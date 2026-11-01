"""Real Cohere/AgentTune pilot. Requires an authenticated free CUDA runtime.

No uploads or paid APIs. Writes real raw completions and measured results locally.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

from toolboundary import evaluate, load_cases, prompt

MODEL = 'CohereLabs/tiny-aya-global'


def parse_completion(raw):
    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate JSON key')
            obj[key] = value
        return obj
    try:
        return json.loads(raw, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return None


def training_pairs(cases):
    return [(json.dumps(prompt(c), sort_keys=True), json.dumps(c['expected'], sort_keys=True))
            for c in cases if c['split'] == 'development']


def infer(model, tokenizer, cases, output, label):
    import torch
    model.eval()
    predictions, completions = {}, []
    for case in cases:
        messages = [{'role': 'user', 'content': json.dumps(prompt(case), sort_keys=True)}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors='pt').to(model.device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**inputs, do_sample=False, max_new_tokens=128,
                                       pad_token_id=tokenizer.pad_token_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        raw = tokenizer.decode(generated[0, inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        predictions[case['id']] = parse_completion(raw)
        completions.append({'id': case['id'], 'raw': raw, 'seconds': elapsed,
                            'generated_tokens': generated.shape[-1] - inputs['input_ids'].shape[-1]})
        print(f'{label}: {case["id"]}', flush=True)
    result = evaluate(cases, predictions, label)
    result['total_generation_seconds'] = sum(r['seconds'] for r in completions)
    (output / f'{label}-raw.json').write_text(json.dumps(completions, indent=2), encoding='utf-8')
    (output / f'{label}-predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    (output / f'{label}-evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


class QuantizedFactory:
    """AgentTune-supported trainer_factory; identical loaded weights for baseline/SFT."""
    def __init__(self, revision):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=revision, trust_remote_code=False, device_map={'': 0}, dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
            attn_implementation='sdpa')
        self.trainer = None

    def __call__(self, *, model, train_dataset, **kwargs):
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer
        self.model = prepare_model_for_kbit_training(self.model)
        self.model.config.use_cache = False
        self.trainer = SFTTrainer(model=self.model, processing_class=self.tokenizer,
            train_dataset=Dataset.from_list(train_dataset), args=SFTConfig(**kwargs),
            peft_config=LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                target_modules=['q_proj', 'v_proj'], bias='none', task_type='CAUSAL_LM'))
        self.model = self.trainer.model
        # The checkpoint defaults to BF16; T4 FP16 scaling requires FP32 adapter gradients.
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(torch.float32)
        print('Trainable dtypes:', sorted({str(p.dtype) for p in self.model.parameters() if p.requires_grad}), flush=True)
        return self.trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/pilot'))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists: preserve the previous run and use a fresh directory.')
    import torch
    from huggingface_hub import HfApi
    from transformers import set_seed
    from agenttune import create_distill_trainer
    if not torch.cuda.is_available():
        raise SystemExit('No CUDA GPU; no model loaded and no training attempted.')
    revision = HfApi().model_info(MODEL).sha
    cases = load_cases()
    pairs = training_pairs(cases)
    if not pairs:
        raise SystemExit('No development demonstrations')
    args.output.mkdir(parents=True)
    manifest = {'model': MODEL, 'model_revision': revision,
        'agenttune_commit': 'd4db4c12d52342227c4113a70183d0f64aa2f345',
        'seed': 42, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
        'data_sha256': hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        'training_demonstrations': len(pairs), 'training_steps': 30,
        'status': 'started', 'claim_boundary': 'Small synthetic pilot, not external validation or a completed competition entry'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    set_seed(42)
    factory = QuantizedFactory(revision)
    before = infer(factory.model, factory.tokenizer, cases, args.output, 'before')
    trainer = create_distill_trainer(student=MODEL, demonstrations=pairs, trainer_factory=factory,
        output_dir=str(args.output/'training'), max_steps=30, max_length=768,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=2e-4,
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False}, logging_steps=5,
        save_strategy='no', report_to='none', seed=42, data_seed=42)
    started = time.perf_counter()
    trainer.train()
    manifest['training_seconds'] = time.perf_counter()-started
    factory.model.save_pretrained(args.output/'adapter')
    factory.tokenizer.save_pretrained(args.output/'adapter')
    factory.model.config.use_cache = True
    after = infer(factory.model, factory.tokenizer, cases, args.output, 'after')
    manifest['status'] = 'completed'
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    comparison = {'before': before['metrics'], 'after': after['metrics'],
        'guard_baseline': 'Guard blocks all unauthorized writes by construction; blocking is not task completion.',
        'conclusion': 'Inspect actual differences and raw failures; no automatic improvement claim.'}
    (args.output/'comparison.json').write_text(json.dumps(comparison, indent=2), encoding='utf-8')
    print(json.dumps(comparison, indent=2))
    del trainer, factory
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
