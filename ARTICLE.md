# ToolBoundary: better tool-call reliability still needs an authorization guard

I built a synthetic lab for single-step record operations. It separates model proposals from a deterministic permission check: an execute update must match the exact trusted tool, record, current version and approved replacement value. Read/propose operations are checked for tool and record scope. Untrusted document text never creates permission. No real account or record is changed.

## Research question and scope
Can completion-only supervised LoRA training improve a small model's strict tool-action reliability without confusing syntactic validity with authorization? This is a Track3 AgentTune research pipeline. It uses supervised demonstrations from a deterministic policy oracle, not reinforcement learning or stronger-model trajectory distillation. I do not claim to have implemented those suggested example tasks. The event permits other within-scope ideas.

## Reproducible pipeline
Base: CohereLabs/tiny-aya-global revision af89d219b53ed9b13b8a4645f9c8028973510324. Source experiment commit d26cd07487224918981af33eed6d9a6978699453. AgentTune d4db4c12d52342227c4113a70183d0f64aa2f345 via its trainer_factory hook with TRL. NF4 base, FP16 compute and FP32 trainable adapters; LoRA rank8/alpha16/dropout0.05 on q_proj/v_proj; seed42;60steps, effective batch4, learning rate2e-4, max length768. Free Colab TeslaT4. Wrapper training time348.884seconds; not an end-to-end deployment cost measurement.

240 synthetic training cases and48 distinct evaluation cases (24development and24heldout wording variants). Evaluation IDs are disjoint from training IDs. Eight authorization modes include missing/wrong scope, stale version, changed value, injected instructions and decoy-plus-valid grants. Labels are constructed from explicit policy conditions and checked against the replay guard. Raw prompts exclude expected answers. The corpus/source hashes were frozen before v2 outputs. All templates remain inspectable and families are shared across splits: this is NOT an external blinded benchmark or proof of generalization.

## Actual results
Strict JSON, exact-action scoring; identical deterministic generation before and after, no repair into success.

| Metric /24 | Development before | Development after | Heldout before | Heldout after |
|---|---:|---:|---:|---:|
| Schema-valid raw actions |7|24|6|24|
| Exact correct action |2|20|2|19|
| Unauthorized execute proposal |1|2|0|3|

Heldout exact completion improved8.3% to79.2%, a70.8percentage-point gain; validity25% to100%. These results support narrow format/task reliability, NOT a safety improvement. Invalid baseline responses are unclassifiable as executable actions, so zero counted unsafe baseline proposals is not evidence of safety. The trained model still makes3unauthorized heldout proposals. A production service must independently enforce trusted grants; the model must never grant itself authority.

A guard blocking a wrong proposal does not make the model's answer correct. Always-request-approval also avoids writes but fails legitimate tasks. Both are simple baselines for interpreting risk; this submission does not claim a learned agent outperforms deterministic policy enforcement. No claim of real users, time saved, commercial ROI or production security validation.

## Failures retained
The first attempt failed before optimizer step1: BF16 gradients were incompatible with the T4 FP16 scaler. A corrected30step pilot completed but strict raw correctness was0/24 before and after due to Markdown fences. A separately labeled post-hoc fence diagnostic stayed13/24 correct; unsafe heldout proposals increased3to4. The revised experiment used rawJSON instructions for BOTH baseline and adapter, separate240/48datasets and completion-only loss. Negative pilot evidence remains in reports/pilot-r2; it was not silently replaced.

## Reproduce
Use Python3.10+ for the stdlib sandbox: `python -m unittest -v` and `python review.py`.14unit tests passed locally and in Colab. For actual inference/training use a CUDA runtime, install requirements-training.txt, obtain authorized access to the gated original base model, and load HF_TOKEN from the runtime secret store. Never put credentials in code.

```sh
python run_experiment.py --cases evaluation-v2.json --training-cases training-v2.json --steps 60 --output runs/reproduction
```

The runner refuses to overwrite results, records revision/data/source hashes and saves predictions, raw completions, evaluation and adapter. `toolboundary.py demo` is deliberately scripted unsafe behavior, not model inference. The local interactive reviewer does not run the model; inspect saved raw completions for actual model behavior.

## Limits and attribution
Single-step English, string-valued synthetic sandbox; no multi-tool ordering, multi-turn attacks or independently collected users. One seed and small sample. Before external deployment create independent policy families, multiple seeds, realistic authenticated tool wrappers and failure-cost evaluation. No unsupported pricing/throughput or closed-model comparisons.

I used ChatGPT/Codex extensively for implementation, fixtures, testing, experiment operation and drafting under my direction. My project code is MIT; Cohere model C4AI CC-BY-NC terms/AUP and AgentTune LSAL1.1 remain applicable. The published adapter does not include base weights. This article preserves the actual assistance and measured limitations.

Artifacts: https://huggingface.co/Aditya0427/ToolBoundary-tiny-aya-lora

Official event: https://hackathon.lexsi.ai/
