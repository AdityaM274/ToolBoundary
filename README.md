# ToolBoundary

I built a reproducible synthetic tool-action reliability lab using Tiny Aya Global and AgentTune, with substantial Codex assistance. **Completed experiment:** heldout exact correct2/24 to19/24; raw schema-valid6/24 to24/24. The adapter still proposes3unauthorized writes: this is not a safety guarantee.

Read [the full write-up](ARTICLE.md), [real v2 outputs](reports/completion-v2/) and [negative first pilot](reports/pilot-r2/). Base weights and tokenizer are not redistributed.

## Local sandbox
Python3.10+, stdlib only. Run `python -m unittest -v`, then `python review.py` or open `Start ToolBoundary.cmd`. The reviewer simulates synthetic in-memory tools; it is not live model inference or production authorization.

## Actual training reproduction
Install optional requirements-training.txt in a free CUDA runtime with authorized original base-model access. Keep HF_TOKEN in runtime secrets, never source. Run:

```sh
python run_experiment.py --cases evaluation-v2.json --training-cases training-v2.json --steps 60 --output runs/reproduction
```

The script refuses overwrites and preserves real raw completions/hashes. Training cases240, separate evaluation48; shared inspectable synthetic families are not an external blinded test.14tests passed locally and in Colab. The original cases.json and scripted demo remain as the24case pilot sandbox; demo is not model performance evidence.

## Permission boundary
Exact action keys tool,record,version,value. Trusted grants scope each tool/record; execute additionally matches current version and approved value. Never accept arbitrary user-supplied grants as authenticated authority in a real service. Guard-blocked model errors are not correct model answers.

## Publication and licenses
Adapter/results: https://huggingface.co/Aditya0427/ToolBoundary-tiny-aya-lora

Own code MIT. Upstream Cohere C4AI CC-BY-NC/AUP and AgentTune LSAL1.1 retain their conditions. No production users, external validation, safety guarantee or fabricated personal learning claims. Actual ChatGPT/Codex contribution is disclosed in the article.
