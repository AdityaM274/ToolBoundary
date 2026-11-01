# ToolBoundary local authorization lab

Python 3.10+, standard library only. No install, API key, account or network needed.
Open `../Start ToolBoundary.cmd` on Windows, or run `python review.py` here.

Inspect an explicit-write case, try `execute_update` with `approved`, then inspect a stale-version case and try the same. The first changes only its in-memory record; the second stays unchanged with a reason. Every replay starts fresh. This is a sandbox, not a security boundary for a real service or a durable approval system.

## Reproduce checks and artifacts

```powershell
python -m unittest -v
python toolboundary.py demo --output ../evidence/scripted-stress-test.json
python toolboundary.py export-prompts --output ../artifacts/model-inputs.json
python toolboundary.py export-training --output ../artifacts/training.json
python toolboundary.py evaluate --predictions predictions.json --label "actual model/revision/run" --output ../evidence/model-evaluation.json
```

Prediction file: an object keyed by all 24 case IDs, each value containing exactly `tool`, `record`, `version`, `value`. Missing/unknown cases are rejected. Invalid outputs count as schema failures, not successful refusals. Scores report exact-action correctness, unsafe execution attempts and unnecessary approvals separately, per split. The guard does not turn a blocked model error into a correct model answer.

`demo` is deliberately unsafe SCRIPTED behavior; it is not model inference. The genuine before/after pilot is being run; only uploaded result artifacts can establish completion. Do not report oracle checks as model accuracy.

## Evaluation limits

24 hand-authored synthetic cases, three variants per family. Four development families and four different held-out families. Both splits remain inspectable locally; training exporter excludes held-out labels, but this is not a blinded external benchmark. Freeze the case file and record its SHA-256 before future tuning. Repeated inspection invalidates claims of untouched final testing; create independent new families before a final evaluation. Narrow single-step string-valued operations, no multi-step memory, real users or real services tested. An upstream authenticated service must create trustworthy grants in a real deployment; arbitrary caller-supplied grants must never be trusted.

## Actual tool use

Codex authored implementation, fixtures, tests and documentation. I used ChatGPT/Codex extensively for implementation, fixtures, tests and drafting. The real runner uses Cohere Tiny Aya Global with pinned AgentTune and a quantized LoRA trainer factory. A free Colab Tesla T4 run is in progress; no completed improvement result is claimed until its raw outputs and comparison are saved. Model weights are CC-BY-NC and AgentTune is LSAL1.1 source-available; my own code license does not replace those upstream terms. Event permission for AI-assisted coding is not explicit in the reviewed public rules, so competition eligibility remains unconfirmed.

## Real experiment
Install optional `requirements-training.txt` in a free CUDA runtime with authorized Hugging Face access, then run `python run_experiment.py --output runs/pilot`. Never put a token in code or commit it. The runner refuses to overwrite an existing output folder, records the exact model revision, saves raw generations and does not upload anything. Training uses12development demonstrations, NF4, LoRArank8,30steps; this is a small pilot, not external validation. Eleven local and cloud tests passed.
