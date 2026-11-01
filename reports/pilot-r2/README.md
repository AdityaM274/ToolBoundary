# Real pilot r2 — negative result
Cohere Tiny Aya Global + AgentTune, Tesla T4, 30 real training steps, 12 training examples, 24 synthetic evaluation cases. Exact model revision and actual timings in manifest.json.

Strict raw-JSON evaluator: baseline0/24, after0/24 (both used Markdown code fences). A separate POST-HOC single-fence-unwrapped diagnostic: baseline13/24, after13/24. Heldout unsafe execution attempts worsened3to4. No improvement or safety claim is justified. Raw outputs and strict scores remain unchanged and available here.

The next run changes completion-only loss, expands training to240examples, adds explicit no-Markdown instructions equally to baseline and adapter, and uses48different evaluation cases frozen before any new outputs. It is a new synthetic experiment, not a revised score for this failed pilot. No blinded real-world benchmark or winning outcome claimed.
