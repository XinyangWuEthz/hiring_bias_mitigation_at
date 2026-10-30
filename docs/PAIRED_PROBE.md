# DAPO final / SFT-v3 paired diagnostic

Requested 2026-10-22. Run after the margin-monitor changes have been committed. This is a
training-pool diagnosis, not a held-out estimate of generalization or a monitor calibration.

Both SFT-v3 and DAPO-v3-all's final step-85 adapter are evaluated on every one of the 9,140
original `grpo-v3-all/grpo_train.parquet` rows. Each row has its original attributed prompt
and its source pool's exact `attr_free` prompt, joined by `pair_id`. Each view is sampled 16
times for each model: 584,960 completions total. The 1,767 underlying pairs repeat across
rows. Free prompts are deliberately sampled separately for every original row.

The serving path uses existing audited BF16 merged weights. DAPO's chain is base model,
SFT-v3, then the final DAPO adapter. The native vLLM LoRA path is not faithful for this
repository's Qwen3.5 adapters. The run manifest records input and merged-weight SHA256,
adapter hashes, merge retention, tokenizer/template files, code hashes, and the commit.

Sampling matches the DAPO rollout settings: temperature 1, top_p 1, max_tokens 96, thinking
off. A fresh base seed of 20261022 avoids reusing the selection samples. SHA256 of base seed,
row_id, view, and prompt gives a separate random stream per request; requests reserve 32
seed values, of which vLLM uses 16. Both models use the same request seeds. The classic
vLLM runner is explicitly checked. Each model uses two GPU shards on one four-GPU node.

The unit of analysis is an original attributed row. Correct means agreement with its
existing anchor, not an independently verified correct hiring decision.

- Attributed 16/16 wrong, free 16/16 right: `attribute_bias_AB`.
- Both 16/16 wrong: `anchor_disagreement_CD_pause_RL`, review C+D and seriously consider
  pausing RL. This establishes anchor disagreement, not that either side is correct.
- Free results mixed: preserve the exact 0–16 correct count and report `mixed_unresolved`.
- Parse failures and refusals: separate from semantic errors. Truncation is an overlapping
  diagnostic. A parseable hire/reject remains a decided response even if truncated.

Also report a free-majority sensitivity analysis, with ties and invalids separate. Report
counts and denominators by model, hire/reject reference, exact anchor hire/valid votes, and
winning/valid votes. The anchor vote files come from `new-pairs` and `new-pairs-2`; their
reference verdict and candidate/job IDs must match every selected row. Row-level tables,
unique-pair counts and SFT-to-DAPO classification transitions remain available. Repeated
rows are correlated, and 0/16 does not establish a truly zero success probability.

`scripts/paired_probe.py prepare --run RUN_DIR` validates all joins and writes `rows.parquet`
and `manifest.json`. `RUN_DIR=... scripts/cscs/submit.sh scripts/cscs/paired_probe.sbatch`
starts four workers; each persists atomic sample chunks. Resubmission resumes valid chunks.
The launcher automatically runs analysis only after all workers succeed. The analyzer
rejects missing, duplicate, foreign-model, and out-of-range samples before writing results.

Final files in RUN_DIR: `report.md`, `summary.json`, `row_classifications.parquet`, and
`all_wrong_rows.parquet`. Raw outputs and finish reasons stay under `samples/`. No training
job is stopped by this diagnostic; route priorities are interpreted from the measured
counts and their hire/reject and vote-strength breakdowns.
