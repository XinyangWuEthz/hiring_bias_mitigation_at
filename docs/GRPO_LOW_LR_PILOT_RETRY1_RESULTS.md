# GRPO lower-LR pilot retry 1: startup gate failure

Job **5131944** failed before any optimizer update. At the unchanged checkpoint 425, the training-side serving model disagreed with the frozen historical cache on **52 of 31,050 decisions**. The strict startup gate stopped the job. This was not a TPR guardrail trigger and provides no result about the effect of lowering the learning rate.

The job ran on 2026-10-23 from **19:39:31 to 19:48:26 Europe/Zurich**, for 8 minutes 55 seconds, and exited `FAILED 1:0`. Its frozen code was `b9e8a828a242abb143f74c5f6e47f8d1b0f58aa8`. The authorized bounds remain checkpoint 425 through at most 500, LR `3e-6`, and irreversible stopping at the first full-benchmark TPR decline of at least 8 pp relative to SFT-v3. See [the retry registration](GRPO_LOW_LR_PILOT_RETRY1.md) and [the original protocol](GRPO_LOW_LR_PILOT.md).

## What executed

The previous W&B startup race did not recur. CPU preflight passed. All four ranks loaded checkpoint 425, retained the 256 Adam states at optimizer step 425, and set LR from `3e-5` to `3e-6` without restarting warmup. Each rank recorded only `restored_and_lr_set`; none recorded `before_optimizer_update` or `after_optimizer_update`. No new checkpoint or adapter was saved.

All four ranks passed the existing 256-row synchronization probe. The saved rank-0 receipt reports 100% decision agreement, margin rank correlation 0.99924268, and merge retention 96.6887168%. This establishes agreement on that probe between the live serving engine and the current in-memory merged policy. It does not establish equivalence with the historical cache over all 31,050 rows.

The full startup pass then completed all 31,050 rows with no invalid decisions, found 52 mismatches, and raised from `on_train_begin`. Thus `startup_gate_passed=false`, `training_completed=false`, and no terminal independent audit ran. `run_config.status=training` was written before training and is not evidence of updates. The planned 3,400-microbatch skip and HF RNG restoration occur after `on_train_begin`; this attempt never reached those operations.

## Existing-output comparison

The following are diagnostic measurements of unchanged checkpoint 425 under two execution paths. They are not measurements of a trained lower-LR model.

| Full benchmark quantity | Historical frozen 425 | Retry startup 425 | Difference |
| --- | ---: | ---: | ---: |
| Hire decisions on 10,350 positive rows | 6,204 | 6,192 | -12 |
| TPR | 59.942029% | 59.826087% | -0.115942 pp |
| TPR change relative to SFT-v3 | +2.212560 pp | +2.096618 pp | -0.115942 pp |
| Hire decisions on 20,700 negative rows | 1,158 | 1,158 | 0 |
| FPR | 5.594203% | 5.594203% | 0 pp |
| Invalid decisions | 0 | 0 | 0 |

SFT-v3 full TPR is 5,975 / 10,350 = 57.729469%. The startup TPR is above this baseline; the 8 pp stopping threshold was not reached. The original final full 10 pp guardrail remains unchanged, including equality passing. No new model exists to select or qualify under either rule.

| Anchor label | Historical decision | Startup decision | Rows |
| --- | --- | --- | ---: |
| hire | hire | reject | 26 |
| hire | reject | hire | 14 |
| reject | hire | reject | 6 |
| reject | reject | hire | 6 |

The 52 mismatches are 0.167472% of the bank and cover 14 pairs, 14 candidates and 14 jobs. Per-rank mismatch counts are 9, 15, 13 and 15. In total 5,065 generated strings differ, including feedback; only 52 change the parsed binary decision. Reapplying the official parser to both sets of saved text reproduces every saved decision, with no invalid output.

The comparison also verifies exact saved prompt text and metadata, all 31,050 unique keys and coverage, the frozen source bindings, selected source/tokenizer/code fingerprints, and both SFT/GRPO adapter inputs. A separate recheck found all ten checkpoint-425 files and the training/validation parquet hashes unchanged. This is not an independent check of rendered tokenizer output or hidden GPU logits.

## What remains unresolved

The historical audit and the live monitor share the intended scorer settings: greedy decoding, at most 512 generated tokens, `top_p=1`, thinking disabled, and TP1. Their execution paths still differ:

| Execution setting | Historical audit | Training-side startup |
| --- | --- | --- |
| Weight merge | CPU FP32 LoRA merge, save bf16 model, then load | GPU FP32 LoRA merge, cast and synchronize live weights |
| Prompt calls | Contiguous chunks of 4,000 | Four strided rank shards of 7,762 or 7,763 |
| Maximum active sequences | 256 | 64 |
| Batched token budget | 16,384 | 4,096 |
| Maximum model length | 8,192 | 4,096 |
| GPU memory fraction | 0.8 | 0.3 |

The frozen inputs fit within 4,096 tokens: maximum input length 1,847 plus 512 output tokens. The length setting therefore does not by itself explain prompt truncation. Historical and current merge-retention summaries differ only around 1e-8, but these summaries do not prove byte-identical merged tensors. Batch-dependent numerical differences and CPU/GPU merge rounding remain hypotheses. The available evidence does not isolate a specific kernel, tokenizer-rendering or merge cause.

The next useful experiment is one controlled, inference-only check of checkpoint 425: compare actual merged tensor hashes and tokenizer-rendered input IDs, then hold weights fixed while reproducing historical and live batching separately. A full canonical 31,050-row replay is required to certify the exact gate; the 52 differing rows and agreeing controls can help localize the cause. This would require separate authorization for a new GPU job. No additional diagnostic job was submitted, no failed output was adopted as a new reference, and no tolerance, threshold, source, label or split was changed.

## Evidence and interpretation

Remote experiment directory:
`/ritom/scratch/cscs/zhang/xw/outputs/grpo_low_lr_pilot/grpo_v3_all_lr3e6_from425_retry1_20261023/`

- `failure_diagnosis.json` records the failed gate and zero updates.
- `failure_evidence/` contains Slurm accounting, cost, training inventory and the source-integrity recheck.
- `startup_diagnostics/summary.json` records reproducible existing-data checks and metrics; `decision_mismatches.csv` lists all 52 rows with both decisions and outputs.
- The training directory retains `continuation_rank*.jsonl`, `sync_validation.jsonl`, monitor state/history and full/positive startup decision parquet files.
- The job log is `logs/xw-grpo-lr3e6-r1-5131944.out` under `/ritom/scratch/cscs/zhang/xw/`.

Copies, analysis scripts and a readable report are saved in the local task's `outputs/grpo_low_lr_pilot/retry1/`. The training process cost record reports 502.2 seconds, 0.558 GPU-hours and 638.2 kJ; Slurm's 535-second allocation includes setup and reports node energy separately. No training trajectory, terminal comparison or lower-LR benefit can be inferred from this failed startup. The historical dev first-stop replay and published benchmark tables remain unchanged. Automatic resubmission remains disabled; the completion check is paused after this failure report.
