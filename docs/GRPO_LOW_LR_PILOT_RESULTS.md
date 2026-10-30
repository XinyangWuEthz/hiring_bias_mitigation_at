# GRPO lower-LR pilot: startup failure

The single authorized attempt, Slurm **5130709**, failed before training. It provides **no evidence about the effect of lowering the learning rate**.

## Observed outcome

| Item | Evidence |
| --- | --- |
| Frozen source commit | `64019e2d0c43f33fa5739057fde3d57fe60eb496` |
| Job state / exit | `FAILED`, `1:0` |
| Start / end, Europe/Zurich | 2026-10-23 16:26:21 / 16:27:23 |
| Allocated duration | 62 seconds, four GPUs, about 0.0689 allocated GPU-hours |
| Training-process duration | 23.9 seconds; recorder observed no GPU processes, 0 active GPU-hours |
| New optimizer updates | 0 |
| New benchmark inference | None |
| Startup GPU decision gate | Not reached |
| New checkpoints / monitor observations | None |

The frozen-input preflight passed on the compute node. The error occurred before model/Trainer construction and before `on_train_begin`, so neither optimizer restoration nor the GPU startup check was reached. The new run directory contains only a nine-byte `wandb_run_id`. There is no `run_config.json`, continuation receipt, benchmark monitor history, checkpoint, final audit or model-result file.

Consequently, there is no new full/dev/historical-remainder TPR, FPR, c or d′ trajectory. The 8 pp stop rule was not evaluated. This is neither a guardrail pass nor a model-quality failure. Original checkpoint425 is unchanged; there is no new safe pilot checkpoint. The historical dev8 stopping replay and all previous benchmark results remain unchanged.

## Root cause

The new continuation path checked that a fresh output directory was empty independently on all four ranks. A rank that passed could immediately proceed. Rank 0 then called `C.wandb_resume`, which created `wandb_run_id` in that directory. Rank 3 arrived later and interpreted the newly created file as an incomplete previous run:

```text
ValueError: continuation output is not empty and has no completed checkpoint
```

The file's recorded modification time is 16:27:20.125737 Europe/Zurich. Three other ranks logged loading the training and validation data at 16:27:22.269–.287; torchrun terminated those ranks after rank 3 failed. The code had no synchronization between the per-rank freshness check and rank 0's first file write. Two independent code reviews reached the same diagnosis.

The defect is in the new distributed startup ordering, not in W&B availability or the learning-rate intervention. The earlier 98 CPU tests missed this concurrent initialization sequence.

## Correction and validation

The correction changes only continuation startup ordering. All ranks must pass the existing strict resume check and complete Trainer construction before a barrier permits rank 0 to write tracking metadata. The write outcome is broadcast to every rank; success is followed by a second barrier. A write error is raised consistently on all ranks. Existing non-continuation configurations keep their original initialization order.

The installed Transformers W&B callback initializes its run at `on_train_begin`, so the delayed run ID is still available before W&B setup. For this frozen configuration, Trainer construction does not write output content: `push_to_hub=false` and `log_completions=false`.

Four new CPU tests use real spawned processes to reproduce the exact old interleaving, verify that all four resume checks precede the first metadata write, retain rejection of pre-existing partial output, and propagate a simulated rank-0 write failure without hanging. The combined GRPO/continuation/monitor/finalizer suite passed **102 tests and four subtests in 39.33 seconds**. Relevant lint and whitespace checks passed.

This is CPU evidence for the startup correction. The four-GPU startup decision gate and actual update426 LR are still unvalidated. The change does not authorize another job or relax any model/data/TPR condition.

The failed output directory, frozen manifests, original prompt/data, labels, source checkpoint and thresholds are preserved. No replacement job has been submitted. Any approved retry must use a new output directory and a new frozen manifest for the corrected code; the failed attempt must not be overwritten or silently reused.

## Evidence

Remote experiment directory:

`/ritom/scratch/cscs/zhang/xw/outputs/grpo_low_lr_pilot/grpo_v3_all_lr3e6_from425_20261023/`

The `failure_evidence/` subdirectory contains the full job log, `sacct.txt`, cost record, training-directory inventory, and 18 exact source/configuration snapshots from commit64019e2. All 18 archived SHA-256 values match the original frozen manifests. These snapshots preserve the actual failed code even after a separate fix is committed. The source checkpoint and training directory have not been deleted or reset. Slurm automatic requeue is disabled.

Local evidence and this report are under `outputs/grpo_low_lr_pilot/`. Since no model result was produced, no TPR or SDT result chart is presented.
