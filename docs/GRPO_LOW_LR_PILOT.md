# GRPO lower learning rate pilot

Status: preregistered for one bounded training job, authorized by the user's 2026-10-23 request, translated from Chinese, to "try a small run with a lower LR." No other training arm is authorized by this protocol.

## Question and intervention

Continue the original GRPO-v3-all checkpoint 425 for at most 75 optimizer updates, ending at absolute step 500. Change the constant post-warmup learning rate from 3e-5 to 3e-6. The fixed source is the beginning of the previously examined 425–571 window, not the checkpoint with the best benchmark result. The historical 425/450/475/500 trajectory covers the observed decline and rebound.

Restore the checkpoint's Adam moments, scheduler counters, global step and four HF rank RNG files. Change the optimizer and scheduler LR fields only after their checkpoint states are restored. Keep the original five-step warmup counter; do not restart warmup. Record actual LR before update 426 and subsequent updates. Preserve data skipping and the original four-rank topology.

Keep the original prompt, all 9,140 training rows, SFT-v3 KL reference, ordinary rank-32 LoRA, beta 0.04, 16 generations, sampling settings, four GPUs, per-device batch 8, accumulation 8, seed 42 and reward implementation. The saved TRL arguments use `loss_type=dapo` normalization and group reward scaling. This does not enable the project's separate DAPO trainer. Record and compare the resolved training arguments with the source checkpoint.

The HF Python/NumPy/Torch RNG states are restored. vLLM's internal sampling RNG was not checkpointed and restarts with the engine. This run is therefore not a bit-identical continuation. The existing 3e-5 trajectory is a historical comparison, not a randomized same-restart control. A separate restarted 3e-5 arm would require its own authorization.

## Frozen inputs and scope

- Source run: `outputs/grpo/qwen3.5-9b_en_only_grpo_v3_all/checkpoint-425`.
- New training output: `outputs/grpo/qwen3.5-9b_en_only_grpo_v3_all_lr3e6_from425_75step`.
- New experiment output: `outputs/grpo_low_lr_pilot/grpo_v3_all_lr3e6_from425_20261023`.
- Original trajectory binding: `d66a7315927d9b7b079a0529396b73b76d7aed61982f6f8f897c2f6ea326ce58`.
- Full benchmark: the same 31,050 prompts, 150 candidates and 450 pairs, including 10,350 reference-hire rows. Labels, prompt text, scoring and existing dev/remainder split stay fixed.
- Original 9,140-row training data and validation files remain byte-identical. Checks found zero candidate, pair or job ID overlap between the training set and the full benchmark.
- The new monitor bank freezes source hashes, SFT and source425 decisions, historical comparison outputs, split, scorer and software identities. The outer experiment manifest pins the final YAML, preregistration, batch script, implementation and Git commit.

All old manifests with `training_authorized=false` retain that setting. Authorization in the new experiment is restricted to this pilot. Do not alter the historical benchmark tables, the previous average-model protocol or historical stopping decisions. The benchmark has already informed development and selection; none of these results constitutes a new untouched test.

## Direct monitoring and stopping

The previous dev-only threshold missed the full-benchmark 10 pp guardrail at historical step 150. This pilot uses direct full-benchmark positive-row measurement instead of that projection or dev-only gate.

Before any optimizer update, score all 31,050 benchmark prompts using the serving model restored from checkpoint425. Require exact decision agreement with the frozen checkpoint425 cache. Keep the existing HF/merged/vLLM synchronization check as an additional startup check. CPU tests do not establish GPU callback correctness. Any mismatch stops before training and is a numerical or implementation diagnosis, not a model result.

Every five optimizer updates, score all 10,350 reference-hire rows with the original greedy, thinking-disabled, 512-token decision protocol. At absolute steps 450, 475 and 500, and at any stop trigger, also score the reject rows to obtain a complete benchmark result. Reuse the positive-row generations from the same step.

SFT-v3 has 5,975 true positives out of 10,350, or 57.729468599%. Stop and save irreversibly at the first measured full TPR change of at most −8 pp, equivalently at most 5,147 true positives. Save every five updates and retain both the preceding safe checkpoint and the trigger checkpoint. A later recovery cannot cancel the stop. There is no automatic resubmission after failure, timeout or a stopping event.

The original final full-benchmark guardrail remains a decrease of no more than 10 pp. Exactly −10 pp passes; on these discrete counts, at least 4,940 true positives are required. Final agreement decrease must be no worse than 5 pp, coverage at least 99%, and effective-update retention after bf16 merge at least 80%. These final checks do not override the earlier 8 pp training stop.

The five-step cadence and two-point buffer reduce delay but cannot guarantee that TPR never crosses the guardrail between checks. Full/dev/historical-remainder reporting uses the original frozen split. The full benchmark is primary; dev and historical remainder are descriptive.

## Terminal audit and interpretation

After a normal training-loop exit, audit the terminal saved checkpoint: step500 if all 75 updates finish, or the first trigger checkpoint if monitoring stops training. Do not choose a checkpoint by its benchmark score. Merge common SFT once and then the ordinary rank-32 terminal update, validate merge retention, and run the same standalone TP1 full-benchmark inference protocol as the original trajectory. Require exact row-level decisions against the callback's complete terminal result. A mismatch invalidates runtime-monitor comparability and requires diagnosis.

Report actual LR and restored-state receipts, startup agreement, all five-step TPR measurements, first stop and preceding safe checkpoint, and complete TPR/FPR, agreement, coverage, c and d′ at 425/450/475/500 when available. Compare the same available historical steps. A stopped pilot is reported as stopped; missing future steps are not extrapolated. The terminal audit retains the existing evaluator's instability metrics.

Smaller TPR or c fluctuations would support running a better-controlled learning-rate comparison. They would not prove a fixed point or optimization noise: reduced parameter movement and restarted sampling are alternative explanations. A decline in d′, a failed guardrail, or an early stop is reported directly. TPR/c stability alone does not establish a fairness improvement. The earlier historical dev8 first-stop at400 remains in force for its original replay, and this explicitly authorized exploratory branch does not reverse it.

## Execution budget

Submit one Slurm job on the normal partition, one four-GH200 node, with a two-hour limit. The original run averaged about 37 seconds per optimizer update, so 75 updates alone take about 46 minutes. Startup checks, direct monitoring and the terminal standalone audit add time; the planning estimate is roughly 80–110 minutes, excluding queue time. The 30-minute debug limit cannot accommodate the planned experiment. On timeout, inspect saved state and evidence before proposing a continuation; do not silently submit another job.
