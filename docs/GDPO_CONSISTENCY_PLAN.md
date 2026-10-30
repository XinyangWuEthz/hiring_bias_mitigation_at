# GDPO with a cross-variant consistency reward: plan

A plan for the final RL arm, written on 2026-10-22 at the user's request ("Plan a GDPO run with a cross-variant consistency reward (sibling variants in the same generation batch) for the final run: design, code changes, cost"). Nothing here is implemented or run yet. Before anything trains, a pre-registration goes in its own card, as `docs/GRPO_V3.md` has one.

## What GRPO has shown so far

The numbers come from `docs/GRPO_V3.md`, `reports/reference_metrics.csv` and the runs' `trainer_state.json` and `decision_eval.jsonl`. Each finding comes with the change it suggests.

| # | Observation | Evidence | Change it suggests |
|---|---|---|---|
| 1 | Prompts that SFT-v3 trained on carry almost no signal | GRPO-v3, on SFT-v3's own pairs: +0.74 pp [−0.15, +1.63]. GRPO-v3-all, on unseen pairs: −0.56 (final adapter), −1.02 (checkpoint-525) | Unseen pairs only. GRPO-v3-all already does this |
| 2 | The signal runs out within one epoch | Groups whose 16 rewards are identical: 48% in steps 1–57, 96% in steps 457–571, all on prompts not seen before | Dynamic selection by the current policy, or DAPO-style dynamic sampling. More epochs or more of the same data will not help |
| 3 | Pre-selection by SFT-v3's error rate goes stale | At SFT-v3's own p, 87% of groups should split; 52% did in the first 57 steps | Re-select during training (as 2) |
| 4 | The anchor reward makes the model stricter than GPT-4o | 25 of SFT-v3's 107 attribute-free hires become rejects, 20 of them GPT-4o hires. TPR −10.11 pp, which fails the guardrail | Reward invariance against the model's own attribute-free answer, not the anchor (this plan) |
| 5 | Half the prompts train agreement with the anchor, not invariance | 4,078 of 9,140 prompts are untipped. SFT-v3's p on them has a median of 0.63–0.69 | An ablation on tipped rows only, or the consistency reward |
| 6 | The hire-rate monitor does not track the benchmark | Checkpoint-525: projected 23.5%, audited 19.09%. The 19–27% swing in the projection does not show on the benchmark | Monitor counterfactual flips on held-out pairs directly |
| 7 | Decisions oscillate, and the verdict depends on the checkpoint | Checkpoint-525 and the final adapter agree on 444 of 450 attribute-free decisions, yet one is "better" and the other "not better" | lr decay instead of constant; a checkpoint rule declared in advance on a validation metric that works; a second seed |
| 8 | The format reward is constant, so GDPO alone changes nothing | Format reward mean 0.200, group std 0.0006 | GDPO only matters once a second reward varies (this plan) |
| 9 | Kept: lr 3e-5 makes the merge keep the update | Merge kept 85.9% (GRPO-v3) and 97.7% (GRPO-v3-all), against the pilot's 58% at 1e-5 | Keep |
| 10 | Kept: the k=32 hire top-up | On round 1's pool, without the top-up, tipped hire rows fall from 1,964 to 958 | Keep |

## Design

### The unit: a pair block

Each round-3 pair has 12 attributed variants (3 groups × 2 values × explicit and implicit) and one attribute-free prompt. On the pilot, 2,777 of the 2,787 pairs have all 12. SFT-v3 has already sampled every attribute-free prompt 8 times (`onpolicy-new{,2}/raw/samples`).

- **Block.** One attribute-free prompt and 3 attributed variants of the same pair, highest pull first. Four blocks are 16 prompts, one generation batch: 16 prompts × 16 completions = 256, as in GRPO-v3-all.
- **Order.**
  - The builder writes each block's rows next to each other and shuffles whole blocks, never single rows.
  - Training sets `shuffle_dataset: false`. TRL 1.15's `RepeatSampler` then walks the dataset in order. Its batch is `generation_batch_size // num_generations` = 16 prompts, so every block lands in one generation batch (`trl/trainer/grpo_trainer.py`, `_get_train_sampler`).
- **Supply.** On round 3's training pool:
  - 1,757 pairs have at least one variant with a sampled error, 964 of them hire-anchor. 1,237 pairs have at least 3.
  - Blocks are balanced by pair anchor and summed pull, as `signal_balanced_train` does with rows now.
  - A pair with many informative variants gives several blocks. Each block carries its own copy of the attribute-free prompt.

### Rewards

| reward | rows | value | weight |
|---|---|---|---|
| `decision` (unchanged) | all, including the attribute-free prompt | 1 if the decision equals the vote anchor | 1.0 |
| `consistency` (new) | attributed rows | the share of the pair's 16 attribute-free completions in this batch with the same decision: soft agreement with the model's own attribute-free answer | 1.0 |
| `format` (unchanged) | all | 1 if the answer is exactly the JSON object | 0.2 |

- **Rows with no consistency value.** `consistency` returns `None` for attribute-free rows, and for any row whose attribute-free prompt is missing from the batch. TRL turns `None` into NaN, and `normalize_then_sum` uses NaN-aware means and deviations per reward (`grpo_trainer.py`, `_generate_and_score_completions`).
- **Aggregation.** `multi_objective_aggregation: normalize_then_sum` (GDPO), `reward_weights: [1, 1, 0.2]`.
- **Why the target is the model's own attribute-free answer.**
  - It rewards a decision that does not move with the attribute. It does not ask the model to adopt the anchor's stricter standard, which observation 4 shows cost TPR.
  - The anchor term stays: without it, answering everything the same way, for example all rejects, would satisfy consistency.
- **Why GDPO here.**
  - The two rewards vary differently. `decision` splits a group whenever the 16 answers split. `consistency` also varies with how split the attribute-free answers are.
  - `sum_then_normalize` would let the one with the larger variance steer.
  - `normalize_then_sum` gives each the same weight inside every group. That is the point of research question ① on the Notion page, and it is TRL's one-line option.

### What it does not fix

The group is still one prompt's 16 completions. A variant that flips deterministically, all 16 answers the same and all against the attribute-free answer, has no variance in either reward and gives no gradient. That is observation 2's limit.

Counterfactual grouping would reach those variants: advantages computed across a block's variants instead of across resamples. It means overriding TRL's advantage code (the normalization in `_generate_and_score_completions`), so it is left as a stretch, not part of the first run.

### Monitor

A counterfactual monitor is added next to the hire-rate one. Every 25 steps it takes greedy decisions on held-out pairs, the attribute-free prompt and all 12 variants, and logs:

- the share of variants whose decision differs from the attribute-free one;
- the hire rate.

This is the benchmark's own quantity on held-out data (observation 6). The 15–35% band stays as the stop condition.

## Code changes

| # | Where | Change | Tests (CPU) |
|---|---|---|---|
| 1 | `generation/grpo_prompts.py`, `scripts/build_grpo_prompts.py` | `--blocks 4`: attach each pair's attribute-free prompt (`prompts.parquet`, `kind == "attr_free"`) and its samples; columns `pair_key`, `is_free`; balance blocks by anchor and pull; write blocks contiguously, shuffled as blocks | blocks are contiguous and whole; the count is a multiple of 4; anchors are balanced; the default mode still rebuilds grpo-v3 and grpo-v3-all byte for byte |
| 2 | `mitigation/grpo.py` | `consistency_reward`: all-gathers (`pair_key`, `is_free`, decision) over ranks with `torch.distributed.all_gather_object`, since TRL scores each rank's slice before it gathers; soft agreement; `None` as above. Reward list chosen by config. `multi_objective_aggregation` added to `GRPO_CONFIG_PASSTHROUGH`. A start-up check that `shuffle_dataset` is false and that 16 prompts make whole blocks | soft agreement values; `None` rows; the single-process path; refusal when blocks would straddle a batch |
| 3 | `tests/test_grpo.py` | TRL 1.15's `RepeatSampler(shuffle=False)` with this run's sizes keeps each block in one generation batch | as stated |
| 4 | `mitigation/grpo.py` (decision monitor) | the counterfactual flip monitor | the flip rate on a toy set |
| 5 | `configs/` | data, training (GRPO-v3-all's with the changes above) and audit configs | `load_config` and `GRPOConfig` build on CPU, as for GRPO-v3-all |

About a day of work, then a 20-step smoke run on `debug` (about 1 GPU-h). It checks the all-gather, the block order in real batches and the GDPO path.

## Experiments and cost

**The measured unit is GRPO-v3-all:** 571 steps of 256 completions took 23.46 GPU-h. An evaluation is 0.50 GPU-h. A block run at the same budget is 571 steps, 2,284 blocks: 6,852 attributed and 2,284 attribute-free prompts, about 24 GPU-h. At GRPO-v3-all's 9,140 attributed prompts, it is about 31 GPU-h.

| arm | rewards and aggregation | what it isolates | GPU-h, train and evaluation |
|---|---|---|---|
| A0 | GRPO, `decision` + `format`, `sum_then_normalize` (GRPO-v3-all's recipe) | the reference on the same data | ~24 |
| A1 | GRPO + `consistency`, `sum_then_normalize` | the consistency reward | ~25 |
| A2 | **GDPO** + `consistency`, `normalize_then_sum` | GDPO, against A1 | ~25 |

Two ways to run it:

- **Cheapest: A2 alone, on round 3's data, against GRPO-v3-all.** About 26 GPU-h with the smoke run, and no new data. It shows whether consistency plus GDPO moves the TPR problem. It cannot separate the reward from the aggregation.
- **The ablation on the final data: A0, A1 and A2.** About 75 GPU-h, plus data. New pairs cost about 10 GPU-h per 6,000. For round 3's second batch that was teacher 6.06 (job 5116018), k=8 samples 2.25 (job 5116328) and k=32 hire top-up 1.94 (job 5116767). The jobs run one at a time.

A second seed of the chosen arm adds about 25 GPU-h.

## Risks

- **Sparsity.** It is the same as GRPO's for deterministic flips (above). The consistency reward changes what is rewarded, not how often a group splits.
- **A noisy target.** The attribute-free answer is itself sampled. The soft agreement over 16 completions damps the noise, but a pair whose attribute-free answers split 8–8 gives a weak target.
- **Dependence on TRL internals.** The design relies on TRL 1.15's sampler order, so the CPU test pins it, and an upgrade of TRL reruns it.
- **The anchor keeps some pull toward strictness.** With equal weights, its pull is half of what it is now. If TPR still falls, the next lever is its weight.

## Decisions for the user

1. A2 on round 3's data now, or the full ablation on the final data?
2. Which arms: A2 alone, A1 and A2, or all three?
3. Counterfactual grouping as a stretch, or not?
