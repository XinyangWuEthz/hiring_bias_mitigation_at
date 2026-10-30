# GRPO-improve: short experiments on why GRPO from SFT-v3 is unstable

**Status (2026-10-26, the user's decision): R3 is recorded as better than SFT-v3.** The model is
R3, GRPO-v3-all with the learning rate decayed linearly to 0 over its last 146 steps, final
adapter (step 571, declared before training): −0.71 pp unstable within-group sets at matched
hire rate, [−1.89, +0.39], every guardrail passing (TPR −5.03 pp, agreement −0.75 pp, merge kept
97.2%, undecided 0%; job 5146713). The user, translated from Chinese: "Δ −0.71 [−1.89, +0.39] can be regarded as an improvement."

What stays on the record:

- The standard's own verdict is "not better: the interval reaches 0"; the decision above is the
  user's, as for GRPO-v3-all's checkpoint 525 (`GRPO_V3.md`).
- One seed. The standard asks for a second seed with the same sign at ≤ 1 pp; none was run.
- The benchmark had been read many times in development; R3 is exploratory in that sense.
- Against GRPO-v3-all's final adapter the change is the operating point (TPR −5.0 against
  −10.1 pp), not a clearly larger gain (−0.71 against −0.56 pp).

Started 2026-10-25 on `xinyang/grpo-improve` (from main 4f9f970, with `xinyang/dapo` merged to
4b5061a for its checkpoint-trajectory scorer and rollout diagnostics). The user's goal: lower the
share of unstable decisions while keeping TPR and utility, under the comparison of
[`EVAL_STANDARD.md`](EVAL_STANDARD.md) unchanged. Training reward is a diagnostic, never the
success criterion.

**Budget (user, 2026-10-25).** 150 GPU-h in total, sampling, failed attempts and evaluation
included; at most 50 short experiments; at most 3 jobs at once and 3 retries. Costs are recorded
under `HBM_COST_ARM=grpo_improve`. The ledger is at the end of this page.

## 1. What is already known

| Work | State | Result | Source |
|---|---|---|---|
| GRPO-v3 (SFT-v3's own pairs, 187 steps) | done, valid | not better: +0.74 pp [−0.15, +1.63]; guardrails pass | `GRPO_V3.md` |
| GRPO-v3-all final, step 571 | done, valid | not better: −0.56 pp [−1.61, +0.41]; **TPR guardrail fails**, −10.11 pp | `GRPO_V3.md` |
| GRPO-v3-all checkpoint 525 | done, exploratory (chosen after the audit) | better: −1.02 pp [−1.99, −0.12]; TPR −9.30 pp; the user recorded GRPO as better | `GRPO_V3.md` |
| DAPO-v3-all (β 0, strict verdict balance) | done, no improvement | +0.85 pp [−0.23, +2.12]; TPR fails, −10.10 pp; pool ran dry at step 85 | `DAPO_V3.md` |
| Checkpoint trajectory, 20 checkpoints of GRPO-v3-all | done, valid diagnostic | full-benchmark TPR swings between −14.4 and +15.6 pp of SFT-v3's from one saved checkpoint to the next; first −10 pp failure at step 150 | `GRPO_CHECKPOINT_TRAJECTORY.md`, job 5127818 |
| Mean of the 7 late checkpoints' updates | done, no improvement on its preregistered test | c central, d′ below step 500's (−0.31 [−0.53, −0.10]); no margin pass, so no fairness verdict | `GRPO_CHECKPOINT_WEIGHT_AVERAGE_RESULTS.md`, job 5130000 |
| Margin monitor, 696 held-out rows | done, **calibration failed** | projects 5–6 pp too little TPR loss for DAPO, GRPO-v3-all and step 525; the RJDB/Djinni source mix explains most of it | `GRPO_MONITOR8.md`, job 5123830 |
| Direct benchmark-dev monitor (90 candidates) | done, **failed its replay** | dev −8 pp first fires at step 400; the full benchmark already failed −10 pp at 150 | `BENCHMARK_DEV_MONITOR.md` |
| Lower LR 3e-6 from checkpoint 425 | **failed twice before any update** | 5130709: a W&B startup race between ranks. 5131944: the startup gate found 52 of 31,050 decisions differing from the frozen cache; zero updates | `GRPO_LOW_LR_PILOT*_RESULTS.md` |
| GRPO-v3-new | stopped by the user at step 4 | nothing | `GRPO_V3.md`, round 2 |

Not yet tested at all: the effect of a lower learning rate, a reward that targets consistency
across variants (`GDPO_CONSISTENCY_PLAN.md` is a plan), a larger batch, a second seed of
GRPO-v3-all, and DAPO's verdict balance apart from its β 0.

## 2. Diagnosis from what is on disk (no GPU)

`scripts/grpo_trajectory_diagnostics.py` reads the 20 checkpoints' benchmark decisions (job
5127818), SFT-v3's op-margin-v2 margins and GRPO-v3-all's `log_history`. Outputs are under
`$HBM_OUTPUT_ROOT/outputs/grpo_improve/diag_grpo_v3_all/` (`trajectory_diagnostics.csv`,
`transitions.csv`, `train_log.csv`, `summary.json`).

It adds a **greedy proxy of the decision number**: the checkpoint's greedy decisions stand in for
its margin pass, and SFT-v3 is moved to the same hire count by
`operating_point_control.matched_comparison`, unchanged. Checked against the two official
reports: step 525, proxy −1.04 against official −1.02 pp; step 571, −0.71 against −0.56 pp;
SFT-v3 against itself, −0.07. It is a development reading, not a verdict.

**Explanation 1, rollouts without reward variance: supported.** The share of groups whose 16
rewards are identical is 0.58 over steps 1–100 and 0.83 over 101–125, and stays at 0.86–0.98
from step 150 on. A step trains 16 groups, so from about step 150 on a step has two or fewer
groups with any advantage.

**Explanation 2, updates too large for the signal: supported, in a specific form.** The
benchmark decisions move as one global threshold:

- In 14 of the 20 windows between saved checkpoints, at least 95% of the changed decisions go
  the same way (`one_way_share`). Steps 475→500 change 3,111 decisions, all to hire; steps
  500→525 change 3,232, all to reject.
- Criterion c ranges 0.41–0.99 across checkpoints, while d′ stays within 1.64–2.06.
- The proxy matched change wanders within −1.07 to +0.72 pp without a trend. At step 100 it is
  −0.64, at 571 −0.71.
- The changed decisions concentrate on prompts SFT-v3 is unsure about: 41–95% of them lie
  in SFT-v3's 10–90% hire-probability band, which holds 8.1% of all prompts.

**Explanation 3, the training distribution is not the evaluation target: supported.** On the
training rollouts the sampled hire rate where the anchor hires rises from 0.72 to 0.82–0.97 and
stays there, while the benchmark TPR swings. Across checkpoints the two are uncorrelated
(r = −0.04; window change against window change, r = −0.22). Training reward climbs, the
matched instability does not move.

**A data fact behind explanation 3.** On 3,366 of the 9,140 training rows (37%) the vote anchor
and SFT-v3's own attribute-free majority (8 samples at temperature 1, from the round-3 pools)
disagree: 1,580 hire-anchored rows where SFT-v3 rejects without the attribute, and 1,786
reject-anchored rows where it hires (`artifacts/semisynthetic-v3/grpo-v3-all-free/report.json`,
built by `scripts/build_grpo_free_target.py`). On those rows the reward teaches a different
hiring standard, not invariance to the attribute, and the two kinds push the decision bias in
opposite directions, about equally often.

The paired probe of 2026-10-22 points the same way (`$HBM_OUTPUT_ROOT/outputs/paired_probe/
20261022-dapo-final-sft-v3-k16/report.md`; 16 samples of each training row with and without
the attribute). Rows where all 16 attributed samples miss the anchor rise from 336 under SFT-v3
to 1,170 under DAPO-v3-all's final adapter, and 1,081 of the 1,170 also miss it on all 16
attribute-free samples. What RL moved on its own training rows is mostly the case judgement,
not the response to the attribute.

**One reading that ties them together, not confirmed.** By step ~125 the policy is sure of
almost every training prompt. From then on neither the reward nor the KL term (β 0.04 on a
per-token mean of ~38 tokens) holds the hire/reject bias where the benchmark's borderline
prompts sit, because no training prompt is near its own boundary. One or two informative groups
a step, at a constant lr of 3e-5, then move that bias at random. The operating point an audit
sees depends on where the walk happened to stop. That explains the TPR guardrail failures and
the verdict flip between checkpoints 525 and 571.

**A constraint on the obvious fix.** A lower learning rate meets the merge guardrail: the GRPO
pilot at 1e-5 kept 58% of its update in the bf16 merge, below the standard's 80%. A short run at
lower LR from SFT-v3 would not pass the merge check that the scorer enforces. That is why the
low-LR pilot continued from checkpoint 425.

## 3. The short-run protocol, shared by every arm

- **Start.** SFT-v3 (`outputs/sft/qwen3.5-9b_en_only_sft_v3_additional`), fresh LoRA, GRPO-v3-all's
  data (`configs/data/en_only_grpo_v3_all.yaml`, 9,140 prompts) and settings, unless the arm says
  otherwise. 200 optimizer steps; checkpoints every 25.
- **Readout.** Checkpoints 100, 125, 150, 175 and 200, scored on all 31,050 benchmark prompts by
  `scripts/checkpoint_benchmark_trajectory.py` (merged bf16, the audit's decoding, the 80% merge
  gate). For each: full TPR change against SFT-v3, the greedy proxy of the matched change, c, d′,
  raw unstable sets; dev and historical remainder separately.
- **Summary per arm, fixed now.**
  - TPR swing: the standard deviation of full TPR over the five checkpoints, and the largest
    change between neighbours. GRPO-v3-all at 100–200: 6.8 pp and 15.2 pp.
  - Mean proxy matched change over the five. GRPO-v3-all at 100–200: −0.52 pp.
  - Checkpoints below −10 pp TPR. GRPO-v3-all at 100–200: one (150).
- **What it is.** The benchmark has been read many times during development, so these are
  development readings on an exposed set, not tests. An arm that looks better goes to the
  standard's `evaluate_arm.sbatch` at a checkpoint declared before that audit, and is labelled
  exploratory there too.
- **Cost.** Training about 2.1 h on one 4-GH200 node (37 s a step) plus start-up, about 9 GPU-h;
  scoring five checkpoints about 2 GPU-h. About 11 GPU-h an arm.

## 4. Arms

Each arm is registered before it is submitted. Parameter arms change one setting of the control;
method arms are listed apart.

### A0 — control: GRPO-v3-all, seed 43 (parameter)

- **Hypothesis.** The TPR swing is a random walk driven by rollout noise, not by the data order.
- **Change.** `seed: 43` (data order, rollouts and monitor rows), `max_steps: 200`,
  `resume: false`, `rollout_diagnostics: {enabled: true}` (per-verdict counts; logging only).
  Config `configs/mitigation/grpo_improve/a0_seed43.yaml`.
- **Expected if it holds.** A TPR swing of the same size as seed 42's (SD ≥ 4 pp over 100–200),
  at different steps. The per-verdict counts of groups with an advantage, summed over a window,
  predict the sign of the TPR change over that window.
- **Against it.** A swing under 2 pp SD (the oscillation was seed 42's), or the same pattern at
  the same steps (data order drives it).
- **Cost.** About 11 GPU-h.
- **Why first.** Every other arm is read against it, and seed 42's checkpoints 25–75 were not kept.

### A1 — bias projection (method)

- **Hypothesis.** The TPR swing is the first-order push of each step on one shared decision bias:
  with one or two informative groups a step, the step's net push toward hire or reject is close
  to a coin flip. Removing that push removes the swing, and leaves the learning that orders
  prompts against each other.
- **Change.** `bias_projection: {enabled: true}` on A0's config (`mitigation/grpo_bias_projection.py`).
  With s_i = ±1 for a hire or reject decision, the gradient of the policy loss with respect to a
  bias on every prompt's decision logit is ½ Σ A_i s_i. Each step subtracts α (s_i − mean_g s)
  from every advantage, with one α for the whole gathered batch, so that Σ A′_i s_i = 0. Group
  sums stay zero, groups whose decisions agree are untouched, and a batch with informative
  groups on one verdict only still trains the difference between them. Detailed rollout
  diagnostics are off (the plain trainer only); `bias_projection/*` is logged instead. Config
  `configs/mitigation/grpo_improve/a1_bias_projection.yaml`; CPU tests
  `tests/test_grpo_bias_projection.py`.
- **Expected if it holds.** TPR SD over checkpoints 100–200 under 2 pp and no checkpoint below
  −8 pp, against A0's swing. The mean proxy matched change no worse than A0's (within 0.3 pp).
  The logged `push_before` changes sign from step to step without a trend.
- **Against it.** A swing within 30% of A0's: the drift does not come from the first-order bias
  push, and some other direction (feedback wording, the KL term, second-order effects) moves the
  operating point. Or a stable TPR with a proxy matched change clearly worse than A0's (by more
  than 0.5 pp): the bias push carried the fairness gain.
- **Cost.** About 11 GPU-h; the projection itself is two gathers a step.

### A2 — consistency reward (method)

- **Hypothesis.** Rewarding agreement with the model's own attribute-free answer targets the
  evaluated quantity (explanation 3), and on the 37% of rows where the anchor contradicts that
  answer it cancels the push toward a new standard, which is also a source of the bias swing.
- **Change.** A third reward, `grpo.free_consistency_reward`: SFT-v3's sampled attribute-free
  hire share for the pair when the completion hires, one minus it when it rejects. Summed with
  the decision and format rewards before the usual group normalisation (`sum_then_normalize`,
  weight 1 each). The data are grpo-v3-all's rows, order and splits unchanged, plus
  `free_hire_rate` (`configs/data/en_only_grpo_v3_all_free.yaml`). Static target: SFT-v3's
  answer, not the current policy's; the plan's on-policy pair blocks
  (`GDPO_CONSISTENCY_PLAN.md`) are left for later. Config
  `configs/mitigation/grpo_improve/a2_consistency.yaml`; tests `tests/test_grpo_free_target.py`.
- **Expected if it holds.** Mean proxy matched change at least 0.3 pp better than A0's, with TPR
  change against SFT-v3 within ±5 pp at every scored checkpoint.
- **Against it.** A proxy matched change within 0.3 pp of A0's or worse: the conflicting rows
  are not what keeps GRPO from adding invariance. A TPR drop past −8 pp: the target pulls the
  operating point.
- **Cost.** About 11 GPU-h; the reward adds a lookup.

### Later candidates, not registered

- A1 + A2 together, if each helps on its own.
- A bias anchor on held-out Djinni prompts in SFT-v3's 10–90% band (penalise the policy's mean
  decision margin moving from SFT-v3's), if A1 shows the drift is not first-order.
- Twice the prompts per step at the same lr (explanation 2 through noise), if A0's swing is
  large and A1 does not remove it.

## 5. Round 1 result (A0, A1, A2)

**In one sentence.** None of the three short runs lowers the proxy matched instability against
SFT-v3, and the bias projection doubled the TPR swing instead of removing it. Development
readings on the exposed benchmark, one seed each.

| arm (200 steps from SFT-v3) | TPR SD, 100–200 | largest neighbour TPR change | TPR change range vs SFT-v3 | checkpoints < −10 pp | mean proxy matched change | proxy range | raw unstable, mean | zero-std groups, mean |
|---|---|---|---|---|---|---|---|---|
| GRPO-v3-all, seed 42 (history, job 5127818) | 6.8 pp | 15.2 | −10.2 to +5.0 | 1 | −0.53 pp | −0.88 to +0.04 | 1.71% | 0.81 |
| A0 control, seed 43 | 5.9 | 13.9 | −2.5 to +11.4 | 0 | −0.12 | −0.55 to +0.39 | 1.80% | 0.79 |
| A1 bias projection | **12.2** | **27.3** | −15.6 to +16.2 | 2 | −0.05 | −0.65 to +0.24 | 1.75% | 0.78 |
| A2 consistency reward | 5.2 | 7.7 | −5.6 to +7.5 | 0 | +0.05 | −0.46 to +0.97 | 2.04% | 0.85 |

Source: `$HBM_OUTPUT_ROOT/outputs/grpo_improve/compare_round1/{summary.csv,curves.csv,curves.svg}`;
per checkpoint `outputs/grpo_improve/diag_{a0,a1,a2}/trajectory_diagnostics.csv` and
`transitions.csv`; raw decisions `outputs/checkpoint_benchmark/grpo_improve_{a0_seed43,
a1_bias_projection,a2_consistency}/steps/step-NNNN/raw.parquet`. SFT-v3: TPR 57.73%, raw
unstable 1.85%.

**Against each registration.**

- **A0.** For: the swing reproduces at seed 43 (SD 5.9 pp ≥ 4), at other steps and in the other
  direction (lenient: +11.4 pp at step 200 against seed 42's −10.2 at 150). Against: the
  per-verdict counts of mixed groups do not predict the sign of a window's TPR change. Over
  176–200 both verdicts had 25 mixed groups and TPR rose 13.9 pp; correlation of the net count
  with the TPR change over the five windows −0.14 (`rollout_diagnostics.jsonl`). In training
  rollouts, the sampled hire rate where the anchor hires fell from 0.58 to 0.01 by step ~19 and
  was back at 0.49 by step ~28 (job 5140665's log).
- **A1. Hypothesis refuted.** With the first-order push on a shared decision bias set to zero
  every step (logged `push_after` ~1e-15), the swing doubled. The registered reading: the
  drift does not come from the first-order bias push. The likelier role of that push is a
  restoring force: when the policy drifts to reject, the hire-anchored groups become
  informative and push back. Removing it removed the brake, not the engine.
- **A2. Not better.** Mean proxy +0.05 against A0's −0.12 (registered: at least 0.3 pp better),
  and TPR leaves ±5 pp at two checkpoints. It changes fewer decisions against SFT-v3 (656–1,302
  against A0's 1,121–2,031) and its largest neighbour change is about half A0's.

**What this says about the three explanations.**

- *Rollout signal (1).* Unchanged by any arm: 0.78–0.85 zero-std groups over 100–200.
- *Update size (2).* The operating point moves by several pp per 25 steps in every arm, and
  not through the decision-bias gradient. Candidates left: Adam's per-parameter scaling with a
  constant lr (the update does not shrink when the signal does, and its direction is not the
  gradient's), the KL term, or learning on feedback tokens. Not separated yet.
- *Target mismatch (3).* Supported again: the training-rollout hire rates and the benchmark TPR
  stay uncorrelated in every arm (r between −0.14 and 0.19).
- *Seed noise in the readout.* The same config at seeds 42 and 43 gives mean proxies of −0.53
  and −0.12. A difference between two single-seed short arms below ~0.5 pp is not readable.

**Two facts from existing outputs, CPU only.**

- Over GRPO-v3-all's whole history the proxy is −0.53 pp on average at 100–200, +0.09 at
  225–400 and −0.83 at 425–571, where all seven checkpoints sit at −0.70 or below while TPR
  swings by 30 pp (`compare_history/`). Whatever gain GRPO has shows late and survives the
  operating-point swing.
- The mean-of-7 weight average (`checkpoint_average/.../average/raw.parquet`): TPR −3.0 pp,
  proxy −0.32 pp [−1.37, +0.67]. Averaging centred the operating point and lost about half of
  the late checkpoints' gain.

**Which sets move (CPU, the same 20 checkpoints).** SFT-v3 has 50 unstable within-group sets.
Across GRPO-v3-all's seven late checkpoints (425–571), 27 of them are stable in all seven and
every one of the 50 in at least four; but only one new unstable set recurs in at least four
of the seven. The early checkpoints (100–200) look the same: 18 of the 50 fixed in all five,
5 new sets recurring in at least three. GRPO's fixes persist; its breaks are transient and
differ from checkpoint to checkpoint, as the threshold walk would make them. (Computed inline
from the `steps/step-NNNN/raw.parquet` files with `reference_metrics.SET`; to be scripted if
it enters a claim.)

## 6. Round 2: registrations

Both arms start from SFT-v3 for 200 steps and are read as in section 3, against A0.

### R2-a — operating-point anchor (method)

- **Hypothesis.** The benchmark's threshold drifts because nothing holds it once the training
  rows are settled (round 1: A1's removal of the restoring push doubled the swing). A restoring
  force on held-out prompts like the benchmark's borderline ones holds it, and the late GRPO
  gain does not depend on the drift.
- **Evidence before running.** On the monitor calibration's seven finished arms (job 5123830),
  the mean SFT-relative margin shift on these 222 rows correlates 0.93 with the arm's
  benchmark TPR change (development evidence, 30 candidates).
- **Change.** A0's config plus `operating_point_anchor` (`mitigation/grpo_operating_anchor.py`):
  each micro-step and rank forwards 4 anchor prompts with the decision prefix, and the loss
  gains 10 · (mean p_policy − mean p_SFT)², p the hire probability of the decision token, p_SFT
  with the LoRA disabled. The anchor rows (`scripts/build_grpo_anchor_rows.py` →
  `artifacts/semisynthetic-v3/grpo-anchor-djinni-band/`) are the 222 Djinni monitor rows, 30
  held-out candidates, with SFT-v3's margin in its 10–90% band; none is in training or the
  benchmark. Weight 10 is set before running: a shift of 0.1 in mean hire probability costs
  0.1, five times GRPO-v3-all's typical loss. Logged as `anchor/shift`, `anchor/penalty`.
- **Expected if it holds.** TPR SD over 100–200 under 3 pp, every checkpoint within ±8 pp, and
  the proxy matched change within 0.3 pp of A0's or better. `anchor/shift` stays under 0.05.
- **Against it.** A swing within 30% of A0's (5.9 pp): held-out Djinni rows do not carry the
  benchmark's threshold. A stable TPR with a proxy worse than A0's by more than 0.5 pp: the
  anchor costs the fairness gain.
- **Check before the run.** A three-step smoke on `debug` (`r2a_anchor_smoke.yaml`) for the
  anchor's forward and backward under DDP. CPU tests: `tests/test_grpo_operating_anchor.py`.
- **Cost.** About 12 GPU-h with the smoke; the anchor adds 4 short forwards per micro-step.
- **Amendment after a failure, before any result (2026-10-25, 15:30).** Run 5141612 hung at step
  2: a DDP bucket all-reduce that rank 3 never reached, NCCL timeout, 2.72 GPU-h lost; nothing
  is evaluated. The penalty was added to each micro-step's loss with its forward through the
  unwrapped module beside DDP's; the three-step smoke (5141458, 0.79 GPU-h) had passed. The
  anchor is now a callback after gradient clipping (`grpo_operating_anchor.OperatingAnchorCallback`):
  its gradient comes from `torch.autograd.grad`, which DDP's hooks never see, and is summed over
  ranks with one all_reduce per step. `rows_per_step` now counts rows per optimizer step and rank:
  8, so 32 rows a step against the first version's 128, at a quarter of its forward cost. The
  weight, rows, hypothesis and criteria are unchanged. A six-step smoke checks it first. This
  is retry 1 of 3.
- **Second amendment, from the smoke, before any result.** The six-step smoke (5142306, 0.83
  GPU-h) ran clean at 25–35 s a step. Its `anchor/shift` changed sign every step at about
  ±0.014, the sampling noise of 32 rows, while the anchor's gradient norm was 3–7 against
  the policy gradient's 0.5–0.8 after clipping. At weight 10 the anchor would add a noise
  direction ten times the policy's every step. The weight is now **1**: about the policy
  gradient's size at noise level, and about 4.6 at a real shift of 0.1.

### R2-b — a third of the learning rate (parameter)

- **Hypothesis.** The operating point moves because each update is too large for the signal
  (explanation 2), whatever its direction.
- **Change.** `learning_rate: 1.0e-05` on A0's config, nothing else (`r2b_lr1e5.yaml`).
- **Expected if it holds.** TPR SD under 2.5 pp and fewer decisions changed against SFT-v3,
  roughly in proportion to the learning rate.
- **Against it.** A swing within 30% of A0's: the step size does not set the swing, and the
  walk comes from something that does not scale with lr (Adam's normalised steps hold
  their size only in proportion to lr, so this would point at the data or the sampling).
- **What it cannot be.** An arm of the standard: the bf16 merge will likely keep under 80% of
  so small an update (the pilot at 1e-5 kept 58%). Its checkpoints are scored with a declared
  floor of 30% (`min_merge_retention` in its trajectory config) and the kept share is reported
  beside each reading. The served model is then only part of the trained one, which weakens
  any null.
- **Cost.** About 11 GPU-h.

## 7. Round 2 results

### R2-b — lr 1e-5: the swing shrinks, less than in proportion (training 5141460, scoring 5142302)

| | TPR SD, 100–200 | largest neighbour change | TPR change range | mean proxy matched | decisions changed vs SFT-v3 | changed in SFT-v3's 10–90% band | merge kept |
|---|---|---|---|---|---|---|---|
| A0, lr 3e-5 | 5.9 pp | 13.9 | −2.5 to +11.4 | −0.12 pp | 1,121–2,031 | 44–72% | ≥ 80% (gate) |
| R2-b, lr 1e-5 | 3.4 | 5.6 | −3.7 to +4.3 | −0.31 | 543–750 | 87–92% | 49–63% |

(`outputs/grpo_improve/diag_r2b/`, `compare_round2_partial/`; merge kept from each step's
`result.json`.)

- **Against the registration.** Neither line is crossed: the SD is above the 2.5 pp of "for"
  and below the 4.1 pp of "against". A third of the learning rate takes 42% off the SD and 60%
  off the largest neighbour change.
- **Reading.** Partial support for explanation 2: the swing depends on the step size, but
  less than linearly. The served models keep only half of their updates in the bf16 merge, so
  the readout understates what was trained; the training-rollout hire rates are, for once,
  correlated with the benchmark TPR (r = −0.86, five points), sign unexplained.
- **Not a candidate.** Every checkpoint fails the standard's 80% merge guardrail.

### R2-a — operating-point anchor: the anchor rows held, the benchmark did not (training 5142344, scoring 5143013, 5143090)

| | TPR SD | largest neighbour change | TPR change range | below −8 pp | mean proxy matched | merge kept |
|---|---|---|---|---|---|---|
| A0 | 5.9 pp (5 checkpoints) | 13.9 | −2.5 to +11.4 | 0 | −0.12 pp | ≥ 80% |
| R2-a | 5.6 (4 checkpoints) | 12.5 | −8.1 to +4.3 | 1 (step 150) | −0.19 | 76.4% at 100 (not scored), 80.4–85.3% after |

- **Hypothesis refuted.** The SD is within 30% of A0's. Checkpoint 100 failed the 80% merge gate
  (the anchor held the update small early) and was not scored.
- **The anchor did its own job.** `anchor/shift`, the mean hire-probability shift on the 222
  held-out Djinni rows, stayed between −0.014 and +0.011 per 25-step window through step 175
  (+0.052 over 176–200), while the benchmark TPR fell 12.5 pp from step 125 to 150 and rose 10.7
  pp to 175. The held-out rows SFT-v3 is unsure of do not carry the benchmark's 25-step swings.
  The 0.93 correlation across finished arms was a correlation of large, lasting shifts.
- **Side effects.** Zero-std groups 0.23 over steps 1–25 against A0's 0.72 (A0 had collapsed to
  reject there), training-rollout hire rates steadier; neither showed on the benchmark.
- Outputs: `outputs/grpo_improve/diag_r2a/`, `compare_round2/`. The first scoring job
  (5143008) failed in 10 s on a manifest the login node had not finished writing; the second
  (5143013) stopped at checkpoint 100's merge gate after scoring 125–175; 5143090 scored 200.

### Round 2 in one paragraph

The step size moves the swing (R2-b: −42% SD at a third of the lr, at the cost of the merge
guardrail); restoring forces do not, whether on the shared decision bias (A1, the swing
doubled) or on held-out borderline rows (R2-a, unchanged). Whatever moves the benchmark
threshold every 25 steps is not visible in the training rollouts or in the held-out rows.

## 8. Round 3: registration

### R3 — a full epoch with the learning rate decayed at the end (parameter)

- **Why this one.** Three facts so far. GRPO's gain shows late: GRPO-v3-all's seven checkpoints
  425–571 all sit at a proxy of −0.70 pp or below (section 5). Its fixes persist across
  checkpoints while its breaks are transient (section 5, "Which sets move"). And the swing
  shrinks with the step size (R2-b), while neither a decision-bias projection (A1) nor an
  anchor on held-out rows (R2-a) stops it. Decaying the learning rate over the late phase
  should freeze the walk and keep the fixes.
- **Hypothesis.** GRPO-v3-all with its learning rate decayed linearly to 0 over its last 146
  steps beats SFT-v3 under the standard at its final checkpoint, with every guardrail passing.
- **Change.** GRPO-v3-all's config (`qwen3.5-9b_en_only_grpo_v3_all_gh200.yaml`), seed 42 and
  every setting the same, except `lr_scheduler_type: warmup_stable_decay` with
  `num_decay_steps: 146` (3e-5 to step 425, then linear to 0 at 571). Config
  `configs/mitigation/grpo_improve/r3_wsd.yaml`. *Correction after the run:* steps 1–425 turned
  out to be bit-identical to GRPO-v3-all's (the adapters of checkpoints 100, 300 and 425 have
  the same SHA-256); they first differ at 450. Training is deterministic at a fixed seed, so
  R3 is GRPO-v3-all itself with the decay applied from step 425: a paired comparison.
- **Control and decision.** SFT-v3, by `evaluate_arm.sbatch` (op-margin-v2 compare-arms on all
  31,050 prompts and 2,700 sets). **The audited checkpoint is the final adapter, step 571,
  declared now.** Audit config `configs/audit/qwen3.5-9b_en_grpo_improve_r3_wsd_gh200.yaml`.
  The benchmark has been read many times in development, so the verdict is labelled
  exploratory, as checkpoint 525's was.
- **For.** Δ < 0 with the 95% interval below 0, all four guardrails passing. Over 475, 500, 525
  and 550 (trajectory scorer), the TPR SD falls from its 425–475 level and stays under 3 pp.
- **Against.** An interval reaching 0, a Δ above 0, or a failed guardrail. A late TPR swing like
  GRPO-v3-all's (SD over 4 pp at 475–550) would say the decay does not freeze the walk.
- **Within-run readings.** Checkpoints 425, 450, 475, 500, 525 and 550 on the trajectory scorer
  (`configs/analysis/grpo_improve_r3_wsd_trajectory.json`); they do not choose the checkpoint.
- **Second seed.** If the verdict is "better" by 1 pp or less, the standard asks for a second
  seed. That is another ~25 GPU-h, inside the budget; it is run only if the first seed passes.
- **Cost.** Training ≈ 571 steps × 33 s + start-up ≈ 5.5 h on 4 GPUs ≈ 22 GPU-h; trajectory
  scoring ≈ 2.5; evaluate_arm ≈ 1.2. About 26 GPU-h. Time limit 7 h 30 min, `resume: true`.

## 9. Round 3 result: R3 freezes the operating point, the gain stays inside the noise

**In one sentence.** By the standard R3 is not better than SFT-v3, −0.71 pp [−1.89, +0.39]
at the declared final checkpoint, with every guardrail passing; against GRPO-v3-all's own
final adapter it keeps the operating point (TPR −5.0 against −10.1 pp) but does not widen the
gain. Evaluation job 5146713, commit 3d89a72, `conforms: true`.

| step 571, against SFT-v3 | GRPO-v3-all (constant lr) | R3 (lr decayed 425→571) |
|---|---|---|
| matched change, pp [95% CI] | −0.56 [−1.61, +0.41] | **−0.71 [−1.89, +0.39]** |
| verdict | not better (TPR guardrail fails) | not better: the interval reaches 0 |
| TPR change vs SFT-v3 | −10.11 pp | **−5.03 pp** |
| agreement with GPT-4o change | −2.70 pp | −0.75 pp |
| hire rate, all prompts | 18.71% | 20.15% |
| raw unstable within-group sets | 1.11% | 1.52% |
| merge kept | 97.7% | 97.2% |

(`reports/reference_metrics.csv`, `reports/operating_point/v2/…r3_wsd--vs--…sft_v3_additional.json`.)

**Within the run** (trajectory job 5146764; `outputs/grpo_improve/diag_r3/`, `compare_round3/`):

| checkpoints 425–571 (7) | TPR SD | largest neighbour change | TPR change range | below −8 pp | mean proxy matched |
|---|---|---|---|---|---|
| GRPO-v3-all | 10.4 pp | 24.9 | −14.4 to +15.6 | 4 | −0.83 pp |
| R3 | **3.0** | 5.9 | −5.0 to +2.2 | 0 | −0.44 |

Over 475–550, the registered window, R3's TPR SD is 2.8 pp: under the 3 pp of "for". From step
525 on, fewer than 125 decisions change per 25 steps, against 600–3,200 at constant lr.

- **Against the registration.** The swing line holds; the decision line fails (the interval
  reaches 0). Hypothesis not supported as a whole.
- **Reading.** Decaying the lr freezes the threshold walk and with it the TPR lottery, which is
  what made GRPO-v3-all fail its guardrail. It does not add invariance: the proxy over the
  late phase is weaker than at constant lr (−0.44 against −0.83), as the weight average's was
  (−0.32). Not confirmed which way: a difference of 0.4 pp is inside one seed's spread.
- **Checked: the proxy gain is not an operating-point artefact.** Over all 51 scored
  checkpoints of the seven arms, the proxy correlates −0.09 with |TPR change| and 0.84 with the
  arm's own raw instability; SFT-v3's matched instability stays at 1.9–2.3% whichever way the
  arm's TPR sits.

## 10. Round 4: registration (cancelled)

### R4 — tipped rows only, with R3's late decay (method: the data)

- **Why.** The one hypothesis on explanation 3 still untested at the standard: on 37% of the
  training rows the anchor contradicts SFT-v3's own attribute-free answer, so the reward teaches
  a standard, not invariance (section 2). A2 tried to cancel those rows with a reward and did
  not move the proxy; removing them is the direct test. R3's decay is kept because it is the
  one change that kept the TPR guardrail.
- **Hypothesis.** GRPO on rows where the attribute flips SFT-v3's decision, and only those,
  lowers matched instability against SFT-v3 more than R3 does, and beats SFT-v3 under the
  standard at its final checkpoint.
- **Change.** R3's config with the data replaced by grpo-v3-all's tipped rows, balanced: all
  2,077 tipped reject rows and 2,077 of the 2,985 tipped hire rows (seed 42),
  `scripts/build_grpo_tipped_subset.py` → `artifacts/semisynthetic-v3/grpo-v3-all-tipped/`
  (919 pairs). One epoch is 259 steps; the decay covers the last 67 (26%, as R3's 146 of
  571). Validation and monitor rows unchanged. Config `configs/mitigation/grpo_improve/r4_tipped.yaml`.
- **Control and decision.** SFT-v3 by `evaluate_arm.sbatch`; the final adapter, declared now
  (`configs/audit/qwen3.5-9b_en_grpo_improve_r4_tipped_gh200.yaml`). R3 is context.
  Exploratory label as before (exposed benchmark).
- **For.** Δ < 0 with the interval below 0 and every guardrail passing.
- **Against.** The interval reaches 0, Δ above R3's −0.71, or a guardrail fails. 4,154 rows is
  45% of R3's data, so a weaker result could also be the smaller dose; that would be stated.
- **Cost.** ≈ 259 × 36 s + start-up ≈ 2.8 h on 4 GPUs ≈ 11 GPU-h, plus 0.5 for the evaluation.
  No trajectory scoring.
- **Cancelled (2026-10-26, 09:38), no result.** After the user recorded R3 as better, they asked
  to cancel R4. Job 5147006 was at step 25 of 259 when cancelled, with no checkpoint saved;
  nothing of it is evaluated, and it says nothing about the hypothesis. 1.36 GPU-h. The data
  (`artifacts/semisynthetic-v3/grpo-v3-all-tipped/`), configs and this registration stay as
  built, for a later run.

## 11. Where this leaves the goal

What the evidence supports:

1. *Rollout signal (explanation 1).* Real and unfixed by any arm: zero-variance groups pass 0.8
   by step ~125 in every arm. More signal on its own did not help (A2, R2-a had more signal
   early, no better proxy; DAPO before this round).
2. *Update size (explanation 2).* The operating-point swing scales with the step size (R2-b,
   R3) and not with the decision-bias gradient (A1) or an anchor (R2-a). Decaying the lr is
   the one change that removed the TPR guardrail failure without breaking another guardrail.
3. *Target mismatch (explanation 3).* The benchmark's threshold moves while the training
   rollouts and held-out borderline rows stay put; 37% of the training rows reward a different
   standard, not invariance; a consistency reward against SFT-v3's own answer did not move the
   proxy (A2).
4. *Readout resolution.* Two seeds of one config differ by 0.4 pp of mean proxy; the standard's
   interval is ±1.1 pp. Every GRPO variant here sits at −1.0 to +0.1 pp. Within this
   benchmark and one seed per arm, no variant still on the list is likely to reach an
   interval below 0.

**Which benchmark rows swing (CPU, GRPO-v3-all's 20 checkpoints; computed inline).** 3,215 of
31,050 rows flip at least 4 times over the 19 windows. They come in whole pairs: 68 of the 450
pairs, with a median 92% of a pair's 69 variants swinging together, so the swing is a
case-level judgement, not a response to the attribute, which is why it moves TPR far more
than instability. The pairs are ones SFT-v3 is unsure of (median |margin| 1.75 against 14.5
elsewhere), 76% of them GPT-4o hires, spread over 55 candidates (the top 10 carry 34%). One
reading of R2-a's failure, not confirmed: its 30 anchor pairs are few enough for the penalty
to be met on those pairs alone, without holding a threshold shared with these 68.

What it does not support: any GRPO variant beating SFT-v3 under the standard at a declared
checkpoint. Checkpoint 525's "better" remains exploratory, and R3 at the same seed did not
reproduce it at its final step.

## 12. Ledger

| # | Arm | Job(s) | GPU-h | State |
|---|---|---|---|---|
| — | CPU diagnosis (section 2) | none | 0 | done |
| 1 | A0 control: training, scoring | 5140665, 5141030 | 8.47 + 2.03 | done |
| 2 | A1 bias projection: training, scoring | 5140670, 5141105 | 8.44 + 2.04 | done, refuted |
| 3 | A2 consistency reward: training, scoring | 5140696, 5141121 | 8.40 + 2.03 | done, not better |
| 4 | R2-a anchor, first smoke (debug) | 5141458 | 0.79 | passed 3 steps |
| 5 | R2-a anchor, first run | 5141612 | 2.72 | **failed**: DDP all-reduce hang at step 2 |
| 6 | R2-b lr 1e-5: training, scoring | 5141460, 5142302 | 8.47 + 2.04 | done, partial support |
| 7 | R2-a anchor, second smoke (callback) | 5142306 | 0.83 | passed 6 steps; weight amended to 1 |
| 8 | R2-a anchor v2: training, scoring (3 jobs) | 5142344, 5143008, 5143013, 5143090 | 11.00 + 0.00 + 1.06 + 0.26 | done, refuted (retry 1 of 3) |
| 9 | R3 lr decay, full epoch: training, evaluate_arm, trajectory | 5143102, 5146713, 5146764 | 23.66 + 0.50 + 2.06 | done, not better; guardrails pass |
| 10 | R4 tipped rows + decay: training | 5147006 | 1.36 | **cancelled** at step 25 by the user; no result |
| | **Total** | | **86.2 of 150** | 8 arms of 50 (7 with a result), 1 retry |
