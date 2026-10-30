# DAPO-improve: DAPO-v3-all with GRPO-improve's learning-rate shape

**Result (2026-10-26).** R3's learning-rate shape turns DAPO-v3-all around: D1 (lr held to step
62, decayed to 0 at 84) reaches −0.83 pp [−1.95, +0.31] against DAPO-v3-all's +0.85 pp
[−0.23, +2.12], and passes the TPR guardrail it failed (−5.42 against −10.10 pp). By the
standard it is still not better: the interval reaches 0. Adding R3's KL term (D2) does not
help: −0.34 pp [−1.49, +0.83]. One seed each; DAPO training is not bit-deterministic.

Started 2026-10-26 on `xinyang/dapo` (with `xinyang/grpo-improve` merged at b39c501), at the
user's request, translated from Chinese: "Use this lr configuration and spend the remaining time exploring how to improve DAPO." "This lr configuration" is GRPO-improve
R3's (`GRPO_IMPROVE.md`): the learning rate held at 3e-5, then decayed linearly to 0 over the
last 26% of the run. R3 was recorded as better than SFT-v3 on the user's decision (−0.71 pp
[−1.89, +0.39], every guardrail passing). The comparison is [`EVAL_STANDARD.md`](EVAL_STANDARD.md),
unchanged; the reference arm is SFT-v3.

**Budget.** What is left of GRPO-improve's 150 GPU-h, 3 jobs at once, 3 retries (one used):
63.0 GPU-h at the start of this page. Costs go under `HBM_COST_ARM=dapo_improve`.

## 1. What DAPO-v3-all did (inference only, job 5147613)

DAPO-v3-all (`DAPO_V3.md`; training 5119438, evaluation 5120574) stopped dry at step 85 and
is not better: +0.85 pp [−0.23, +2.12], TPR −10.10 pp (guardrail fails). Its checkpoints 24, 48
and 72 were scored on all 31,050 benchmark prompts by `scripts/checkpoint_benchmark_trajectory.py`
(`configs/analysis/dapo_v3_all_trajectory.json`, declared merge floor 30%: the merges kept
62.5%, 77.8% and 85.0% of the updates); step 85 is the audit's own output.
`scripts/grpo_trajectory_diagnostics.py` → `$HBM_OUTPUT_ROOT/outputs/dapo_improve/diag_dapo_v3_all/`.

| step | TPR vs SFT-v3 | proxy matched change | raw unstable | trained-batch grad norm (pre-clip, window mean) |
|---|---|---|---|---|
| 24 | +1.0 pp | −0.58 pp | 1.74% | 0.59 |
| 48 | −8.7 | −0.36 | 1.67% | 1.00 |
| 72 | −8.3 | −0.40 | 1.56% | 2.94 |
| 85 | −10.1 | +0.93 (official +0.85) | 2.22% | 7.46 |

- The TPR loss is one move, 24→48: 1,442 decisions change, 98% of them to reject. It then
  stays at −8 to −10 pp: a level shift, not GRPO's back-and-forth swing.
- The instability is lost in the last 13 steps: the proxy goes from −0.40 to +0.93 while
  the gradient norm before clipping climbs from ~1 to 7.5, as the dry pool leaves fewer groups
  per step (DAPO normalises by the kept tokens) and β = 0 leaves nothing to hold the policy.

## 2. D1 — R3's learning-rate shape on DAPO's own horizon (parameter)

- **Hypothesis.** Decaying the learning rate over DAPO's last 26% removes the late damage (the
  72→85 loss of stability) and keeps checkpoint 72's state, at −8 pp TPR, inside the guardrail.
- **Change.** DAPO-v3-all's config with `max_steps: 84` (it stopped dry at 84–85) and
  `warmup_stable_decay`, `num_decay_steps: 22`: 3e-5 to step 62, then linear to 0 at 84.
  Nothing else (`configs/mitigation/dapo/dapo_improve_d1_decay.yaml`). GRPO training was
  bit-deterministic at a fixed seed (`GRPO_IMPROVE.md`, R3); if DAPO's is too, steps 1–62
  repeat DAPO-v3-all's exactly, and the checkpoints 24 and 48 must hash the same. That is
  checked, not assumed.
- **Control and decision.** SFT-v3 by `evaluate_arm.sbatch`; the final adapter (step 84, or the
  dry stop if it comes first), declared now; audit
  `configs/audit/qwen3.5-9b_en_dapo_improve_d1_decay_gh200.yaml`. DAPO-v3-all and R3 are
  context. Exploratory label (exposed benchmark).
- **For.** Δ < 0 with the interval below 0 and every guardrail passing. Context reading: Δ
  below DAPO-v3-all's +0.85 and TPR above −10 pp.
- **Against.** Δ ≥ DAPO-v3-all's +0.85, or a failed TPR guardrail: the late decay does not
  undo what DAPO's last steps do.
- **Cost.** DAPO-v3-all trained 85 steps in 65 min; ≈ 5 GPU-h with start-up, 0.5 for the
  evaluation.

## 3. D2 — D1 with the KL term back (parameter)

- **Why.** R3, the arm recorded as better, is GRPO with β 0.04 toward SFT-v3; DAPO-v3-all ran at
  β 0 (the user's choice for that arm, `DAPO_V3.md`). Both of DAPO-v3-all's failures, the
  24→48 move to reject and the late loss of stability with a rising gradient norm, are what a
  missing pull toward the parent would allow. D1 tests the lr shape alone; D2 adds the one
  other setting R3 has and DAPO lacks.
- **Hypothesis.** With β 0.04 and D1's schedule, DAPO keeps its TPR within −5 pp and does not
  lose stability late.
- **Change.** `beta: 0.04` on D1's config, nothing else
  (`configs/mitigation/dapo/dapo_improve_d2_decay_kl.yaml`; audit
  `configs/audit/qwen3.5-9b_en_dapo_improve_d2_decay_kl_gh200.yaml`). The dry-pool stop and the
  verdict balance stay on.
- **Decision.** As D1: the final adapter by `evaluate_arm.sbatch` against SFT-v3, declared now.
- **For.** Δ < 0 with the interval below 0, every guardrail passing. Context: TPR change above
  D1's and Δ below D1's.
- **Against.** Δ and TPR no better than D1's: the KL term is not what DAPO is missing.
- **Cost.** ≈ 6 GPU-h with the reference forward, 0.5 for the evaluation. Runs beside D1.

## 4. Results (training 5147717 and 5147722, evaluation 5148330 and 5148359)

Every comparison ran at commit b8d106f with no uncommitted code, `conforms: true`, on all
31,050 prompts and 2,700 sets (`reports/operating_point/v2/…dapo_improve_d{1_decay,2_decay_kl}--vs--…sft_v3_additional.json`;
`reports/reference_metrics.csv`). The audited models are the final adapters, step 84, as
declared; both runs reached `max_steps` (`status: done`), not the dry-pool stop.

| arm, against SFT-v3 | matched change, pp [95% CI] | TPR | agreement | merge kept | hire rate | raw unstable | verdict |
|---|---|---|---|---|---|---|---|
| DAPO-v3-all (constant lr, β 0) | +0.85 [−0.23, +2.12] | −10.10 pp | −2.03 pp | 87.4% | 18.0% (margin) | 2.22% | not better, TPR fails |
| **D1** (decay 62→84, β 0) | **−0.83 [−1.95, +0.31]** | **−5.42** | −0.97 | 85.0% | 20.11% | 1.37% | not better: the interval reaches 0 |
| D2 (D1 + β 0.04) | −0.34 [−1.49, +0.83] | −5.78 | −1.28 | 84.7% | 20.18% | 1.85% | not better: the interval reaches 0 |
| context: GRPO-improve R3 | −0.71 [−1.89, +0.39] | −5.03 | −0.75 | 97.2% | 20.15% | 1.52% | not better; recorded as better by the user |

- **D1 against its registration.** For needs the interval below 0: not met. Against needs
  Δ ≥ +0.85 or a TPR failure: not met either. D1 is the best point estimate of any arm in
  GRPO-improve or DAPO-improve, and on its context reading (Δ below DAPO-v3-all's, TPR above
  −10 pp) it holds.
- **D2 against its registration.** Against: Δ and TPR are no better than D1's. The KL term is
  not what DAPO is missing; at β 0.04 the point estimate is half of D1's.
- **Not a paired comparison.** Unlike GRPO (`GRPO_IMPROVE.md`, R3), DAPO training is not
  bit-deterministic at a fixed seed: D1's checkpoint 24 already hashes differently from
  DAPO-v3-all's. So D1 against DAPO-v3-all mixes the schedule with a new draw of the run; the
  1.7 pp between them is larger than the 0.4 pp seed spread seen in GRPO-improve, but one draw
  each.
- **Not measured.** No checkpoint trajectory of D1 or D2 (whether the decay window is where the
  difference arises), no second seed.

**Cost.** 0.80 (DAPO-v3-all's checkpoints) + 4.22 + 4.51 (training) + 0.50 + 0.50 (evaluation) =
10.5 GPU-h under `dapo_improve`. With GRPO-improve's 86.2: **96.7 of 150 GPU-h** in all.

## 5. Reproducing

```bash
export REPO=$PWD HBM_COST_ARM=dapo_improve
CONFIG=configs/mitigation/dapo/dapo_improve_d1_decay.yaml \
  scripts/cscs/submit.sh -p normal -t 02:30:00 scripts/cscs/train_dapo.sbatch
AUDIT=configs/audit/qwen3.5-9b_en_dapo_improve_d1_decay_gh200.yaml \
AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh scripts/cscs/evaluate_arm.sbatch
# D2: the same with dapo_improve_d2_decay_kl
```
