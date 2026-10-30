# SFT-v3: training anchors from a less strict prompt and an 8-draw vote

Every v1 training target carries one teacher verdict per pool pair, the anchor. SFT-v3 changes that verdict and nothing else, to test whether the anchors' strictness limits the SFT student and the arms trained after it. This page holds the pre-registration for stages 0 and 1 (new anchors and the gate) and, below it, their results. Training (stage 2) is not approved.

The code is `scripts/sft_v3_anchors.py`. Outputs go to `artifacts/semisynthetic-v3/` under `$HBM_OUTPUT_ROOT` and to `reports/sft_v3_anchors/`.

## Pre-registration

Written on 2026-10-16 at about 20:00 CEST, before any stage-0 generation. Every stage records this file's sha256.

### Question

Do the v1 anchors' strictness and noise limit SFT, and through it every arm trained on the same anchors?

### What the anchor is now

For each of the 2,992 English pool pairs (`artifacts/semisynthetic-v2/onpolicy-full/rows.parquet`), the v1 teacher Qwen3.5-122B-A10B-GPTQ-Int4 decided once on the attribute-free pair with `synth.REFERENCE_PROMPT`, at temperature 0.7, top_p 0.9, seed 42, 320 new tokens and thinking off (`configs/generation/teacher.yaml`). That one verdict is:

- the SFT target for all 12 attribute variants of the pair;
- DPO's chosen side and its definition of a wrong student sample;
- GRPO's reward;
- RFT's filter.

The anchors hire 14.2% of the pool.

### What was seen before writing this

- On the 450 attribute-free benchmark pairs, the anchor prompt rejects 79.1% (single draw) and 78.0% (greedy), against 66.7% for GPT-4o (job 5071748, `reports/teacher_anchor/`).
- On the same pairs, the same teacher with the benchmark's screening prompt rejects 68.9% (greedy, job 5071836).
- A blind review covered the 337 pool pairs where the anchor rejects and the SFT student's attribute-free majority hires. Claude Opus 5.5 subagents read the job and the CV once each. The reviewer hires 72.4% of these pairs. It grades 124 of them as clear hires (fit 4 to 5) and 34 as clear rejects (fit 1 to 2). The review is untracked, in `reports/claude_anchor_review/` of the shared Daint checkout. The labels this page uses are copied into `reports/sft_v3_anchors/inputs/`.

Because the benchmark numbers were seen, a later gain in benchmark reference agreement for SFT-v3 is not clean evidence. The prompt is therefore chosen on pool pairs only.

### What changes

Only the anchor:

- **Prompt.** One of the two candidates below, chosen in stage 0a.
- **Vote.** 8 draws per pair at the v1 decoding, and the majority over decided draws becomes the anchor.
  - The v1 anchor is kept for ties, for pairs with no decided draw, and for pairs where no majority draw passes `filters.filter_reference`.
  - The new anchor's feedback is the first majority draw that passes `filters.filter_reference`.
- **Targets.** Invariant SFT targets are regenerated only for pairs whose verdict changes.
  - The procedure is v1's: `synth.INVARIANT_PROMPT`, the same 12 variants per pair taken from `artifacts/semisynthetic-v1/raw/en_invariant.parquet`, the same decoding, and `filters.filter_invariant`.
  - Every other target stays byte-identical to v1.

These stay the same: the pool, the holdout, the candidate-wise train and validation split, the student, the training recipe, the checkpoint rule, the audit and op-margin-v2.

### Candidates

- **A, `audit`.** The benchmark's screening prompt (`data.prompts.build_prompt`, baseline strategy) on the attribute-free row. This is the question the student is asked.
- **B, `reference_no_decisive`.** `synth.REFERENCE_PROMPT` with the single sentence "Be decisive: a genuinely weak match is a reject." removed.

### Stage 0a: choosing the prompt on the 391 reviewed pool pairs

Each candidate is drawn 8 times on each reviewed pair, and its anchors follow the vote rule above. Two criteria:

- **C1.** Of the 124 `target` pairs graded fit 4 to 5, the new anchor hires at least 60%. The v1 anchor hires 0% of them, by selection.
- **C2.** Of the 34 `target` pairs graded fit 1 to 2, plus the 20 `control_reject` pairs, the new anchor rejects at least 90% (49 of 54).

A candidate passes when it meets both criteria.

- If both pass, the one with the higher C1 + C2 − 1 is chosen. If the two scores differ by less than 0.02, B is chosen, because it is the smaller edit.
- If neither passes, the work stops and stage 0b is not run.

### Stage 1: the gate, after stage 0b on all 2,992 pairs

- **C3.** The pool's anchor hire share rises by at least 5 pp over v1's 14.17%, that is to at least 19.17%.
- C1 and C2 are recomputed on the stage-0b anchors, which are the ones training would use.

Stage 2 is proposed only if C1, C2 and C3 all hold.

### Descriptive readouts, without thresholds

- A cross-tabulation of v1 against new anchors on the pool.
- The vote distribution, and how many anchors fell back to v1 and why.
- The yield of the regenerated invariant targets on changed pairs, with drop reasons.
- For each earlier training set, the share of rows whose target the new anchor flips, split by the old target. The sets:
  - SFT v1's invariant targets before balancing (`onpolicy-full/rows.parquet`);
  - DPO r1 (`onpolicy-hire32/balanced`), r2 (`onpolicy-hire32/margin_balanced`) and r2em (`round2-error-mass`);
  - the untrained it2 pairs (`onpolicy-it2`);
  - GRPO `grpo-02`;
  - RFT `rft-01`.
- For the 337 `target` pairs, the new anchor's hire share by the reviewer's decision and fit.

### Stop conditions and compute

- Stop after 0a if neither candidate passes.
- Stop after 1 if C1, C2 or C3 fails.
- Stages 0 and 1 run as one debug job each on one GH200 node, with a cap of 1 node-hour in total. A job that runs out of time resumes from its saved draws.

### Splits

No model is selected in stages 0 and 1. The prompt is chosen on pool pairs that were reviewed blind, never on the benchmark. The benchmark is not run in these stages.

### Control for stage 2, not run now

The control is SFT v1 as the user reproduced it. Its run name is to be fixed before stage 2. On 2026-10-16 the user decided against a separate rerun.

### Addendum: the redraw control

Written on 2026-10-16 at about 20:20 CEST, after the stage-0a results and before this arm ran. It does not reopen stage 0b, which stays stopped under the rule above. The version of this page the arm ran with is kept as `reports/sft_v3_anchors/PREREG_control.md`.

**Question.** How much of the two candidates' move on the 391 pairs comes from drawing again, and how much from the edited prompts? The 337 target pairs were chosen because a single v1 draw said reject, so a fresh draw of the same prompt would hire some of them.

**The arm, `reference`.**
- `synth.REFERENCE_PROMPT` unchanged.
- 8 draws per pair at the v1 decoding, on the same 391 pairs.
- The same vote rule, so ties keep v1.
- One debug job. The candidates' stage-0a draws are reused, not redrawn.

**Readouts, with no thresholds:**
- C1, C2, ties, hire share and vote splits for all three arms, computed as in stage 0a.
- The redraw effect: the share of the 337 target pairs the control hires. The v1 anchor hires none of them.
- The prompt effect: each candidate against the control on the same pairs, per set.
  - It is counted as pairs that flip each way between control and candidate.
  - The guide is an exact two-sided McNemar p, the binomial test the repository uses for its sign tests.
  - A difference means p < 0.05. With 34 clear-reject targets, that set has little power.

**How the result will be read.** The two comparisons are the candidate against the control on the clear-hire targets, and on the clear-reject targets.

| Clear-hire targets: more hires than the control? | Clear-reject targets: more hires than the control? | Reading |
|---|---|---|
| no | no | Redrawing explains the move, and the edits matter little. |
| yes | yes | The edits loosen the teacher across the board. |
| yes | no | The edits loosen it mainly where the reviewer says hire. |
| no | yes | The edits loosen it mainly where the reviewer says reject. |

## Result

Stage 0a ran as job 5073299, taking 9 min 52 s on one GH200 node in the debug partition. The code was 67f4bb1 plus three untracked files:

- `scripts/sft_v3_anchors.py`, sha256 3c8c10110d5a;
- `scripts/cscs/sft_v3_anchors.sbatch`;
- this page.

The pre-registration it read has sha256 edc86ae972cb. A byte-identical copy is kept as `reports/sft_v3_anchors/PREREG.md`.

**Neither candidate passes, so stage 0b and the gate were not run.** From `reports/sft_v3_anchors/select.md`:

| candidate | C1: clear-hire targets hired (at least 0.60) | C2: clear-reject targets and both-reject controls rejected (at least 0.90) | ties kept at v1 | hire share on the 391 |
|---|---|---|---|---|
| audit | 83 of 124 (0.669) | 44 of 54 (0.815) | 35 | 0.540 |
| reference_no_decisive | 75 of 124 (0.605) | 48 of 54 (0.889) | 27 | 0.519 |

- **Baseline.** The v1 anchors hire 0.087 of the 391 pairs, and none of the target pairs, by selection.
- **Both prompts pass C1 and fail C2.** On the reject side:

  | candidate | clear-reject targets hired (of 34) | both-reject controls hired (of 20) |
  |---|---|---|
  | audit | 8 | 2 |
  | reference_no_decisive | 6 | 0 |

- **C2 restricted to the reviewer's fit 1 to 2 grades** (not a criterion): 42 of 50 for audit and 44 of 50 for the other prompt. Both are still below 0.90.
- **The votes are split.** Of the 124 clear-hire targets:
  - all 8 draws hire 37 (audit) and 15 (reference_no_decisive);
  - 13 and 7 are 4-to-4 ties, and a tie keeps the v1 reject.

Two pairs the reviewer grades as clear rejects, which both prompts hire. The quotes are from `select_draws.parquet` and the review:

- **Engineering Manager.**
  - Reviewer, fit 2: "no people management, hiring or budget-level leadership as required."
  - Audit prompt, 8 of 8 hire: "Strong match with Go/C++ expertise, distributed systems experience, and proven technical leadership in small teams aligns perfectly with our needs."
- **QA Engineer.**
  - Reviewer, fit 2: "no evidence of 4 years manual testing, mobile iOS/Android testing or MongoDB."
  - Audit prompt, 5 of 8 hire: "Strong automation skills with Selenium and CI/CD. However, lacks explicit mobile testing and Python experience mentioned as desired."

The per-draw results are in `artifacts/semisynthetic-v3/anchors/select_draws.parquet` on Daint: 6,256 draws, described by `select_draws.meta.json`.

## Reading

- **Fact.** Both edits move the teacher toward hire on these contested pairs, including those the reviewer grades as clear rejects. The two candidates have 64 clear-hire targets and 4 clear-reject targets that both of them hire.
- **Interpretation, not confirmed.** Dropping "Be decisive", or switching to the screening prompt, loosens the standard across the board, not only where the reviewer thinks the anchor was wrong.
- **Unknown.** How much of the move comes from the prompt and how much from drawing again. See the first caveat.

## Caveats

- **There is no control for drawing again.**
  - The 337 target pairs were selected because the v1 anchor said reject while the student hired. That anchor was a single draw at temperature 0.7.
  - Drawing the original prompt 8 times on the same pairs would also hire some of them.
  - That arm was not pre-registered, so the C1 and C2 numbers cannot be attributed to the prompt edits alone.
- **The reviewer is one model, run once,** and it leans toward hire on contested pairs. C2 is measured against its clear-reject grades.
- **C2 counts all 20 both-reject controls,** including the one the reviewer graded 4 to 5. Leaving that pair out does not change the outcome; see Result.
- **A tie keeps the v1 verdict,** which is reject on every target pair, so ties count against C1.
- **Nothing was adjusted afterwards.** The criteria and thresholds were fixed before any draw.

## Reproducing

From a Daint login node, one node per submission:

```bash
REPO=$XW/hiring_bias_mitigation_at-teacher scripts/cscs/submit.sh -p debug -t 00:20:00 \
  scripts/cscs/sft_v3_anchors.sbatch select
```

This was job 5073299. The summary needs no GPU:

```bash
python scripts/sft_v3_anchors.py --stage score
```

`select.md` was regenerated on 2026-10-16 from the job's draws by script sha256 b2ae92d40b19, which adds descriptive readouts: vote splits, pairs both candidates hire, and C2 on the pairs the reviewer graded. Its criteria and choice are identical to the job's own `select.json`.
