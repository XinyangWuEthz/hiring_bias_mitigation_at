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

## Result

Not run yet.

## Reading

## Caveats

## Reproducing
