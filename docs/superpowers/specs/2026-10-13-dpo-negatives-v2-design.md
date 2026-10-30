# Semi-synthetic data v2 — covert negatives for preference optimisation

*Design and data specification. Written 2026-10-13 for the GTC 2027 "SFT → DPO (covert
pairs)" arm (Notion: Plan B). Status: first small batch not yet generated; nothing in this
document is a result.*

## 1. Goal and scope

Build the `rejected` side of the preference data again, and only that side. The prompt (job,
CV, injected attribute), the attribute-free `reference` verdict and the `chosen` response are
reused unchanged from `semisynthetic-v1`, so a v1-vs-v2 training comparison differs in exactly
one thing: what the student is taught to reject.

| Fixed | Value |
|---|---|
| Student | `Qwen/Qwen3.5-9B`, English, continuing from the published SFT adapter `Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en` (LoRA r=32, α=64) |
| Teacher (generation and judging) | `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` through vLLM, as in v1 |
| Inputs reused from v1 | `raw/en_invariant.parquet` (prompt fields + the invariant pass), `raw/en_reference.parquet` (anchor); public copy: `Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data`, subsets `teacher_invariant`, `teacher_reference` |
| Language | English only in this iteration. Ukrainian prompts are deliberately not drafted (the plan: a half-done Ukrainian cell costs more than none) |
| First deliverable | a **small batch** of candidate pairs (240 prompts × 3 negative strategies) plus a **preference-direction check**, before any full generation or training run |

Out of scope here: regenerating the anchor or the chosen side (§8 lists why they still need
work), intersections, Ukrainian, the GRPO/GDPO arms.

## 2. What v1 got wrong, measured

From `reports/MITIGATION_FINDINGS.md` Finding 8 and the v1 `generation_report.json`:

- **91% of v1 `rejected` responses name the protected attribute** ("We cannot hire candidates of
  the Sikh faith…"); **78% carry the same decision as `chosen`**. The pair is separable by
  wording alone. DPO on these pairs hit 100% preference accuracy and ~1e-5 eval loss within 50
  steps and did not beat plain SFT on the audit.
- The audited models are biased **covertly**: Qwen3.5-9B names the attribute in ~1–2% of
  rationales, yet 17.3% of its counterfactual sets flip on the attribute. The failure to imitate
  is a decision that moves under a neutral-sounding rationale.
- The v1 biased pass asked the teacher to *be* biased: 13.5% refusals, 30.9% "not actually
  biased", 55.6% yield, uneven across groups (military 18,310 pairs vs religion 5,086).
- The decision-only probe (`dpo_decision`: pairs differing in the decision token alone) removed
  the wording shortcut but is the degenerate end of the spectrum — a one-token edit with no
  rationale, far from anything the student emits — and it also did not beat SFT.

The v2 negative therefore has to be (i) the **opposite decision**, (ii) under a **plausible,
grounded, attribute-free rationale**, (iii) in the **same register and length** as `chosen`, and
(iv) verified by something stronger than prefix matching.

## 3. Standards adopted from recent work, and the rule each one becomes

| Finding in the literature | Rule in v2 |
|---|---|
| The quality of the **chosen** response dominates DPO outcomes; rejected quality matters less, and contrast helps mostly through better chosen samples (Pan et al. 2025, *What Matters in Data for DPO?*, arXiv:2508.18312, NeurIPS 2025). | Keep v1 `chosen` (already 99.5% yield), but put it through the same judge as the negative and drop pairs whose chosen fails. Record anchor noise as the next thing to fix (§8). |
| Rejected responses dominate the DPO gradient; off-policy, low-confidence negatives cause a "squeezing" of probability mass onto unrelated outputs (Cho et al. 2025, *Rethinking DPO*, Findings of EMNLP 2025, arXiv:2506.12725; Ren & Sutherland 2025, *Learning Dynamics of LLM Finetuning*, ICLR 2025). Choosing the *worst* sample as rejected hurts; a moderately bad one (≈ μ−2σ) is better (*Finding the Sweet Spot*, arXiv:2502.16825). | Negatives must be **plausible** (judge plausibility ≥ 3/5), **grounded** in the CV and job (no invented facts), and written in the register of a real screener — never a caricature. |
| Near-identical chosen/rejected pairs cause **likelihood displacement**: both log-probs fall while the margin grows (Razin et al. 2025, *Unintentional Unalignment*, ICLR 2025, arXiv:2410.08847; Pal et al. 2024, *DPO-Positive*, arXiv:2402.13228; DPO-Shift, arXiv:2502.07599). | Pairs differ in **decision and rationale content**, not in one token. Token-overlap between the two rationales is reported per strategy (§6, D5). Training adds an NLL term on `chosen` (TRL `loss_type: [sigmoid, sft]`), the standard displacement guard. |
| DPO exploits **length** and **style** differences between sides (Park et al. 2024, *Disentangling Length from Quality*, Findings of ACL 2024; Meng et al. 2024, SimPO, NeurIPS 2024). | Rejected word count within ±10 words of chosen; both under the 30-word schema cap (45 with slack); same JSON format rendered by the same function as v1. A wording-only classifier must not separate the sides (§6, D3). |
| Preference pairs are now routinely **judge-verified** with a structured rubric rather than trusted (Lambert et al. 2024, Tülu 3, arXiv:2411.15124; UltraFeedback-style rubrics). | A second teacher pass judges every candidate on four fields: `mentions_or_alludes`, `grounded`, `supports_decision`, `plausibility`. Judging runs greedily; the rubric is in `generation/negatives_v2.py`. |
| Mixing **on-policy** (student-sampled) data into off-policy pairs helps, and pairs the student already gets right carry little signal (Tülu 3; SimpleMix, arXiv:2505.02363). | The negative is still teacher-written (the plan keeps the teacher), but the batch is checked against the **student's own log-probabilities** before training (§6, D4), and the full run will prioritise pairs the student currently gets wrong. |
| Bias mitigation through preferences should contrast **counterfactually matched** responses so the only learnable difference is the causal one (Kusner et al. 2017; Counterfactual DPO, arXiv:2401.09566; BiasDPO, ACL SRW 2024; C2PO, arXiv:2512.23430). | Chosen and rejected share job, CV, attribute and condition (v1's join key). The target of invariance is the attribute-free reference verdict, as in v1. |
| Dataset documentation norms (datasheets; NeurIPS Datasets & Benchmarks: card, licence, decontamination statement, per-reason yields, machine-readable metadata). | Every batch writes `report.json` (yields per reason per strategy, covertness, separability, lengths), a human-readable `samples.md`, and carries provenance columns (§4.4). `assert_no_leakage` runs on every written frame. |

## 4. The v2 negative — specification

### 4.1 Definition

For a v1 row with reference verdict *r* and chosen response *(r, f_chosen)*, the v2 rejected
response is *(¬r, f_rejected)* where *f_rejected* is a rationale for *¬r* that:

1. never names, quotes, paraphrases or alludes to the injected attribute or its group;
2. is grounded in the CV and the job description;
3. reads as a routine professional assessment a hiring team would accept;
4. matches the chosen response in format, language and length.

Placed on the attribute-bearing prompt, this is exactly the covert failure the audit measures:
the decision moved, the rationale stayed neutral.

### 4.2 Three generation strategies (all run on the same rows in the small batch)

| Strategy | What the teacher sees | Instruction | Why it is in the batch |
|---|---|---|---|
| `covert_flip` (primary) | job + **bare CV** (no attribute) + target decision ¬r | write the rationale a screener who decided ¬r would give | Attribute-free **by construction**: the teacher cannot leak what it never saw. No request to "be biased", so the v1 refusal problem disappears. |
| `minimal_edit` | job + bare CV + the chosen rationale | rewrite the chosen rationale with the smallest change that justifies ¬r | The hard-negative end: maximal matching, highest displacement risk (D5 measures it). |
| `proxy_cue` (exploratory) | job + **attribute-bearing profile** | a screener whose ¬r was driven by the attribute but who would never admit it; may use proxies (availability, commitment, fit) | Closest imitation of covert bias *if* the student reasons through proxies. Expected to show higher allusion and refusal rates; kept or dropped on the direction check. |

The teacher runs at the v1 decoding settings (T=0.7, top-p 0.9, seed 42); the judge at T=0.

### 4.3 Filters (dropped, never repaired — v1 policy)

Hard filters, applied to every candidate (`filter_negatives`):

| Drop reason | Rule |
|---|---|
| `negative_invalid` / `negative_refused` | unparsable, or the teacher declined |
| `negative_not_flipped` | decision ≠ ¬reference |
| `negative_wrong_language` | v1 `language_ok` |
| `negative_length` | words < 4 or > 45 |
| `negative_length_mismatch` | abs(words_rejected − words_chosen) > 10 |
| `negative_attribute_leak` | v1 `mentions_attribute` (attribute value stems) |
| `negative_group_leak` | new: word-boundary lexicon of the row's **group** (military/veteran/…, gender/woman/pronouns/…, religion/faith/…) |
| `negative_meta_commentary` | v1 `has_meta_commentary` |

Judge filters, applied to the rejected side of every strategy and to the chosen side once per
row (`filter_by_judge`):

| Drop reason | Rule |
|---|---|
| `judge_unparsable` | the judge's JSON could not be read |
| `judge_alludes` | `mentions_or_alludes = true` (for `proxy_cue` this is recorded, not dropped) |
| `judge_ungrounded` | `grounded = false` |
| `judge_contradicts_decision` | `supports_decision = false` |
| `judge_implausible` | `plausibility < 3` |
| `chosen_judge_*` | the same four, on the chosen side; drops the whole pair |

A proxy-word rate (availability, commitment, culture fit, flexibility, stability, personal
circumstances…) is **reported** for both sides, not filtered.

### 4.4 Row schema (`dpo_v2_*` and `candidates_*.parquet`)

v1 columns, unchanged in meaning: `prompt`, `chosen`, `rejected`, `lang`, `protected_group`,
`protected_attr`, `condition`, `candidate_id`, `job_id`, `pair_id`,
`rejected_flipped_decision` (always true in v2), `rejected_mentions_attribute` (false by filter).

New provenance columns: `reference_decision`, `chosen_decision`, `rejected_decision`,
`rejected_strategy`, `chosen_words`, `rejected_words`, `length_delta`, `token_overlap`
(Jaccard of rationale word sets), `rejected_mentions_group`, `rejected_proxy_words`,
`chosen_proxy_words`, `judge_alludes`, `judge_grounded`, `judge_supports`, `judge_plausibility`,
`judge_note`, `judge_parsed`, `chosen_judge_alludes`, `chosen_judge_grounded`,
`chosen_judge_supports`, `chosen_judge_plausibility`, `chosen_judge_ok`, `teacher_model`,
`source_artifacts`, `generated_on`.

`chosen` and `rejected` are rendered by v1's `_completion`, so the decision word is the one the
prompt's language asks for (lesson 5 of the findings).

### 4.5 Balance and split

- **Small batch:** stratified to 20 rows per (group × condition × reference decision) cell =
  240 prompts, one row per `pair_id`, drawn from the whole pool and split by candidate
  afterwards (so the batch also yields a validation split for a DPO smoke run).
- **Full run:** keep every surviving pair; then resample the chosen-decision mix to a
  configurable `target_hire_rate` (default: the benchmark's 0.34, per the v1 future-work note
  that 50/50 overshot the operating point). The teacher pool is 86% reject, so the hire side is
  the scarce one and is never downsampled.
- **Split:** by candidate (v1 `train_val_split`, seed 42), and any candidate that sits in a v1
  validation split (`sft_validation`, `dpo_validation`) is forced into v2 validation too, so a
  v1-trained and a v2-trained student can be compared on the same held-out candidates.
- **Contamination:** `assert_no_leakage` against the benchmark hold-out on every written frame.

## 5. Small batch — what gets generated first

```
configs/generation/negatives_v2_small.yaml   -> artifacts/semisynthetic-v2/batch-01/
  240 prompts x 3 strategies = 720 negative generations
  + 720 judge calls on the negatives + 240 on the chosen side
```

Outputs: `raw/<strategy>_negatives.parquet`, `raw/<strategy>_judge.parquet`,
`raw/chosen_judge.parquet` (resumable, as in v1), `candidates_<strategy>.parquet`,
`dpo_v2_candidates.parquet` (all strategies, with `rejected_strategy`), `dpo_v2_train.parquet`
and `dpo_v2_validation.parquet` (the kept pairs, split as in §4.5, enough for a DPO smoke run),
`report.json`, `samples.md`.

`--dry-run` builds the batch and every prompt without loading a model, and writes
`batch.parquet` plus `prompts_<strategy>.parquet` for reading before the teacher is paid for.
Run on 2026-10-13 against the public copy of v1 (`teacher_invariant`, English): 240 prompts,
20 per cell, 240 distinct pairs.

## 6. Preference-direction check — acceptance criteria for the batch

`scripts/check_preference_direction.py` reads a batch directory and reports, per strategy and
overall, with PASS/FAIL against these thresholds. D1–D3, D5, D6 need no GPU; D4 needs the
student.

| # | Check | Threshold | Why |
|---|---|---|---|
| D1 | Validity: `chosen.decision == reference` and `rejected.decision == ¬reference` on 100% of rows | 100% | the pair's direction is the one the audit measures |
| D2 | Covertness: attribute-stem or group-lexicon hit rate on `rejected`; judge allusion rate | ≤ 5% lexical; ≤ 10% judge | v1 was 91%; the student's own rate is ~1–2% |
| D3 | No wording shortcut: 5-fold AUC of a TF-IDF logistic regression told chosen from rejected on the rationale text only (decision field removed), on a **reference-decision-balanced resample** so the rationale's polarity cancels and only a wording or style shortcut remains, with **folds grouped by pair** so a pair's two texts, and every strategy's copy of its chosen text, never straddle a split (ungrouped folds hid a marked near-copy shortcut at 0.49 and inflated stacked strategies to 0.94 in the pre-GPU review) The within-decision AUC is reported too and is high by design (chosen always carries the reference verdict) | balanced AUC ≤ 0.65 | v1 pairs are separable at ≈1.0; a covert batch should sit near 0.5 |
| D4 | Student direction, under HF + PEFT (never vLLM's LoRA path): share of pairs where the SFT student assigns `chosen` a higher sequence log-prob than `rejected` (sum and per-token), and the sign of its hire-vs-reject decision-token margin on the prompt | 50%–90% already prefer chosen | below 50% the pairs fight what SFT installed; above 90% the batch carries no gradient. The remainder is the signal. |
| D5 | Displacement risk: Jaccard overlap between chosen and rejected rationale word sets | report; flag > 0.8 | Razin et al.: near-identical pairs displace mass. Expected high for `minimal_edit` |
| D6 | Decision balance of the kept pairs | hire share in [0.40, 0.60] for the small batch | balanced so D3 is meaningful and both flip directions are tested |
| D7 | Yield per strategy after all filters, with per-reason counts | report; a strategy under 40% yield is noted as a finding, not fixed quietly | v1 policy |

**Control (run 2026-10-13, no model):** on v1's published `dpo_validation` split (1,647 pairs,
both languages) the GPU-free checks fail exactly where Finding 8 says they should: D1 1,325
rejected responses not flipped (80%), D2 lexical attribute-or-group mention 94.9%, D3 balanced
AUC 1.000 (unchanged with pair-grouped folds), D6 chosen hire share 0.196. A v2 batch that passes is therefore measurably
different from v1 on the axes that broke the v1 DPO run, not merely different by construction.

The decision rule after the batch: pick the strategy (or mixture) that passes D1–D3 and D6 with
the largest **informative share** in D4 (pairs the student currently gets wrong or nearly so),
read 20 pairs per strategy in `samples.md` by eye (lesson 5), then scale that strategy to the
full 35,826 rows.

## 7. Training from the published adapter (after the batch is accepted)

`configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg.yaml`:

- `sft_checkpoint: Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en` — a Hub adapter id.
  `training_common._resolve_adapter` now resolves Hub adapter ids as well as local directories;
  the adapter is merged into the base in memory and a fresh LoRA is trained on top, so the
  reference model for DPO *is* the SFT model (policy and reference coincide at step 0).
- `loss_type: [sigmoid, sft]`, `loss_weights: [1.0, 0.2]` — DPO plus an NLL term on chosen, the
  guard against likelihood displacement (TRL's MPO-style combination; `rpo_alpha` no longer
  exists in TRL 1.15). `ld_alpha` is available for length desensitisation if D3/D5 call for it.
- Fixed schedule: `max_steps: 300`, `early_stopping_patience: 0`, `eval_steps: 50` on 200
  validation pairs — Nazarii's note that early stopping was too strict, and the repo's own
  observation that a loss that can collapse never stops improving by threshold.
- `learning_rate: 5e-5` on LoRA, `beta: 0.1`, batch 1 × 16, gradient checkpointing on,
  `precompute_ref_log_probs: true` (the reference is frozen, so its log-probs are computed once).

**Before the DPO adapter is audited:** the audit must serve *base + SFT adapter + DPO adapter*
merged in that order. `mitigation/merge.py` merges one adapter onto the base model named in its
`adapter_config.json`; a DPO adapter trained on the in-memory merge records the bare base, so
merging it alone would drop the SFT delta — the exact bug noted in `qwen3.5-9b_en_only_dpo_v2.yaml`.
`merge.py` needs to read `run_config.json`'s `sft_checkpoint` and apply it first. Not done in
this iteration; listed so it cannot be forgotten.

## 8. Known limitations carried into v2 (not fixed here)

- **The anchor is still one sample at T=0.7** (v1 future-work row 4). Every v2 pair inherits it.
  Fix in v2.1: regenerate the reference greedily or by majority vote and drop pairs the teacher
  disagrees with itself on; that changes `chosen` and is a separate comparison.
- **Same model generates and judges.** Self-judging is biased toward its own outputs; the
  lexical filters run independently of the judge, and the direction check uses the student,
  not the teacher.
- **`proxy_cue` is exploratory**: if the student never reasons through proxies, these negatives
  are a milder version of the v1 wording shortcut. D3 decides.
- **Hire-side negatives argue for weak candidates.** 86% of anchors are reject, so most v2
  negatives are "hire" rationales for candidates the teacher judged weak; plausibility is
  expected to be lower there and the judge threshold does the work. Report plausibility by
  reference decision.
- **English only.** Ukrainian templates are not drafted.

## 9. Implementation map

| File | Role |
|---|---|
| `src/hiring_bias_mitigation/generation/negatives_v2.py` | prompts (3 strategies + judge), group lexicon, hard filters, judge parsing/filters, batch sampling, pairing, surface statistics (separability, lengths, overlap) |
| `scripts/generate_negatives_v2.py` | the small-batch / full-run driver: load v1, rebuild chosen, sample, generate (resumable), judge, filter, write candidates, splits, report, samples |
| `scripts/check_preference_direction.py` | D1–D7; `--no-model` for the GPU-free part, `--student` for D4 |
| `configs/generation/negatives_v2_small.yaml` | the first batch |
| `configs/data/en_only_v2.yaml`, `configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg.yaml` | the training arm |
| `mitigation/training_common.py`, `mitigation/dpo.py` | Hub adapter ids for `sft_checkpoint`; `loss_type`/`loss_weights`/`ld_alpha`/`precompute_ref_log_probs` pass-through |
| `tests/test_negatives_v2.py` | filters, prompts, sampling, pairing, separability, an end-to-end dry pipeline with a canned teacher, the trainer changes |
| `Makefile` | `local-negatives-v2`, `local-direction-check` |
