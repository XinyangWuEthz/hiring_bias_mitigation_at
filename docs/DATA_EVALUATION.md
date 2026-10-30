# Evaluating generated preference data

This page describes how we check a batch of generated DPO pairs before any GPU time goes into training on it. It applies to the semisynthetic-v2 negatives. The design spec is `docs/superpowers/specs/2026-10-13-dpo-negatives-v2-design.md`, the checks live in `generation/direction_check.py`, and the generators live in `generation/negatives_v2.py` and `generation/onpolicy.py`.

The checks exist because v1 failed in a way no single metric showed. Its negatives named the protected attribute in 95% of cases, DPO separated the pairs by wording in 50 steps, and the student's decisions did not move. Every check below asks one question about a batch: does the only learnable difference between `chosen` and `rejected` sit in the decision?

## What a negative has to be

A `chosen` response carries the attribute-free reference verdict and a rationale that never mentions the attribute. A `rejected` response on the same prompt has to:

1. carry the opposite verdict;
2. never name or allude to the attribute or its group;
3. read like something a real screener would accept, grounded in the CV and the job;
4. look like the student's own mistakes, so the pair differs in the decision and nothing else that is easy to learn.

The fourth requirement turned out to be the hard one.

## Per-item gates during generation

Every candidate negative passes two gates. A failure drops it. We never repair a generation, and `report.json` counts every drop by reason.

The hard filters are cheap string checks:

| Rule | Drop reason |
|---|---|
| Unparsable output, or the generator refused | `negative_invalid`, `negative_refused` |
| Decision is not the opposite of the reference | `negative_not_flipped` |
| Wrong language | `negative_wrong_language` |
| Rationale outside 4 to 45 words | `negative_length` |
| More than 10 words longer or shorter than the chosen rationale | `negative_length_mismatch` |
| Names the attribute value, by whole-word lexicon (English) or prefix match (Ukrainian) | `negative_attribute_leak` |
| Names the attribute's group, by a whole-word lexicon | `negative_group_leak` |
| Announces neutrality, as in "regardless of" | `negative_meta_commentary` |

The teacher then judges each surviving negative at temperature 0 on four questions. Does it name or allude to the attribute, including through proxies such as availability or cultural fit? Is it grounded in the profile and the job? Does it support the decision it carries? How plausible is it, on a scale of 1 to 5? A negative is dropped if it alludes, is ungrounded, contradicts its decision, or scores below 3. The teacher judges the `chosen` side with the same rubric, and a failing `chosen` drops the whole pair.

## Batch checks

These run on the finished batch. D4 needs a GPU; the rest do not.

| Check | Question | How it is computed | Threshold |
|---|---|---|---|
| D1 validity | Does every pair point the right way? | `chosen` carries the reference verdict and `rejected` the opposite, read from the rendered completions in either language | every pair |
| D2 covertness | Does the negative mention the attribute? | Lexical hit rate on attribute value or group, plus the judge's allusion rate | lexical at most 5%, judge at most 10% |
| D3 wording separability | Can you tell the two sides apart without the decision? | 5-fold AUC of a TF-IDF logistic regression on the rationale text alone, on a sample balanced by reference decision, with folds grouped by pair | balanced AUC at most 0.65 |
| D4 student preference | Does the SFT student already prefer `chosen`? | Sequence log-probability of each side under HF and PEFT, never vLLM's LoRA path, for the student and for the base model | 50% to 90% prefer `chosen` |
| D5 overlap | Are the two rationales near copies? | Jaccard overlap of their word sets | flag above 0.8 |
| D6 balance | Are both flip directions tested? | Share of pairs whose `chosen` verdict is hire | 0.4 to 0.6 |
| D7 yield | How much of what the generator wrote survived? | Kept pairs over prompts, per strategy, with every drop reason | note anything below 40% |

D3 is the check that matters most, and it took two corrections to get right.

Balancing by decision is what makes D3 measure a shortcut and not polarity. A rationale for "hire" praises and one for "reject" criticises, so without balancing a classifier wins by reading tone. With balancing, praise and criticism appear equally on both sides.

The first version split folds by document, which leaked in both directions. A near-copy pair puts its two texts in different folds, so the classifier scores the rejected copy as `chosen`. On a simulated batch with a real wording marker, that version reported 0.49 where the grouped version reports 0.80. Stacking several strategies repeats each `chosen` text, and a copy in the test fold matches its twin in the training fold, which pushed one simulated mixture to 0.94. Grouping folds by `pair_id` fixes both.

D4 has two edges. Below 50%, the pairs fight what SFT already taught. Above 90%, the student already ranks `chosen` first on almost every pair. That does not mean the batch carries no gradient: when the SFT model is also the DPO reference, every pair starts with the same weight, σ(0) = ½, because the loss compares the policy's preference with the reference's. A high D4 means the batch may not reach the mistakes the student still makes. The useful number is the "informative share", the fraction of pairs the student currently gets wrong. (Corrected 2026-10-21; this sentence used to say "carries no gradient".)

## Controls that make the checks believable

A check that never fails, or fails for the wrong reason, is worse than no check. We ran six controls.

**Known-bad input.** v1's published `dpo_validation` pairs fail exactly where we expected. 80% of the rejected responses keep the chosen verdict, 94.9% name the attribute or its group, D3 scores 1.000 and D6 scores 0.196. The grouped folds do not change the 1.000, so v1's shortcut is real and not a fold artefact.

**A natural level for D3.** The threshold of 0.65 is only meaningful next to how real mistakes look. We took the audited Qwen3.5-9B's own baseline responses and found 156 attribute-tipped sets, where the model gave the right verdict for one attribute and the wrong one for another on the same CV. Its wrong-verdict rationales and its right-verdict rationales score an AUC of 0.47 under the same protocol. Real covert bias reads as confident and grounded as a correct decision. Negatives that score far above 0.5 do not look like the failure we want to train away.

**The serving stack.** vLLM's LoRA path does not reproduce these adapters, so the student is merged into its base before vLLM serves it. On 187 batch rows the merged model's greedy decisions agree with the HF and PEFT decision margins on 98.9% of rows.

**Memorisation in D4.** The SFT student trained on v1's `chosen` texts, so a high preference for `chosen` could just be recall. On the 29 pairs whose candidates v1 held out of SFT, the student prefers `chosen` 86% of the time, against 89% overall. The preference generalises.

**Matched comparisons with honest intervals.** Strategies are compared on the same prompts with the same `chosen` texts. Confidence intervals come from a bootstrap over pairs in which a resampled copy keeps its `pair_id`. My first bootstrap gave each copy a fresh ID, which recreated the D3 leak and put 0.883 outside its own interval. Keep copies grouped.

**Reading the data.** Every batch writes `samples.md` with 20 pairs per strategy. v1 hid a broken output contract behind clean metrics for weeks, and only reading the text found it.

When D3 fails, two diagnostics show why. The classifier's top-weighted n-grams name the signal, and an ablation removes word classes one at a time and re-scores. On the teacher's negatives, removing person and subject words such as "candidate" and "your" moved D3 only from 0.88 to 0.86. Also removing hedging and intensity words such as "perfectly", "despite" and "but" brought it to about 0.78. The rest is content. A rationale argued for the wrong verdict has weaker evidence.

## Results on batch-01

batch-01 holds 240 English prompts, stratified 20 per group, condition and reference decision. The table compares the teacher's best strategy with the student's own mistakes on the same rows.

| | Teacher, `covert_flip` | Student, `student_tipped` |
|---|---|---|
| Who writes the negative | the 122B teacher, shown the CV without the attribute | the SFT 9B student, sampled 8 times per prompt at temperature 1 |
| Pairs kept | 154 of 240 | 34 of 240 |
| Negatives naming the attribute, before filters | 0% | 0 of 241 wrong samples |
| Main drop reason | judge: ungrounded, 54 | judge: ungrounded, 16 |
| Judge plausibility, mean of 1 to 5 | 3.81 | 3.76 |
| D3 balanced AUC, 95% CI | 0.883, from 0.792 to 0.911 | 0.574, from 0.319 to 0.750 |
| D4, SFT student prefers `chosen` | 81% | 50% |
| D4, base model prefers `chosen` | 49% | 47% |
| D5 mean overlap | 0.117 | 0.125 |
| D6 hire share | 0.519 | 0.500 |

The other strategies did worse. `minimal_edit` kept 150 pairs but scored 0.882 on D3 and 95% on D4, so it is separable and too easy. `proxy_cue` kept 14 pairs, because the teacher could not lean on proxies without alluding to the attribute; the judge flagged 62% of its outputs. Student mistakes that were not attribute-tipped kept 20 pairs and failed D6 with a hire share of 0.2.

The teacher's negatives fail D3 clearly, with the whole interval above 0.65. The student's negatives pass D3 on the point estimate and sit at 50% on D4, the most informative position possible. Thirty-four pairs are too few to certify D3, though, because the interval still crosses 0.65.

The student strategy has a yield problem by construction. The student answers correctly on 87.3% of samples with the attribute and 88.3% without. Only 76 of 240 rows produced any wrong sample, and 50 of those were attribute-tipped. If the rate holds, the full pool of 35,826 rows gives roughly 5,000 pairs. That is an estimate from 240 rows, not a measurement.

## Decision rule and open questions

A strategy goes to full scale when it passes D1, D2, D3 and D6, sits inside the D4 band with the largest informative share, and its `samples.md` reads well. On batch-01 only `student_tipped` qualifies, and only provisionally.

Three things are still open:

1. **D3 at a size that can decide it.** We need several hundred student pairs, from more rows or more samples per row.
2. **The reference verdict.** The reference is a single teacher sample at temperature 0.7, so some "wrong" student answers on borderline candidates may be defensible. The judge's decision-support question catches some of these, not all.
3. **Self-judging.** The teacher judges the `chosen` texts it wrote itself. The lexical filters and the student-based D4 do not depend on it, but an independent judge, or a 50-item human calibration, would make the judge's numbers more trustworthy. Partly answered: on hire rows, the student's own correct samples pass the judge 95.8–96.0% against 97.1% for `chosen`, so the gap with `rejected` follows the verdict, not the author (`docs/JUDGE_AUTHOR_CONTROL.md`).

## The sets DPO trained on

`student_tipped` went to full scale, and two training sets came out of it. Round 1 is the full-pool pairs plus a hire-side top-up at 32 samples per prompt, balanced by reference verdict. r2em adds round 1's reject-side surplus and the v1 pairs the judge found covert, and balances by the SFT student's error mass. `docs/V2_NEGATIVES_DPO_DATA.md` describes how each was built, its checks and what training on it gave. `docs/DPO_DATA_CREATION.md` walks through the stages from v1's invariant pool and tabulates the attributes in each set. Round 1's final set fails D3 at 0.684, and r2em's final set has no direction check.

## On-policy negatives from a DPO model

The same construction with the round-1 DPO model as the student (r2em checkpoint-100) found 409 tipped rows on the full pool instead of SFT's 5,614. The judge kept 305 pairs, and they fail D6 with a `chosen` hire share of 0.013. `docs/ITERATIVE_DPO.md` explains why and why that round was not trained. GRPO from SFT on the same pool, with a reward instead of negatives, is in `docs/GRPO_PILOT.md`: −0.95 pp at matched hire rate, reached by rejecting more, as round 1 did. A second SFT stage on the student's own right answers (`docs/RFT_CONTROL.md`) gains nothing at matched hire rate (+0.31 pp) and also rejects more.

## Running the checks

From a Daint login node, with one node per submission:

```bash
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/small_batch.sbatch generate
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy.sbatch sample
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy.sbatch judge
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy.sbatch check
```

Any parquet of pairs can be checked without a GPU:

```bash
python scripts/check_preference_direction.py --pairs artifacts/semisynthetic-v1/dpo_validation.parquet
```
