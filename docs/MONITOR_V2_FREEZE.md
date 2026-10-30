# Djinni monitor correction, freeze 1

2026-10-23, experiment `monitor-v2-djinni-intake-20261023-v1`.
`configs/calibration/monitor_v2_djinni_freeze.json` defines this round's rules.
This commit freezes sources, candidate order and partitions, the unified annotation procedure, target weights, 8 models, and the proposed alert rule.
It generates no new labels, runs no new model scoring, and enables no training.

## 2026-10-23: subsequent switch to direct benchmark-dev monitoring

The user subsequently changed the plan explicitly: run benchmark inference only for GRPO-v3-all checkpoints, submitting one job at a time,
and replace projection-based monitoring with direct TPR measurement on the benchmark dev split.
The current entry points are `docs/BENCHMARK_DEV_MONITOR.md` and
`configs/analysis/benchmark_dev_monitor_v1.json`. Candidate partitions were frozen before reading new checkpoint results.
The historical benchmark has already informed development, so the remaining partition is not an independent unseen test. The new dev stop threshold of 8 pp is unvalidated;
the original full-benchmark final guardrail of 10 pp remains unchanged, and the checkpoint-525 conclusion remains exploratory.

The independent Djinni candidate intake, unified blind binary annotation, and projection-validation plan below remain as a historical record and a basis for future label research.
The direct-dev experiment does not advance that plan or treat its unfinished blind annotation and projection calibration as passed.
The conclusion that **RJDB reject anchors are questionable** remains pending confirmation by blind review under the unified protocol.
No training is authorized in this round; the direct-monitoring manifest retains `training_authorized=false`.

## Existing monitor calculation and required corrections

`mitigation/grpo.py::MarginMonitorCallback` reads SFT-v3 margins before updates and reads margins from the currently served model every 25 steps.
The margin is `log p(hire) - log p(reject)` after a forced JSON decision prefix;
`margin > 0` counts as hire. The 8 bins use fixed boundaries of -4.595, -2.197, -0.847, 0, 0.847, 2.197, and 4.595.
These boundaries are not calibrated hiring probabilities.

The old estimator mixes all sources and labels within each bin, computes the overall hire-rate change, and multiplies it by the benchmark's hire/reject
bin weights separately. The 696 rows comprise 504 Djinni and 192 RJDB rows. Within a bin, new RJDB hires offset lost Djinni hires,
giving a DAPO projection of -3.24 pp versus a measured benchmark-margin change of -9.61 pp. The old calibration failed and cannot authorize training.

The retained conclusion is that **RJDB reject anchors are questionable**, pending blind confirmation under the unified binary-annotation protocol.

The old implementation also treats empty bins as zero change and, on resume, checks only the parent-margin count rather than prompt, label,
weight, and model fingerprints. The revised version must save and stop under these conditions rather than continue reporting a supposedly reliable TPR estimate.
This round freezes data and procedures; runtime callback repair, deployment, and independent validation still follow the second freeze.

## Sources and candidates

The fixed source is `djinni_new` in `additional/pool/pairs_en.parquet`, with SHA256
`cb64431723a73f6a1fe2107503a0a80d09a3883da2c0b319582b7cd3f2df2a90`.
The English candidate HF revision is `86255174c6c378a2b52cd7e81d79002c64d899a6`,
and the job revision is `b56a6054c10f1266a141e37c8df66c79ff2863af`.

Starting from 144,091 Djinni candidates, exclusions cover registered-model training, historical train/validation data, complete previous voting batches, benchmark
candidates and jobs, this round's reviews, and existing attribute-marker screening, leaving 127,681 candidates. The source manifest retains fingerprints and exclusion reasons for 121 input files.
Normalized-CV deduplication applies NFKC, whitespace collapsing, and casefold before SHA256.
The near-duplicate screening method is frozen but has not run; current ID separation must not be described as complete absence of text exposure.

The first batch takes 2,000 candidates in ascending order of `SHA256("monitor-v2-intake-20261023:42:" + candidate_id)`,
retaining the original CV/JD pairings. Use at most one pair per candidate, without selecting positives from old anchors or model predictions.
If annotation leaves coverage insufficient, add batches of 2,000 in the same order, up to 10,000 candidates. All additional sampling must precede disclosure
of training-arm results. Annotate each complete batch and retain every resolved hire/reject case, rather than favorable samples only.

Candidate partitions are fixed before annotation. With the independent salt `monitor-v2-split-20261023:42:`, a SHA256 integer below
`2**255` assigns monitor; all others assign validation. The first batch splits into 1,003 and 997 candidates. All text and attribute variants of one person
must stay in one partition. The partitions share 14 jobs; another 258 candidates have jobs that appeared in registered-model training.
Report sensitivity to job clustering rather than claiming job independence.

The old 696 rows, 75-row/40-pair reviews, original 167-candidate subpool, and all previously inspected arm results are development evidence.
The 8 old Qwen hire labels among the original 167 candidates do not count as newly confirmed positives.
Some historical training inputs lack training-time SHA values. Current paths and row counts match, and current bytes are locked, but historical immutability cannot be reconstructed as proven.

## Unified binary annotation

Reuse the single `hiring-evidence-v1` policy file and the two committed prompts; do not define a separate set of qualification thresholds.
Freeze checklists from JDs alone, then have two independent reviewers assess attribute-free CVs. Reviewers cannot see old labels, vote counts, SFT/DAPO
answers, margins, or final partitions. Register reviewer identities and serving versions before starting.

- hire: every explicitly mandatory requirement has direct or permitted functional evidence, with unknown count U=0 and contradiction count F=0, and at least one
  verbatim CV passage supports the main job duties. When H>0, the support rate must be 100%.
- reject: no conflict in decisive material, and affirmative counterevidence for at least one explicitly mandatory requirement, F>=1.
- Missing information, conflicts, or disagreement: `label=null, eligibility=unresolved`; exclude from the binary-label set.
  This is sample eligibility, not a third model output class, and must not automatically become reject.

Evidence ledgers retain the original policy's review state. Export a binary label only when two valid ledgers agree on hire/reject.
Every table must disclose both the resolved proportion and unresolved count, rather than reporting retained samples alone.

Separate mandatory requirements from preferences. Missing a preference cannot justify rejection or offset a failed mandatory requirement. Unmentioned does not mean unable.
Use the original duration policy: year differences without inclusive endpoint counting, month differences when available, unions of overlapping intervals, and >= for minimum thresholds.
Open-ended dates use the record's own as-of date; the cache download date of 2026-10-20 is not the CV observation date.
Without a valid cutoff date, do not accumulate ongoing experience through today.

The existing benchmark retains GPT-4o labels. New-protocol labels and old labels are not the same ground truth merely because both are binary.
The preregistered label comparison covers all original 450 pairs: blind-review them under the same standard and report, by old label × new label × parent-margin bin,
agreement, unresolved rates, and parent-model TPR under both definitions. Preserve the original benchmark values. Until label-meaning differences are resolved,
do not claim the new monitor measures the original TPR. This is an approval condition, not a reason to retune weights afterward.

## Stratification, weights, and models

Use all 68 attributed variants from the original benchmark for each pair; report attribute-free controls separately.
Assign the original 8 bins using fresh SFT-v3 margins, not the old k8/k32 sampled hire rates.
Within each bin, estimate separately by label, protected group, and condition. For each label side:

`delta = sum(W_cell * (gain_cell - loss_cell))`

W is the exact frequency proportion for each bin × group × condition within that label in the old GPT-4o benchmark, saved in
`monitor_v2_target_weights.json`. Weights use only parent-model margins, with no training-arm performance used in their selection.
The complete values contain 48 cells per label side and a list of 68 variants. They define the projection target for the old benchmark;
they do not themselves establish that new labels transfer to that target.

New labels, final row order, and parent margins are not yet available, so per-row `W_cell/n_cell` weights are not yet frozen.
Lock them in the second freeze before revealing training-arm scores.
Do not fill empty positive-weight cells with zero or renormalize the other weights.

Minimum-support screening requires at least 20 distinct candidates in each label × margin bin with target weight, and observations in every positive-weight fine-grained cell.
Before revealing arms, also rehearse all 40 fixed bootstrap seeds using parent-model cell assignments. If any resample has an empty positive-weight
cell, add the next batch in the fixed order. This structural precheck does not require arm scores.
Candidate-clustered 95% intervals for the actual delta and transfer error must have half-widths no greater than 2 pp. These are screening thresholds, not accuracy guarantees.
Interval precision is checked after arms are revealed; do not add samples based on it and still claim the original validation is independent.

The fixed model list is SFT-v3, DAPO-v3-all, GRPO-v3-all, GRPO-v3-all-step525, GRPO-v3,
DPO-v3-step240, DPO-v3-onpolicy-step98, DPO-v3-4880-step305.
The model lock records full-weight SHA256 values, tokenizer/config, and merge provenance. A mutable directory name alone cannot establish model identity.
step525 and GRPO final belong to the same training trajectory and are not two independent experiments.

## Alerts and validation

Check every 25 optimizer steps, plus a final check if fewer than 25 steps remain. Use the net TPR change relative to SFT-v3.
Save and stop training if the point-estimated decline reaches 8 pp or the conservative upper bound on decline reaches 8 pp.
In signed-change form, this is `point_delta <= -0.08 OR lower_delta <= -0.08`.
Only a measured decline below 8 pp is on the safe side for false-stop checks; stopping at exactly 8 pp follows the rule.

The proposed alert-interval method is frozen as whole-candidate bootstrap with 65,536 resamples, PCG64, and base seed 20261023,
with each seed constructed from `[base_seed, check_index]` and linear quantiles. Allow at most 40 post-update
checks, each with one-sided alpha 0.05/40. Keep target weights fixed and recompute cell means in each resample.
Any empty positive-weight cell in a resample, scoring/source/serving-sync failure, or exhausted check budget triggers save-and-stop for insufficient evidence.
Resuming requires verification of the complete parent model, prompt order, labels, weights, configuration fingerprints, and cumulative check count.

This approximate bootstrap rule still needs validation. Bonferroni coverage depends on each interval being valid; it does not yet justify
a claim of a 95% guarantee over the full run. It also does not cover label differences, source-transfer bias, or changes between checks.

Independent validation still requires all 7 arms to pass: absolute error in original-benchmark margin TPR <=2 pp,
optimistic bias relative to full-generation TPR <2 pp, and an alert for every arm with a measured decline greater than 8 pp.
Check false stops under the actual conservative rule above. False stops block routine deployment; do not retune thresholds on the spot and declare validation passed.
Also report the new validation bank, sensitivity to job clustering, FPR, loss/gain, and label coverage.

## Artifacts and reproduction

Remote artifacts are under `$HBM_OUTPUT_ROOT/outputs/monitor_v2_freeze/20261023-intake-v1/`.
`source/` stores provenance, the full candidate inventory, first-batch source text, and model locks; `lock-final/` stores candidate partitions, the blind-review queue, and the manifest.
`lock/` is the pre-commit review draft, retained for traceability rather than as the official entry point.
The blind-review queue cannot yet be executed directly: complete near-duplicate screening and freeze JD-only checklists before binding CVs.

```bash
uv run python scripts/prepare_monitor_v2_freeze.py \
  --config configs/calibration/monitor_v2_djinni_freeze.json \
  --output "$HBM_OUTPUT_ROOT/outputs/monitor_v2_freeze/20261023-intake-v1/lock-final"
uv run python scripts/prepare_monitor_v2_freeze.py \
  --config configs/calibration/monitor_v2_djinni_freeze.json \
  --output "$HBM_OUTPUT_ROOT/outputs/monitor_v2_freeze/20261023-intake-v1/lock-final" --verify
```

The entry point verifies source fingerprints, exclusions, fixed order, text fingerprints, partitions, and target weights, and refuses to overwrite an existing frozen directory.
`--verify --require-stage arm-scoring` or `training` must exit with code 2 under the first lock.
This is not a global block on general training scripts; no training jobs will be submitted in this round.

The first lock fixes procedures and inputs. Newly confirmed positives remain at 0. The second lock still lacks new labels, parent margins, final row weights, and
label-comparison conclusions. Runtime repairs and independent validation must follow; the first lock does not mean the monitor is ready for use.
