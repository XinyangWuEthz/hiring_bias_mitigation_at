# GRPO late-checkpoint effective-update average

## Setup

Requested on 2026-10-23, after the completed checkpoint trajectory. This is one inference-only
test of an equal-weight average of GRPO-v3-all checkpoints 425, 450, 475, 500, 525, 550 and 571.
No learning-rate trial, optimizer step, new training seed, alternative averaging window or
search over averaging weights is authorized. The common parent is SFT-v3, and the original
9,140-row training run is unchanged. The seven-member window was selected with prior benchmark
results visible, so this is an exploratory diagnostic on an already-exposed benchmark.

Question: does averaging the actual LoRA updates produce a less extreme decision criterion
while retaining reference-label discrimination? A result consistent with this prediction
would support a useful smoothing interpretation. It would not establish that the optimizer
was sampling noise around a fixed point, or demonstrate stability across seeds or future data.
If discrimination degrades, that is evidence against the practical benefit of this particular
average; it does not uniquely establish why the individual checkpoints differ.

Controls are the seven individual checkpoints and the frozen SFT-v3 parent. Reuse their existing
complete raw generations from `outputs/checkpoint_benchmark/grpo_v3_all_20261023/`. Only the new
average receives a new generation pass. Primary scope is all 31,050 original English benchmark
prompts, including 450 attribute-free controls, evaluated against the original GPT-4o labels.
The frozen 90-candidate dev and 60-candidate historical remainder are descriptive partitions,
not alternative primary outcomes or fresh tests. No labels or split IDs change.

### Effective-update construction

The seven source adapters have identical ordinary LoRA configurations: rank 32, alpha 64,
scaling 2, 128 linear modules, FP32 A/B factors, and a common SFT-v3 parent. The intended model is

`W_average = W_SFT + (1/7) * sum_i [2 * B_i @ A_i]`.

Construct a rank-224 adapter by stacking all seven A matrices vertically and concatenating all
seven B matrices horizontally. Keep alpha 64, so the new scale is `64/224 = 2/7` and the product
is exactly the mean effective update in real arithmetic. Preserve all original factor blocks;
do not average A and B separately and do not perform an SVD truncation. Reject unsupported
configuration variants instead of silently approximating them. Verify factor identities,
dimensions, scaling and deterministic numerical action for every affected module.

Apply the common SFT adapter once, then the average adapter. Use the same bf16 merged serving
protocol as the original audits and record rounding retention against the effective average
update. If retention is below 80%, stop before interpreting model quality. Preserve the new
adapter, merged model, input hashes and construction/merge receipts in the experiment directory;
never overwrite a source checkpoint or an existing shared merge.

### Inference and metrics

Use the original binary baseline prompt, original prompt ordering, greedy generation,
temperature 0, top-p 1, seed 42, maximum 512 generated tokens, thinking disabled, official parser,
TP1 and the same serving software and base-model revision as the completed trajectory.
No new forced-prefix margin pass is part of this diagnostic. Derive SDT quantities from the
generated binary decisions, not from the old margin scorer.

Let H be the fraction of reference-hire rows called hire, and F the fraction of reference-reject
rows called hire. Add 0.5 to all four confusion-matrix cells consistently, so
`H=(TP+0.5)/(N_positive+1)` and `F=(FP+0.5)/(N_negative+1)`. Report both the original uncorrected
TPR/FPR and corrected rates used for finite z-scores. Define

`d_prime = Phi_inverse(H) - Phi_inverse(F)`

`c = -0.5 * [Phi_inverse(H) + Phi_inverse(F)]`.

Larger c describes a more conservative operating point. d-prime describes separation relative
to these reference labels under the SDT interpretation. Invariance of d-prime to criterion
depends on distributional assumptions such as equal-variance Gaussian evidence; a single
binary operating point is not a full ROC curve or a direct measurement of model capability.

Sources: the [MIT signal-detection lecture](https://ocw.mit.edu/courses/9-63-laboratory-in-visual-cognition-fall-2009/db92e2fb140ea77d5b1a30a78f0da154_MIT9_63F09_lec03.pdf),
[Stanislaw and Todorov](https://doi.org/10.3758/BF03207704), and
[Hautus on extreme-proportion corrections](https://doi.org/10.3758/BF03203619).

### Frozen reading rules

Quantify "criterion in the middle" as average-model c inside the seven single-checkpoint
interquartile range, using linear quantiles. Also show the full range and median. Compare
average d-prime separately with all seven checkpoints, their maximum, median and mean.
The strict point-estimate condition "no worse than every single checkpoint" is
`d_prime_average >= max(d_prime_single)`. No noninferiority margin is introduced.
Improvement over a median checkpoint and preservation of the best checkpoint are separate
observations; neither may be substituted for the other after seeing the average.

Bootstrap candidates jointly across all compared models, retaining all rows of each selected
candidate. Use 10,000 draws with PCG64 and `SeedSequence([20261023, scope_index])` separately for
full, dev and remainder. Report paired 95% intervals and, for the seven pairwise d-prime
differences, one-sided Bonferroni lower quantiles at `0.05/7`. These describe sensitivity to
candidate composition on this development benchmark. An interval containing zero does not
establish noninferiority, and these intervals do not validate the fixed-point mechanism.

Report original TPR/FPR, reference agreement, coverage and descriptive counterfactual set
instability alongside c/d-prime. Preserve the existing TPR drop <=10 pp, agreement drop <=5 pp,
coverage >=99%, and merge-retention >=80% guardrails against SFT-v3. No new matched-hire-rate
margin comparison is run, so this experiment cannot award a new formal fairness-win verdict.

The frozen sources and software must match. Abort on unsupported adapter layout, failed
equivalence checks, model/input hash mismatch, insufficient merge retention, incomplete or
duplicated benchmark rows, inference failure, or unusable decisions that prevent complete
paired SDT analysis. Report the failure instead of choosing a different average or decoder.

One submitted Slurm job owns the new model's entire inference pass. Expected runtime is
20–40 minutes after allocation, with a one-hour wall limit. No further job is automatically
submitted by the runner. Record actual wall time, active GPU cost and scheduler allocation.

## Result

Pending construction verification and the average-model inference. The source-only c/d-prime
table is computed before the average is evaluated and stored with its exact raw-input hashes.
All numerical conclusions must come from the experiment's generated JSON/CSV and per-row raw
outputs under `$HBM_OUTPUT_ROOT/outputs/checkpoint_average/grpo_v3_all_delta_avg_425_571_20261023/`.

## Reading

A central c with preserved d-prime would be consistent with useful averaging. It would not
uniquely identify optimizer noise, a common basin, a fixed point, or the reason checkpoint
500 differs from 525. A worse average also admits alternatives: a nonlinear interpolation
path, different representations, or numerical serving effects. The integrity and merge checks
address some numerical alternatives but do not establish a training mechanism.

The seven late checkpoints occur after the retrospective dev 8 pp stop at 400. This average
is a new post-hoc diagnostic model; it does not reverse the stopping-rule replay or become a
checkpoint that the frozen prospective rule would have selected. No training is enabled by
this experiment's result.

## Caveats

There are 150 candidates and 450 pairs, not 31,050 independent observations. Candidate
bootstrap does not remove historical benchmark exposure or make shared jobs independent.
The diagnostic uses original benchmark reference decisions, which are not an independent
adjudication of each hiring case. A single static average does not demonstrate lower
variance across repeated training runs. Native set instability remains confounded by the
operating point and is reported descriptively, without replacing the existing standard.

## Reproducing

The runner/config, average-construction helper and SDT analyzer are committed before the
formal manifest is frozen and before the one inference job is submitted. The result card
will record the exact code commit, configuration, manifest SHA, job ID, model identity,
per-row outputs, construction/merge checks and analysis artifacts.

Files are `configs/analysis/grpo_v3_all_checkpoint_average.json`,
`scripts/checkpoint_average_benchmark.py`, `scripts/average_lora_updates.py`,
`src/hiring_bias_mitigation/mitigation/lora_average.py`,
`scripts/analyze_checkpoint_average_sdt.py`, and
`scripts/cscs/checkpoint_average_benchmark.sbatch`.

After committing this protocol and code, prepare on CPU with the repository environment:

```sh
uv run --no-project --python "$XW/venv/bin/python" python \
  scripts/checkpoint_average_benchmark.py prepare \
  --config configs/analysis/grpo_v3_all_checkpoint_average.json
sbatch scripts/cscs/checkpoint_average_benchmark.sbatch
```

The frozen manifest includes this protocol's SHA. Keep this document unchanged after prepare.
Record results in `docs/GRPO_CHECKPOINT_WEIGHT_AVERAGE_RESULTS.md`. Main machine-readable outputs
are `manifest.json`, `adapter/average_manifest.json`, `merge_receipt.json`,
`average/raw.parquet`, `average/result.json`, `sdt_analysis.json`, and `sdt_metrics.csv`.
