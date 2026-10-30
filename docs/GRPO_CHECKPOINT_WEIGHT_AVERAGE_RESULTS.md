# GRPO late-checkpoint effective-update average: results

## Conclusion

The equal effective-update average places criterion c inside the preregistered seven-checkpoint IQR, but does not preserve the highest single-checkpoint d-prime. Full-benchmark mean7 c is 0.756152 and d-prime is 1.749208. Step500 d-prime is 2.059507; the paired difference is -0.310300, with candidate-bootstrap 95% interval [-0.526442, -0.095204]. This is exploratory evidence against the strict prediction of an average that is no worse than all seven checkpoints, not proof of a particular training mechanism.

The average improves the full d-prime point estimate over four late checkpoints and the seven-point median, but is slightly below the seven-point mean and below SFT-v3. Its candidate-paired interval versus the median crosses zero. Do not describe it as a free improvement over the best model or as established noninferiority. No training was started.

## Execution and integrity

- Preregistered code and protocol: `0f50ff61537476286aac976128f329754fc3e13e`.
- Manifest binding: `b16b43d9a3592a641834743bfc4859052ade5307411bb6b650920a840303986a`.
- Slurm job `5130000`, exit `0:0`, debug node `nid006557`, 2026-10-23 14:31:13 to 14:47:52 Europe/Zurich, elapsed 16m39s.
- The user authorized debug after submission. The same pending job was changed to eligible partitions debug/normal and a 30-minute limit, then ran on debug. No second job was submitted. `scheduling_amendment.json` records this scheduler-only change; frozen scientific files remain unchanged.
- Seven equal sources: 425, 450, 475, 500, 525, 550, 571; common SFT-v3 applied once. Rank224/alpha64 concatenation averages effective `2 * B @ A`; source A/B blocks remain unchanged.
- All 128 module FP64 equivalence checks passed. Final mean-adapter bf16 merge retention: 97.128758%, above the 80% requirement.
- All 31,050 original prompts have usable binary decisions. Full coverage 100%; same original prompt/parser, greedy512, thinking off and TP1 protocol.
- Independent CPU audit passed with no issues: 86 file hashes, all 128 modules and 896 source-module A/B block pairs, independent FP64 probes, eight generation chunks, official-parser reparse, complete row metadata and all 27 model/scope SDT point estimates. The independent audit did not rehash the large base/merged weights or recompute bootstrap intervals; it checked their bound receipts and existing output structure. See `validation.json` for exact checks and limits.
- GPU recorder: 0.2753 active GPU-hours on one GH200, GPU energy 405,934.8J. Scheduler allocation: four GPUs for 999s, 1.11 allocated GPU-hours, node energy 983,674J. Active-GPU and allocated-node costs measure different quantities.

## Primary full-benchmark results

| Model | TPR % | ΔTPR vs SFT pp | FPR % | c | d-prime | Reference agreement % | Raw set instability % |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT-v3 | 57.7295 | +0.0000 | 5.2657 | 0.712230 | 1.814377 | 82.3994 | 1.8519 |
| step425 | 59.9420 | +2.2126 | 5.5942 | 0.668885 | 1.841413 | 82.9179 | 1.5926 |
| step450 | 60.2319 | +2.5024 | 6.0000 | 0.647634 | 1.813923 | 82.7440 | 1.5926 |
| step475 | 49.3043 | -8.4251 | 4.5604 | 0.853134 | 1.671395 | 80.0612 | 1.5926 |
| step500 | 73.3720 | +15.6425 | 7.5556 | 0.405720 | 2.059507 | 86.0870 | 1.6296 |
| step525 | 48.4348 | -9.2947 | 4.4106 | 0.871954 | 1.665426 | 79.8712 | 1.0370 |
| step550 | 43.3140 | -14.4155 | 3.5121 | 0.989215 | 1.641691 | 78.7633 | 1.0741 |
| step571 | 47.6232 | -10.1063 | 4.2609 | 0.890274 | 1.661333 | 79.7005 | 1.1111 |
| mean7 | 54.7150 | -3.0145 | 5.1449 | 0.756152 | 1.749208 | 81.4750 | 1.4074 |

The frozen c IQR is [0.658259, 0.881114], median 0.853134. Mean7 is inside it. Its TPR change is -3.0145 pp and reference-agreement change is -0.9243 pp, so both original full guardrails pass: TPR drop at most 10 pp and agreement drop at most 5 pp. Passing these guards does not establish a fairness win.

| Full d-prime comparison | Mean7 minus comparator | Paired 95% interval | Bonferroni one-sided lower bound |
|---|---:|---:|---:|
| step425 | -0.092206 | [-0.203906, +0.009653] | -0.230440 |
| step450 | -0.064716 | [-0.174036, +0.040327] | -0.202451 |
| step475 | +0.077812 | [-0.037960, +0.179489] | -0.077714 |
| step500 | -0.310300 | [-0.526442, -0.095204] | -0.573373 |
| step525 | +0.083781 | [-0.041711, +0.190857] | -0.081551 |
| step550 | +0.107516 | [-0.101430, +0.281254] | -0.170845 |
| step571 | +0.087874 | [-0.044896, +0.211347] | -0.085298 |
| vs_seven_max | -0.310300 | [-0.527147, -0.106074] | n/a |
| vs_seven_median | +0.077812 | [-0.076939, +0.147356] | n/a |
| vs_seven_mean | -0.015748 | [-0.105144, +0.066254] | n/a |

The maximum/median/mean comparator is recomputed within each paired candidate-bootstrap draw. This explains why the interval against the resampled maximum differs from that against the fixed step500 model. Full mean7-minus-step500 is below zero throughout its ordinary paired 95% interval. The other six individual full comparisons cross zero; those intervals do not establish noninferiority.

## Frozen descriptive partitions

| Scope | TPR % | ΔTPR pp | FPR % | c | d-prime | c in seven-point IQR | d-prime ≥ all seven | d-prime ≥ median | Raw instability % |
|---|---:|---:|---:|---:|---:|---|---|---|---:|
| full | 54.7150 | -3.0145 | 5.1449 | 0.756152 | 1.749208 | True | False | True | 1.4074 |
| dev | 53.4362 | -2.6648 | 6.2556 | 0.723573 | 1.619598 | True | False | True | 0.6173 |
| historical_remainder | 56.8014 | -3.5850 | 3.5466 | 0.816968 | 1.976489 | True | False | False | 2.5926 |

All three c values are central under their frozen IQR definition. None meets the strict d-prime condition. Historical remainder d-prime is even below its seven-point median and mean. Raw instability is lower on dev but higher on historical remainder than SFT and every individual checkpoint, so an unqualified claim of greater stability is also unsupported.

## Post-hoc audit of existing decisions

After the average metrics were observed, compare the already-saved decisions around 475/500/525 and mean7. This is an additional descriptive analysis, not a replacement for the preregistered outcome. All eight raw files have the same 31,050 unique row keys and identical input/label metadata.

| Transition | Reference hire: H to R / R to H | Reference reject: H to R / R to H | Changed rows | Candidates | Pairs |
|---|---:|---:|---:|---:|---:|
| 475 to 500 | 0 / 2,491 | 0 / 620 | 3,111 | 59 | 67 |
| 500 to 525 | 2,581 / 0 | 651 / 0 | 3,232 | 54 | 66 |
| 475 to 525 | 314 / 224 | 102 / 71 | 711 | 23 | 26 |
| 500 to mean7 | 1,931 / 0 | 499 / 0 | 2,430 | 49 | 57 |

Step500's hire set strictly contains the hire sets of 475, 525 and mean7 on these prompts. Mean7 loses 1,931 reference-hire decisions and corrects 499 reference-reject decisions relative to 500, with no reject-to-hire transitions. That is compatible with tightening a threshold; the d-prime decrease alone does not establish changes beyond a threshold. Conversely, the two-way changes between 475 and 525 mean the entire trajectory cannot be described exactly by moving one threshold over a single fixed ranking. Neither observation identifies the optimizer's mechanism.

There are 2,816 reject/hire/reject rows across 475/500/525: 2,267 reference hires and 549 reference rejects. No hire/reject/hire rows occur. The 500-to-mean7 changes span 49 candidates and 57 pairs; the five largest candidate counts account for 24.07% of changed rows, so the difference is not confined to one or two candidates. Rows include repeated protected-attribute variants and are not independent people.

Mean7 agrees with seven-source majority decisions on 98.7633% of rows. The 384 disagreements span 19 candidates and 19 pairs: 322 improve agreement with the reference and 62 worsen it. Mean7 opposes none of the seven-unanimous decisions. Weight averaging was used to build the model; these vote counts are only a post-hoc comparison, not a prediction-ensemble implementation.

Reproducible evidence is in `posthoc_transition_diagnostics.json`, its compact CSV and `posthoc_transitions.py` in the experiment directory. The JSON records raw-source SHA values and per-candidate/pair counts without reassigning reference labels.

## Interpretation and next analysis

This average softens the conservative late-run operating point relative to 525/550/571, with a smaller TPR loss. It does not retain step500's stronger reference-label discrimination. The result fails the combined preregistered condition of central c and d-prime no worse than all singles. It neither proves nor rules out fixed-point optimization noise: c/d-prime from one binary operating point rely on SDT assumptions, and nonlinear model averaging can change responses beyond a threshold shift.

The row-level audit above narrows the next question to the 57 changed pairs and the rank changes between 475 and 525. Reviewing their job requirements/reference labels and, if separately authorized, comparing continuous margins at matched operating points would help separate criterion shifts from ranking changes. Retain mean7 as an exploratory comparison model. This result alone does not establish that lowering the learning rate is the remedy. No new training, average window, weights, model inference or label change is authorized by this result.

All seven averaged checkpoints follow the retrospective first dev8 trigger at step400. This new post-hoc average does not reverse that stop or become prospectively selectable under it. The earlier full/dev stopping-monitor failure remains. `training_authorized=false` is unchanged; this inference does not validate a GPU training callback.

## Limits and artifacts

- The original benchmark has already influenced development. Neither dev nor historical remainder is a fresh independent test set.
- Candidate bootstrap uses 150 full candidates, not 31,050 independent rows, and does not remove dependence through shared jobs. Its intervals describe candidate-composition sensitivity on exposed data.
- Reference decisions are GPT-4o labels, not verified hiring outcomes. The earlier audit conclusion "RJDB reject anchors are questionable" remains; this experiment does not re-adjudicate labels.
- No matched-hire-rate margin pass was run, so native instability and agreement cannot establish the published fairness-win criterion. No published historical benchmark table was changed.
- Formula, correction, bootstrap seed/count and comparison definitions remain exactly in [the frozen preregistration](GRPO_CHECKPOINT_WEIGHT_AVERAGE.md).

Remote evidence root: `$HBM_OUTPUT_ROOT/outputs/checkpoint_average/grpo_v3_all_delta_avg_425_571_20261023/`. Main evidence: `manifest.json`, `adapter/average_manifest.json`, `adapter_receipt.json`, `merge_receipt.json`, `average/raw.parquet`, `average/result.json`, `sdt_analysis.json`, `sdt_metrics.csv`, `scheduling_amendment.json`. Local report mirrors the recorded JSON without resampling.

### Evidence hashes

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `3fb053ff39c8a4f07ee02341c28252130059f126a2db810cefefb5e12f18f75e` |
| `adapter/average_manifest.json` | `e760aa3ef8530c47794448a38d24416affd9e2e5e8f082d079d86b05a1cf1478` |
| `adapter_receipt.json` | `e618610219625a2531be7ec0eb71f7ea8c9ca51cc62fd9a85590b69684d30fe1` |
| `merge_receipt.json` | `601a80b399401faa30bdc32565d52864abf5373940b3a5c27669fd27d7bf6337` |
| `average/raw.parquet` | `9d1b14c9a90e531c88ea0277b7b25cd2c6da0de3aecd9df779980b3be6db2075` |
| `average/result.json` | `59e09b6eef76e3801b4313b007aefc78e1da3be1254bff4891a07965636a6545` |
| `sdt_analysis.json` | `5d562c705e4060e9af8be298be5e8ad8460851ce4bbbf9144f01300c7fe95e5b` |
| `sdt_metrics.csv` | `60164a656eb8bda7634fb8d576d24322eb3d82537e598d7480e1cd370fa9ea1f` |
| `posthoc_transition_diagnostics.json` | `07966b8dc21f07abf10443b83e87ca8c4b5bcc6cd5b28f5ee064d4aafcaa3721` |
| `posthoc_transition_diagnostics.csv` | `14c423c9484fb634b2cfee32323fd78bb6109745c69679d5b3d5e575271e1270` |
| `cost_receipt.json` | `405104e1d0d8bed9706eb05960d568ee55d156218ec4e23f1787da7fc6494375` |
| `validation.json` | `751351694037f23f3f5190b8d4360ee583cbbdecb1dd8df74349365eeba39f67` |
