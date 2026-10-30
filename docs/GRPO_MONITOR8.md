# GRPO with a calibrated 8 pp TPR stop

## Setup

User authorization, 2026-10-22: first calibrate the 696 held-out monitor rows. Only if calibration passes, train a GRPO comparison on the original prompt and 9,140 rows with an 8 pp projected-TPR stop. Do not retrain the other models.

Question: can the new monitor detect the TPR losses missed by the original monitor, and can its stop rule preserve the original GRPO arm's stability benefit without exceeding the TPR guardrail?

The parent and final comparison reference are SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`. The existing GRPO-v3-all run is the training-procedure control. DAPO and the other completed arms are transfer checks only. This is one seed and one GRPO run, conditional on calibration.

The original binary prompt, rewards, 9,140 training rows, 256 validation rows, SFT-v3 checkpoint, optimizer, rollout settings and data split remain the control settings. The 696 monitor rows come from 222 held-out candidates. No v1 evidence-ledger prompt, review output or teacher-correction candidate is used in this experiment.

## Calibration acceptance, fixed before GPU scoring

The registered config is `configs/calibration/grpo_v3_all_monitor8.json`. It pins 13 source files, including the training/validation/monitor Parquets, parent benchmark responses, all eight benchmark margin files and the existing generated-response metrics. Eight existing merged models score the same ordered monitor prompts with the existing op-margin-v2 decision scorer.

Every condition below must pass:

1. Exactly 696 finite scores per model, identical ordered prompt digest and model identity, with 222 monitor candidates and no training-candidate overlap. The original training set has 9,140 rows and the registered hashes still match.
2. Every benchmark-weighted bin has at least one row when binned by the freshly scored SFT-v3 decision margins. Report both rows and distinct candidates per bin. Sampled hire-rate bins in `monitor.json` do not establish this coverage.
3. For all seven other arms, absolute error between monitor-projected TPR change and actual attributed-benchmark margin TPR change is at most 2.0 percentage points.
4. For every arm, optimism against its existing all-condition generated-response TPR change is strictly less than 2.0 pp. Optimism is projected change minus actual change, so a positive value hides some of the loss.
5. Any arm whose observed margin-based or generated-response TPR change is below -8 pp must also have a projected change below -8 pp. This includes DAPO final, GRPO final and GRPO checkpoint 525.

FPR errors and per-bin row and candidate counts are reported. They do not replace these predeclared TPR gates. Weights come only from the SFT-v3 benchmark margin distribution, using the existing calibration formula; no arm's transfer result is used to refit weights or move thresholds. A failed gate means no training submission.

This checks transfer on completed arms, not every future optimization trajectory. It also accounts for the difference between attributed margin TPR and the final guardrail's all-condition generated-response TPR. It is not a guarantee that a run cannot cross the guardrail between the 25-step checks.

## Conditional training and comparison

If calibration passes, create a new output directory and GRPO config cloned from `qwen3.5-9b_en_only_grpo_v3_all_gh200.yaml`. Add the frozen SFT-only `margin_monitor` calibration, `steps: 25` and `tpr_drop_stop: 0.08`; start fresh rather than resuming the old run. Keep the original 15–35% hire-rate band and memory floor as additional stop conditions.

Budget: one epoch, approximately 571 optimizer steps, with an 11 h 15 min wall-time cap on one four-GH200 node, at most 45 allocated GPU-hours. The calibration has a separate 45-minute cap on the same node size. No automatic extension or second seed is included.

Checkpoint selection is declared before training: use the latest saved checkpoint that has a recorded margin-monitor reading at or above -8 pp and before any safety stop. Keep the checkpoint that triggers a stop for diagnosis. If the first monitored checkpoint already breaches the line, report that no eligible new GRPO checkpoint exists and retain SFT-v3 as the baseline. Do not select using benchmark results.

Evaluate the selected checkpoint once using the original binary audit protocol and `docs/EVAL_STANDARD.md`: matched-hire-rate instability versus SFT-v3, TPR/FPR, agreement and undecided coverage. Report actual training steps and the stopping reason alongside the existing GRPO-v3-all result. An interval touching zero or any failed guardrail does not support improvement.

## Result

Calibration failed. No GRPO comparison was submitted, and no other model was retrained.

Job `5123830` completed successfully on one four-GH200 node in **5 min 15 s**, scoring all eight existing merged models. The scoring/pre-registration commit was `1703309`; the gate serialization fix is `c637bf1`. The fix changes NumPy scalar serialization and atomic report publication only. The registered data, numerical projection and acceptance thresholds are unchanged. The updated gate and calibration tests passed: 34 tests and 5 subtests.

The formal result is `$HBM_OUTPUT_ROOT/outputs/monitor_calibration/grpo_v3_all_monitor8_20261022/gate-v2.json`, status `calibration_failed`, exit code 2. The first attempted `gate.json` is a partial file from the serialization error and is not an acceptance report.

All data-integrity checks passed: 696 finite margins for every model, identical ordered prompt digest, 222 held-out candidates, no training-candidate overlap, unchanged 9,140 training rows and all 13 source hashes matching. Fresh parent-margin bins contain `[100, 91, 81, 61, 65, 103, 84, 111]` rows, so every positive-weight bin is covered.

All values below are percentage-point changes relative to SFT-v3. Margin results use the 30,600 attributed benchmark rows and the same scorer as the monitor. Generated results use the frozen full-audit metrics, including the attribute-free control; they are a separate guardrail check, not interchangeable measurements.

| Arm | Monitor ΔTPR | Benchmark margin ΔTPR | Generated ΔTPR | Monitor minus margin | Gate |
|---|---:|---:|---:|---:|---|
| DAPO-v3-all | -3.24 | -9.61 | -10.10 | +6.37 | fail |
| GRPO-v3-all | -4.59 | -9.61 | -10.11 | +5.01 | fail |
| GRPO-v3-all-step525 | -2.99 | -8.82 | -9.30 | +5.83 | fail |
| GRPO-v3 | +0.33 | -1.03 | -1.24 | +1.36 | pass |
| DPO-v3 | -2.25 | -0.49 | -0.98 | -1.76 | pass |
| DPO-v3-onpolicy | -0.81 | -2.39 | -2.92 | +1.58 | fail |
| DPO-v3-4880 | -0.84 | +1.24 | +0.70 | -2.08 | fail |

Only GRPO-v3 and DPO-v3 passed every registered arm check. DAPO-v3-all, GRPO-v3-all and step 525 each failed both error checks and the historical-stop check: all three should trigger an 8 pp stop, but none would. DPO-v3-onpolicy hides 2.11 pp of its generated TPR decline, above the strict <2 pp bound. DPO-v3-4880 has 2.08 pp absolute margin error, above the 2 pp bound.

FPR also transfers poorly in the major failing arms. DAPO's benchmark margin FPR change is -1.94 pp but the monitor projects +2.60 pp; step 525 is -0.77 pp versus +4.43 pp. FPR was diagnostic only and did not determine acceptance.

## Reading

Covering the parent's margin bins does not make the held-out rows respond like benchmark rows within each bin. The CPU benchmark-self estimate remains close: DAPO actual -9.61 pp versus estimate -10.41 pp; GRPO-v3-all -9.61 versus -9.20 pp; step 525 -8.82 versus -8.28 pp. Substituting the held-out changes yields the much smaller losses above. Thus the unverified transfer assumption identified before this run did not hold for this monitor cohort.

The existing-score diagnostic `scripts/diagnose_monitor_transfer.py`, commit `1b6e19d`, separates the projection error without fitting any parameter. Results are in the same score directory as `diagnosis.json`. The three terms below sum to monitor projection minus actual benchmark margin TPR change. All independently computed production projections match the gate to 1e-10 pp.

| Arm | Same-bin cohort transfer | Benchmark label pooling | Weight rounding | Total error |
|---|---:|---:|---:|---:|
| DAPO-v3-all | +7.169 | -0.798 | -0.003 | +6.368 |
| GRPO-v3-all | +4.611 | +0.408 | -0.004 | +5.014 |
| GRPO-v3-all-step525 | +5.289 | +0.551 | -0.005 | +5.835 |

The dominant term is the difference between pooled monitor and pooled benchmark decision changes within the same parent-margin bins. This term includes all within-bin differences in sample composition; it does not identify their causal origin. For DAPO, the bin with parent margins 0.847 to 2.197 loses 36.89 pp hire share on monitor rows versus 63.78 pp on benchmark rows. The -2.197 to -0.847 bin gains 25.27 pp on monitor rows versus 0.00 pp on benchmark rows. These gains offset some projected losses.

The monitor parent hires 68.23% of its 362 anchor-hire rows, versus 57.17% of the 10,200 attributed benchmark reference-hire rows under the same margin scorer. These are descriptive unweighted rates on differently sampled populations, not a replacement acceptance test. Simply conditioning the monitor changes on anchor-hire rows is not a validated repair: the exploratory exact-weight projections become +7.80 pp for DAPO and +9.56 pp for step 525, while their benchmark TPR falls. Some label-specific bins are sparse; the lowest bin has only eight anchor-hire monitor rows. Scores alone cannot determine whether semisynthetic anchors, benchmark labels or other text differences explain the mismatch.

The 8 pp stopping code cannot provide the intended guardrail with these readings. This experiment remains stopped at calibration. No weights or thresholds were refitted after seeing the failed results, and the prepared checkpoint-selection helper was not deployed into training.

## Caveats

This diagnoses the monitor's transfer failure. It does not establish whether a newly monitored GRPO run would improve fairness or preserve benchmark TPR, because that run was not authorized after a failed calibration. It also does not identify a unique cause of the within-bin mismatch. The monitoring rows share 222 candidates, so the 696 rows are not independent samples. No confidence interval was computed for this calibration.

Any revised monitor evaluated against these same seven completed arms is now development work on observed results. A credible pass for deployment needs a separately declared validation check, rather than relaxing this gate or fitting the weights to these losses.


## Reproducing

The calibration job uses `scripts/cscs/monitor_calibration.sbatch` with the eight registered `LABEL=model` entries, a unique `SCORES_DIR`, and the seven registered `LABEL=benchmark-run` comparisons. `scripts/check_monitor_transfer.py` will verify source integrity and write the machine-readable acceptance result. A nonzero gate exit prevents training.

All new outputs are under `outputs/monitor_calibration/grpo_v3_all_monitor8_20261022`. Original models and original train/validation/monitor data are retained. The job log is `$HBM_OUTPUT_ROOT/logs/xw-monitor8-cal-5123830.out`; each model has a `xw-monitor-cal-5123830-<label>.log`. The score directory contains eight `<label>.parquet` files, the original `submission.json`, and the formal `gate-v2.json`. The registered config and report include source and score SHA256 hashes. The only unrelated untracked repository file at scoring/result recording was `docs/FRONTIER_RL_SURVEY.md`, which was not staged.

To repeat the CPU checks without overwriting the retained reports, use a fresh output filename with each command:

```bash
uv run python scripts/check_monitor_transfer.py --config configs/calibration/grpo_v3_all_monitor8.json --scores-dir "$HBM_OUTPUT_ROOT/outputs/monitor_calibration/grpo_v3_all_monitor8_20261022" --output <fresh-gate.json>
uv run python scripts/diagnose_monitor_transfer.py --config configs/calibration/grpo_v3_all_monitor8.json --scores-dir "$HBM_OUTPUT_ROOT/outputs/monitor_calibration/grpo_v3_all_monitor8_20261022" --output <fresh-diagnosis.json>
```

## Existing-score source diagnosis and proposed repair

User instruction: explain the within-bin mismatch with existing data, propose a correction and validation plan, and do not start training. This follow-up used no new model scoring and changed no production monitor, gate threshold, prompt, training data or benchmark label.

The monitor mixes 504 Djinni rows from 171 candidates with 192 RJDB rows from 51 candidates. The benchmark is entirely Djinni. Within the same parent-margin bands, the model updates often increase hires on RJDB while reducing hires on Djinni. The source-stratified diagnostic below uses the same frozen rounded weights as the original projection; all changes are pp relative to SFT-v3. It is exploratory and does not reverse the failed gate.

| Arm | Original monitor | Djinni only | RJDB only | Benchmark actual |
|---|---:|---:|---:|---:|
| DAPO-v3-all | -3.239 | -8.973 | +11.533 | -9.608 |
| GRPO-v3-all | -4.593 | -8.172 | +3.638 | -9.608 |
| GRPO-v3-all-step525 | -2.989 | -10.725 | +20.343 | -8.824 |
| GRPO-v3 | +0.329 | -2.711 | +7.929 | -1.029 |
| DPO-v3 | -2.253 | -2.262 | -2.449 | -0.490 |
| DPO-v3-onpolicy | -0.810 | -2.991 | +5.699 | -2.392 |
| DPO-v3-4880 | -0.844 | -1.054 | -0.235 | +1.235 |

Using exact parent weights, the algebraic source-mixture contributions to projection optimism are +5.733 pp for DAPO, +3.578 pp for GRPO final and +7.736 pp for step525. The remaining Djinni-cohort residuals are +0.639, +1.441 and -1.896 pp. Rounding contributes less than 0.006 pp. This separates observed sources of the error; source is correlated with text construction, occupations and labels, so it does not isolate a causal mechanism.

RJDB is synthetic English JD/CV data generated from a shared skills/occupation seed. The configured pool takes matched pairs and disables weakened CV variants (`docs/EXTERNAL_DATA.md:109`, `generation/sources.py:50,674`, `configs/generation/v3_new_pairs.yaml`). In pair `rjdb:rjdb-train-12957-match`, CV and JD share the same four roles and 5/3/5/1-year requirements; DAPO gains hires on 10 of its 11 monitor rows. Pair `rjdb:rjdb-train-03727-match` repeats the same BSS/OSS/Java/CRM/POS skills and one-year lead role on both sides, with 9/9 gains. Full input examples and anchor votes are retained in `source_audit.json`. These examples support a text-domain hypothesis; they are not new truth labels.

All 696 monitor prompts were reconstructed byte-for-byte through the original shared binary `build_prompt(strategy="baseline")`. The calibration did not use the later audit prompt. Monitor anchors nevertheless differ from benchmark references: Qwen-122B eight-vote majority versus the released GPT-4o field. Old sampled-hire-rate selection bins and fresh scorer-margin bins differ for 215/696 rows. All monitor anchor-hire rows used k32 estimates and anchor-reject rows used k8 estimates. Monitor pairs have 1–12 selected attribute rows, while each benchmark pair has 68 attributed rows.

### Why the source-only adjustment is not a validated repair

DPO-v3-4880 still has about 2.29 pp absolute TPR error. Adding group/condition alignment to Djinni gives GRPO final -7.886 pp, which again misses the 8 pp line. Neither 16 parent-only bins nor candidate-equal weighting solves the original mixed-source failure.

The Djinni-only projection has candidate-cluster bootstrap 95% intervals [-13.64,-4.82] for DAPO, [-12.66,-3.89] for GRPO final and [-15.48,-5.98] for step525. This uses 3,000 draws, seed20261022, and keeps the benchmark fixed. It estimates empirical-monitor sampling variation, not selection/transport bias. The Djinni subset has 171 distinct candidates, candidate-weight ESS about129, minimum13 candidates per margin bin and minimum4 after group/condition refinement. Removing one candidate can move DAPO's estimate above the stop threshold. Directly conditioning on Qwen anchor-hire is especially unstable: the lowest bin has only8 rows/4 candidates but26.91% target hire weight.

### Proposed correction and independent validation

1. Match the monitoring domain to the target: use Djinni for Djinni benchmark TPR and report RJDB separately. Keep original binary prompt and hire meaning. Use fresh SFT scorer margins for sampling/weighting, not historical k8/k32 bins.
2. Sample independent candidates first; declare attribute/condition sampling weights, record job reuse, and check job-cluster sensitivity. Add coverage for tail and critical margin regions. Attribute duplicates do not supply independent information.
3. Align the binary reference-label protocol before using label-specific TPR. Current Qwen-hire cannot be substituted for GPT-4o-hire without checking comparability. Prefer a labeled, same-source direct TPR monitor over relying only on cross-domain projections. Report loss, gain and net separately; loss-only is an additional diagnostic, not TPR decline.
4. Proposed support screen: at least20 distinct candidates per positive-weight coarse label×margin cell, plus a candidate-cluster95% half-width target of2pp. Twenty is not a sufficiency guarantee. Unsupported target mass or inadequate precision must be explicit and prevents release, rather than silently assigning zero change.
5. All seven completed arms and the current696 rows are development evidence. Freeze the candidate pools, labels, estimator, thresholds and model list before scoring independent candidate-disjoint Djinni monitor/validation banks. Respect all existing training/benchmark exclusions and keep the original benchmark unchanged. Step525 and final are one trajectory, not independent validation arms.
6. Keep the original acceptance requirements: every arm's margin absolute error≤2pp; optimism against generated TPR<2pp; every actual>8pp decline triggers a stop. Also report false stops on safer arms, FPR and uncertainty. Validate the frozen proposal with existing models only; no new training is needed. If validation fails, do not tune on it and then call it independent.
7. Before eventual deployment, declare the repeated-check interval rule. Retain25-step checks and an8pp net-TPR stop; a conservative calibrated bound and missing-support stop are proposed safeguards. Single-check95% intervals do not establish a whole-run guarantee, and domain bias/25-step overshoot remain separate risks. A loss-only stop needs its own declared threshold and false-stop validation.

### Available independent validation data

The previous1,155 count is the union of historical validation-ID registries. The two relevant on-policy pools actually contain420 held-out candidates/5,039 rows. Djinni contributes338 candidates/4,055 rows, with74 anchor-hire and264 anchor-reject pairs. After excluding the current monitor candidates and all train/benchmark candidate/job exclusions, only167 unused Djinni candidates/2,004 rows remain: **8 hire pairs and159 reject pairs**. The unused31 RJDB pairs do not repair Djinni target coverage.

Existing SFT k8/k32 caches cover this pool. No DAPO/GRPO per-row margins were found for the unused candidates in the relevant score inventories; the16-sample paired experiment covers the9,140 training rows and old training monitors retained only aggregate checks. Therefore the next step is first to obtain enough untouched Djinni hire cases and check binary labeling, then freeze an inference-only validation manifest. Current unused rows are insufficient for a precise, fully stratified independent TPR validation.

### Evidence and reproduction

All files are under `$HBM_OUTPUT_ROOT/outputs/monitor_calibration/grpo_v3_all_monitor8_20261022/`. `diagnostic_inputs.jsonl.gz` contains frozen existing rows and margins; `monitor_provenance.json` supplies source and anchor information. `numeric.json` includes hashes, all seven arms, fixed ablations, source-mixture identities and bootstrap results. `source_audit.json` contains source citations and input cases. `validation_plan.json` is a proposal, not deployed config. `verification.json` records21 successful checks against the original gate and independent source analysis.

Run the CPU-only analysis with a fresh output destination:

```bash
uv run python scripts/analyze_monitor_transfer.py --rows "$HBM_OUTPUT_ROOT/outputs/monitor_calibration/grpo_v3_all_monitor8_20261022/diagnostic_inputs.jsonl.gz" --output <fresh-numeric.json>
```

No training was started by this follow-up.
