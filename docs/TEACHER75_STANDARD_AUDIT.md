# Full re-audit of 75 rows and 40 pairs under the unified standard

Standard `hiring-evidence-v1`, commit `323d4f5`. Two independent reviewers assessed each pair against the same frozen JD checklist, producing 80 evidence ledgers.
Standard fingerprint: `37dda6072e102f04274bb94fdce75bdd6063de0792c50e3ef6907202fc70ff53`, using canonical-json-sha256.

Hire requires supporting evidence for every explicitly mandatory requirement and positive evidence of job fit. Reject requires explicit evidence that a mandatory requirement is not met. Unknowns, critical conflicts, and disagreements between reviewers go to the review queue. No third reviewer adjudicated disagreements.

## Updated categories

| Category | Pairs | Rows |
|---|---:|---:|
| DAPO decision degradation | 3 | 3 |
| Anchor problem | 9 | 13 |
| Insufficient information or reviewer disagreement | 28 | 59 |

Audit-label denominators: hire, 12 pairs / 16 rows; reject, 0 pairs / 0 rows; review, 28 pairs / 59 rows.
Resolved binary coverage is 12/40 = 30.0% of pairs and 16/75 = 21.3% of rows. Review is not a binary error and is excluded from the binary-comparison denominator for the resolved subset.

If both reviewers agree on hire/reject and that label matches the anchor, classify the case as DAPO decision degradation. If their agreed label opposes the anchor, classify it as an anchor problem. Any review outcome or disagreement is classified as insufficient information.

The original samples for this subset are unchanged: SFT-v3 agrees with the anchor in 16/16 attribute-free samples, and DAPO agrees in 0/16 samples for both attributed and attribute-free inputs. Historically, "all wrong" meant disagreement with the old anchor; the current categories reflect evidence judgments under v1. Only the first category, confirmed by review, supplies label-correction targets for the frozen SFT teacher.

## Stratification by anchor label and vote count

| Anchor | Winning/valid votes | Pairs/rows | DAPO degradation pairs/rows | Anchor-problem pairs/rows | Insufficient-information pairs/rows |
|---|---|---|---|---|---|
| hire | 5/8 | 3/11 | 0/0 | 0/0 | 3/11 |
| hire | 6/8 | 3/4 | 0/0 | 0/0 | 3/4 |
| hire | 7/8 | 7/20 | 1/1 | 0/0 | 6/19 |
| hire | 8/8 | 7/12 | 2/2 | 0/0 | 5/10 |
| reject | 5/8 | 4/7 | 0/0 | 1/3 | 3/4 |
| reject | 6/8 | 7/9 | 0/0 | 6/8 | 1/1 |
| reject | 8/8 | 9/12 | 0/0 | 2/2 | 7/10 |

## Correction targets for the frozen SFT teacher

Retain 3 pairs and 3 rows. This confirms only the label direction; it does not mean training has corrected the rows or authorize automatic reuse of SFT's original feedback.

| Pair | Row ID | Target | Attributed SFT samples matching target | Review counts H/D/I/U/F |
|---|---:|---|---:|---|
| P02 | 3930 | hire | 15/16 | 9/6/3/0/0; 9/7/2/0/0 |
| P31 | 4653 | hire | 16/16 | 0/0/0/0/0; 0/0/0/0/0 |
| P04 | 9117 | hire | 15/16 | 19/13/6/0/0; 19/13/6/0/0 |

Attributed SFT samples match the target in 46/48 cases, or 95.8%. Per-row agreement ranges from 15 to 16/16.

Each candidate's target_reason stores job-fit grounds and mandatory-requirement evidence from both reviews in this round. correction_conditions stores the applicable fixed inputs and data-quality flags. Year-only durations use nominal differences and must not be interpreted as verified exact completed years. Original SFT rationales were not checked individually; rationale_reuse_unchecked=true.

Of the original 4 pairs and 7 candidate rows, 3 rows were retained and 4 removed; 0 new rows were added.
Retained rows: [3930, 4653, 9117]. Removed rows: [2405, 4411, 4556, 6152]. Added rows: [].

## Changes from the earlier 40-pair audit

| Previous category | New category | Pairs | Rows | Pair IDs |
|---|---|---:|---:|---|
| DAPO decision degradation | DAPO decision degradation | 3 | 3 | P02, P04, P31 |
| DAPO decision degradation | Insufficient information or reviewer disagreement | 1 | 4 | P25 |
| Anchor problem | Anchor problem | 9 | 13 | P07, P08, P09, P22, P24, P27, P32, P34, P37 |
| Anchor problem | Insufficient information or reviewer disagreement | 7 | 10 | P05, P06, P14, P16, P17, P30, P39 |
| Insufficient information or reviewer disagreement | Insufficient information or reviewer disagreement | 20 | 45 | P01, P03, P10, P11, P12, P13, P15, P18, P19, P20, P21, P23, P26, P28, P29, P33, P35, P36, P38, P40 |

## Complete evidence and limitations

For item-level evidence on all 40 pairs, reasons the 59 rows remain pending, and the three final-decision disagreements, see the [full report](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/report.md) and [40-pair index](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/case_index.md). Updated categories for the 75 rows are in [JSON](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/audit_rows_75.json); confirmed targets are in the [candidate file](/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/teacher_correction_candidates.json).

The three final-decision disagreements are P06, P25, and P39, covering 7 rows, all excluded from hard correction targets. Across the 80 evidence tables, 2,413 quotation instances were checked and located in the sources. These are model text reviews, not human gold labels; confirmation of a correction direction does not establish a training benefit.

## Reproducing

The standard and English prompts were committed first as `323d4f5e44cbda9da3ae5f75fa05f1ca98776a52`. Complete review artifacts are stored at:

`/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full/`

`manifest.json` records policy and input-file hashes, reviewer assignments, and validation results. `frozen_job_rubrics.json` stores all 40 JD checklists and their extraction boundaries; `source_cases.json` contains attribute-free JD/CV text; `reviews/` stores 80 evidence tables from 8 independent review tasks. The previous 4 v1 JD checklists were retained; the other 36 were extracted from JDs alone. All 40 pairs received two fresh CV reviews.

Recompute the tables in the repository using the existing CPU environment, without calling a model:

```bash
audit_dir=/ritom/scratch/cscs/zhang/xw/outputs/paired_probe/20261022-dapo-final-sft-v3-k16/teacher75_v1_full
uv run --no-project --python /ritom/scratch/cscs/zhang/xw/venv/bin/python \
  python scripts/teacher75_standard_analysis.py \
  --policy configs/hiring_decision_standard_v1.json \
  --sources "$audit_dir/source_cases.json" \
  --rubrics "$audit_dir/frozen_job_rubrics.json" \
  --reviews "$audit_dir/reviews" \
  --old-rows "$audit_dir/legacy_audit_rows_75.json" \
  --pair-metadata "$audit_dir/pair_metadata.json" \
  --output "$audit_dir"
uv run --no-project --python /ritom/scratch/cscs/zhang/xw/venv/bin/python \
  python "$audit_dir/export_parquet.py" --directory "$audit_dir"
```

`audited_rows_75.parquet` contains the updated 75-row detail; `teacher_correction_candidates.parquet` contains only confirmed correction candidates. Nested evidence is stored in explicit `*_json` Parquet columns. See `parquet_schema.json` for column mappings and roundtrip checks; native structures remain in the corresponding JSON files.

The policy validator's 18 tests and the aggregator's 16 tests cover evidence references and decision thresholds, agreement between reviews, mapping of the 75 rows, vote stratification, and exclusion of unresolved cases. The actual data also underwent an independent aggregation check; see `independent_qa.json`.

This reproduces the saved review results and their statistics; it does not guarantee verbatim-identical opinions from fresh model calls. The original train/validation splits, benchmark, model weights, and sampling records were not modified. No new training or DAPO/SFT sampling was started.
