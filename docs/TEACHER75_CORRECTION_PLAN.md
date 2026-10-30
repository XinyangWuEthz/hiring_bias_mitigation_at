# Correction plan and current entry point for the 75 rows

The current policy is `hiring-evidence-v1`; see the [standard](HIRING_DECISION_STANDARD.md) for the unified rules and two-stage prompts. This update applies the standard to input validation, category recomputation, and correction-data export. The categories still derive from the completed 80 blind-review records.

## Setup

The question is how many correction directions on these 75 rows have review evidence supporting the frozen SFT teacher. The scope is fixed at 40 pairs and 75 rows from the existing paired probe: SFT agreed with the old anchor in 16/16 attribute-free samples, while DAPO agreed in 0/16 samples for either input view. The comparison uses the original samples and two independent v1 text reviews, with no new model generation.

Expectation: if DAPO truly degraded, the reviews should support the SFT direction. If they support DAPO or retain review, the old anchor cannot justify forcing DAPO to follow the teacher. Export stops on any mismatch in versions, quotations, checklists, counts, or saved results. This step does not select a model, change data splits, or train.

## Result

| Category | Pairs / rows | Current action |
|---|---:|---|
| DAPO decision degradation | 3 / 3 | 3930, 4653, 9117; retain only hire-label correction candidates |
| Anchor problem | 9 / 13 | Queue for anchor review; do not correct toward the old teacher direction |
| Insufficient information or reviewer disagreement | 28 / 59 | Queue for evidence review; assign no hard binary target |

Across the three target rows, SFT produced hire in 48/48 attribute-free samples and 46/48 attributed samples; DAPO produced hire in 0/48 samples for each input view. The four P25 rows among the original seven candidates, 2405, 4411, 4556, and 6152, were excluded. Final-decision disagreements on P06, P25, and P39 remain recorded.

Sources are `teacher75_v1_full/summary.json`, `audit_rows_75.json`, and `teacher_correction_candidates.json`; see the [audit report](TEACHER75_STANDARD_AUDIT.md) for complete per-pair evidence. The new entry point recomputes these results from the original evidence and checks them against the saved outputs.

## Reading

The frozen SFT teacher has evidence-supported correction candidates on 3/75 rows in this subset. The other 72 rows do not become hard correction targets. Of these, 13 require anchor review and 59 require evidence for mandatory requirements or resolution of reviewer disagreements. Unknown cannot be changed to reject.

The current plan keeps new RL paused while evidence and anchors are addressed. The 3 candidates may support a later, explicitly confirmed teacher-correction experiment, but cannot establish that large-scale training will improve TPR or bias. These 75 rows are a selected C+D subset and cannot estimate the proportion of A/B versus C+D across all 9,140 rows.

Every new row is marked `training_ready=false`. Candidate `target_reason` comes from this review round; the original SFT rationale is not approved for reuse. Preparing training text later requires an explicit supervision format, item-by-item rationale checks, confirmation of data use and splits, and a parent-model baseline under the same prompt for the new three-way task.

## Caveats

The reviewers are models operating in isolated contexts, not human gold annotators. The validator checks quotation existence, fields, hashes, counts, and the decision function; it cannot establish that natural-language inferences are correct or JD checklists complete.

The new prompts enforce two-stage inputs and a three-way output contract; the v1 decision rules and policy hash remain unchanged. The 40 new prompts are prepared with `request_executed=false`; the existing 80 reviews have not been replaced with runs of the new prompt version. Existing DAPO/SFT samples use the old prompt and are not reported as results under the new task definition.

## Reproducing

The sole preparation configuration for this round is `configs/teacher75_correction_workflow_v1.json`. It points to the completed `teacher75_v1_full` sources and pins 40 pairs / 75 rows, the policy hash, and SHA-256 values for 15 source files. If those files change, old reviews cannot directly authorize targets for the new inputs; sources and reviews must be reconfirmed. Historical files in the source directory remain intact.

```bash
uv run python scripts/prepare_teacher_corrections.py \
  --config configs/teacher75_correction_workflow_v1.json \
  --output /path/to/a/new/teacher75_v1_application
```

Outputs include all 75 rows, the 3 candidate rows, anchor-review and evidence-review queues, per-row eligibility decisions, and 40 unexecuted review requests. `manifest.json` records source, script, and artifact SHA-256 values, plus policy and prompt versions. Only a previously nonexistent output directory is accepted; no model calls or training jobs are submitted.
