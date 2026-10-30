# Unified hiring-text audit standard v1

Version `hiring-evidence-v1`, 2026-10-22. Applies to this English JD/CV text audit, confirmation of correction targets, and future anchor annotation that explicitly adopts this version.

`hire` means that the available professional evidence supports the job match and every explicitly mandatory requirement has supporting evidence. This retains the meaning used in this text-based qualification audit. It is not a final job offer, and missing mandatory requirements cannot be deferred to an interview.

The single source of rules is `policy.json`, stored in the repository at `configs/hiring_decision_standard_v1.json`. The English audit prompt is generated from it. Rule changes require a version update, prompt regeneration, and revalidation of the affected checklists and decisions.

## Quantifiable decision criteria

Freeze the checklist from the JD alone before reading the CV. H is the total number of mandatory requirements; D counts direct support, I counts support through permitted functional inference, U counts unknowns, and F counts explicit contradictions. Require `H = D + I + U + F`.

| Decision | Condition, evaluated in this order |
|---|---|
| review | A material conflict exists in decisive evidence or mandatory JD requirements |
| reject | No such conflict exists and F ≥ 1. Cite explicit evidence that a requirement is not met |
| hire | U = 0 and F = 0; mandatory-requirement support is 100% when H > 0. At least one genuine CV quotation must also support the main work of the role |
| review | All remaining cases, including unknown mandatory requirements or insufficient evidence for the main job match |

When H = 0, record the support rate as `null`, not 100%. Positive evidence of job fit is still required. Preferred qualifications do not enter the mandatory-requirement denominator and cannot offset a failed mandatory requirement. There is no aggregate score that allows requirements to compensate for one another.

### Mandatory requirements, preferences, and background

- Treat `must`, `required`, `essential`, `minimum`, eligibility conditions, and conditions under Requirements/Required Skills/Required Experience as mandatory.
- Local qualifiers in the same sentence, such as `plus`, `preferred`, and `desirable`, take precedence over the section heading. These are preferences; their absence cannot justify rejection.
- Project background, client information, future responsibilities, and the technology stack do not by themselves require prior experience with every item. Impose such a requirement only when the JD explicitly states a prior-experience threshold.
- Split independently verifiable AND conditions; retain alternatives in OR conditions; deduplicate repeated requirements. Do not truncate to a maximum of six items. Use the same checklist for every candidate and attribute variant of the same JD.

### Evidence and missing information

Record each item as `direct`, `entailed`, `unknown`, or `contradicted`, retaining verbatim JD/CV quotations. CV self-reports count as textual evidence; this audit does not replace external verification.

Limited functional inference is allowed. For example, explicit experience leading a team can support leadership ability. Relevant roles combined with matching activities or skills can support the usual functions of that role. A job title alone does not establish specialized competence; mark an inference unknown if it requires inventing facts.

Do not infer language proficiency from an English CV, a degree from a manager title, years of experience from an undated project, explicitly required Azure DevOps from generic Azure experience, or linkbuilding from generic SEO experience. An unmentioned ability is unknown, not evidence of inability.

`review` is a valid audit outcome. Do not force it into reject or treat it as a formatting error or refusal. Every table must report hire/reject/review and resolved coverage. Compute binary agreement only on resolved cases and disclose the denominator.

### Years of experience

| Source information | Consistent calculation |
|---|---|
| Years only | Nominal duration = end year − start year; record `date_precision=year` and `nominal=true`. Do not count both endpoint years inclusively |
| Months available | Compare the actual month difference with `12 × required years` |
| Exact dates available | Compare with the corresponding anniversary date |
| Explicit statement of X years | Accept it if it concerns the required work and does not conflict with explicit dates |
| Present/current | Use the CV as-of date or a documented source snapshot date; without one, do not calculate duration or substitute the audit date |
| Overlapping experience | Use the union of intervals when accumulating duration for the same qualification; assess separately required functions separately, without inventing a required sequence |

Use `>=` for minimum duration. For example, 2018 to 2023 gives a nominal 5 years at the source data's year-level precision and can meet a textual requirement of "at least 5 years." It does not establish exactly 60 completed months. Retain this limitation when precise external verification is needed, and use the same calculation across cases. Do not invent requirements for recent experience, exceeding the minimum, a maximum duration, or being overqualified.

Store the nominal duration and `nominal=true` in `data_quality_flags`; retain `date_precision` on each corresponding mandatory requirement. If an explicit duration statement conflicts with dates for the same experience, record a material conflict and return review.

## Prompt integration

Generate the complete rules, numeric thresholds, output fields, and policy hash from the same configuration. New reviews use the following two entry points, each recording its prompt_id and input-prompt SHA-256:

| Stage | prompt_id | Inputs and constraints |
|---|---|---|
| Extract a checklist from the JD alone | `hiring-evidence-v1-jd-only-1` | Accepts only the JD and case ID, with no input for a CV, anchor, or model result |
| Review the CV against a frozen checklist | `hiring-evidence-v1-frozen-review-1` | Validate the checklist policy hash, fields, and JD quotations before binding the JD, checklist, and CV; stop if the checklist is absent or inconsistent |

Complete the checklist before seeing the CV. If the second stage identifies a material extraction problem, return review with an explanation and send the checklist back to the JD stage. Do not revise requirements ad hoc to fit the current candidate. `build_review_prompt` binds inputs through JSON and does not recursively substitute placeholders in source text.

The two new templates are `docs/prompts/hiring_evidence_v1_jd.txt` and `docs/prompts/hiring_evidence_v1_review.txt`. Retain the historical `hiring_evidence_v1.txt` template to reproduce earlier reviews. The new templates implement the same v1 decision rules without relaxing or raising the hire threshold. The existing 80 reviews remain records of the original prompt; do not claim they were rerun with the new templates.

Apply this standard to text review first. The existing benchmark, SFT/GRPO/DAPO training prompts, parsers, and metric protocols have not switched. The old workflow only supports hire/reject and also parses interview, shortlist, and conditional hire as hire. A direct replacement would change the task definition and valid-output denominator; historical numbers could no longer be treated as model differences within the same experiment.

The repository's existing `structured_rubric` experimental strategy does have quantitative thresholds: at most six items, scored 0/1/2 each, with hire at 60% of the total score provided no essential item scores 0. It is not the default standard shared by every workflow. v1 uses the complete mandatory-requirement coverage rule above and retains the earlier strategy as a historical experiment.

Future training under the new standard must explicitly handle review, establish a parent-model baseline under the same task definition, and preserve separation between training and the benchmark. Do not interpret this diagnostic exercise directly as a training gain.

## Confirming correction targets

1. Freeze the checklist from the JD alone, without seeing the old anchor or model answers.
2. Two independent reviewers use the same policy and checklist to assess the attribute-free CV; both must provide evidence tables.
3. The validator checks the policy hash, missing mandatory requirements, verbatim quotations, counts, and final decision.
4. Only matching hire/reject decisions from both reviewers become resolved labels. Any review outcome or disagreement excludes the case from hard correction targets for now.
5. Reveal the model results afterward. A label that agrees with SFT and opposes DAPO is an evidence-supported candidate for correcting DAPO.

This procedure confirms a correction direction; it does not guarantee that the frozen SFT teacher can achieve the correction. Check SFT rationales separately rather than copying them because the decision is correct.

Correction data are currently prepared with `configs/teacher75_correction_workflow_v1.json`. The configuration pins the SHA-256 of 15 original audit source files, including JD/CV text, checklists, ledgers, sampling summaries, and the old review prompt. The entry point first compares fingerprints, then reads 40 frozen checklists and 80 review records, recomputes all 75 rows, and checks the saved categories and candidates. Even one added source sentence, added or removed reviews, an invalid quotation or hash, or a stale candidate stops export. Neither the original 7-row nor the 75-row list can replace this validation.

The current output marks 3 rows as `label_only_candidate`, 13 as `excluded_anchor_problem`, and 59 as `excluded_pending_review`. Every row has `training_ready=false`; the original teacher rationales retain `teacher_rationale_approved_for_reuse=false`. These fields distinguish an evidence-supported label direction from a training-ready example. See the [correction plan](TEACHER75_CORRECTION_PLAN.md).

Both reviewers are models, not human gold annotators. The validator checks quotations and arithmetic; it cannot automatically establish the validity of functional inferences, the relevance of experience duration, OR-clause judgments, or completeness of JD extraction.

## Archiving the existing audits

| Artifact | Scope | Purpose |
|---|---|---|
| Original research benchmark | Historical binary prompt and established metrics | Compare model bias, stability, TPR, and other measures |
| 16-sample paired probe | Two models, attributed/attribute-free inputs, 9,140 rows | Distinguish persistent anchor disagreement from attribute effects; not independent ground truth |
| First text spot-check | 18 pairs | Explore typical errors and evidence gaps |
| Second full-text blind review | 75 rows deduplicated into 40 pairs, two reviews each, 80 total | Establish the earlier degradation/anchor-problem/insufficient-information categories |
| Targeted v1 reconfirmation | 4 pairs underlying the original 7 correction-candidate rows, two new reviews each | Reconfirm candidates under the frozen quantitative standard |
| Full v1 re-audit | All 40 pairs and 75 rows, with 80 new reviews | Update all categories and frozen-SFT-teacher correction targets |

There have now been four rounds of text review; the last two use the same v1 standard. The latest full re-audit covers all 40 pairs and 75 rows, with two new reviews per pair. Earlier reviews remain as historical records. Samples overlap and must not be added together as independent observations. See the [full re-audit report](TEACHER75_STANDARD_AUDIT.md) for current categories and correction targets. Entry-point validation does not add another model-review round.

## Repository usage

```bash
uv run python scripts/hiring_standard.py \
  --policy configs/hiring_decision_standard_v1.json \
  --render-job-prompt docs/prompts/hiring_evidence_v1_jd.txt \
  --render-review-prompt docs/prompts/hiring_evidence_v1_review.txt
uv run python scripts/prepare_teacher_corrections.py \
  --config configs/teacher75_correction_workflow_v1.json \
  --output /path/to/a/new/teacher75_v1_application
uv run python -m pytest tests/test_hiring_standard.py tests/test_hiring_workflow.py \
  tests/test_teacher75_standard_analysis.py tests/test_prepare_teacher_corrections.py -q
```

`policy_sha256` is the SHA-256 of canonical JSON: UTF-8, sorted keys, compact separators, preserved Unicode, and no NaN/Infinity. It is not a file-byte hash. New callers bind inputs with `build_job_rubric_prompt` and `build_review_prompt`. Export requires a new directory, preserves historical sources, and neither calls a model nor starts training.

Source references: baseline/structured_rubric/second_pass in `src/hiring_bias_mitigation/data/prompts.py`, REFERENCE_PROMPT in `generation/synth.py`, binary parsing in `eval/parsing.py`, `docs/EVAL_STANDARD.md`, and `docs/METRICS.md`. The existing monitor commit is `8d7063f`; the paired-probe commit is `0f8c856`.
