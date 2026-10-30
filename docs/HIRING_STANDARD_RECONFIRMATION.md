# Reconfirming correction targets under the unified standard

2026-10-22, policy `hiring-evidence-v1`. This round re-examined only the 4 pairs underlying the original 7 correction-candidate rows. Result: **3 pairs and 3 rows retained; 1 pair and 4 rows moved to review**. All had original hire anchors and became newly all-wrong rows under DAPO.

Mandatory requirements and preferences were frozen from the JD alone before two model reviewers independently assessed the cases in isolated contexts. They could not see the original anchors, SFT/DAPO outputs, or previous audit conclusions. Their final decisions agreed on all four cases. The two model reviews may still share judgment biases and are not human gold labels.

| Pair | Row IDs | Original anchor votes | Mandatory items H | Direct support D, reviewers A/B | Inferred support I, A/B | Unknown U | Contradicted F | v1 conclusion |
|---|---|---|---:|---|---|---:|---:|---|
| P02 | 3930 | hire 8/8 | 9 | 6/6 | 3/3 | 0 | 0 | hire, retained |
| P04 | 9117 | hire 7/8 | 19 | 13/13 | 6/6 | 0 | 0 | hire, retained |
| P25 | 2405, 4411, 4556, 6152 | hire 7/8 | 5 | 2/1 | 2/3 | 1 | 0 | review, removed from hard correction targets |
| P31 | 4653 | hire 8/8 | 0 | 0/0 | 0/0 | 0 | 0 | hire, positive evidence of job fit |

Mandatory-requirement support is 100% for P02 and P04, but 80% for P25, which cannot receive hire. P31 has no explicitly mandatory qualifications, so its support rate is null; its hire decision rests on clear software-development work matching the main duties. H is a checklist-item count, not a difficulty measure or score comparable across jobs.

## Why P25 is no longer changed directly to hire

The JD states the required skill as follows:

> Strong practical understanding of Application Architecture techniques for designing scale- universal systems.

The relevant CV evidence is:

> Comfortable with handling Application Architecture or decentralized strategy definitions and implementation for tailored decentralized applications.

Both reviewers found `scale- universal systems` undefined. It cannot be silently rewritten as ordinary application architecture, nor can decentralized-systems experience be assumed to cover the qualifier. The item is therefore unknown.

Under the unified rule, 2018 to 2021 still gives a nominal 3 years and meets an "at least 3 years" threshold at year-level precision. A DAPO rejection claiming this experience is "less than 3 years" or "not recent enough" remains problematic. Invalidating that rationale does not fill the other unknown requirement, so these 4 rows are not currently assigned to the frozen teacher for forced correction. Clarify the JD first, apply the same interpretation to all candidates for that JD, and then re-review.

## The three retained targets and their limits

- P02: 2018 to 2023 gives a nominal 5 years, equal to the 5-year minimum; professional evidence supports the other explicit requirements. Do not change the minimum to "more than 5 years." Exact months remain unavailable.
- P04: PR manager from 2018 to 2020 and PR specialist from 2020 to 2022 each give a nominal 2 years. The JD does not require specialist experience to precede manager experience. Planning, communication, relationship maintenance, working under pressure, on-time delivery, and implementing strategic communication plans are the 6 items supported by limited functional inference from matching skills and relevant roles. Both evidence tables retain the item-level basis. The conclusion depends on v1 explicitly allowing such inference; not every item is supported by a verbatim self-report.
- P31: The JD describes a project without an explicit prior-experience threshold. Actual development experience in .NET, SQL, Azure, microservices, testing, and Scrum supports the main job match. Specific medical equipment, SQL Server versions, Azure SQL, and Azure DevOps remain unconfirmed background details and cannot automatically become rejection thresholds.

Across these 3 rows, SFT-v3 produced hire in 48/48 attribute-free samples and 46/48 attributed samples, with per-row attributed counts of 15/16, 15/16, and 16/16. DAPO produced hire in 0/48 samples for each view. These counts come from the previous probe; the new standard did not trigger additional model sampling.

This only establishes a frozen-SFT signal aligned with the confirmed direction. It does not predict that training will correct all 3 rows, and cannot be extrapolated to all 1,081 C+D rows. Do not copy teacher rationales directly.

## Denominators and validation

Across the 4 pairs, hire/reject/review counts are 3/0/1, with 75% resolved coverage. Across the 7 rows, they are 3/0/4, with 42.86% coverage. Review is neither counted as an error or refusal nor hidden from the denominator.

All 8 evidence tables passed checks of policy hashes, frozen checklists, 225 quotation references, counts, and the decision function. The 225 checks include quotations repeated across items and reviewers; they are not 225 independent pieces of evidence. The validator passed 18 boundary tests; semantic judgments remain the reviewers' responsibility.

The old 40-pair / 75-row audit was not overwritten. The remaining 36 pairs have not yet been reconfirmed under v1, so this round cannot be described as a new-standard audit of all 75 rows.

Keep new RL training paused. Retain these 3 rows as evidence-supported correction candidates and hold the 4 P25 rows pending JD clarification. v1 is now included in the new audit prompts; the original training and benchmark prompts, data, and metric definitions retain their existing versions.

Full rules: `standard.md`; machine-readable rules: `policy.json`; item-level evidence: `review_a.json` and `review_b.json`; candidates: `correction_candidates_3.json`. The policy hash is `37dda6072e102f04274bb94fdce75bdd6063de0792c50e3ef6907202fc70ff53`, using canonical JSON SHA-256.
