# Claude blind review: 337 pairs rejected by the training anchor and accepted by the SFT student

Date: 2026-10-16. Reviewers: 8 Claude Opus 5.5 subagents, with each case assessed once. See `PREREG.md`, written before scoring and placed on Daint at 14:38.

Data: `reports/anchor_agreement/pairs.csv`, from `scripts/anchor_agreement.py` at 08052c8, and
`artifacts/semisynthetic-v2/onpolicy-full/{prompts,rows}.parquet`. SHA256 values for all three files are in `sealed/daint_provenance.txt`.

## Method

1. Target set: the anchor, a 122B teacher sampled once without attributes at temperature 0.7, says reject; the SFT student's majority across 8 attribute-free samples says hire. There are 337 pairs.
2. Reverse set: anchor hire and student reject, 14 pairs. Controls: 20 pairs where both hire and 20 where both reject, sampled with seed 42.
3. Shuffle the 391 cases into 8 batches with `build_batches.py`. Reviewers see only the job description and CV, matching the content of the attribute-free prompts seen by the anchor and student. They see no job title, attributes, decisions, or rationales. The answer key is in `sealed/key.csv`; reviewers were not directed to it.
4. For each case, return hire or reject, a fit score from 1 to 5, and one sentence of reasoning. Scores 1 to 2 mean clear reject, 3 borderline, and 4 to 5 clear hire. `score.py` unblinds and scores; `by_training_set.py` maps cases to training sets.

## Validity

| Check | Result |
|---|---|
| Returned cases | 391 / 391, no missing or duplicate cases |
| Decision/fit-score inconsistencies | 0 |
| Control agreement with anchor/student consensus | 38 / 40 = 95%, 95% CI 87.5% to 100% |

The preregistered threshold was at least 70% control agreement; it passed.

## Results

### Whom the reviewer supports in the 337-pair target set

| | Pairs | Share |
|---|---|---|
| Reviewer hires, agreeing with student | 244 | 72.4%, 95% CI 67.7% to 77.4% |
| Reviewer rejects, agreeing with anchor | 93 | 27.6% |

Breakdown by fit score:

| Reviewer fit score | Pairs | Hire | Reject |
|---|---|---|---|
| 4 to 5, clear hire | 124 | 124 | 0 |
| 3, borderline | 179 | 120 | 59 |
| 1 to 2, clear reject | 34 | 0 | 34 |

Among the 158 clear cases, the reviewer supports the student on 124 and the anchor on 34. More than half of all cases are borderline.

Reviewer support for the student increases with student certainty, but only modestly:

| Hire votes among the student's 8 samples | Pairs | Reviewer hire rate |
|---|---|---|
| 5 votes | 71 | 70.4% |
| 6 to 7 votes | 155 | 70.3% |
| 8 votes | 111 | 76.6% |

The rate is 67.5% on the 40 pairs unseen during SFT training and 73.1% on the 297 seen pairs.

### The 14-pair reverse set

The reviewer hires on 9 pairs and rejects on 5, agreeing with the anchor on 9 and the student on 5. The hire rate is 64.3%, with a 95% CI of 35.7% to 85.7%. This sample is too small for more than the observation that the reviewer tends to hire on disputed pairs, regardless of which side originally hired.

### Which training sets contain these pairs

| Training set | Target pairs used | Training rows | Reviewer-hire rows | Clear hire | Borderline | Clear reject |
|---|---|---|---|---|---|---|
| DPO r1, r2, r2em | 0 | 0 | | | | |
| GRPO grpo-02 | 83 | 97 | 75 | 39 | 54 | 4 |
| RFT rft-01 | 88 | 111 | 78 | 47 | 51 | 13 |

All these GRPO and RFT rows have reject targets. The reviewer would hire on roughly three quarters of them; only 4 and 13 rows, respectively, are clear rejects.

## Individual cases

- **C056, Senior IT Recruiter.** The student hires in 8/8 samples. The reviewer hires with score 5 because more than six years of IT recruiting exceeds the four-year requirement. The anchor rejects for missing business experience in Microsoft project management and contractor accounting.
- **C004, Intern Business Analyst.** The student hires in 5/8 samples. The reviewer rejects with score 2 because the CV omits Atlassian, English proficiency, and the required science/engineering degree. The anchor's rejection rationale is largely the same.
- **C002, Frontend Developer.** The student hires in 7/8 samples. The reviewer hires with score 3 because freelance work and React practice projects meet the technical requirements, although education is unmentioned. The anchor rejects for lacking formal work experience and a university degree.

## Interpretation

- On anchor-reject/student-hire pairs, the independent reviewer mostly agrees with the student. Among clear cases, the ratio is approximately four to one.
- More than half the cases are borderline to the reviewer; a single-sample anchor on these pairs is close to a coin flip.
- The reviewer also tends to hire on the reverse set, and Claude's benchmark hire rate is higher than GPT-4o's. Some apparent student correctness may reflect the reviewer's more lenient prior. Among the 20 reject-consensus controls, however, the reviewer rejects 19, so it does not hire indiscriminately.
- GRPO and RFT each contain roughly a hundred training rows that push toward rejection for candidates whom both the attribute-free student and reviewer would hire. This is consistent with both runs becoming stricter after training. None of the three DPO rounds used these pairs, so their increased strictness has another cause.

## Limitations

- One reviewer model and one assessment per case; no human labels or inter-reviewer agreement measurement.
- The reviewer is Opus 5.5, unlike the Fable 5.1 reviewer used for the 153-pair benchmark blind review.
- The reverse set contains only 14 pairs.
- This only suggests that the training targets may be wrong on these rows. Training without these rows has not been tested.

## Reproducing

```bash
python build_batches.py <work_dir>        # 391 cases, 8 batches, sealed/key.csv
# grade each batches/batch_XX.md blind into grades/batch_XX.csv (item,decision,fit,reason)
python score.py <work_dir>                # graded_cases.csv, results.json
python by_training_set.py <work_dir>      # by_training_set.json
```
