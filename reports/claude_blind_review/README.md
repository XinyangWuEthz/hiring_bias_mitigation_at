# Claude blind review: 113 model/4o disagreements plus 40 agreement controls

Date: 2026-10-15. Reviewer: Claude, Fable 5.1; one reviewer, one assessment per case.
Data: `attr_free` rows from `outputs/raw/*.parquet`, covering 450 pairs, with `reference_decision` from GPT-4o as the reference.
Per-case results: `claude_grades_153.csv`, containing my decision, 1–5 fit score, one-sentence rationale, and the decisions and feedback from 4o and the four models.

## Method

1. Take the 113 pairs where at least one of four models, SFT, r1-300, r2-102, or r2em-100, disagrees with 4o. Add 40 controls where all five agree, comprising 20 unanimous hires and 20 unanimous rejects, for 153 pairs total. Shuffle the order.
2. The blind-review form contains only job title, job description, and CV, with no decisions or feedback. Judge each case using the same "hire or reject" task framing as the baseline prompt and assign a 1–5 fit score: 1–2 clear reject, 3 borderline, 4–5 clear hire.
3. Unblind against the key only after writing scores to a file.

## Results

### Controls: 40 pairs with five-way agreement

| Metric | Value |
|---|---|
| My agreement with consensus | 33/40 = 82.5%, kappa 0.65 |
| The 7 disagreements | All borderline cases that I scored 3 |
| Hire rate | Mine 52.5%, consensus 50% |

Interpretation: I mostly agree where all five agree. Cases where the consensus is collectively wrong according to my review occur only among borderline cases.

### Disagreement set: 113 pairs

| Model/reviewer | Agreement with me | kappa | Hire rate in this set |
|---|---|---|---|
| 4o | 74/113 = 65.5% | 0.11 | 73.5% |
| SFT | 71/113 = 62.8% | 0.22 | 54.9% |
| r2em-100 | 68/113 = 60.2% | 0.14 | 57.5% |
| r2-102 | 44/113 = 38.9% | 0.07 | 18.6% |
| r1-300 | 39/113 = 34.5% | 0.03 | 14.2% |
| Me | — | — | 74.3% |

My hire rate is nearly the same as 4o's, 74% versus 73%; r1 / r2 hire on only 14–19% of this set.

### Direction of disagreement: whom I support

| Model | Model reject, 4o hire | I support model: reject | I support 4o: hire | Model hire, 4o reject | I support model: hire | I support 4o: reject |
|---|---|---|---|---|---|---|
| SFT | 48 | 17 | 31 | 27 | 19 | 8 |
| r1-300 | 79 | 19 | 60 | 12 | 9 | 3 |
| r2-102 | 76 | 19 | 57 | 14 | 11 | 3 |
| r2em-100 | 48 | 16 | 32 | 30 | 20 | 10 |

Among the 55 pairs where all four models disagree with 4o: 44 have 4o hire/model reject, on which I hire 28 and reject 16; 11 have 4o reject/model hire, on which I hire 8 and reject 3.

Interpretation: when the model rejects and 4o hires, I support 4o by about 2:1. When the model hires and 4o rejects, I support the model by about 2–3:1. Thus 4o tends to be strict when rejecting and lenient when hiring, while I judge most of r1 / r2's many additional rejections to be false rejections.

### Borderline versus clear cases

Disagreement cases broken down by my fit score:

| My score | Pairs | I support 4o | I support model |
|---|---|---|---|
| 2, clear reject | 6 | 4 | 2 |
| 3, borderline | 56 | 26 | 30 |
| 4–5, clear hire | 51 | 44 | 7 |

Half the disagreements, 56/113, are borderline; deciding which side is correct is close to a coin flip. In the other half, the clear cases, 4o is correct 48 times and the model 9 times according to this review.

Among model-reject/4o-hire pairs, the shares I score 4–5, clear hire, are SFT 20/48 = 42%, r1-300 42/79 = 53%, r2-102 38/76 = 50%, and r2em-100 21/48 = 44%.

### Agreement across 450 pairs with me as the reference

The 337 five-way-agreement pairs were not all reviewed; extrapolate the control agreement of 82.5% to them. Use observed results for the 113 disagreement pairs.

| Model | Agreement with 4o, existing metric | Agreement with me, extrapolated |
|---|---|---|
| 4o | — | 78.2% |
| SFT | 83.3% | 77.6% |
| r2em-100 | 82.7% | 76.9% |
| r2-102 | 80.0% | 71.6% |
| r1-300 | 79.8% | 70.4% |

The ranking does not change. The SFT versus r1 / r2 gap widens from 3.5 pp to 6–7 pp: most candidates whom r1 / r2 reject in addition to SFT are candidates I would also hire.

## Factual errors in rationales: individual review of 45 pairs with four-model disagreement

- Model reject, 4o hire, and my clear hire, 18 pairs: 6 r1-300 rationales contain checkable factual errors, such as understating experience or denying skills explicitly listed in the CV. For example, I057 claims less than three years where the CV gives four, and I127 claims no IT-industry background where the CV lists three IT companies. Another 2 treat preferred qualifications as mandatory, Mandarin in I075 and FinTech in I147. The remaining roughly 10 are strict but evidence-based rejections.
- Model reject, 4o hire, and my reject, 16 pairs: 4 of 4o's rationales overstate the CV, claiming CMS experience in I040 or two years of automation experience in I135, neither of which the CV states.
- Model hire, 4o reject, 11 pairs: 4 r2em-100 rationales invent qualifications. I059 claims "5+ years of iOS" where the CV gives 2.8 years; I089 claims Swift and Objective-C where the CV does not mention Objective-C.

This is a separate quality dimension that decision agreement misses: DPO models in both directions invent facts to support their decisions.

## Implications for the existing conclusions

- The conclusion that utility declined in the first two rounds holds, and is stronger under my review: a 6–7 pp gap rather than 3.5 pp.
- The conclusion that r2em recovered utility holds: r2em stays within 1 pp of SFT under both references.
- Agreement has a ceiling: my agreement with 4o over 450 pairs is approximately 78%, while SFT agrees with 4o on 83%. With only about 80% agreement between two independent reviewers, agreement cannot and should not be pushed to 90%; it is suitable as a guardrail, not a score.
- About half the disagreements are borderline. Human blind review should use a 1–5 scale rather than a binary choice, or coin flips will be counted as errors.
- Human blind review should separately label whether each rationale matches the CV.

## Limitations

- There is only one reviewer, so there is no inter-reviewer kappa. I am also an LLM, and my prior may be closer to 4o's than a human's would be. Blinding removes anchoring, not shared priors.
- Extrapolation to 450 pairs assumes the 337 agreement cases behave like the 40 controls.
- Before constructing the blind-review form, I saw one 4o decision while inspecting the benchmark CSV structure, for the Data Scientist/AdTech pair. That pair was not among the final 153 cases.
