 prepare data: prepare pairs,attribute_free prompt + rows(different attribtes included),attributed prompt
 `mentions_attribute` has limitations: English has little inflection, so prefix truncation adds almost no recall and mainly causes false positives. Fixed by switching to whole-word vocabulary matching. Paraphrases such as "your faith" and "served in the army" still escape lexical matching; the judge stage handles semantic implications.
 v1 anchors were overly strict because all prompts shared seed 42. Switching to majority voting across 8 samples produced the SFT-v3 data. v3 also added new Djinni candidates, RJDB train, and Kaggle data. The full candidate pool contains 186,970 pairs; this round generated only 9,034 of them. See sft_v3_swap and hiring_bias_mitigation_at-v3additional/docs/V3_NEW_DATA.md.
 New DPO data source: train SFT-v3 → have SFT-v3 answer attribute variants → identify actual attribute-tipped errors → teacher verification → construct preference pairs, plus the old r2em data.
 New evaluation tables: reports/reference_metrics.csv is the summary with verdicts; reports/comparisons.csv is the long-form pairwise comparison table.
 GPU usage
 DPO>SFT
 GRPO~DPO
 DAPO<SFT