"""Checks for the teacher's attribute bias in its invariant targets and its judging
(scripts/teacher_leakage_checks.py). No GPU, no network."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "teacher_leakage_checks",
    Path(__file__).resolve().parents[1] / "scripts" / "teacher_leakage_checks.py")
TL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TL)


def test_features_count_whole_words_from_each_list():
    f = TL.features(pd.Series(["Strong match, but Python depth is unproven; however, proven "
                               "leadership and availability concerns."]))
    assert f.loc[0, "hedges"] == 2       # but, however
    assert f.loc[0, "intensity"] == 2    # strong, proven ("unproven" is another word)
    assert f.loc[0, "proxies"] == 1      # availability
    assert f.loc[0, "words"] == 13
    assert f.loc[0, "addresses"] == 0
    addressed = TL.features(pd.Series(["Your background meets what we require."]))
    assert addressed.loc[0, "addresses"] == 1


def _kept(pair, group, cond, text):
    return {"pair_id": pair, "protected_group": group, "condition": cond,
            "reference_decision": "hire", "chosen_feedback": text}


def test_deviation_is_read_against_each_pairs_own_mean():
    kept = pd.DataFrame([_kept("p1", "a", "explicit", "one two three four"),
                         _kept("p1", "b", "explicit", "one two"),
                         _kept("p2", "a", "explicit", "one two three four five six"),
                         _kept("p2", "b", "explicit", "one two")])
    table = TL.within_pair_deviation(kept, n_boot=50).set_index("protected_group")
    assert table.loc["a", "dev_words"] == 1.5 and table.loc["b", "dev_words"] == -1.5
    assert table.loc["a", "words"] == 5.0 and table.loc["a", "pairs"] == 2


def _r(decision, feedback):
    return json.dumps({"decision": decision, "feedback": feedback})


def test_invariant_filter_reports_yield_and_reasons_per_cell():
    base = {"pair_id": "p", "protected_group": "religion", "protected_attr": "Jain",
            "reference_decision": "hire", "lang": "en"}
    raw = pd.DataFrame([
        {**base, "condition": "implicit",
         "raw_output": _r("hire", "Five years of Django and PostgreSQL meet the requirements.")},
        {**base, "condition": "implicit",
         "raw_output": _r("reject", "Lacks the PostgreSQL depth the role needs here.")},
        {**base, "condition": "explicit",
         "raw_output": _r("hire", "As a Jain candidate with Django skills, a solid match.")},
    ])
    kept, table = TL.invariant_filter_by_cell(raw)
    table = table.set_index("condition")
    assert len(kept) == 1
    assert table.loc["implicit", "yield"] == 0.5
    assert table.loc["implicit", "drop_invariant_decision_drift"] == 1
    assert table.loc["explicit", "drop_invariant_attribute_leak"] == 1


def _judge(alludes=False, grounded=True, supports=True, plausibility=4):
    return json.dumps({"mentions_or_alludes": alludes, "grounded": grounded,
                       "supports_decision": supports, "plausibility": plausibility,
                       "note": "ok"})


def test_judge_pass_rates_and_reasons_per_cell():
    frame = pd.DataFrame([{"pair_id": f"p{i}", "protected_group": "religion",
                           "condition": "implicit", "reference_decision": "hire",
                           "protected_attr": "Jain", "rejected_strategy": "student_tipped"}
                          for i in range(4)])
    rejected = [_judge(), _judge(), _judge(grounded=False), _judge(plausibility=2)]
    chosen = [_judge(), _judge(alludes=True), _judge(), _judge()]
    judged = TL.N.attach_judgements(frame, rejected, "rejected")
    judged = TL.N.attach_judgements(judged, chosen, "chosen")
    row = TL.judge_by_cell(judged).iloc[0]
    assert row["rejected_pass"] == 0.5 and row["chosen_pass"] == 0.75
    assert row["both_pass"] == 0.25     # only p0 passes both sides
    assert row["rejected_judge_ungrounded"] == 1 and row["rejected_judge_implausible"] == 1
    assert row["chosen_judge_alludes"] == 1


def test_the_cell_classifier_finds_a_cell_marker_and_the_null_does_not():
    rows = []
    for i in range(10):
        rows.append(_kept(f"p{i}", "a", "explicit", f"shared words alpha marker {i}"))
        rows.append(_kept(f"p{i}", "b", "explicit", f"shared words beta marker {i}"))
    table = TL.cell_classifier(pd.DataFrame(rows), n_perm=2, folds=5).set_index("cell")
    assert table.loc["a / explicit", "auc"] > 0.9
    assert table.loc["a / explicit", "null_mean"] < table.loc["a / explicit", "auc"]


def test_wilson_interval_and_homogeneity_test():
    lo, hi = TL.wilson([50], [100])
    assert round(float(lo[0]), 3) == 0.404 and round(float(hi[0]), 3) == 0.596
    table = pd.DataFrame({"round": "r", "reference_decision": "hire", "n": [100, 100],
                          "rejected_pass": [0.5, 0.9], "chosen_pass": [0.9, 0.9],
                          "both_pass": [0.45, 0.81]})
    h = TL.judge_homogeneity(table).set_index("side")
    assert h.loc["rejected", "p"] < 0.001 and h.loc["chosen", "p"] > 0.99


def test_group_words_and_top_features():
    f = TL.features(pd.Series(["Ready for a civilian role.", "Ready for a new role."]),
                    pd.Series(["military_status", "religion"]))
    assert f["group_words"].tolist() == [1, 0]
    rows = []
    for i in range(10):
        rows.append(_kept(f"p{i}", "a", "explicit", f"shared words alpha marker {i}"))
        rows.append(_kept(f"p{i}", "b", "explicit", f"shared words beta marker {i}"))
    top = TL.cell_top_features(pd.DataFrame(rows), k=1).set_index("cell")
    assert "alpha" in top.loc["a / explicit", "ngram"]   # "alpha" or a bigram around it
    assert top.loc["a / explicit", "rationales_with_it"] == 10
    assert "beta" in top.loc["b / explicit", "ngram"]
