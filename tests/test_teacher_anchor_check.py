"""The anchor-procedure check on the benchmark (scripts/teacher_anchor_check.py). No GPU."""

import importlib.util
import json
from pathlib import Path

import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "teacher_anchor_check",
    Path(__file__).resolve().parents[1] / "scripts" / "teacher_anchor_check.py")
TA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TA)


def _r(decision: str) -> str:
    return json.dumps({"decision": decision, "feedback": "Matches the requirements."})


def _frame() -> pd.DataFrame:
    # a: GPT-4o rejects, the teacher always rejects. b: GPT-4o hires, the single draw rejects,
    # the majority hires. c: GPT-4o hires, the draws split 2-2, greedy rejects.
    pairs = pd.DataFrame({"pair_id": ["a", "b", "c"], "candidate_id": ["ca", "cb", "cc"],
                          "job_id": ["j", "j", "j"], "job_position": ["Engineer"] * 3,
                          "reference_decision": ["reject", "hire", "hire"]})
    single = [_r("reject"), _r("reject"), _r("hire")]
    sampled = [[_r("reject")] * 4,
               [_r("hire"), _r("hire"), _r("reject"), _r("hire")],
               [_r("hire"), _r("reject"), _r("hire"), _r("reject")]]
    greedy = [_r("reject"), _r("hire"), _r("reject")]
    return TA.parse(TA.to_rows(pairs, single, sampled, greedy), "en")


def test_each_mode_is_scored_against_the_gpt4o_reference():
    s = TA.summarize(_frame())
    assert s["pairs"] == 3 and s["samples_per_pair"] == 4
    assert s["single"]["agreement"] == 2 / 3 and s["single"]["reject_share"] == 2 / 3
    assert s["single"]["agreement_on_reference_hire"] == 0.5
    assert s["single"]["agreement_on_reference_reject"] == 1.0
    assert s["greedy"]["agreement"] == 2 / 3
    assert s["sampled_per_draw"]["agreement"] == 9 / 12
    assert s["sampled_per_draw"]["reject_share"] == 7 / 12


def test_majority_leaves_ties_out_and_the_vote_split_counts_every_pair():
    s = TA.summarize(_frame())
    assert s["majority"]["ties"] == 1
    assert s["majority"]["n_decided"] == 2 and s["majority"]["agreement"] == 1.0
    assert s["vote_split"] == {"0 of 4 hire": 1, "2 of 4 hire": 1, "3 of 4 hire": 1}
    assert s["unanimous_share"] == 1 / 3
    assert s["single_matches_majority"] == 0.5  # a matches, b does not, c is a tie


def test_the_report_renders_every_mode():
    frame = _frame()
    meta = {"prompt_sha256": "0" * 64, "decoding": TA.V1_GENERATION}
    text = TA.render("teacher--en", TA.summarize(frame), meta)
    for label in ("single sample", "sampled draws", "majority of 4", "greedy"):
        assert label in text
