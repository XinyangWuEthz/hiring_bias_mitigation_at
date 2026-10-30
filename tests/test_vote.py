"""The vote anchor: the rule of docs/SFT_V3_ANCHORS.md, for pools with and without a v1 anchor."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_mitigation.generation import vote as V


def _out(decision: str, words: int = 8) -> str:
    return json.dumps({"decision": decision, "feedback": " ".join(["solid"] * words)})


def _draws(per_pair: dict[str, list[str]]) -> pd.DataFrame:
    return V.draw_frame(list(per_pair), list(per_pair.values()), "en")


def test_majority_of_decided_draws_and_first_usable_feedback():
    draws = _draws({
        "a": [_out("reject", 2), _out("hire")] + [_out("hire")] * 4 + [_out("reject")] * 2,
        "b": [_out("reject")] * 8,
        "c": ["not json"] * 3 + [_out("hire")] * 3 + [_out("reject")] * 2,
    })
    a = V.vote_anchors(draws, ["a", "b", "c"], "en").set_index("pair_id")
    assert a.loc["a", "reference_decision"] == "hire"
    assert a.loc["a", "hire_votes"] == 5 and a.loc["a", "decided"] == 8
    assert a.loc["a", "vote_sample_index"] == 1          # the first majority draw
    assert a.loc["b", "reference_decision"] == "reject" and a.loc["b", "majority_share"] == 1.0
    # Undecided draws do not count: 3 of 5 decided is a majority, and at most 5/8 is unstable.
    assert a.loc["c", "decided"] == 5 and a.loc["c", "reference_decision"] == "hire"
    assert bool(a.loc["c", "unstable"]) and not bool(a.loc["b", "unstable"])
    assert bool(a.loc["a", "unstable"])                  # 5 of 8


def test_without_a_fallback_ties_and_unusable_majorities_get_no_anchor():
    draws = _draws({
        "tie": [_out("hire")] * 4 + [_out("reject")] * 4,
        "none": ["garbage"] * 8,
        "short": [_out("hire", 1)] * 8,                   # majority, but no draw passes the filter
    })
    a = V.vote_anchors(draws, ["tie", "none", "short"], "en").set_index("pair_id")
    assert a["anchor_source"].to_dict() == {"tie": "tie", "none": "no_decided_draw",
                                     "short": "no_usable_feedback"}
    assert not a["anchored"].any()
    assert a["reference_decision"].isna().all()


def test_with_a_fallback_the_v1_rule_is_reproduced():
    draws = _draws({"tie": [_out("hire")] * 4 + [_out("reject")] * 4,
                    "flip": [_out("hire")] * 7 + [_out("reject")]})
    fallback = pd.DataFrame({"pair_id": ["tie", "flip"],
                             "fallback_decision": ["reject", "reject"],
                             "fallback_feedback": ["v1 tie", "v1 flip"]})
    a = V.vote_anchors(draws, ["tie", "flip"], "en", fallback=fallback).set_index("pair_id")
    assert a.loc["tie", "reference_decision"] == "reject"
    assert a.loc["tie", "reference_feedback"] == "v1 tie"
    assert a.loc["flip", "reference_decision"] == "hire"
    assert bool(a.loc["flip", "unstable"])               # overturned v1
    assert a["anchored"].all()


SAVED = (Path(os.environ.get("HBM_OUTPUT_ROOT", "/nonexistent"))
         / "artifacts/semisynthetic-v3/anchors")


@pytest.mark.skipif(not (SAVED / "pool_votes_seed43_anchors.parquet").exists(),
                    reason="job 5073453's vote is not on this machine")
def test_port_reproduces_the_saved_vote_anchors():
    """The ported rule gives job 5073453's anchors exactly (sft_v3_anchors.new_anchors)."""
    draws = pd.read_parquet(SAVED / "pool_votes_seed43.parquet")
    saved = pd.read_parquet(SAVED / "pool_votes_seed43_anchors.parquet")
    fallback = saved[["pair_id", "v1_decision", "v1_feedback"]].rename(
        columns={"v1_decision": "fallback_decision", "v1_feedback": "fallback_feedback"})
    mine = V.vote_anchors(draws, saved["pair_id"], "en", fallback=fallback)
    m = mine.merge(saved, on="pair_id", suffixes=("", "_saved"), validate="one_to_one")
    assert (m["reference_decision"] == m["new_decision"]).all()
    assert (m["reference_feedback"] == m["new_feedback"]).all()
    assert (m["anchor_source"] == m["source"]).all()
    assert (m["unstable"] == m["unstable_saved"]).all()
