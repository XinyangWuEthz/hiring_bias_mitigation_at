"""scripts/monitor_calibration.py: the margin monitor's bin shares, and its checks against the
benchmark."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("monitor_calibration",
                                               ROOT / "scripts" / "monitor_calibration.py")
MC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MC)

EDGES = (-1.0, 1.0)


def _parent(margins, hire) -> pd.DataFrame:
    n = len(margins)
    return pd.DataFrame({"pair_id": [f"p{i}" for i in range(n)], "protected_group": "gender",
                         "protected_attr": "a", "condition": "explicit",
                         "margin": margins, "gpt4o_hire": hire})


def test_bench_shares_split_each_reference_side_over_the_parents_bins():
    margins = [-3, -2, 0.5, 2, 3, -3, -3, -0.5]
    hire = [False, False, True, True, True, False, False, True]
    s = MC.bench_shares(margins, hire, EDGES)
    assert s["share_ref_hire"] == [0.0, 0.5, 0.5] and s["share_ref_reject"] == [1.0, 0.0, 0.0]
    assert s["reference_hire_share"] == 0.5
    assert s["parent_tpr"] == pytest.approx(0.75) and s["parent_fpr"] == 0.0


def test_validate_recovers_the_tpr_change_when_bins_move_alike_whatever_gpt4o_said():
    margins = np.array([-3.0, -3.0, 0.5, 0.5, 0.5, 0.5, 3.0, 3.0])
    hire = np.array([False, True, True, True, False, False, True, True])
    parent = _parent(margins, hire)
    # borderline bin: one hire of each label lost; confident hires: one of two lost
    flipped = np.isin(np.arange(8), [2, 4, 6])
    arm = parent[MC.KEY].assign(margin=np.where(flipped, -margins, margins))
    shares = MC.bench_shares(margins, hire, EDGES)
    r = MC.validate(parent, arm, shares)
    assert r["tpr_actual_pp"] == pytest.approx(-40.0)
    assert r["tpr_estimate_pp"] == pytest.approx(r["tpr_actual_pp"])
    with pytest.raises(ValueError, match="without the arm's margin"):
        MC.validate(parent, arm.iloc[:7], shares)


def test_transfer_refuses_scores_of_different_monitor_rows():
    parent = _parent([-3.0, 0.5, 3.0], [False, True, True])
    shares = MC.bench_shares(parent["margin"], parent["gpt4o_hire"], EDGES)
    scored = pd.DataFrame({"margin": [-2.0, 0.4, 2.0], "prompts_sha1": "x"})
    moved = scored.assign(margin=[-2.0, -0.4, 2.0])
    r = MC.transfer(scored, moved, parent, parent[MC.KEY].assign(margin=[-3.0, -0.5, 3.0]),
                    shares)
    assert r["tpr_monitor_pp"] == pytest.approx(r["tpr_benchmark_pp"]) == pytest.approx(-50.0)
    assert r["rows_per_bin"] == [1, 1, 1]
    with pytest.raises(ValueError, match="different monitor rows"):
        MC.transfer(scored, moved.assign(prompts_sha1="y"), parent, parent[MC.KEY].assign(
            margin=[-3.0, -0.5, 3.0]), shares)


def test_prompt_digest_depends_on_order_and_text():
    assert MC.prompt_digest(["a", "b"]) != MC.prompt_digest(["b", "a"])
    assert MC.prompt_digest(["ab"]) != MC.prompt_digest(["a", "b"])
