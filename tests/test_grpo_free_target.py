"""A2's consistency reward and its data: the reward's values, and the built artifact left every
grpo-v3-all row unchanged."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation.grpo import free_consistency_reward  # noqa: E402

HIRE = '{"decision": "hire", "feedback": "Strong match for the role and its stack."}'
REJECT = '{"decision": "reject", "feedback": "Lacks the required years of experience."}'


def test_reward_is_agreement_with_the_attribute_free_share():
    out = free_consistency_reward([HIRE, REJECT, "no json at all"], [0.75, 0.75, 0.75])
    assert out == [0.75, 0.25, 0.0]


def test_reward_rows_must_line_up():
    with pytest.raises(ValueError):
        free_consistency_reward([HIRE, REJECT], [0.5])


ROOT = Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
SRC = ROOT / "artifacts/semisynthetic-v3/grpo-v3-all"
OUT = ROOT / "artifacts/semisynthetic-v3/grpo-v3-all-free"


@pytest.mark.skipif(not OUT.exists(), reason="grpo-v3-all-free not built on this machine")
@pytest.mark.parametrize("split", ["grpo_train", "grpo_validation", "grpo_monitor"])
def test_built_artifact_keeps_every_row(split):
    src, out = pd.read_parquet(SRC / f"{split}.parquet"), pd.read_parquet(OUT / f"{split}.parquet")
    assert out[src.columns].equals(src)
    assert out["free_hire_rate"].between(0, 1).all() and (out["free_samples"] >= 1).all()
