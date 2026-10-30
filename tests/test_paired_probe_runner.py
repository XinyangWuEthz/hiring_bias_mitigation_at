"""Runner invariants for a resumable per-row paired sampling experiment."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve()
RUNNER = HERE.with_name("paired_probe.py")
if not RUNNER.exists():
    RUNNER = HERE.parents[1] / "scripts" / "paired_probe.py"
SPEC = importlib.util.spec_from_file_location("paired_probe_runner_under_test", RUNNER)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def _chunk(ids=(2, 4)):
    return pd.DataFrame(
        [
            {"model": "dapo_final", "row_id": row_id, "view": view, "sample_index": sample}
            for row_id in ids
            for view in ("attributed", "attr_free")
            for sample in range(16)
        ]
    )


def test_repeated_free_prompts_get_distinct_reproducible_streams():
    prompt = "One shared attribute-free prompt."
    first = PROBE.request_seed(10, "attr_free", prompt)
    assert first == PROBE.request_seed(10, "attr_free", prompt)
    assert first != PROBE.request_seed(11, "attr_free", prompt)
    assert first != PROBE.request_seed(10, "attributed", prompt)
    assert first != PROBE.request_seed(10, "attr_free", prompt + " changed")


def test_all_9140_rows_have_disjoint_child_seed_ranges_for_both_views():
    # Same text deliberately maximizes repetition; row and view must still distinguish it.
    parents = [
        PROBE.request_seed(i, view, "shared prompt")
        for i in range(9140)
        for view in ("attributed", "attr_free")
    ]
    ordered = sorted(parents)
    assert ordered[0] >= 0 and ordered[-1] + 15 < 2**63
    assert all(right - left >= 16 for left, right in itertools.pairwise(ordered))


def test_chunk_validation_accepts_complete_shuffled_samples():
    PROBE.check_chunk(_chunk().sample(frac=1, random_state=3), [2, 4], "dapo_final")


def test_chunk_validation_rejects_missing_duplicate_and_foreign_samples():
    good = _chunk()
    variants = [good.iloc[:-1], pd.concat([good.iloc[:-1], good.iloc[[0]]], ignore_index=True)]
    for column, value in (
        ("row_id", 9),
        ("view", "other"),
        ("sample_index", 16),
        ("model", "sft_v3"),
    ):
        bad = good.copy()
        bad.loc[0, column] = value
        variants.append(bad)
    for bad in variants:
        with pytest.raises(AssertionError):
            PROBE.check_chunk(bad, [2, 4], "dapo_final")
