"""Classification and completeness checks, with an optional real parquet integration test."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Support normal repository pytest collection from tests/, as well as local work/.
_scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
if (_scripts_dir / "paired_probe_analysis.py").is_file():
    sys.path.insert(0, str(_scripts_dir))

from paired_probe_analysis import (  # noqa: E402
    MODELS,
    analyze,
    build_model_rows,
    build_summary,
    validate_metadata,
    validate_samples,
)


def metadata(count=7):
    # Pair IDs deliberately overlap, and 8/8 must remain distinct from 8/10.
    return pd.DataFrame(
        [
            dict(
                row_id=i,
                reference_decision="hire" if i % 2 == 0 else "reject",
                pair_id=i // 2,
                anchor_hire_votes=8,
                anchor_reject_votes=0 if i < 3 else 2,
                anchor_valid_votes=8 if i < 3 else 10,
                anchor_total_votes=16,
                anchor_winner_votes=8,
            )
            for i in range(count)
        ]
    )


def samples_for(rows, model="sft_v3", patterns=None, n=16):
    patterns = patterns or {}
    records = []
    for row in rows.itertuples():
        right = row.reference_decision
        wrong = "reject" if right == "hire" else "hire"
        for view in ("attributed", "attr_free"):
            pattern = patterns.get((row.row_id, view), "W" * n)
            assert len(pattern) == n
            for k, outcome in enumerate(pattern):
                records.append(
                    dict(
                        model=model,
                        row_id=row.row_id,
                        view=view,
                        sample_index=k,
                        decision={
                            "R": right,
                            "W": wrong,
                            "I": "invalid output",
                            "F": "model refuses to decide",
                        }[outcome],
                        outcome={"R": "decided", "W": "decided", "I": "invalid", "F": "refused"}[
                            outcome
                        ],
                        finish_reason="stop",
                        raw_output="fixture",
                        seed=1000 + k,
                        output_tokens=3,
                    )
                )
    return pd.DataFrame(records)


class ProbeAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.rows = metadata()
        self.patterns = {
            (0, "attr_free"): "R" * 16,
            (1, "attr_free"): "W" * 16,
            (2, "attr_free"): "R" * 9 + "W" * 7,
            (3, "attr_free"): "R" * 8 + "W" * 8,
            (4, "attr_free"): "R" * 15 + "I",
            (5, "attributed"): "W" * 15 + "F",
            (6, "attributed"): "R" * 16,
        }
        self.samples = samples_for(self.rows, patterns=self.patterns)

    def table(self):
        validate_metadata(self.rows, 7)
        validate_samples(self.samples, "sft_v3", 7, 16)
        return build_model_rows(self.rows, self.samples, "sft_v3", 16)

    def test_strict_majority_tie_and_invalid(self):
        table = self.table().set_index("row_id")
        self.assertEqual(table.loc[0, "strict_route"], "attribute_bias_AB")
        self.assertEqual(table.loc[1, "strict_route"], "anchor_disagreement_CD_pause_RL")
        self.assertEqual(table.loc[2, "strict_route"], "mixed_unresolved")
        self.assertEqual(table.loc[2, "majority_route"], "attribute_bias_AB")
        self.assertEqual(table.loc[3, "majority_route"], "tie_unresolved")
        self.assertEqual(table.loc[4, "strict_route"], "invalid_unresolved")
        self.assertEqual(table.loc[4, "majority_route"], "invalid_unresolved")
        self.assertEqual(table.loc[4, "attr_free_right"], 15)
        self.assertEqual(table.loc[5, "attributed_status"], "invalid")
        self.assertEqual(table.loc[5, "attributed_wrong"], 15)
        self.assertEqual(table.loc[5, "attributed_refused"], 1)
        self.assertEqual(table.loc[5, "strict_route"], "not_applicable")
        self.assertTrue(table.loc[5, "attributed_zero_agreement"])
        self.assertEqual(table.loc[6, "strict_classification"], "attributed_all_right")

    def test_majority_wrong_and_partial_agreement(self):
        self.samples = samples_for(
            self.rows,
            patterns={
                (0, "attr_free"): "W" * 9 + "R" * 7,
                (1, "attributed"): "W" * 15 + "R",
            },
        )
        table = self.table().set_index("row_id")
        self.assertEqual(table.loc[0, "majority_route"], "anchor_disagreement_CD_pause_RL")
        self.assertEqual(table.loc[0, "strict_route"], "mixed_unresolved")
        self.assertEqual(table.loc[1, "attributed_status"], "mixed")

    def test_majority_cannot_override_invalid_or_refused(self):
        self.samples = samples_for(
            self.rows,
            patterns={
                (0, "attr_free"): "R" * 9 + "W" * 6 + "I",
                (1, "attr_free"): "W" * 9 + "R" * 6 + "F",
            },
        )
        table = self.table().set_index("row_id")
        for row_id in (0, 1):
            self.assertEqual(table.loc[row_id, "majority_route"], "invalid_unresolved")
            self.assertEqual(table.loc[row_id, "strict_route"], "invalid_unresolved")

    def test_parseable_truncation_is_overlapping_diagnostic(self):
        self.samples.loc[0, "finish_reason"] = "length"
        table = self.table().set_index("row_id")
        self.assertEqual(table.loc[0, "attributed_truncated"], 1)
        self.assertEqual(table.loc[0, "attributed_wrong"], 16)
        self.assertEqual(table.loc[0, "strict_route"], "attribute_bias_AB")

    def test_duplicate_rejected_even_if_total_count_unchanged(self):
        corrupt = pd.concat([self.samples.iloc[:-1], self.samples.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_samples(corrupt, "sft_v3", 7, 16)

    def test_missing_sample_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_samples(self.samples.iloc[:-1], "sft_v3", 7, 16)

    def test_foreign_model_view_row_index_and_bad_parse_rejected(self):
        for column, value, message in [
            ("model", "unknown", "model labels"),
            ("view", "unknown", "unknown view"),
            ("row_id", 7, "integers"),
            ("sample_index", 16, "integers"),
            ("decision", "invalid output", "requires"),
            ("outcome", "unknown", "unknown parser"),
        ]:
            with self.subTest(column=column):
                corrupt = self.samples.copy()
                corrupt.loc[0, column] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_samples(corrupt, "sft_v3", 7, 16)

    def test_metadata_invalid_votes_and_missing_rows_rejected(self):
        bad = self.rows.copy()
        bad.loc[0, "anchor_valid_votes"] = 7
        with self.assertRaisesRegex(ValueError, "hire \\+ reject"):
            validate_metadata(bad, 7)
        with self.assertRaisesRegex(ValueError, "unique rows"):
            validate_metadata(self.rows.iloc[:-1], 7)

    def test_exact_grouping_denominators_zero_agreement_and_transitions(self):
        left = self.table()
        other = samples_for(self.rows, "dapo_final", patterns={(0, "attributed"): "R" * 16})
        right = build_model_rows(self.rows, other, "dapo_final", 16)
        summary = build_summary(pd.concat([left, right], ignore_index=True), 16, 7)
        model = next(g for g in summary["by_model"] if g["model"] == "sft_v3")
        self.assertEqual(model["all_rows"], 7)
        self.assertEqual(model["unique_pairs"], 4)
        self.assertEqual(model["attributed_all_wrong_rows"], 5)
        self.assertEqual(model["strict_routes"]["attribute_bias_AB"]["pct_all_wrong_rows"], 20.0)
        self.assertAlmostEqual(model["strict_routes"]["attribute_bias_AB"]["pct_all_rows"], 100 / 7)
        self.assertEqual(model["zero_agreement_sensitivity"]["rows"], 6)
        self.assertEqual(model["zero_agreement_sensitivity"]["rows_with_invalid_or_refused"], 1)
        groups = [
            g
            for g in summary["by_model_reference_anchor_hire_votes"]
            if g["model"] == "sft_v3" and g["reference_decision"] == "hire"
        ]
        self.assertEqual({g["anchor_valid_votes"]: g["all_rows"] for g in groups}, {8: 2, 10: 2})
        vote10 = next(g for g in groups if g["anchor_valid_votes"] == 10)
        self.assertEqual(vote10["attributed_all_wrong_rows"], 1)
        self.assertEqual(sum(g["rows"] for g in summary["model_transitions"]), 7)
        transition = next(
            g
            for g in summary["model_transitions"]
            if g["sft_v3_classification"] == "attribute_bias_AB"
        )
        self.assertEqual(transition["dapo_final_classification"], "attributed_all_right")
        self.assertEqual(transition["rows"], 1)
        changes = summary["all_wrong_changes_sft_to_dapo"]
        self.assertEqual(changes["new_all_wrong_rows"], 2)
        self.assertEqual(changes["persistent_all_wrong_rows"], 4)
        self.assertEqual(changes["resolved_all_wrong_rows"], 1)
        self.assertEqual(changes["neither_all_wrong_rows"], 0)
        json.dumps(summary, allow_nan=False)

    @unittest.skipUnless(
        importlib.util.find_spec("pyarrow"), "pyarrow required for parquet integration"
    )
    def test_parquet_end_to_end_and_no_partial_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.rows.to_parquet(root / "rows.parquet", index=False)
            for model in MODELS:
                dest = root / "samples" / model / "shard-000"
                dest.mkdir(parents=True)
                frame = samples_for(self.rows, model, self.patterns)
                frame.iloc[:100].to_parquet(dest / "chunk-000.parquet", index=False)
                frame.iloc[100:].to_parquet(dest / "chunk-001.parquet", index=False)
            result = analyze(root, 7, 16)
            self.assertTrue(result["complete"])
            self.assertEqual(len(pd.read_parquet(root / "row_classifications.parquet")), 14)
            self.assertEqual(len(pd.read_parquet(root / "all_wrong_rows.parquet")), 10)
            self.assertIn("锚点", (root / "report.md").read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.rows.to_parquet(root / "rows.parquet", index=False)
            for model in MODELS:
                dest = root / "samples" / model / "shard-000"
                dest.mkdir(parents=True)
                frame = samples_for(self.rows, model, self.patterns)
                if model == "dapo_final":
                    frame = frame.iloc[:-1]
                frame.to_parquet(dest / "chunk-000.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                analyze(root, 7, 16)
            self.assertFalse((root / "summary.json").exists())
            self.assertFalse((root / "row_classifications.parquet").exists())


if __name__ == "__main__":
    unittest.main()
