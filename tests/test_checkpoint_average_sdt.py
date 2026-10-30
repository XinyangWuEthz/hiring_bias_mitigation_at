"""SDT corrections, alignment and paired candidate uncertainty."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "average_sdt", SCRIPTS / "analyze_checkpoint_average_sdt.py"
)
sdt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sdt)


def fixtures():
    base = []
    for candidate in range(8):
        for reference in ("hire", "reject"):
            for variant in range(3):
                base.append({
                    "candidate_id": f"candidate{candidate}", "job_id": f"job-{reference}",
                    "pair_id": f"candidate{candidate}__job-{reference}",
                    "protected_group": "gender" if variant < 2 else "none",
                    "protected_attr": ["A", "B", "none"][variant],
                    "condition": "explicit" if variant < 2 else "attr_free",
                    "reference_decision": reference, "outcome": "decided",
                    "decision": reference,
                })
    raw = {}
    for model, label in enumerate([sdt.PARENT, *sdt.SINGLES]):
        frame = pd.DataFrame(base)
        for index in range(len(frame)):
            if (index + model) % (model + 7) == 0:
                frame.loc[index, "decision"] = (
                    "hire" if frame.loc[index, "decision"] == "reject" else "reject")
        raw[label] = frame
    partitions = {"dev": {f"candidate{i}" for i in range(4)},
                  "historical_remainder": {f"candidate{i}" for i in range(4, 8)}}
    return raw, partitions


class SDTTests(unittest.TestCase):
    def test_log_linear_changes_nonextreme_rates(self):
        value = sdt.sdt_counts(6, 4, 1, 9)
        self.assertEqual(value["tpr_pct"], 60)
        self.assertAlmostEqual(value["corrected_hit_rate"], 6.5 / 11)
        self.assertAlmostEqual(value["corrected_false_alarm_rate"], 1.5 / 11)

    def test_extreme_rates_are_finite(self):
        value = sdt.sdt_counts(10, 0, 0, 10)
        self.assertTrue(np.isfinite(value["dprime"]))
        self.assertAlmostEqual(value["criterion_c"], 0)
        self.assertGreater(value["dprime"], 0)

    def test_missing_label_and_negative_counts_rejected(self):
        for counts in ((0, 0, 2, 3), (-1, 2, 3, 4), (1, 2, np.nan, 4)):
            with self.assertRaises(ValueError):
                sdt.sdt_counts(*counts)

    def test_identical_model_comparison_is_paired_exactly(self):
        raw, partitions = fixtures()
        raw["mean7"] = raw["step500"].sample(frac=1, random_state=11)
        result = sdt.analyze(raw, partitions, bootstrap_replicates=200, expected_prompt_count=48)
        for scope in sdt.SCOPES:
            comparison = result["scopes"][scope]["comparisons"]["vs_each_single"]["step500"]
            self.assertEqual(comparison["delta_dprime"], 0)
            self.assertEqual(comparison["delta_dprime_paired_ci95"], [0, 0])
            self.assertEqual(comparison["delta_c_paired_ci95"], [0, 0])
            self.assertFalse(result["scopes"][scope]["comparisons"]["decision"]["mechanism_proven"])
        json.dumps(result, allow_nan=False)

    def test_duplicate_key_rejected(self):
        raw, partitions = fixtures()
        raw["step425"].iloc[1] = raw["step425"].iloc[0]
        with self.assertRaises(ValueError):
            sdt.analyze(raw, partitions, expected_prompt_count=48)

    def test_changed_reference_rejected(self):
        raw, partitions = fixtures()
        raw["step425"].loc[0, "reference_decision"] = "reject"
        with self.assertRaises(ValueError):
            sdt.analyze(raw, partitions, expected_prompt_count=48)

    def test_unusable_decision_rejected(self):
        raw, partitions = fixtures()
        raw["step425"].loc[0, "outcome"] = "parse_failure"
        with self.assertRaises(ValueError):
            sdt.analyze(raw, partitions, expected_prompt_count=48)

    def test_overlapping_partitions_rejected(self):
        raw, partitions = fixtures()
        partitions["dev"].add("candidate7")
        with self.assertRaises(ValueError):
            sdt.analyze(raw, partitions, expected_prompt_count=48)

    def test_awaiting_average_has_no_inferred_result(self):
        raw, partitions = fixtures()
        result = sdt.analyze(raw, partitions, expected_prompt_count=48)
        self.assertEqual(result["status"], "awaiting_average_inference")
        self.assertIsNone(result["scopes"]["full"]["comparisons"])
        self.assertFalse(result["contract"]["published_standard_changed"])


if __name__ == "__main__":
    unittest.main()
