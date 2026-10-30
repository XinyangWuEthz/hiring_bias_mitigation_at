"""Validate parent-only weight extraction on a complete synthetic benchmark."""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("freeze_monitor_v2_target",
                                            SCRIPTS / "freeze_monitor_v2_target.py")
target = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(target)


class FreezeMonitorV2TargetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temporary.name)
        cls.grid = [
            (group, condition, f"{group}-{number:02d}")
            for group, attributes in (("gender", 20), ("military_status", 5), ("religion", 9))
            for condition in ("explicit", "implicit")
            for number in range(attributes)
        ]
        cls.margins = [-5.0, -4.595, -2.197, -0.847, 0.0, 0.847, 2.197, 4.595, 5.0]
        cls.rows = []
        for pair in range(450):
            label = "hire" if pair < 150 else "reject"
            for index, (group, condition, attribute) in enumerate(cls.grid):
                cls.rows.append({
                    "cohort": "benchmark", "reference_decision": label,
                    "margin__SFT-v3": cls.margins[index % 9] if label == "hire" else 0.0,
                    "pair_id": f"pair-{pair:03d}", "protected_group": group,
                    "condition": condition, "protected_attr": attribute,
                })
        cls.source = cls.base / "parent-only.jsonl.gz"
        cls.write_rows(cls.source, cls.rows)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @staticmethod
    def write_rows(path, rows):
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def altered_source(self, change):
        rows = [dict(row) for row in self.rows]
        change(rows)
        path = self.base / (self._testMethodName + ".jsonl.gz")
        self.write_rows(path, rows)
        return path

    def test_exact_counts_weights_and_boundary_bins_without_arm_columns(self):
        self.assertTrue(all(set(row).intersection({"margin__DAPO-v3-all", "margin__GRPO-v3-all"})
                            == set() for row in self.rows))
        result = target.extract(self.source)
        self.assertEqual(result["rows"], 30600)
        self.assertEqual(result["pairs"], 450)
        self.assertEqual(len(result["grid"]), 68)
        self.assertEqual(result["edges"], [-4.595, -2.197, -0.847, 0, 0.847, 2.197, 4.595])
        # The expected boundary map is explicit, independent of the implementation's bisect.
        bins = {-5.0: 0, -4.595: 1, -2.197: 2, -0.847: 3, 0.0: 4,
                0.847: 5, 2.197: 6, 4.595: 7, 5.0: 7}
        expected = Counter()
        for index, (group, condition, _) in enumerate(self.grid):
            expected[("hire", bins[self.margins[index % 9]], group, condition)] += 150
            expected[("reject", 4, group, condition)] += 300
        for label, denominator in (("hire", 10200), ("reject", 20400)):
            self.assertEqual(len(result["cells"][label]), 48)
            self.assertAlmostEqual(sum(c["weight"] for c in result["cells"][label]), 1.0)
            for cell in result["cells"][label]:
                wanted = expected[(label, cell["bin"], cell["protected_group"], cell["condition"])]
                self.assertEqual(cell["count"], wanted)
                self.assertEqual(cell["denominator"], denominator)
                self.assertEqual(cell["weight"], wanted / denominator)
        self.assertEqual(result["parent_rates"]["hire"], 28 / 68)
        self.assertEqual(result["parent_rates"]["reject"], 0.0)
        self.assertEqual(result["decision_rule"], "margin > 0; exact zero is reject")
        self.assertNotIn("margin__DAPO-v3-all", result["used_columns"])
        self.assertTrue(result["not_validated_as_unified_protocol_labels"])

    def test_trained_arm_payloads_are_not_read(self):
        path = self.altered_source(lambda rows: [row.update({
            "margin__DAPO-v3-all": "deliberately not a numeric margin",
            "margin__GRPO-v3-all": None,
        }) for row in rows])
        clean = target.extract(self.source)
        poisoned = target.extract(path)
        self.assertEqual(poisoned["cells"], clean["cells"])
        self.assertEqual(poisoned["parent_rates"], clean["parent_rates"])

    def test_missing_benchmark_row_is_rejected(self):
        path = self.altered_source(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "450-pair"):
            target.extract(path)

    def test_different_68_variant_grid_is_rejected_without_changing_row_count(self):
        path = self.altered_source(
            lambda rows: rows[0].update(protected_attr="unexpected-attribute"))
        with self.assertRaisesRegex(ValueError, "same 68 attributed variants"):
            target.extract(path)

    def test_duplicate_pair_variant_is_rejected(self):
        path = self.altered_source(lambda rows: rows.__setitem__(1, dict(rows[0])))
        with self.assertRaisesRegex(ValueError, "Duplicate benchmark key"):
            target.extract(path)

    def test_nonfinite_parent_margin_is_rejected(self):
        for number in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(number=number):
                path = self.altered_source(
                    lambda rows, n=number: rows[0].update({"margin__SFT-v3": n}))
                with self.assertRaisesRegex(ValueError, "Invalid reference label or parent margin"):
                    target.extract(path)

    def test_unknown_reference_label_is_rejected(self):
        path = self.altered_source(lambda rows: rows[0].update(reference_decision="review"))
        with self.assertRaisesRegex(ValueError, "Invalid reference label or parent margin"):
            target.extract(path)

    def test_one_pair_cannot_change_reference_label_across_attributes(self):
        path = self.altered_source(lambda rows: rows[0].update(reference_decision="reject"))
        with self.assertRaisesRegex(ValueError, "label"):
            target.extract(path)

    def test_missing_label_population_is_rejected_explicitly(self):
        path = self.altered_source(lambda rows: [row.update(reference_decision="hire")
                                                for row in rows])
        with self.assertRaisesRegex(ValueError, "Both reference labels must have target support"):
            target.extract(path)

    def test_nonbenchmark_rows_do_not_contribute_to_target(self):
        path = self.altered_source(lambda rows: rows.append({"cohort": "monitor"}))
        result = target.extract(path)
        clean = target.extract(self.source)
        self.assertEqual(result["cells"], clean["cells"])
        self.assertEqual(result["rows"], 30600)


if __name__ == "__main__":
    unittest.main()
