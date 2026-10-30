import importlib.util
import random
import unittest
from pathlib import Path

script = Path(__file__).parent / "freeze_benchmark_dev.py"
if not script.exists():
    script = Path(__file__).resolve().parents[1] / "scripts/freeze_benchmark_dev.py"
spec = importlib.util.spec_from_file_location("freeze_benchmark_dev", script)
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def metadata():
    return [
        {
            "candidate_id": f"role{role:02d}-candidate{candidate}",
            "pair_id": f"role{role:02d}-candidate{candidate}__job{job}",
            "job_position": f"Role {role:02d}",
            "reference_decision": "hire" if job < [0, 0, 1, 2, 3][candidate] else "reject",
        }
        for role in range(30)
        for candidate in range(5)
        for job in range(3)
    ]


class FrozenSplitTests(unittest.TestCase):
    def test_candidate_partition_keeps_roles_and_pairs_together(self):
        split = freeze.select_split(metadata())
        dev, rest = split["splits"]["dev"], split["splits"]["historical_remainder"]
        self.assertEqual((dev["candidate_count"], rest["candidate_count"]), (90, 60))
        self.assertEqual((dev["pair_count"], rest["pair_count"]), (270, 180))
        self.assertFalse(set(dev["candidate_ids"]) & set(rest["candidate_ids"]))
        self.assertEqual(
            set(dev["candidate_ids"]) | set(rest["candidate_ids"]),
            {r["candidate_id"] for r in metadata()},
        )
        for role in split["strata"]:
            self.assertEqual(len(role["dev_candidate_ids"]), 3)
            self.assertEqual(role["dev_hire_pairs"], 4)

    def test_input_order_cannot_change_split(self):
        rows = metadata()
        before = freeze.select_split(rows)
        random.Random(42).shuffle(rows)
        self.assertEqual(freeze.select_split(rows), before)

    def test_model_outcomes_do_not_enter_selection(self):
        rows = metadata()
        before = freeze.select_split(rows)
        for i, row in enumerate(rows):
            row.update(
                parent_decision="hire" if i % 2 else "reject",
                current_margin=i * 12345,
                all_models="changed",
            )
        self.assertEqual(freeze.select_split(rows), before)

    def test_missing_pair_fails(self):
        with self.assertRaises(ValueError):
            freeze.select_split(metadata()[:-1])

    def test_duplicate_pair_fails(self):
        rows = metadata()
        rows[1] = dict(rows[0])
        with self.assertRaises(ValueError):
            freeze.select_split(rows)

    def test_nonbinary_label_fails(self):
        rows = metadata()
        rows[0]["reference_decision"] = "review"
        with self.assertRaises(ValueError):
            freeze.select_split(rows)


if __name__ == "__main__":
    unittest.main()
