"""CPU contract tests; source files are tiny fixtures, never the frozen experiment."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch

MODULE = (Path(__file__).resolve().parents[1] / "src/hiring_bias_mitigation/mitigation"
          / "grpo_anchor_metadata.py")
if not MODULE.is_file():
    MODULE = Path(__file__).with_name("anchor_metadata.py")
SPEC = importlib.util.spec_from_file_location("anchor_metadata", MODULE)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class FakeDataset:
    """Only HuggingFace Dataset's immutable column-addition interface is needed."""

    def __init__(self, records):
        self.records = copy.deepcopy(records)
        self.column_names = list(self.records[0])

    def __iter__(self):
        return iter(copy.deepcopy(self.records))

    def __len__(self):
        return len(self.records)

    def add_column(self, name, values):
        assert name not in self.column_names
        return FakeDataset([{**row, name: value} for row, value in zip(self, values)])


def row(pair, side):
    return {"pair_id": pair, "protected_attr": "example", "condition": "explicit",
            "candidate_id": f"candidate-{pair}", "job_id": f"job-{pair}", "lang": "en",
            "reference_decision": side, "prompt": f"The original raw prompt for {pair}.",
            "sft_hire_rate": 0.5, "tipped": True}


def anchor(pair, side, winning, decided):
    return {"pair_id": pair, "reference_decision": side, "majority": side,
            "hire_votes": winning if side == "hire" else decided - winning,
            "decided": decided, "majority_share": winning / decided,
            "anchored": True, "anchor_source": "vote"}


class AnchorMetadataTest(TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.train = [row("a", "hire"), row("b", "reject"), row("c", "reject")]
        self.validation = [row("d", "hire")]
        self.records = {
            "train": self.train, "validation": self.validation,
            "new_rows": [self.train[0], self.train[1], self.validation[0]],
            "new2_rows": [self.train[2]],
            "new_anchors": [anchor("a", "hire", 8, 8), anchor("b", "reject", 4, 6),
                            anchor("d", "hire", 5, 8)],
            "new2_anchors": [anchor("c", "reject", 5, 7)],
        }
        for prefix in M.SOURCE_PREFIXES:
            self.records[f"{prefix}_prompts"] = [
                {"kind": "attributed", "key": str(i), "prompt": value["prompt"]}
                for i, value in enumerate(self.records[f"{prefix}_rows"])
            ]
        self.fingerprints = {}
        self.sync()
        self.addCleanup(patch.stopall)
        patch.object(M, "FROZEN_FILES", self.fingerprints).start()
        patch.object(M, "SPLIT_ROWS", {"train": 3, "validation": 1}).start()
        patch.object(M, "_read_records", self.read_records).start()

    def sync(self):
        for name, records in self.records.items():
            path = self.root / f"{name}.parquet"
            path.write_text(json.dumps(records), encoding="utf-8")
            self.fingerprints[name] = M._fingerprint(path)

    @staticmethod
    def read_records(path, columns=None):
        for record in json.loads(path.read_text(encoding="utf-8")):
            yield {name: record[name] for name in columns} if columns else record

    def attach(self, records=None, split="train"):
        return M.attach_anchor_metadata(FakeDataset(self.train if records is None else records),
                                        split=split, resolve_path=Path)

    def test_exact_join_keeps_original_fields_order_and_source_bytes(self):
        before = {name: Path(fp["path"]).read_bytes() for name, fp in self.fingerprints.items()}
        dataset = FakeDataset(self.train)
        result, receipt = M.attach_anchor_metadata(dataset, split="train", resolve_path=str)
        self.assertEqual(dataset.records, self.train)
        self.assertEqual([r["vote_stratum"] for r in result], ["8_of_8", "4_of_6", "5_of_7"])
        self.assertEqual([r["winning_votes"] for r in result], [8, 4, 5])
        self.assertEqual([r["decided_draws"] for r in result], [8, 6, 7])
        for actual, original in zip(result, self.train):
            self.assertEqual({k: actual[k] for k in original}, original)
        self.assertEqual(receipt["files_before"], receipt["files_after"])
        self.assertTrue(receipt["row_order_preserved"])
        self.assertEqual(before, {name: Path(fp["path"]).read_bytes()
                                  for name, fp in self.fingerprints.items()})

    def test_validation_is_explicit_and_preserves_five_of_eight(self):
        result, receipt = self.attach(self.validation, split="validation")
        self.assertEqual(result.records[0]["vote_stratum"], "5_of_8")
        self.assertEqual(receipt["split"], "validation")

    def test_dataset_must_be_full_ordered_raw_split(self):
        for records in (self.train[:-1], list(reversed(self.train)),
                        [{**self.train[0], "prompt": "chat wrapped"}, *self.train[1:]]):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, "frozen"):
                self.attach(records)

    def test_unknown_split_fails(self):
        with self.assertRaisesRegex(ValueError, "Only the frozen"):
            self.attach(split="other")

    def test_existing_prompt_id_conflict_fails_but_exact_reuse_is_idempotent(self):
        original, receipt = self.attach()
        repeated, second_receipt = self.attach(original.records)
        self.assertEqual(original.records, repeated.records)
        self.assertEqual(receipt, second_receipt)
        conflicting = copy.deepcopy(original.records)
        conflicting[1]["prompt_id"] = "another-prompt"
        with self.assertRaisesRegex(ValueError, "Existing metadata conflicts"):
            self.attach(conflicting)

    def test_duplicate_or_missing_anchor_fails(self):
        self.records["new_anchors"].append(copy.deepcopy(self.records["new_anchors"][0]))
        self.sync()
        with self.assertRaisesRegex(ValueError, "Duplicate anchor"):
            self.attach()
        self.records["new_anchors"] = self.records["new_anchors"][1:-1]
        self.sync()
        with self.assertRaisesRegex(ValueError, "missing its anchor"):
            self.attach()

    def test_anchor_label_or_share_conflict_fails(self):
        self.records["new_anchors"][0]["reference_decision"] = "reject"
        self.sync()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.attach()
        self.records["new_anchors"][0]["reference_decision"] = "hire"
        self.records["new_anchors"][0]["majority_share"] = 0.5
        self.sync()
        with self.assertRaisesRegex(ValueError, "majority share"):
            self.attach()

    def test_unknown_reference_label_fails(self):
        self.train[0]["reference_decision"] = "consider"
        self.sync()
        with self.assertRaisesRegex(ValueError, "Unknown reference"):
            self.attach()

    def test_tied_or_nonnumeric_vote_counts_fail(self):
        for update in ({"hire_votes": 4, "decided": 8}, {"hire_votes": 8.0},
                       {"decided": 0}, {"decided": True}, {"hire_votes": 9}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                M._checked_vote({**anchor("a", "hire", 8, 8), **update}, "hire")

    def test_fallback_anchor_fails(self):
        self.records["new_anchors"][0]["anchor_source"] = "tie"
        self.sync()
        with self.assertRaisesRegex(ValueError, "fallback anchor"):
            self.attach()

    def test_missing_duplicate_or_changed_source_prompt_fails(self):
        original = copy.deepcopy(self.records["new_prompts"])
        for variant in (original[1:], [*original, original[0]],
                        [{**original[0], "prompt": "wrong prompt"}, *original[1:]]):
            self.records["new_prompts"] = variant
            self.sync()
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, "source prompt"):
                self.attach()

    def test_duplicate_source_key_and_cross_pool_row_fail(self):
        self.records["new_rows"].append(copy.deepcopy(self.records["new_rows"][0]))
        self.sync()
        with self.assertRaisesRegex(ValueError, "Duplicate source row"):
            self.attach()
        self.records["new_rows"].pop()
        self.records["new2_rows"].append(copy.deepcopy(self.records["new_rows"][0]))
        self.sync()
        with self.assertRaisesRegex(ValueError, "more than one source"):
            self.attach()

    def test_missing_source_row_and_source_label_conflict_fail(self):
        self.records["new2_rows"][0] = {**self.train[2], "reference_decision": "hire"}
        self.sync()
        with self.assertRaisesRegex(ValueError, "conflicts with its source"):
            self.attach()
        self.records["new2_rows"] = []
        self.sync()
        with self.assertRaisesRegex(ValueError, "missing from its source"):
            self.attach()

    def test_changed_source_hash_fails_before_join(self):
        path = Path(self.fingerprints["validation"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "fingerprint changed: validation"):
            self.attach()

    def test_changed_source_during_join_is_detected_by_second_hash(self):
        original = self.read_records
        changed = False

        def mutate_after_read(path, columns=None):
            nonlocal changed
            yield from original(path, columns)
            if not changed and path.name == "new2_anchors.parquet":
                victim = Path(self.fingerprints["train"]["path"])
                victim.write_bytes(victim.read_bytes() + b" ")
                changed = True

        with (patch.object(M, "_read_records", mutate_after_read),
              self.assertRaisesRegex(ValueError, "fingerprint changed: train")):
            self.attach()


if __name__ == "__main__":
    main()
