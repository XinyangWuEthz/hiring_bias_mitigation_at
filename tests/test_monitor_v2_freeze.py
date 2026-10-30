"""Exercise intake isolation and immutable-freeze failure paths without model calls."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_monitor_v2_freeze.py"
SPEC = importlib.util.spec_from_file_location("prepare_monitor_v2_freeze", SCRIPT)
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def hashed(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MonitorV2FreezeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.config_path = self.base / "config.json"
        self.output = self.base / "frozen"
        self.registry = []
        for i in range(1, 7):
            candidate = f"candidate-{i:02d}"
            self.registry.append({
                "candidate_id": candidate,
                "pair_id": f"{candidate}__job-{i:02d}",
                "job_id": f"job-{i:02d}",
                "source_part": "djinni_new-fixture",
                "cv_sha256_norm": hashed(f"cv evidence for {candidate}"),
                "jd_sha256_norm": hashed(f"required evidence for job-{i:02d}"),
                "selection_hash": hashed("monitor-v2-intake-20261023:42:" + candidate),
                "split_hash": hashed("monitor-v2-split-20261023:42:" + candidate),
            })
        self.registry.sort(key=lambda row: row["selection_hash"])
        for rank, row in enumerate(self.registry, 1):
            row["selection_rank"] = rank
        self.batch = [self.intake_row(row) for row in self.registry[:3]]
        self.config = {
            "experiment_id": "unit-test-monitor-v2",
            "inputs": {},
            "annotation": {"labels": ["hire", "reject"], "source_as_of": "2026-10-01"},
            "partition": {
                "salt": "monitor-v2-split-20261023:42:",
                "intake_salt": "monitor-v2-intake-20261023:42:",
            },
            "intake": {"first_batch_size": 3},
            "release": {
                "arm_scoring_allowed": False,
                "training_allowed": False,
                "pending_second_lock": ["resolved labels", "parent margins", "validation"],
            },
        }
        self.write_input("eligible_registry", self.registry, jsonl=True)
        self.write_input("intake_batch", self.batch, jsonl=True)
        self.write_input("excluded_candidates", {"all_candidate_ids": ["excluded-other"]})
        self.weights = {
            "edges": [-4.595, -2.197, -0.847, 0, 0.847, 2.197, 4.595],
            "grid": [{"protected_group": "gender", "condition": condition,
                      "protected_attr": "fixture"}
                     for condition in ("explicit", "implicit")],
            "cells": {
                label: [{"bin": bin_id, "protected_group": "gender",
                         "condition": condition, "count": 1, "denominator": 16,
                         "weight": 0.0625}
                        for bin_id in range(8)
                        for condition in ("explicit", "implicit")]
                for label in ("hire", "reject")
            },
        }
        self.write_input("legacy_target_weights", self.weights)

    @staticmethod
    def intake_row(row):
        return dict(row,
                    cv=f"CV evidence for {row['candidate_id']}",
                    job_description=f"Required evidence for {row['job_id']}",
                    label=None, label_status="unlabelled",
                    source="djinni_new", development_exposed=False,
                    old_anchor="hire", sft_margin=7.3, dapo_margin=-4.1)

    def save_config(self):
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")

    def write_input(self, key, value, *, jsonl=False):
        path = self.base / (key + (".jsonl" if jsonl else ".json"))
        content = ("".join(json.dumps(row) + "\n" for row in value)
                   if jsonl else json.dumps(value))
        path.write_text(content, encoding="utf-8")
        self.config["inputs"][key] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        self.save_config()
        return path

    def prepare(self):
        return freeze.prepare(self.config_path, self.output)

    def test_prepare_and_verify_preserve_blinding_and_block_release(self):
        manifest = self.prepare()
        verified = freeze.verify_freeze(self.config_path, self.output)
        self.assertTrue(verified["intake_integrity_passed"])
        self.assertFalse(verified["arm_scoring_allowed"])
        self.assertFalse(verified["training_allowed"])
        self.assertEqual(manifest["eligible_candidates"], 6)
        self.assertEqual(manifest["first_batch_candidates"], 3)
        self.assertEqual(manifest["confirmed_hire_candidates"], 0)
        self.assertEqual(manifest["confirmed_reject_candidates"], 0)
        queue = freeze.read_jsonl(self.output / "annotation_queue.jsonl")
        self.assertEqual([r["candidate_id"] for r in queue],
                         [r["candidate_id"] for r in self.batch])
        for row in queue:
            self.assertEqual(set(row), {
                "candidate_id", "pair_id", "job_id", "job_description", "cv",
                "source_as_of", "label", "eligibility",
            })
            self.assertIsNone(row["label"])
            self.assertEqual(row["eligibility"], "awaiting_jd_rubric")
        for row in freeze.read_jsonl(self.output / "jd_only_queue.jsonl"):
            self.assertEqual(set(row), {"job_id", "job_description"})

    def test_partition_is_fixed_across_candidate_order(self):
        expected = {
            "candidate-01": "monitor", "candidate-02": "monitor",
            "candidate-03": "monitor", "candidate-04": "validation",
            "candidate-05": "validation", "candidate-06": "validation",
        }
        self.assertEqual({candidate: freeze.partition(candidate)
                          for candidate in reversed(list(expected))}, expected)
        self.prepare()
        partitions = freeze.read_jsonl(self.output / "candidate_partitions.jsonl")
        self.assertEqual([r["candidate_id"] for r in partitions],
                         [r["candidate_id"] for r in self.registry])
        self.assertEqual({r["candidate_id"]: r["bank"] for r in partitions}, expected)

    def test_input_byte_drift_is_rejected_even_when_parsed_rows_unchanged(self):
        path = self.base / self.config["inputs"]["eligible_registry"]["path"]
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_excluded_candidate_cannot_enter_even_with_updated_fingerprint(self):
        self.write_input("excluded_candidates", {
            "all_candidate_ids": [self.registry[0]["candidate_id"]]})
        with self.assertRaisesRegex(ValueError, "excluded candidate"):
            self.prepare()

    def test_distinct_ids_with_duplicate_normalized_cvs_are_rejected(self):
        self.registry[-1]["cv_sha256_norm"] = self.registry[0]["cv_sha256_norm"]
        self.write_input("eligible_registry", self.registry, jsonl=True)
        with self.assertRaisesRegex(ValueError, "duplicate CV"):
            self.prepare()

    def test_registry_reordering_is_rejected_even_with_updated_fingerprint(self):
        self.write_input("eligible_registry", list(reversed(self.registry)), jsonl=True)
        with self.assertRaisesRegex(ValueError, "hash order"):
            self.prepare()

    def test_first_batch_cannot_skip_candidate_in_fixed_intake_order(self):
        self.batch[-1] = self.intake_row(self.registry[3])
        self.write_input("intake_batch", self.batch, jsonl=True)
        with self.assertRaisesRegex(ValueError, "First batch"):
            self.prepare()

    def test_preselected_binary_label_is_forbidden(self):
        self.batch[0]["label"] = "hire"
        self.write_input("intake_batch", self.batch, jsonl=True)
        with self.assertRaisesRegex(ValueError, "preselected labels"):
            self.prepare()

    def test_null_label_with_non_unlabelled_status_is_forbidden(self):
        self.batch[0]["label_status"] = "resolved"
        self.write_input("intake_batch", self.batch, jsonl=True)
        with self.assertRaisesRegex(ValueError, "preselected labels"):
            self.prepare()

    def test_output_tampering_is_detected(self):
        self.prepare()
        path = self.output / "annotation_queue.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Frozen output changed"):
            freeze.verify_freeze(self.config_path, self.output)

    def test_config_change_is_detected_even_if_inputs_are_unchanged(self):
        self.prepare()
        self.config["experiment_id"] = "post-hoc-change"
        self.save_config()
        with self.assertRaisesRegex(ValueError, "Config changed"):
            freeze.verify_freeze(self.config_path, self.output)

    def test_arm_scoring_and_training_stage_gates_always_refuse(self):
        self.prepare()
        for stage in ("arm-scoring", "training"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "Blocked"):
                freeze.verify_freeze(self.config_path, self.output, stage)

    def test_release_flags_cannot_authorize_actions_during_prepare(self):
        for flag in ("arm_scoring_allowed", "training_allowed"):
            with self.subTest(flag=flag):
                self.config["release"][flag] = True
                self.save_config()
                with self.assertRaisesRegex(ValueError, "cannot authorize"):
                    self.prepare()
                self.config["release"][flag] = False

    def test_manifest_release_flag_tampering_is_rejected(self):
        self.prepare()
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["training_allowed"] = True
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "invalid release flags"):
            freeze.verify_freeze(self.config_path, self.output)

    def test_existing_freeze_directory_is_never_overwritten(self):
        self.prepare()
        previous = (self.output / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual((self.output / "manifest.json").read_bytes(), previous)

    def test_invalid_weight_sum_is_rejected(self):
        self.weights["cells"]["hire"][0]["weight"] = 0.25
        self.write_input("legacy_target_weights", self.weights)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            self.prepare()

    def test_nonfinite_weights_cannot_bypass_sum_validation(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.weights["cells"]["hire"][0]["weight"] = value
                self.write_input("legacy_target_weights", self.weights)
                with self.assertRaisesRegex(ValueError, "Nonfinite"):
                    self.prepare()

    def test_target_weights_must_follow_counts_not_just_sum_to_one(self):
        self.weights["cells"]["hire"][0]["weight"] += 0.01
        self.weights["cells"]["hire"][1]["weight"] -= 0.01
        self.write_input("legacy_target_weights", self.weights)
        with self.assertRaisesRegex(ValueError, "count proportions"):
            self.prepare()

    def test_benchmark_job_is_excluded_even_with_unseen_candidate(self):
        self.write_input("excluded_candidates", {
            "all_candidate_ids": [],
            "benchmark_job_ids": [self.registry[0]["job_id"]],
        })
        with self.assertRaisesRegex(ValueError, "benchmark job"):
            self.prepare()

    def test_cv_text_must_match_frozen_normalized_digest(self):
        self.batch[0]["cv"] += " Added professional experience."
        self.write_input("intake_batch", self.batch, jsonl=True)
        with self.assertRaisesRegex(ValueError, "registry digest"):
            self.prepare()

    def test_non_djinni_or_exposed_intake_is_rejected(self):
        for key, value in (("source", "rjdb"), ("development_exposed", True)):
            with self.subTest(key=key):
                original = self.batch[0][key]
                self.batch[0][key] = value
                self.write_input("intake_batch", self.batch, jsonl=True)
                with self.assertRaisesRegex(ValueError, "source or development exposure"):
                    self.prepare()
                self.batch[0][key] = original

    def test_manifest_cannot_drop_a_required_output_to_hide_tampering(self):
        self.prepare()
        path = self.output / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["outputs"]["annotation_queue.jsonl"]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            freeze.verify_freeze(self.config_path, self.output)

    def test_jsonl_unicode_line_separator_is_text_not_record_boundary(self):
        path = self.base / "unicode.jsonl"
        path.write_text(json.dumps({"cv": "First\u2028second"}, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        self.assertEqual(freeze.read_jsonl(path), [{"cv": "First\u2028second"}])


if __name__ == "__main__":
    unittest.main()
