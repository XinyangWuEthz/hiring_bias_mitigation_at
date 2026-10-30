"""Failure-path tests for correction preparation and historical isolation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prep = load("prepare_teacher_corrections", HERE.parent / "scripts/prepare_teacher_corrections.py")
fixtures = load("teacher75_test_fixtures", HERE / "test_teacher75_standard_analysis.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "historical"
        self.output = self.root / "prepared"
        self.config_path = self.root / "configs/prepare.json"
        self.data = fixtures.fixture()
        for rubric in self.data["rubrics"]:
            rubric.update({"background": [], "role_definition": {
                "main_work": "Developer", "jd_quote": "Developer"}, "extraction_notes": []})
            for item in rubric["mandatory"]:
                item.update({"logic": "atomic", "alternatives": []})
        computed = prep.analysis.analyze(**self.data)
        names = {"source_cases.json": "sources", "frozen_job_rubrics.json": "rubrics",
                 "legacy_audit_rows_75.json": "old_rows", "pair_metadata.json": "pair_metadata"}
        for name, key in names.items():
            write_json(self.source / name, self.data[key])
        for review in self.data["review_files"]:
            write_json(self.source / review["file"], review["cases"])
        for name in ("audit_rows_75", "teacher_correction_candidates"):
            write_json(self.source / (name + ".json"), computed[name])
        (self.source / "audit_prompt.txt").write_text(
            prep.analysis.standard.render_prompt(self.data["policy"]), encoding="utf-8")
        write_json(self.root / "policy.json", self.data["policy"])
        self.config = {
            "workflow_id": prep.WORKFLOW_ID,
            "review_prompt_id": "hiring-evidence-v1-frozen-review-1",
            "policy": "../policy.json", "source_audit_dir": "../historical",
            "expected_counts": {"pairs": 2, "rows": 3},
            "provenance": {"expected_policy_id": fixtures.POLICY["policy_id"],
                           "expected_policy_sha256": fixtures.SHA, "standard_commit": "323d4f5",
                           "expected_source_file_sha256": tree_hashes(self.source)},
        }
        write_json(self.config_path, self.config)

    def read(self, path):
        return json.loads(path.read_text())

    def repin_fixture(self):
        """Create a new authorized fixture snapshot for non-pinning test paths."""
        self.config["provenance"]["expected_source_file_sha256"] = tree_hashes(self.source)
        write_json(self.config_path, self.config)

    def fail_without_write(self, pattern):
        with self.assertRaisesRegex(ValueError, pattern):
            prep.prepare(self.config_path, self.output)
        self.assertFalse(self.output.exists())

    def test_recomputes_targets_and_preserves_historical_snapshot(self):
        before = tree_hashes(self.source)
        manifest = prep.prepare(self.config_path, self.output)
        self.assertEqual(tree_hashes(self.source), before)
        self.assertEqual(manifest["observed_counts"], {
            "pairs": 2, "rows": 3, "evidence_ledgers": 4,
            "label_only_candidates": 2, "review_requests_prepared": 2})
        self.assertFalse(manifest["training_started"])
        self.assertEqual(manifest["model_calls_made"], 0)
        self.assertEqual(manifest["legacy_review_prompt_sha256"],
                         self.config["provenance"]["expected_source_file_sha256"]["audit_prompt.txt"])
        self.assertEqual((self.output / "historical_review_prompt.txt").read_bytes(),
                         (self.source / "audit_prompt.txt").read_bytes())
        rows = self.read(self.output / "audit_rows_75.json")
        self.assertEqual([r["correction_status"] for r in rows],
                         ["label_only_candidate", "label_only_candidate", "excluded_anchor_problem"])
        for row in rows:
            self.assertFalse(row["training_ready"])
            self.assertFalse(row["teacher_rationale_approved_for_reuse"])
            self.assertFalse(row["sample_provenance"]["resampled_under_current_prompt"])
            self.assertFalse(row["sample_provenance"]["existing_reviews_rerun_with_current_prompt"])
        for request in self.read(self.output / "review_requests.json"):
            self.assertFalse(request["request_executed"])
            self.assertNotIn("reference_decision", request)
            self.assertNotIn("audit_decision", request)
            self.assertNotIn("pair-P", request["prompt"])
            self.assertIn("Python required", request["prompt"])
        for name, expected in manifest["artifact_sha256"].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), expected)

    def test_stale_target_set_is_rejected_not_trusted(self):
        # Represents the prior seven-target export left behind after re-audit.
        path = self.source / "teacher_correction_candidates.json"
        targets = self.read(path)
        targets.append(self.read(self.source / "audit_rows_75.json")[-1])
        write_json(path, targets)
        self.repin_fixture()
        self.fail_without_write("stale or missing row IDs")

    def test_stale_category_is_rejected(self):
        path = self.source / "audit_rows_75.json"
        rows = self.read(path)
        rows[0]["audit_category"] = "anchor_problem"
        write_json(path, rows)
        self.repin_fixture()
        self.fail_without_write("stale audit_category")

    def test_tampered_source_evidence_is_rejected(self):
        path = self.source / "reviews/a.json"
        reviews = self.read(path)
        reviews["P01"]["mandatory"][0]["cv_quotes"] = ["Fabricated qualification"]
        write_json(path, reviews)
        self.fail_without_write("Pinned historical source SHA256 mismatch: reviews/a.json")

    def test_appended_contradiction_cannot_reuse_old_ledgers(self):
        path = self.source / "source_cases.json"
        sources = self.read(path)
        # Existing quoted spans remain intact: quote-presence checks alone
        # would miss this new decisive evidence and retain stale hire targets.
        sources[0]["candidate_profile"] += " I have never used Python and cannot program in Python."
        write_json(path, sources)
        self.fail_without_write("Pinned historical source SHA256 mismatch: source_cases.json")

    def test_modified_frozen_rubric_and_prompt_are_rejected(self):
        path = self.source / "frozen_job_rubrics.json"
        original = path.read_bytes()
        rubrics = self.read(path)
        rubrics[0]["extraction_notes"].append("Changed checklist interpretation")
        write_json(path, rubrics)
        self.fail_without_write("Pinned historical source SHA256 mismatch: frozen_job_rubrics.json")
        path.write_bytes(original)
        with (self.source / "audit_prompt.txt").open("a") as prompt:
            prompt.write("Changed historical instructions")
        self.fail_without_write("Pinned historical source SHA256 mismatch: audit_prompt.txt")

    def test_expected_policy_hash_must_match(self):
        self.config["provenance"]["expected_policy_sha256"] = "0" * 64
        write_json(self.config_path, self.config)
        self.fail_without_write("Unexpected policy_sha256")

    def test_missing_and_extra_reviews_are_rejected(self):
        path = self.source / "reviews/a.json"
        original = self.read(path)
        partial = dict(original)
        partial.pop("P01")
        write_json(path, partial)
        self.fail_without_write("Pinned historical source SHA256 mismatch: reviews/a.json")
        write_json(path, original)
        write_json(self.source / "reviews/c.json", original)
        self.fail_without_write("keys must match all consumed historical files exactly")
        (self.source / "reviews/c.json").unlink()
        (self.source / "reviews/b.json").unlink()
        self.fail_without_write("keys must match all consumed historical files exactly")

    def test_incomplete_extra_or_malformed_source_pins_are_rejected(self):
        pins = self.config["provenance"]["expected_source_file_sha256"]
        original = dict(pins)
        pins.pop("source_cases.json")
        write_json(self.config_path, self.config)
        self.fail_without_write("keys must match all consumed historical files exactly")
        pins.update(original)
        pins["unconsumed.json"] = "0" * 64
        write_json(self.config_path, self.config)
        self.fail_without_write("keys must match all consumed historical files exactly")
        pins.pop("unconsumed.json")
        pins["source_cases.json"] = "bad"
        write_json(self.config_path, self.config)
        self.fail_without_write("Malformed expected source SHA256")

    def test_wrong_prompt_id_and_training_configuration_are_rejected(self):
        self.config["review_prompt_id"] = "legacy"
        write_json(self.config_path, self.config)
        self.fail_without_write("Unexpected review_prompt_id")
        self.config["review_prompt_id"] = "hiring-evidence-v1-frozen-review-1"
        self.config["training_enabled"] = True
        write_json(self.config_path, self.config)
        self.fail_without_write("Config must contain exactly")

    def test_existing_output_and_nested_historical_output_are_rejected(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("Keep historical content")
        with self.assertRaisesRegex(ValueError, "fresh directory"):
            prep.prepare(self.config_path, self.output)
        self.assertEqual(sentinel.read_text(), "Keep historical content")
        with self.assertRaisesRegex(ValueError, "outside the historical"):
            prep.prepare(self.config_path, self.source / "new")
        self.assertFalse((self.source / "new").exists())

    def test_unknown_mandatory_excludes_target_and_keeps_review_queue(self):
        for review in self.data["review_files"]:
            review["cases"]["P01"] = fixtures.ledger("unknown")
            write_json(self.source / review["file"], review["cases"])
        computed = prep.analysis.analyze(**self.data)
        for name in ("audit_rows_75", "teacher_correction_candidates"):
            write_json(self.source / (name + ".json"), computed[name])
        self.repin_fixture()
        prep.prepare(self.config_path, self.output)
        self.assertEqual(self.read(self.output / "teacher_correction_candidates.json"), [])
        rows = self.read(self.output / "eligibility_decisions.json")
        self.assertEqual(rows[0]["correction_status"], "excluded_pending_review")
        self.assertEqual(len(self.read(self.output / "review_queue.json")), 1)


if __name__ == "__main__":
    unittest.main()
