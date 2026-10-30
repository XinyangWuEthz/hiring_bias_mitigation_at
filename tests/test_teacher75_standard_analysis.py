"""Synthetic tests for provenance, consensus, and pair/row aggregation."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ANALYSIS_PATH = HERE / "analysis.py"
if not ANALYSIS_PATH.exists():
    ANALYSIS_PATH = HERE.parent / "scripts/teacher75_standard_analysis.py"
SPEC = importlib.util.spec_from_file_location("teacher75_standard_analysis", ANALYSIS_PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)
POLICY_PATH = analysis.ROOT / "configs/hiring_decision_standard_v1.json"
if not POLICY_PATH.exists():
    POLICY_PATH = analysis.ROOT / "outputs/hiring_standard_v1/policy.json"
POLICY = json.loads(POLICY_PATH.read_text())
SHA = analysis.standard.canonical_policy_sha256(POLICY)


def ledger(state="direct", role="supported", conflict=False):
    mapping = {"direct": "D", "entailed": "I", "unknown": "U", "contradicted": "F"}
    counts = {"H": 1, "D": 0, "I": 0, "U": 0, "F": 0, "mandatory_supported_fraction": 0}
    counts[mapping[state]] = 1
    counts["mandatory_supported_fraction"] = int(state in {"direct", "entailed"})
    decision = "review" if conflict else ("reject" if state == "contradicted" else
               ("hire" if state in {"direct", "entailed"} and role == "supported" else "review"))
    return {
        "policy_id": POLICY["policy_id"], "policy_sha256": SHA, "decision": decision,
        "mandatory": [{"id": "M1", "requirement": "Python proficiency", "jd_quote": "Python required",
                       "state": state, "cv_quotes": [] if state == "unknown" else ["Python proficient"],
                       "basis": "Synthetic evidence judgment for aggregation test.", "date_precision": None}],
        "preferred": [], "background_unconfirmed": [],
        "role_fit": {"state": role, "jd_quote": "Developer", "cv_quotes": ["Developer experience"] if role == "supported" else [],
                     "basis": "Synthetic role match."},
        "material_conflict": conflict, "data_quality_flags": [], "counts": counts,
        "feedback": "Synthetic fixture for validation and grouping tests.",
    }


def row(row_id, case_id, anchor="hire", prior="insufficient_information"):
    r = {
        "row_id": row_id, "case_id": case_id, "pair_id": f"pair-{case_id}", "reference_decision": anchor,
        "anchor_reference": anchor, "anchor_hire_votes": 8 if anchor == "hire" else 0,
        "anchor_reject_votes": 8 if anchor == "reject" else 0,
        "anchor_valid_votes": 8, "anchor_total_votes": 8, "anchor_winner_votes": 8,
        "sft_attributed_right": 14, "sft_attributed_wrong": 2, "sft_attributed_valid": 16,
        "sft_attributed_invalid": 0, "sft_attributed_refused": 0,
        "audit_category": prior, "teacher_correction_supported": prior == "dapo_degradation",
        "correction_evidence_basis": "legacy_reason", "correction_condition": "legacy_condition",
        "anchor_reason_issue": False, "blind_agreement": True,
    }
    for prefix, right, wrong, status in (("sft_attr_free", 16, 0, "all_right"),
                                        ("dapo_attributed", 0, 16, "all_wrong"),
                                        ("dapo_attr_free", 0, 16, "all_wrong")):
        r.update({f"{prefix}_right": right, f"{prefix}_wrong": wrong, f"{prefix}_valid": 16,
                  f"{prefix}_invalid": 0, f"{prefix}_refused": 0, f"{prefix}_truncated": 0,
                  f"{prefix}_status": status})
    return r


def fixture():
    sources, rubrics, pairs = [], [], []
    rows = [row(1, "P01", prior="dapo_degradation"), row(2, "P01", prior="dapo_degradation"),
            row(3, "P02", anchor="reject", prior="anchor_problem")]
    for case_id in ("P01", "P02"):
        sources.append({"case_id": case_id, "job_description": "Developer. Python required.",
                        "candidate_profile": "Developer experience. Python proficient."})
        rubrics.append({"case_id": case_id, "policy_id": POLICY["policy_id"], "policy_sha256": SHA,
                        "mandatory": [{"id": "M1", "requirement": "Python proficiency", "jd_quote": "Python required"}],
                        "preferred": []})
        selected = [r for r in rows if r["case_id"] == case_id]
        pairs.append({"case_id": case_id, "pair_id": f"pair-{case_id}", "rows": len(selected),
                      "row_ids": [r["row_id"] for r in selected],
                      "reference_decision": selected[0]["reference_decision"], "anchor_winner_votes": 8, "anchor_valid_votes": 8})
    reviews = [{"reviewer": name, "file": f"reviews/{name}.json", "cases": {"P01": ledger(), "P02": ledger()}}
               for name in ("a", "b")]
    return dict(policy=POLICY, sources=sources, rubrics=rubrics, review_files=reviews,
                old_rows=rows, pair_metadata=pairs, expected_pairs=2, expected_rows=3)


class AnalysisTests(unittest.TestCase):
    def test_consensus_table_includes_every_binary_direction(self):
        self.assertEqual(analysis.classify("hire", ["hire", "hire"]), ("dapo_degradation", "hire", True))
        self.assertEqual(analysis.classify("reject", ["reject", "reject"]), ("dapo_degradation", "reject", True))
        self.assertEqual(analysis.classify("hire", ["reject", "reject"]), ("anchor_problem", "reject", False))
        self.assertEqual(analysis.classify("reject", ["hire", "hire"]), ("anchor_problem", "hire", False))
        for decisions in (["review", "review"], ["hire", "review"], ["hire", "reject"], ["review", "reject"]):
            self.assertEqual(analysis.classify("hire", decisions), ("insufficient_information", "review", False))

    def test_pair_row_aggregation_and_teacher_sampling_denominator(self):
        result = analysis.analyze(**fixture())
        s = result["summary"]
        self.assertEqual(s["categories"]["dapo_degradation"], {"pairs": 1, "rows": 2})
        self.assertEqual(s["categories"]["anchor_problem"], {"pairs": 1, "rows": 1})
        self.assertEqual(s["teacher_targets"]["sft_attributed_agreement"], 28)
        self.assertEqual(s["teacher_targets"]["sft_attributed_draws"], 32)
        self.assertEqual(s["teacher_targets"]["sft_attributed_agreement_per_draw"], 0.875)
        self.assertEqual(s["source_quote_validation"]["ledger_quote_instances"], 16)
        self.assertEqual(s["source_quote_validation"]["distinct_case_source_quotes"], 8)
        self.assertEqual(s["prior_seven_candidates"]["retained_row_ids"], [1, 2])

    def test_legacy_reasons_are_not_reused(self):
        result = analysis.analyze(**fixture())
        target = result["teacher_correction_candidates"][0]
        self.assertEqual(target["legacy_correction_evidence_basis"], "legacy_reason")
        self.assertEqual(target["legacy_correction_condition"], "legacy_condition")
        self.assertNotEqual(target["correction_evidence_basis"], "legacy_reason")
        self.assertEqual(len(target["target_reason"]), 2)
        self.assertEqual(target["target_reason"][0]["mandatory_evidence"][0]["basis"], "Synthetic evidence judgment for aggregation test.")
        self.assertTrue(target["rationale_reuse_unchecked"])
        self.assertFalse(target["teacher_rationale_approved_for_reuse"])

    def test_review_disagreement_excludes_both_rows_and_tracks_transition(self):
        data = fixture()
        data["review_files"][1]["cases"]["P01"] = ledger("unknown")
        result = analysis.analyze(**data)
        self.assertEqual(result["teacher_correction_candidates"], [])
        self.assertEqual(result["summary"]["decision_disagreement"]["rows"], 2)
        self.assertEqual(result["summary"]["prior_seven_candidates"]["removed_row_ids"], [1, 2])
        self.assertEqual(result["summary"]["audit_decision_counts"]["review"], {"pairs": 1, "rows": 2})
        self.assertEqual(result["summary"]["resolved_coverage"], {
            "pairs_resolved": 1, "pairs_total": 2, "pair_fraction": 0.5,
            "rows_resolved": 1, "rows_total": 3, "row_fraction": 1/3})
        self.assertEqual(result["review_queue"][0]["review_queue_reason"], "reviewer_decision_disagreement")
        self.assertEqual(len(result["review_queue"][0]["mandatory_state_disagreements"]), 1)

    def test_same_decision_different_direct_entailed_states_remains_target(self):
        data = fixture()
        data["review_files"][1]["cases"]["P01"] = ledger("entailed")
        result = analysis.analyze(**data)
        self.assertEqual(len(result["teacher_correction_candidates"]), 2)
        self.assertEqual(len(result["audit_cases_40"][0]["mandatory_state_disagreements"]), 1)

    def test_two_review_decisions_are_not_binary_consensus(self):
        data = fixture()
        for r in data["review_files"]:
            r["cases"]["P01"] = ledger("unknown")
        result = analysis.analyze(**data)
        self.assertEqual(result["review_queue"][0]["review_queue_reason"], "both_review")
        self.assertEqual(result["summary"]["decision_disagreement"]["rows"], 0)
        self.assertEqual(result["summary"]["categories"]["insufficient_information"]["rows"], 2)

    def test_missing_review_fails_before_outputs(self):
        data = fixture()
        del data["review_files"][1]["cases"]["P02"]
        with self.assertRaisesRegex(ValueError, "exactly two"):
            analysis.analyze(**data)

    def test_third_review_is_rejected(self):
        data = fixture()
        data["review_files"].append({"reviewer": "c", "file": "reviews/c.json", "cases": {"P01": ledger()}})
        with self.assertRaisesRegex(ValueError, "exactly two"):
            analysis.analyze(**data)

    def test_invalid_source_quote_is_rejected(self):
        data = fixture()
        data["review_files"][0]["cases"]["P01"]["mandatory"][0]["cv_quotes"] = ["Invented evidence"]
        with self.assertRaisesRegex(ValueError, "exact source"):
            analysis.analyze(**data)

    def test_wrong_fingerprint_is_rejected(self):
        data = fixture()
        data["review_files"][0]["cases"]["P01"]["policy_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "policy_sha256"):
            analysis.analyze(**data)

    def test_omitted_frozen_mandatory_is_rejected(self):
        data = fixture()
        data["rubrics"][0]["mandatory"].append({"id": "M2", "requirement": "Developer role", "jd_quote": "Developer"})
        with self.assertRaisesRegex(ValueError, "frozen_rubric"):
            analysis.analyze(**data)

    def test_miscounted_ledger_is_rejected(self):
        data = fixture()
        data["review_files"][0]["cases"]["P01"]["counts"]["D"] = 2
        with self.assertRaisesRegex(ValueError, "counts.D"):
            analysis.analyze(**data)

    def test_changed_subset_invariant_is_rejected(self):
        data = fixture()
        data["old_rows"][0]["sft_attr_free_right"] = 15
        with self.assertRaisesRegex(ValueError, "sft_attr_free_right"):
            analysis.analyze(**data)

    def test_pair_row_mapping_mismatch_is_rejected(self):
        data = fixture()
        data["pair_metadata"][0]["row_ids"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "row IDs mismatch"):
            analysis.analyze(**data)

    def test_duplicate_row_and_reviewer_are_rejected(self):
        data = fixture()
        data["old_rows"].append(copy.deepcopy(data["old_rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate row_id"):
            analysis.analyze(**data)
        data = fixture()
        data["review_files"][1]["reviewer"] = "a"
        with self.assertRaisesRegex(ValueError, "Duplicate reviewer"):
            analysis.analyze(**data)

    def test_report_preserves_limits_and_ledger_links(self):
        text = analysis.report(analysis.analyze(**fixture()))
        self.assertIn("历史上的“全错”表示与旧锚点不一致", text)
        self.assertIn("reviews/a.json", text)
        self.assertIn("rationale_reuse_unchecked=true", text)
        self.assertIn("未启动训练", text)


if __name__ == "__main__":
    unittest.main()
