"""Boundary checks for source provenance and the policy decision function."""

from copy import deepcopy
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


TEST_PATH = Path(__file__).resolve()
LAYOUTS = (
    (TEST_PATH.parent.parent / "scripts",
     TEST_PATH.parent.parent / "configs/hiring_decision_standard_v1.json"),
    (TEST_PATH.parent,
     TEST_PATH.parents[2] / "outputs/hiring_standard_v1/policy.json"),
)
for SCRIPT_DIRECTORY, POLICY_PATH in LAYOUTS:
    if (SCRIPT_DIRECTORY / "hiring_standard.py").is_file() and POLICY_PATH.is_file():
        sys.path.insert(0, str(SCRIPT_DIRECTORY))
        break
else:
    raise RuntimeError("Cannot find the hiring standard script and policy in a supported layout")

from hiring_standard import (
    ValidationError, canonical_policy_sha256, load_policy, main,
    render_prompt, validate_record,
)


JD = "Build Python services. Python is required. French B2 is required. Rust is preferred."
CV = "Built Python services. French B2. Rust projects. Worked with French-speaking customers. Led a team."


class AuditValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy(POLICY_PATH)

    def record(self):
        return {
            "policy_id": self.policy["policy_id"],
            "policy_sha256": canonical_policy_sha256(self.policy),
            "decision": "hire",
            "mandatory": [
                {"id": "H1", "requirement": "Python", "jd_quote": "Python is required.",
                 "state": "direct", "cv_quotes": ["Built Python services."],
                 "basis": "The CV states Python service work.", "date_precision": None},
                {"id": "H2", "requirement": "French B2", "jd_quote": "French B2 is required.",
                 "state": "direct", "cv_quotes": ["French B2."],
                 "basis": "The CV states French B2.", "date_precision": None},
            ],
            "preferred": [
                {"requirement": "Rust", "jd_quote": "Rust is preferred.",
                 "state": "direct", "cv_quotes": ["Rust projects."]},
            ],
            "background_unconfirmed": [],
            "role_fit": {"state": "supported", "jd_quote": "Build Python services.",
                         "cv_quotes": ["Built Python services."],
                         "basis": "Python service work matches the role."},
            "material_conflict": False,
            "data_quality_flags": [],
            "counts": {"H": 2, "D": 2, "I": 0, "U": 0, "F": 0,
                       "mandatory_supported_fraction": 1.0},
            "feedback": "Python service work and French B2 support the requirements.",
        }

    def rubric(self, record):
        return {
            "policy_id": record["policy_id"], "policy_sha256": record["policy_sha256"],
            "case_id": "fixture",
            "mandatory": [dict(
                {key: item[key] for key in ("id", "requirement", "jd_quote")},
                logic="atomic", alternatives=[],
            ) for item in record["mandatory"]],
            "preferred": [dict(
                {key: item[key] for key in ("requirement", "jd_quote")}, id=f"P{index}",
            ) for index, item in enumerate(record["preferred"], 1)],
            "background": [],
            "role_definition": {"main_work": "Build services", "jd_quote": "Build Python services."},
        }

    def check(self, record, rubric=None, jd=JD, cv=CV):
        return validate_record(record, jd, cv, self.policy, rubric)

    def test_all_supported_and_entailed(self):
        record = self.record()
        record["mandatory"][0]["state"] = "entailed"
        record["counts"].update(D=1, I=1)
        self.assertEqual(self.check(record, self.rubric(record)),
                         {"decision": "hire", "counts": record["counts"]})

    def test_unknown_with_no_or_related_evidence_is_review(self):
        for quotations in ([], ["Worked with French-speaking customers."]):
            with self.subTest(quotations=quotations):
                record = self.record()
                record["mandatory"][1].update(
                    state="unknown", cv_quotes=quotations,
                    basis="The cited language statement is not resolved by this review.",
                )
                record["counts"].update(D=1, U=1, mandatory_supported_fraction=0.5)
                record["decision"] = "review"
                self.assertEqual(self.check(record, cv=CV.replace("French B2. ", ""))["decision"], "review")

    def test_affirmative_contradiction_rejects(self):
        record = self.record()
        record["mandatory"][1].update(
            state="contradicted", cv_quotes=["My French is A1."],
            basis="The reviewer labels the A1 statement as contrary evidence.",
        )
        record["counts"].update(D=1, F=1, mandatory_supported_fraction=0.5)
        record["decision"] = "reject"
        self.assertEqual(self.check(record, cv=CV.replace("French B2.", "My French is A1."))["decision"], "reject")

    def test_material_conflict_precedes_rejection(self):
        record = self.record()
        record["mandatory"][1].update(state="contradicted", cv_quotes=["My French is A1."])
        record["counts"].update(D=1, F=1, mandatory_supported_fraction=0.5)
        record["material_conflict"] = True
        record["decision"] = "review"
        self.assertEqual(self.check(record, cv=CV + " My French is A1.")["decision"], "review")

    def test_zero_mandatory_requires_positive_role_evidence(self):
        for state, quotations, expected in (
            ("supported", ["Built Python services."], "hire"),
            ("unknown", [], "review"),
        ):
            with self.subTest(state=state):
                record = self.record()
                record["mandatory"] = []
                record["counts"].update(H=0, D=0, mandatory_supported_fraction=None)
                record["role_fit"].update(state=state, cv_quotes=quotations)
                record["decision"] = expected
                self.assertEqual(self.check(record, self.rubric(record),
                                            jd="Build Python services. Rust is preferred.")["decision"], expected)
                record["counts"]["mandatory_supported_fraction"] = 1.0
                with self.assertRaisesRegex(ValidationError, "must be null"):
                    self.check(record)

    def test_supported_role_requires_real_nonempty_quote(self):
        for quotations in ([], ["Imagined relevant work"], [""]):
            with self.subTest(quotations=quotations):
                record = self.record()
                record["role_fit"]["cv_quotes"] = quotations
                with self.assertRaises(ValidationError):
                    self.check(record)

    def test_nonsilent_evidence_states_require_cv_quote(self):
        for state in ("direct", "entailed", "contradicted"):
            with self.subTest(state=state):
                record = self.record()
                record["mandatory"][0].update(state=state, cv_quotes=[])
                with self.assertRaisesRegex(ValidationError, "requires a CV quotation"):
                    self.check(record)

    def test_frozen_or_aggregate_counts_once(self):
        jd = "Build Python services. Python OR Rust is required."
        record = self.record()
        record["mandatory"] = [record["mandatory"][0]]
        record["mandatory"][0].update(requirement="Python OR Rust", jd_quote="Python OR Rust is required.")
        record["preferred"] = []
        record["counts"].update(H=1, D=1)
        rubric = self.rubric(record)
        rubric["mandatory"][0].update(logic="OR", alternatives=["Python", "Rust"])
        result = self.check(record, rubric, jd=jd)
        self.assertEqual(result["counts"]["H"], 1)
        self.assertEqual(result["decision"], "hire")
        self.assertIn("OR/AND alternative semantics", render_prompt(self.policy))

    def test_wrong_source_quotes_are_rejected(self):
        mutations = [
            lambda r: r["mandatory"][0].update(jd_quote="Python must be used"),
            lambda r: r["mandatory"][0].update(cv_quotes=["Built Rust services."]),
            lambda r: r["preferred"][0].update(cv_quotes=["Python is required."]),
            lambda r: r["role_fit"].update(jd_quote="Manage teams"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = self.record()
                mutate(record)
                with self.assertRaisesRegex(ValidationError, "exact source quotation"):
                    self.check(record)

    def test_frozen_missing_added_reworded_and_duplicate_items(self):
        base = self.record()
        frozen = self.rubric(base)
        mutations = [
            lambda r: r["mandatory"].pop(),
            lambda r: r["mandatory"].append(dict(r["mandatory"][0], id="H3", requirement="Python again")),
            lambda r: r["mandatory"][0].update(requirement="Expert Python"),
            lambda r: r["mandatory"].append(deepcopy(r["mandatory"][0])),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = deepcopy(base)
                mutate(record)
                with self.assertRaises(ValidationError):
                    self.check(record, frozen)
        frozen["mandatory"].append(deepcopy(frozen["mandatory"][0]))
        with self.assertRaisesRegex(ValidationError, "duplicate id"):
            self.check(base, frozen)

    def test_frozen_order_can_change_but_preferred_must_match(self):
        record = self.record()
        frozen = self.rubric(record)
        frozen["mandatory"].reverse()
        self.assertEqual(self.check(record, frozen)["decision"], "hire")
        record["preferred"] = []
        with self.assertRaisesRegex(ValidationError, "preferred checklist"):
            self.check(record, frozen)

    def test_policy_identity_and_hash_are_required_on_both_objects(self):
        for target in ("record", "rubric"):
            for field in ("policy_id", "policy_sha256"):
                with self.subTest(target=target, field=field):
                    record = self.record()
                    rubric = self.rubric(record)
                    (record if target == "record" else rubric)[field] = "wrong"
                    with self.assertRaisesRegex(ValidationError, field):
                        self.check(record, rubric)

    def test_incorrect_declared_counts_and_decisions_are_rejected(self):
        mutations = [
            lambda r: r["counts"].update(H=3),
            lambda r: r["counts"].update(D=1),
            lambda r: r["counts"].update(I=True),
            lambda r: r["counts"].update(mandatory_supported_fraction=0.9),
            lambda r: r["counts"].update(mandatory_supported_fraction=1.0 + 1e-13),
            lambda r: r["counts"].update(mandatory_supported_fraction=10 ** 400),
            lambda r: r["counts"].update(mandatory_supported_fraction=float("nan")),
            lambda r: r.update(decision="reject"),
            lambda r: r.update(decision="review"),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = self.record()
                mutate(record)
                with self.assertRaises(ValidationError):
                    self.check(record)

    def test_preferred_never_changes_mandatory_decision(self):
        record = self.record()
        record["preferred"][0]["state"] = "contradicted"
        record["preferred"][0]["cv_quotes"] = ["I have never used Rust."]
        self.assertEqual(self.check(record, cv=CV.replace("Rust projects.", "I have never used Rust."))["decision"], "hire")

    def test_missing_extra_keys_invalid_state_and_nonboolean_conflict(self):
        mutations = [
            lambda r: r.pop("feedback"),
            lambda r: r.update(reasoning="extra"),
            lambda r: r["mandatory"][0].update(logic="OR"),
            lambda r: r["mandatory"][0].update(state="supported"),
            lambda r: r.update(material_conflict=1),
            lambda r: r.update(feedback="word " * 31),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                record = self.record()
                mutate(record)
                with self.assertRaises(ValidationError):
                    self.check(record)

    def test_missing_jd_remains_review_without_invented_quote(self):
        record = self.record()
        record.update(mandatory=[], preferred=[], material_conflict=True, decision="review")
        record["counts"].update(H=0, D=0, mandatory_supported_fraction=None)
        record["role_fit"].update(state="unknown", jd_quote="", cv_quotes=[], basis="The JD is missing.")
        record["data_quality_flags"] = ["Missing JD"]
        self.assertEqual(self.check(record, jd="")["decision"], "review")

    def test_fingerprint_is_canonical_and_sensitive_to_policy_changes(self):
        policy = json.loads(json.dumps(self.policy, sort_keys=True, indent=4))
        self.assertEqual(canonical_policy_sha256(policy), canonical_policy_sha256(self.policy))
        policy["version"] += "-changed"
        self.assertNotEqual(canonical_policy_sha256(policy), canonical_policy_sha256(self.policy))

    def test_render_contains_every_rule_placeholders_hash_and_exact_schema(self):
        prompt = render_prompt(self.policy)
        for rule in self.policy["rules"]:
            self.assertIn(rule["text"], prompt)
        for placeholder in ("{job_description}", "{candidate_profile}", "{frozen_job_rubric}"):
            self.assertEqual(prompt.count(placeholder), 1)
        self.assertIn(canonical_policy_sha256(self.policy), prompt)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "prompt.txt"
            self.assertEqual(main(["--policy", str(POLICY_PATH), "--render-prompt", str(destination)]), 0)
            self.assertEqual(destination.read_text(encoding="utf-8"), prompt)
            policy_copy = Path(directory) / "policy.json"
            original = POLICY_PATH.read_text(encoding="utf-8")
            policy_copy.write_text(original, encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--policy", str(policy_copy), "--render-prompt", str(policy_copy)])
            self.assertEqual(policy_copy.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
