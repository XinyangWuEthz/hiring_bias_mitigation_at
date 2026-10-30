"""Acceptance-gate regressions: malformed evidence must never authorize training."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_monitor_transfer.py"
spec = importlib.util.spec_from_file_location("check_monitor_transfer", SCRIPT)
gate = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = gate
spec.loader.exec_module(gate)


class MonitorGateFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.scores_dir = self.root / "scores"
        self.scores_dir.mkdir()
        self.frames = {}
        self.keys = ["pair_id", "protected_group", "protected_attr", "condition"]
        self.edges = [-4.595, -2.197, -0.847, 0.0, 0.847, 2.197, 4.595]
        self.margins = np.repeat([-6., -3., -1., -.5, .5, 1., 3., 6.], 2)
        self.specs = [{"label": f"model{i}", "run": f"run{i}", "model": f"merged/model{i}"}
                      for i in range(8)]
        self.config = {
            "experiment_id": "frozen-fixture", "output_root": str(self.root),
            "parent": self.specs[0], "arms": self.specs[1:], "rows": "monitor.parquet",
            "train": "train.parquet", "validation": "validation.parquet",
            "expected_monitor_rows": 16, "expected_monitor_candidates": 8,
            "expected_train_rows": 2, "source_sha256": {},
            "reference_metrics": str(self.root / "reference_metrics.csv"),
            "limits": {"max_abs_tpr_margin_error_pp": 2.,
                       "max_optimism_generated_tpr_pp": 2., "tpr_stop_pp": 8.},
        }
        monitor = pd.DataFrame({"prompt": [f"prompt{i}" for i in range(16)],
                                "candidate_id": [f"candidate{i//2}" for i in range(16)],
                                "pair_id": [f"pair{i//2}" for i in range(16)]})
        self.put("monitor.parquet", monitor)
        self.put("train.parquet", pd.DataFrame({"prompt": ["train0", "train1"],
                 "candidate_id": ["train0", "train1"], "pair_id": ["train0", "train1"]}))
        self.put("validation.parquet", pd.DataFrame({"prompt": ["validation"],
                 "candidate_id": ["validation"], "pair_id": ["validation"]}))
        benchmark = pd.DataFrame({"pair_id": [f"bench{i}" for i in range(16)],
                                  "protected_group": "gender", "protected_attr": "woman",
                                  "condition": "explicit",
                                  "reference_decision": ["hire", "reject"] * 8})
        self.put("outputs/raw/run0.parquet", benchmark)
        for model in self.specs:
            margins = benchmark[self.keys].assign(margin=self.margins)
            self.put(f"outputs/operating_point/v2/{model['run']}--adapter.parquet", margins)
            scores = pd.DataFrame({"margin": self.margins, "row_index": np.arange(16),
                                   "prompts_sha1": "frozen-digest", "model": model["model"],
                                   "rows": str(self.root / "monitor.parquet")})
            self.put(f"scores/{model['label']}.parquet", scores, register=False)
        metrics_path = Path(self.config["reference_metrics"])
        pd.DataFrame({"run": [s["run"] for s in self.specs], "tpr_vs_gpt4o_pct": 50.,
                       "fpr_vs_gpt4o_pct": 50.}).to_csv(metrics_path, index=False)
        self.config["source_sha256"]["reference_metrics.csv"] = gate.sha256(metrics_path)
        # Production numerical functions are separately covered in test_grpo; this fixture
        # isolates the gate's evidence checks without needing GPU imports or Parquet engines.
        self.mc = SimpleNamespace(
            KEY=self.keys, prompt_digest=lambda prompts: "frozen-digest",
            margin_bins=lambda margins, edges: np.searchsorted(edges, margins, side="right"),
            bench_shares=lambda margins, hire: {
                "edges": self.edges, "share_ref_hire": [.125] * 8,
                "share_ref_reject": [.125] * 8, "parent_tpr": .5, "parent_fpr": .5,
                "reference_hire_share": .5},
            project_from_margins=lambda *args: {"delta_tpr": 0., "delta_fpr": 0.})

    def put(self, name, frame, register=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture parquet backing bytes: " + name)
        self.frames[path.resolve()] = frame
        if register:
            self.config["source_sha256"][name] = gate.sha256(path)

    def run_gate(self):
        with patch.object(gate.pd, "read_parquet", side_effect=lambda p: self.frames[Path(p).resolve()].copy()):
            return gate.check_transfer(self.config, self.scores_dir, calibration=self.mc)

    def score(self, label="model1"):
        return self.frames[(self.scores_dir / f"{label}.parquet").resolve()]

    def test_complete_registered_evidence_passes(self):
        result = self.run_gate()
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["arms"]), 7)
        self.assertEqual([b["rows"] for b in result["parent_margin_bins"]], [2] * 8)
        self.assertEqual([b["candidates"] for b in result["parent_margin_bins"]], [1] * 8)

    def test_truncated_scores_cannot_pass(self):
        self.frames[self.scores_dir / "model1.parquet"] = self.score().iloc[:-1]
        with self.assertRaisesRegex(gate.GateInputError, "expected 16 scores"):
            self.run_gate()

    def test_nonfinite_score_cannot_pass(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.score().loc[15, "margin"] = value
                with self.assertRaisesRegex(gate.GateInputError, "nonfinite margin"):
                    self.run_gate()

    def test_digest_is_checked_on_every_row(self):
        self.score().loc[15, "prompts_sha1"] = "different"
        with self.assertRaisesRegex(gate.GateInputError, "inconsistent prompt digest"):
            self.run_gate()

    def test_row_order_cannot_change(self):
        self.score().loc[[0, 1], "row_index"] = [1, 0]
        with self.assertRaisesRegex(gate.GateInputError, "original ordered row sequence"):
            self.run_gate()

    def test_duplicate_row_indices_cannot_pass(self):
        self.score().loc[15, "row_index"] = 14
        with self.assertRaisesRegex(gate.GateInputError, "original ordered row sequence"):
            self.run_gate()

    def test_wrong_model_cannot_pass(self):
        self.score().loc[15, "model"] = "merged/unregistered-model"
        with self.assertRaisesRegex(gate.GateInputError, "unexpected model"):
            self.run_gate()

    def test_wrong_rows_path_cannot_pass(self):
        self.score().loc[15, "rows"] = "other-monitor.parquet"
        with self.assertRaisesRegex(gate.GateInputError, "unexpected monitor"):
            self.run_gate()

    def test_positive_weight_empty_bins_fail_even_when_estimate_is_zero(self):
        self.score("model0")["margin"] = 10.
        result = self.run_gate()
        self.assertFalse(result["passed"])
        self.assertEqual(sum(not b["covered"] for b in result["parent_margin_bins"]), 7)

    def test_stale_source_hash_cannot_pass(self):
        (self.root / "train.parquet").write_text("changed training data")
        with self.assertRaisesRegex(gate.GateInputError, "stale source hash"):
            self.run_gate()

    def test_missing_source_hash_cannot_pass(self):
        del self.config["source_sha256"]["outputs/operating_point/v2/run1--adapter.parquet"]
        with self.assertRaisesRegex(gate.GateInputError, "missing registered source hashes"):
            self.run_gate()

    def test_training_overlap_cannot_pass(self):
        self.frames[self.root / "train.parquet"].loc[0, "candidate_id"] = "candidate0"
        with self.assertRaisesRegex(gate.GateInputError, "overlaps training candidates"):
            self.run_gate()

    def test_duplicate_benchmark_keys_cannot_pass(self):
        path = self.root / "outputs/operating_point/v2/run1--adapter.parquet"
        self.frames[path].loc[1, self.keys] = self.frames[path].loc[0, self.keys].to_numpy()
        with self.assertRaisesRegex(gate.GateInputError, "duplicate join keys"):
            self.run_gate()

    def test_missing_benchmark_margin_cannot_pass(self):
        path = self.root / "outputs/operating_point/v2/run1--adapter.parquet"
        self.frames[path] = self.frames[path].iloc[:-1]
        with self.assertRaisesRegex(gate.GateInputError, "nonfinite margin"):
            self.run_gate()

    def test_missing_comparison_arm_cannot_pass(self):
        self.config["arms"] = self.config["arms"][:-1]
        with self.assertRaisesRegex(gate.GateInputError, "seven registered"):
            self.run_gate()


class ThresholdBoundaries(unittest.TestCase):
    limits = {"max_abs_tpr_margin_error_pp": 2.,
              "max_optimism_generated_tpr_pp": 2., "tpr_stop_pp": 8.}

    def test_absolute_error_exactly_two_is_inclusive(self):
        self.assertTrue(gate.evaluate_arm(-2., -4., -2., self.limits)["passed"])

    def test_absolute_error_above_two_fails(self):
        result = gate.evaluate_arm(-2., -4.00001, -2., self.limits)
        self.assertFalse(result["checks"]["margin_error_within_limit"])

    def test_generated_optimism_exactly_two_fails(self):
        result = gate.evaluate_arm(-2., -2., -4., self.limits)
        self.assertFalse(result["checks"]["generated_optimism_below_limit"])

    def test_generated_optimism_just_below_two_passes(self):
        self.assertTrue(gate.evaluate_arm(-2., -2., -3.99999, self.limits)["passed"])

    def test_decimal_csv_roundoff_cannot_accept_exact_two_pp_optimism(self):
        result = gate.evaluate_arm(-8.1, -8.1, 47.63 - 57.73, self.limits)
        self.assertFalse(result["checks"]["generated_optimism_below_limit"])

    def test_monitor_exactly_minus_eight_does_not_trigger(self):
        result = gate.evaluate_arm(-8., -9., -9., self.limits)
        self.assertFalse(result["checks"]["historical_drop_triggers_stop"])

    def test_either_historical_metric_can_require_stop(self):
        for margin, generated in ((-9., -7.), (-7., -9.)):
            with self.subTest(margin=margin, generated=generated):
                result = gate.evaluate_arm(-7.5, margin, generated, self.limits)
                self.assertFalse(result["checks"]["historical_drop_triggers_stop"])

    def test_nonfinite_result_fails_closed(self):
        with self.assertRaisesRegex(gate.GateInputError, "nonfinite TPR"):
            gate.evaluate_arm(np.nan, 0., 0., self.limits)

    def test_numpy_projection_stop_checks_are_native_json_booleans(self):
        result = gate.evaluate_arm(np.float64(-3.24), np.float64(-9.61),
                                   np.float64(-10.10), self.limits)
        self.assertTrue(all(type(value) is bool for value in result["checks"].values()))
        self.assertFalse(json.loads(json.dumps(result))["passed"])


class CliFailureReporting(unittest.TestCase):
    def test_realistic_numpy_calibration_failure_is_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text("{}")
            output = Path(tmp) / "gate-v2.json"
            result = {"passed": np.bool_(False), "status": "calibration_failed",
                      "failures": ["DAPO: historical_drop_triggers_stop"],
                      "arms": [{"tpr_monitor_pp": np.float64(-3.24),
                                "tpr_actual_pp": np.float32(-9.61),
                                "rows": np.int64(696),
                                "check": np.bool_(False)}]}
            with patch.object(gate, "check_transfer", return_value=result):
                code = gate.main(["--config", str(config), "--scores-dir", tmp,
                                  "--output", str(output)])
            self.assertEqual(code, 2)
            report = json.loads(output.read_text())
            self.assertIs(report["passed"], False)
            self.assertIs(report["arms"][0]["check"], False)
            self.assertEqual(report["arms"][0]["rows"], 696)
            self.assertEqual(report["arms"][0]["tpr_monitor_pp"], -3.24)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_serialization_error_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gate.json"
            with self.assertRaises(TypeError):
                gate.publish_report(output, {"passed": True, "unserializable": object()})
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_publish_race_preserves_existing_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gate.json"
            output.write_text("existing evidence")
            with self.assertRaises(FileExistsError):
                gate.publish_report(output, {"passed": False})
            self.assertEqual(output.read_text(), "existing evidence")
            self.assertEqual(list(Path(tmp).iterdir()), [output])

    def test_unexpected_report_type_becomes_complete_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config.json"
            config.write_text("{}")
            output = Path(tmp) / "gate.json"
            with patch.object(gate, "check_transfer", return_value={"passed": True,
                              "status": "passed", "bad_value": object()}):
                code = gate.main(["--config", str(config), "--scores-dir", tmp,
                                  "--output", str(output)])
            self.assertEqual(code, 2)
            report = json.loads(output.read_text())
            self.assertIs(report["passed"], False)
            self.assertEqual(report["status"], "invalid_report")

    def test_invalid_input_writes_failed_report_and_returns_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_text("{}")
            output = root / "gate.json"
            with patch.object(gate, "check_transfer", side_effect=ValueError("bad input")):
                code = gate.main(["--config", str(config), "--scores-dir", tmp,
                                  "--output", str(output)])
            self.assertEqual(code, 2)
            report = json.loads(output.read_text())
            self.assertIs(report["passed"], False)
            self.assertEqual(report["status"], "invalid_input")
            self.assertIn("bad input", report["failures"][0])

    def test_existing_report_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "gate.json"
            output.write_text("original pass evidence")
            code = gate.main(["--config", str(Path(tmp) / "missing.json"),
                              "--scores-dir", tmp, "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertEqual(output.read_text(), "original pass evidence")


if __name__ == "__main__":
    unittest.main()
