"""CPU checks for terminal selection, protocol identity, and diagnostic-failure gates."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location(
    "pilot_results_tested", ROOT / "scripts/grpo_low_lr_pilot_results.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def frame():
    return pd.DataFrame(
        [
            {
                "prompt_id": f"prompt-{i}",
                "pair_id": f"pair-{i}",
                "candidate_id": f"candidate-{i // 2}",
                "job_id": f"job-{i}",
                "lang": "en",
                "protected_group": "gender",
                "protected_attr": "female",
                "condition": "explicit",
                "reference_decision": "hire" if i < 2 else "reject",
                "cv": f"cv-{i}",
                "job_description": f"job text {i}",
                "prompt": f"frozen prompt {i}",
            }
            for i in range(4)
        ]
    )


def raw(expected, decisions=("hire", "reject", "hire", "reject")):
    from hiring_bias_mitigation.eval.parsing import parse_frame

    outputs = [
        json.dumps({"decision": value, "reason": "Saved test output"}) for value in decisions
    ]
    return parse_frame(expected.drop(columns="prompt").assign(raw_output=outputs))


def state(step=500, stop=False):
    return {
        "normal_train_end": True,
        "startup_gate_passed": True,
        "stop": stop,
        "stop_step": step if stop else None,
        "last_step": step,
        "terminal_checkpoint": f"/pilot/checkpoint-{step}",
        "last_safe_step": step - 5 if stop else step,
    }


class PilotResultTests(unittest.TestCase):
    def test_full_tpr_guardrail_includes_exact_ten_pp_boundary(self):
        passed = M.count_guardrails(4940, 5975, 10350, 25000, 25584, 31050)
        failed = M.count_guardrails(4939, 5975, 10350, 25000, 25584, 31050)
        self.assertTrue(passed["passes_original_tpr_10pp_guardrail"])
        self.assertFalse(failed["passes_original_tpr_10pp_guardrail"])

    def test_agreement_guardrail_includes_exact_five_pp_boundary(self):
        self.assertTrue(
            M.count_guardrails(50, 50, 100, 750, 800, 1000)[
                "passes_original_agreement_5pp_guardrail"
            ]
        )
        self.assertFalse(
            M.count_guardrails(50, 50, 100, 749, 800, 1000)[
                "passes_original_agreement_5pp_guardrail"
            ]
        )

    def test_terminal_is_fixed_not_best_checkpoint(self):
        value = state()
        value["best_checkpoint"] = "/pilot/checkpoint-475"
        self.assertEqual(M.terminal_step(value), 500)

    def test_early_stop_and_later_recovery_cannot_be_erased(self):
        self.assertEqual(M.terminal_step(state(435, True)), 435)
        recovered = state()
        recovered["stop_step"] = 435
        with self.assertRaisesRegex(ValueError, "cannot be cleared"):
            M.terminal_step(recovered)
        continued = state(435, True)
        continued["last_step"] = 500
        with self.assertRaisesRegex(ValueError, "Terminal event"):
            M.terminal_step(continued)

    def test_abnormal_or_failed_startup_does_not_finalize(self):
        for field in ("normal_train_end", "startup_gate_passed"):
            value = state()
            value[field] = False
            with self.assertRaisesRegex(ValueError, "normally ended"):
                M.terminal_step(value)

    def test_exact_decision_gate_aligns_keys_and_finds_one_disagreement(self):
        expected = frame()
        previous = raw(expected)
        same = previous.sample(frac=1, random_state=42).reset_index(drop=True)
        self.assertTrue(M.exact_decision_check(same, previous, expected)["exact_match"])
        current = raw(expected, ("hire", "hire", "hire", "reject"))
        check = M.exact_decision_check(current, previous, expected)
        self.assertFalse(check["exact_match"])
        self.assertEqual(check["mismatch_count"], 1)
        self.assertEqual(check["mismatch_sample"][0]["prompt_id"], "prompt-1")

    def test_relabeled_candidate_or_duplicate_row_rejected(self):
        expected = frame()
        changed = raw(expected)
        changed.loc[0, "candidate_id"] = "different-candidate"
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            M.align_raw(changed, expected)
        duplicate = pd.concat([raw(expected).iloc[:3], raw(expected).iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique key"):
            M.align_raw(duplicate, expected)

    def test_callback_text_is_reparsed_and_missing_ids_rejected(self):
        expected = frame()
        decisions = raw(expected)[["prompt_id", "decision", "raw_output"]]
        M.callback_raw(decisions.iloc[::-1], expected)
        forged = decisions.copy()
        forged.loc[0, "decision"] = "reject"
        with self.assertRaisesRegex(ValueError, "official parser"):
            M.callback_raw(forged, expected)
        with self.assertRaisesRegex(ValueError, "exact frozen"):
            M.callback_raw(decisions.iloc[:-1], expected)

    def test_invalid_decision_cannot_enter_sdt(self):
        expected = frame()
        invalid = raw(expected)
        invalid.loc[0, "outcome"] = "invalid"
        with self.assertRaisesRegex(ValueError, "Unusable"):
            M.align_raw(invalid, expected)

    def test_retention_and_chunk_identity_fail_closed(self):
        M.validate_retention([{"kept_fraction": 0.8}])
        for value in (0.7999, float("nan"), None, True):
            with self.assertRaises(ValueError):
                M.validate_retention([{"kept_fraction": value}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunk.json"
            path.write_text(
                json.dumps(
                    {
                        "binding_sha256": "run",
                        "model_binding": "model",
                        "start": 0,
                        "outputs": ["one"],
                    }
                )
            )
            self.assertEqual(M.read_chunk(path, "run", "model", 0, 1), ["one"])
            with self.assertRaisesRegex(ValueError, "different model"):
                M.read_chunk(path, "run", "another-model", 0, 1)

    def test_fixed_grid_range_and_early_stop_censoring(self):
        def curve(values):
            return {
                step: {
                    scope: {"tpr_pct": value, "criterion_c": value / 100, "dprime": value / 10}
                    for scope in M.SCOPES
                }
                for step, value in zip(M.GRID, values)
            }

        pilot, original = curve([60, 61, 59, 60]), curve([60, 60, 49, 73])
        complete = M.curve_summary(pilot, original, False)
        self.assertTrue(complete["complete_fixed_grid"])
        tpr = complete["scopes"]["full"]["tpr_pct"]
        self.assertEqual(tpr["low_lr"]["range"], 2)
        self.assertEqual(tpr["low_lr"]["total_absolute_adjacent_change"], 4)
        self.assertFalse(complete["learning_rate_causal_effect_established"])
        stopped = M.curve_summary(pilot, original, True)
        self.assertFalse(stopped["complete_fixed_grid"])
        self.assertIsNone(stopped["scopes"])
        self.assertTrue(stopped["stop_latch_preserved"])

    def test_outer_freeze_rejects_changed_training_config(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            config, monitor = out / "pilot.yaml", out / "monitor.json"
            config.write_text("learning_rate: 3e-6\n")
            monitor.write_text("{}\n")
            outer = {
                "schema": "grpo-low-lr-pilot-run-v1",
                "training_authorized": True,
                "bounds": {
                    "start_step": 425,
                    "max_step": 500,
                    "learning_rate": 3e-6,
                    "max_new_optimizer_steps": 75,
                },
                "config": M.T.fingerprint(config),
                "monitor_manifest": M.T.fingerprint(monitor),
                "code": [M.T.fingerprint(Path(M.__file__))],
                "prepared_commit": "fixed-commit",
            }
            (out / "run_manifest.json").write_text(json.dumps(outer))
            settings = {"manifest_sha256": M.T.sha(monitor)}
            M.validate_outer(out, config, settings)
            config.write_text("learning_rate: 3e-5\n")
            with self.assertRaisesRegex(ValueError, "frozen input changed"):
                M.validate_outer(out, config, settings)

    def test_monitor_history_cannot_skip_check_or_continue_after_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            contract = {"identity_sha256": "identity", "manifest_sha256": "manifest"}
            records = []
            for step in (425, 430):
                folder = run / "full_benchmark_monitor" / f"step-{step:04d}"
                folder.mkdir(parents=True)
                decisions = folder / "decisions.parquet"
                decisions.write_text("fingerprint-only fixture")
                receipt = {"decisions_file": M.T.fingerprint(decisions)}
                record = {
                    "step": step,
                    "identity_sha256": "identity",
                    "manifest_sha256": "manifest",
                    "valid": True,
                    "stop": step == 430,
                    "positive": receipt,
                    "full": receipt,
                }
                (folder / "metrics.json").write_text(json.dumps(record))
                records.append(record)
            history = run / "full_benchmark_monitor.jsonl"
            history.write_text("\n".join(json.dumps(record) for record in records))
            final = {**state(430, True), "history_checks": 2}
            self.assertEqual(len(M.load_history(run, final, contract)), 2)
            history.write_text(json.dumps(records[1]))
            final["history_checks"] = 1
            with self.assertRaisesRegex(ValueError, "Missing"):
                M.load_history(run, final, contract)

    def test_terminal_mismatch_is_saved_without_interpretation_or_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            expected = frame()
            baseline = raw(expected)
            decisions = out / "callback.parquet"
            baseline[["prompt_id", "decision", "raw_output"]].to_parquet(decisions, index=False)
            changed = raw(expected, ("hire", "hire", "hire", "reject"))
            raw_path = out / "raw.parquet"
            changed.to_parquet(raw_path, index=False)
            (out / "terminal_merge_receipt.json").write_text("{}")
            inputs = {
                "out": out,
                "expected": expected,
                "binding_sha256": "binding",
                "records": [
                    {"step": step, "full": {"decisions_file": M.T.fingerprint(decisions)}}
                    for step in (425, 500)
                ],
                "cached_raw": {"step425": baseline},
                "step": 500,
                "terminal": out / "checkpoint-500",
                "state": state(),
                "files": [],
                "contract": {
                    "identity_sha256": "identity",
                    "manifest_sha256": "manifest",
                    "manifest": {"source_trajectory": {"cached_models": {}}},
                },
            }
            with (
                patch.object(M, "load_inputs", return_value=inputs),
                patch.object(M, "prepare_merged", return_value=(out / "merged", {})),
                patch.object(
                    M,
                    "generate_terminal",
                    return_value=(changed, M.T.fingerprint(raw_path), "model"),
                ) as generate,
                patch.object(M, "scope_metrics", side_effect=AssertionError("must not interpret")),
            ):
                with self.assertRaisesRegex(ValueError, "decisions differ"):
                    M.finalize(out / "config.yaml")
                result = json.loads((out / "pilot_results.json").read_text())
                self.assertEqual(result["status"], "diagnostic_failed")
                self.assertFalse(result["interpretation_allowed"])
                self.assertEqual(result["callback_terminal_exact_check"]["mismatch_count"], 1)
                self.assertNotIn("trajectory", result)
                with self.assertRaisesRegex(ValueError, "Prior diagnostic failure"):
                    M.finalize(out / "config.yaml")
                self.assertEqual(generate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
