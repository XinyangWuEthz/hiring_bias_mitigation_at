#!/usr/bin/env python3
"""Review an existing generated-TPR trajectory without inference or policy edits.

The first crossings reported here concern retained checkpoints. Missing early
checkpoints and changes between checks prevent a global first-crossing claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

SCOPES = ("full", "dev", "historical_remainder")


def first_crossing(rows, scope, threshold_pp, *, strict=False):
    def breached(row):
        delta = row["scopes"][scope]["delta_tpr_pp"]
        return delta < -threshold_pp if strict else delta <= -threshold_pp

    indices = [i for i, row in enumerate(rows) if breached(row)]
    first = indices[0] if indices else None
    previous = rows[first - 1] if first is not None and first > 0 else None
    return {
        "scope": scope,
        "threshold_pp": threshold_pp,
        "comparison": "<" if strict else "<=",
        "crossing_steps": [rows[i]["step"] for i in indices],
        "first_observed_crossing_step": rows[first]["step"] if first is not None else None,
        "preceding_retained_step": previous["step"] if previous else None,
        "preceding_retained_safe_step": previous["step"] if previous else None,
        "first_crossing_scope": "retained_checkpoints_only",
    }


def analyze(trajectory, check_every_steps=25):
    if not trajectory.get("complete") or trajectory.get("pending_steps"):
        raise ValueError("A complete registered trajectory is required for this policy review")
    rows = sorted(trajectory["rows"], key=lambda row: row["step"])
    steps = [row["step"] for row in rows]
    if not rows or steps[0] != 0 or len(set(steps)) != len(steps):
        raise ValueError("Expected one parent at step 0 and unique checkpoint steps")
    if check_every_steps <= 0:
        raise ValueError("Check spacing must be positive")
    for row in rows:
        for scope in SCOPES:
            metric = row["scopes"][scope]
            parent = rows[0]["scopes"][scope]
            if metric["invalid_n"] or metric["decided_n"] != metric["n"]:
                raise ValueError(f"Unusable decisions at {row['step']} / {scope}")
            if metric["reference_hire_n"] != parent["reference_hire_n"]:
                raise ValueError(f"Changed positive denominator at {row['step']} / {scope}")
            delta = metric["delta_tpr_pp"]
            tpr = metric["tpr_vs_gpt4o_pct"]
            if not math.isfinite(delta) or not math.isfinite(tpr):
                raise ValueError("Nonfinite TPR")
            if not math.isclose(delta, tpr - parent["tpr_vs_gpt4o_pct"], abs_tol=1e-10):
                raise ValueError("Parent-subtraction identity failed")
            if not math.isclose(delta, metric["gain_pp"] - metric["loss_pp"], abs_tol=1e-10):
                raise ValueError("Loss/gain identity failed")

    crossings = {
        "full_8pp_alarm": first_crossing(rows, "full", 8),
        "full_10pp_guardrail_failure": first_crossing(rows, "full", 10, strict=True),
        "dev_8pp_alarm": first_crossing(rows, "dev", 8),
        "dev_10pp_alarm": first_crossing(rows, "dev", 10),
        "remainder_8pp_alarm": first_crossing(rows, "historical_remainder", 8),
        "remainder_10pp_guardrail_failure": first_crossing(
            rows, "historical_remainder", 10, strict=True
        ),
    }
    full_failures = set(crossings["full_10pp_guardrail_failure"]["crossing_steps"])
    full_alarm_steps = set(crossings["full_8pp_alarm"]["crossing_steps"])
    dev_alarm_steps = set(crossings["dev_8pp_alarm"]["crossing_steps"])
    full_first = crossings["full_10pp_guardrail_failure"]["first_observed_crossing_step"]
    dev_first = crossings["dev_8pp_alarm"]["first_observed_crossing_step"]
    missing_checks = [
        step
        for step in range(check_every_steps, steps[-1] + 1, check_every_steps)
        if step not in steps
    ]
    row525 = next((row for row in rows if row["step"] == 525), None)
    new_stop_precedes525 = dev_first is not None and dev_first < 525
    full8_first = crossings["full_8pp_alarm"]["first_observed_crossing_step"]
    full8_already_fails10 = full8_first is not None and full8_first in full_failures
    missed_full10 = sorted(full_failures - dev_alarm_steps)
    curve = [
        {
            "step": row["step"],
            "source": row.get("source"),
            "scopes": {
                scope: {
                    key: row["scopes"][scope][key]
                    for key in (
                        "tpr_vs_gpt4o_pct",
                        "delta_tpr_pp",
                        "reference_hire_n",
                        "loss_count",
                        "gain_count",
                        "loss_pp",
                        "gain_pp",
                        "invalid_n",
                    )
                }
                for scope in SCOPES
            },
        }
        for row in rows
    ]
    result = {
        "schema": "checkpoint-trajectory-threshold-review-v1",
        "trajectory_complete_for_registered_retained_checkpoints": True,
        "points_including_parent": len(rows),
        "checkpoint_count": len(rows) - 1,
        "retained_steps": steps[1:],
        "curve": curve,
        "crossings": crossings,
        "sampling_limitations": {
            "check_every_optimizer_steps": check_every_steps,
            "unobserved_scheduled_steps": missing_checks,
            "global_first_crossing_known": False,
            "global_first_crossing_reason": (
                "Missing saved checkpoints and unobserved changes between checks. "
                "Complete means every registered retained checkpoint was scored."
            ),
            "previous_safe_checkpoint_scope": (
                "Latest retained point before the first observed breach; missing earlier "
                "checks may already have triggered an irreversible stop."
            ),
        },
        "dev_8pp_detection": {
            "first_full_10pp_failure_step": full_first,
            "first_dev_8pp_alarm_step": dev_first,
            "delay_optimizer_steps": (
                dev_first - full_first if dev_first is not None and full_first is not None else None
            ),
            "missed_full_10pp_failure_steps": missed_full10,
            "missed_full_8pp_alarm_steps": sorted(full_alarm_steps - dev_alarm_steps),
            "passes_detection_of_every_observed_full_10pp_failure": not missed_full10,
            "pointwise_miss_caveat": (
                "After the first irreversible dev stop, later misses describe endpoint "
                "sensitivity, not a second policy run. The earlier miss still matters."
            ),
        },
        "checkpoint_525": {
            "present": row525 is not None,
            "full_delta_tpr_pp": row525["scopes"]["full"]["delta_tpr_pp"] if row525 else None,
            "dev_delta_tpr_pp": row525["scopes"]["dev"]["delta_tpr_pp"] if row525 else None,
            "passes_original_full_10pp_guardrail": (
                row525["scopes"]["full"]["delta_tpr_pp"] >= -10 if row525 else None
            ),
            "earlier_observed_dev_8pp_stop_precedes_checkpoint": new_stop_precedes525,
            "eligible_under_frozen_dev_8pp_first_stop_rule": (
                False if new_stop_precedes525 else None
            ),
            "later_recovery_revokes_stop": False,
            "historical_fairness_conclusion": (
                "Existing matched-stability result remains exploratory; this TPR-only "
                "trajectory neither erases it nor independently confirms a fairness gain."
            ),
        },
        "full_8pp_buffer": {
            "first_alarm_step": full8_first,
            "full_10pp_guardrail_already_failed_at_first_alarm": full8_already_fails10,
            "check_spacing_and_2pp_buffer_guarantee_advance_warning": False,
            "reason": (
                "The first retained full 8pp alarm can already be beyond the 10pp guardrail; "
                "TPR can cross both lines between scheduled checks."
            ),
        },
        "recommendation": {
            "early_alarm_drop_pp": 8,
            "final_full_benchmark_guardrail_drop_pp": 10,
            "change_original_final_standard": False,
            "fixed_dev_8pp_status": (
                "failed_observed_full_guardrail_detection"
                if missed_full10
                else "no_observed_miss_not_independent_validation"
            ),
            "training_authorized": False,
            "relax_to_10_to_select_525": False,
            "next_policy_change_requires_new_freeze": True,
            "later_scope_or_frequency_tuning_is_independent_validation": False,
            "benchmark_previously_exposed": True,
            "historical_remainder_is_fresh_test": False,
        },
        "verification": {
            "all_rows_usable": True,
            "fixed_positive_denominators": True,
            "parent_subtraction_and_loss_gain_identities": True,
            "new_inference": False,
            "frozen_manifest_changed": False,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--check-every-steps", type=int, default=25)
    args = parser.parse_args()
    if args.trajectory.resolve() == args.out.resolve():
        parser.error("The output must not replace the trajectory input")
    payload = args.trajectory.read_bytes()
    result = analyze(json.loads(payload), args.check_every_steps)
    result["trajectory_source"] = {
        "path": str(args.trajectory),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "points": result["points_including_parent"],
                "crossings": result["crossings"],
                "dev_8pp_detection": result["dev_8pp_detection"],
                "recommendation": result["recommendation"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
