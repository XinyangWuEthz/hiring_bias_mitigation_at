#!/usr/bin/env python3
"""Descriptive SDT analysis of frozen late checkpoints and their weight average.

This reads existing generated decisions. It does not generate, train, alter the
benchmark, or replace the original matched-fairness comparison standard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if (ROOT / "src").is_dir():
    sys.path.insert(0, str(ROOT / "src"))

KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
STEPS = [425, 450, 475, 500, 525, 550, 571]
SINGLES = [f"step{step}" for step in STEPS]
PARENT = "SFT-v3"
SCOPES = ["full", "dev", "historical_remainder"]
NORMAL = NormalDist()


def sdt_counts(hits, misses, false_alarms, correct_rejections):
    """Hautus log-linear correction of every contingency cell, including nonextremes."""
    counts = [hits, misses, false_alarms, correct_rejections]
    if any(not np.isfinite(n) or n < 0 for n in counts):
        raise ValueError("SDT counts must be finite and nonnegative")
    positive, negative = hits + misses, false_alarms + correct_rejections
    if not positive or not negative:
        raise ValueError("Both reference labels require observations")
    h = (hits + 0.5) / (positive + 1)
    f = (false_alarms + 0.5) / (negative + 1)
    zh, zf = NORMAL.inv_cdf(h), NORMAL.inv_cdf(f)
    return {
        "hits": int(hits),
        "misses": int(misses),
        "false_alarms": int(false_alarms),
        "correct_rejections": int(correct_rejections),
        "reference_hire_n": int(positive),
        "reference_reject_n": int(negative),
        "tpr_pct": float(100 * hits / positive),
        "fpr_pct": float(100 * false_alarms / negative),
        "corrected_hit_rate": float(h),
        "corrected_false_alarm_rate": float(f),
        "criterion_c": -0.5 * (zh + zf),
        "dprime": zh - zf,
    }


def _bootstrap_sdt(totals):
    positive = totals[..., 0] + totals[..., 1]
    negative = totals[..., 2] + totals[..., 3]
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError("A candidate bootstrap draw has no positive or negative label")
    inverse = np.vectorize(NORMAL.inv_cdf, otypes=[float])
    zh = inverse((totals[..., 0] + 0.5) / (positive + 1))
    zf = inverse((totals[..., 2] + 0.5) / (negative + 1))
    return -0.5 * (zh + zf), zh - zf


def interval(values):
    return [float(v) for v in np.quantile(values, [0.025, 0.975], method="linear")]


def _summary(c, d):
    q = np.quantile(c, [0, 0.25, 0.5, 0.75, 1], method="linear")
    return {
        "criterion_c": dict(zip(["min", "q25", "median", "q75", "max"], map(float, q))),
        "dprime": {
            "min": float(min(d)),
            "median": float(np.median(d)),
            "mean": float(np.mean(d)),
            "max": float(max(d)),
        },
        "dprime_best_checkpoint": SINGLES[int(np.argmax(d))],
    }


def compare_average(point_c, point_d, bootstrap_c, bootstrap_d, average_index):
    """Jointly resampled comparisons against the seven fixed checkpoints at indices 1:8."""
    singles_c, singles_d = point_c[1:8], point_d[1:8]
    summary = _summary(singles_c, singles_d)
    avg_c, avg_d = float(point_c[average_index]), float(point_d[average_index])
    q = summary["criterion_c"]
    per_single = {}
    lower_bounds = []
    for index, label in enumerate(SINGLES, start=1):
        dc = bootstrap_c[:, average_index] - bootstrap_c[:, index]
        dd = bootstrap_d[:, average_index] - bootstrap_d[:, index]
        lower = float(np.quantile(dd, 0.05 / len(SINGLES), method="linear"))
        lower_bounds.append(lower)
        per_single[label] = {
            "delta_c": float(avg_c - point_c[index]),
            "delta_c_paired_ci95": interval(dc),
            "delta_dprime": float(avg_d - point_d[index]),
            "delta_dprime_paired_ci95": interval(dd),
            "delta_dprime_bonferroni_one_sided95_lower": lower,
        }
    combined = {}
    for name, reducer in (("max", np.max), ("median", np.median), ("mean", np.mean)):
        delta = avg_d - float(reducer(singles_d))
        boot = bootstrap_d[:, average_index] - reducer(bootstrap_d[:, 1:8], axis=1)
        combined[f"vs_seven_{name}"] = {"delta_dprime": delta, "paired_ci95": interval(boot)}
    return {
        "seven_checkpoint_summary": summary,
        "average_c_position": {
            "criterion_c": avg_c,
            "in_minmax": q["min"] <= avg_c <= q["max"],
            "in_iqr": q["q25"] <= avg_c <= q["q75"],
            "delta_from_median": avg_c - q["median"],
            "normalized_minmax_position": (
                (avg_c - q["min"]) / (q["max"] - q["min"]) if q["max"] != q["min"] else None
            ),
        },
        "vs_each_single": per_single,
        "dprime_comparisons": combined,
        "decision": {
            "c_in_seven_iqr_point": q["q25"] <= avg_c <= q["q75"],
            "dprime_ge_all7_point": avg_d >= float(max(singles_d)),
            "dprime_ge_median_point": avg_d >= float(np.median(singles_d)),
            "dprime_ge_mean_point": avg_d >= float(np.mean(singles_d)),
            "all7_dprime_bonferroni_lower_bounds_ge_zero": all(v >= 0 for v in lower_bounds),
            "noninferiority_margin": None,
            "ci_crossing_zero_establishes_noninferiority": False,
            "mechanism_proven": False,
            "fairness_gain_established": False,
            "cross_seed_or_training_dynamics_stability_established": False,
        },
    }


def analyze(
    raw_by_model,
    partitions,
    average_label="mean7",
    bootstrap_replicates=10000,
    seed=20261023,
    expected_prompt_count=31050,
):
    """Analyze DataFrames aligned to the unchanged full greedy benchmark.

    Required labels: SFT-v3, step425, step450, step475, step500, step525,
    step550, step571. Optional average_label adds paired comparisons. The same
    candidate multiplicities are applied to every model in a bootstrap draw.
    """
    from hiring_bias_mitigation.analysis.reference_metrics import (
        flip_rates,
        prepare,
        unstable_within_group,
    )

    labels = [PARENT, *SINGLES]
    if average_label in raw_by_model:
        labels.append(average_label)
    if set(raw_by_model) != set(labels):
        raise ValueError("Expected the parent and exactly seven preregistered single checkpoints")
    if bootstrap_replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    required = [*KEY, "candidate_id", "job_id", "reference_decision", "decision", "outcome"]
    parent = raw_by_model[PARENT]
    if len(parent) != expected_prompt_count or parent.duplicated(KEY).any():
        raise ValueError("Parent expected row count or unique keys mismatch")
    base = parent.set_index(KEY)
    aligned = {}
    for label in labels:
        raw = raw_by_model[label]
        if (
            any(c not in raw for c in required)
            or len(raw) != len(parent)
            or raw.duplicated(KEY).any()
        ):
            raise ValueError(f"Incomplete or duplicate raw rows: {label}")
        indexed = raw.set_index(KEY)
        if set(indexed.index) != set(base.index):
            raise ValueError(f"Row keys differ: {label}")
        indexed = indexed.loc[base.index]
        for field in ("candidate_id", "job_id", "reference_decision"):
            if not indexed[field].equals(base[field]):
                raise ValueError(f"Frozen {field} differs: {label}")
        if (
            not indexed.outcome.eq("decided").all()
            or not indexed.decision.isin(["hire", "reject"]).all()
        ):
            raise ValueError(f"SDT paired comparison requires all usable binary outputs: {label}")
        if not indexed.reference_decision.isin(["hire", "reject"]).all():
            raise ValueError("Nonbinary reference label")
        aligned[label] = indexed.reset_index()
    all_candidates = set(parent.candidate_id)
    if set(partitions) != {"dev", "historical_remainder"}:
        raise ValueError("Expected frozen dev and historical_remainder partitions")
    if partitions["dev"] & partitions["historical_remainder"]:
        raise ValueError("Candidate partitions overlap")
    if partitions["dev"] | partitions["historical_remainder"] != all_candidates:
        raise ValueError("Candidate partitions do not exactly cover benchmark")
    scopes = {"full": all_candidates, **partitions}
    result = {
        "schema": "checkpoint-average-sdt-v1",
        "status": "complete" if average_label in labels else "awaiting_average_inference",
        "new_inference_performed_by_analysis": False,
        "average_raw_present": average_label in labels,
        "model_order": labels,
        "contract": {
            "single_steps": STEPS,
            "primary_scope": "full",
            "secondary_scopes": SCOPES[1:],
            "criterion_c": "-0.5 * (Phi_inverse(H) + Phi_inverse(F))",
            "dprime": "Phi_inverse(H) - Phi_inverse(F)",
            "rate_correction": (
                "H=(hits+0.5)/(N_hire+1); F=(false_alarms+0.5)/(N_reject+1), all rates"
            ),
            "tpr_fpr_reports_use_uncorrected_rates": True,
            "includes_attribute_free": True,
            "middle_c_definition": "Within the fixed seven checkpoints' inclusive c IQR",
            "iqr_quantile_method": "linear",
            "strict_dprime_not_worse_definition": "average dprime >= max(seven single dprimes)",
            "noninferiority_margin": None,
            "bootstrap": {
                "unit": "candidate_id",
                "paired_across_all_models": True,
                "replicates": bootstrap_replicates,
                "rng": "PCG64",
                "seed": seed,
                "scope_seeds": "SeedSequence([seed, index of full/dev/historical_remainder])",
                "ci95_quantiles": [0.025, 0.975],
                "quantile_method": "linear",
                "all7_one_sided_lower_quantile": 0.05 / 7,
                "interpretation": "Candidate sensitivity on development data only",
            },
            "published_standard_changed": False,
        },
        "scopes": {},
        "limitations": [
            "The benchmark labels are GPT-4o reference decisions, not verified hiring outcomes.",
            "Gaussian equal-variance SDT is assumed; one binary point is not a full ROC.",
            "Observed c,dprime do not identify a fixed criterion or zero-mean checkpoint noise.",
            "Candidate bootstrap does not model all dependence from shared jobs.",
            "Checkpoints are correlated states of one training run, not independent seeds.",
            "Nonlinear averaging and merge retention can affect results.",
            "Raw set instability is not the original matched-operating-point fairness comparison.",
        ],
    }
    for scope_index, scope in enumerate(SCOPES):
        candidates = sorted(scopes[scope])
        counts = np.zeros((len(candidates), len(labels), 4), dtype=np.int64)
        candidate_index = {candidate: i for i, candidate in enumerate(candidates)}
        metrics = {}
        for model_index, label in enumerate(labels):
            raw = aligned[label]
            frame = raw[raw.candidate_id.isin(candidates)].reset_index(drop=True)
            reference_hire = frame.reference_decision.eq("hire").to_numpy()
            predicted_hire = frame.decision.eq("hire").to_numpy()
            categories = np.where(
                reference_hire, np.where(predicted_hire, 0, 1), np.where(predicted_hire, 2, 3)
            )
            np.add.at(
                counts[:, model_index, :],
                (frame.candidate_id.map(candidate_index).to_numpy(), categories),
                1,
            )
            value = sdt_counts(*counts[:, model_index, :].sum(axis=0))
            prepared = prepare(frame)
            instability, complete, expected = unstable_within_group(prepared)
            value.update(
                {
                    "row_count": len(frame),
                    "candidate_count": len(candidates),
                    "pair_count": int(frame.pair_id.nunique()),
                    "invalid_count": 0,
                    "agreement_pct": 100
                    * float(frame.decision.eq(frame.reference_decision).mean()),
                    "raw_unstable_within_group_pct": instability,
                    "complete_within_group_sets": complete,
                    "expected_within_group_sets": expected,
                    "attribute_flip_rates": flip_rates(prepared),
                }
            )
            metrics[label] = value
        point_c = np.array([metrics[label]["criterion_c"] for label in labels])
        point_d = np.array([metrics[label]["dprime"] for label in labels])
        for value in metrics.values():
            value["delta_tpr_vs_sft_pp"] = value["tpr_pct"] - metrics[PARENT]["tpr_pct"]
            value["delta_agreement_vs_sft_pp"] = (
                value["agreement_pct"] - metrics[PARENT]["agreement_pct"]
            )
            value["passes_original_tpr_10pp_guardrail"] = (
                value["delta_tpr_vs_sft_pp"] >= -10 if scope == "full" else None
            )
            value["passes_original_agreement_5pp_guardrail"] = (
                value["delta_agreement_vs_sft_pp"] >= -5 if scope == "full" else None
            )
        result["scopes"][scope] = {
            "metrics_by_model": metrics,
            "seven_checkpoint_summary": _summary(point_c[1:8], point_d[1:8]),
            "comparisons": None,
        }
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, scope_index])))
        n = len(candidates)
        weights = rng.multinomial(n, [1 / n] * n, size=bootstrap_replicates)
        totals = (weights @ counts.reshape(n, -1)).reshape(bootstrap_replicates, len(labels), 4)
        bc, bd = _bootstrap_sdt(totals)
        for model_index, label in enumerate(labels):
            metrics[label]["criterion_c_candidate_ci95"] = interval(bc[:, model_index])
            metrics[label]["dprime_candidate_ci95"] = interval(bd[:, model_index])
        if average_label in labels:
            result["scopes"][scope]["comparisons"] = compare_average(
                point_c, point_d, bc, bd, len(labels) - 1
            )
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(output, path):
    rows = []
    for scope, scoped in output["scopes"].items():
        for label, metric in scoped["metrics_by_model"].items():
            row = {"scope": scope, "model": label}
            row.update({key: value for key, value in metric.items()
                        if not isinstance(value, (dict, list))})
            row["c_ci95_low"], row["c_ci95_high"] = metric["criterion_c_candidate_ci95"]
            row["dprime_ci95_low"], row["dprime_ci95_high"] = metric["dprime_candidate_ci95"]
            rows.append(row)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--average-raw", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = {PARENT: args.trajectory_dir / "steps/step-0000/raw.parquet"}
    files.update(
        {
            f"step{step}": args.trajectory_dir / "steps" / f"step-{step:04d}" / "raw.parquet"
            for step in STEPS
        }
    )
    if args.average_raw:
        files["mean7"] = args.average_raw
    split = json.loads(args.split.read_text())
    partitions = {name: set(spec["candidate_ids"]) for name, spec in split["splits"].items()}
    output = analyze({label: pd.read_parquet(path) for label, path in files.items()}, partitions)
    output["sources"] = {
        "split": {"path": str(args.split), "sha256": sha(args.split)},
        "raw_files": {
            label: {"path": str(path), "sha256": sha(path)} for label, path in files.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    write_csv(output, args.out.with_suffix(".csv"))
    print(
        json.dumps(
            {"output": str(args.out), "status": output["status"], "full": output["scopes"]["full"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
