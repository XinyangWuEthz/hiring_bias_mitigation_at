"""Fail-closed, predeclared held-out monitor transfer gate; never starts training.

The config freezes all input hashes, eight models and thresholds before GPU scoring.
Exit 0 means every registered check passed; exit 2 means calibration failed or an
input was malformed. A fresh JSON report is always written when possible.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


class GateInputError(ValueError):
    """Registered inputs cannot safely be interpreted."""


def require(condition, message):
    if not condition:
        raise GateInputError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rooted(root, value):
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def finite_column(frame, column, label):
    require(column in frame, f"{label}: missing {column}")
    try:
        values = frame[column].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise GateInputError(f"{label}: nonnumeric {column}") from exc
    require(np.isfinite(values).all(), f"{label}: nonfinite {column}")
    return values


def valid_keys(frame, keys, label):
    require(set(keys).issubset(frame.columns), f"{label}: missing join keys")
    require(not frame[keys].isna().any().any(), f"{label}: null join keys")
    require(not frame.duplicated(keys).any(), f"{label}: duplicate join keys")


def validate_score(frame, spec, root, rows_path, digest, n):
    label = spec["label"]
    require(len(frame) == n, f"{label}: expected {n} scores, got {len(frame)}")
    margins = finite_column(frame, "margin", label)
    require("row_index" in frame, f"{label}: missing row_index")
    require(np.array_equal(frame["row_index"].to_numpy(), np.arange(n)),
            f"{label}: row_index is not the original ordered row sequence")
    for column in ("prompts_sha1", "model", "rows"):
        require(column in frame and frame[column].notna().all(),
                f"{label}: missing {column}")
    require(frame["prompts_sha1"].eq(digest).all(), f"{label}: inconsistent prompt digest")
    expected_model = rooted(root, spec["model"])
    require(all(rooted(root, p) == expected_model for p in frame["model"]),
            f"{label}: unexpected model path")
    require(all(rooted(root, p) == rows_path for p in frame["rows"]),
            f"{label}: unexpected monitor rows path")
    return margins


def evaluate_arm(tpr_monitor, tpr_margin, tpr_generated, limits):
    """Strict optimism/stop boundaries; inclusive 2 pp absolute-error boundary."""
    values = (tpr_monitor, tpr_margin, tpr_generated)
    require(all(math.isfinite(float(x)) for x in values), "nonfinite TPR comparison")
    error = float(tpr_monitor - tpr_margin)
    optimism = float(tpr_monitor - tpr_generated)
    needs_stop = min(tpr_margin, tpr_generated) < -limits["tpr_stop_pp"]
    # CSV decimal percentages can land a few ulps below an exact boundary after
    # subtraction. Do not accidentally accept exactly 2 pp optimism as <2 pp.
    optimism_at_limit = math.isclose(optimism, limits["max_optimism_generated_tpr_pp"],
                                     rel_tol=0., abs_tol=1e-10)
    checks = {
        "margin_error_within_limit": abs(error) <= limits["max_abs_tpr_margin_error_pp"]
        or math.isclose(abs(error), limits["max_abs_tpr_margin_error_pp"],
                        rel_tol=0., abs_tol=1e-10),
        "generated_optimism_below_limit": optimism < limits["max_optimism_generated_tpr_pp"]
        and not optimism_at_limit,
        "historical_drop_triggers_stop": not needs_stop or tpr_monitor < -limits["tpr_stop_pp"],
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    return {"tpr_margin_error_pp": error, "tpr_generated_optimism_pp": optimism,
            "historical_stop_required": bool(needs_stop), "checks": checks,
            "passed": bool(all(checks.values()))}


def check_transfer(config, scores_dir, calibration=None):
    """Read only registered inputs; use the unchanged production projection functions."""
    mc = calibration or importlib.import_module("monitor_calibration")
    require(Path(config["output_root"]).is_absolute(), "output_root must be absolute")
    root = Path(config["output_root"]).resolve()
    require(isinstance(config["experiment_id"], str) and config["experiment_id"],
            "missing experiment_id")
    arms, parent_spec = config["arms"], config["parent"]
    require(len(arms) == 7, "expected seven registered comparison arms")
    specs = [parent_spec, *arms]
    labels = [s["label"] for s in specs]
    runs = [s["run"] for s in specs]
    require(len(set(labels)) == len(set(runs)) == 8, "model labels and runs must be unique")
    require(all(isinstance(x, str) and x and Path(x).name == x for x in labels + runs),
            "labels and runs must be plain nonempty names")
    limits = config["limits"]
    for key in ("max_abs_tpr_margin_error_pp", "max_optimism_generated_tpr_pp", "tpr_stop_pp"):
        require(math.isfinite(float(limits[key])) and limits[key] > 0,
                f"invalid threshold {key}")
    expected = {k: config[k] for k in ("expected_monitor_rows", "expected_monitor_candidates",
                                      "expected_train_rows")}
    require(all(isinstance(n, int) and not isinstance(n, bool) and n > 0
                for n in expected.values()), "expected counts must be positive integers")
    paths = {k: rooted(root, config[k]) for k in ("rows", "train", "validation")}
    raw_path = root / "outputs/raw" / f"{parent_spec['run']}.parquet"
    margin_paths = {s["label"]: root / "outputs/operating_point/v2" /
                    f"{s['run']}--adapter.parquet" for s in specs}
    metrics_path = rooted(root, config["reference_metrics"])
    required_sources = set(paths.values()) | {raw_path, metrics_path} | set(margin_paths.values())
    hashes = config["source_sha256"]
    require(isinstance(hashes, dict) and hashes, "missing source hashes")
    registered = {}
    for name, expected_hash in hashes.items():
        path = rooted(root, name)
        require(path not in registered, f"duplicate hash registration: {path}")
        require(isinstance(expected_hash, str) and len(expected_hash) == 64,
                f"invalid SHA256 registration: {name}")
        actual = sha256(path)
        require(actual == expected_hash, f"stale source hash: {name}")
        registered[path] = actual
    require(required_sources.issubset(registered),
            "missing registered source hashes: " +
            ", ".join(str(p) for p in sorted(required_sources - registered.keys())))

    monitor, train, validation = [pd.read_parquet(paths[k]) for k in ("rows", "train", "validation")]
    for name, frame in (("monitor", monitor), ("train", train), ("validation", validation)):
        for column in ("candidate_id", "pair_id", "prompt"):
            require(column in frame and frame[column].notna().all(), f"{name}: missing {column}")
        require(frame["prompt"].map(lambda p: isinstance(p, str) and bool(p)).all(),
                f"{name}: invalid prompt")
        require(not frame["prompt"].duplicated().any(), f"{name}: duplicate prompt")
    require(len(monitor) == expected["expected_monitor_rows"], "monitor row count mismatch")
    require(len(train) == expected["expected_train_rows"], "training row count mismatch")
    require(monitor["candidate_id"].nunique() == expected["expected_monitor_candidates"],
            "monitor candidate count mismatch")
    require(not (set(monitor["candidate_id"]) & set(train["candidate_id"])),
            "monitor overlaps training candidates")
    require(not (set(validation["candidate_id"]) & set(train["candidate_id"])),
            "validation overlaps training candidates")
    digest = mc.prompt_digest(monitor["prompt"].tolist())

    scores, score_hashes = {}, {}
    for spec in specs:
        score_path = Path(scores_dir) / f"{spec['label']}.parquet"
        score_hashes[spec["label"]] = sha256(score_path)
        scores[spec["label"]] = validate_score(pd.read_parquet(score_path), spec, root,
                                               paths["rows"], digest, len(monitor))

    raw = pd.read_parquet(raw_path)
    require("condition" in raw and "reference_decision" in raw, "parent raw missing reference")
    raw = raw[raw["condition"] != "attr_free"].copy()
    valid_keys(raw, mc.KEY, "parent benchmark")
    require(len(raw) > 0, "empty parent benchmark")
    references = raw["reference_decision"].astype(str).str.lower()
    require(references.isin(["hire", "reject"]).all(), "invalid benchmark reference decisions")
    require(references.nunique() == 2, "benchmark requires both reference decisions")
    raw["gpt4o_hire"] = references.eq("hire")
    benchmark_margins = {}
    for spec in specs:
        frame = pd.read_parquet(margin_paths[spec["label"]])
        valid_keys(frame, mc.KEY, f"{spec['label']} benchmark margins")
        finite_column(frame, "margin", f"{spec['label']} benchmark margins")
        aligned = raw[mc.KEY].merge(frame[[*mc.KEY, "margin"]], on=mc.KEY,
                                    how="left", validate="one_to_one", sort=False)
        require(len(aligned) == len(raw), f"{spec['label']}: unmatched benchmark length")
        benchmark_margins[spec["label"]] = finite_column(aligned, "margin", spec["label"])
    parent_bench = benchmark_margins[parent_spec["label"]]
    hire = raw["gpt4o_hire"].to_numpy(dtype=bool)
    shares = mc.bench_shares(parent_bench, hire)
    parent_scores = scores[parent_spec["label"]]
    bins = mc.margin_bins(parent_scores, shares["edges"])
    weights_hire = np.asarray(shares["share_ref_hire"], dtype=float)
    weights_reject = np.asarray(shares["share_ref_reject"], dtype=float)
    require(len(weights_hire) == len(weights_reject) == 8, "expected eight frozen margin bins")
    require(np.isfinite(weights_hire).all() and np.isfinite(weights_reject).all(),
            "nonfinite calibration weights")
    bin_rows = []
    failures = []
    for b in range(8):
        mask = bins == b
        n = int(mask.sum())
        candidates = int(monitor.loc[mask, "candidate_id"].nunique())
        covered = bool(n > 0 or (weights_hire[b] == 0 and weights_reject[b] == 0))
        if not covered:
            failures.append(f"positive-weight parent-margin bin {b} has no monitor rows")
        bin_rows.append({"bin": b, "rows": n, "candidates": candidates,
                         "share_ref_hire": float(weights_hire[b]),
                         "share_ref_reject": float(weights_reject[b]), "covered": covered})

    metrics = pd.read_csv(metrics_path)
    require({"run", "tpr_vs_gpt4o_pct", "fpr_vs_gpt4o_pct"}.issubset(metrics.columns),
            "reference metrics missing required columns")
    reference = {}
    for spec in specs:
        selected = metrics[metrics["run"] == spec["run"]]
        require(len(selected) == 1, f"{spec['label']}: need exactly one reference-metrics row")
        tpr = float(finite_column(selected, "tpr_vs_gpt4o_pct", spec["label"])[0])
        fpr = float(finite_column(selected, "fpr_vs_gpt4o_pct", spec["label"])[0])
        require(0 <= tpr <= 100 and 0 <= fpr <= 100, "generated rates must be percentages")
        reference[spec["label"]] = (tpr, fpr)
    parent_generated = reference[parent_spec["label"]]
    arm_results = []
    for spec in arms:
        label = spec["label"]
        projection = mc.project_from_margins(
            parent_scores, scores[label], shares["edges"], weights_hire, weights_reject,
            {"hire_rate_ref_hire": shares["parent_tpr"],
             "hire_rate_ref_reject": shares["parent_fpr"],
             "reference_hire_share": shares["reference_hire_share"]})
        current_bench = benchmark_margins[label]
        actual_tpr = float(100 * ((current_bench[hire] > 0).mean() - (parent_bench[hire] > 0).mean()))
        actual_fpr = float(100 * ((current_bench[~hire] > 0).mean() - (parent_bench[~hire] > 0).mean()))
        monitor_tpr, monitor_fpr = 100 * projection["delta_tpr"], 100 * projection["delta_fpr"]
        generated_tpr = reference[label][0] - parent_generated[0]
        generated_fpr = reference[label][1] - parent_generated[1]
        result = {"label": label, "run": spec["run"], "tpr_monitor_pp": monitor_tpr,
                  "tpr_benchmark_margin_pp": actual_tpr, "tpr_benchmark_generated_pp": generated_tpr,
                  "fpr_monitor_pp": monitor_fpr, "fpr_benchmark_margin_pp": actual_fpr,
                  "fpr_benchmark_generated_pp": generated_fpr,
                  "fpr_margin_error_pp": monitor_fpr - actual_fpr,
                  **evaluate_arm(monitor_tpr, actual_tpr, generated_tpr, limits)}
        require(math.isfinite(float(monitor_fpr)), f"{label}: nonfinite projected FPR")
        if not result["passed"]:
            failures.extend(f"{label}: {check}" for check, passed in result["checks"].items()
                            if not passed)
        arm_results.append(result)
    return {"experiment_id": config["experiment_id"], "passed": not failures,
            "status": "passed" if not failures else "calibration_failed", "failures": failures,
            "limits": limits, "calibration": shares, "parent_margin_bins": bin_rows,
            "monitor_rows": len(monitor), "monitor_candidates": int(monitor["candidate_id"].nunique()),
            "training_rows": len(train), "validation_rows": len(validation),
            "monitor_prompts_sha1": digest, "source_sha256": hashes, "score_sha256": score_hashes,
            "arms": arm_results,
            "caveats": ["Empirical transfer gate; does not guarantee a future training trajectory.",
                        "Rows share candidates; row count is not an independent sample count.",
                        "FPR errors are diagnostic, not acceptance criteria.",
                        "Generated TPR is rounded in the frozen reference metrics CSV."]}


def json_scalar(value):
    """NumPy computations produce scalar booleans/integers as well as floats."""
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def publish_report(output, report):
    """Serialize completely, then publish atomically without replacing any old report."""
    text = json.dumps(report, indent=2, allow_nan=False, default=json_scalar) + "\n"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link creation is atomic and fails if output already exists.
        # Unlike replace(), it cannot overwrite a report created by a concurrent process.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--scores-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        print(f"Cannot create fresh gate report: {args.output} already exists", file=sys.stderr)
        return 2
    provenance = {"config": str(args.config.resolve()),
                  "scores_dir": str(args.scores_dir.resolve()),
                  "computed_at": datetime.now(timezone.utc).isoformat()}
    try:
        provenance["config_sha256"] = sha256(args.config)
        config = json.loads(args.config.read_text())
        report = check_transfer(config, args.scores_dir)
    except Exception as exc:
        report = {"passed": False, "status": "invalid_input",
                  "failures": [f"{type(exc).__name__}: {exc}"]}
    report.update(provenance)
    try:
        publish_report(args.output, report)
    except (TypeError, ValueError) as exc:
        # Serialization failures also yield a complete failure report, never partial JSON.
        report = {**provenance, "passed": False, "status": "invalid_report",
                  "failures": [f"{type(exc).__name__}: {exc}"]}
        try:
            publish_report(args.output, report)
        except OSError as write_exc:
            print(f"Cannot create fresh gate report: {write_exc}", file=sys.stderr)
            return 2
    except OSError as exc:
        print(f"Cannot create fresh gate report: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"passed": bool(report["passed"]), "status": report["status"],
                      "report": str(args.output), "failures": report.get("failures", [])}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
