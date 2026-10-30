"""Explain a failed monitor transfer using existing scores, without fitting anything.

The three main terms sum to the production TPR projection error: cohort transfer
at pooled labels, benchmark label pooling, and four-decimal weight rounding.
This diagnostic does not replace or revise the preregistered acceptance gate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
EDGES = np.array([-4.595, -2.197, -.847, 0., .847, 2.197, 4.595])
TARGETS = {"DAPO-v3-all", "GRPO-v3-all", "GRPO-v3-all-step525"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subset(frame, parent, current, mask):
    n = int(mask.sum())
    if not n:
        return {"rows": 0, "candidates": 0 if "candidate_id" in frame else None,
                "pairs": 0, "reference_hire_share": None, "parent_hire_rate": None,
                "arm_hire_rate": None, "delta_hire_pp": None, "lost_hire_rows": 0,
                "gained_hire_rows": 0, "parent_margin_quantiles": None}
    before, after = parent[mask] > 0, current[mask] > 0
    return {"rows": n,
            "candidates": int(frame.loc[mask, "candidate_id"].nunique())
                          if "candidate_id" in frame else None,
            "pairs": int(frame.loc[mask, "pair_id"].nunique()),
            "reference_hire_share": float(frame.loc[mask, "reference_decision"].eq("hire").mean()),
            "parent_hire_rate": float(before.mean()), "arm_hire_rate": float(after.mean()),
            "delta_hire_pp": float(100 * (after.mean() - before.mean())),
            "lost_hire_rows": int((before & ~after).sum()),
            "gained_hire_rows": int((~before & after).sum()),
            "parent_margin_quantiles": [float(x) for x in np.quantile(parent[mask], [0, .25, .5, .75, 1])]}


def strata(frame, parent, current, mask):
    hires = frame["reference_decision"].eq("hire").to_numpy()
    return {"all": subset(frame, parent, current, mask),
            "reference_hire": subset(frame, parent, current, mask & hires),
            "reference_reject": subset(frame, parent, current, mask & ~hires)}


def weighted(records, cohort, label, weights):
    result = 0.
    for record, weight in zip(records, weights):
        value = record[cohort][label]["delta_hire_pp"]
        if weight and value is None:
            return None
        if value is not None:
            result += weight * value
    return float(result)


def diagnose(config_path, scores_dir):
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    hashes = {}

    def source(relative):
        path = Path(relative)
        path = path if path.is_absolute() else root / path
        key = str(path.relative_to(root))
        actual = sha256(path)
        if config["source_sha256"].get(key) != actual:
            raise ValueError("Unregistered or changed source: " + key)
        hashes[str(path)] = actual
        return pd.read_parquet(path)

    def scores(label, monitor):
        path = scores_dir / (label + ".parquet")
        hashes[str(path)] = sha256(path)
        frame = pd.read_parquet(path)
        if len(frame) != len(monitor) or not np.array_equal(frame["row_index"], np.arange(len(monitor))):
            raise ValueError("Score row alignment mismatch: " + label)
        digest = hashlib.sha1()
        for prompt in monitor["prompt"]:
            digest.update(prompt.encode())
            digest.update(b"\x1e")
        if not frame["prompts_sha1"].eq(digest.hexdigest()).all():
            raise ValueError("Score prompt digest mismatch: " + label)
        values = frame["margin"].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite monitor scores: " + label)
        return values

    def aligned_benchmark(spec, benchmark):
        margins = source("outputs/operating_point/v2/" + spec["run"] + "--adapter.parquet")
        joined = benchmark[KEY].merge(margins[KEY + ["margin"]], on=KEY,
                                      how="left", validate="one_to_one", sort=False)
        values = joined["margin"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or len(values) != len(benchmark):
            raise ValueError("Missing/nonfinite benchmark margins: " + spec["label"])
        return values

    monitor = source(config["rows"]).reset_index(drop=True)
    parent_spec = config["parent"]
    benchmark = source("outputs/raw/" + parent_spec["run"] + ".parquet")
    benchmark = benchmark[benchmark["condition"] != "attr_free"].reset_index(drop=True)
    for name, frame in (("monitor", monitor), ("benchmark", benchmark)):
        frame["reference_decision"] = frame["reference_decision"].str.lower()
        if not frame["reference_decision"].isin(["hire", "reject"]).all():
            raise ValueError("Unknown reference label in " + name)
    pm = scores(parent_spec["label"], monitor)
    pb = aligned_benchmark(parent_spec, benchmark)
    mbins, bbins = np.searchsorted(EDGES, pm, side="right"), np.searchsorted(EDGES, pb, side="right")
    bhire = benchmark["reference_decision"].eq("hire").to_numpy()
    exact_weights = np.bincount(bbins[bhire], minlength=8) / bhire.sum()
    rounded_weights = np.array([round(float(value), 4) for value in exact_weights])
    output = {
        "diagnostic_only": True, "thresholds_or_weights_refitted": False,
        "note": "Monitor reference labels are semisynthetic anchors; benchmark references are GPT-4o labels. This analysis cannot establish which labels are correct.",
        "parent_monitor": strata(monitor, pm, pm, np.ones(len(monitor), dtype=bool)),
        "parent_benchmark": strata(benchmark, pb, pb, np.ones(len(benchmark), dtype=bool)),
        "edges": EDGES.tolist(), "exact_benchmark_hire_weights": exact_weights.tolist(),
        "production_rounded_hire_weights": rounded_weights.tolist(), "arms": []}
    for arm in config["arms"]:
        if arm["label"] not in TARGETS:
            continue
        am = scores(arm["label"], monitor)
        ab = aligned_benchmark(arm, benchmark)
        bins = [{"bin": b,
                 "monitor": strata(monitor, pm, am, mbins == b),
                 "benchmark": strata(benchmark, pb, ab, bbins == b)} for b in range(8)]
        monitor_pool = weighted(bins, "monitor", "all", exact_weights)
        benchmark_pool = weighted(bins, "benchmark", "all", exact_weights)
        benchmark_hire = weighted(bins, "benchmark", "reference_hire", exact_weights)
        production = weighted(bins, "monitor", "all", rounded_weights)
        actual = 100 * ((ab[bhire] > 0).mean() - (pb[bhire] > 0).mean())
        if None in (monitor_pool, benchmark_pool, benchmark_hire, production):
            raise ValueError("Missing positive-weight bin prevents decomposition")
        cohort = monitor_pool - benchmark_pool
        pooling = benchmark_pool - benchmark_hire
        rounding = production - monitor_pool
        error = production - actual
        if not np.isclose(benchmark_hire, actual, atol=1e-10, rtol=0):
            raise ValueError("Exact reference-hire decomposition failed")
        if not np.isclose(cohort + pooling + rounding, error, atol=1e-10, rtol=0):
            raise ValueError("Error decomposition failed")
        output["arms"].append({"label": arm["label"], "bins": bins,
            "overall_monitor": strata(monitor, pm, am, np.ones(len(monitor), dtype=bool)),
            "overall_benchmark": strata(benchmark, pb, ab, np.ones(len(benchmark), dtype=bool)),
            "tpr_projection_error_decomposition_pp": {
                "production_projection": production, "actual_benchmark_tpr_change": float(actual),
                "total_error": error, "cohort_transfer_pooled_labels": cohort,
                "benchmark_label_pooling": pooling, "weight_rounding": rounding},
            "diagnostic_exact_weight_projections_pp": {
                "monitor_all": monitor_pool, "benchmark_all": benchmark_pool,
                "monitor_reference_hire": weighted(bins, "monitor", "reference_hire", exact_weights),
                "benchmark_reference_hire": benchmark_hire}})
    if {arm["label"] for arm in output["arms"]} != TARGETS:
        raise ValueError("Missing requested historical arms")
    output["source_sha256"] = hashes
    output["config_sha256"] = sha256(config_path)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite " + str(args.output))
    report = diagnose(args.config, args.scores_dir)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"parent_monitor": report["parent_monitor"],
                      "parent_benchmark": report["parent_benchmark"],
                      "arms": [{"label": arm["label"],
                                **arm["tpr_projection_error_decomposition_pp"],
                                **arm["diagnostic_exact_weight_projections_pp"]}
                               for arm in report["arms"]]}, indent=2))


if __name__ == "__main__":
    main()
