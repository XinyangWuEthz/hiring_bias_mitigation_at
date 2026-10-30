#!/usr/bin/env python
"""CPU-only diagnostics of a GRPO run's saved checkpoints against its SFT parent.

Reads what is already on disk and runs no model:

- the per-checkpoint benchmark decisions written by `checkpoint_benchmark_trajectory.py`
  (`steps/step-NNNN/raw.parquet`, greedy, all 31,050 prompts);
- the parent's op-margin-v2 margin file;
- the trainer's `log_history` (`trainer_state.json` of the last checkpoint).

Per checkpoint it writes TPR, FPR, signal-detection c and d', raw unstable within-group sets,
attribute-free hires, and a *greedy proxy* of the standard's decision number: the run's greedy
decisions stand in for its margin pass (margin > 0), and the parent is moved to the same hire
count with `operating_point_control.matched_comparison`, unchanged. The proxy is checked
against the official op-margin-v2 reports where both exist (`--check`). It is a development
reading on an exposed benchmark, not a verdict under docs/EVAL_STANDARD.md.

Per window between checkpoints it averages the training log: share of zero-variance groups,
KL, gradient norm, entropy, and the sampled hire rate on each anchor verdict.

    python scripts/grpo_trajectory_diagnostics.py \
        --trajectory $HBM_OUTPUT_ROOT/outputs/checkpoint_benchmark/grpo_v3_all_20261023 \
        --run-dir $HBM_OUTPUT_ROOT/outputs/grpo/qwen3.5-9b_en_only_grpo_v3_all \
        --parent Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional \
        --out $HBM_OUTPUT_ROOT/outputs/grpo_improve/diag_grpo_v3_all
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hiring_bias_mitigation.analysis.reference_metrics import KEY, SET  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "opc", REPO / "scripts" / "operating_point_control.py")
opc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(opc)

LOG_KEYS = ["frac_reward_zero_std", "kl", "grad_norm", "entropy", "reward",
            "rewards/decision_reward/mean", "sampled_hire_rate_ref_hire",
            "sampled_hire_rate_ref_reject", "clip_ratio/region_mean"]


def sdt(tpr: float, fpr: float, n_pos: int, n_neg: int) -> tuple[float, float]:
    """Criterion c and d' with the 1/(2N) correction at 0 and 1."""
    tpr = min(max(tpr, 1 / (2 * n_pos)), 1 - 1 / (2 * n_pos))
    fpr = min(max(fpr, 1 / (2 * n_neg)), 1 - 1 / (2 * n_neg))
    zt, zf = norm.ppf(tpr), norm.ppf(fpr)
    return float(-(zt + zf) / 2), float(zt - zf)


def decisions(raw: Path) -> pd.DataFrame:
    frame = pd.read_parquet(raw, columns=[*KEY, "decision", "reference_decision"])
    return frame.rename(columns={"reference_decision": "ref"})


def checkpoint_row(step: int, frame: pd.DataFrame, parent_margins: pd.DataFrame,
                   parent_dec: pd.DataFrame | None) -> dict:
    hire = frame["decision"] == "hire"
    pos, neg = frame["ref"] == "hire", frame["ref"] == "reject"
    tpr, fpr = float(hire[pos].mean()), float(hire[neg].mean())
    c, d = sdt(tpr, fpr, int(pos.sum()), int(neg.sum()))
    att = frame[frame["condition"] != "attr_free"]
    unstable = float((att.groupby(SET)["decision"].nunique() > 1).mean())
    free = frame[frame["condition"] == "attr_free"]
    run = frame[KEY].assign(margin=np.where(hire, 1.0, -1.0))
    matched = opc.matched_comparison(parent_margins[[*KEY, "margin"]], run)
    row = {
        "step": step, "tpr_pct": 100 * tpr, "fpr_pct": 100 * fpr, "c": c, "dprime": d,
        "hire_pct": 100 * float(hire.mean()),
        "agree_pct": 100 * float((frame["decision"] == frame["ref"]).mean()),
        "unstable_raw_pct": 100 * unstable,
        "free_hires": int((free["decision"] == "hire").sum()),
        "proxy_delta_matched_pp": matched["delta_matched_pp"],
        "proxy_ci_low": matched["ci_low"], "proxy_ci_high": matched["ci_high"],
        "proxy_unstable_ref_matched_pct": 100 * matched["unstable_ref_matched"],
    }
    if parent_dec is not None:
        joined = frame.merge(parent_dec, on=KEY, suffixes=("", "_parent"))
        row["changed_vs_parent"] = int((joined["decision"] != joined["decision_parent"]).sum())
    return row


def transitions(frames: dict[int, pd.DataFrame], parent_margins: pd.DataFrame) -> pd.DataFrame:
    """Decision changes between consecutive checkpoints: direction, and where they sit on the
    parent's margin (|margin| < 2.197 is the parent's 10-90% hire-probability band)."""
    margin = parent_margins.set_index(KEY)["margin"]
    steps, rows = sorted(frames), []
    for a, b in itertools.pairwise(steps):
        left = frames[a].set_index(KEY)["decision"]
        right = frames[b].set_index(KEY)["decision"].reindex(left.index)
        ref = frames[a].set_index(KEY)["ref"]
        changed = left != right
        to_hire = changed & (right == "hire")
        m = margin.reindex(left.index)
        rows.append({
            "from": a, "to": b, "changed": int(changed.sum()),
            "to_hire": int(to_hire.sum()), "to_reject": int((changed & ~to_hire).sum()),
            "one_way_share": float(max(to_hire.sum(), (changed & ~to_hire).sum())
                                   / max(changed.sum(), 1)),
            "changed_ref_hire": int((changed & (ref == "hire")).sum()),
            "changed_in_parent_band": (float((m[changed].abs() < 2.197).mean())
                                       if changed.any() else np.nan),
            "all_in_parent_band": float((m.abs() < 2.197).mean()),
        })
    return pd.DataFrame(rows)


def window_stats(history: list[dict], steps: list[int]) -> pd.DataFrame:
    log = pd.DataFrame([h for h in history if "frac_reward_zero_std" in h])
    rows, prev = [], 0
    for s in steps:
        part = log[(log["step"] > prev) & (log["step"] <= s)]
        rows.append({"step": s, "window": f"{prev + 1}-{s}", "n_logs": len(part),
                     **{k: float(part[k].mean()) for k in LOG_KEYS if k in part}})
        prev = s
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trajectory", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--parent", required=True, help="parent run name (op-margin-v2 margins)")
    ap.add_argument("--parent-raw", type=Path, help="parent raw parquet, for decision changes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extra", nargs="*", default=[],
                    help="STEP=RAW pairs: a checkpoint audited outside the trajectory, e.g. the "
                         "final adapter's outputs/raw/<run>.parquet")
    ap.add_argument("--check", nargs="*", default=[],
                    help="STEP=RUN pairs whose official op-margin-v2 report checks the proxy")
    args = ap.parse_args()

    root = Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
    margins = pd.read_parquet(
        root / "outputs/operating_point/v2" / f"{args.parent}--adapter.parquet")
    parent_raw = args.parent_raw or root / "outputs/raw" / f"{args.parent}.parquet"
    parent_dec = decisions(parent_raw)[[*KEY, "decision"]] if parent_raw.exists() else None

    steps_dir = args.trajectory / "steps"
    frames: dict[int, pd.DataFrame] = {}
    if parent_dec is not None:
        frames[0] = decisions(parent_raw)
    for d in sorted(steps_dir.glob("step-*")):
        step = int(d.name.split("-")[1])
        if (d / "raw.parquet").exists() and step not in frames:
            frames[step] = decisions(d / "raw.parquet")
    for item in args.extra:
        step, raw = item.split("=", 1)
        frames.setdefault(int(step), decisions(Path(raw)))
    rows = [checkpoint_row(s, f, margins, parent_dec) for s, f in sorted(frames.items())]
    ckpt = pd.DataFrame(rows).sort_values("step")
    moves = transitions(frames, margins)

    last = max(args.run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    history = json.loads((last / "trainer_state.json").read_text())["log_history"]
    win = window_stats(history, [s for s in ckpt["step"] if s > 0])
    table = ckpt.merge(win, on="step", how="left")
    table["delta_tpr_step_pp"] = table["tpr_pct"].diff()
    table["delta_c_step"] = table["c"].diff()

    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "trajectory_diagnostics.csv", index=False)
    moves.to_csv(args.out / "transitions.csv", index=False)
    full = pd.DataFrame([h for h in history if "frac_reward_zero_std" in h])
    full.to_csv(args.out / "train_log.csv", index=False)

    checks = []
    for item in args.check:
        step, run = item.split("=", 1)
        rep = REPO / "reports/operating_point/v2" / f"{run}--vs--{args.parent}.json"
        official = json.loads(rep.read_text()) if rep.exists() else None
        if official:
            m = official.get("matched") or official
            official = {k: m.get(k) for k in ("delta_matched_pp", "ci_low", "ci_high",
                                              "hire_rate_run", "unstable_run") if k in m} or m
        mine = table.loc[table["step"] == int(step), "proxy_delta_matched_pp"]
        checks.append({"step": int(step), "run": run, "report": str(rep),
                       "proxy": float(mine.iloc[0]) if len(mine) else None,
                       "official": official})
    summary = {"parent": args.parent, "trajectory": str(args.trajectory),
               "run_dir": str(args.run_dir), "checks": checks,
               "corr_tpr_vs_train_hire_ref_hire": float(
                   table["tpr_pct"].corr(table["sampled_hire_rate_ref_hire"])),
               "corr_dtpr_vs_dtrain_hire_ref_hire": float(
                   table["delta_tpr_step_pp"].corr(table["sampled_hire_rate_ref_hire"].diff()))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = ["step", "tpr_pct", "fpr_pct", "c", "dprime", "unstable_raw_pct", "free_hires",
            "proxy_delta_matched_pp", "proxy_ci_low", "proxy_ci_high", "changed_vs_parent",
            "frac_reward_zero_std", "kl", "grad_norm", "sampled_hire_rate_ref_hire",
            "sampled_hire_rate_ref_reject"]
    print(table[[c for c in cols if c in table]].round(3).to_string(index=False))
    print(moves.round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("corr")}, indent=2))


if __name__ == "__main__":
    main()
