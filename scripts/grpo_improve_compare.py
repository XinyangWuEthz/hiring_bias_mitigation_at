#!/usr/bin/env python
"""Compare GRPO-improve short runs on the summary fixed in docs/GRPO_IMPROVE.md, section 3.

Reads each arm's `trajectory_diagnostics.csv` (written by grpo_trajectory_diagnostics.py),
keeps the declared checkpoints, and writes one summary row per arm plus a two-panel SVG:
full-benchmark TPR change against SFT-v3, and the greedy proxy of the matched change in
unstable sets, by optimizer step. No plotting library is needed.

    python scripts/grpo_improve_compare.py \
        --arm "GRPO-v3-all (seed 42)=$HBM_OUTPUT_ROOT/outputs/grpo_improve/diag_grpo_v3_all" \
        --arm "A0=$HBM_OUTPUT_ROOT/outputs/grpo_improve/diag_a0" \
        --out $HBM_OUTPUT_ROOT/outputs/grpo_improve/compare_round1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SFT_TPR = 100 * 5975 / 10350
STEPS = (100, 125, 150, 175, 200)
COLORS = ("#7a7a7a", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")


def summary(name: str, table: pd.DataFrame, steps) -> dict:
    t = table[table["step"].isin(steps)].sort_values("step")
    missing = sorted(set(steps) - set(t["step"]))
    d = t["tpr_pct"] - SFT_TPR
    row = {
        "arm": name, "checkpoints": len(t), "missing_steps": missing,
        "tpr_sd_pp": float(t["tpr_pct"].std(ddof=1)) if len(t) > 1 else None,
        "max_neighbour_tpr_change_pp": (float(t["tpr_pct"].diff().abs().max())
                                        if len(t) > 1 else None),
        "min_delta_tpr_pp": float(d.min()), "max_delta_tpr_pp": float(d.max()),
        "below_minus8": int((d <= -8).sum()), "below_minus10": int((d < -10).sum()),
        "mean_proxy_matched_pp": float(t["proxy_delta_matched_pp"].mean()),
        "min_proxy_matched_pp": float(t["proxy_delta_matched_pp"].min()),
        "max_proxy_matched_pp": float(t["proxy_delta_matched_pp"].max()),
        "c_sd": float(t["c"].std(ddof=1)) if len(t) > 1 else None,
        "mean_dprime": float(t["dprime"].mean()),
        "mean_unstable_raw_pct": float(t["unstable_raw_pct"].mean()),
    }
    if "frac_reward_zero_std" in t:
        row["mean_frac_zero_std"] = float(t["frac_reward_zero_std"].mean())
    return row


def svg(curves: dict[str, pd.DataFrame], steps, path: Path) -> None:
    w, h, pad, gap = 760, 300, 48, 40
    panels = [("tpr_delta", "Full TPR change vs SFT-v3 (pp)", [(-8, "#999", "4 3"),
                                                              (-10, "#c00", "2 2")]),
              ("proxy_delta_matched_pp", "Proxy matched change in unstable sets (pp)",
               [(0, "#999", "4 3")])]
    pw = (w - 3 * pad - gap) / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'font-family="sans-serif" font-size="11"><rect width="{w}" height="{h}" fill="white"/>']
    xs = (min(steps), max(steps))
    for k, (col, title, refs) in enumerate(panels):
        x0 = pad + k * (pw + gap + pad)
        y0, ph = 30, h - 80
        vals = [v for t in curves.values() for v in t[col]] + [r[0] for r in refs]
        lo, hi = min(vals), max(vals)
        lo, hi = lo - 0.08 * (hi - lo + 1e-9), hi + 0.08 * (hi - lo + 1e-9)

        def X(s, x0=x0):
            return x0 + (s - xs[0]) / (xs[1] - xs[0]) * pw

        def Y(v, lo=lo, hi=hi, y0=y0, ph=ph):
            return y0 + (hi - v) / (hi - lo) * ph

        out.append(f'<text x="{x0}" y="18" font-weight="bold">{title}</text>')
        out.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" '
                   'fill="none" stroke="#ccc"/>')
        for v, color, dash in refs:
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" '
                       f'stroke="{color}" stroke-dasharray="{dash}"/>'
                       f'<text x="{x0 + pw - 2}" y="{Y(v) - 3:.1f}" text-anchor="end" '
                       f'fill="{color}">{v:g}</text>')
        for s in steps:
            out.append(f'<text x="{X(s):.1f}" y="{y0 + ph + 14}" text-anchor="middle">{s}</text>')
        for frac in (0, 0.5, 1):
            v = lo + frac * (hi - lo)
            out.append(f'<text x="{x0 - 4}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.1f}</text>')
        for i, t in enumerate(curves.values()):
            color = COLORS[i % len(COLORS)]
            pts = " ".join(f"{X(s):.1f},{Y(v):.1f}" for s, v in zip(t["step"], t[col]))
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
            out.extend(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="2.5" fill="{color}"/>'
                       for s, v in zip(t["step"], t[col]))
    for i, name in enumerate(curves):
        x = pad + i * 150
        out.append(f'<rect x="{x}" y="{h - 22}" width="12" height="3" '
                   f'fill="{COLORS[i % len(COLORS)]}"/>'
                   f'<text x="{x + 16}" y="{h - 17}">{name}</text>')
    out.append(f'<text x="{w / 2}" y="{h - 34}" text-anchor="middle">optimizer step</text></svg>')
    path.write_text("\n".join(out) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, help="NAME=DIAG_DIR")
    ap.add_argument("--steps", default=",".join(map(str, STEPS)))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    steps = tuple(int(s) for s in args.steps.split(","))
    rows, curves, long = [], {}, []
    for spec in args.arm:
        name, folder = spec.split("=", 1)
        table = pd.read_csv(Path(folder) / "trajectory_diagnostics.csv")
        rows.append(summary(name, table, steps))
        t = table[table["step"].isin(steps)].sort_values("step").assign(
            tpr_delta=lambda f: f["tpr_pct"] - SFT_TPR)
        curves[name] = t
        long.append(t.assign(arm=name))
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "summary.csv", index=False)
    pd.concat(long).to_csv(args.out / "curves.csv", index=False)
    svg(curves, steps, args.out / "curves.svg")
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2, default=str) + "\n")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
