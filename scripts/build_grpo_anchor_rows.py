#!/usr/bin/env python
"""The operating-point anchor set for GRPO-improve R2-a: held-out Djinni rows SFT-v3 is unsure of.

Takes grpo-v3-all's 696 monitor rows (held-out candidates, none in training or the benchmark)
with SFT-v3's op-margin-v2 margins as scored by the monitor calibration (job 5123830), keeps the
Djinni rows (RJDB rows moved the opposite way in that calibration, docs/GRPO_MONITOR8.md) whose
SFT-v3 margin lies in its 10-90% hire-probability band, |margin| < 2.197, and writes them with
the margin. Nothing is scored anew.

    python scripts/build_grpo_anchor_rows.py \
        --scores $HBM_OUTPUT_ROOT/outputs/monitor_calibration/grpo_v3_all_monitor8_20261022 \
        --monitor artifacts/semisynthetic-v3/grpo-v3-all/grpo_monitor.parquet \
        --out artifacts/semisynthetic-v3/grpo-anchor-djinni-band
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

BAND = 2.197  # logit(0.9)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--monitor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    monitor = pd.read_parquet(args.monitor)
    sft = pd.read_parquet(args.scores / "SFT-v3.parquet").sort_values("row_index")
    if len(sft) != len(monitor) or list(sft["row_index"]) != list(range(len(monitor))):
        raise ValueError("SFT-v3 monitor scores do not cover the monitor rows in order")
    rows = monitor.assign(sft_margin=sft["margin"].to_numpy())
    djinni = rows[~rows["pair_id"].str.startswith("rjdb")]
    anchor = djinni[djinni["sft_margin"].abs() < BAND].reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=False)
    dest = args.out / "anchor.parquet"
    anchor.to_parquet(dest, index=False)
    report = {
        "rows": len(anchor), "candidates": int(anchor["candidate_id"].nunique()),
        "pairs": int(anchor["pair_id"].nunique()),
        "by_reference": anchor["reference_decision"].value_counts().to_dict(),
        "sft_hire_share": float((anchor["sft_margin"] > 0).mean()),
        "band": BAND, "monitor": str(args.monitor), "monitor_sha256": sha(args.monitor),
        "scores": str(args.scores / "SFT-v3.parquet"),
        "scores_sha256": sha(args.scores / "SFT-v3.parquet"), "sha256": sha(dest),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
