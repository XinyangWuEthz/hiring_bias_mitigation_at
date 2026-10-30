#!/usr/bin/env python
"""grpo-v3-all restricted to its tipped rows, balanced by anchor verdict (GRPO-improve R4).

A tipped row is one where SFT-v3 matched the anchor without the attribute and missed it with
the attribute in at least one sample (`tipped` in grpo-v3-all). Every tipped row of the scarcer
verdict is kept, and as many of the other verdict's tipped rows, drawn with a fixed seed. The
validation and monitor parquets are copied unchanged. Rows keep grpo-v3-all's order.

    python scripts/build_grpo_tipped_subset.py --source artifacts/semisynthetic-v3/grpo-v3-all \
        --out artifacts/semisynthetic-v3/grpo-v3-all-tipped
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train = pd.read_parquet(args.source / "grpo_train.parquet")
    tipped = train[train["tipped"].astype(bool)]
    sizes = tipped["reference_decision"].value_counts()
    n = int(sizes.min())
    keep = pd.concat([part.sample(n=n, random_state=args.seed) if len(part) > n else part
                      for _, part in tipped.groupby("reference_decision")]).index
    subset = train.loc[sorted(keep)].reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=False)
    subset.to_parquet(args.out / "grpo_train.parquet", index=False)
    for name in ("grpo_validation.parquet", "grpo_monitor.parquet"):
        shutil.copyfile(args.source / name, args.out / name)
    report = {
        "source": str(args.source), "source_train_sha256": sha(args.source / "grpo_train.parquet"),
        "rows": len(subset), "per_verdict": subset["reference_decision"].value_counts().to_dict(),
        "tipped_available": sizes.to_dict(), "seed": args.seed,
        "candidates": int(subset["candidate_id"].nunique()),
        "pairs": int(subset["pair_id"].nunique()),
        "sha256": sha(args.out / "grpo_train.parquet"),
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
