#!/usr/bin/env python
"""grpo-v3-all with each pair's attribute-free hire rate under SFT-v3, for the consistency reward.

Copies grpo-v3-all's train, validation and monitor parquets unchanged and adds two columns to
every row: `free_hire_rate`, the share of SFT-v3's 8 temperature-1 samples of the pair's
attribute-free prompt that decide hire, and `free_samples`, how many of them carry a decision.
The samples are the round-3 on-policy pools' own (`onpolicy-new{,2}/raw/samples`, items whose
`prompts.parquet` row has kind `attr_free`, keyed by `pair_id`). Nothing is sampled anew, and
no prompt, row or split changes: `tests/test_grpo_free_target.py` and the report's hashes say so.

    python scripts/build_grpo_free_target.py \
        --source artifacts/semisynthetic-v3/grpo-v3-all \
        --pool artifacts/semisynthetic-v3/onpolicy-new \
        --pool artifacts/semisynthetic-v3/onpolicy-new2 \
        --out artifacts/semisynthetic-v3/grpo-v3-all-free
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402

SPLITS = ("grpo_train", "grpo_validation", "grpo_monitor")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_rates(pool: Path) -> pd.DataFrame:
    """Per pair: SFT-v3's attribute-free hire share and the number of decided samples."""
    prompts = pd.read_parquet(pool / "prompts.parquet").reset_index(drop=True)
    free = prompts[prompts["kind"] == "attr_free"]
    chunks = sorted((pool / "raw/samples").glob("*.parquet"))
    samples = pd.concat(pd.read_parquet(f) for f in chunks)
    samples = samples[samples["item"].isin(free.index)]
    if not (samples.drop_duplicates("item").set_index("item")["prompt"]
            == free.loc[samples["item"].unique(), "prompt"]).all():
        raise ValueError(f"{pool}: sample prompts differ from prompts.parquet")
    decisions = [parse_output(t, "en").decision for t in samples["raw_output"]]
    samples = samples.assign(decision=decisions,
                             pair_id=free.loc[samples["item"], "key"].to_numpy())
    decided = samples[samples["decision"].isin(["hire", "reject"])]
    out = decided.groupby("pair_id")["decision"].agg(
        free_hire_rate=lambda d: float((d == "hire").mean()), free_samples="size")
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--pool", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rates = pd.concat([free_rates(p) for p in args.pool])
    if rates["pair_id"].duplicated().any():
        raise ValueError("a pair has attribute-free samples in more than one pool")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"source": str(args.source), "pools": [str(p) for p in args.pool], "splits": {}}
    for name in SPLITS:
        src = args.source / f"{name}.parquet"
        frame = pd.read_parquet(src)
        joined = frame.merge(rates, on="pair_id", how="left", validate="many_to_one")
        missing = int(joined["free_hire_rate"].isna().sum())
        if missing:
            raise ValueError(f"{name}: {missing} rows without attribute-free samples")
        if not joined[frame.columns].equals(frame):
            raise ValueError(f"{name}: rows changed in the join")
        dest = args.out / f"{name}.parquet"
        joined.to_parquet(dest, index=False)
        report["splits"][name] = {
            "rows": len(joined), "pairs": int(joined["pair_id"].nunique()),
            "source_sha256": sha(src), "sha256": sha(dest),
            "free_hire_rate_mean": float(joined["free_hire_rate"].mean()),
            "by_reference": {side: {
                "rows": int((part := joined[joined["reference_decision"] == side]).shape[0]),
                "free_hire_rate_mean": float(part["free_hire_rate"].mean()),
                "free_agrees_with_anchor_share": float(
                    ((part["free_hire_rate"] > 0.5) == (side == "hire")).mean()),
            } for side in ("hire", "reject")},
        }
    for extra in ("report.json", "monitor.json"):
        if (args.source / extra).exists():
            report[f"source_{extra}_sha256"] = sha(args.source / extra)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
