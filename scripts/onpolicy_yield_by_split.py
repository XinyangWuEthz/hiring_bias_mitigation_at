"""Tipped negatives per row of an on-policy run, by pool source, SFT split and anchor.

    python scripts/onpolicy_yield_by_split.py \\
        --run artifacts/semisynthetic-v3/onpolicy-full \\
        --validation artifacts/semisynthetic-v3/additional/sft_validation.parquet \\
        --sft-train artifacts/semisynthetic-v3/additional/additional_sft_train.parquet \\
        --out reports/v3_new_data/onpolicy_yield_by_split.json

A row is one attributed prompt of the run (`rows.parquet`); it yields when the run selected an
attribute-tipped negative for it (`selected.parquet`, `tipped`), before the judge. Rows are
split by candidate: a candidate in `--validation` was held out of the student's SFT, any other
candidate of the run was in its SFT training split. With `--sft-train`, a training candidate's
rows split again: the rows SFT trained on (pair, attribute and condition all in the file) and
the rows its per-source balance left out. The student's error rate where it never trained is
what a round on new pairs should expect (docs/V3_NEW_DATA.md, round 3). CPU only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

KEY = ["pair_id", "protected_attr", "condition"]


def yields(rows: pd.DataFrame, selected: pd.DataFrame, held_out: set[str],
           trained_rows: pd.DataFrame | None = None) -> dict:
    rows = rows.assign(split=rows["candidate_id"].astype(str).isin(held_out)
                       .map({True: "sft_validation", False: "sft_train"}))
    if trained_rows is not None:
        seen = rows[KEY].astype(str).merge(trained_rows[KEY].astype(str).drop_duplicates()
                                           .assign(seen=True), on=KEY, how="left")["seen"]
        seen = seen.astype("boolean").fillna(False).astype(bool).to_numpy()
        train = (rows["split"] == "sft_train").to_numpy()
        rows.loc[train & seen, "split"] = "sft_train_row_trained"
        rows.loc[train & ~seen, "split"] = "sft_train_row_left_out"
    tipped = selected.loc[selected["tipped"].astype(bool), KEY].drop_duplicates()
    rows = rows.merge(tipped.assign(yielded=True), on=KEY, how="left")
    rows["yielded"] = rows["yielded"].astype("boolean").fillna(False).astype(bool)

    def summary(part: pd.DataFrame) -> dict:
        return {"rows": len(part), "candidates": int(part["candidate_id"].nunique()),
                "tipped": int(part["yielded"].sum()),
                "tipped_per_row_pct": round(100 * part["yielded"].mean(), 2) if len(part) else None}

    out: dict = {}
    for (source, split), part in rows.groupby(["pool_source", "split"], sort=True):
        out.setdefault(source, {})[split] = {
            "all": summary(part),
            "by_anchor": {d: summary(p) for d, p in part.groupby("reference_decision", sort=True)}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="an on-policy run's artifacts_dir")
    parser.add_argument("--validation", required=True,
                        help="the student's SFT validation file; its candidates are held out")
    parser.add_argument("--sft-train",
                        help="the student's SFT training rows, to split training candidates' "
                             "rows into those it trained on and those it did not")
    parser.add_argument("--out", help="write the report here as JSON")
    args = parser.parse_args()

    run = Path(resolve_output_path(args.run))
    rows = pd.read_parquet(run / "rows.parquet",
                           columns=[*KEY, "candidate_id", "pool_source", "reference_decision"])
    selected = pd.read_parquet(run / "selected.parquet", columns=[*KEY, "tipped"])
    held_out = set(pd.read_parquet(resolve_output_path(args.validation),
                                   columns=["candidate_id"])["candidate_id"].astype(str))
    trained = (pd.read_parquet(resolve_output_path(args.sft_train), columns=KEY)
               if args.sft_train else None)
    report = {"run": args.run, "validation": args.validation, "sft_train": args.sft_train,
              "by_pool_source": yields(rows, selected, held_out, trained)}
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
