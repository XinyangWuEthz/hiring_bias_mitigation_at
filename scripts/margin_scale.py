"""How far an arm's decision margins separate pairs, against how much the attribute moves them,
on each model's own scale. CPU only; reads op-margin-v2 margin files.

Why. `margin_set_sd_nats` (analysis/reference_metrics.py) is in nats, and margins are on each
model's own scale. Training can widen every margin: DPO_V3_4880 and GRPO-v3-all have larger
within-set spreads than SFT-v3 yet fewer unstable sets. Multiplying all margins by a constant
changes no decision, so stability depends on the attribute's spread relative to how far the
sets sit from the cut. Over the within-group sets (pair x group x condition), per margin file:

  within_sd      mean over sets of the SD across the set's variants (= margin_set_sd_nats)
  between_sd     SD across sets of the sets' mean margins
  ratio          within_sd / between_sd: the attribute's spread on the arm's own scale
  at_risk_pct    sets whose mean margin is within one of their own SDs of the cut
  unstable_pct   sets with variants on both sides of the cut (margin > 0 hires, as op-margin-v2)
  hire_pct       attributed and attribute-free prompts with margin > 0
  af_mean, af_sd the attribute-free margins
  af_shift_mean, af_shift_sd, af_corr
                 against --reference, per pair: the change in the attribute-free margin, its SD,
                 and the correlation of the two files' attribute-free margins

    python scripts/margin_scale.py \\
        --reference Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional--adapter \\
        --files Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305--adapter ... \\
        --out reports/margin_scale.csv

Files are read from $HBM_OUTPUT_ROOT/outputs/operating_point/v2/<name>.parquet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

SET = ["pair_id", "protected_group", "condition"]
ATTR_FREE = "attr_free"


def load(name: str) -> pd.DataFrame:
    root = Path(resolve_output_path("outputs/operating_point/v2"))
    return pd.read_parquet(root / f"{name}.parquet")


def scale_row(margins: pd.DataFrame, reference: pd.DataFrame | None) -> dict:
    att = margins[margins["condition"] != ATTR_FREE]
    sets = att.groupby(SET)["margin"].agg(mean="mean", sd=lambda m: m.std(ddof=0),
                                          lo="min", hi="max")
    af = margins[margins["condition"] == ATTR_FREE].set_index("pair_id")["margin"]
    row = {"sets": len(sets), "within_sd": sets["sd"].mean(),
           "between_sd": sets["mean"].std(ddof=0),
           "at_risk_pct": 100 * ((sets["mean"].abs() < sets["sd"]) & (sets["sd"] > 0)).mean(),
           "unstable_pct": 100 * ((sets["lo"] <= 0) & (sets["hi"] > 0)).mean(),
           "hire_pct": 100 * (margins["margin"] > 0).mean(),
           "af_hire_pct": 100 * (af > 0).mean(), "af_mean": af.mean(), "af_sd": af.std(ddof=0)}
    row["ratio"] = row["within_sd"] / row["between_sd"]
    if reference is not None:
        ref = reference[reference["condition"] == ATTR_FREE].set_index("pair_id")["margin"]
        both = pd.concat({"run": af, "ref": ref}, axis=1, join="inner")
        shift = both["run"] - both["ref"]
        row.update(af_shift_mean=shift.mean(), af_shift_sd=shift.std(ddof=0),
                   af_corr=float(np.corrcoef(both["run"], both["ref"])[0, 1]))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", required=True,
                        help="margin file the shifts are taken against")
    parser.add_argument("--files", nargs="+", required=True, help="margin files, without .parquet")
    parser.add_argument("--out", type=Path, default=Path("reports/margin_scale.csv"))
    args = parser.parse_args()

    reference = load(args.reference)
    rows = [{"file": name, **scale_row(load(name), None if name == args.reference else reference)}
            for name in dict.fromkeys([args.reference, *args.files])]
    table = pd.DataFrame(rows).round(4)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
