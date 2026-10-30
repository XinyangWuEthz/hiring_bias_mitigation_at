"""D4: the contrast-pair manifest every patching experiment draws its pairs from.

    uv run python scripts/interp/04_manifest.py --dataset pool   # discovery (base only)
    uv run python scripts/interp/04_manifest.py --dataset grid   # confirmation, SFT/DPO

Needs 03_segments.py for the dataset and 02_margins.py for at least the base spec; margins of
any other spec present on disk are joined as extra columns. Selection flags use base margins
only. Writes outputs/interp/manifest/contrasts-{dataset}-{lang}.parquet and a per-contrast
summary to reports/interp/manifest_summary-{dataset}.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from hiring_bias_mitigation.interp import manifest as MF  # noqa: E402
from hiring_bias_mitigation.interp.io import (  # noqa: E402
    CONFIG_DIR,
    data_config,
    interp_dir,
    report_dir,
    write_meta,
)
from hiring_bias_mitigation.utils.config import load_yaml  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("interp.manifest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("grid", "pool"), required=True)
    parser.add_argument("--contrasts", default=str(CONFIG_DIR / "contrasts.yaml"))
    parser.add_argument("--data-config", default=None)
    args = parser.parse_args()

    cfg = data_config(args.data_config)
    name = f"{args.dataset}-{cfg['lang']}"
    segments = pd.read_parquet(interp_dir("segments") / f"{name}.parquet").drop(
        columns=["input_ids"])
    margins = {}
    for path in sorted(interp_dir("margins").glob(f"*/{name}.parquet")):
        margins[path.parent.name] = pd.read_parquet(path, columns=["prompt_uid", "margin_fp32"])
    if "base" not in margins:
        raise SystemExit(f"no base margins for {name}; run 02_margins.py --spec base first")
    log.info("margins for specs: %s", sorted(margins))

    contrasts = MF.expand_contrasts(load_yaml(args.contrasts))
    sel = cfg.get("selection", {})
    manifest = MF.build_manifest(segments, contrasts, margins,
                                 threshold=float(sel.get("min_abs_effect", 0.5)),
                                 random_share=float(sel.get("random_share", 0.3)),
                                 seed=int(cfg.get("seed", 42)))
    path = interp_dir("manifest") / f"contrasts-{name}.parquet"
    manifest.to_parquet(path, index=False)
    write_meta(path, dataset=args.dataset, rows=len(manifest), specs=sorted(margins),
               selection=sel)

    agg = {"pairs": ("pair_id", "size"), "aligned": ("aligned", "mean"),
           "len_match": ("len_match", "mean"), "sel_effect": ("sel_effect", "sum"),
           "sel_flip": ("sel_flip", "sum")}
    agg.update({f"mean_dm_{s}": (f"dm_{s}", "mean") for s in margins})
    agg.update({f"mean_abs_dm_{s}": (f"dm_{s}", lambda x: x.abs().mean()) for s in margins})
    summary = manifest.groupby("contrast_id").agg(**agg).round(3)
    summary.to_csv(report_dir() / f"manifest_summary-{args.dataset}.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
