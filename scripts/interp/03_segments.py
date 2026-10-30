"""D3: token segmentation of every model input, with the layout assertions (CPU only).

    uv run python scripts/interp/03_segments.py --dataset grid
    uv run python scripts/interp/03_segments.py --dataset pool

Writes outputs/interp/segments/{dataset}-{lang}.parquet (one row per prompt: token ids,
[start, end) of every segment, alignment flags), the rows that failed to violations.parquet,
and a summary to reports/interp/segments_summary-{dataset}.json. Stops when hard violations
exceed `max_violation_share` of rows: a suffix-aligned patch on a broken layout compares
different tokens, and nothing downstream would notice.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hiring_bias_mitigation.interp import render  # noqa: E402
from hiring_bias_mitigation.interp import segment as S  # noqa: E402
from hiring_bias_mitigation.interp.io import data_config, interp_dir, report_dir, write_meta  # noqa: E402
from hiring_bias_mitigation.interp.spec import load_spec_set  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("interp.segments")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("grid", "pool"), required=True)
    parser.add_argument("--data-config", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    cfg = data_config(args.data_config)
    tokenizer = AutoTokenizer.from_pretrained(load_spec_set().base_model, trust_remote_code=True)
    kwargs = cfg.get("chat_template_kwargs")
    frame = render.render(render.build_frame(args.dataset, cfg), tokenizer, kwargs)
    log.info("%d rows to segment", len(frame))

    segs = S.segment_frame(frame, tokenizer, kwargs)
    out_dir = interp_dir("segments")
    path = out_dir / f"{args.dataset}-{cfg['lang']}.parquet"
    segs.to_parquet(path, index=False)
    write_meta(path, dataset=args.dataset, rows=len(segs), data_config=cfg)

    hard = segs["violations"] != ""
    bad = segs[hard | ~segs["set_ok"]]
    bad.drop(columns=["input_ids"]).to_parquet(out_dir / f"violations-{args.dataset}.parquet",
                                               index=False)
    attr = segs["protected_attr"] != "__attr_free__"
    summary = {
        "dataset": args.dataset, "rows": len(segs),
        "rows_with_hard_violations": int(hard.sum()),
        "rows_with_whitespace_merges": int((segs["merges"] != "").sum()),
        "attribute_sets": int(segs[attr].groupby(["pair_id", "protected_group",
                                                 "condition"]).ngroups),
        "attribute_sets_not_ok": int((~segs[attr].groupby(
            ["pair_id", "protected_group", "condition"])["set_ok"].all()).sum()),
        "presence_suffix_aligned_share": float(segs.loc[attr, "presence_suffix_ok"].mean()),
        "presence_prefix_aligned_share": float(segs.loc[attr, "presence_prefix_ok"].mean()),
        "n_tokens_p50_p95_max": [float(x) for x in
                                 segs["n_tokens"].quantile([0.5, 0.95, 1.0])],
    }
    (report_dir() / f"segments_summary-{args.dataset}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    share = (hard | ~segs["set_ok"]).mean()
    if share > float(cfg.get("max_violation_share", 0.005)):
        raise SystemExit(f"{share:.2%} of rows violate the segment layout; see {out_dir}")


if __name__ == "__main__":
    main()
