"""Token lengths of an SFT or DPO config's rows as its trainer renders them, against `max_seq_len`.

    python scripts/sft_token_lengths.py \\
        --config configs/mitigation/sft/qwen3.5-9b_en_only_sft_v3_additional_gh200.yaml \\
        --limit 2048 --out reports/v3_new_data/sft_token_lengths.json

TRL cuts a row longer than `max_seq_len` at the end, and under a completion-only loss the end is
the target. This reads the splits through the trainer's own loader and chat rendering
(`training_common.load_generated_dataset`, `sft.render_chat`), every validation row included,
and counts per pool source the rows over the config's `max_seq_len` and over each `--limit`.
For a DPO config (`stage: dpo`) the rows go through `dpo._to_conversational`, and a row's
length is the prompt with the longer of its two responses: TRL 1.x truncates DPO rows at
`max_length` from the end too (`truncation_mode` keep_start), which would cut the responses
the preference margin is taken over.
CPU only; the tokenizer is the base model's.

    python scripts/sft_token_lengths.py \\
        --config configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v3_gh200.yaml \\
        --limit 2048 --out reports/v3_new_data/dpo_token_lengths.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.mitigation import sft as S  # noqa: E402
from hiring_bias_mitigation.mitigation import training_common as C  # noqa: E402
from hiring_bias_mitigation.utils.config import load_config  # noqa: E402


def length_summary(lengths: pd.Series, limits: list[int]) -> dict:
    values = lengths.to_numpy()
    return {"rows": len(values), "tokens": int(values.sum()),
            "mean": round(float(values.mean()), 1), "p50": int(np.percentile(values, 50)),
            "p99": int(np.percentile(values, 99)), "max": int(values.max()),
            **{f"over_{limit}": int((values > limit).sum()) for limit in limits}}


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="an SFT or DPO training config")
    parser.add_argument("--limit", type=int, action="append", default=[],
                        help="also count the rows over this many tokens; repeatable")
    parser.add_argument("--by", default="pool_source", help="the column to break counts down by")
    parser.add_argument("--out", help="write the report here as JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.pop("eval_limit_rows", None)  # every validation row, not the trainer's subsample
    dpo = cfg.get("stage") == "dpo"
    splits = C.load_generated_dataset(cfg, cfg.get("data_kind", "dpo" if dpo else "sft"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    max_seq_len = int(cfg.get("max_seq_len", 2048))
    limits = sorted({max_seq_len, *args.limit})

    report = {"config": args.config, "base_model": cfg["base_model"],
              "data_config": cfg.get("data_config"), "max_seq_len": max_seq_len,
              "by": args.by, "splits": {}}
    for name, dataset in splits.items():
        frame = dataset.to_pandas()
        if dpo:
            from hiring_bias_mitigation.mitigation import dpo as P

            rendered = P._to_conversational(dataset, tokenizer, cfg.get("chat_template_kwargs"))
            sides = [[len(ids) for ids in tokenizer(
                [p + r for p, r in zip(rendered["prompt"], rendered[side])],
                add_special_tokens=False)["input_ids"]] for side in ("chosen", "rejected")]
            frame["n_tokens"] = [max(c, r) for c, r in zip(*sides)]
        else:
            rendered = S.render_chat(dataset, tokenizer, cfg.get("chat_template_kwargs"))
            texts = [p + c for p, c in zip(rendered["prompt"], rendered["completion"])]
            frame["n_tokens"] = [len(ids) for ids in
                                 tokenizer(texts, add_special_tokens=False)["input_ids"]]
        groups = (frame[args.by].fillna("missing") if args.by in frame.columns
                  else pd.Series("all", index=frame.index))
        report["splits"][name] = {
            "all": length_summary(frame["n_tokens"], limits),
            "by": {str(key): length_summary(part, limits)
                   for key, part in frame.groupby(groups, sort=True)["n_tokens"]},
        }
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
