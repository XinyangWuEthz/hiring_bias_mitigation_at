"""D2: fp32 decision margins for one model spec over one dataset, HF + PEFT, resumable.

    uv run python scripts/interp/02_margins.py --spec base --dataset grid --device cuda:0
    uv run python scripts/interp/02_margins.py --spec sft  --dataset grid --device cuda:1
    uv run python scripts/interp/02_margins.py --spec dpo  --dataset grid --adapter dpo=PATH

Each chunk of rows is written as its own part file, so an interrupted pass resumes where it
stopped. With --resid the last-position residual after every layer (33 x 4096, fp16) is saved
beside each part, for the probes in H5 and the final-position analyses.

Then, on the grid, the pass is checked against the audit it is meant to reproduce: the share
of decisions agreeing with the audit's greedy decision on the same rows, and the unstable
share recomputed from these margins next to the audit's own on the same sets. A margin pass
that does not reproduce the audit is not the model the behavioural results describe
(lesson 4 in reports/MITIGATION_FINDINGS.md).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.interp import extract as EX  # noqa: E402
from hiring_bias_mitigation.interp import margins as M  # noqa: E402
from hiring_bias_mitigation.interp import render  # noqa: E402
from hiring_bias_mitigation.interp import spec as SP  # noqa: E402
from hiring_bias_mitigation.interp.io import data_config, interp_dir, report_dir, write_meta  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("interp.margins")

KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
SET = ["pair_id", "condition"]  # 34 variants per set: the unit MITIGATION_FINDINGS reports
AUDIT_RUN = {"base": "Qwen3.5-9B--en--baseline",
             "sft": "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional",
             "sft_v1": "Qwen3.5-9B--en--sft--adapter--en_only"}
# SFT-v3's audit responses are not in the public responses dataset. Its release carries the
# audit's summary, so until the parquet is copied into outputs/raw/ the pass is checked
# against that summary's hire rate and within-group unstable share instead of row by row.
AUDIT_SUMMARY = {"sft": (EX.SFT_V3, EX.SFT_V3_REVISION, "evaluation.json")}
META_COLUMNS = ["dataset", *KEY, "candidate_id", "job_id", "group_id", "prompt_uid"]


def parse_overrides(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--adapter expects NAME=PATH, got {item!r}")
        out[name] = path
    return out


def run_pass(frame: pd.DataFrame, model, tokenizer, ro, out_dir: Path, batch_size: int,
             chunk: int, resid: bool) -> None:
    from safetensors.numpy import save_file

    n_parts = (len(frame) + chunk - 1) // chunk
    for part in range(n_parts):
        path = out_dir / f"part-{part:05d}.parquet"
        if path.exists():
            continue
        rows = frame.iloc[part * chunk:(part + 1) * chunk]
        # Length-sorted batches waste least compute on padding; results go back in row order.
        order = np.argsort(rows["text"].str.len().to_numpy(), kind="stable")
        texts = rows["text"].to_numpy()[order].tolist()
        collected: dict[str, list] = {}
        started = time.time()
        for start in range(0, len(texts), batch_size):
            out = M.score_batch(model, tokenizer, ro, texts[start:start + batch_size], resid)
            for name, values in out.items():
                collected.setdefault(name, []).append(values)
        merged = {name: np.concatenate(v)[np.argsort(order)] for name, v in collected.items()}
        n_tokens = [len(tokenizer(t, add_special_tokens=False)["input_ids"])
                    for t in rows["text"]]
        result = rows[META_COLUMNS].assign(
            n_tokens=n_tokens, **{k: v for k, v in merged.items() if k != "resid"})
        result["decision_hf"] = np.where(result["margin_fp32"] > 0, "hire", "reject")
        if resid:
            save_file({"resid": merged["resid"]}, str(out_dir / f"resid-{part:05d}.safetensors"))
        result.to_parquet(path, index=False)  # written last: its presence marks the part done
        log.info("part %d/%d: %d rows in %.0fs", part + 1, n_parts, len(rows),
                 time.time() - started)


def audit_check(margins: pd.DataFrame, spec: str) -> dict:
    run = AUDIT_RUN.get(spec)
    path = Path(resolve_output_path("outputs/raw")) / f"{run}.parquet" if run else None
    if path is None or not path.exists():
        out = {"audit_run": run, "available": False}
        if spec in AUDIT_SUMMARY:
            out.update(summary_check(margins, *AUDIT_SUMMARY[spec]))
        return out
    audit = pd.read_parquet(path, columns=[*KEY, "decision"])
    both = margins.merge(audit, on=KEY, how="inner")
    decided = both[both["decision"].isin(["hire", "reject"])]
    attr = decided[decided["condition"] != "attr_free"]

    def unstable(col: str) -> float:
        return float(100 * (attr.groupby(SET)[col].nunique() > 1).mean())

    return {
        "audit_run": run, "available": True, "rows_matched": len(both),
        "agreement_pct": float(100 * (decided["decision"] == decided["decision_hf"]).mean()),
        "hire_rate_hf_pct": float(100 * (decided["decision_hf"] == "hire").mean()),
        "hire_rate_audit_pct": float(100 * (decided["decision"] == "hire").mean()),
        "unstable_sets_hf_pct": unstable("decision_hf"),
        "unstable_sets_audit_pct": unstable("decision"),
        "margin_bf16_minus_fp32_abs_mean": float(
            (margins["margin_bf16"] - margins["margin_fp32"]).abs().mean()),
    }


def summary_check(margins: pd.DataFrame, repo: str, revision: str, filename: str) -> dict:
    """The pass against an audit known only by its summary: aggregate rates, not agreement.

    Within-group sets are (pair, group, condition) over attributed rows, as
    `analysis/reference_metrics.unstable_within_group` counts them for `evaluation.json`.
    """
    from huggingface_hub import hf_hub_download

    reported = json.loads(Path(hf_hub_download(repo, filename, revision=revision)).read_text())
    attr = margins[margins["condition"] != "attr_free"]
    unstable = attr.groupby(["pair_id", "protected_group", "condition"])["decision_hf"].nunique()
    return {
        "summary_source": f"{repo}@{revision[:7]}/{filename}",
        "summary_run": reported.get("run"),
        "hire_rate_hf_pct": float(100 * (margins["decision_hf"] == "hire").mean()),
        "hire_rate_audit_pct": reported.get("hire_pct"),
        "unstable_within_group_hf_pct": float(100 * (unstable > 1).mean()),
        "unstable_within_group_audit_pct": reported.get("unstable_within_group_pct"),
        "margin_bf16_minus_fp32_abs_mean": float(
            (margins["margin_bf16"] - margins["margin_fp32"]).abs().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dataset", choices=("grid", "pool"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=1024, help="rows per resumable part file")
    parser.add_argument("--resid", action="store_true",
                        help="also save the last-position residual of every layer (fp16)")
    parser.add_argument("--limit", type=int, default=None, help="first N rows only (pilot)")
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--data-config", default=None)
    args = parser.parse_args()

    cfg = data_config(args.data_config)
    specs = SP.load_spec_set(overrides=parse_overrides(args.adapter))
    spec = specs.specs[args.spec]
    model, tokenizer = SP.load_models(specs, [args.spec], device=args.device)
    ro = M.readout(model, tokenizer, cfg["lang"])
    frame = render.render(render.build_frame(args.dataset, cfg), tokenizer,
                          cfg.get("chat_template_kwargs"))
    if args.limit:
        frame = frame.head(args.limit)

    name = f"{args.dataset}-{cfg['lang']}" + (f"-limit{args.limit}" if args.limit else "")
    out_dir = interp_dir("margins", args.spec, name)
    with SP.use(model, spec):
        run_pass(frame, model, tokenizer, ro, out_dir, args.batch_size, args.chunk, args.resid)

    margins = pd.concat([pd.read_parquet(p) for p in sorted(out_dir.glob("part-*.parquet"))],
                        ignore_index=True)
    if len(margins) != len(frame) or set(margins["prompt_uid"]) != set(frame["prompt_uid"]):
        raise SystemExit(f"{out_dir} does not match the current frame; delete it and rerun")
    final = out_dir.parent / f"{name}.parquet"
    margins.to_parquet(final, index=False)
    fingerprints = {a: SP.adapter_fingerprint(SP.resolve_adapter(specs.adapter_paths[a]))
                    for a in spec.adapters}
    write_meta(final, spec=args.spec, adapters=fingerprints, base_model=specs.base_model,
               dataset=args.dataset, rows=len(margins), batch_size=args.batch_size,
               dtype="bfloat16 weights, fp32 readout", data_config=cfg)

    summary = {"spec": args.spec, "dataset": name, "rows": len(margins),
               "hire_rate_pct": float(100 * (margins["margin_fp32"] > 0).mean())}
    if args.dataset == "grid":
        summary["audit"] = audit_check(margins, args.spec)
    (report_dir() / f"margins_summary-{args.spec}-{name}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
