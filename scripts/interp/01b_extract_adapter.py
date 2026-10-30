"""Recover a LoRA adapter from a merged release, and check it reproduces that release.

SFT-v3 (`Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-v3-en`, the parent of the DPO-v3 and
GRPO releases) is published as a merged bf16 checkpoint only; its adapter stayed on Daint. The
pipeline needs an adapter (`interp/spec.py`), so this recovers one (`interp/extract.py`).

    extract   merged - base on every LoRA target module, rank-r SVD -> a PEFT adapter under
              outputs/interp/adapters/<name>/. Refuses if any other tensor of the language
              model differs from the base. No GPU forward; the SVDs run on --device.
    check     fp32 margins on a seeded sample of grid rows from (a) the merged release loaded
              as is and (b) the base with the recovered adapter, the way 02_margins.py loads
              it; base and SFT-v1 are scored too, for scale. Passes when (b) agrees with (a)
              on at least 99% of decisions and the mean |margin difference| is at most 0.1
              logits (2.5x the bf16 readout error in reports/interp/README.md). Fixed before
              the first run; a failure stops the pipeline here.

    uv run python scripts/interp/01b_extract_adapter.py --stage extract
    uv run python scripts/interp/01b_extract_adapter.py --stage check --device cuda:1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402

from hiring_bias_mitigation.interp import extract as EX  # noqa: E402
from hiring_bias_mitigation.interp import margins as M  # noqa: E402
from hiring_bias_mitigation.interp import render  # noqa: E402
from hiring_bias_mitigation.interp import spec as SP  # noqa: E402
from hiring_bias_mitigation.interp.io import data_config, interp_dir, report_dir, write_meta  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("interp.extract")

MAX_DISAGREE_PCT = 1.0
MAX_MEAN_ABS = 0.1


def snapshot(repo: str, revision: str | None = None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=[
        "*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*"]))


def tensor_index(root: Path) -> dict[str, Path]:
    """Every tensor name -> its file, read from the files themselves, not from an index."""
    from safetensors import safe_open

    index = {}
    for f in sorted(root.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            index.update(dict.fromkeys(fh.keys(), f))
    return index


def load(path: Path, name: str):
    from safetensors import safe_open

    with safe_open(str(path), "pt") as fh:
        return fh.get_tensor(name)


def extract(args, specs) -> None:
    import torch
    from safetensors.torch import save_file

    base_dir, merged_dir = snapshot(specs.base_model), snapshot(args.merged, args.revision)
    base, merged = tensor_index(base_dir), tensor_index(merged_dir)
    lm = lambda k: k.startswith(("model.language_model.", "lm_head."))  # noqa: E731
    missing = sorted(k for k in set(base) ^ set(merged) if lm(k))
    tensors, stats, changed, recast, unchanged_targets = {}, {}, [], [], []
    for name in sorted(set(base) & set(merged)):
        b, m = load(base[name], name), load(merged[name], name)
        if not EX.TARGET.match(name):
            # A release may store a tensor at lower precision than the base (SFT-v3 keeps the
            # GDN A_log and norm weights in bf16, the base in fp32). Only the cast is allowed:
            # the tensor must equal the base's cast to the release's dtype, bit for bit.
            if b.shape != m.shape or not torch.equal(b.to(m.dtype), m):
                changed.append(name)
            elif b.dtype != m.dtype:
                recast.append(name)
            continue
        delta = m.to(args.device, torch.float32) - b.to(args.device, torch.float32)
        if not delta.any():
            unchanged_targets.append(name)
            continue
        a, bb, st = EX.lora_from_delta(delta, args.rank)
        tensors[EX.peft_key(name, "A")] = a.cpu()
        tensors[EX.peft_key(name, "B")] = bb.cpu()
        stats[name] = st
    if missing or [k for k in changed if lm(k)]:
        raise SystemExit(f"{args.merged} is not base + LoRA on the target modules: tensors "
                         f"missing on one side {missing[:5]}, changed outside the targets "
                         f"{[k for k in changed if lm(k)][:5]}")

    out = Path(interp_dir("adapters", args.name))
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(
        json.dumps(EX.adapter_config(specs.base_model, args.rank, list(stats)), indent=2))
    kept = np.array([s["kept_energy"] for s in stats.values()])
    ratio = np.array([s["sv_ratio"] for s in stats.values()])
    template = [d / "chat_template.jinja" for d in (base_dir, merged_dir)]
    summary = {
        "merged": args.merged, "revision": args.revision, "base_model": specs.base_model,
        "rank": args.rank, "modules": len(stats), "unchanged_target_modules": unchanged_targets,
        "changed_outside_language_model": changed,
        # The adapter runs on the base's values of these; check measures what that costs.
        "stored_at_lower_precision": {"count": len(recast), "dtype": sorted({
            f"{load(base[k], k).dtype}->{load(merged[k], k).dtype}" for k in recast}),
            "tensors": recast},
        "kept_energy_min": float(kept.min()), "kept_energy_median": float(np.median(kept)),
        "sv_ratio_max": float(ratio.max()), "sv_ratio_median": float(np.median(ratio)),
        "chat_template_matches_base": all(t.is_file() for t in template)
        and template[0].read_text() == template[1].read_text(),
        "fingerprint": SP.adapter_fingerprint(str(out)),
    }
    (out / "extraction.json").write_text(json.dumps({**summary, "per_module": stats}, indent=2))
    write_meta(out / "extraction", **summary)
    (report_dir() / f"adapter_extract-{args.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def score(model, tokenizer, ro, texts: list[str], batch_size: int) -> np.ndarray:
    order = np.argsort([len(t) for t in texts], kind="stable")
    parts = [M.score_batch(model, tokenizer, ro, [texts[i] for i in order[s:s + batch_size]])
             ["margin_fp32"] for s in range(0, len(texts), batch_size)]
    return np.concatenate(parts)[np.argsort(order)]


def check(args, specs) -> None:
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = data_config()
    tokenizer = AutoTokenizer.from_pretrained(specs.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    frame = render.render(render.build_frame("grid", cfg), tokenizer,
                          cfg.get("chat_template_kwargs"))
    frame = frame.sample(n=args.rows, random_state=args.seed).reset_index(drop=True)
    texts = frame["text"].tolist()

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot(args.merged, args.revision)), dtype=torch.bfloat16, device_map=args.device,
        trust_remote_code=True).eval()
    result = {"merged": score(model, tokenizer, M.readout(model, tokenizer, cfg["lang"]), texts,
                              args.batch_size)}
    del model
    gc.collect()
    torch.cuda.empty_cache()

    names = ["base", args.spec, *[s for s in ("sft_v1",) if s in specs.specs]]
    model, _ = SP.load_models(specs, names, device=args.device)
    ro = M.readout(model, tokenizer, cfg["lang"])
    for name in names:
        with SP.use(model, specs.specs[name]):
            result[name] = score(model, tokenizer, ro, texts, args.batch_size)

    ref = result["merged"]
    rows = frame[["prompt_uid", "protected_group", "protected_attr", "condition"]].assign(
        **{f"margin_{k}": v for k, v in result.items()})
    rows.to_parquet(interp_dir("adapters", args.name) / "check.parquet", index=False)
    versus = {name: {
        "mean_abs_diff": float(np.abs(result[name] - ref).mean()),
        "max_abs_diff": float(np.abs(result[name] - ref).max()),
        "decision_agreement_pct": float(100 * ((result[name] > 0) == (ref > 0)).mean()),
    } for name in names}
    rec = versus[args.spec]
    passed = (100 - rec["decision_agreement_pct"] <= MAX_DISAGREE_PCT
              and rec["mean_abs_diff"] <= MAX_MEAN_ABS)
    summary = {"merged": args.merged, "revision": args.revision, "spec": args.spec,
               "rows": len(frame), "seed": args.seed, "hire_rate_merged_pct":
               float(100 * (ref > 0).mean()), "against_merged": versus,
               "criteria": {"max_disagree_pct": MAX_DISAGREE_PCT, "max_mean_abs": MAX_MEAN_ABS},
               "passed": passed}
    (report_dir() / f"adapter_check-{args.name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not passed:
        raise SystemExit(f"the recovered adapter does not reproduce {args.merged}; stop here")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("extract", "check"), required=True)
    parser.add_argument("--merged", default=EX.SFT_V3)
    parser.add_argument("--revision", default=EX.SFT_V3_REVISION)
    parser.add_argument("--name", default="sft-v3-svd32", help="outputs/interp/adapters/<name>")
    parser.add_argument("--rank", type=int, default=32, help="the LoRA rank the release trained")
    parser.add_argument("--spec", default="sft", help="the models.yaml spec using the adapter")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    specs = SP.load_spec_set()
    (extract if args.stage == "extract" else check)(args, specs)


if __name__ == "__main__":
    main()
