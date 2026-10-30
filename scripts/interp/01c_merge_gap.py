"""How far merging an adapter into bf16 weights moves the margin, next to the S1 gap.

Gate S1 of docs/INTERP_SFT_V3.md compares the recovered SFT-v3 adapter, unmerged, with the merged
release. Merging is itself a bf16 rounding of every adapted weight, so this measures that
rounding on the same rows: each SFT adapter scored unmerged (as the pipeline runs it) and after
merge_and_unload into bf16 weights (as the releases and audits were built). Needs the
check.parquet that `01b_extract_adapter.py --stage check` writes.

    uv run python scripts/interp/01c_merge_gap.py cuda:1
"""
import gc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from hiring_bias_mitigation.interp import margins as M  # noqa: E402
from hiring_bias_mitigation.interp import render  # noqa: E402
from hiring_bias_mitigation.interp import spec as SP  # noqa: E402
from hiring_bias_mitigation.interp.io import data_config, interp_dir  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
check = pd.read_parquet(interp_dir("adapters", "sft-v3-svd32") / "check.parquet")
cfg = data_config()
specs = SP.load_spec_set()


def score(model, tok, texts):
    ro = M.readout(model, tok, cfg["lang"])
    order = np.argsort([len(t) for t in texts], kind="stable")
    parts = [M.score_batch(model, tok, ro, [texts[i] for i in order[s:s + 8]])["margin_fp32"]
             for s in range(0, len(texts), 8)]
    return np.concatenate(parts)[np.argsort(order)]


texts = None
out = {}
for name in ("sft", "sft_v1"):
    model, tok = SP.load_models(specs, [name], device=DEV)
    if texts is None:
        frame = render.render(render.build_frame("grid", cfg), tok, cfg.get("chat_template_kwargs"))
        texts = frame.set_index("prompt_uid").loc[check["prompt_uid"], "text"].tolist()
    with SP.use(model, specs.specs[name]):
        unmerged = score(model, tok, texts)
    merged = score(model.merge_and_unload(), tok, texts)
    out[name] = {"unmerged": unmerged, "merged_bf16": merged}
    del model
    gc.collect()
    torch.cuda.empty_cache()

release = check["margin_merged"].to_numpy()


def cmp(a, b):
    d = np.abs(a - b)
    return {"mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
            "decision_agreement_pct": float(100 * ((a > 0) == (b > 0)).mean())}


v3, v1 = out["sft"], out["sft_v1"]
summary = {
    "rows": len(check),
    "sft_v3: recovered adapter unmerged vs release": cmp(v3["unmerged"], release),
    "sft_v3: recovered adapter re-merged in bf16 vs release": cmp(v3["merged_bf16"], release),
    "sft_v3: recovered unmerged vs re-merged": cmp(v3["unmerged"], v3["merged_bf16"]),
    "sft_v1: published adapter unmerged vs merged in bf16": cmp(v1["unmerged"],
                                                                v1["merged_bf16"]),
    "unmerged sft check reproduces S1 run": cmp(v3["unmerged"], check["margin_sft"].to_numpy()),
}
print(json.dumps(summary, indent=2))
dest = REPO / "reports" / "interp" / "adapter_merge_gap-sft-v3-svd32.json"
dest.write_text(json.dumps(summary, indent=2))
