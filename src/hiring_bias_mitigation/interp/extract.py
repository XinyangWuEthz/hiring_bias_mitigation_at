"""A LoRA adapter recovered from a merged checkpoint.

SFT-v3, and the DPO-v3 and GRPO releases built on it, are published as merged bf16 weights
only, while this pipeline switches adapters over one base model (`spec.py`). A LoRA update has
rank r by construction, so on each adapted module `merged - base` is that update plus the bf16
rounding of the merge. Its best rank-r approximation recovers the update and drops most of the
rounding, which is spread over every direction rather than concentrated in r of them.

Every other tensor has to be bit-identical to the base. `margins.py` reads the decision through
the base model's final norm and `lm_head`, and a module changed outside the LoRA targets would
be silently lost from the recovered adapter -- so `compare` reports it and the script refuses.
"""

from __future__ import annotations

import re

import torch

#: The SFT-v3 release, pinned: the revision every artifact derived from it was produced from.
SFT_V3 = "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-v3-en"
SFT_V3_REVISION = "b6e5f2b361f19e07fa14f1f7fd7c7864b4eb97f1"

#: The modules a Qwen3.5 LoRA here adapts: attention on the full-attention layers and every MLP,
#: as the published SFT adapters target them (`q,k,v,o,gate,up,down_proj`).
TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\.(self_attn|mlp)\.(q|k|v|o|gate|up|down)_proj\.weight$"
)


def peft_key(name: str, part: str) -> str:
    """Checkpoint tensor name -> adapter key, as the published SFT adapters name them.

    The checkpoint is a `Qwen3_5ForConditionalGeneration` (`model.language_model.layers...`);
    the pipeline loads it as `Qwen3_5ForCausalLM`, whose decoder is `model.layers...`.
    """
    module = name[: -len(".weight")].replace("model.language_model.", "model.", 1)
    return f"base_model.model.{module}.lora_{part}.weight"


@torch.no_grad()
def lora_from_delta(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """(A, B) in fp32 with B @ A the best rank-`rank` approximation of `delta`, and the fit.

    The singular values are split evenly between the factors, so neither carries the scale.
    `kept_energy` is the share of ||delta||^2 the approximation keeps; `sv_ratio` is the
    (r+1)-th singular value over the r-th, near 0 when delta really has rank r.
    """
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    root = s[:rank].sqrt()
    b = (u[:, :rank] * root).contiguous()
    a = (root[:, None] * vh[:rank]).contiguous()
    energy = s.pow(2)
    total = float(energy.sum())
    stats = {
        "norm": total ** 0.5,
        "kept_energy": float(energy[:rank].sum()) / total if total else 1.0,
        "sv_ratio": float(s[rank] / s[rank - 1]) if len(s) > rank and s[rank - 1] > 0 else 0.0,
    }
    return a, b, stats


def adapter_config(base_model: str, rank: int, modules: list[str]) -> dict:
    """PEFT config for the recovered factors: alpha = r, so the scaling is 1 and B @ A is delta."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "target_modules": sorted({m.rsplit(".", 2)[-2] for m in modules}),
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "use_dora": False,
        "use_rslora": False,
        "modules_to_save": None,
        "layers_to_transform": None,
    }
