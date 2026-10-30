"""Base, SFT and DPO as one PEFT model whose adapters are switched, never re-loaded.

Cross-model patching (base activations into the SFT run) and adapter masking (the SFT update
on some modules only) both need the models side by side, and a second 19 GB copy of the
weights would leave one 48 GB card no room for activations. LoRA makes that unnecessary: the
three models differ only in which low-rank deltas are active, so one base plus named adapters
is all three.

A spec names its adapters in order. A DPO adapter trained on top of the SFT model is the spec
`[sft, dpo]`: LoRA deltas add on the weights, so the stacked spec equals SFT merged and DPO
applied -- an identity `scripts/interp/06_e0_sanity.py` checks once the adapter arrives. A DPO
run that kept training the SFT adapter itself is the spec `[dpo]`. Which of the two the other
team ran is declared in `configs/interp/models.yaml`, not guessed.
"""

from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.config import load_yaml, resolve_output_path
from ..utils.logging import get_logger
from .io import CONFIG_DIR

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    adapters: tuple[str, ...] = ()
    #: the spec a DPO policy's implicit reward is measured against
    reference: str | None = None


@dataclass
class SpecSet:
    base_model: str
    adapter_paths: dict[str, str]
    specs: dict[str, ModelSpec] = field(default_factory=dict)

    def needed_adapters(self, names: list[str]) -> list[str]:
        order: list[str] = []
        for name in names:
            for adapter in self.specs[name].adapters:
                if adapter not in order:
                    order.append(adapter)
        return order


def load_spec_set(path: str | Path | None = None, overrides: dict[str, str] | None = None
                  ) -> SpecSet:
    """Reads models.yaml; `${VAR}` in an adapter path is read from the environment."""
    raw = load_yaml(path or CONFIG_DIR / "models.yaml")
    paths = {name: os.path.expandvars(str(p)) for name, p in raw["adapters"].items()}
    paths.update(overrides or {})
    specs = {
        name: ModelSpec(name, tuple(body.get("adapters") or ()), body.get("reference"))
        for name, body in raw["specs"].items()
    }
    return SpecSet(raw["base_model"], paths, specs)


def resolve_adapter(path: str) -> str:
    """A Hub repo id stays as is; a local path resolves against the output drive."""
    if "${" in path or not path:
        raise FileNotFoundError(
            f"adapter path {path!r} is unset. Pass --adapter NAME=PATH or export the "
            "variable named in configs/interp/models.yaml (e.g. HBM_DPO_ADAPTER)."
        )
    local = Path(resolve_output_path(path))
    if local.exists():
        if not (local / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{local} has no adapter_config.json")
        return str(local)
    if path.count("/") == 1 and not path.startswith((".", "/")):
        return path  # a Hub repo id
    raise FileNotFoundError(f"adapter {path} not found locally and is not a Hub repo id")


def adapter_fingerprint(path: str) -> str:
    """sha256 of the adapter weights, so an artifact names exactly which update produced it."""
    local = Path(path)
    if not local.exists():
        from huggingface_hub import hf_hub_download

        local = Path(hf_hub_download(path, "adapter_model.safetensors")).parent
    weights = local / "adapter_model.safetensors"
    digest = hashlib.sha256()
    with open(weights, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def load_models(spec_set: SpecSet, names: list[str], device: str = "cuda:0",
                dtype: str = "bfloat16"):
    """One model carrying every adapter the named specs need, plus its tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec_set.base_model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec_set.base_model, dtype=getattr(torch, dtype), device_map=device,
        trust_remote_code=True)
    adapters = spec_set.needed_adapters(names)
    if adapters:
        from peft import PeftModel

        first, *rest = adapters
        model = PeftModel.from_pretrained(model, resolve_adapter(spec_set.adapter_paths[first]),
                                          adapter_name=first)
        for name in rest:
            model.load_adapter(resolve_adapter(spec_set.adapter_paths[name]), adapter_name=name)
        for name in adapters:
            log.info("adapter %s: %d LoRA modules", name, lora_module_count(model, name))
    model.eval()
    return model, tokenizer


def lora_module_count(model, adapter: str) -> int:
    return sum(1 for m in model.modules()
               if hasattr(m, "lora_A") and adapter in getattr(m, "lora_A", {}))


@contextmanager
def use(model, spec: ModelSpec):
    """Runs the block with exactly this spec's adapters active."""
    if not spec.adapters:
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                yield model
        else:
            yield model
        return
    if not hasattr(model, "base_model") or not hasattr(model, "peft_config"):
        raise RuntimeError(f"spec {spec.name} needs adapters but the model has none loaded")
    previous = model.base_model.active_adapters
    model.base_model.set_adapter(list(spec.adapters), inference_mode=True)
    try:
        yield model
    finally:
        model.base_model.set_adapter(previous, inference_mode=True)
