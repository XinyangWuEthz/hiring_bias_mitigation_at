"""Recovering a LoRA adapter from merged weights (interp/extract.py).

The SFT-v3 adapter the analysis runs on is recovered from a merged release, so the recovery is
tested as an identity on a tiny random Qwen3.5: base plus a rank-r update on every target
module, then recovered and loaded through PEFT the way `spec.py` loads it, must give the
merged model's logits.
"""

import copy
import json

import pytest
import torch

pytest.importorskip("peft")

from hiring_bias_mitigation.interp import extract as EX  # noqa: E402

from .test_hooks import DEV, VOCAB  # noqa: E402

RANK = 4


def test_svd_recovers_a_rank_r_update_and_drops_noise():
    g = torch.Generator().manual_seed(0)
    update = torch.randn(48, RANK, generator=g) @ torch.randn(RANK, 80, generator=g)
    noise = 1e-3 * torch.randn(48, 80, generator=g)
    a, b, stats = EX.lora_from_delta(update + noise, RANK)
    assert a.shape == (RANK, 80) and b.shape == (48, RANK)
    # Closer to the update than the noisy delta it was given: the noise outside rank r is gone.
    assert (b @ a - update).norm() < noise.norm()
    assert stats["kept_energy"] > 0.999 and stats["sv_ratio"] < 0.01


def test_keys_and_targets_match_the_published_adapters():
    assert (EX.peft_key("model.language_model.layers.3.self_attn.q_proj.weight", "A")
            == "base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight")
    assert EX.TARGET.match("model.language_model.layers.0.mlp.down_proj.weight")
    assert not EX.TARGET.match("model.language_model.layers.0.linear_attn.out_proj.weight")
    assert not EX.TARGET.match("lm_head.weight")


@torch.no_grad()
def test_recovered_adapter_reproduces_the_merged_model(model, tmp_path):
    from peft import PeftModel
    from safetensors.torch import save_file

    torch.manual_seed(3)
    merged = copy.deepcopy(model)
    tensors, modules = {}, []
    for name, p in merged.named_parameters():
        ckpt = name.replace("model.", "model.language_model.", 1)
        if not EX.TARGET.match(ckpt):
            continue
        delta = 0.05 * (torch.randn(p.shape[0], RANK, device=p.device)
                        @ torch.randn(RANK, p.shape[1], device=p.device))
        p += delta
        a, b, _ = EX.lora_from_delta(delta, RANK)
        tensors[EX.peft_key(ckpt, "A")], tensors[EX.peft_key(ckpt, "B")] = a.cpu(), b.cpu()
        modules.append(ckpt)
    # Every MLP of the four layers, and the q/k/v/o of the one full-attention layer.
    assert len(modules) == 4 * 3 + 4
    save_file(tensors, str(tmp_path / "adapter_model.safetensors"))
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(EX.adapter_config("tiny", RANK, modules)))
    adapted = PeftModel.from_pretrained(copy.deepcopy(model), str(tmp_path)).eval()

    x = torch.randint(1, VOCAB, (2, 12), device=DEV)
    want = merged(x).logits
    effect = (want - model(x).logits).abs().max()
    assert effect > 1e-2  # the update is not a no-op
    # (W + BA)x and Wx + B(Ax) round differently, and the fla GDN kernels carry the difference
    # through the stack, so the identity holds to a small share of the update, not bit for bit.
    assert (adapted(x).logits - want).abs().max() < 0.01 * effect
