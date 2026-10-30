import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from hiring_bias_mitigation.mitigation import lora_average as L

AK = "base_model.model.proj.lora_A.weight"
BK = "base_model.model.proj.lora_B.weight"


def config():
    return {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 2, "lora_alpha": 4,
            "target_modules": ["proj"], "base_model_name_or_path": "test/base",
            "bias": "none", "fan_in_fan_out": False, "rank_pattern": {}, "alpha_pattern": {},
            "modules_to_save": None, "use_dora": False, "use_rslora": False,
            "lora_dropout": 0.0, "inference_mode": True}


@pytest.fixture
def sources(tmp_path):
    parent, run = tmp_path / "sft-parent", tmp_path / "grpo-run"
    parent.mkdir()
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({"sft_checkpoint": str(parent)}))
    gen = torch.Generator().manual_seed(9)
    source = []
    for folder in [parent, *[run / f"checkpoint-{s}" for s in [425, 450, 475]]]:
        folder.mkdir(exist_ok=True)
        (folder / "adapter_config.json").write_text(json.dumps(config()))
        save_file({AK: torch.randn(2, 5, generator=gen), BK: torch.randn(4, 2, generator=gen)},
                  folder / "adapter_model.safetensors")
        if folder != parent:
            source.append(folder)
    return source


def test_dense_equivalence_unchanged_blocks_and_resume(sources, tmp_path):
    output = tmp_path / "average"
    manifest = L.build_average(sources, output)
    assert manifest["layer_count"] == 1
    assert manifest["construction"]["output_rank"] == 6
    assert manifest["construction"]["output_alpha"] == 4
    state = load_file(output / "adapter_model.safetensors")
    original = [load_file(p / "adapter_model.safetensors") for p in sources]
    expected = sum(s[BK].double() @ s[AK].double() * 2 for s in original) / 3
    actual = state[BK].double() @ state[AK].double() * (4 / 6)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    for i, source in enumerate(original):
        assert torch.equal(state[AK][i * 2:(i + 1) * 2], source[AK])
        assert torch.equal(state[BK][:, i * 2:(i + 1) * 2], source[BK])
    assert L.validate_average(output, sources)["binding_sha256"] == manifest["binding_sha256"]
    assert L.MG.adapter_chain(output)[:-1] == L.MG.adapter_chain(sources[0])[:-1]


def test_separate_factor_averaging_is_not_the_update_average():
    a = [torch.tensor([[1., 2.]]), torch.tensor([[5., 7.]])]
    b = [torch.tensor([[2.], [3.]]), torch.tensor([[11.], [13.]])]
    ac, bc, scale, identities = L.concatenate_factors(a, b, alpha=2)
    expected = sum(bb @ aa * 2 for aa, bb in zip(a, b)) / 2
    naive = (sum(b) / 2) @ (sum(a) / 2) * 2
    assert not torch.allclose(naive, expected)
    torch.testing.assert_close((bc @ ac) * scale, expected)
    assert all(i["output_blocks_bit_identical"] for i in identities)


@pytest.mark.parametrize("field,value", [
    ("rank_pattern", {"proj": 3}), ("alpha_pattern", {"proj": 8}),
    ("fan_in_fan_out", True), ("bias", "all"), ("lora_bias", True),
    ("modules_to_save", ["head"]), ("use_dora", True), ("use_rslora", True),
    ("use_qalora", True), ("target_parameters", ["weight"]),
    ("some_future_lora_variant", True)])
def test_unsupported_configuration_fails_before_output(sources, tmp_path, field, value):
    for source in sources:
        cfg = config()
        cfg[field] = value
        (source / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        L.build_average(sources, tmp_path / "bad-average")
    assert not (tmp_path / "bad-average").exists()


def test_mismatched_config_nonuniform_weights_and_duplicate_sources_rejected(sources, tmp_path):
    with pytest.raises(ValueError, match="equal normalized"):
        L.build_average(sources, tmp_path / "weighted", weights=[.5, .25, .25])
    with pytest.raises(ValueError, match="distinct"):
        L.build_average([sources[0], sources[0]], tmp_path / "duplicate")
    cfg = config()
    cfg["lora_alpha"] = 8
    (sources[1] / "adapter_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="identical"):
        L.build_average(sources, tmp_path / "mixed-config")


@pytest.mark.parametrize("mutation", ["bf16", "nan", "extra_bias", "missing_b"])
def test_unsupported_tensor_payload_rejected(sources, tmp_path, mutation):
    state = load_file(sources[0] / "adapter_model.safetensors")
    if mutation == "bf16":
        state = {k: v.bfloat16() for k, v in state.items()}
    elif mutation == "nan":
        state[AK][0, 0] = float("nan")
    elif mutation == "extra_bias":
        state["base_model.model.proj.bias"] = torch.zeros(4)
    else:
        del state[BK]
    save_file(state, sources[0] / "adapter_model.safetensors")
    with pytest.raises(ValueError):
        L.build_average(sources, tmp_path / "bad-tensors")
    assert not (tmp_path / "bad-tensors").exists()


def test_output_never_overwritten_and_changed_source_refuses_resume(sources, tmp_path):
    output = tmp_path / "average"
    L.build_average(sources, output)
    with pytest.raises(FileExistsError):
        L.build_average(sources, output)
    tensor_path = sources[0] / "adapter_model.safetensors"
    state = load_file(tensor_path)
    state[AK][0, 0] += 1
    save_file(state, tensor_path)
    with pytest.raises(ValueError, match="inputs mismatch"):
        L.validate_average(output, sources)


def test_modified_output_refuses_resume(sources, tmp_path):
    output = tmp_path / "average"
    L.build_average(sources, output)
    path = output / "adapter_model.safetensors"
    state = load_file(path)
    state[AK][0, 0] += 1
    save_file(state, path)
    with pytest.raises(ValueError, match="fingerprint"):
        L.validate_average(output, sources)


def test_real_peft_load_keeps_fp32_factors_and_bf16_merge_rounding(sources, tmp_path):
    from peft import PeftModel
    from transformers import LlamaConfig, LlamaForCausalLM

    real_a = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    real_b = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"
    for source in sources:
        cfg = config()
        cfg["target_modules"] = ["q_proj"]
        (source / "adapter_config.json").write_text(json.dumps(cfg))
        state = load_file(source / "adapter_model.safetensors")
        save_file({real_a: state[AK][:, :4].contiguous(), real_b: state[BK]},
                  source / "adapter_model.safetensors")
    output = tmp_path / "average"
    L.build_average(sources, output)
    state = load_file(output / "adapter_model.safetensors")
    base = LlamaForCausalLM(LlamaConfig(hidden_size=4, intermediate_size=8, num_hidden_layers=1,
                          num_attention_heads=1, num_key_value_heads=1, vocab_size=16)).bfloat16()
    model = PeftModel.from_pretrained(base, output)
    layer = model.base_model.model.model.layers[0].self_attn.q_proj
    assert layer.lora_A["default"].weight.dtype == torch.float32
    assert torch.equal(layer.lora_A["default"].weight.detach(), state[real_a])
    assert torch.equal(layer.lora_B["default"].weight.detach(), state[real_b])
    before = layer.get_base_layer().weight.detach().float().clone()
    expected = (before + state[real_b] @ state[real_a] * (4 / 6)).bfloat16()
    retained = L.MG.merge_measured(model)
    assert retained["layers"] == 1
    assert layer.get_base_layer().weight.dtype == torch.bfloat16
    torch.testing.assert_close(layer.get_base_layer().weight, expected, atol=0, rtol=0)


@pytest.mark.parametrize("kept,expected_pass", [(.79, False), (.8, True), (float("nan"), False)])
def test_merge_retention_gate_never_changes_dtype_or_original_merge_function(
        sources, tmp_path, monkeypatch, kept, expected_pass):
    output, merged = tmp_path / "average", tmp_path / "merged"
    L.build_average(sources, output)
    original_dir_function = L.MG.merged_dir_for
    retention = [{"adapter": str(sources[0].parent.parent / "sft-parent"), "kept_fraction": 1.0},
                 {"adapter": str(output), "kept_fraction": kept}]

    def fake_merge(path):
        target = L.MG.merged_dir_for(path)
        target.mkdir()
        (target / "fake-merged-weights").write_bytes(b"test")
        return target

    monkeypatch.setattr(L.MG, "merge", fake_merge)
    monkeypatch.setattr(L.MG, "recorded_retention", lambda p: retention)
    if expected_pass:
        result = L.merge_average(output, merged)
        assert result["minimum_retention_pass"]
        assert result["merged_dtype"] == "bfloat16"
    else:
        with pytest.raises(ValueError):
            L.merge_average(output, merged)
    assert L.MG.merged_dir_for is original_dir_function
