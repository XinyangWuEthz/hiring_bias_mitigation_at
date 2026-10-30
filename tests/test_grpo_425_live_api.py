"""Installed vLLM CPU contracts; no model initialization, inference or CUDA.

The loader test uses one-element stand-ins, so it verifies name routing, shard
metadata and per-tensor acknowledgements, not full-size packing or GPU execution.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    path = Path(
        os.environ.get(
            "GRPO_DIAGNOSTIC_SCRIPT",
            str(Path(__file__).parents[1] / "scripts/grpo_425_diagnostic.py"),
        )
    )
    spec = importlib.util.spec_from_file_location("diagnostic_live_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_427_names_load_one_at_a_time_through_installed_qwen_loaders():
    import torch
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5Model,
    )

    fixture = json.loads(
        (Path(__file__).parent / "fixtures/grpo_425_serving_names.json").read_text()
    )
    names = fixture["names"]
    assert len(names) == len(set(names)) == 427
    cuda_before = torch.cuda.is_initialized()

    # These are plain CPU module trees, not instances of a Qwen model. Their
    # loader methods and name mappers are the real installed implementations.
    class Root(torch.nn.Module):
        hf_to_vllm_mapper = Qwen3_5ForConditionalGeneration.hf_to_vllm_mapper
        load_weights = Qwen3_5ForConditionalGeneration.load_weights

    class Causal(torch.nn.Module):
        hf_to_vllm_mapper = Qwen3_5ForCausalLM.hf_to_vllm_mapper
        load_weights = Qwen3_5ForCausalLM.load_weights

    class Decoder(torch.nn.Module):
        hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper
        load_weights = Qwen3_5Model.load_weights

    root = Root()
    root.language_model = Causal()
    root.language_model.model = Decoder()
    root.language_model.model.config = SimpleNamespace(model_type="qwen3_5_text")
    routing = {}
    shard_metadata = {}
    for name in names:
        mapped = root.hf_to_vllm_mapper.map_name(name)
        assert mapped is not None and mapped.startswith("language_model.")
        causal_name = root.language_model.hf_to_vllm_mapper.map_name(
            mapped.removeprefix("language_model.")
        )
        assert causal_name is not None
        if causal_name.startswith("model."):
            inner_name = causal_name.removeprefix("model.")
            mapped_inner, shard = root.language_model.model.hf_to_vllm_mapper._map_name_with_shard(
                inner_name
            )
            final = "language_model.model." + mapped_inner
        else:
            final, shard = "language_model." + causal_name, None
        routing[name], shard_metadata[name] = final, shard

    observed = []
    for target in sorted(set(routing.values())):
        module = root
        *components, leaf = target.split(".")
        for component in components:
            if component not in module._modules:
                module.add_module(component, torch.nn.Module())
            module = module._modules[component]
        parameter = torch.nn.Parameter(torch.zeros(1, device="cpu"), requires_grad=False)

        def record_load(param, incoming, target=target):
            observed.append((target, getattr(incoming, "shard_id", None), incoming.item()))
            param.data.copy_(incoming)

        parameter.weight_loader = record_load
        module.register_parameter(leaf, parameter)

    params = dict(root.named_parameters())
    for pass_number in (1, 2):
        for index, name in enumerate(names):
            value = float(pass_number * 1000 + index)
            incoming = torch.tensor([value], device="cpu")
            result = root.load_weights([(name, incoming)])
            assert result == {routing[name]}, name
            assert observed[-1] == (routing[name], shard_metadata[name], value), name
            assert params[routing[name]].item() == value
            assert params[routing[name]].device.type == "cpu"
    assert len(observed) == 2 * 427
    assert torch.cuda.is_initialized() == cuda_before


@pytest.mark.parametrize("flat_logprobs", [False, True])
def test_actual_sampling_and_output_contracts(diagnostic, flat_logprobs):
    from vllm import SamplingParams
    from vllm.logprobs import FlatLogprobs, Logprob
    from vllm.outputs import CompletionOutput, RequestOutput

    warm = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=[11, 12])
    warm._validate_logprobs(SimpleNamespace(max_logprobs=20, get_vocab_size=lambda: 100))
    assert warm.num_logprobs == 2
    assert warm.clone().logprob_token_ids == [11, 12]
    params = SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0, n=1, seed=42)
    assert params.max_tokens == 512 and params.temperature == 0 and params.seed == 42
    selected = {11: Logprob(-0.25), 12: Logprob(-1.25)}
    if flat_logprobs:
        logprobs = FlatLogprobs()
        logprobs.append(selected)
    else:
        logprobs = [selected]
    completion = CompletionOutput(
        index=0,
        text='{"decision":"hire"}',
        token_ids=[11],
        cumulative_logprob=None,
        logprobs=logprobs,
        finish_reason="stop",
    )
    request = RequestOutput(
        request_id="0",
        prompt="wrapped",
        prompt_token_ids=[1, 2],
        prompt_logprobs=None,
        outputs=[completion],
        finished=True,
    )
    margin = request.outputs[0].logprobs[0][11].logprob - request.outputs[0].logprobs[0][12].logprob
    assert margin == 1.0
    records, differences = diagnostic.check_request_outputs(
        [request], [{"prompt_id": "row", "wrapped_prompt": "wrapped", "input_ids": [1, 2]}]
    )
    assert differences == []
    assert records == [
        {
            "prompt_id": "row",
            "raw_output": '{"decision":"hire"}',
            "actual_input_ids": [1, 2],
            "generated_token_ids": [11],
            "finish_reason": "stop",
        }
    ]


def test_served_parameter_hashes_accept_actual_vllm_parameter(diagnostic):
    import torch
    from vllm.model_executor.parameter import BasevLLMParameter

    parameter = BasevLLMParameter.__new__(
        BasevLLMParameter, torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16, device="cpu")
    )
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=model))
            )
        )
    )
    before = parameter.detach().view(torch.uint8).numpy().tobytes()
    pointer = parameter.data_ptr()
    result = diagnostic.served_parameters(llm)
    assert result == [
        {
            "name": "weight",
            "shape": [1, 2],
            "dtype": "torch.bfloat16",
            "sha256": hashlib.sha256(before).hexdigest(),
        }
    ]
    assert parameter.data_ptr() == pointer
    assert parameter.detach().view(torch.uint8).numpy().tobytes() == before
