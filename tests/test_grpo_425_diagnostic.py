"""CPU-only checks for diagnostic identities, coverage and attribution inputs."""

import dataclasses
import importlib.util
import json
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "grpo425_diagnostic", Path(__file__).parents[1] / "scripts/grpo_425_diagnostic.py"
)
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_authorized", True),
        ("rows_per_arm", 52),
        ("slurm_jobs_authorized", 2),
        ("strict_startup_gate_unchanged", False),
    ],
)
def test_protocol_rejects_scope_changes(field, value):
    protocol = D.protocol()
    protocol[field] = value
    with pytest.raises(ValueError, match="protocol differs"):
        D.validate_protocol(protocol)


def test_protocol_preserves_real_greedy_seed_difference():
    protocol = D.protocol()
    assert protocol["historical"]["sampling_seed"] is None
    assert protocol["live"]["sampling_seed"] == 42
    assert protocol["live"]["engine_seed"] == "rank"


@pytest.mark.parametrize(
    "records",
    [
        [{"prompt_id": "a"}],
        [{"prompt_id": "a"}, {"prompt_id": "a"}],
        [{"prompt_id": "a"}, {"prompt_id": "x"}],
    ],
)
def test_unique_coverage_rejects_missing_duplicate_extra(records):
    with pytest.raises(ValueError, match="prompt identities"):
        D.unique_order(records, ["a", "b"])


def test_unique_coverage_restores_order():
    assert (
        D.unique_order([{"prompt_id": "b"}, {"prompt_id": "a"}], ["a", "b"])[0]["prompt_id"] == "a"
    )


def test_controls_are_same_pair_agreeing_and_distinct():
    rows = [
        {"prompt_id": p, "pair_id": pair, "condition": condition, "protected_group": "gender"}
        for p, pair, condition in [
            ("a", "p", "explicit"),
            ("b", "p", "explicit"),
            ("c", "p", "explicit"),
            ("d", "p", "implicit"),
            ("e", "q", "explicit"),
        ]
    ]
    result = D.matched_controls(rows, {"a", "b"})
    assert [r["control_prompt_id"] for r in result] == ["c", "d"]
    assert all(r["pair_id"] == "p" for r in result)


def test_controls_never_cross_pair_to_fill_shortfall():
    rows = [
        {"prompt_id": "a", "pair_id": "p", "condition": "explicit", "protected_group": "gender"}
    ]
    with pytest.raises(ValueError, match="same-pair"):
        D.matched_controls(rows, {"a"})


def test_manifest_binding_rejects_mutation():
    value = {"training_authorized": False}
    value["binding_sha256"] = D.digest(value)
    D.check_binding(value)
    value["training_authorized"] = True
    with pytest.raises(ValueError, match="binding mismatch"):
        D.check_binding(value)


def request(ids=(1, 2), text="wrapped"):
    return SimpleNamespace(
        prompt=text,
        prompt_token_ids=ids,
        outputs=[SimpleNamespace(text='{"decision":"hire"}', token_ids=[3], finish_reason="stop")],
    )


def test_actual_vllm_ids_checked_even_when_text_matches():
    expected = [{"prompt_id": "a", "wrapped_prompt": "wrapped", "input_ids": [1, 2]}]
    records, diffs = D.check_request_outputs([request((1, 4))], expected)
    assert records[0]["actual_input_ids"] == [1, 4]
    assert diffs == [
        {"prompt_id": "a", "text_equal": True, "token_ids_equal": False, "actual_token_ids": [1, 4]}
    ]


def test_actual_vllm_missing_ids_cannot_pass():
    expected = [{"prompt_id": "a", "wrapped_prompt": "wrapped", "input_ids": [1, 2]}]
    _, diffs = D.check_request_outputs([request(None)], expected)
    assert diffs[0]["token_ids_equal"] is False


def test_actual_vllm_complete_inputs_pass():
    expected = [{"prompt_id": "a", "wrapped_prompt": "wrapped", "input_ids": [1, 2]}]
    assert D.check_request_outputs([request()], expected)[1] == []


def test_comparison_keeps_both_directions_and_reference_labels():
    result = D.comparison(
        ["hire", "reject", "hire", "reject"],
        ["reject", "hire", "reject", "hire"],
        ["hire", "hire", "reject", "reject"],
    )
    assert result["decision_differences"] == 4
    assert result["by_reference"] == {
        label: {"hire_to_reject": 1, "reject_to_hire": 1} for label in ["hire", "reject"]
    }


@pytest.mark.parametrize("right", [["hire"], ["hire", None]])
def test_comparison_rejects_partial_or_invalid(right):
    with pytest.raises(ValueError):
        D.comparison(["hire", "reject"], right, ["hire", "reject"])


def test_cached_partial_phase_is_not_success(tmp_path):
    path = tmp_path / "result.json"
    D.atomic(path, {"binding_sha256": "m", "status": "failed"})
    with pytest.raises(ValueError, match="incomplete or foreign"):
        D.bound_result(path, {"binding_sha256": "m"})


def test_completed_phase_fingerprint_is_verified(tmp_path):
    data = tmp_path / "raw.json"
    data.write_text("original")
    path = tmp_path / "result.json"
    D.atomic(path, {"binding_sha256": "m", "status": "complete", "files": [D.fingerprint(data)]})
    data.write_text("changed")
    with pytest.raises(ValueError, match="fingerprint changed"):
        D.bound_result(path, {"binding_sha256": "m"})


def test_base_snapshot_symlink_change_is_detected_even_if_old_blob_is_unchanged(tmp_path):
    first, second = tmp_path / "blob1", tmp_path / "blob2"
    first.write_text("original")
    second.write_text("replacement")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    link = snapshot / "model.safetensors"
    link.symlink_to(first)
    mapping = {link.name: str(first.resolve())}
    D.verify_snapshot_mapping(snapshot, mapping)
    link.unlink()
    link.symlink_to(second)
    with pytest.raises(ValueError, match="snapshot mapping changed"):
        D.verify_snapshot_mapping(snapshot, mapping)


def test_engine_snapshot_preserves_declared_settings_without_object_serializers():
    class Mode(IntEnum):
        COMPILE = 3

    class RuntimeObject:
        def __deepcopy__(self, memo):
            raise AssertionError("must not deepcopy runtime object")

        def __repr__(self):
            raise AssertionError("must not repr runtime object")

        def tolist(self):
            raise AssertionError("must not materialize runtime object")

        def model_dump(self):
            raise AssertionError("must not call runtime serializer")

    @dataclasses.dataclass
    class Settings:
        seed: int
        enabled: bool
        custom_ops: list
        nested: dict
        mode: object
        path: object
        static_forward_context: dict
        runtime: object

    runtime = RuntimeObject()
    settings = Settings(
        42,
        True,
        ["none"],
        {"limit": 16384, "optional": None},
        Mode.COMPILE,
        Path("/local/model"),
        {"layer": runtime},
        runtime,
    )
    config = SimpleNamespace(
        model_config=settings,
        scheduler_config={"max_num_seqs": 256},
        cache_config={"gpu_memory_utilization": 0.7},
        parallel_config={"tensor_parallel_size": 1},
        compilation_config={"mode": Mode.COMPILE, "sizes": (1, 4, 8)},
    )
    result = D.engine_info(SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=config)))
    assert result["model_config"] == {
        "seed": 42,
        "enabled": True,
        "custom_ops": ["none"],
        "nested": {"limit": 16384, "optional": None},
        "mode": 3,
        "path": "/local/model",
        "static_forward_context": {
            "_type": "builtins.dict",
            "_omitted": "runtime layer registry; not traversed",
            "count": 1,
        },
        "runtime": {
            "_type": f"{RuntimeObject.__module__}.{RuntimeObject.__qualname__}",
            "_omitted": "runtime object",
        },
    }
    assert result["scheduler_config"] == {"max_num_seqs": 256}
    assert result["cache_config"] == {"gpu_memory_utilization": 0.7}
    assert result["parallel_config"] == {"tensor_parallel_size": 1}
    assert result["compilation_config"] == {"mode": 3, "sizes": [1, 4, 8]}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert settings.static_forward_context["layer"] is runtime


def test_config_snapshot_detects_cycles_but_keeps_repeated_settings():
    repeated = {"seed": 42}
    circular = [repeated, repeated]
    circular.append(circular)
    assert D.config_snapshot(circular) == [
        {"seed": 42},
        {"seed": 42},
        {"_type": "builtins.list", "_omitted": "cycle"},
    ]
    mapping = {}
    mapping["self"] = mapping
    assert D.config_snapshot(mapping)["self"]["_omitted"] == "cycle"


def test_config_snapshot_handles_non_string_keys_and_nonfinite_values():
    result = D.config_snapshot({1: "one", "inf": float("inf"), "names": {"b", "a"}})
    assert result == {
        "_type": "builtins.dict",
        "entries": [
            [1, "one"],
            ["inf", {"_type": "builtins.float", "_value": "inf"}],
            ["names", ["a", "b"]],
        ],
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_config_snapshot_reads_pydantic_fields_without_model_dump():
    pydantic = pytest.importorskip("pydantic")

    class Settings(pydantic.BaseModel):
        seed: int = 42
        modes: list[str] = pydantic.Field(default_factory=lambda: ["generate"])

        def model_dump(self, **kwargs):
            raise AssertionError("must read declared fields, not call serializer")

    assert D.config_snapshot(Settings()) == {"seed": 42, "modes": ["generate"]}


def test_actual_vllm_runtime_parameter_snapshot_is_cpu_only_and_read_only(monkeypatch):
    torch = pytest.importorskip("torch")
    parameter_module = pytest.importorskip("vllm.model_executor.parameter")
    compilation_module = pytest.importorskip("vllm.config.compilation")
    parameter_type = parameter_module.BasevLLMParameter
    parameter = parameter_type.__new__(parameter_type, torch.arange(4, device="cpu"))
    layer = torch.nn.Module()
    layer.register_parameter("weight", parameter)
    config = compilation_module.CompilationConfig()
    config.static_forward_context["toy_layer"] = layer
    config.inductor_compile_config["diagnostic_parameter"] = parameter
    config.inductor_compile_config["diagnostic_layer"] = layer
    config.inductor_compile_config["diagnostic_dtype"] = torch.bfloat16
    context_identity = id(config.static_forward_context)
    pointer = parameter.data_ptr()
    before = parameter.detach().view(torch.uint8).numpy().tobytes()
    cuda_initialized = torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot must not copy, repr or materialize runtime data")

    monkeypatch.setattr(parameter_type, "__deepcopy__", forbidden)
    monkeypatch.setattr(parameter_type, "__repr__", forbidden)
    monkeypatch.setattr(parameter_type, "tolist", forbidden)
    monkeypatch.setattr(torch.nn.Module, "__repr__", forbidden)
    result = D.config_snapshot(config)
    assert result["static_forward_context"] == {
        "_type": "builtins.dict",
        "_omitted": "runtime layer registry; not traversed",
        "count": 1,
    }
    assert result["inductor_compile_config"]["diagnostic_parameter"] == {
        "_type": "vllm.model_executor.parameter.BasevLLMParameter",
        "_omitted": "runtime tensor or module",
    }
    assert result["inductor_compile_config"]["diagnostic_layer"]["_omitted"] == (
        "runtime tensor or module"
    )
    assert result["inductor_compile_config"]["diagnostic_dtype"] == "torch.bfloat16"
    assert result["backend"] == config.backend
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert id(config.static_forward_context) == context_identity
    assert config.static_forward_context["toy_layer"] is layer
    assert layer.weight is parameter
    assert parameter.device.type == "cpu"
    assert parameter.data_ptr() == pointer
    assert parameter.detach().view(torch.uint8).numpy().tobytes() == before
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("acknowledged", [True, False, None])
def test_live_weight_push_requires_actual_cache_reset_acknowledgement(monkeypatch, acknowledged):
    tensor = SimpleNamespace(to=lambda device: "device-value")
    model = SimpleNamespace(load_weights=lambda values: {"packed.weight"})
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(model_runner=SimpleNamespace(model=model))
            )
        ),
        reset_prefix_cache=lambda: acknowledged,
    )
    monkeypatch.setattr(D, "tensor_inventory", lambda path: {"source.weight": path})
    monkeypatch.setattr(D, "read_tensor", lambda path, name: tensor)
    if acknowledged is True:
        assert D.push_weights(llm, "/unused", ["source.weight"])["reset_prefix_cache"] is True
    else:
        with pytest.raises(RuntimeError, match="cache reset"):
            D.push_weights(llm, "/unused", ["source.weight"])


@pytest.mark.parametrize("smoke_fails", [False, True])
def test_retry_waits_for_disposable_smoke_before_any_formal_arm(tmp_path, monkeypatch, smoke_fails):
    calls = []
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")

    def run(command, **kwargs):
        command_name = next(
            word
            for word in command
            if word
            in {
                "token-check",
                "make-weights",
                "compare-weights",
                "smoke-live",
                "collect-smoke",
                "score-historical",
                "score-live",
                "collect-live",
                "summarize",
            }
        )
        calls.append(command_name)
        return SimpleNamespace(returncode=int(smoke_fails and command_name == "smoke-live"))

    monkeypatch.setattr(D.subprocess, "run", run)
    manifest = {"binding_sha256": "test-binding", "retry": {}}
    if smoke_fails:
        with pytest.raises(RuntimeError, match="live_smoke"):
            D.run(tmp_path, manifest)
        assert "score-historical" not in calls and "score-live" not in calls
    else:
        D.run(tmp_path, manifest)
        assert calls.index("collect-smoke") < calls.index("score-historical")
        assert [name for name in calls if name.startswith("score-")] == [
            "score-historical",
            "score-historical",
            "score-live",
            "score-live",
        ]
