"""No extra source-model inference, no changed membership, no stale average artifacts."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/checkpoint_average_benchmark.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_average_runner", PATH)
A = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(A)
CONFIG = ROOT / "configs/analysis/grpo_v3_all_checkpoint_average.json"


def config():
    return json.loads(CONFIG.read_text())


def raw_rows():
    return pd.DataFrame({"pair_id": list("abc"), "candidate_id": ["dev", "dev", "rem"],
        "protected_group": ["gender"] * 3, "protected_attr": ["Male"] * 3,
        "condition": ["explicit"] * 3, "reference_decision": ["hire", "hire", "reject"],
        "decision": ["hire", "reject", "reject"], "outcome": ["decided"] * 3})


def cache_fixture(tmp_path):
    frame = raw_rows()
    manifest = {"binding_sha256": "same-experiment", "steps": []}
    results = {}
    parent = [{"path": "sft", "files": [{"sha256": "frozen-sft"}]}]
    for step in (0, *A.STEPS):
        raw_path = tmp_path / f"raw-{step}.parquet"
        frame.to_parquet(raw_path, index=False)
        manifest["steps"].append({"step": step, "adapter": f"adapter-{step}",
            "chain": parent if step == 0 else [*parent, {"path": f"adapter-{step}"}]})
        results[step] = {"binding_sha256": "same-experiment", "raw": A.T.fingerprint(raw_path),
                         "metrics": {"invalid_n": 0}}
    return frame, manifest, results


def test_protocol_fixes_exact_sources_and_one_new_inference_only():
    original = config()
    A.validate_config(original)
    for key, value in [("steps", list(reversed(A.STEPS))), ("steps", list(A.STEPS[:-1])),
                       ("new_model_inference_count", 8), ("training_authorized", True),
                       ("expected_output_lora_r", 32), ("minimum_merge_retention", .7),
                       ("tensor_parallel_size", 4), ("no_extra_margin_pass", False)]:
        with pytest.raises(ValueError):
            A.validate_config({**original, key: value})


def test_analysis_rule_cannot_be_tuned_after_freeze():
    changed = config()
    changed["analysis"]["seed"] += 1
    with pytest.raises(ValueError, match="analysis settings"):
        A.validate_config(changed)


def test_average_outputs_cannot_overlap_source_trajectory(tmp_path):
    for out in (tmp_path, tmp_path / "derived"):
        with pytest.raises(ValueError, match="separate"):
            A.checked_output({"output_dir": str(out), "source_trajectory_dir": str(tmp_path)})
    paths = A.checked_output({"output_dir": str(tmp_path / "average"),
                              "source_trajectory_dir": str(tmp_path / "source")})
    assert paths[0].name == "average"


def test_existing_models_are_read_only_verified_cache_inputs(tmp_path):
    frame, manifest, results = cache_fixture(tmp_path)
    selected = A.validate_cached_models(manifest, results, frame)
    assert list(selected) == ["SFT-v3", *[f"step{s}" for s in A.STEPS]]
    assert len(selected) == 8
    del results[475]
    with pytest.raises(ValueError, match="do not regenerate"):
        A.validate_cached_models(manifest, results, frame)


def test_cached_source_wrong_experiment_labels_or_unusable_outputs_refuse(tmp_path):
    frame, manifest, results = cache_fixture(tmp_path)
    bad = copy.deepcopy(results)
    bad[425]["binding_sha256"] = "another-experiment"
    with pytest.raises(ValueError, match="another trajectory"):
        A.validate_cached_models(manifest, bad, frame)
    bad = copy.deepcopy(results)
    bad[425]["metrics"]["invalid_n"] = 1
    with pytest.raises(ValueError, match="unusable"):
        A.validate_cached_models(manifest, bad, frame)
    with pytest.raises(ValueError, match="labels mismatch"):
        A.validate_cached_models(manifest, results,
                                 frame.assign(reference_decision=["reject"] * 3))


def test_all_seven_checkpoints_must_share_identical_sft_chain(tmp_path):
    frame, manifest, results = cache_fixture(tmp_path)
    manifest["steps"][-1]["chain"] = [{"path": "different-sft"}, {"path": "last"}]
    with pytest.raises(ValueError, match="share the frozen SFT"):
        A.validate_cached_models(manifest, results, frame)


def test_manifest_and_input_fingerprints_fail_closed(tmp_path):
    path = tmp_path / "weights"
    path.write_bytes(b"abc")
    one = A.T.fingerprint(path)
    assert A.deduplicate_fingerprints([one, one]) == [one]
    with pytest.raises(ValueError, match="different frozen identities"):
        A.deduplicate_fingerprints([one, {**one, "sha256": "other"}])
    manifest = {"config": config(), "inputs": [one]}
    manifest["binding_sha256"] = A.T.digest(manifest)
    A.T.atomic_json(tmp_path / "manifest.json", manifest)
    assert A.load_manifest(tmp_path)["inputs"] == [one]
    manifest["config"]["steps"] = list(reversed(A.STEPS))
    A.T.atomic_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="binding"):
        A.load_manifest(tmp_path)


def test_receipt_binds_derived_files_and_original_experiment(tmp_path):
    output = tmp_path / "adapter"
    output.write_bytes(b"abc")
    receipt = tmp_path / "receipt.json"
    A.T.atomic_json(receipt, {"binding_sha256": "binding", "files": [A.T.fingerprint(output)]})
    A.check_owned_receipt(receipt, "binding")
    with pytest.raises(ValueError, match="another experiment"):
        A.check_owned_receipt(receipt, "other")
    output.write_bytes(b"xyz")
    with pytest.raises(ValueError, match="frozen input changed"):
        A.check_owned_receipt(receipt, "binding")


def test_chunks_cannot_be_reused_across_different_average_weights_or_order(tmp_path):
    path = tmp_path / "chunk.json"
    chunk = {"binding_sha256": "binding", "model_binding": "mean-model", "start": 0,
             "outputs": ["first", "second"]}
    A.T.atomic_json(path, chunk)
    assert A.read_chunk(path, "binding", "mean-model", 0, 2) == ["first", "second"]
    for args in [("other", "mean-model", 0, 2), ("binding", "other", 0, 2),
                 ("binding", "mean-model", 4, 2), ("binding", "mean-model", 0, 1)]:
        with pytest.raises(ValueError, match="identity/order/count"):
            A.read_chunk(path, *args)


def test_partial_adapter_or_merge_without_success_receipt_is_never_overwritten(tmp_path,
                                                                            monkeypatch):
    def never(*args, **kwargs):
        pytest.fail("partial products must not be overwritten")
    fake = SimpleNamespace(build_average=never, merge_average=never, validate_average=never)
    monkeypatch.setitem(sys.modules, "hiring_bias_mitigation.mitigation.lora_average", fake)
    (tmp_path / "adapter").mkdir()
    with pytest.raises(RuntimeError, match="partial average adapter"):
        A.prepare_merged_model(tmp_path, {"average_adapter_dirs": [], "binding_sha256": "x"})


def test_completed_average_is_not_generated_again(tmp_path, monkeypatch):
    (tmp_path / "average").mkdir()
    (tmp_path / "average" / "result.json").write_text("{}")
    monkeypatch.setattr(A, "load_manifest", lambda *a, **k: {})
    called = []
    monkeypatch.setattr(A, "summarize", lambda out: called.append(out))
    monkeypatch.setattr(A, "prepare_merged_model", lambda *a: pytest.fail("do not rebuild mean"))
    A.run(tmp_path)
    assert called == [tmp_path]


@pytest.mark.parametrize("value", [.79, float("nan"), float("inf"), None, True])
def test_below_gate_or_nonfinite_retention_blocks_inference_and_sdt(value):
    with pytest.raises(ValueError, match="retention"):
        A.validate_retention([{"kept_fraction": value}])


def test_exact_eighty_percent_retention_passes():
    A.validate_retention([{"kept_fraction": .8}])
