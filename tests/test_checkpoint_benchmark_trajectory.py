import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/checkpoint_benchmark_trajectory.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_trajectory", PATH)
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)


def rows():
    return pd.DataFrame({"pair_id": ["a", "b", "c"], "protected_group": ["gender"] * 3,
        "protected_attr": ["Male"] * 3, "condition": ["explicit"] * 3,
        "reference_decision": ["hire", "hire", "reject"],
        "decision": ["hire", None, "reject"], "outcome": ["decided", "unparsed", "decided"]})


def test_align_reorders_but_rejects_missing_duplicate_extra_and_relabelled():
    raw = rows()
    assert T.aligned_raw(raw.iloc[::-1], raw)["pair_id"].tolist() == ["a", "b", "c"]
    for bad in [raw.iloc[:2], pd.concat([raw.iloc[:2], raw.iloc[:1]]),
                raw.assign(pair_id=["a", "b", "extra"]),
                raw.assign(reference_decision=["reject", "hire", "reject"])]:
        with pytest.raises(ValueError):
            T.aligned_raw(bad, raw)


def test_legacy_tpr_denominator_is_not_silently_changed():
    result = T.metrics(rows())
    assert result["tpr_vs_gpt4o_pct"] == 100.0
    assert result["tpr_full_denominator_pct"] == 50.0
    assert result["invalid_n"] == 1
    assert result["reference_hire_n"] == 2
    assert result["decided_reference_hire_n"] == 1


def test_no_usable_label_is_fatal():
    with pytest.raises(ValueError, match="no usable"):
        T.metrics(rows().assign(decision=None, outcome="unparsed"))


def test_fingerprint_detects_same_size_content_change(tmp_path):
    path = tmp_path / "input"
    path.write_text("abc")
    bound = T.fingerprint(path)
    path.write_text("xyz")
    with pytest.raises(ValueError, match="frozen input"):
        T.check_fingerprint(bound)


def test_manifest_edit_is_detected(tmp_path):
    record = {"steps": [100]}
    record["binding_sha256"] = T.digest(record)
    T.atomic_json(tmp_path / "manifest.json", record)
    assert T.load_manifest(tmp_path)["steps"] == [100]
    record["steps"].append(125)
    T.atomic_json(tmp_path / "manifest.json", record)
    with pytest.raises(ValueError, match="binding"):
        T.load_manifest(tmp_path)


def test_exact_negative_ten_passes_legacy_guard_but_alerts(tmp_path):
    manifest = {"steps": [{"step": 0}, {"step": 100}, {"step": 125}]}
    manifest["binding_sha256"] = T.digest(manifest)
    T.atomic_json(tmp_path / "manifest.json", manifest)
    for step, tpr in [(0, 60.0), (100, 50.0), (125, 49.99)]:
        folder = T.step_path(tmp_path, step)
        folder.mkdir(parents=True)
        raw = folder / "raw.parquet"
        raw.write_bytes(b"unit-test-placeholder")
        T.atomic_json(folder / "result.json", {"binding_sha256": manifest["binding_sha256"],
            "raw": T.fingerprint(raw), "metrics": {"tpr_vs_gpt4o_pct": tpr},
            "scopes": {"dev": {"delta_tpr_pp": -7.0}},
            "provenance": {"kind": "test"}})
    T.summarize(tmp_path)
    curve = json.loads((tmp_path / "trajectory.json").read_text())["rows"]
    assert curve[1]["full_8pp_alert"] is True
    assert curve[1]["full_fails_10pp_guardrail"] is False
    assert curve[2]["full_fails_10pp_guardrail"] is True
    assert curve[2]["dev_8pp_candidate_alert"] is False


def test_partition_direct_tpr_and_paired_loss_gain_have_separate_denominators():
    current = pd.DataFrame({"pair_id": list("abcde"), "candidate_id": ["dev"] * 3 + ["rem"] * 2,
        "protected_group": ["gender"] * 5, "protected_attr": ["Male"] * 5,
        "condition": ["explicit"] * 5, "reference_decision": ["hire", "hire", "reject",
        "hire", "reject"], "decision": ["reject", "hire", "reject", "reject", "reject"],
        "outcome": ["decided"] * 5})
    parent = current.assign(decision=["hire", "reject", "reject", "hire", "reject"])
    results = T.partition_metrics(current, parent,
                    {"dev": {"dev"}, "historical_remainder": {"rem"}})
    assert results["dev"]["delta_tpr_pp"] == 0
    assert results["dev"]["loss_pp"] == results["dev"]["gain_pp"] == 50
    assert results["historical_remainder"]["delta_tpr_pp"] == -100
    assert results["historical_remainder"]["loss_count"] == 1
    assert results["full"]["delta_tpr_pp"] == pytest.approx(-100 / 3)


def test_first_stop_remains_latched_after_recovery_and_waits_for_missing_prior_steps():
    curve = [{"step": s, "alarm": alarm} for s, alarm in
             [(0, False), (100, False), (125, True), (150, False)]]
    result = T.first_crossing(curve, [0, 100, 125, 150], "alarm")
    assert result["first_observed_crossing_step"] == 125
    assert result["preceding_retained_safe_step"] == 100
    assert result["later_recovery_does_not_revoke_stop"] is True
    incomplete = T.first_crossing(curve[:1] + curve[2:], [0, 100, 125, 150], "alarm")
    assert incomplete["first_crossing_confirmed_among_retained_steps"] is False
    assert incomplete["preceding_retained_safe_step"] is None
    assert incomplete["missing_earlier_steps"] == [100]


def test_four_lanes_keep_one_process_per_gpu_and_score_each_step_once(tmp_path, monkeypatch):
    import threading
    import time
    from types import SimpleNamespace

    manifest = {"steps": [{"step": n} for n in range(18)], "config": {"workers": 4}}
    completed, active, seen = {}, set(), []
    lock = threading.Lock()
    monkeypatch.setattr(T, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(T, "validated_results", lambda *a: dict(completed))
    monkeypatch.setattr(T, "summarize", lambda *a: None)

    def worker(args, env, **kwargs):
        gpu = env["CUDA_VISIBLE_DEVICES"]
        step = int(args[-1])
        with lock:
            assert gpu not in active
            active.add(gpu)
            seen.append(step)
        time.sleep(.005)
        with lock:
            active.remove(gpu)
            completed[step] = {}
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(T.subprocess, "run", worker)
    T.run(tmp_path)
    assert sorted(seen) == list(range(18))
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "complete"


def test_worker_failure_prevents_later_wave(tmp_path, monkeypatch):
    from types import SimpleNamespace

    manifest = {"steps": [{"step": n} for n in range(18)], "config": {"workers": 4}}
    seen = []
    monkeypatch.setattr(T, "load_manifest", lambda *a, **k: manifest)
    monkeypatch.setattr(T, "validated_results", lambda *a: {})
    monkeypatch.setattr(T, "summarize", lambda *a: None)

    def worker(args, **kwargs):
        seen.append(int(args[-1]))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(T.subprocess, "run", worker)
    with pytest.raises(RuntimeError, match="later work stopped"):
        T.run(tmp_path)
    assert len(seen) <= 4
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "failed"
