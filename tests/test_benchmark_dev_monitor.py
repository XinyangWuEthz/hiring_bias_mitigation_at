"""Direct development TPR must never silently fall back to a projection or a new parent."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hiring_bias_mitigation.mitigation import benchmark_dev_monitor as M


def rows(n=25, rejects=2):
    return [{"prompt_id": f"p{i}", "prompt": f"screen {i}", "candidate_id": f"c{i // 2}",
             "job_id": f"j{i % 3}", "reference_decision": "hire" if i < n else "reject",
             "lang": "en"} for i in range(n + rejects)]


def decisions(bank, values=None):
    values = values or [r["reference_decision"] for r in bank]
    return [{"prompt_id": r["prompt_id"], "decision": d} for r, d in zip(bank, values)]


def contract(tmp_path, bank=None):
    bank = rows() if bank is None else bank
    parent = decisions(bank)
    paths = {}
    for name, value in (("rows", bank), ("parent_decisions", parent)):
        p = tmp_path / f"{name}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in value))
        paths[f"{name}_file"] = str(p)
        paths[f"{name}_sha256"] = M.sha256_file(p)
    manifest = {"schema": M.SCHEMA, **paths,
                "parent_model": {"run": "SFT-v3", "identity_sha256": "a"*64,
                                 "sft_checkpoint": "outputs/sft/v3"},
                "scorer": {"max_new_tokens": 512, "temperature": 0.0, "top_p": 1.0,
                           "chat_template_kwargs": {"enable_thinking": False}},
                "split": {"candidate_ids": sorted({r["candidate_id"] for r in bank}),
                          "remaining_candidate_ids": ["unseen-candidate"],
                          "benchmark_previously_exposed": True}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    settings = {"manifest_file": str(manifest_path),
                "manifest_sha256": M.sha256_file(manifest_path), "steps": 25,
                "tpr_drop_stop": .08}
    return M.load_contract(settings, "outputs/sft/v3", lambda x: x), settings


def state(step=0, zero=True):
    return SimpleNamespace(global_step=step, is_world_process_zero=zero)


def control():
    return SimpleNamespace(should_save=False, should_training_stop=False)


def monitor(tmp_path, score=None, bank=None, sync=None):
    pytest.importorskip("transformers")
    frozen, _ = contract(tmp_path, bank)
    return M.callback_class()(frozen, score or (lambda r: decisions(r)), sync or (lambda: None),
                              tmp_path / "run")


def test_direct_label_conditional_rates_and_loss_gain():
    bank = rows(3, 2)
    parent = decisions(bank, ["hire", "hire", "reject", "hire", "reject"])
    current = decisions(bank, ["reject", "hire", "hire", "reject", "reject"])
    out = M.direct_metrics(bank, parent, current)
    assert out["tpr"] == out["parent_tpr"] == pytest.approx(2/3)
    assert out["tpr_loss"] == out["tpr_gain"] == pytest.approx(1/3)
    assert out["delta_tpr"] == 0
    assert out["delta_fpr"] == -.5 and out["fpr_loss_count"] == 1
    assert out["n_candidates"] == 3 and out["n_jobs"] == 3


def test_positive_only_scope_never_invents_fpr():
    bank = rows(3, 0)
    out = M.direct_metrics(bank, decisions(bank), decisions(bank))
    assert out["scope"] == "reference_hire_only" and out["tpr"] == 1
    assert out["fpr"] is None and out["delta_fpr"] is None and out["n_ref_reject"] == 0


@pytest.mark.parametrize("delta,stop", [(-.079, False), (-.08, True), (-.081, True), (.1, False)])
def test_threshold_is_inclusive(delta, stop):
    assert M.stop_for_tpr(delta, .08) is stop


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_reading_fails(value):
    with pytest.raises(ValueError):
        M.stop_for_tpr(value, .08)


@pytest.mark.parametrize("threshold", [0, -1, 1.01, float("nan")])
def test_invalid_threshold_fails(threshold):
    with pytest.raises(ValueError):
        M.stop_for_tpr(0, threshold)


@pytest.mark.parametrize("bad", [None, "review", "", float("nan"), 1])
def test_invalid_decisions_fail_closed(bad):
    bank = rows()
    current = decisions(bank)
    current[0]["decision"] = bad
    with pytest.raises(ValueError, match="invalid or missing"):
        M.direct_metrics(bank, decisions(bank), current)


def test_missing_duplicate_and_reordered_decisions_fail():
    bank = rows()
    for current in (decisions(bank)[:-1], decisions(bank)[::-1],
                    [decisions(bank)[0]] * len(bank)):
        with pytest.raises(ValueError):
            M.direct_metrics(bank, decisions(bank), current)


def test_no_positive_or_duplicate_rows_fail():
    for bank in (rows(0, 2), [rows()[0], rows()[0]]):
        with pytest.raises(ValueError):
            M.validate_rows(bank)


def test_contract_freezes_parent_rows_and_order(tmp_path):
    frozen, settings = contract(tmp_path)
    assert frozen["identity"]["scorer"]["max_new_tokens"] == 512
    assert frozen["identity_sha256"] == M.identity_hash(frozen["identity"])
    with pytest.raises(ValueError, match="training SFT parent"):
        M.load_contract(settings, "other-parent", lambda x: x)
    p = Path(settings["manifest_file"])
    p.write_text(p.read_text() + " ")
    with pytest.raises(ValueError, match="manifest SHA256"):
        M.load_contract(settings, "outputs/sft/v3", lambda x: x)


def test_changed_row_file_rejected_even_with_same_length(tmp_path):
    frozen, settings = contract(tmp_path)
    p = Path(frozen["manifest"]["rows_file"])
    p.write_text(p.read_text().replace("screen 0", "different prompt"))
    with pytest.raises(ValueError, match="rows SHA256"):
        M.load_contract(settings, "outputs/sft/v3", lambda x: x)


@pytest.mark.parametrize("kind", ["candidate_overlap", "unacknowledged_exposure", "scorer"])
def test_invalid_frozen_contract_rejected(tmp_path, kind):
    frozen, settings = contract(tmp_path)
    manifest = frozen["manifest"]
    if kind == "candidate_overlap":
        manifest["split"]["remaining_candidate_ids"].append("c0")
    elif kind == "unacknowledged_exposure":
        manifest["split"]["benchmark_previously_exposed"] = False
    else:
        manifest["scorer"]["max_new_tokens"] = 96
    path = Path(settings["manifest_file"])
    path.write_text(json.dumps(manifest))
    settings["manifest_sha256"] = M.sha256_file(path)
    with pytest.raises(ValueError):
        M.load_contract(settings, "outputs/sft/v3", lambda x: x)


def test_positive_subset_of_dev_candidates_allowed(tmp_path):
    frozen, settings = contract(tmp_path, rows(3, 0))
    frozen["manifest"]["split"]["candidate_ids"].append("dev-reject-only")
    path = Path(settings["manifest_file"])
    path.write_text(json.dumps(frozen["manifest"]))
    settings["manifest_sha256"] = M.sha256_file(path)
    assert len(M.load_contract(settings, "outputs/sft/v3", lambda x: x)["rows"]) == 3


def test_training_overlap_rejected():
    with pytest.raises(ValueError, match="in training"):
        M.validate_train_isolation(rows(), ["c0"])
    M.validate_train_isolation(rows(), ["other"])


def test_monitor_checks_initial_periodic_and_final_and_stops_at_exact_threshold(tmp_path):
    calls = []
    def score(bank):
        calls.append(len(bank))
        out = decisions(bank)
        if len(calls) == 3:
            out[0]["decision"] = out[1]["decision"] = "reject"
        return out
    m = monitor(tmp_path, score, rows(25, 0))
    ctl = control()
    m.on_train_begin(None, state(), ctl)
    m.on_step_end(None, state(24), ctl)
    m.on_step_end(None, state(25), ctl)
    m.on_train_end(None, state(28), ctl)
    assert calls == [25, 25, 25]
    assert m.aborted and ctl.should_save and ctl.should_training_stop
    lines = (tmp_path / "run" / "benchmark_dev_monitor.jsonl").read_text().splitlines()
    results = [json.loads(x) for x in lines]
    assert [x["event"] for x in results] == ["start", "periodic", "final"]
    assert results[-1]["delta_tpr"] == -.08 and results[-1]["checks"] == 3
    m.on_train_end(None, state(28), ctl)
    assert len(calls) == 3


def test_startup_parent_mismatch_aborts_before_updates(tmp_path):
    def score(bank):
        out = decisions(bank)
        out[0]["decision"] = "reject"
        return out
    m = monitor(tmp_path, score)
    with pytest.raises(RuntimeError, match="starting policy differs"):
        m.on_train_begin(None, state(), control())


def test_resume_requires_same_identity_and_accumulates_checks(tmp_path):
    m = monitor(tmp_path)
    m.on_train_begin(None, state(), control())
    m.on_step_end(None, state(25), control())
    resumed = monitor(tmp_path)
    resumed.on_train_begin(None, state(25), control())
    assert resumed.checks == 3
    changed = monitor(tmp_path)
    changed.identity = "changed-label-prompt-or-threshold"
    with pytest.raises(RuntimeError, match="identity mismatch"):
        changed.on_train_begin(None, state(25), control())


def test_resume_missing_ahead_or_stopped_state_is_rejected(tmp_path):
    m = monitor(tmp_path)
    with pytest.raises(RuntimeError, match="without monitor state"):
        m.on_train_begin(None, state(25), control())
    m.on_train_begin(None, state(), control())
    m.on_step_end(None, state(50), control())
    with pytest.raises(RuntimeError, match="ahead of checkpoint"):
        monitor(tmp_path).on_train_begin(None, state(25), control())
    saved = json.loads(m._state_path().read_text())
    saved["stop"] = True
    m._state_path().write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="previously stopped"):
        monitor(tmp_path).on_train_begin(None, state(50), control())


def test_score_and_sync_exceptions_become_saved_stop(tmp_path):
    m = monitor(tmp_path)
    m.on_train_begin(None, state(), control())
    def fail(_):
        raise ValueError("missing output")
    m.score = fail
    ctl = control()
    m.on_step_end(None, state(25), ctl)
    assert m.aborted and ctl.should_save and "missing output" in m.reason
    assert json.loads(m._state_path().read_text())["stop"] is True
    other = tmp_path / "sync-failure"
    other.mkdir()
    def broken_sync():
        raise ValueError("weight sync failed")
    with pytest.raises(RuntimeError, match="weight sync failed"):
        monitor(other, sync=broken_sync).on_train_begin(None, state(), control())


def test_all_ranks_sync_and_receive_rank_zero_failure(tmp_path, monkeypatch):
    m = monitor(tmp_path)
    calls = []
    m.sync = lambda: calls.append("sync")
    m.score = lambda _: pytest.fail("nonzero rank must not score")
    from hiring_bias_mitigation.mitigation import grpo
    monkeypatch.setattr(grpo, "_broadcast", lambda x: x)
    # callback captures broadcast at class creation, so construct after patch.
    m = M.callback_class()(m.contract, m.score, m.sync, tmp_path / "run")
    message = {"checks": 2, "stop": True, "reason": "rank zero invalid result"}
    monkeypatch.setattr(grpo, "_broadcast", lambda x: message)
    m = M.callback_class()(m.contract, m.score, m.sync, tmp_path / "run")
    ctl = control()
    m.on_step_end(None, state(25, False), ctl)
    assert calls == ["sync"] and m.aborted and ctl.should_training_stop


def test_incomplete_resume_history_is_rejected(tmp_path):
    m = monitor(tmp_path)
    m.on_train_begin(None, state(), control())
    m.on_step_end(None, state(25), control())
    path = tmp_path / "run" / "benchmark_dev_monitor.jsonl"
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(RuntimeError, match="incomplete or mismatched"):
        monitor(tmp_path).on_train_begin(None, state(25), control())


def test_jsonl_uses_physical_lines_not_unicode_text_separators(tmp_path):
    bank = rows(2, 0)
    bank[0]["prompt"] = "CV line one\u2028CV line two\u0085JD"
    path = tmp_path / "unicode.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in bank))
    assert M._read_records(path) == bank


def test_contract_identity_pins_runtime_scorer_and_callback_code(tmp_path):
    frozen, _ = contract(tmp_path)
    hashes = frozen["identity"]["runtime_code_sha256"]
    assert set(hashes) == {"benchmark_dev_monitor.py", "grpo.py",
                           "eval/parsing.py", "eval/backends.py"}
    assert all(len(v) == 64 for v in hashes.values())


def test_manifest_relative_artifact_paths_resolve_beside_manifest(tmp_path):
    frozen, settings = contract(tmp_path)
    manifest = frozen["manifest"]
    manifest["rows_file"] = "rows.jsonl"
    manifest["parent_decisions_file"] = "parent_decisions.jsonl"
    path = Path(settings["manifest_file"])
    path.write_text(json.dumps(manifest))
    settings["manifest_sha256"] = M.sha256_file(path)
    assert len(M.load_contract(settings, "outputs/sft/v3", lambda x: x)["rows"]) == 27


def test_safe_selector_requires_saved_checkpoint_and_stops_at_first_stop():
    def record(step, delta, stop=False):
        return {"step": step, "delta_tpr": delta, "tpr_drop_stop": .08,
                "valid": True, "stop": stop}
    history = [record(0, 0), record(25, -.01), record(50, -.08, True), record(75, 0)]
    assert M.choose_latest_safe_record(history, lambda s: True)["step"] == 25
    assert M.choose_latest_safe_record(history, lambda s: False)["step"] == 0
    history[2]["stop"] = False  # corrupted flag cannot make a threshold breach safe
    assert M.choose_latest_safe_record(history, lambda s: True)["step"] == 25


def test_last_safe_checkpoint_advances_only_after_saved_pass_and_never_to_trigger(tmp_path):
    m = monitor(tmp_path, bank=rows(25, 0))
    m.on_train_begin(None, state(), control())
    m.on_step_end(None, state(25), control())
    assert m.last_safe_step == 0
    saved = tmp_path / "run" / "checkpoint-25"
    saved.mkdir()
    (saved / "trainer_state.json").write_text("{}")
    (saved / "adapter_model.safetensors").write_bytes(b"fixture")
    m.on_save(None, state(25), control())
    assert m.last_safe_step == 25 and m.last_safe_checkpoint == str(saved)
    def fail(bank):
        out = decisions(bank)
        out[0]["decision"] = out[1]["decision"] = "reject"
        return out
    m.score = fail
    m.on_step_end(None, state(50), control())
    trigger = tmp_path / "run" / "checkpoint-50"
    trigger.mkdir()
    (trigger / "trainer_state.json").write_text("{}")
    (trigger / "adapter_model.safetensors").write_bytes(b"fixture")
    m.on_save(None, state(50), control())
    assert m.last_safe_step == 25
    status = json.loads(m._state_path().read_text())
    assert status["stop"] and status["last_safe_step"] == 25


def test_prepared_monitor_contract_cannot_authorize_training(tmp_path):
    frozen, _ = contract(tmp_path)
    with pytest.raises(ValueError, match="does not authorize training"):
        M.require_training_ready(frozen)
    frozen["manifest"]["policy"] = {"training_authorized": True,
                                    "check_every_optimizer_steps": 25,
                                    "stop_delta_tpr_pp": -8.0}
    M.require_training_ready(frozen)
    frozen["manifest"]["policy"]["stop_delta_tpr_pp"] = -10.0
    with pytest.raises(ValueError, match="alarm differs"):
        M.require_training_ready(frozen)


def test_training_candidate_in_dev_reject_subset_is_also_rejected():
    positive_rows = rows(3, 0)
    all_dev_candidates = [r["candidate_id"] for r in positive_rows] + ["dev-reject-only"]
    with pytest.raises(ValueError, match="in training"):
        M.validate_train_isolation(positive_rows, ["dev-reject-only"], all_dev_candidates)
