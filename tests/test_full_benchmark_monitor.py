"""The pilot must fail before an update on identity errors, and never erase its first stop."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hiring_bias_mitigation.mitigation import full_benchmark_monitor as M

SOURCE_FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
                "trainer_state.json", "training_args.bin", *[f"rng_state_{i}.pth" for i in range(4)]]


def rows(positive=100, negative=20):
    return [{"prompt_id": f"p{i}", "prompt": f"screen {i}", "candidate_id": f"c{i}",
             "pair_id": f"pair{i}", "job_id": f"j{i}", "lang": "en",
             "reference_decision": "hire" if i < positive else "reject"}
            for i in range(positive + negative)]


def decisions(bank, flipped=()):
    return [{"prompt_id": r["prompt_id"],
             "decision": ("reject" if r["reference_decision"] == "hire" else "hire")
                         if r["prompt_id"] in flipped else r["reference_decision"],
             "raw_output": json.dumps({"decision": r["reference_decision"], "feedback": "example"})}
            for r in bank]


def checkpoint(folder, step, missing=()):
    folder = Path(folder) / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_FILES:
        if name not in missing:
            (folder / name).write_text(json.dumps({"global_step": step}) if name == "trainer_state.json" else "x")
    return folder


def contract(tmp_path):
    bank = rows()
    source = checkpoint(tmp_path / "source", 425)
    return {"rows": bank, "positive_rows": bank[:100], "parent": decisions(bank), "start": decisions(bank),
            "manifest": {"continuation_source": {"source_checkpoint": str(source), "world_size": 1,
                        "source_files_sha256": {name: "fixture" for name in SOURCE_FILES}}},
            "manifest_sha256": "frozen-manifest", "identity": {"source": "fixture"}, "identity_sha256": "identity"}


def args():
    return SimpleNamespace(save_steps=5)


def state(step=425):
    return SimpleNamespace(global_step=step, is_world_process_zero=True, save_steps=5)


def control():
    return SimpleNamespace(should_save=False, should_training_stop=False)


@pytest.fixture(autouse=True)
def no_distributed(monkeypatch):
    monkeypatch.setattr(M, "_dist", lambda: None)


def callback(tmp_path, score=None, sync=None):
    pytest.importorskip("transformers")
    return M.callback_class()(contract(tmp_path), score or decisions, sync or (lambda: None), tmp_path / "run")


@pytest.mark.parametrize("delta,expected", [(-.079, False), (-.08, True), (-.081, True), (.02, False)])
def test_inclusive_eight_pp(delta, expected):
    assert M.stop_for_tpr(delta) is expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_delta_fails(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        M.stop_for_tpr(bad)


def test_merge_shards_orders_by_identity_and_rejects_duplicate_missing():
    bank = rows(4, 1)
    full = decisions(bank)
    assert M.merge_records(bank, full[::2], full[1::2]) == full
    with pytest.raises(ValueError, match="duplicate"):
        M.merge_records(bank, full, full[:1])
    with pytest.raises(ValueError, match="missing"):
        M.merge_records(bank, full[:-1])


def test_collective_rank_shards_and_sync_before_scoring(monkeypatch):
    bank = rows(6, 2)
    all_records = decisions(bank)
    events = []
    monkeypatch.setattr(M, "_dist", lambda: SimpleNamespace(get_rank=lambda: 1, get_world_size=lambda: 2))

    def gather(value):
        if value is None:
            return [None, None]
        return [{"error": None, "records": all_records[::2]}, value]

    monkeypatch.setattr(M, "_all_gather", gather)

    def score(local):
        events.append("score")
        assert [r["prompt_id"] for r in local] == [r["prompt_id"] for r in bank[1::2]]
        return decisions(local)

    assert M.collective_score(bank, score, lambda: events.append("sync")) == all_records
    assert events == ["sync", "score"]


def test_other_rank_sync_or_generation_failure_is_fail_closed(monkeypatch):
    bank = rows(3, 1)
    called = []
    monkeypatch.setattr(M, "_all_gather", lambda _: [None, "other rank sync failed"])
    with pytest.raises(RuntimeError, match="sync failed"):
        M.collective_score(bank, lambda r: called.append(r), lambda: None)
    assert not called
    calls = iter([[None, None], [{"error": None, "records": decisions(bank)},
                                {"error": "other rank OOM", "records": None}]])
    monkeypatch.setattr(M, "_all_gather", lambda _: next(calls))
    with pytest.raises(RuntimeError, match="other rank OOM"):
        M.collective_score(bank, decisions, lambda: None)


def test_other_rank_post_generation_validation_cannot_be_overwritten(tmp_path, monkeypatch):
    monitor = callback(tmp_path)
    monkeypatch.setattr(M, "collective_score", lambda bank, *_: decisions(bank))
    monkeypatch.setattr(M, "_all_gather", lambda _: [None, "rank 1 checkpoint decision read failed"])
    with pytest.raises(RuntimeError, match="rank 1 checkpoint decision read failed"):
        monitor.on_train_begin(args(), state(), control())
    assert monitor.aborted and not monitor.startup_gate_passed


def test_startup_requires_exact_frozen_425_before_training(tmp_path):
    monitor = callback(tmp_path, lambda bank: decisions(bank, {"p0"}))
    with pytest.raises(RuntimeError, match="differs on 1"):
        monitor.on_train_begin(args(), state(), control())
    saved = json.loads(monitor._state_path().read_text())
    assert saved["stop"] and not saved["startup_gate_passed"]
    assert monitor.last_record["full"] is not None
    assert not monitor.last_record["valid"]


def test_startup_sync_failure_and_stale_save_cadence_reject(tmp_path):
    def sync():
        raise ValueError("bad sync")

    monitor = callback(tmp_path, sync=sync)
    with pytest.raises(RuntimeError, match="bad sync"):
        monitor.on_train_begin(args(), state(), control())
    changed = state()
    changed.save_steps = 25
    with pytest.raises(RuntimeError, match="cadence"):
        monitor.on_train_begin(args(), changed, control())


def test_periodic_positive_only_saves_and_full_grid_reuses_one_pass(tmp_path):
    counts = []

    def score(bank):
        counts.append(len(bank))
        return decisions(bank)

    monitor = callback(tmp_path, score)
    monitor.on_train_begin(args(), state(), control())
    ctl = control()
    monitor.on_step_end(args(), state(430), ctl)
    assert ctl.should_save and not ctl.should_training_stop
    assert monitor.last_record["full"] is None
    assert monitor.last_record["positive"]["metrics"]["fpr"] is None
    monitor.on_step_end(args(), state(450), control())
    assert counts == [120, 100, 120]
    assert monitor.last_record["full"]["metrics"]["n"] == 120


def test_exact_eight_pp_triggers_full_and_keeps_prior_saved_safe(tmp_path):
    flipped, counts = set(), []

    def score(bank):
        counts.append(len(bank))
        return decisions(bank, flipped)

    monitor = callback(tmp_path, score)
    monitor.on_train_begin(args(), state(), control())
    monitor.on_step_end(args(), state(430), control())
    saved = checkpoint(monitor.output_dir, 430)
    monitor.on_save(args(), state(430), control())
    assert monitor.last_safe_checkpoint == str(saved)
    flipped.update(f"p{i}" for i in range(8))
    ctl = control()
    monitor.on_step_end(args(), state(435), ctl)
    assert counts[-2:] == [100, 20]  # Positive pass is reused in the full triggering result.
    assert ctl.should_training_stop and ctl.should_save and monitor.stop_step == 435
    assert monitor.last_record["full"]["metrics"]["delta_tpr"] == -.08
    checkpoint(monitor.output_dir, 435)
    monitor.on_save(args(), state(435), control())
    monitor.on_train_end(args(), state(435), control())
    final = json.loads(monitor._state_path().read_text())
    assert final["last_safe_step"] == 430 and final["terminal_step"] == 435
    assert final["normal_train_end"] and not final["training_completed"]
    flipped.clear()
    before = len(counts)
    monitor.on_step_end(args(), state(440), control())
    assert len(counts) == before and monitor.stop_step == 435


def test_failed_supplementary_reject_pass_does_not_erase_first_stop(tmp_path):
    monitor = callback(tmp_path)
    monitor.on_train_begin(args(), state(), control())

    def score(bank):
        if bank[0]["reference_decision"] == "reject":
            raise RuntimeError("reject pass failure")
        return decisions(bank, {f"p{i}" for i in range(8)})

    monitor.score_local = score
    monitor.on_step_end(args(), state(430), control())
    saved = json.loads(monitor._state_path().read_text())
    assert saved["stop"] and saved["stop_step"] == 430
    assert saved["reason"] == "direct_full_benchmark_tpr_drop_8pp"
    assert not monitor.last_record["valid"] and monitor.last_record["full"] is None
    monitor.on_train_end(args(), state(430), control())
    assert not monitor.normal_train_end


def test_incomplete_rng_checkpoint_cannot_become_safe_or_normal_terminal(tmp_path):
    monitor = callback(tmp_path)
    monitor.on_train_begin(args(), state(), control())
    monitor.on_step_end(args(), state(500), control())
    checkpoint(monitor.output_dir, 500, missing={"rng_state_3.pth"})
    ctl = control()
    monitor.on_save(args(), state(500), ctl)
    assert ctl.should_training_stop and monitor.aborted and monitor.last_safe_step == 425
    monitor.on_train_end(args(), state(500), control())
    assert not monitor.normal_train_end


def test_complete_target_has_exact_terminal_and_last_safe(tmp_path):
    monitor = callback(tmp_path)
    monitor.on_train_begin(args(), state(), control())
    monitor.on_step_end(args(), state(500), control())
    saved = checkpoint(monitor.output_dir, 500)
    monitor.on_save(args(), state(500), control())
    monitor.on_train_end(args(), state(500), control())
    result = json.loads(monitor._state_path().read_text())
    assert result["normal_train_end"] and result["training_completed"] and result["terminal_full_valid"]
    assert result["terminal_checkpoint"] == result["last_safe_checkpoint"] == str(saved)


def test_resume_matches_record_and_keeps_immutable_history_files(tmp_path):
    monitor = callback(tmp_path)
    monitor.on_train_begin(args(), state(), control())
    monitor.on_step_end(args(), state(430), control())
    checkpoint(monitor.output_dir, 430)
    monitor.on_save(args(), state(430), control())
    old_fp = monitor.last_record["positive"]["decisions_file"]
    resumed = callback(tmp_path)
    resumed.on_train_begin(args(), state(430), control())
    assert resumed.checks == 3 and resumed.startup_gate_passed
    M.verify_file(old_fp)
    assert resumed.last_record["positive"]["decisions_file"]["path"] != old_fp["path"]
    resumed_bad = callback(tmp_path, lambda bank: decisions(bank, {"p0"}))
    with pytest.raises(RuntimeError, match="resumed policy differs"):
        resumed_bad.on_train_begin(args(), state(430), control())


def test_stopped_run_and_missing_history_cannot_resume(tmp_path):
    monitor = callback(tmp_path)
    with pytest.raises(RuntimeError, match="only the frozen 425"):
        monitor.on_train_begin(args(), state(430), control())
    monitor.on_train_begin(args(), state(), control())
    monitor.score_local = lambda bank: decisions(bank, {f"p{i}" for i in range(8)})
    monitor.on_step_end(args(), state(430), control())
    with pytest.raises(RuntimeError, match="stopped or completed"):
        callback(tmp_path).on_train_begin(args(), state(430), control())


def test_resume_rejects_decision_file_tampering(tmp_path):
    monitor = callback(tmp_path)
    monitor.on_train_begin(args(), state(), control())
    monitor.on_step_end(args(), state(430), control())
    checkpoint(monitor.output_dir, 430)
    monitor.on_save(args(), state(430), control())
    path = Path(monitor.last_record["positive"]["decisions_file"]["path"])
    path.write_bytes(path.read_bytes() + b"tampering")
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        callback(tmp_path).on_train_begin(args(), state(430), control())


def test_training_overlap_checks_full_bank_including_reject_candidates(tmp_path):
    frozen = contract(tmp_path)
    train = pd.DataFrame({"candidate_id": [f"t{i}" for i in range(9140)],
                         "pair_id": [f"tp{i}" for i in range(9140)],
                         "job_id": [f"tj{i}" for i in range(9140)]})
    M.validate_train_isolation(frozen, train)
    for key in ("candidate_id", "pair_id", "job_id"):
        changed = train.copy()
        changed.loc[0, key] = frozen["rows"][-1][key]
        with pytest.raises(ValueError, match=key):
            M.validate_train_isolation(frozen, changed)


def test_record_failure_stops_instead_of_continuing(tmp_path, monkeypatch):
    monitor = callback(tmp_path)
    monitor.on_train_begin(args(), state(), control())

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(monitor, "_persist_decisions", fail)
    ctl = control()
    monitor.on_step_end(args(), state(430), ctl)
    assert monitor.aborted and ctl.should_training_stop and "record_failed" in monitor.reason


def freeze_fixture(tmp_path):
    bank = rows(10350, 20700)
    files = {}
    for key, value in [("rows_file", bank), ("parent_decisions_file", decisions(bank)),
                       ("start_decisions_file", decisions(bank))]:
        path = tmp_path / f"{key}.parquet"
        pd.DataFrame(value).to_parquet(path, index=False)
        files[key] = M.fingerprint(path)
    lengths = tmp_path / "lengths.parquet"
    pd.DataFrame({"input_tokens": [100]}).to_parquet(lengths)
    manifest = {"schema": M.SCHEMA, "training_sft_checkpoint": "SFT",
        "benchmark_previously_exposed": True, "independent_test_set": False,
        "inputs": [], "runtime_code": [], "software": {},
        "policy": {"training_authorized": True, "scope": "this_pilot_only", "source_step": 425,
            "target_step": 500, "check_every_steps": 5, "save_every_steps": 5,
            "full_steps": [425, 450, 475, 500], "stop_delta_tpr_pp": -8.0,
            "original_final_tpr_guardrail_pp": -10.0, "first_stop_irreversible": True},
        "tokenization": {"rows": 31050, "max_input_tokens": 100, "max_model_len": 4096,
                         "lengths_file": M.fingerprint(lengths)},
        "scorer": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "seed": 42,
                   "chat_template_kwargs": {"enable_thinking": False}}, **files}
    return manifest


def save_manifest(tmp_path, manifest):
    manifest = dict(manifest)
    manifest.pop("binding_sha256", None)
    manifest["binding_sha256"] = M.digest(manifest)
    path = tmp_path / "manifest.json"
    M.atomic_json(path, manifest)
    return {"manifest_file": str(path), "manifest_sha256": M.fingerprint(path)["sha256"]}


def test_load_frozen_bank_checks_hash_coverage_policy_and_parent(tmp_path):
    manifest = freeze_fixture(tmp_path)
    settings = save_manifest(tmp_path, manifest)
    loaded = M.load_contract(settings, "SFT", lambda p: p)
    assert len(loaded["rows"]) == 31050 and len(loaded["positive_rows"]) == 10350
    assert loaded["identity_sha256"] == M.digest(loaded["identity"])
    with pytest.raises(ValueError, match="SFT parent"):
        M.load_contract(settings, "other-SFT", lambda p: p)
    manifest["policy"]["training_authorized"] = False
    with pytest.raises(ValueError, match="policy"):
        M.load_contract(save_manifest(tmp_path, manifest), "SFT", lambda p: p)
    manifest["policy"]["training_authorized"] = True
    manifest["tokenization"]["max_input_tokens"] = 3585
    with pytest.raises(ValueError, match="capacity"):
        M.load_contract(save_manifest(tmp_path, manifest), "SFT", lambda p: p)
    manifest["tokenization"]["max_input_tokens"] = 100
    settings = save_manifest(tmp_path, manifest)
    Path(manifest["rows_file"]["path"]).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="fingerprint"):
        M.load_contract(settings, "SFT", lambda p: p)


def test_freeze_length_limit_includes_512_completion_tokens(monkeypatch):
    import importlib.util

    from hiring_bias_mitigation.eval import backends

    path = Path(__file__).resolve().parents[1] / "scripts/freeze_grpo_low_lr_full_monitor.py"
    spec = importlib.util.spec_from_file_location("freeze_pilot_test", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    monkeypatch.setattr(backends, "_chat_wrap", lambda tokenizer, text, kwargs: text)
    frame = pd.DataFrame({"prompt": ["example"]})

    def tokenizer(texts, **kwargs):
        assert kwargs == {"add_special_tokens": True, "padding": False, "truncation": False}
        return {"input_ids": [[1] * 3584 for _ in texts]}

    assert helper.tokenize_lengths(frame, tokenizer) == [3584]
    with pytest.raises(ValueError, match="exceeds"):
        helper.tokenize_lengths(frame, lambda texts, **kwargs: {"input_ids": [[1] * 3585]})
