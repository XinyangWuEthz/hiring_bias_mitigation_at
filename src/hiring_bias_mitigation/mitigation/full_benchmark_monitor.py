"""Pilot-only direct benchmark guardrail; the historical development monitor is untouched."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .benchmark_dev_monitor import direct_metrics, validate_rows, validated_decisions

SCHEMA = "grpo-low-lr-full-benchmark-monitor-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def verify_file(record):
    if fingerprint(record["path"]) != record:
        raise ValueError(f"full benchmark monitor: fingerprint mismatch: {record['path']}")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def stop_for_tpr(delta):
    if not math.isfinite(float(delta)):
        raise ValueError("full benchmark monitor: nonfinite TPR change")
    return delta <= -.08 or math.isclose(delta, -.08, abs_tol=1e-12)


def subset_records(rows, records, wanted):
    validated_decisions(rows, records)
    by_id = {record["prompt_id"]: record for record in records}
    return [by_id[row["prompt_id"]] for row in wanted]


def merge_records(rows, *parts):
    joined = [record for part in parts for record in part]
    if len(joined) != len(rows) or len({r["prompt_id"] for r in joined}) != len(rows):
        raise ValueError("full benchmark monitor: missing or duplicate gathered rows")
    by_id = {record["prompt_id"]: record for record in joined}
    if set(by_id) != {r["prompt_id"] for r in rows}:
        raise ValueError("full benchmark monitor: gathered prompt identities differ")
    ordered = [by_id[row["prompt_id"]] for row in rows]
    validated_decisions(rows, ordered)
    return ordered


def load_contract(settings, training_sft_checkpoint, resolve_path):
    import importlib.metadata

    import pandas as pd

    path = Path(resolve_path(settings["manifest_file"]))
    manifest_fp = fingerprint(path)
    if manifest_fp["sha256"] != settings["manifest_sha256"]:
        raise ValueError("full benchmark monitor: manifest SHA mismatch")
    manifest = json.loads(path.read_text())
    frozen_binding = manifest["binding_sha256"]
    if digest({k: v for k, v in manifest.items() if k != "binding_sha256"}) != frozen_binding:
        raise ValueError("full benchmark monitor: manifest binding mismatch")
    if manifest["schema"] != SCHEMA or manifest["training_sft_checkpoint"] != training_sft_checkpoint:
        raise ValueError("full benchmark monitor: schema or SFT parent mismatch")
    policy = manifest["policy"]
    expected = {"training_authorized": True, "scope": "this_pilot_only", "source_step": 425,
                "target_step": 500, "check_every_steps": 5, "save_every_steps": 5,
                "full_steps": [425, 450, 475, 500], "stop_delta_tpr_pp": -8.0,
                "original_final_tpr_guardrail_pp": -10.0, "first_stop_irreversible": True}
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError("full benchmark monitor: pilot policy mismatch")
    if manifest["benchmark_previously_exposed"] is not True or manifest["independent_test_set"] is not False:
        raise ValueError("full benchmark monitor: prior exposure declaration missing")
    for item in manifest["inputs"]:
        verify_file(item)
    if {name: importlib.metadata.version(name) for name in manifest["software"]} != manifest["software"]:
        raise ValueError("full benchmark monitor: software differs from frozen benchmark")
    verify_file(manifest["tokenization"]["lengths_file"])
    loaded = {}
    for field, key in [("rows", "rows_file"), ("parent", "parent_decisions_file"),
                       ("start", "start_decisions_file")]:
        verify_file(manifest[key])
        loaded[field] = pd.read_parquet(manifest[key]["path"]).to_dict("records")
    rows, parent, start = loaded["rows"], loaded["parent"], loaded["start"]
    validate_rows(rows)
    validated_decisions(rows, parent)
    validated_decisions(rows, start)
    if len(rows) != 31050 or sum(row["reference_decision"] == "hire" for row in rows) != 10350:
        raise ValueError("full benchmark monitor: bank coverage differs")
    tokenization = manifest["tokenization"]
    if (tokenization["max_input_tokens"] + 512 > tokenization["max_model_len"]
            or tokenization["max_model_len"] != 4096 or tokenization["rows"] != len(rows)):
        raise ValueError("full benchmark monitor: frozen bank exceeds training engine capacity")
    scorer = manifest["scorer"]
    if scorer != {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "seed": 42,
                  "chat_template_kwargs": {"enable_thinking": False}}:
        raise ValueError("full benchmark monitor: original greedy scorer differs")
    runtime = {item["path"]: item["sha256"] for item in manifest["runtime_code"]}
    for item in manifest["runtime_code"]:
        verify_file(item)
    identity = {"schema": SCHEMA, "manifest_sha256": manifest_fp["sha256"],
                "binding_sha256": frozen_binding, "runtime_code_sha256": runtime}
    return {**loaded, "positive_rows": [r for r in rows if r["reference_decision"] == "hire"],
            "manifest": manifest, "manifest_sha256": manifest_fp["sha256"],
            "identity": identity, "identity_sha256": digest(identity)}


def validate_train_isolation(contract, train_rows):
    if len(train_rows) != 9140:
        raise ValueError("full benchmark monitor: original 9140 training rows required")
    for key in ("candidate_id", "pair_id", "job_id"):
        bank_ids = {r[key] for r in contract["rows"]}
        training_ids = set(train_rows[key])
        if bank_ids & training_ids:
            raise ValueError(f"full benchmark monitor: training overlap in {key}")


def served_decisions(vllm_generation, tokenizer, rows, scorer):
    from vllm import SamplingParams

    from ..eval.backends import _chat_wrap
    from ..eval.parsing import parse_output

    if not rows:
        return []
    texts = [_chat_wrap(tokenizer, r["prompt"], scorer["chat_template_kwargs"]) for r in rows]
    params = SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0, n=1, seed=scorer["seed"])
    outputs = vllm_generation.llm.generate(texts, params, use_tqdm=False)
    if len(outputs) != len(rows):
        raise ValueError("full benchmark monitor: vLLM output count differs")
    records = []
    for row, prompt, output in zip(rows, texts, outputs):
        if output.prompt != prompt or len(output.outputs) != 1:
            raise ValueError("full benchmark monitor: vLLM prompt order or sample count differs")
        raw = output.outputs[0].text
        records.append({"prompt_id": row["prompt_id"], "raw_output": raw,
                        "decision": parse_output(raw, row["lang"]).decision})
    validated_decisions(rows, records)
    return records


def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


def _all_gather(value):
    dist = _dist()
    if dist is None:
        return [value]
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    return gathered


def _broadcast(value):
    dist = _dist()
    if dist is None:
        return value
    values = [value]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def collective_score(rows, score_local, sync):
    """Every rank syncs and resets its vLLM cache before its deterministic row shard."""
    error = None
    try:
        sync()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = _all_gather(error)
    if any(errors):
        raise RuntimeError(f"weight sync failed: {next(e for e in errors if e)}")
    dist = _dist()
    rank, world = (dist.get_rank(), dist.get_world_size()) if dist is not None else (0, 1)
    local_rows = rows[rank::world]
    records, error = None, None
    try:
        records = score_local(local_rows)
        validated_decisions(local_rows, records)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    gathered = _all_gather({"error": error, "records": records})
    errors = [x["error"] for x in gathered if x["error"]]
    if errors:
        raise RuntimeError(f"benchmark generation failed: {errors[0]}")
    return merge_records(rows, *(x["records"] for x in gathered))


def callback_class():
    from transformers import TrainerCallback

    class FullBenchmarkMonitorCallback(TrainerCallback):
        def __init__(self, contract, score_local, sync, output_dir):
            self.contract, self.score_local, self.sync = contract, score_local, sync
            self.rows, self.parent, self.start = contract["rows"], contract["parent"], contract["start"]
            self.positive = contract["positive_rows"]
            self.negative = [r for r in self.rows if r["reference_decision"] == "reject"]
            self.parent_positive = subset_records(self.rows, self.parent, self.positive)
            self.output_dir = Path(output_dir)
            self.identity = contract["identity_sha256"]
            self.checks, self.last_step = 0, None
            self.aborted, self.reason, self.stop_step = False, None, None
            self.startup_gate_passed, self.last_full_valid = False, False
            self.last_safe_step = 425
            self.last_safe_checkpoint = contract["manifest"]["continuation_source"]["source_checkpoint"]
            self.last_record = None
            self.normal_train_end, self.training_completed = False, False

        def _state_path(self):
            return self.output_dir / "full_benchmark_monitor_state.json"

        def _history_path(self):
            return self.output_dir / "full_benchmark_monitor.jsonl"

        def _checkpoint(self, step):
            path = self.output_dir / f"checkpoint-{step}"
            required = self.contract["manifest"]["continuation_source"]["source_files_sha256"]
            if all((path / name).is_file() and (path / name).stat().st_size > 0 for name in required):
                saved = json.loads((path / "trainer_state.json").read_text())
                if saved["global_step"] == step:
                    return str(path)
            return None

        def _state(self):
            return {"schema": SCHEMA, "identity": self.contract["identity"],
                "identity_sha256": self.identity, "manifest_sha256": self.contract["manifest_sha256"],
                "startup_gate_passed": self.startup_gate_passed, "history_checks": self.checks,
                "last_step": self.last_step, "stop": self.aborted, "stop_step": self.stop_step,
                "reason": self.reason, "last_safe_step": self.last_safe_step,
                "last_safe_checkpoint": self.last_safe_checkpoint,
                "normal_train_end": self.normal_train_end, "training_completed": self.training_completed,
                "terminal_step": self.last_step if self.normal_train_end else None,
                "terminal_checkpoint": self._checkpoint(self.last_step) if self.normal_train_end else None,
                "terminal_full_valid": self.last_full_valid}

        def _save_state(self):
            atomic_json(self._state_path(), self._state())

        def _restore(self, step):
            path = self._state_path()
            if not path.exists():
                if step != 425 or self._history_path().exists():
                    raise ValueError("full benchmark monitor: only the frozen 425 may start without history")
                return
            saved = json.loads(path.read_text())
            if saved["identity_sha256"] != self.identity or saved["identity"] != self.contract["identity"]:
                raise ValueError("full benchmark monitor: resume identity changed")
            if saved["stop"] or saved["normal_train_end"]:
                raise ValueError("full benchmark monitor: stopped or completed pilot cannot resume")
            with self._history_path().open() as stream:
                history = [json.loads(line) for line in stream if line.strip()]
            if (not history or len(history) != saved["history_checks"] or
                    history[-1]["step"] != saved["last_step"] or saved["last_step"] != step or
                    any(x["identity_sha256"] != self.identity or x["check_index"] != i + 1
                        or not x["valid"] or x["stop"]
                        for i, x in enumerate(history))):
                raise ValueError("full benchmark monitor: resume history/checkpoint mismatch")
            for record in history:
                for scope in ("positive", "full"):
                    if record[scope] is not None:
                        verify_file(record[scope]["decisions_file"])
            if not saved["startup_gate_passed"]:
                raise ValueError("full benchmark monitor: startup GPU gate never passed")
            for name, key in [("checks", "history_checks"), ("last_step", "last_step"),
                              ("last_safe_step", "last_safe_step"),
                              ("last_safe_checkpoint", "last_safe_checkpoint"),
                              ("startup_gate_passed", "startup_gate_passed")]:
                setattr(self, name, saved[key])
            self.last_record = history[-1]
            if not Path(self.last_safe_checkpoint).is_dir():
                raise ValueError("full benchmark monitor: prior safe checkpoint is missing")

        def _persist_decisions(self, step, scope, rows, records, parent):
            import pandas as pd

            folder = self.output_dir / "full_benchmark_monitor" / f"step-{step:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{scope}_decisions_check{self.checks:04d}.parquet"
            temp = path.with_suffix(".parquet.tmp")
            pd.DataFrame(records).to_parquet(temp, index=False)
            temp.replace(path)
            return {"metrics": direct_metrics(rows, parent, records), "decisions_file": fingerprint(path)}

        def _latch(self, step, reason):
            if not self.aborted:
                self.aborted, self.stop_step, self.reason = True, step, reason

        def _run(self, state, control, event, full=False, initial=False):
            step = state.global_step
            records, positive_records, error = None, None, None
            try:
                if full:
                    records = collective_score(self.rows, self.score_local, self.sync)
                    positive_records = subset_records(self.rows, records, self.positive)
                else:
                    positive_records = collective_score(self.positive, self.score_local, self.sync)
                validation_error, pm = None, None
                try:
                    if initial:
                        before = validated_decisions(self.rows, self.start)
                        after = validated_decisions(self.rows, records)
                        mismatches = sum(a != b for a, b in zip(before, after))
                        if mismatches:
                            raise ValueError(f"served checkpoint 425 differs on {mismatches} of 31050 frozen decisions")
                    if event == "resume":
                        import pandas as pd

                        scope = "full" if self.last_record["full"] is not None else "positive"
                        expected_rows = self.rows if scope == "full" else self.positive
                        path = self.last_record[scope]["decisions_file"]["path"]
                        previous = pd.read_parquet(path).to_dict("records")
                        current = subset_records(self.rows, records, expected_rows)
                        if validated_decisions(expected_rows, previous) != validated_decisions(expected_rows, current):
                            raise ValueError("served resumed policy differs from its saved checkpoint decisions")
                    pm = direct_metrics(self.positive, self.parent_positive, positive_records)
                except Exception as exc:
                    validation_error = f"{type(exc).__name__}: {exc}"
                failures = _all_gather(validation_error)
                if any(failures):
                    raise RuntimeError(f"post-generation validation failed: {next(e for e in failures if e)}")
                trigger = stop_for_tpr(pm["delta_tpr"])
                if trigger:
                    self._latch(step, "direct_full_benchmark_tpr_drop_8pp")
                    # Persist the first stop before a supplementary full pass can fail.
                    self.last_step = step
                    latch_error = None
                    if state.is_world_process_zero:
                        try:
                            self._save_state()
                        except Exception as exc:
                            latch_error = f"cannot persist first stop: {exc}"
                    latch_error = _broadcast(latch_error)
                    if latch_error:
                        raise RuntimeError(latch_error)
                if trigger and records is None:
                    negative_records = collective_score(self.negative, self.score_local, self.sync)
                    records = merge_records(self.rows, positive_records, negative_records)
                if initial:
                    self.startup_gate_passed = True
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                self._latch(step, f"full_benchmark_monitor_failed: {error}")
            result = None
            if state.is_world_process_zero:
                self.checks += 1
                result = {"step": step, "event": event, "check_index": self.checks,
                    "identity_sha256": self.identity, "manifest_sha256": self.contract["manifest_sha256"],
                    "valid": error is None, "stop": self.aborted, "stop_step": self.stop_step,
                    "reason": self.reason, "error": error, "positive": None, "full": None,
                    "last_safe_step": self.last_safe_step, "last_safe_checkpoint": self.last_safe_checkpoint,
                    "startup_gate_passed": self.startup_gate_passed}
                try:
                    if positive_records is not None:
                        result["positive"] = self._persist_decisions(step, "positive", self.positive,
                                                                    positive_records, self.parent_positive)
                    if records is not None:
                        result["full"] = self._persist_decisions(step, "full", self.rows, records, self.parent)
                    folder = self.output_dir / "full_benchmark_monitor" / f"step-{step:04d}"
                    atomic_json(folder / f"metrics-check{self.checks:04d}.json", result)
                    atomic_json(folder / "metrics.json", result)
                    with self._history_path().open("a") as stream:
                        stream.write(json.dumps(result, allow_nan=False) + "\n")
                    self.last_record, self.last_step = result, step
                    self.last_full_valid = bool(result["valid"] and result["full"] is not None)
                    self._save_state()
                except Exception as exc:
                    self._latch(step, f"full_benchmark_record_failed: {exc}")
                    result.update(valid=False, stop=True, stop_step=self.stop_step,
                                  reason=self.reason, error=f"record failed: {exc}")
            result = _broadcast(result)
            self.checks, self.last_step, self.last_record = result["check_index"], step, result
            self.aborted, self.reason, self.stop_step = result["stop"], result["reason"], result["stop_step"]
            self.startup_gate_passed = result["startup_gate_passed"]
            self.last_full_valid = bool(result["valid"] and result["full"] is not None)
            if event == "periodic":
                # A restored TrainerState may still carry the historical save_steps=25.
                control.should_save = True
            if self.aborted:
                control.should_save = True
                control.should_training_stop = True
                if event in ("startup", "resume"):
                    raise RuntimeError(self.reason)
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            if args.save_steps != 5 or state.save_steps != 5:
                raise RuntimeError("full benchmark monitor: save cadence must be five after resume")
            payload = None
            if state.is_world_process_zero:
                try:
                    self._restore(state.global_step)
                    payload = {"error": None, "state": self._state(), "last_record": self.last_record}
                except Exception as exc:
                    payload = {"error": f"{type(exc).__name__}: {exc}"}
            payload = _broadcast(payload)
            if payload["error"]:
                raise RuntimeError(payload["error"])
            restored = payload["state"]
            self.checks, self.last_step = restored["history_checks"], restored["last_step"]
            self.last_safe_step, self.last_safe_checkpoint = restored["last_safe_step"], restored["last_safe_checkpoint"]
            self.startup_gate_passed = restored["startup_gate_passed"]
            self.last_record = payload["last_record"]
            dist = _dist()
            world = dist.get_world_size() if dist is not None else 1
            if world != self.contract["manifest"]["continuation_source"]["world_size"]:
                raise RuntimeError("full benchmark monitor: world size differs from frozen continuation")
            initial = not self.startup_gate_passed
            return self._run(state, control, "startup" if initial else "resume", full=True, initial=initial)

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step
            if not self.aborted and step > 425 and (step - 425) % 5 == 0:
                return self._run(state, control, "periodic", full=step in (450, 475, 500))
            return control

        def on_save(self, args, state, control, **kwargs):
            # All ranks reach this after writing their RNG state.
            _all_gather(None)
            payload = None
            if state.is_world_process_zero:
                try:
                    path = self._checkpoint(state.global_step)
                    if not path:
                        raise ValueError("finished checkpoint missing after save")
                    if (not self.aborted and self.last_record and self.last_record["valid"]
                            and self.last_step == state.global_step):
                        self.last_safe_step, self.last_safe_checkpoint = state.global_step, path
                    self._save_state()
                    payload = {"error": None, "step": self.last_safe_step, "path": self.last_safe_checkpoint}
                except Exception as exc:
                    payload = {"error": f"full benchmark safe-checkpoint record failed: {exc}"}
            payload = _broadcast(payload)
            if payload["error"]:
                self._latch(state.global_step, payload["error"])
                control.should_training_stop = True
                record_error = None
                if state.is_world_process_zero:
                    try:
                        self._save_state()
                    except Exception as exc:
                        record_error = str(exc)
                record_error = _broadcast(record_error)
                if record_error:
                    raise RuntimeError(f"full benchmark failure state could not be saved: {record_error}")
            else:
                self.last_safe_step, self.last_safe_checkpoint = payload["step"], payload["path"]
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if not self.aborted and (self.last_step != state.global_step or not self.last_full_valid):
                control = self._run(state, control, "final", full=True)
            if not self.aborted and state.global_step != 500:
                self._latch(state.global_step, "unexpected_early_train_end")
            self.training_completed = state.global_step == 500 and not self.aborted
            complete_checkpoint = self._checkpoint(state.global_step) is not None
            self.normal_train_end = self.last_full_valid and complete_checkpoint and (
                self.training_completed or self.reason == "direct_full_benchmark_tpr_drop_8pp")
            error = None
            if state.is_world_process_zero:
                try:
                    self._save_state()
                except Exception as exc:
                    error = f"full benchmark terminal state failed: {exc}"
            error = _broadcast(error)
            if error:
                raise RuntimeError(error)
            return control

    return FullBenchmarkMonitorCallback
