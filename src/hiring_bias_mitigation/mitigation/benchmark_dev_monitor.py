"""Direct greedy TPR on a frozen, candidate-disjoint benchmark development split.

This is an opt-in training safety monitor, not an unbiased final evaluation. Labels are the
original benchmark labels. Neither sampled margins nor pool-to-benchmark weights enter it.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..eval.parsing import parse_output

SCHEMA = "benchmark-dev-monitor-v1"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"benchmark_dev_monitor: invalid {name}")
    return value


def _read_records(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate_rows(rows):
    if not rows:
        raise ValueError("benchmark_dev_monitor: empty development split")
    for row in rows:
        for key in ("prompt_id", "prompt", "candidate_id", "job_id", "lang"):
            _nonempty(row.get(key), key)
        if row.get("reference_decision") not in ("hire", "reject"):
            raise ValueError("benchmark_dev_monitor: invalid reference_decision")
        if row["lang"] != "en":
            raise ValueError("benchmark_dev_monitor: only the frozen English scorer is supported")
    ids = [row["prompt_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("benchmark_dev_monitor: duplicate prompt_id")
    if "hire" not in {row["reference_decision"] for row in rows}:
        raise ValueError("benchmark_dev_monitor: reference hire rows are required")


def validated_decisions(rows, records):
    if len(records) != len(rows):
        raise ValueError("benchmark_dev_monitor: missing or extra decisions")
    if [r.get("prompt_id") for r in records] != [r["prompt_id"] for r in rows]:
        raise ValueError("benchmark_dev_monitor: decision order or prompt identity mismatch")
    decisions = [r.get("decision") for r in records]
    if any(d not in ("hire", "reject") for d in decisions):
        raise ValueError("benchmark_dev_monitor: invalid or missing binary decision")
    return decisions


def direct_metrics(rows, parent_records, current_records):
    """Original-label conditional rates, with paired losses and gains on identical rows."""
    validate_rows(rows)
    parent = validated_decisions(rows, parent_records)
    current = validated_decisions(rows, current_records)
    result = {"n": len(rows), "n_candidates": len({r["candidate_id"] for r in rows}),
              "n_jobs": len({r["job_id"] for r in rows}), "invalid_rate": 0.0,
              "scope": "reference_hire_only" if all(r["reference_decision"] == "hire"
                                                         for r in rows) else "both_labels",
              "hire_rate": sum(d == "hire" for d in current) / len(rows),
              "agreement": sum(r["reference_decision"] == d
                               for r, d in zip(rows, current)) / len(rows)}
    for label, metric in (("hire", "tpr"), ("reject", "fpr")):
        ix = [i for i, r in enumerate(rows) if r["reference_decision"] == label]
        n = len(ix)
        if not n:
            result.update({f"n_ref_{label}": 0, f"n_ref_{label}_candidates": 0,
                           **{key: None for key in (f"parent_{metric}", metric,
                               f"delta_{metric}", f"{metric}_loss", f"{metric}_gain",
                               f"{metric}_loss_count", f"{metric}_gain_count")}})
            continue
        before = sum(parent[i] == "hire" for i in ix)
        after = sum(current[i] == "hire" for i in ix)
        loss = sum(parent[i] == "hire" and current[i] == "reject" for i in ix)
        gain = sum(parent[i] == "reject" and current[i] == "hire" for i in ix)
        result.update({f"n_ref_{label}": n, f"parent_{metric}": before / n,
                       metric: after / n, f"delta_{metric}": (after - before) / n,
                       f"{metric}_loss": loss / n, f"{metric}_gain": gain / n,
                       f"{metric}_loss_count": loss, f"{metric}_gain_count": gain,
                       f"n_ref_{label}_candidates": len({rows[i]["candidate_id"] for i in ix})})
    return result


def stop_for_tpr(delta_tpr, threshold):
    if not math.isfinite(delta_tpr) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("benchmark_dev_monitor: invalid delta or stop threshold")
    # Inclusive: 8 percentage points lost must stop at an 8 pp threshold.
    return delta_tpr <= -threshold or math.isclose(delta_tpr, -threshold, abs_tol=1e-12)


def choose_latest_safe_record(history, checkpoint_exists):
    """Last passing, saved step before the first stop; step zero means the frozen parent.

    Later recovery cannot undo an earlier stop. Existence is supplied by the caller so an
    unsaved passing reading is never offered as a checkpoint to export.
    """
    latest = None
    for record in history:
        if record.get("stop") or not record.get("valid"):
            break
        if stop_for_tpr(record["delta_tpr"], record["tpr_drop_stop"]):
            break
        if record["step"] == 0 or checkpoint_exists(record["step"]):
            latest = record
    return latest


def load_contract(settings, training_sft_checkpoint, resolve_path):
    """Verify every frozen input before model loading; return records and complete identity."""
    manifest_path = Path(resolve_path(settings["manifest_file"]))
    digest = sha256_file(manifest_path)
    if digest != settings["manifest_sha256"]:
        raise ValueError("benchmark_dev_monitor: manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("benchmark_dev_monitor: unsupported manifest schema")
    loaded = {}
    for stem in ("rows", "parent_decisions"):
        path = Path(manifest[f"{stem}_file"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if sha256_file(path) != manifest[f"{stem}_sha256"]:
            raise ValueError(f"benchmark_dev_monitor: {stem} SHA256 mismatch")
        loaded[stem] = _read_records(path)
    rows, parent = loaded["rows"], loaded["parent_decisions"]
    validate_rows(rows)
    validated_decisions(rows, parent)
    model = manifest["parent_model"]
    _nonempty(model["run"], "parent run")
    _nonempty(model["identity_sha256"], "parent model identity")
    if model["sft_checkpoint"] != training_sft_checkpoint:
        raise ValueError("benchmark_dev_monitor: training SFT parent differs from frozen parent")
    split = manifest["split"]
    dev_ids = set(split["candidate_ids"])
    remaining_ids = set(split["remaining_candidate_ids"])
    if not {r["candidate_id"] for r in rows} <= dev_ids:
        raise ValueError("benchmark_dev_monitor: rows contain candidates outside development split")
    if not remaining_ids or dev_ids & remaining_ids:
        raise ValueError("benchmark_dev_monitor: candidate overlap or empty remaining split")
    if split.get("benchmark_previously_exposed") is not True:
        raise ValueError("benchmark_dev_monitor: prior benchmark exposure must be acknowledged")
    scorer = manifest["scorer"]
    if (scorer.get("temperature") != 0.0 or scorer.get("top_p") != 1.0
            or scorer.get("max_new_tokens") != 512
            or scorer.get("chat_template_kwargs") != {"enable_thinking": False}):
        raise ValueError("benchmark_dev_monitor: scorer differs from original greedy benchmark")
    every = settings.get("steps", 25)
    if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
        raise ValueError("benchmark_dev_monitor: steps must be a positive integer")
    threshold = float(settings["tpr_drop_stop"])
    stop_for_tpr(0.0, threshold)
    identity = {"schema": SCHEMA, "manifest_sha256": digest,
                "rows_sha256": manifest["rows_sha256"],
                "parent_decisions_sha256": manifest["parent_decisions_sha256"],
                "ordered_rows_sha256": identity_hash(rows), "parent_model": model,
                "scorer": scorer, "steps": every, "tpr_drop_stop": threshold,
                "runtime_code_sha256": {
                    "benchmark_dev_monitor.py": sha256_file(__file__),
                    "grpo.py": sha256_file(Path(__file__).with_name("grpo.py")),
                    "eval/parsing.py": sha256_file(
                        Path(__file__).parents[1] / "eval" / "parsing.py"),
                    "eval/backends.py": sha256_file(
                        Path(__file__).parents[1] / "eval" / "backends.py")}}
    return {"rows": rows, "parent": parent, "manifest": manifest, "identity": identity,
            "identity_sha256": identity_hash(identity)}


def require_training_ready(contract):
    """A prepared, unvalidated manifest may be inspected but cannot start optimization."""
    policy = contract["manifest"].get("policy", {})
    if policy.get("training_authorized") is not True:
        raise ValueError("benchmark_dev_monitor: frozen policy does not authorize training")
    if (policy.get("check_every_optimizer_steps") != contract["identity"]["steps"]
            or policy.get("stop_delta_tpr_pp") != -100 * contract["identity"]["tpr_drop_stop"]):
        raise ValueError("benchmark_dev_monitor: configured alarm differs from frozen policy")


def validate_train_isolation(rows, train_candidate_ids, dev_candidate_ids=None):
    dev_ids = {r["candidate_id"] for r in rows} | set(dev_candidate_ids or ())
    if dev_ids & set(train_candidate_ids):
        raise ValueError("benchmark_dev_monitor: development candidate is in training")


def served_decisions(vllm_generation, tokenizer, rows, scorer):
    """Same complete greedy generation and parser as the benchmark; caller syncs all ranks."""
    from vllm import SamplingParams

    from ..eval.backends import _chat_wrap

    texts = [_chat_wrap(tokenizer, r["prompt"], scorer["chat_template_kwargs"]) for r in rows]
    params = SamplingParams(max_tokens=scorer["max_new_tokens"], temperature=0.0, top_p=1.0, n=1)
    outputs = vllm_generation.llm.generate(texts, params, use_tqdm=False)
    if len(outputs) != len(rows):
        raise ValueError("benchmark_dev_monitor: vLLM returned missing or extra outputs")
    records = []
    for row, text, out in zip(rows, texts, outputs):
        if out.prompt != text or len(out.outputs) != 1:
            raise ValueError("benchmark_dev_monitor: vLLM output order or sample count differs")
        raw = out.outputs[0].text
        records.append({"prompt_id": row["prompt_id"],
                        "decision": parse_output(raw, row["lang"]).decision, "raw_output": raw})
    validated_decisions(rows, records)
    return records


def _atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    tmp.replace(path)


def _sync_error(sync):
    """Every rank enters weight sync; exchange Python failures before rank-zero generation."""
    import torch.distributed as dist

    error = None
    try:
        sync()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    if dist.is_available() and dist.is_initialized():
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        error = next((e for e in errors if e), None)
    return error


def callback_class():
    from transformers import TrainerCallback

    from ..utils.logging import get_logger
    from .grpo import _broadcast

    logger = get_logger(__name__)

    class BenchmarkDevMonitorCallback(TrainerCallback):
        def __init__(self, contract, score, sync, output_dir):
            self.contract, self.score, self.sync = contract, score, sync
            self.rows, self.parent = contract["rows"], contract["parent"]
            self.identity = contract["identity_sha256"]
            self.every = contract["identity"]["steps"]
            self.threshold = contract["identity"]["tpr_drop_stop"]
            self.output_dir = Path(output_dir)
            self.checks, self.last_step = 0, None
            self.aborted, self.reason = False, None
            self.last_safe_step = 0
            self.last_safe_checkpoint = contract["manifest"]["parent_model"]["sft_checkpoint"]

        def _state_path(self):
            return self.output_dir / "benchmark_dev_monitor_state.json"

        def _restore(self, state):
            path = self._state_path()
            if path.is_file():
                saved = json.loads(path.read_text())
                if (saved["identity_sha256"] != self.identity
                        or saved["identity"] != self.contract["identity"]):
                    raise ValueError("benchmark_dev_monitor: resume identity mismatch")
                history_path = self.output_dir / "benchmark_dev_monitor.jsonl"
                history = [json.loads(line) for line in history_path.read_text().splitlines()]
                if (not history or saved["checks"] != len(history)
                        or history[-1]["step"] != saved["last_step"]
                        or any(r["identity_sha256"] != self.identity or r["checks"] != i + 1
                               for i, r in enumerate(history))):
                    raise ValueError(
                        "benchmark_dev_monitor: incomplete or mismatched check history")
                if saved["last_step"] > state.global_step:
                    raise ValueError("benchmark_dev_monitor: history is ahead of checkpoint")
                if saved["stop"]:
                    raise ValueError("benchmark_dev_monitor: previously stopped run cannot resume")
                self.checks, self.last_step = saved["checks"], saved["last_step"]
                self.last_safe_step = saved["last_safe_step"]
                self.last_safe_checkpoint = saved["last_safe_checkpoint"]
                if self.last_safe_step and not self._checkpoint_exists(self.last_safe_step):
                    raise ValueError("benchmark_dev_monitor: last safe checkpoint is missing")
            elif state.global_step or (self.output_dir / "benchmark_dev_monitor.jsonl").exists():
                raise ValueError("benchmark_dev_monitor: cannot resume without monitor state")

        def _checkpoint_exists(self, step):
            path = self.output_dir / f"checkpoint-{step}"
            return (path / "trainer_state.json").is_file() and any(
                (path / name).is_file() for name in ("adapter_model.safetensors",
                                                    "adapter_model.bin", "model.safetensors"))

        def _save_state(self, step, stop):
            _atomic_json(self._state_path(), {"identity_sha256": self.identity,
                         "identity": self.contract["identity"], "checks": self.checks,
                         "last_step": step, "stop": stop,
                         "last_safe_step": self.last_safe_step,
                         "last_safe_checkpoint": self.last_safe_checkpoint})

        def _record(self, result, records):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            if records is not None:
                name = f"benchmark_dev_monitor_step{result['step']:06d}_check{self.checks:04d}.json"
                _atomic_json(self.output_dir / name,
                             {"identity_sha256": self.identity, "rows": records})
                result["decisions_file"] = name
            result.update(last_safe_step=self.last_safe_step,
                          last_safe_checkpoint=self.last_safe_checkpoint,
                          safe_step_candidate=(result["step"] if result["valid"]
                                               and not result["stop"] else None))
            with (self.output_dir / "benchmark_dev_monitor.jsonl").open("a") as fh:
                fh.write(json.dumps(result, allow_nan=False) + "\n")
            self._save_state(result["step"], result["stop"])

        def _run(self, state, control, event, initial=False):
            # FSDP/ZeRO sync can contain collectives. All ranks sync before rank zero scores.
            error = _sync_error(self.sync)
            result = None
            if state.is_world_process_zero:
                records = None
                try:
                    if error:
                        raise RuntimeError(error)
                    records = self.score(self.rows)
                    metrics = direct_metrics(self.rows, self.parent, records)
                    if initial and validated_decisions(self.rows, records) != validated_decisions(
                            self.rows, self.parent):
                        raise ValueError("served starting policy differs from frozen SFT decisions")
                    stop = stop_for_tpr(metrics["delta_tpr"], self.threshold)
                    reason = (f"direct benchmark-dev TPR fell at least {100*self.threshold:g} pp"
                              if stop else None)
                    result = {**metrics, "stop": stop, "reason": reason, "valid": True}
                except Exception as exc:
                    reason = f"benchmark-dev monitor failed: {type(exc).__name__}: {exc}"
                    result = {"stop": True, "valid": False, "reason": reason}
                self.checks += 1
                result.update(step=state.global_step, event=event, checks=self.checks,
                              identity_sha256=self.identity, tpr_drop_stop=self.threshold)
                try:
                    self._record(result, records)
                except Exception as exc:
                    result.update(stop=True, valid=False, reason=f"monitor record failed: {exc}")
            result = _broadcast(result)
            self.checks, self.last_step = result["checks"], state.global_step
            if result["stop"]:
                self.aborted, self.reason = True, result["reason"]
                control.should_save = True
                control.should_training_stop = True
                logger.error("%s at step %d", self.reason, state.global_step)
            elif state.is_world_process_zero:
                logger.info("benchmark dev at step %d: TPR %.4f (%+.2f pp), FPR %s; check %d",
                            state.global_step, result["tpr"], 100 * result["delta_tpr"],
                            result["fpr"], self.checks)
            if event in ("start", "resume") and result["stop"]:
                raise RuntimeError(result["reason"])
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            error = None
            if state.is_world_process_zero:
                try:
                    self._restore(state)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            error = _broadcast(error)
            if error:
                raise RuntimeError(error)
            return self._run(state, control, "start" if not state.global_step else "resume",
                             initial=not state.global_step)

        def on_step_end(self, args, state, control, **kwargs):
            if not self.aborted and state.global_step % self.every == 0:
                return self._run(state, control, "periodic")
            return control

        def on_save(self, args, state, control, **kwargs):
            result = None
            if state.is_world_process_zero:
                try:
                    path = self.output_dir / "benchmark_dev_monitor.jsonl"
                    with path.open() as fh:
                        history = [json.loads(line) for line in fh]
                    safe = choose_latest_safe_record(history, self._checkpoint_exists)
                    if safe is not None:
                        self.last_safe_step = safe["step"]
                        self.last_safe_checkpoint = (
                            self.contract["manifest"]["parent_model"]["sft_checkpoint"]
                            if safe["step"] == 0
                            else str(self.output_dir / f"checkpoint-{safe['step']}"))
                    self._save_state(self.last_step, self.aborted)
                    result = {"step": self.last_safe_step, "path": self.last_safe_checkpoint}
                except Exception as exc:
                    result = {"error": f"safe-checkpoint tracking failed: {exc}"}
            result = _broadcast(result)
            if "error" in result:
                self.aborted, self.reason = True, result["error"]
                control.should_training_stop = True
                logger.error(self.reason)
            else:
                self.last_safe_step, self.last_safe_checkpoint = result["step"], result["path"]
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if not self.aborted and self.last_step != state.global_step:
                return self._run(state, control, "final")
            return control

    return BenchmarkDevMonitorCallback
