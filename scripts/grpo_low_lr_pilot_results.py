"""Evaluate only the declared terminal state of the separately authorized low-LR pilot.

This post-training stage never trains, selects a best checkpoint, or submits a job.
The final merged-model pass must reproduce every terminal in-training binary decision
before SDT or trajectory interpretation is emitted. Source controls remain immutable.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
START, END = 425, 500
GRID = (425, 450, 475, 500)
SCOPES = ("full", "dev", "historical_remainder")


def script_module(name):
    key = f"low_lr_results_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


T = script_module("checkpoint_benchmark_trajectory")


def terminal_step(state):
    """Select the fixed terminal event; a later recovery cannot erase an earlier stop."""
    if state.get("normal_train_end") is not True or state.get("startup_gate_passed") is not True:
        raise ValueError("Only a normally ended pilot with a passed startup gate can finalize")
    if not isinstance(state.get("stop"), bool):
        raise ValueError("Missing explicit monitor stop latch")
    step = state.get("stop_step") if state["stop"] else END
    if isinstance(step, bool) or not isinstance(step, int) or not START <= step <= END:
        raise ValueError("Terminal step is outside the authorized pilot")
    if step % 5 or state.get("last_step") != step:
        raise ValueError("Terminal event and final monitor observation differ")
    if not state["stop"] and state.get("stop_step") is not None:
        raise ValueError("A previous stop cannot be cleared by later recovery")
    if Path(state.get("terminal_checkpoint", "")).name != f"checkpoint-{step}":
        raise ValueError("Terminal checkpoint path disagrees with the terminal event")
    safe = state.get("last_safe_step")
    if safe is not None and (not isinstance(safe, int) or safe > step or safe < START):
        raise ValueError("Invalid last-safe checkpoint metadata")
    if state["stop"] and safe is not None and safe >= step:
        raise ValueError("The triggering checkpoint is not the previous safe checkpoint")
    return step


def align_raw(raw, expected):
    """Align immutable row keys and verify all source metadata, not only reference labels."""
    aligned = T.aligned_raw(raw, expected)
    identity = [column for column in expected.columns if column not in ("prompt", "prompt_id")]
    if any(column not in aligned.columns for column in identity):
        raise ValueError("Raw decisions lack frozen row identity columns")
    for column in identity:
        left, right = expected[column].reset_index(drop=True), aligned[column]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError(f"Frozen row identity changed: {column}")
    if "prompt_id" in aligned and aligned.prompt_id.tolist() != expected.prompt_id.tolist():
        raise ValueError("Raw prompt_id does not match the frozen row keys")
    if (
        not aligned.decision.isin(["hire", "reject"]).all()
        or not aligned.outcome.eq("decided").all()
    ):
        raise ValueError("Unusable decisions prevent paired interpretation")
    return aligned.assign(prompt_id=expected.prompt_id.tolist())


def callback_raw(decisions, expected):
    """Reparse saved in-training text with the official parser before comparing decisions."""
    from hiring_bias_mitigation.eval.parsing import parse_frame

    required = {"prompt_id", "decision", "raw_output"}
    if not required <= set(decisions.columns) or decisions.prompt_id.duplicated().any():
        raise ValueError("Malformed or duplicated callback prompt identity")
    if len(decisions) != len(expected) or set(decisions.prompt_id) != set(expected.prompt_id):
        raise ValueError("Callback did not score the exact frozen full benchmark")
    values = decisions.set_index("prompt_id").loc[expected.prompt_id].reset_index()
    parsed = parse_frame(
        expected.drop(columns="prompt").assign(raw_output=values.raw_output.tolist())
    )
    if parsed.decision.tolist() != values.decision.tolist():
        raise ValueError("Callback stored decisions disagree with the official parser")
    return align_raw(parsed, expected)


def exact_decision_check(current, callback, expected):
    current, callback = align_raw(current, expected), align_raw(callback, expected)
    mismatches = current.decision.ne(callback.decision)
    sample = current.loc[mismatches, ["prompt_id", *T.KEY, "decision"]].head(20).copy()
    sample = sample.rename(columns={"decision": "merged_decision"})
    sample["callback_decision"] = callback.loc[mismatches, "decision"].head(20).tolist()
    count = int(mismatches.sum())
    return {
        "exact_match": count == 0,
        "rows": len(current),
        "mismatch_count": count,
        "mismatch_pct": 100 * count / len(current),
        "mismatch_sample": sample.to_dict("records"),
        "comparison": "exact binary decisions on identical frozen prompt IDs",
    }


def count_guardrails(hits, parent_hits, positive_n, agreement, parent_agreement, total_n):
    """Inclusive original limits using integer counts, avoiding rounded-rate boundaries."""
    values = (hits, parent_hits, positive_n, agreement, parent_agreement, total_n)
    if (
        any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
        or positive_n <= 0
        or total_n <= 0
        or max(hits, parent_hits) > positive_n
        or max(agreement, parent_agreement) > total_n
    ):
        raise ValueError("Guardrail counts must be valid nonnegative integer counts")
    return {
        "passes_original_tpr_10pp_guardrail": 100 * (hits - parent_hits) >= -10 * positive_n,
        "passes_original_agreement_5pp_guardrail": 100 * (agreement - parent_agreement)
        >= -5 * total_n,
    }


def scope_metrics(raw, parent, partitions):
    from hiring_bias_mitigation.analysis.reference_metrics import (
        flip_rates,
        prepare,
        unstable_within_group,
    )

    sdt_counts = script_module("analyze_checkpoint_average_sdt").sdt_counts
    output = T.partition_metrics(raw, parent, partitions)
    scopes = {"full": set(raw.candidate_id), **partitions}
    for scope, candidates in scopes.items():
        frame = raw[raw.candidate_id.isin(candidates)]
        prepared = prepare(frame)
        hire, yes = frame.reference_decision.eq("hire"), frame.decision.eq("hire")
        sdt = sdt_counts(
            int((hire & yes).sum()),
            int((hire & ~yes).sum()),
            int((~hire & yes).sum()),
            int((~hire & ~yes).sum()),
        )
        instability, complete, expected = unstable_within_group(prepared)
        parent_frame = parent[parent.candidate_id.isin(candidates)]
        parent_metric = T.metrics(parent_frame)
        output[scope].update(sdt)
        output[scope].update(
            {
                "raw_unstable_within_group_pct": instability,
                "complete_within_group_sets": complete,
                "expected_within_group_sets": expected,
                "attribute_flip_rates": flip_rates(prepared),
                "delta_agreement_vs_sft_pp": output[scope]["agreement_pct"]
                - parent_metric["agreement_pct"],
            }
        )
        if scope == "full":
            output[scope].update(
                count_guardrails(
                    int((hire & yes).sum()),
                    int(
                        (
                            parent_frame.reference_decision.eq("hire")
                            & parent_frame.decision.eq("hire")
                        ).sum()
                    ),
                    int(hire.sum()),
                    int(frame.decision.eq(frame.reference_decision).sum()),
                    int(parent_frame.decision.eq(parent_frame.reference_decision).sum()),
                    len(frame),
                )
            )
        else:
            output[scope].update(
                {
                    "passes_original_tpr_10pp_guardrail": None,
                    "passes_original_agreement_5pp_guardrail": None,
                }
            )
    return output


def curve_summary(pilot, controls, stopped):
    """Fixed four-point comparison only when complete; never fill censored pilot points."""
    complete = not stopped and set(GRID) <= set(pilot) and set(GRID) <= set(controls)
    output = {
        "fixed_grid": list(GRID),
        "complete_fixed_grid": complete,
        "pilot_observed_steps": sorted(pilot),
        "stop_latch_preserved": bool(stopped),
        "comparison_is_historical_not_restarted_control": True,
        "learning_rate_causal_effect_established": False,
        "fixed_point_noise_mechanism_established": False,
        "scopes": None,
    }
    if not complete:
        output["reason"] = (
            "Early stop or missing fixed-grid observation; no complete-grid range comparison"
        )
        return output
    output["scopes"] = {}
    for scope in SCOPES:
        output["scopes"][scope] = {}
        for key in ("tpr_pct", "criterion_c", "dprime"):

            def summarize(points, scope=scope, key=key):
                values = [float(points[step][scope][key]) for step in GRID]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("Nonfinite trajectory metric")
                return {
                    "values": values,
                    "range": max(values) - min(values),
                    "total_absolute_adjacent_change": sum(abs(b - a) for a, b in pairwise(values)),
                    "endpoint_minus_start": values[-1] - values[0],
                }

            new, old = summarize(pilot), summarize(controls)
            output["scopes"][scope][key] = {
                "low_lr": new,
                "historical_original_lr": old,
                "range_difference": new["range"] - old["range"],
                "total_absolute_change_difference": new["total_absolute_adjacent_change"]
                - old["total_absolute_adjacent_change"],
            }
    return output


def validate_retention(records):
    value = records[-1].get("kept_fraction") if records else None
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or value < 0.8
    ):
        raise ValueError("Terminal adapter merge retention must be finite and at least 0.8")


def read_chunk(path, binding, model_binding, start, count):
    data = json.loads(path.read_text())
    if (
        data.get("binding_sha256") != binding
        or data.get("model_binding") != model_binding
        or data.get("start") != start
        or len(data.get("outputs", [])) != count
        or any(not isinstance(value, str) for value in data["outputs"])
    ):
        raise ValueError("Cached generation chunk has different model, protocol, or row identity")
    return data["outputs"]


def write_csv(result, destination):
    rows = []
    for arm in ("pilot", "historical_original_lr"):
        for entry in result["trajectory"][arm]:
            for scope, metric in entry["scopes"].items():
                row = {"arm": arm, "step": entry["step"], "scope": scope}
                row.update(
                    {
                        key: value
                        for key, value in metric.items()
                        if not isinstance(value, (dict, list))
                    }
                )
                rows.append(row)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_history(run, state, contract):
    """Require all prescribed monitor checks and preserve the first latched stop."""
    path = run / "full_benchmark_monitor.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    terminal = terminal_step(state)
    steps = [record["step"] for record in records]
    if steps != sorted(steps) or sorted(set(steps)) != list(range(START, terminal + 1, 5)):
        raise ValueError("Missing, reordered, or post-stop monitor check")
    if state.get("history_checks") != len(records):
        raise ValueError("Monitor state and history count differ")
    stops = [record["step"] for record in records if record.get("stop")]
    if set(stops) != ({terminal} if state["stop"] else set()):
        raise ValueError("Monitor history disagrees with the irreversible first stop")
    by_step = {record["step"]: record for record in records}
    for record in records:
        if (
            record.get("identity_sha256") != contract["identity_sha256"]
            or record.get("manifest_sha256") != contract["manifest_sha256"]
            or record.get("valid") is not True
        ):
            raise ValueError("Monitor identity or valid-decision gate failed")
        metrics_path = (
            run / "full_benchmark_monitor" / f"step-{record['step']:04d}" / "metrics.json"
        )
        if json.loads(metrics_path.read_text()) != by_step[record["step"]]:
            raise ValueError("Monitor JSONL and per-step metric receipt differ")
        T.check_fingerprint(record["positive"]["decisions_file"])
        if record["full"] is not None:
            T.check_fingerprint(record["full"]["decisions_file"])
    required = {step for step in GRID if step <= terminal} | {terminal}
    if any(by_step[step]["full"] is None for step in required):
        raise ValueError("Required full-grid or terminal callback decisions are missing")
    return records


def validate_outer(out, config_path, settings):
    path = out / "run_manifest.json"
    outer = json.loads(path.read_text())
    if (
        outer.get("schema") != "grpo-low-lr-pilot-run-v1"
        or outer.get("training_authorized") is not True
        or outer.get("bounds")
        != {
            "start_step": START,
            "max_step": END,
            "learning_rate": 3e-6,
            "max_new_optimizer_steps": 75,
        }
    ):
        raise ValueError("Outer pilot authorization or training bounds differ")
    T.check_fingerprint(outer["config"])
    if Path(outer["config"]["path"]).resolve() != config_path.resolve():
        raise ValueError("Outer freeze refers to another training config")
    T.check_fingerprint(outer["monitor_manifest"])
    if outer["monitor_manifest"]["sha256"] != settings["manifest_sha256"]:
        raise ValueError("Outer freeze and config refer to different monitor manifests")
    for item in outer["code"]:
        T.check_fingerprint(item)
    if not outer.get("prepared_commit"):
        raise ValueError("Missing committed pilot source identity")
    if not any(Path(item["path"]).resolve() == Path(__file__).resolve() for item in outer["code"]):
        raise ValueError("Finalizer source is not frozen in the outer experiment")
    return outer, T.fingerprint(path)


def load_inputs(config_path, require_terminal=True):
    import pandas as pd
    from hiring_bias_mitigation.mitigation.full_benchmark_monitor import load_contract
    from hiring_bias_mitigation.utils.config import load_config, resolve_output_path

    cfg = load_config(config_path)
    settings = cfg["pilot_results"]
    if settings.get("chunk_size") != 512 or cfg.get("learning_rate") != 3e-6:
        raise ValueError("Pilot finalization requires the frozen LR and chunk size")
    contract = load_contract(
        cfg["full_benchmark_monitor"], cfg["sft_checkpoint"], resolve_output_path
    )
    manifest = contract["manifest"]
    source = manifest["source_trajectory"]
    if T.software() != manifest["software"]:
        raise ValueError("Serving software changed since pilot freeze")
    run = Path(resolve_output_path(cfg["output_dir"])).resolve()
    out = Path(resolve_output_path(settings["experiment_dir"])).resolve()
    outer, outer_file = validate_outer(out, config_path, cfg["full_benchmark_monitor"])
    original = Path(source["directory"]).resolve()
    for protected in (run, original):
        if out == protected or out in protected.parents or protected in out.parents:
            raise ValueError(
                "Final evaluation directory must be isolated from training/source runs"
            )
    T.check_fingerprint(source["manifest"])
    source_manifest = T.load_manifest(original, verify=True)
    if source_manifest["binding_sha256"] != source["binding_sha256"]:
        raise ValueError("Frozen source trajectory binding differs")
    expected = pd.DataFrame(contract["rows"]).copy()
    if len(expected) != 31050 or expected.prompt_id.duplicated().any():
        raise ValueError("Finalizer requires all 31,050 unique frozen prompts")
    T.check_fingerprint(source["prompts"])
    frozen_prompts = pd.read_parquet(source["prompts"]["path"])
    if expected.prompt.tolist() != frozen_prompts.prompt.tolist():
        raise ValueError("Monitor bank prompt text differs from the original benchmark")
    for column in frozen_prompts.columns:
        left, right = expected[column], frozen_prompts[column]
        if not (left.eq(right) | (left.isna() & right.isna())).all():
            raise ValueError(f"Monitor bank source metadata changed: {column}")
    audit = source["audit_config"]
    generation = audit["generation"]
    if (
        audit.get("backend") != "vllm"
        or audit.get("tensor_parallel_size", 1) != 1
        or audit.get("chat_template_kwargs") != {"enable_thinking": False}
        or generation["max_new_tokens"] != 512
        or generation["temperature"] != 0.0
        or generation["top_p"] != 1.0
        or generation.get("n_samples", 1) != 1
    ):
        raise ValueError("Original TP1 greedy512 thinking-off benchmark scoring is required")
    cached = source["cached_models"]
    if set(cached) != {"SFT-v3", *(f"step{step}" for step in GRID)}:
        raise ValueError("Exactly SFT-v3 and the four fixed original-LR checkpoints are required")
    raw_by_model = {}
    for label, spec in cached.items():
        T.check_fingerprint(spec["raw"])
        raw_by_model[label] = align_raw(pd.read_parquet(spec["raw"]["path"]), expected)
    partitions = T.load_partitions({"benchmark_partition": source["benchmark_partition"]})
    dev, remainder = partitions["dev"], partitions["historical_remainder"]
    if dev & remainder or dev | remainder != set(expected.candidate_id):
        raise ValueError("Candidate split does not partition the unchanged benchmark")
    loaded = {
        "config": cfg,
        "contract": contract,
        "source_manifest": source_manifest,
        "out": out,
        "run": run,
        "expected": expected,
        "cached_raw": raw_by_model,
        "partitions": partitions,
        "outer": outer,
        "outer_file": outer_file,
    }
    if not require_terminal:
        return loaded
    state_path = run / "full_benchmark_monitor_state.json"
    state = json.loads(state_path.read_text())
    if (
        state.get("identity_sha256") != contract["identity_sha256"]
        or state.get("manifest_sha256") != contract["manifest_sha256"]
    ):
        raise ValueError("Final monitor state belongs to another experiment")
    step = terminal_step(state)
    terminal = Path(state["terminal_checkpoint"]).resolve()
    if terminal != run / f"checkpoint-{step}":
        raise ValueError("Terminal checkpoint must belong to the new training run")
    trainer_state = json.loads((terminal / "trainer_state.json").read_text())
    if trainer_state["global_step"] != step:
        raise ValueError("Saved trainer state and selected terminal step differ")
    records = load_history(run, state, contract)
    terminal_config = json.loads((terminal / "adapter_config.json").read_text())
    if terminal_config.get("r") != 32 or terminal_config.get("lora_alpha") != 64:
        raise ValueError(
            "Pilot terminal adapter must retain the original ordinary r32/alpha64 LoRA"
        )
    files = [
        T.fingerprint(terminal / name)
        for name in ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")
    ]
    files += [
        outer_file,
        T.fingerprint(run / "run_config.json"),
        T.fingerprint(state_path),
        T.fingerprint(run / "full_benchmark_monitor.jsonl"),
        T.fingerprint(config_path),
    ]
    binding = T.digest(
        {"monitor_identity_sha256": contract["identity_sha256"], "terminal_files": files}
    )
    return {
        **loaded,
        "terminal": terminal,
        "step": step,
        "state": state,
        "records": records,
        "files": files,
        "binding_sha256": binding,
    }


def prepare_merged(inputs):
    from hiring_bias_mitigation.mitigation import merge as merger

    out, terminal = inputs["out"], inputs["terminal"]
    merged = out / "merged" / f"terminal-{inputs['step']:04d}"
    path = out / "terminal_merge_receipt.json"
    parent = next(row for row in inputs["source_manifest"]["steps"] if row["step"] == 0)
    expected_chain = [Path(item["path"]).resolve() for item in parent["chain"]] + [terminal]
    if [item.resolve() for item in merger.adapter_chain(terminal)] != expected_chain:
        raise ValueError(
            "Pilot terminal does not have the frozen SFT parent followed by its r32 adapter"
        )
    if path.exists():
        receipt = json.loads(path.read_text())
        if receipt["binding_sha256"] != inputs["binding_sha256"]:
            raise ValueError("Existing terminal merge belongs to another checkpoint/protocol")
        for item in receipt["files"]:
            T.check_fingerprint(item)
    else:
        if merged.exists():
            raise ValueError("Partial terminal merge has no receipt; do not silently replace it")
        merged.parent.mkdir(parents=True, exist_ok=True)
        original_resolver = merger.merged_dir_for
        try:
            merger.merged_dir_for = lambda _adapter: merged
            actual = merger.merge(terminal)
        finally:
            merger.merged_dir_for = original_resolver
        if Path(actual).resolve() != merged.resolve():
            raise ValueError("Merge escaped the isolated pilot evaluation directory")
        retention = merger.recorded_retention(merged)
        receipt = {
            "binding_sha256": inputs["binding_sha256"],
            "step": inputs["step"],
            "adapter_chain": [str(item) for item in expected_chain],
            "merge_retention": retention,
            "files": [T.fingerprint(p) for p in sorted(merged.iterdir()) if p.is_file()],
        }
        T.atomic_json(path, receipt)
    validate_retention(receipt["merge_retention"])
    return merged, receipt


def generate_terminal(inputs, merged, receipt):
    from hiring_bias_mitigation.eval.backends import build_backend
    from hiring_bias_mitigation.eval.parsing import parse_frame

    out = inputs["out"] / "terminal"
    out.mkdir(parents=True, exist_ok=True)
    model_binding = T.digest(receipt)
    expected = inputs["expected"]
    prompts = expected.prompt.tolist()
    cfg = {
        **inputs["contract"]["manifest"]["source_trajectory"]["audit_config"],
        "model": str(merged),
    }
    cfg.pop("mitigation", None)
    chunk_size = inputs["config"]["pilot_results"]["chunk_size"]
    backend = None
    outputs = []
    try:
        for start in range(0, len(prompts), chunk_size):
            count = min(chunk_size, len(prompts) - start)
            path = out / f"chunk-{start:05d}.json"
            if path.exists():
                chunk = read_chunk(path, inputs["binding_sha256"], model_binding, start, count)
            else:
                if backend is None:
                    backend = build_backend(cfg)
                chunk = backend.generate(prompts[start : start + count])
                if len(chunk) != count or any(not isinstance(value, str) for value in chunk):
                    raise ValueError("Terminal generation returned invalid output count or type")
                T.atomic_json(
                    path,
                    {
                        "binding_sha256": inputs["binding_sha256"],
                        "model_binding": model_binding,
                        "start": start,
                        "outputs": chunk,
                    },
                )
            outputs.extend(chunk)
            T.atomic_json(
                inputs["out"] / "finalize_progress.json",
                {
                    "status": "generating_terminal",
                    "step": inputs["step"],
                    "rows_done": len(outputs),
                    "rows_total": len(prompts),
                    "new_training": False,
                    "job_id": os.getenv("SLURM_JOB_ID"),
                },
            )
    finally:
        if backend is not None:
            backend.close()
    raw = parse_frame(expected.drop(columns="prompt").assign(raw_output=outputs))
    T.atomic_parquet(out / "raw.parquet", raw)
    return raw, T.fingerprint(out / "raw.parquet"), model_binding


def finalize(config_path):
    import fcntl

    import pandas as pd

    inputs = load_inputs(config_path)
    out, expected = inputs["out"], inputs["expected"]
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".finalize.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out / "pilot_results.json"
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior["binding_sha256"] != inputs["binding_sha256"]:
                raise ValueError("Prior pilot result belongs to different terminal/protocol")
            T.check_fingerprint(prior["terminal_raw"])
            if prior["status"] != "complete":
                raise ValueError(
                    "Prior diagnostic failure is retained; do not silently retry inference"
                )
            return prior
        started = time.monotonic()
        callback_frames = {}
        for record in inputs["records"]:
            if record["full"] is not None:
                spec = record["full"]["decisions_file"]
                current = callback_raw(pd.read_parquet(spec["path"]), expected)
                previous = callback_frames.get(record["step"])
                if (
                    previous is not None
                    and not exact_decision_check(current, previous, expected)["exact_match"]
                ):
                    raise ValueError("Repeated full callback checks at the same step disagree")
                callback_frames[record["step"]] = current
        startup = exact_decision_check(
            callback_frames[START], inputs["cached_raw"]["step425"], expected
        )
        if not startup["exact_match"]:
            raise ValueError("Full startup gate does not reproduce cached step425 decisions")
        merged, receipt = prepare_merged(inputs)
        raw, raw_file, model_binding = generate_terminal(inputs, merged, receipt)
        try:
            check = exact_decision_check(raw, callback_frames[inputs["step"]], expected)
        except ValueError as error:
            check = {"exact_match": False, "reason": str(error), "rows": len(raw)}
        common = {
            "schema": "grpo-low-lr-pilot-results-v1",
            "binding_sha256": inputs["binding_sha256"],
            "monitor_identity_sha256": inputs["contract"]["identity_sha256"],
            "monitor_manifest_sha256": inputs["contract"]["manifest_sha256"],
            "terminal_step": inputs["step"],
            "terminal_checkpoint": str(inputs["terminal"]),
            "terminal_selection": "first latched stop, otherwise predeclared step500",
            "stop": inputs["state"]["stop"],
            "stop_step": inputs["state"].get("stop_step"),
            "stop_reason": inputs["state"].get("reason"),
            "last_safe_step": inputs["state"].get("last_safe_step"),
            "last_safe_checkpoint": inputs["state"].get("last_safe_checkpoint"),
            "terminal_raw": raw_file,
            "model_binding": model_binding,
            "terminal_merge_receipt": T.fingerprint(out / "terminal_merge_receipt.json"),
            "callback_terminal_exact_check": check,
            "terminal_inputs": inputs["files"],
            "source_controls": inputs["contract"]["manifest"]["source_trajectory"]["cached_models"],
            "monitor_checks": inputs["records"],
            "new_model_inference_count": 1,
            "new_training_performed_by_finalizer": False,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.monotonic() - started,
        }
        if not check["exact_match"]:
            failed = {
                **common,
                "status": "diagnostic_failed",
                "interpretation_allowed": False,
                "reason": (
                    "Merged terminal and in-training full decisions differ; "
                    "no retraining or model conclusion"
                ),
            }
            T.atomic_json(result_path, failed)
            T.atomic_json(
                out / "finalize_progress.json",
                {"status": "diagnostic_failed", "step": inputs["step"]},
            )
            raise ValueError(failed["reason"])
        parent = inputs["cached_raw"]["SFT-v3"]
        pilot = {
            step: scope_metrics(frame, parent, inputs["partitions"])
            for step, frame in callback_frames.items()
        }
        controls = {
            step: scope_metrics(inputs["cached_raw"][f"step{step}"], parent, inputs["partitions"])
            for step in GRID
        }
        result = {
            **common,
            "status": "complete",
            "interpretation_allowed": True,
            "terminal_scopes": scope_metrics(
                align_raw(raw, expected), parent, inputs["partitions"]
            ),
            "parent_scopes": scope_metrics(parent, parent, inputs["partitions"]),
            "trajectory": {
                "pilot": [{"step": step, "scopes": pilot[step]} for step in sorted(pilot)],
                "historical_original_lr": [
                    {"step": step, "scopes": controls[step]} for step in GRID
                ],
            },
            "fixed_grid_comparison": curve_summary(pilot, controls, inputs["state"]["stop"]),
            "limitations": [
                "Historical original-LR continuation is not a contemporaneous restarted control.",
                "Smaller LR reduces the update budget; the mechanism is not identified.",
                "The benchmark and both partitions are previously exposed development data.",
                "Five-step checks and a two-pp buffer cannot guarantee avoiding a ten-pp drop.",
                "SDT uses all-cell Hautus correction; a binary point is not a complete ROC.",
                "Raw counterfactual instability does not replace matched-operating-point fairness.",
                "Early stopping stays latched; recovery or checkpoint selection cannot erase it.",
            ],
        }
        T.atomic_json(result_path, result)
        write_csv(result, out / "pilot_metrics.csv")
        T.atomic_json(
            out / "finalize_progress.json",
            {"status": "complete", "step": inputs["step"], "new_training": False},
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "finalize"])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        inputs = load_inputs(args.config.resolve(), require_terminal=False)
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "monitor_identity_sha256": inputs["contract"]["identity_sha256"],
                    "outer_manifest": inputs["outer_file"],
                    "new_inference": False,
                    "authorized_bounds": inputs["outer"]["bounds"],
                },
                indent=2,
            )
        )
        return
    result = finalize(args.config.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "terminal_step": result["terminal_step"],
                "stop": result["stop"],
                "exact_match": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
