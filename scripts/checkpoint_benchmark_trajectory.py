"""Inference-only benchmark trajectory, isolated from the published benchmark tables.

prepare freezes inputs and imports verified existing audits. run evaluates all 18 missing
checkpoints in one job with four independent TP1 GPU workers. score-next is the worker.
No command in this script trains a model or submits a job.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
CODE = ["scripts/checkpoint_benchmark_trajectory.py",
        "src/hiring_bias_mitigation/eval/runner.py",
        "src/hiring_bias_mitigation/eval/backends.py",
        "src/hiring_bias_mitigation/eval/parsing.py",
        "src/hiring_bias_mitigation/data/prompts.py",
        "src/hiring_bias_mitigation/data/injection.py",
        "src/hiring_bias_mitigation/data/benchmark.py",
        "src/hiring_bias_mitigation/mitigation/merge.py",
        "src/hiring_bias_mitigation/analysis/reference_metrics.py"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def atomic_parquet(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size}


def check_fingerprint(item):
    path = Path(item["path"])
    if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
        raise ValueError(f"frozen input changed: {path}")


def git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def software():
    return {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")}


def aligned_raw(raw, expected):
    """Fail closed on any missing, duplicated, extra, or relabelled benchmark row."""
    if len(raw) != len(expected) or raw.duplicated(KEY).any():
        raise ValueError("raw audit row count or unique key mismatch")
    columns = [*KEY, "reference_decision"]
    left = expected[columns].copy()
    left["_order"] = range(len(left))
    joined = left.merge(raw, on=KEY, how="outer", suffixes=("_expected", ""),
                        validate="one_to_one", indicator=True).sort_values("_order")
    if not (joined["_merge"] == "both").all():
        raise ValueError("raw audit keys mismatch")
    if not (joined["reference_decision_expected"] == joined["reference_decision"]).all():
        raise ValueError("raw audit labels mismatch")
    return joined[raw.columns].reset_index(drop=True)


def metrics(raw):
    """Keep the repository's usable-decision denominator; report coverage separately."""
    from hiring_bias_mitigation.analysis.reference_metrics import hire_by_reference, prepare

    rows = prepare(raw)
    good = rows["d"].notna()
    hire = rows["ref"] == "hire"
    reject = rows["ref"] == "reject"
    out = {**hire_by_reference(rows), "n": len(raw),
           "reference_hire_n": int(hire.sum()), "reference_reject_n": int(reject.sum()),
           "decided_n": int(good.sum()), "invalid_n": int((~good).sum()),
           "invalid_pct": 100 * float((~good).mean()),
           "decided_reference_hire_n": int((good & hire).sum()),
           "decided_reference_reject_n": int((good & reject).sum()),
           "tpr_full_denominator_pct": 100 * float((rows.loc[hire, "d"] == "hire").mean()),
           "fpr_full_denominator_pct": 100 * float((rows.loc[reject, "d"] == "hire").mean()),
           "agreement_pct": 100 * float((rows.loc[good, "d"] == rows.loc[good, "ref"]).mean())}
    if out["decided_reference_hire_n"] == 0 or out["decided_reference_reject_n"] == 0:
        raise ValueError("no usable decisions in a reference label")
    return out


def output_path(config):
    from hiring_bias_mitigation.utils.config import resolve_output_path
    return Path(resolve_output_path(config["output_dir"]))


def partition_metrics(raw, parent, partitions):
    """Direct label-conditional decisions on frozen candidate partitions, no projection."""
    from hiring_bias_mitigation.analysis.reference_metrics import prepare

    parent = aligned_raw(parent, raw)
    out = {}
    scopes = {"full": set(raw["candidate_id"]), **partitions}
    for name, candidates in scopes.items():
        mask = raw["candidate_id"].isin(candidates)
        current_raw, parent_raw = raw.loc[mask], parent.loc[mask]
        current, previous = prepare(current_raw), prepare(parent_raw)
        positive = current["ref"] == "hire"
        both = positive & current["d"].notna() & previous["d"].notna()
        loss = int((both & (previous["d"] == "hire") & (current["d"] == "reject")).sum())
        gain = int((both & (previous["d"] == "reject") & (current["d"] == "hire")).sum())
        current_metrics, parent_metrics = metrics(current_raw), metrics(parent_raw)
        n_positive = int(positive.sum())
        out[name] = {**current_metrics, "parent_tpr_pct": parent_metrics["tpr_vs_gpt4o_pct"],
            "delta_tpr_pp": current_metrics["tpr_vs_gpt4o_pct"] -
                            parent_metrics["tpr_vs_gpt4o_pct"],
            "candidate_count": int(current_raw["candidate_id"].nunique()),
            "pair_count": int(current_raw["pair_id"].nunique()),
            "loss_count": loss, "gain_count": gain,
            "loss_pp": 100 * loss / n_positive, "gain_pp": 100 * gain / n_positive,
            "paired_usable_positive_n": int(both.sum()),
            "loss_gain_denominator": "all reference-hire rows; transitions require both usable"}
    return out


def load_partitions(manifest):
    frozen = manifest["benchmark_partition"]
    check_fingerprint(frozen["split"])
    check_fingerprint(frozen["manifest"])
    data = json.loads(Path(frozen["split"]["path"]).read_text())
    return {name: set(spec["candidate_ids"]) for name, spec in data["splits"].items()}


def step_path(out, step):
    return out / "steps" / f"step-{step:04d}"


def finish(out, step, raw, provenance, binding):
    import pandas as pd

    folder = step_path(out, step)
    folder.mkdir(parents=True, exist_ok=True)
    atomic_parquet(folder / "raw.parquet", raw)
    manifest = load_manifest(out)
    parent = raw if step == 0 else pd.read_parquet(step_path(out, 0) / "raw.parquet")
    result = {"step": step, "binding_sha256": binding, "metrics": metrics(raw),
              "scopes": partition_metrics(raw, parent, load_partitions(manifest)),
              "raw": fingerprint(folder / "raw.parquet"), "provenance": provenance,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(folder / "result.json", result)
    return result


def prepare(config_path):
    import pandas as pd

    from hiring_bias_mitigation.eval.runner import _build_prompts, build_frame
    from hiring_bias_mitigation.mitigation.merge import adapter_chain, recorded_retention
    from hiring_bias_mitigation.utils.config import load_config, resolve_output_path

    config = json.loads(config_path.read_text())
    out = output_path(config)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("manifest already frozen; use score-next or summarize")
    cfg = load_config(config["audit_config"])
    frame = build_frame(cfg)
    prompts = _build_prompts(frame, "baseline")
    if len(frame) != 31050 or frame["candidate_id"].nunique() != 150:
        raise ValueError("unexpected benchmark population")
    frame = frame.assign(prompt=prompts)
    atomic_parquet(out / "prompts.parquet", frame)
    run = Path(resolve_output_path(config["training_run"]))
    checkpoints = sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*")
                         if (p / "adapter_model.safetensors").is_file())
    if checkpoints != config["expected_steps"]:
        raise ValueError(f"retained checkpoint inventory changed: {checkpoints}")
    # `score_steps` scores a declared subset of the retained checkpoints (default: all of them).
    scored = config.get("score_steps", checkpoints)
    if not set(scored) <= set(checkpoints):
        raise ValueError(f"score_steps not retained: {sorted(set(scored) - set(checkpoints))}")
    steps = [{"step": 0, "adapter": str(Path(resolve_output_path(config["parent_adapter"])))},
             *[{"step": n, "adapter": str(run / f"checkpoint-{n}")} for n in scored]]
    inputs = [fingerprint(ROOT / p) for p in CODE]
    inputs += [fingerprint(config_path), fingerprint(ROOT / config["audit_config"]),
               fingerprint(ROOT / "data/benchmark/reference_feedback_en.csv"),
               fingerprint(run / "run_config.json")]
    partition = {}
    for name in ("split", "manifest"):
        spec = config["benchmark_partition"][name]
        file = fingerprint(Path(resolve_output_path(spec["path"])))
        if file["sha256"] != spec["sha256"]:
            raise ValueError(f"frozen benchmark partition {name} hash differs")
        partition[name] = file
        inputs.append(file)
    split = json.loads(Path(partition["split"]["path"]).read_text())["splits"]
    dev, remainder = (set(split[n]["candidate_ids"])
                      for n in ("dev", "historical_remainder"))
    if dev & remainder or dev | remainder != set(frame["candidate_id"]):
        raise ValueError("frozen candidate split does not partition the benchmark")
    for name, spec in split.items():
        subset = frame[frame["candidate_id"].isin(spec["candidate_ids"])]
        if set(subset["pair_id"]) != set(spec["pair_ids"]) or len(subset) != spec["prompt_count"]:
            raise ValueError(f"frozen {name} pair/prompt membership differs")
    hub = Path(os.environ.get("HF_HOME", "/ritom/scratch/cscs/zhang/xw/hf-cache")) / "hub"
    base = hub / "models--Qwen--Qwen3.5-9B"
    if (base / "refs/main").read_text().strip() != config["base_revision"]:
        raise ValueError("cached base main revision differs from preregistration")
    inputs.append(fingerprint(base / "refs/main"))
    snapshot = base / "snapshots" / config["base_revision"]
    inputs += [fingerprint(p) for p in sorted(snapshot.iterdir())
               if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".safetensors"}]
    for item in steps:
        item["chain"] = []
        for adapter in adapter_chain(item["adapter"]):
            files = [fingerprint(adapter / "adapter_model.safetensors"),
                     fingerprint(adapter / "adapter_config.json")]
            item["chain"].append({"path": str(adapter), "files": files})
    cache_imports = []
    for spec in config["cache_imports"]:
        item = next(x for x in steps if x["step"] == spec["step"])
        raw_path = Path(resolve_output_path(f"outputs/raw/{spec['run']}.parquet"))
        meta_path = raw_path.with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text())
        cached_cfg = load_config(meta["config_path"])
        if cached_cfg.get("chat_template_kwargs") != {"enable_thinking": False}:
            raise ValueError("cached audit thinking mode differs")
        if any(meta["generation"][k] != v for k, v in cfg["generation"].items()):
            raise ValueError("cached audit decoding differs")
        raw = aligned_raw(pd.read_parquet(raw_path), frame)
        if _build_prompts(raw, "baseline") != prompts:
            raise ValueError("cached audit prompt text differs")
        margin_meta_path = Path(resolve_output_path(
            f"outputs/operating_point/v2/{spec['run']}--adapter.meta.json"))
        margin_meta = json.loads(margin_meta_path.read_text())
        if margin_meta["software"] != software():
            raise ValueError("serving software differs from cached baseline")
        expected_hashes = [a["files"][0]["sha256"] for a in item["chain"]]
        if [a["sha256"] for a in margin_meta["model"]["adapters"]] != expected_hashes:
            raise ValueError("cached audit adapter chain differs")
        if meta["served_model"] != margin_meta["model"]["served"]:
            raise ValueError("raw/margin cache served model mismatch")
        if margin_meta["model"]["base_revision"] != config["base_revision"]:
            raise ValueError("cached base revision differs")
        retention = recorded_retention(meta["served_model"])
        if not retention or retention[-1]["kept_fraction"] < .8:
            raise ValueError("cached merge does not satisfy retention gate")
        cache = {"step": spec["step"], "run": spec["run"],
                 "raw": fingerprint(raw_path), "meta": fingerprint(meta_path),
                 "margin_meta": fingerprint(margin_meta_path), "merge_retention": retention}
        item["cache"] = cache
        cache_imports.append((item, raw))
    manifest = {"schema": "checkpoint-benchmark-trajectory-v1", "config": config,
                "audit_config": cfg, "steps": steps, "inputs": inputs,
                "prompts": fingerprint(out / "prompts.parquet"),
                "prompt_text_sha256": digest(prompts), "software": software(),
                "benchmark_partition": partition,
                "prepared_commit": git_commit(), "status": "inference_only_exploratory",
                "missing_saved_steps": config.get("missing_saved_steps", [25, 50, 75]),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest["binding_sha256"] = digest(manifest)
    atomic_json(out / "manifest.json", manifest)
    for item, raw in cache_imports:
        finish(out, item["step"], raw, {"kind": "verified_cache", **item["cache"]},
               manifest["binding_sha256"])
    summarize(out)
    print(f"FROZEN {out / 'manifest.json'}; {len(checkpoints)} checkpoints, "
          f"{len(steps) - len(cache_imports)} need inference", flush=True)


def load_manifest(out, verify=False):
    manifest = json.loads((out / "manifest.json").read_text())
    binding = manifest.pop("binding_sha256")
    if digest(manifest) != binding:
        raise ValueError("manifest binding mismatch")
    manifest["binding_sha256"] = binding
    if verify:
        for item in [*manifest["inputs"], manifest["prompts"]]:
            check_fingerprint(item)
        if software() != manifest["software"]:
            raise ValueError("serving software changed since freeze")
    return manifest


def validated_results(out, manifest):
    found = {}
    for step in manifest["steps"]:
        path = step_path(out, step["step"]) / "result.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["binding_sha256"] != manifest["binding_sha256"]:
                raise ValueError("result belongs to a different frozen experiment")
            check_fingerprint(result["raw"])
            found[step["step"]] = result
    return found


def score_next(out, requested_step=None):
    import fcntl

    import pandas as pd

    from hiring_bias_mitigation.eval.backends import build_backend
    from hiring_bias_mitigation.eval.parsing import parse_frame
    from hiring_bias_mitigation.mitigation import merge as MG

    manifest = load_manifest(out, verify=True)
    existing = validated_results(out, manifest)
    if any(r["metrics"]["invalid_pct"] > 1.0 for r in existing.values()):
        raise RuntimeError("prior checkpoint exceeded 1% unusable decisions; review required")
    pending = [s for s in manifest["steps"] if s["step"] not in existing]
    if requested_step is not None:
        pending = [s for s in pending if s["step"] == requested_step]
        if not pending and requested_step not in existing:
            raise ValueError("requested checkpoint is not in the frozen inventory")
    if not pending:
        print("NOTHING TO SCORE", flush=True)
        return
    item = pending[0]
    step = item["step"]
    folder = step_path(out, step)
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / ".score.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / "result.json").exists():
        raise RuntimeError("checkpoint completed while waiting for its lock")
    for adapter in item["chain"]:
        for file in adapter["files"]:
            check_fingerprint(file)
    frame = pd.read_parquet(manifest["prompts"]["path"])
    prompts = frame["prompt"].tolist()
    if digest(prompts) != manifest["prompt_text_sha256"]:
        raise ValueError("prompt text fingerprint mismatch")
    started = time.monotonic()
    atomic_json(folder / "progress.json", {"status": "merging", "step": step,
                "job_id": os.getenv("SLURM_JOB_ID"), "completed_steps": sorted(existing)})
    # Only this experiment's temporary merge may be overwritten/deleted. Global merges remain.
    merged = out / "temporary_merges" / f"step-{step:04d}"
    MG.merged_dir_for = lambda adapter: merged
    MG.merge(item["adapter"])
    retention = MG.recorded_retention(merged)
    # A diagnostic run may declare a lower floor (`min_merge_retention`); the standard's is 80%.
    floor = float(manifest["config"].get("min_merge_retention", .8))
    if not retention or retention[-1]["kept_fraction"] < floor:
        raise ValueError(f"new merge keeps less than {floor:.0%} of the update")
    merged_files = [fingerprint(p) for p in sorted(merged.glob("*.safetensors"))]
    cfg = {**manifest["audit_config"], "model": str(merged)}
    cfg.pop("mitigation", None)
    # Deliberately retain legacy max_num_seqs/default, GPU share, ordering and full generation.
    backend = build_backend(cfg)
    done = []
    chunk_size = int(manifest["config"]["chunk_size"])
    try:
        for start in range(0, len(prompts), chunk_size):
            end = min(start + chunk_size, len(prompts))
            path = folder / f"chunk-{start:05d}.json"
            if path.exists():
                chunk = json.loads(path.read_text())
                if (chunk["binding_sha256"] != manifest["binding_sha256"] or
                        chunk["step"] != step or chunk["start"] != start or
                        len(chunk["outputs"]) != end - start):
                    raise ValueError("partial generation identity mismatch")
                outputs = chunk["outputs"]
            else:
                outputs = backend.generate(prompts[start:end])
                if len(outputs) != end - start:
                    raise ValueError("generation returned wrong number of outputs")
                atomic_json(path, {"binding_sha256": manifest["binding_sha256"], "step": step,
                                   "start": start, "outputs": outputs})
            done.extend(outputs)
            atomic_json(folder / "progress.json", {"status": "generating", "step": step,
                "rows_done": end, "rows_total": len(prompts),
                "job_id": os.getenv("SLURM_JOB_ID"), "completed_steps": sorted(existing)})
            print(f"step {step}: {end}/{len(prompts)}", flush=True)
    finally:
        backend.close()
    raw = parse_frame(frame.drop(columns="prompt").assign(raw_output=done))
    result = finish(out, step, raw,
                    {"kind": "new_inference", "job_id": os.getenv("SLURM_JOB_ID"),
                     "commit": git_commit(), "software": software(), "adapter_chain": item["chain"],
                     "merge_retention": retention, "merged_weight_files": merged_files,
                     "wall_seconds": time.monotonic() - started}, manifest["binding_sha256"])
    # Only derived, experiment-owned weights are removed after complete raw outputs are hashed.
    shutil.rmtree(merged)
    atomic_json(folder / "progress.json", {"status": "complete", "step": step,
                "job_id": os.getenv("SLURM_JOB_ID"), "rows_done": len(prompts)})
    if result["metrics"]["invalid_pct"] > 1.0:
        raise RuntimeError("checkpoint completed but >1% unusable decisions; review required")
    print(f"COMPLETE step={step} TPR={result['metrics']['tpr_vs_gpt4o_pct']:.6f}", flush=True)


def run(out):
    """One allocation, four independent TP1 workers, no job submission or training."""
    import fcntl
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = (out / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = load_manifest(out, verify=True)
    initial = validated_results(out, manifest)
    pending = [s["step"] for s in manifest["steps"] if s["step"] not in initial]
    if len(pending) > 18:
        raise ValueError("more than 18 checkpoints requested")
    events = queue.Queue()
    stop = threading.Event()
    workers = int(manifest["config"]["workers"])
    if workers != 4:
        raise ValueError("this frozen run requires four TP1 workers")

    def lane(gpu):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "8"}
        try:
            for step in pending[gpu::workers]:
                if stop.is_set():
                    break
                log = step_path(out, step) / "inference.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("a") as stream:
                    proc = subprocess.run(
                          [sys.executable, str(Path(__file__).resolve()), "score-next",
                          "--output-dir", str(out), "--step", str(step)], env=env,
                          stdout=stream, stderr=subprocess.STDOUT)
                if proc.returncode:
                    stop.set()
                events.put({"step": step, "gpu": gpu, "returncode": proc.returncode})
                if proc.returncode:
                    break
        except Exception as exc:
            stop.set()
            events.put({"step": None, "gpu": gpu, "returncode": -1, "error": repr(exc)})
        finally:
            events.put({"lane_finished": gpu})

    finished, failures = 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(lane, gpu) for gpu in range(workers)]
        while finished < workers:
            event = events.get()
            if "lane_finished" in event:
                finished += 1
            elif event["returncode"]:
                failures.append(event)
            summarize(out)
            atomic_json(out / "progress.json", {"status": "failed" if failures else "running",
                 "job_id": os.getenv("SLURM_JOB_ID"), "last_event": event,
                 "failed_steps": failures, "workers_finished": finished,
                 "completed_steps": sorted(validated_results(out, manifest))})
        for future in futures:
            future.result()
    if failures:
        raise RuntimeError(f"checkpoint inference failed; later work stopped: {failures}")
    summarize(out)
    atomic_json(out / "progress.json", {"status": "complete", "job_id": os.getenv("SLURM_JOB_ID"),
                "completed_steps": [s["step"] for s in manifest["steps"]]})


def first_crossing(curve, retained_steps, flag):
    """First historical stop stays latched even if later checkpoints recover."""
    by_step = {row["step"]: row for row in curve}
    crossing = next((s for s in retained_steps if s in by_step and by_step[s][flag]), None)
    if crossing is None:
        return {"first_observed_crossing_step": None,
                "complete_trajectory": all(s in by_step for s in retained_steps)}
    before = retained_steps[:retained_steps.index(crossing)]
    immediate = before[-1] if before else None
    known = all(s in by_step for s in before)
    return {"first_observed_crossing_step": crossing,
            "first_crossing_confirmed_among_retained_steps": known,
            "preceding_retained_step": immediate,
            "preceding_retained_safe_step": immediate if known and immediate is not None
                and not by_step[immediate][flag] else None,
            "missing_earlier_steps": [s for s in before if s not in by_step],
            "later_recovery_does_not_revoke_stop": True}


def summarize(out):
    manifest = load_manifest(out)
    records = validated_results(out, manifest)
    if 0 not in records:
        raise ValueError("missing frozen SFT-v3 baseline")
    baseline = records[0]["metrics"]["tpr_vs_gpt4o_pct"]
    curve = []
    for step, result in sorted(records.items()):
        row = {"step": step, **result["metrics"], "delta_tpr_pp":
               result["metrics"]["tpr_vs_gpt4o_pct"] - baseline,
               "source": result["provenance"]["kind"]}
        row["scopes"] = result["scopes"]
        row["full_8pp_alert"] = row["delta_tpr_pp"] <= -8.0
        row["dev_8pp_candidate_alert"] = row["scopes"]["dev"]["delta_tpr_pp"] <= -8.0
        # EVAL_STANDARD allows a drop of at most 10pp; exactly -10 still passes.
        row["full_fails_10pp_guardrail"] = row["delta_tpr_pp"] < -10.0
        curve.append(row)
    retained = [s["step"] for s in manifest["steps"]]
    crossings = {flag: first_crossing(curve, retained, flag) for flag in
                 ("full_8pp_alert", "full_fails_10pp_guardrail", "dev_8pp_candidate_alert")}
    atomic_json(out / "trajectory.json", {"binding_sha256": manifest["binding_sha256"],
                "complete": len(records) == len(manifest["steps"]), "rows": curve,
                "first_crossings": crossings,
                "pending_steps": [s["step"] for s in manifest["steps"] if s["step"] not in records],
                "interpretation": "exploratory repeated benchmark use; no fresh test claim"})
    print(json.dumps({"completed_steps": list(sorted(records)), "pending_count":
                      len(manifest["steps"]) - len(records)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    for command in ("score-next", "summarize", "run"):
        child = sub.add_parser(command)
        child.add_argument("--output-dir", type=Path, required=True)
        if command == "score-next":
            child.add_argument("--step", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config.resolve())
    elif args.command == "score-next":
        score_next(args.output_dir.resolve(), args.step)
    elif args.command == "run":
        run(args.output_dir.resolve())
    else:
        summarize(args.output_dir.resolve())


if __name__ == "__main__":
    main()
