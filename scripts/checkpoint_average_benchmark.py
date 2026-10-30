"""One-model inference for the equal effective-ΔW mean of seven frozen GRPO checkpoints.

prepare is CPU-only and must follow a commit. run builds/merges the mean adapter and evaluates
that model once. Seven checkpoint audits and the SFT parent are immutable cache inputs.
The experiment writes only to its own output directory and never trains or submits a job.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STEPS = (425, 450, 475, 500, 525, 550, 571)
CODE = ["docs/GRPO_CHECKPOINT_WEIGHT_AVERAGE.md",
        "scripts/checkpoint_average_benchmark.py", "scripts/analyze_checkpoint_average_sdt.py",
        "scripts/checkpoint_benchmark_trajectory.py",
        "scripts/cscs/checkpoint_average_benchmark.sbatch",
        "src/hiring_bias_mitigation/mitigation/lora_average.py",
        "src/hiring_bias_mitigation/mitigation/merge.py",
        "src/hiring_bias_mitigation/eval/backends.py",
        "src/hiring_bias_mitigation/eval/parsing.py",
        "src/hiring_bias_mitigation/analysis/reference_metrics.py"]


def script_module(name):
    key = f"frozen_average_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


T = script_module("checkpoint_benchmark_trajectory")


def validate_config(config):
    if tuple(config.get("steps", ())) != STEPS:
        raise ValueError("exactly the seven predeclared checkpoints, in order, are required")
    fixed = {"aggregation": "equal_effective_delta_W", "weight_numerator": 1,
             "weight_denominator": 7, "expected_output_lora_r": 224,
             "expected_output_lora_alpha": 64, "expected_rows": 31050,
             "tensor_parallel_size": 1, "minimum_merge_retention": .8,
             "training_authorized": False, "new_model_inference_count": 1,
             "no_extra_margin_pass": True}
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError("average experiment protocol differs from its frozen declaration")
    if config.get("analysis") != {"average_label": "mean7", "bootstrap_replicates": 10000,
                                  "seed": 20261023}:
        raise ValueError("SDT analysis settings differ from preregistration")
    if not isinstance(config.get("chunk_size"), int) or config["chunk_size"] <= 0:
        raise ValueError("positive chunk_size required")


def require_committed(paths):
    paths = [str(Path(p).resolve().relative_to(ROOT)) for p in paths]
    for path in paths:
        subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    changed = subprocess.check_output(["git", "status", "--porcelain", "--", *paths],
                                      cwd=ROOT, text=True)
    if changed.strip():
        raise ValueError("commit all experiment/scorer changes before prepare or run")


def checked_output(config):
    from hiring_bias_mitigation.utils.config import resolve_output_path

    out = Path(resolve_output_path(config["output_dir"])).resolve()
    source = Path(resolve_output_path(config["source_trajectory_dir"])).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("average output must be separate from the immutable source trajectory")
    return out, source


def deduplicate_fingerprints(items):
    found = {}
    for item in items:
        prior = found.setdefault(item["path"], item)
        if prior != item:
            raise ValueError("different frozen identities for the same path")
    return [found[path] for path in sorted(found)]


def validate_cached_models(source_manifest, source_results, expected):
    """Select only the eight existing raw files, after checking identities and row alignment."""
    import pandas as pd

    selected = {}
    inventory = {row["step"]: row for row in source_manifest["steps"]}
    for step in (0, *STEPS):
        if step not in source_results or step not in inventory:
            raise ValueError(f"missing verified cached source step {step}; do not regenerate it")
        result = source_results[step]
        if result["binding_sha256"] != source_manifest["binding_sha256"]:
            raise ValueError("cached source belongs to another trajectory")
        T.check_fingerprint(result["raw"])
        raw = T.aligned_raw(pd.read_parquet(result["raw"]["path"]), expected)
        if result["metrics"]["invalid_n"] != 0:
            raise ValueError("cached source has unusable decisions; review before averaging")
        if T.metrics(raw)["invalid_n"] != 0:
            raise ValueError("cached invalid-decision metadata differs from raw")
        label = "SFT-v3" if step == 0 else f"step{step}"
        selected[label] = {"step": step, "raw": result["raw"],
                           "adapter": inventory[step]["adapter"],
                           "adapter_chain": inventory[step]["chain"]}
    parent_chain = selected["SFT-v3"]["adapter_chain"]
    for step in STEPS:
        chain = selected[f"step{step}"]["adapter_chain"]
        if not parent_chain or chain[:-1] != parent_chain:
            raise ValueError("source checkpoints do not share the frozen SFT parent chain")
    return selected


def prepare(config_path):
    import pandas as pd

    config = json.loads(config_path.read_text())
    validate_config(config)
    require_committed([ROOT / p for p in CODE] + [config_path])
    out, source = checked_output(config)
    if (out / "manifest.json").exists():
        raise FileExistsError("experiment already frozen; use run/summarize without replacing it")
    source_manifest = T.load_manifest(source, verify=True)
    if T.software() != source_manifest["software"]:
        raise ValueError("serving software differs from the frozen source trajectory")
    if source_manifest["binding_sha256"] != config["source_trajectory_binding_sha256"]:
        raise ValueError("source trajectory binding differs from preregistration")
    source_results = T.validated_results(source, source_manifest)
    if set(source_results) != {r["step"] for r in source_manifest["steps"]}:
        raise ValueError("source trajectory must already be complete")
    frame = pd.read_parquet(source_manifest["prompts"]["path"])
    if len(frame) != 31050 or frame["candidate_id"].nunique() != 150:
        raise ValueError("unexpected full benchmark population")
    from hiring_bias_mitigation.eval.runner import _build_prompts
    if _build_prompts(frame, "baseline") != frame["prompt"].tolist():
        raise ValueError("benchmark prompt code no longer reproduces frozen prompt text")
    cfg = source_manifest["audit_config"]
    generation = cfg["generation"]
    if (generation["temperature"] != 0.0 or generation["top_p"] != 1.0
            or generation["max_new_tokens"] != 512 or generation.get("n_samples", 1) != 1
            or cfg.get("chat_template_kwargs") != {"enable_thinking": False}
            or cfg.get("backend") != "vllm" or cfg.get("tensor_parallel_size", 1) != 1):
        raise ValueError("original greedy512 thinking-off protocol required")
    cached = validate_cached_models(source_manifest, source_results, frame)
    inputs = [*source_manifest["inputs"], source_manifest["prompts"],
              T.fingerprint(source / "manifest.json"), T.fingerprint(source / "trajectory.json"),
              T.fingerprint(config_path), *[T.fingerprint(ROOT / p) for p in CODE]]
    for model in cached.values():
        inputs += [model["raw"], T.fingerprint(T.step_path(source, model["step"]) / "result.json")]
        for adapter in model["adapter_chain"]:
            for item in adapter["files"]:
                T.check_fingerprint(item)
                inputs.append(item)
    manifest = {"schema": "effective-delta-average-benchmark-v1", "config": config,
                "source_trajectory": {"directory": str(source),
                    "binding_sha256": source_manifest["binding_sha256"]},
                "inputs": deduplicate_fingerprints(inputs), "cached_models": cached,
                "average_adapter_dirs": [cached[f"step{s}"]["adapter"] for s in STEPS],
                "prompts": source_manifest["prompts"],
                "prompt_text_sha256": source_manifest["prompt_text_sha256"],
                "audit_config": cfg, "benchmark_partition": source_manifest["benchmark_partition"],
                "software": T.software(), "prepared_commit": T.git_commit(),
                "status": "inference_only_exploratory", "training_authorized": False,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    manifest["binding_sha256"] = T.digest(manifest)
    T.atomic_json(out / "manifest.json", manifest)
    T.atomic_json(out / "progress.json", {"status": "prepared", "new_models_to_infer": 1,
                    "cached_models": list(cached), "training_authorized": False})
    print(json.dumps({"manifest": str(out / "manifest.json"),
                      "binding_sha256": manifest["binding_sha256"],
                      "new_model_inference_count": 1, "cached_models": list(cached)}, indent=2))


def load_manifest(out, verify=False):
    manifest = json.loads((out / "manifest.json").read_text())
    binding = manifest.pop("binding_sha256")
    if T.digest(manifest) != binding:
        raise ValueError("average experiment manifest binding mismatch")
    manifest["binding_sha256"] = binding
    validate_config(manifest["config"])
    if verify:
        require_committed([ROOT / p for p in CODE])
        for item in manifest["inputs"]:
            T.check_fingerprint(item)
        if T.software() != manifest["software"]:
            raise ValueError("serving software changed since freeze")
    return manifest


def check_owned_receipt(path, binding):
    receipt = json.loads(path.read_text())
    if receipt["binding_sha256"] != binding:
        raise ValueError("derived average artifact belongs to another experiment")
    for item in receipt["files"]:
        T.check_fingerprint(item)
    return receipt


def prepare_merged_model(out, manifest):
    from hiring_bias_mitigation.mitigation.lora_average import (
        build_average,
        merge_average,
        validate_average,
    )

    adapters = [Path(p) for p in manifest["average_adapter_dirs"]]
    adapter_dir, merged_dir = out / "adapter", out / "merged"
    adapter_receipt, merged_receipt = out / "adapter_receipt.json", out / "merge_receipt.json"
    binding = manifest["binding_sha256"]
    if adapter_receipt.exists():
        check_owned_receipt(adapter_receipt, binding)
        average = validate_average(adapter_dir, adapters)
    else:
        if adapter_dir.exists():
            raise RuntimeError("partial average adapter has no success receipt; never overwrite")
        average = build_average(adapters, adapter_dir)
        average = validate_average(adapter_dir, adapters)
        files = [T.fingerprint(adapter_dir / name) for name in
                 ("adapter_config.json", "adapter_model.safetensors", "run_config.json",
                  "average_manifest.json")]
        T.atomic_json(adapter_receipt, {"binding_sha256": binding, "files": files,
                                       "average": average})
    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    if adapter_cfg["r"] != 224 or adapter_cfg["lora_alpha"] != 64:
        raise ValueError("mean adapter rank/alpha differs from frozen concatenation protocol")
    if merged_receipt.exists():
        receipt = check_owned_receipt(merged_receipt, binding)
    else:
        if merged_dir.exists():
            raise RuntimeError("partial merged average has no success receipt; never overwrite")
        merge = merge_average(adapter_dir, merged_dir)
        files = [T.fingerprint(p) for p in sorted(merged_dir.iterdir()) if p.is_file()]
        receipt = {"binding_sha256": binding, "files": files, "merge": merge}
        T.atomic_json(merged_receipt, receipt)
    validate_retention(receipt["merge"]["merge_retention"])
    return merged_dir, receipt


def validate_retention(retention):
    value = retention[-1].get("kept_fraction") if retention else None
    if (isinstance(value, bool) or not isinstance(value, (float, int))
            or not math.isfinite(value) or value < .8):
        raise ValueError("mean merge retention is invalid or below 80%; no inference or SDT")


def read_chunk(path, binding, model_binding, start, count):
    chunk = json.loads(path.read_text())
    if (chunk.get("binding_sha256") != binding or chunk.get("model_binding") != model_binding
            or chunk.get("start") != start or len(chunk.get("outputs", [])) != count
            or any(not isinstance(x, str) for x in chunk["outputs"])):
        raise ValueError("partial average generations have different identity/order/count")
    return chunk["outputs"]


def summarize(out):
    import pandas as pd

    manifest = load_manifest(out, verify=True)
    result_path = out / "average" / "result.json"
    if not result_path.exists():
        raise ValueError("average inference is not complete; no SDT conclusion")
    result = json.loads(result_path.read_text())
    if result["binding_sha256"] != manifest["binding_sha256"]:
        raise ValueError("average result belongs to another experiment")
    T.check_fingerprint(result["raw"])
    check_owned_receipt(out / "adapter_receipt.json", manifest["binding_sha256"])
    merge_receipt = check_owned_receipt(out / "merge_receipt.json", manifest["binding_sha256"])
    validate_retention(merge_receipt["merge"]["merge_retention"])
    if result["model_binding"] != T.digest(merge_receipt):
        raise ValueError("average inference and retained merged model identities differ")
    if result["metrics"]["invalid_n"] != 0:
        raise ValueError("mean model has unusable decisions; no SDT interpretation")
    raw_by_model = {label: pd.read_parquet(spec["raw"]["path"])
                    for label, spec in manifest["cached_models"].items()}
    raw_by_model["mean7"] = pd.read_parquet(result["raw"]["path"])
    analyzer = script_module("analyze_checkpoint_average_sdt")
    analysis = analyzer.analyze(
        raw_by_model, T.load_partitions(manifest), **manifest["config"]["analysis"])
    analyzer.write_csv(analysis, out / "sdt_metrics.csv")
    T.atomic_json(out / "sdt_analysis.json", {"binding_sha256": manifest["binding_sha256"],
                    "status": "complete", "analysis": analysis,
                    "raw_sources": {**{k: v["raw"] for k, v in manifest["cached_models"].items()},
                                    "mean7": result["raw"]},
                    "merge_receipt": T.fingerprint(out / "merge_receipt.json"),
                    "training_authorized": False})
    return analysis


def run(out):
    import fcntl

    import pandas as pd

    from hiring_bias_mitigation.eval.backends import build_backend
    from hiring_bias_mitigation.eval.parsing import parse_frame

    lock = (out / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = load_manifest(out, verify=True)
    if (out / "average" / "result.json").exists():
        summarize(out)
        return
    started = time.monotonic()
    def progress(status, **extra):
        T.atomic_json(out / "progress.json", {"status": status, "job_id": os.getenv("SLURM_JOB_ID"),
                    "training_authorized": False, **extra})
    progress("building_and_merging_average")
    try:
        merged_dir, merge_receipt = prepare_merged_model(out, manifest)
        model_binding = T.digest(merge_receipt)
        frame = pd.read_parquet(manifest["prompts"]["path"])
        prompts = frame["prompt"].tolist()
        if T.digest(prompts) != manifest["prompt_text_sha256"]:
            raise ValueError("benchmark prompt identity differs")
        cfg = {**manifest["audit_config"], "model": str(merged_dir)}
        cfg.pop("mitigation", None)
        backend = build_backend(cfg)
        done = []
        chunk_size = manifest["config"]["chunk_size"]
        folder = out / "average"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            for start in range(0, len(prompts), chunk_size):
                count = min(chunk_size, len(prompts) - start)
                path = folder / f"chunk-{start:05d}.json"
                if path.exists():
                    outputs = read_chunk(path, manifest["binding_sha256"],
                                         model_binding, start, count)
                else:
                    outputs = backend.generate(prompts[start:start + count])
                    if len(outputs) != count or any(not isinstance(x, str) for x in outputs):
                        raise ValueError("average generation returned invalid output count/type")
                    T.atomic_json(path, {"binding_sha256": manifest["binding_sha256"],
                                         "model_binding": model_binding, "start": start,
                                         "outputs": outputs})
                done.extend(outputs)
                progress("generating", rows_done=len(done), rows_total=len(prompts))
                print(f"mean7: {len(done)}/{len(prompts)}", flush=True)
        finally:
            backend.close()
        raw = parse_frame(frame.drop(columns="prompt").assign(raw_output=done))
        T.atomic_parquet(folder / "raw.parquet", raw)
        parent = pd.read_parquet(manifest["cached_models"]["SFT-v3"]["raw"]["path"])
        result = {"binding_sha256": manifest["binding_sha256"], "model_binding": model_binding,
                  "label": "mean7", "metrics": T.metrics(raw),
                  "scopes": T.partition_metrics(raw, parent, T.load_partitions(manifest)),
                  "raw": T.fingerprint(folder / "raw.parquet"),
                  "provenance": {"kind": "new_average_inference",
                      "job_id": os.getenv("SLURM_JOB_ID"),
                      "commit": T.git_commit(), "software": T.software(),
                      "merge_retention": merge_receipt["merge"]["merge_retention"],
                      "wall_seconds": time.monotonic() - started},
                  "completed_utc": datetime.now(timezone.utc).isoformat()}
        T.atomic_json(folder / "result.json", result)
        if result["metrics"]["invalid_n"] != 0:
            raise ValueError("mean model has unusable decisions; retained raw, stopped before SDT")
        summarize(out)
        progress("complete", rows_done=len(done), rows_total=len(prompts),
                 wall_seconds=time.monotonic() - started)
    except Exception as exc:
        progress("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, required=True)
    for command in ("run", "summarize"):
        parser_sub = sub.add_parser(command)
        parser_sub.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.config.resolve())
    elif args.command == "run":
        run(args.output_dir.resolve())
    else:
        summarize(args.output_dir.resolve())


if __name__ == "__main__":
    main()
