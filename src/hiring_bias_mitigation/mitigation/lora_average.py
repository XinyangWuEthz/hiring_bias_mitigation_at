"""Exact equal-weight average of ordinary LoRA effective updates by rank concatenation.

For n identical configurations with scaling alpha/r, A*=vertical_concat(A_i),
B*=horizontal_concat(B_i), r*=n*r and alpha*=alpha. Consequently
alpha*/r* B*A* = (1/n) sum_i alpha/r B_i A_i. Source FP32 factor values are copied
unchanged, never averaged, rescaled, quantized or truncated. Equality is algebraic;
floating-point matrix multiplication and the final bf16 merge still round.

This deliberately supports only the ordinary, uniform-rank, FP32 linear LoRA format
of this experiment. Unsupported PEFT variants fail before producing an adapter.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

from . import merge as MG


def file_fingerprint(path: Path, recorded_path: Path | None = None) -> dict:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str((recorded_path or path).resolve()), "sha256": h.hexdigest(),
            "bytes": path.stat().st_size}


def verify_file(record: dict) -> None:
    if file_fingerprint(Path(record["path"])) != record:
        raise ValueError(f"file fingerprint mismatch: {record['path']}")


def _binding(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _equal_weights(n: int, weights=None) -> list[float]:
    if n < 2:
        raise ValueError("averaging requires at least two distinct adapters")
    expected = [1.0 / n] * n
    if weights is not None and (len(weights) != n or any(
            not math.isfinite(float(w)) or abs(float(w) - 1.0 / n) > 1e-15 for w in weights)):
        raise ValueError("only equal normalized weights are supported")
    return expected


def _validate_config(config: dict) -> None:
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("only causal-LM LoRA adapters are supported")
    rank, alpha = config.get("r"), config.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (
            math.isfinite(alpha) and alpha > 0):
        raise ValueError("lora_alpha must be finite and positive")
    if config.get("bias", "none") != "none":
        raise ValueError("bias updates are unsupported")
    unsupported = ("fan_in_fan_out", "lora_bias", "use_dora", "use_rslora", "use_qalora",
                   "rank_pattern", "alpha_pattern", "modules_to_save", "target_parameters",
                   "trainable_token_indices", "layer_replication", "layers_pattern",
                   "layers_to_transform", "exclude_modules", "alora_invocation_tokens",
                   "arrow_config", "corda_config", "eva_config", "kasa_config", "loftq_config",
                   "lora_ga_config", "megatron_config", "monteclora_config", "use_bdlora",
                   "velora_config", "ensure_weight_tying", "auto_mapping", "revision")
    for field in unsupported:
        if config.get(field):
            raise ValueError(f"unsupported LoRA option {field}={config[field]!r}")
    if config.get("lora_dropout", 0) != 0 or config.get("init_lora_weights", True) is not True:
        raise ValueError("only zero-dropout, ordinary initialized LoRA is supported")
    target = config.get("target_modules")
    if not isinstance(target, list) or not target or not all(isinstance(x, str) for x in target):
        raise ValueError("target_modules must be a nonempty explicit list")
    if not config.get("base_model_name_or_path"):
        raise ValueError("missing base_model_name_or_path")
    known = {"peft_type", "task_type", "r", "lora_alpha", "bias", "lora_dropout",
             "init_lora_weights", "target_modules", "base_model_name_or_path", "inference_mode",
             "peft_version", "qalora_group_size", "megatron_core", *unsupported}
    if set(config) - known:
        raise ValueError(f"unrecognized PEFT configuration fields: {sorted(set(config) - known)}")


def _adapter_files(path: Path) -> list[dict]:
    files = [path / "adapter_config.json", path / "adapter_model.safetensors"]
    run = path.parent if path.name.startswith("checkpoint-") else path
    if (run / "run_config.json").exists():
        files.append(run / "run_config.json")
    return [file_fingerprint(p) for p in files]


def _inspect_sources(adapters: list[Path], weights=None) -> dict:
    adapters = [Path(p).resolve() for p in adapters]
    normalized = _equal_weights(len(adapters), weights)
    if len(set(adapters)) != len(adapters):
        raise ValueError("source adapters must be distinct")
    configs = [json.loads((p / "adapter_config.json").read_text()) for p in adapters]
    for config in configs:
        _validate_config(config)
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("source adapter configurations must be identical")
    chains = [MG.adapter_chain(p) for p in adapters]
    parents = [[p.resolve() for p in chain[:-1]] for chain in chains]
    if not parents[0] or any(p != parents[0] for p in parents[1:]):
        raise ValueError("all adapters must share one identical nonempty SFT parent chain")
    if set(adapters) & set(parents[0]):
        raise ValueError("an averaged adapter cannot also be in its common parent chain")
    return {"config": configs[0], "weights": normalized,
            "inputs": [{"path": str(p), "files": _adapter_files(p)} for p in adapters],
            "parent_chain": [{"path": str(p), "files": _adapter_files(p)} for p in parents[0]]}


def _tensor_hash(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def concatenate_factors(a_parts, b_parts, alpha: float):
    """Return untouched FP32 factors and scale; reject inconsistent shapes or values."""
    import torch

    n = len(a_parts)
    _equal_weights(n)
    if len(b_parts) != n:
        raise ValueError("A/B source counts differ")
    rank, input_size = a_parts[0].shape
    output_size = b_parts[0].shape[0]
    for a, b in zip(a_parts, b_parts):
        if a.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("source factors must be FP32; no implicit dtype conversion")
        if a.shape != (rank, input_size) or b.shape != (output_size, rank):
            raise ValueError("source A/B shapes differ")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError("nonfinite source factor")
    a_out, b_out = torch.cat(a_parts, dim=0), torch.cat(b_parts, dim=1)
    identities = []
    for i, (a, b) in enumerate(zip(a_parts, b_parts)):
        ah, bh = _tensor_hash(a), _tensor_hash(b)
        identical = (ah == _tensor_hash(a_out[i * rank:(i + 1) * rank, :]) and
                     bh == _tensor_hash(b_out[:, i * rank:(i + 1) * rank]))
        if not identical:
            raise ValueError("concatenation changed an input factor block")
        identities.append({"source_index": i, "a_sha256": ah, "b_sha256": bh,
                           "output_blocks_bit_identical": identical})
    return a_out, b_out, alpha / (n * rank), identities


def _numerical_check(a_parts, b_parts, a_out, b_out, alpha, seed):
    """Check every module on fixed FP64 probes and sampled dense cells, not a full dense SVD."""
    import torch

    n, rank = len(a_parts), a_parts[0].shape[0]
    scale = alpha / rank
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(a_out.shape[1], 8, dtype=torch.float64, generator=generator)
    x /= math.sqrt(a_out.shape[1])
    indices = sorted({0, a_out.shape[1] // 2, a_out.shape[1] - 1})
    basis = torch.zeros(a_out.shape[1], len(indices), dtype=torch.float64)
    basis[indices, range(len(indices))] = 1.0
    x = torch.cat([x, basis], dim=1)
    records = {}
    for dtype, name in [(torch.float64, "fp64"), (torch.float32, "fp32")]:
        probe = x.to(dtype)
        reference = sum((b.to(dtype) @ (a.to(dtype) @ probe)) * scale
                        for a, b in zip(a_parts, b_parts)) / n
        concatenated = (b_out.to(dtype) @ (a_out.to(dtype) @ probe)) * (alpha / (n * rank))
        if not torch.isfinite(reference).all() or not torch.isfinite(concatenated).all():
            raise ValueError(f"nonfinite {name} effective-update probe")
        error = concatenated - reference
        norm = float(reference.norm())
        records[name] = {"max_abs_error": float(error.abs().max()),
                         "relative_l2_error": float(error.norm()) / max(norm, 1e-300)}
        if name == "fp64" and not torch.allclose(reference, concatenated, atol=1e-12, rtol=1e-10):
            raise ValueError("FP64 effective-update probe equivalence failed")
    row_ids = torch.linspace(0, b_out.shape[0] - 1, min(32, b_out.shape[0])).long().unique()
    col_ids = torch.linspace(0, a_out.shape[1] - 1, min(32, a_out.shape[1])).long().unique()
    reference = sum((b[row_ids].double() @ a[:, col_ids].double()) * scale
                    for a, b in zip(a_parts, b_parts)) / n
    concatenated = (b_out[row_ids].double() @ a_out[:, col_ids].double()) * (alpha / (n * rank))
    if not torch.allclose(reference, concatenated, atol=1e-12, rtol=1e-10):
        raise ValueError("FP64 dense-cell equivalence failed")
    records.update(probe_seed=seed, probe_columns=x.shape[1],
                   dense_cells_checked=int(reference.numel()),
                   dense_fp64_max_abs_error=float((reference - concatenated).abs().max()),
                   fp64_passed=True)
    return records


def build_average(adapters: list[Path], output_dir: Path, weights=None) -> dict:
    """Create a new exact concatenation adapter, or fail without replacing existing outputs."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    source = _inspect_sources(adapters, weights)
    if output_dir in [Path(i["path"]) for i in source["inputs"] + source["parent_chain"]]:
        raise ValueError("output cannot replace an input")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.with_name(output_dir.name + ".building-" + uuid.uuid4().hex)
    stage.mkdir()
    config = dict(source["config"])
    n, rank, alpha = len(adapters), config["r"], config["lora_alpha"]
    config.update(r=n * rank, inference_mode=True)
    layers, state = [], {}
    try:
        with ExitStack() as stack:
            readers = [stack.enter_context(safe_open(Path(item["path"]) /
                      "adapter_model.safetensors", framework="pt", device="cpu"))
                       for item in source["inputs"]]
            keys = set(readers[0].keys())
            if any(set(reader.keys()) != keys for reader in readers[1:]):
                raise ValueError("source tensor key sets differ")
            matches = [re.fullmatch(r"(.+)\.lora_([AB])\.weight", key) for key in sorted(keys)]
            if not keys or not all(matches):
                raise ValueError("only ordinary linear lora_A/lora_B.weight tensors are supported")
            modules = sorted({match.group(1) for match in matches})
            if len(keys) != 2 * len(modules):
                raise ValueError("missing A or B tensor")
            for index, module in enumerate(modules):
                ak, bk = module + ".lora_A.weight", module + ".lora_B.weight"
                if ak not in keys or bk not in keys:
                    raise ValueError(f"missing A/B pair for {module}")
                if not any(module.endswith("." + name) for name in config["target_modules"]):
                    raise ValueError(f"tensor module not declared in target_modules: {module}")
                aa = [reader.get_tensor(ak) for reader in readers]
                bb = [reader.get_tensor(bk) for reader in readers]
                if any(a.ndim != 2 or b.ndim != 2 or a.shape[0] != rank or b.shape[1] != rank
                       for a, b in zip(aa, bb)):
                    raise ValueError(f"tensor rank does not match config for {module}")
                a_out, b_out, scale, identity = concatenate_factors(aa, bb, alpha)
                numerical = _numerical_check(aa, bb, a_out, b_out, alpha, 20261023 + index)
                state[ak], state[bk] = a_out, b_out
                layers.append({"module": module, "input_rank": rank, "output_rank": n * rank,
                    "input_scaling": alpha / rank, "output_scaling": scale,
                    "a_shape": list(a_out.shape), "b_shape": list(b_out.shape),
                    "block_identity": identity, "numerical_equivalence": numerical})
        save_file(state, stage / "adapter_model.safetensors", metadata={"format": "pt"})
        _write_json(stage / "adapter_config.json", config)
        run_config = {"stage": "grpo_effective_update_average", "no_training": True,
            "sft_checkpoint": source["parent_chain"][-1]["path"],
            "base_model": config["base_model_name_or_path"], "status": "derived",
            "source_adapters": [item["path"] for item in source["inputs"]]}
        _write_json(stage / "run_config.json", run_config)
        outputs = {label: file_fingerprint(stage / name, output_dir / name) for label, name in
                   [("adapter_model", "adapter_model.safetensors"),
                    ("adapter_config", "adapter_config.json"), ("run_config", "run_config.json")]}
        # A changed source during construction invalidates the artifact before publication.
        for item in source["inputs"] + source["parent_chain"]:
            for file in item["files"]:
                verify_file(file)
        manifest = {"schema": "equal-effective-lora-update-average-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), **source,
            "outputs": outputs, "layer_count": len(layers), "layers": layers,
            "construction": {"formula": "(alpha/(n*r))*cat_cols(B_i)@cat_rows(A_i) "
                             "= mean_i((alpha/r)*B_i@A_i)",
                "source_dtype": "float32", "output_factor_dtype": "float32",
                "source_blocks_unchanged": True, "factor_averaging": False,
                "svd_or_rank_truncation": False, "source_rank": rank,
                "output_rank": n * rank, "output_alpha": alpha,
                "mathematical_equality": True,
                "floating_point_note": "FP32 GEMM and final bf16 merge round; equality is "
                "algebraic, not a claim of bit-identical dense GEMM across summation orders."}}
        manifest["binding_sha256"] = _binding(manifest)
        _write_json(stage / "average_manifest.json", manifest)
        if output_dir.exists():
            raise FileExistsError(f"output appeared during construction: {output_dir}")
        os.rename(stage, output_dir)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {**manifest, "manifest_file": file_fingerprint(output_dir / "average_manifest.json")}


def validate_average(adapter_dir: Path, adapters: list[Path], weights=None) -> dict:
    """Verify a completed average against all source and output files for safe resume."""
    adapter_dir = Path(adapter_dir).resolve()
    manifest = json.loads((adapter_dir / "average_manifest.json").read_text())
    bound = manifest.pop("binding_sha256")
    if _binding(manifest) != bound:
        raise ValueError("average manifest binding mismatch")
    manifest["binding_sha256"] = bound
    current = _inspect_sources(adapters, weights)
    for key in ("config", "weights", "inputs", "parent_chain"):
        if manifest[key] != current[key]:
            raise ValueError(f"average source {key} mismatch")
    for file in manifest["outputs"].values():
        if Path(file["path"]).parent != adapter_dir:
            raise ValueError("average output fingerprint points outside its directory")
        verify_file(file)
    return {**manifest, "manifest_file": file_fingerprint(adapter_dir / "average_manifest.json")}


def merge_average(adapter_dir: Path, merged_dir: Path) -> dict:
    """Merge common SFT then average with existing bf16 implementation; refuse retention<80%."""
    adapter_dir, merged_dir = Path(adapter_dir).resolve(), Path(merged_dir).resolve()
    manifest = json.loads((adapter_dir / "average_manifest.json").read_text())
    verified = validate_average(adapter_dir, [Path(i["path"]) for i in manifest["inputs"]])
    if merged_dir.exists():
        raise FileExistsError(f"refusing to overwrite merged output {merged_dir}")
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    original = MG.merged_dir_for
    try:
        MG.merged_dir_for = lambda adapter: merged_dir
        produced = MG.merge(adapter_dir)
    finally:
        MG.merged_dir_for = original
    retention = MG.recorded_retention(produced)
    if Path(produced).resolve() != merged_dir:
        raise ValueError("merge wrote outside its declared isolated output")
    if not retention or len(retention) != len(verified["parent_chain"]) + 1:
        raise ValueError("merge retention missing or chain mismatch")
    expected_chain = [item["path"] for item in verified["parent_chain"]] + [str(adapter_dir)]
    if [record["adapter"] for record in retention] != expected_chain:
        raise ValueError("merge retention adapter paths differ from the validated parent chain")
    passed = math.isfinite(retention[-1]["kept_fraction"]) and retention[-1]["kept_fraction"] >= .8
    result = {"schema": "equal-effective-lora-update-average-merge-v1",
        "average_manifest": verified["manifest_file"], "base_dtype": "bfloat16",
        "source_factor_dtype": "float32", "merged_dtype": "bfloat16",
        "merge_retention": retention, "minimum_retention": .8,
        "minimum_retention_pass": passed,
        "rounding": "Existing PEFT merge on common bf16 base; SFT first, average second. "
                    "No automatic dtype change.",
        "files": [file_fingerprint(p) for p in sorted(merged_dir.iterdir()) if p.is_file()]}
    result["binding_sha256"] = _binding(result)
    _write_json(merged_dir / "average_merge_manifest.json", result)
    if not passed:
        raise ValueError("average merge retention below 80%; stop; not model degradation evidence")
    return {**result, "manifest_file": file_fingerprint(merged_dir / "average_merge_manifest.json")}
