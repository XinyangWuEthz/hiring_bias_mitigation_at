"""One allocation, no training: full-bank 2x2 checkpoint425 startup diagnostic.

prepare freezes committed source and existing evidence. run executes child processes
inside its existing Slurm allocation; it never submits or requeues a job. All model
products and measurements are confined to the new experiment directory.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SCHEMA = "grpo425-startup-factorial-v1"
SOURCE_COMMIT = "b9e8a828a242abb143f74c5f6e47f8d1b0f58aa8"
SOURCE_MONITOR_SHA = "a021b6cd14f24652deb85ce7e58b6d5b7c1ac0a7a3292485ba31595074227899"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
ARMS = ("cpu_historical", "gpu_historical", "cpu_live", "gpu_live")
OUTPUT_DIRECTORY = "outputs/grpo_425_diagnostic/startup52_20261025"
RETRY_OUTPUT_DIRECTORY = "outputs/grpo_425_diagnostic/startup52_retry1_20261025"
CODE = [
    "scripts/grpo_425_diagnostic.py",
    "scripts/cscs/grpo_425_diagnostic.sbatch",
    "scripts/checkpoint_benchmark_trajectory.py",
    "scripts/operating_point_control.py",
    "src/hiring_bias_mitigation/mitigation/grpo.py",
    "src/hiring_bias_mitigation/mitigation/training_common.py",
    "src/hiring_bias_mitigation/mitigation/merge.py",
    "src/hiring_bias_mitigation/mitigation/decision_eval.py",
    "src/hiring_bias_mitigation/eval/backends.py",
    "src/hiring_bias_mitigation/eval/parsing.py",
    "src/hiring_bias_mitigation/eval/margins.py",
    "src/hiring_bias_mitigation/eval/runner.py",
    "src/hiring_bias_mitigation/analysis/reference_metrics.py",
    "src/hiring_bias_mitigation/utils/config.py",
    "scripts/cscs/env.sh",
]
ENV_KEYS = [
    "VLLM_FLOAT32_MATMUL_PRECISION",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_ALLREDUCE_USE_FLASHINFER",
    "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
    "NVIDIA_TF32_OVERRIDE",
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
]


def scalar_json(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON-compatible: {type(value)}")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
            default=scalar_json,
        ).encode()
    ).hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}


def verify(item):
    if fingerprint(item["path"]) != item:
        raise ValueError(f"fingerprint changed: {item['path']}")


def verify_snapshot_mapping(snapshot, mapping):
    for relative, blob in mapping.items():
        if str((Path(snapshot) / relative).resolve()) != blob:
            raise ValueError(f"base snapshot mapping changed: {relative}")


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def module(name):
    key = "diagnostic_" + name
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, ROOT / "scripts" / f"{name}.py")
        item = importlib.util.module_from_spec(spec)
        sys.modules[key] = item
        spec.loader.exec_module(item)
    return sys.modules[key]


def software():
    return {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "transformers",
            "vllm",
            "trl",
            "peft",
            "tokenizers",
            "safetensors",
            "pandas",
        ]
    }


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def committed(paths):
    paths = [str(Path(p).resolve().relative_to(ROOT)) for p in paths]
    for path in paths:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=ROOT,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True
    )
    if changed.strip():
        raise ValueError("commit diagnostic and imported source before freezing/executing")


def protocol():
    return {
        "arms": list(ARMS),
        "rows_per_arm": 31050,
        "training_authorized": False,
        "slurm_jobs_authorized": 1,
        "automatic_resubmission": False,
        "max_new_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "thinking": False,
        "strict_startup_gate_unchanged": True,
        "historical": {
            "chunk_size": 4000,
            "max_num_seqs": 256,
            "max_num_batched_tokens": 16384,
            "max_model_len": 8192,
            "gpu_memory_utilization": 0.8,
            "engine_seed": 42,
            "sampling_seed": None,
            "tensor_parallel_size": 1,
        },
        "live": {
            "shards": 4,
            "shard_rule": "canonical_rows[rank::4]",
            "max_num_seqs": 64,
            "max_num_batched_tokens": 4096,
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.3,
            "engine_seed": "rank",
            "sampling_seed": 42,
            "tensor_parallel_size": 1,
            "warmup": "original256_vllm_margin_only",
        },
    }


def validate_protocol(value):
    if value != protocol():
        raise ValueError("frozen two-by-two inference-only protocol differs")


def check_binding(value):
    if digest({k: v for k, v in value.items() if k != "binding_sha256"}) != value["binding_sha256"]:
        raise ValueError("manifest binding mismatch")


def unique_order(records, expected):
    ids = [r["prompt_id"] for r in records]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("missing, duplicated, or unexpected prompt identities")
    lookup = {r["prompt_id"]: r for r in records}
    return [lookup[key] for key in expected]


def matched_controls(rows, mismatch_ids):
    """One distinct agreeing control per mismatch, prefer the same pair/group/condition."""
    selected, used = [], set(mismatch_ids)
    for target in [r for r in rows if r["prompt_id"] in mismatch_ids]:
        choices = [
            r for r in rows if r["prompt_id"] not in used and r["pair_id"] == target["pair_id"]
        ]
        if not choices:
            raise ValueError("no same-pair agreeing control remains")
        choice = min(
            choices,
            key=lambda r: (
                r["condition"] != target["condition"],
                r["protected_group"] != target["protected_group"],
                r["prompt_id"],
            ),
        )
        used.add(choice["prompt_id"])
        selected.append(
            {
                "mismatch_prompt_id": target["prompt_id"],
                "control_prompt_id": choice["prompt_id"],
                "pair_id": target["pair_id"],
                "condition_matched": choice["condition"] == target["condition"],
                "protected_group_matched": choice["protected_group"] == target["protected_group"],
            }
        )
    return selected


def prepare(out, extra_code, retry_of=None):
    import pandas as pd
    from huggingface_hub import snapshot_download

    from hiring_bias_mitigation.mitigation import merge as MG
    from hiring_bias_mitigation.mitigation.decision_eval import eval_rows
    from hiring_bias_mitigation.utils.config import resolve_output_path

    if out.exists():
        raise FileExistsError("new diagnostic output directory required")
    authorized = RETRY_OUTPUT_DIRECTORY if retry_of is not None else OUTPUT_DIRECTORY
    if out != Path(resolve_output_path(authorized)).resolve():
        raise ValueError("only the authorized isolated diagnostic output directory may be written")
    retry = None
    if retry_of is not None:
        original = Path(resolve_output_path(OUTPUT_DIRECTORY)).resolve()
        if Path(resolve_output_path(str(retry_of))).resolve() != original:
            raise ValueError("retry must preserve the original diagnostic as its parent")
        parent_fingerprint = fingerprint(original / "manifest.json")
        if parent_fingerprint["sha256"] != (
            "17801b2f69082e58971faefcd7b6e2648113b76ffa9f86ab72194d86bd20f396"
        ):
            raise ValueError("original diagnostic manifest bytes changed")
        parent = read(original / "manifest.json")
        check_binding(parent)
        if parent["binding_sha256"] != (
            "02f0ee627287d26e01e3a98129dbc5a5fa74c582d8ccd004cb8dd1046f071753"
        ):
            raise ValueError("unexpected original diagnostic binding")
        validate_protocol(parent["protocol"])
        if parent["prepared_commit"] != "4b5061a6cc85e45392d2c1ae29ab28735957c2ba":
            raise ValueError("unexpected original diagnostic commit")
        if read(original / "progress.json")["status"] != "failed":
            raise ValueError("retry parent must be the preserved failed diagnostic")
        if read(original / "execution.json")["job_id"] != "5140642":
            raise ValueError("unexpected original diagnostic job")
        retry = {
            "parent_manifest": parent_fingerprint,
            "parent_progress": fingerprint(original / "progress.json"),
            "parent_execution": fingerprint(original / "execution.json"),
            "parent_binding_sha256": parent["binding_sha256"],
            "parent_job_id": "5140642",
            "reuse_old_measurements": False,
            "reconstruct_all_weights": True,
            "rerun_all_four_cells_in_one_allocation": True,
            "smoke": {
                "weights": "cpu",
                "process_group": "separate four-rank torchrun before formal cells",
                "request_rule": "first frozen canonical row in each rank::4 shard",
                "warmup": "unchanged256; two original weight pushes and cache resets",
                "required": "all four rank receipts; actual token IDs and valid decisions",
                "decision_matching_required": False,
                "included_in_benchmark_metrics": False,
                "persistent_compiler_caches_may_be_warmed": True,
            },
        }
    code = [ROOT / p for p in CODE] + [Path(p).resolve() for p in extra_code]
    if retry is not None:
        code.extend(
            ROOT / p
            for p in [
                "scripts/cscs/grpo_425_diagnostic_retry1.sbatch",
                "docs/GRPO_425_STARTUP_DIAGNOSTIC_RETRY1.md",
            ]
        )
    committed(code)
    prior = Path(
        resolve_output_path("outputs/grpo_low_lr_pilot/grpo_v3_all_lr3e6_from425_retry1_20261023")
    )
    train = Path(
        resolve_output_path(
            "outputs/grpo/qwen3.5-9b_en_only_grpo_v3_all_lr3e6_from425_75step_retry1"
        )
    )
    for source in [prior, train]:
        if out == source or out in source.parents or source in out.parents:
            raise ValueError("diagnostic outputs must be isolated from failed/source runs")
    fp = fingerprint(prior / "monitor_bank/manifest.json")
    if fp["sha256"] != SOURCE_MONITOR_SHA:
        raise ValueError("retry1 source manifest is not the frozen input")
    old = read(fp["path"])
    check_binding(old)
    source = old["source_trajectory"]
    trajectory = read(source["manifest"]["path"])
    check_binding(trajectory)
    if source["binding_sha256"] != trajectory["binding_sha256"]:
        raise ValueError("source trajectory binding differs")
    if {k: software()[k] for k in trajectory["software"]} != trajectory["software"]:
        raise ValueError("software differs from the historical benchmark")
    cached = source["cached_models"]["step425"]
    cached_result = read(Path(cached["raw"]["path"]).with_name("result.json"))
    base_snapshot = Path(snapshot_download("Qwen/Qwen3.5-9B", local_files_only=True))
    if base_snapshot.name != BASE_REVISION:
        raise ValueError("cached base revision changed")
    base_files = [fingerprint(p) for p in sorted(base_snapshot.rglob("*")) if p.is_file()]
    base_file_map = {
        str(p.relative_to(base_snapshot)): str(p.resolve())
        for p in sorted(base_snapshot.rglob("*"))
        if p.is_file()
    }
    startup = read(train / "full_benchmark_monitor/step-0425/metrics-check0001.json")
    inputs = [
        fp,
        source["manifest"],
        source["prompts"],
        old["rows_file"],
        cached["raw"],
        source["cached_models"]["SFT-v3"]["raw"],
        startup["full"]["decisions_file"],
        old["training_data"]["validation"],
        *old["tokenization"]["tokenizer_files"],
        fingerprint(train / "run_config.json"),
        fingerprint(train / "full_benchmark_monitor/step-0425/metrics-check0001.json"),
        fingerprint(Path(cached["raw"]["path"]).with_name("result.json")),
    ]
    for adapter in cached["adapter_chain"]:
        inputs.extend(adapter["files"])
        adapter_dir = Path(adapter["path"])
        run_dir = adapter_dir.parent if adapter_dir.name.startswith("checkpoint-") else adapter_dir
        parent_metadata = run_dir / "run_config.json"
        if parent_metadata.exists():
            inputs.append(fingerprint(parent_metadata))
    expected_chain = [str(Path(p["path"]).resolve()) for p in cached["adapter_chain"]]
    if [str(p.resolve()) for p in MG.adapter_chain(cached["adapter"])] != expected_chain:
        raise ValueError("actual adapter parent chain differs from frozen trajectory")
    inputs.extend(base_files)
    for item in inputs:
        verify(item)
    rows = pd.read_parquet(old["rows_file"]["path"])
    prompts = pd.read_parquet(source["prompts"]["path"])
    if len(rows) != 31050 or rows.prompt_id.duplicated().any() or rows.duplicated(KEY).any():
        raise ValueError("expected31050 unique frozen prompts")
    for column in prompts:
        if not (
            rows[column].eq(prompts[column]) | (rows[column].isna() & prompts[column].isna())
        ).all():
            raise ValueError(f"frozen prompt metadata differs: {column}")
    cached_raw = (
        pd.read_parquet(cached["raw"]["path"])
        .set_index(KEY)
        .loc[pd.MultiIndex.from_frame(rows[KEY])]
    )
    failed_raw = (
        pd.read_parquet(startup["full"]["decisions_file"]["path"])
        .set_index("prompt_id")
        .loc[rows.prompt_id]
    )
    mismatch_ids = set(
        rows.loc[cached_raw.decision.to_numpy() != failed_raw.decision.to_numpy(), "prompt_id"]
    )
    if len(mismatch_ids) != 52:
        raise ValueError("source does not reproduce the52 known disagreements")
    warmup = eval_rows(pd.read_parquet(old["training_data"]["validation"]["path"]), 256, 42)
    if len(warmup) != 256:
        raise ValueError("original synchronization warmup must have256rows")
    out.mkdir(parents=True)
    warmup.to_parquet(out / "warmup.parquet", index=False)
    controls = matched_controls(rows.to_dict("records"), mismatch_ids)
    atomic(out / "matched_controls.json", controls)
    snapshots = []
    for relative in [
        "src/hiring_bias_mitigation/mitigation/grpo.py",
        "src/hiring_bias_mitigation/mitigation/training_common.py",
        "src/hiring_bias_mitigation/mitigation/merge.py",
        "src/hiring_bias_mitigation/eval/backends.py",
    ]:
        data = subprocess.check_output(["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT)
        path = out / "retry1_source_snapshot" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        snapshots.append(fingerprint(path))
    manifest = {
        "schema": SCHEMA,
        "protocol": protocol(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_directory": str(out),
        "adapter_chain": expected_chain,
        "prepared_commit": git_head(),
        "retry1_source_commit": SOURCE_COMMIT,
        "historical_cache_commit": cached_result["provenance"]["commit"],
        "runtime_code": [fingerprint(p) for p in code],
        "retry1_code_snapshot": snapshots,
        "inputs": list({p["path"]: p for p in inputs}.values()),
        "software": software(),
        "source_monitor": fp,
        "source_trajectory": source,
        "rows": old["rows_file"],
        "failed_startup_raw": startup["full"]["decisions_file"],
        "source_training_run_config": fingerprint(train / "run_config.json"),
        "warmup": fingerprint(out / "warmup.parquet"),
        "matched_controls": fingerprint(out / "matched_controls.json"),
        "mismatch_prompt_ids": sorted(mismatch_ids),
        "base_model": "Qwen/Qwen3.5-9B",
        "base_revision": BASE_REVISION,
        "sft_directory": cached["adapter_chain"][0]["path"],
        "base_snapshot": str(base_snapshot),
        "base_files": base_files,
        "base_file_map": base_file_map,
        "checkpoint425": cached["adapter"],
        "inference_only_authorized": True,
        "limitations": [
            "Previously exposed benchmark; no independent performance claim.",
            "Historical CUDA/TF32 environment was not recorded; current live merge follows "
            "installed vLLM environment/default highest.",
            "Live engine reproduces vLLM-only synchronization warmup, "
            "not HF probe forwards or optimizer-resident memory.",
            "Engine-context contrast includes batching, cache/warmup, scheduling "
            "and explicit greedy-seed differences.",
            "No failed startup output becomes a replacement reference "
            "and no strict gate is relaxed.",
        ],
    }
    if retry is not None:
        if (
            manifest["rows"] != parent["rows"]
            or manifest["failed_startup_raw"] != parent["failed_startup_raw"]
            or manifest["source_trajectory"] != parent["source_trajectory"]
            or manifest["mismatch_prompt_ids"] != parent["mismatch_prompt_ids"]
            or controls != read(parent["matched_controls"]["path"])
            or manifest["warmup"]["sha256"] != parent["warmup"]["sha256"]
        ):
            raise ValueError("retry changes original rows, targets, controls, or warmup")
        manifest["retry"] = retry
        manifest["inputs"].extend(
            retry[key] for key in ["parent_manifest", "parent_progress", "parent_execution"]
        )
        manifest["limitations"].append(
            "A separate disposable four-rank smoke process precedes all four fresh engines; "
            "persistent compiler caches may be warmed. No smoke request enters a formal engine."
        )
    manifest["binding_sha256"] = digest(manifest)
    atomic(out / "manifest.json", manifest)
    atomic(out / "progress.json", {"status": "prepared", "training_authorized": False})
    print(
        json.dumps(
            {
                "manifest": fingerprint(out / "manifest.json"),
                "binding_sha256": manifest["binding_sha256"],
            }
        )
    )


def load_manifest(out, full_verify=False):
    m = read(out / "manifest.json")
    check_binding(m)
    if out != Path(m["output_directory"]).resolve():
        raise ValueError("manifest belongs to a different output directory")
    verify_snapshot_mapping(m["base_snapshot"], m["base_file_map"])
    validate_protocol(m["protocol"])
    if m["schema"] != SCHEMA or git_head() != m["prepared_commit"]:
        raise ValueError("diagnostic source commit changed")
    if software() != m["software"]:
        raise ValueError("diagnostic software changed")
    committed([p["path"] for p in m["runtime_code"]])
    for item in m["runtime_code"]:
        verify(item)
    if full_verify:
        for item in [
            *m["inputs"],
            *m["retry1_code_snapshot"],
            m["warmup"],
            m["matched_controls"],
        ]:
            verify(item)
    return m


def runtime_info():
    import torch

    gpu = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .splitlines()
        if torch.cuda.is_available()
        else []
    )
    return {
        "software": software(),
        "python": sys.version,
        "gpu_inventory": gpu,
        "environment": {k: os.environ.get(k) for k in ENV_KEYS},
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_cuda": torch.version.cuda,
        "job_id": os.getenv("SLURM_JOB_ID"),
    }


def config_snapshot(value, _active=None):
    """Read declared settings without copying or inspecting runtime tensors/modules."""
    kind = type(value)
    type_name = f"{kind.__module__}.{kind.__qualname__}"
    if value is None or kind in (str, bool, int):
        return value
    if kind is float:
        return value if math.isfinite(value) else {"_type": type_name, "_value": str(value)}
    if isinstance(value, Enum):
        return config_snapshot(value.value, _active)
    if isinstance(value, Path) or (
        kind.__module__ == "torch" and kind.__name__ in {"dtype", "device"}
    ):
        return str(value)
    if any(
        (base.__module__, base.__name__)
        in {("torch", "Tensor"), ("torch.nn.modules.module", "Module")}
        for base in kind.__mro__
    ):
        return {"_type": type_name, "_omitted": "runtime tensor or module"}

    active = set() if _active is None else _active
    if id(value) in active:
        return {"_type": type_name, "_omitted": "cycle"}
    active.add(id(value))
    try:

        def field_snapshot(name, item):
            if type(name) is str and name == "static_forward_context":
                result = {
                    "_type": f"{type(item).__module__}.{type(item).__qualname__}",
                    "_omitted": "runtime layer registry; not traversed",
                }
                if type(item) is dict:
                    result["count"] = len(item)
                return result
            return config_snapshot(item, active)

        if kind is dict:
            if all(type(key) is str for key in value):
                return {key: field_snapshot(key, item) for key, item in value.items()}
            return {
                "_type": type_name,
                "entries": [
                    [config_snapshot(key, active), field_snapshot(key, item)]
                    for key, item in value.items()
                ],
            }
        if kind in (list, tuple):
            return [config_snapshot(item, active) for item in value]
        if kind in (set, frozenset):
            return sorted((config_snapshot(item, active) for item in value), key=json.dumps)
        if dataclasses.is_dataclass(kind):
            return {
                field.name: field_snapshot(field.name, getattr(value, field.name))
                for field in dataclasses.fields(kind)
            }
        if any(
            base.__module__ == "pydantic.main" and base.__name__ == "BaseModel"
            for base in kind.__mro__
        ):
            return {name: field_snapshot(name, getattr(value, name)) for name in kind.model_fields}
        return {"_type": type_name, "_omitted": "runtime object"}
    finally:
        active.remove(id(value))


def engine_info(llm):
    config = llm.llm_engine.vllm_config
    sections = {}
    for name in [
        "model_config",
        "scheduler_config",
        "cache_config",
        "parallel_config",
        "compilation_config",
    ]:
        sections[name] = config_snapshot(getattr(config, name, None))
    return sections


def bound_result(path, m):
    if not path.exists():
        return None
    value = read(path)
    if value.get("binding_sha256") != m["binding_sha256"] or value.get("status") != "complete":
        raise ValueError("incomplete or foreign cached phase result")
    for item in value.get("files", []):
        verify(item)
    return value


def token_check(out, m):
    import pandas as pd
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    from hiring_bias_mitigation.eval.backends import _chat_wrap

    path = out / "tokenization/result.json"
    if bound_result(path, m):
        return
    current_base = Path(snapshot_download(m["base_model"], local_files_only=True))
    if current_base.name != m["base_revision"]:
        raise ValueError("current cachedbase does not resolve to the frozen revision")
    rows = pd.read_parquet(m["rows"]["path"])
    historical = AutoTokenizer.from_pretrained(
        m["base_model"], trust_remote_code=True, local_files_only=True
    )
    live = AutoTokenizer.from_pretrained(
        m["sft_directory"], trust_remote_code=True, local_files_only=True
    )
    if live.pad_token is None:
        live.pad_token = live.eos_token
    records, differences = [], []
    for row in rows.to_dict("records"):
        a, b = [
            _chat_wrap(tok, row["prompt"], {"enable_thinking": False}) for tok in [historical, live]
        ]
        ia, ib = historical.encode(a), live.encode(b)
        if a != b or ia != ib:
            differences.append(
                {
                    "prompt_id": row["prompt_id"],
                    "wrapped_text_equal": a == b,
                    "historical_ids": ia,
                    "live_ids": ib,
                }
            )
        records.append({"prompt_id": row["prompt_id"], "wrapped_prompt": a, "input_ids": ia})
    folder = path.parent
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(folder / "inputs.parquet", index=False)
    atomic(folder / "differences.json", differences)
    result = {
        "binding_sha256": m["binding_sha256"],
        "status": "complete" if not differences else "failed",
        "rows": len(records),
        "mismatched_rows": len(differences),
        "max_input_tokens": max(len(r["input_ids"]) for r in records),
        "token_ids_sha256": digest([r["input_ids"] for r in records]),
        "wrapped_text_sha256": digest([r["wrapped_prompt"] for r in records]),
        "base_snapshot": str(current_base),
        "files": [fingerprint(folder / "inputs.parquet"), fingerprint(folder / "differences.json")],
        "tokenizer_sources": [
            fingerprint(d / name)
            for d in [current_base, Path(m["sft_directory"])]
            for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]
        ],
    }
    if result["max_input_tokens"] + 512 > 4096:
        result["status"] = "failed"
    atomic(path, result)
    if differences or result["max_input_tokens"] + 512 > 4096:
        raise ValueError("token identity/capacity gate failed; no model inference permitted")


def tensor_sha(tensor):
    import torch

    tensor = tensor.detach().contiguous().cpu()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_inventory(directory):
    from safetensors import safe_open

    found = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as stream:
            for name in stream.keys():  # noqa: SIM118 -- safe_open is not a dictionary
                if name in found:
                    raise ValueError("duplicate tensor key across saved shards")
                found[name] = path
    if not found:
        raise ValueError("no model tensors")
    return found


def read_tensor(path, name):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as stream:
        return stream.get_tensor(name)


def weight_result(out, kind, m):
    path = out / "weights" / kind / "result.json"
    result = bound_result(path, m)
    if result is None:
        raise ValueError(f"missing completed{kind}weight artifact")
    return result


def make_weights(out, m, kind):
    import torch

    from hiring_bias_mitigation.mitigation import merge as MG

    if [str(p.resolve()) for p in MG.adapter_chain(m["checkpoint425"])] != m["adapter_chain"]:
        raise ValueError("adapter parent chain changed before merge")

    folder = out / "weights" / kind
    if bound_result(folder / "result.json", m):
        return
    if folder.exists():
        raise ValueError("partial weight artifact has no success receipt")
    folder.mkdir(parents=True)
    model_dir = folder / "model"
    if kind == "cpu":
        MG.merged_dir_for = lambda adapter: model_dir
        MG.merge(m["checkpoint425"])
        retention = MG.recorded_retention(model_dir)
        if not retention or retention[-1]["kept_fraction"] < 0.8:
            raise ValueError("canonical merge fails original80%retention requirement")
        extra = {
            "merge_retention": retention,
            "construction": "unchanged merge.merge CPU SFT then425",
        }
    else:
        from peft import get_peft_model
        from safetensors.torch import save_file
        from vllm import envs

        from hiring_bias_mitigation.mitigation import grpo
        from hiring_bias_mitigation.mitigation import training_common as C

        cpu = weight_result(out, "cpu", m)
        cpu_model = Path(cpu["model_directory"])
        # This is the installed vLLM Worker.__init__ operation, which precedes live sync.
        before = runtime_info()
        torch.set_float32_matmul_precision(envs.VLLM_FLOAT32_MATMUL_PRECISION)
        cfg = {
            "base_model": m["sft_directory"],
            "dtype": "bfloat16",
            "use_lora": True,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.0,
        }
        frozen_adapter = read(Path(m["checkpoint425"]) / "adapter_config.json")
        actual_config = C.peft_config(cfg).to_dict()
        for field in ["r", "lora_alpha", "lora_dropout", "bias", "task_type", "target_modules"]:
            actual, wanted = actual_config[field], frozen_adapter[field]
            if field == "target_modules":
                actual, wanted = sorted(actual), sorted(wanted)
            if actual != wanted:
                raise ValueError(f"fresh adapter differs from frozen425 configuration: {field}")
        base, _ = C.load_model_and_tokenizer(cfg)
        if any(p.device.type != "cpu" for p in base.parameters()):
            raise ValueError("SFT parent merge did not happen onCPU")
        policy = get_peft_model(base, C.peft_config(cfg)).to("cuda")
        load_result = policy.load_adapter(m["checkpoint425"], "default", is_trainable=True)
        missing_adapter = [key for key in load_result.missing_keys if "lora_" in key]
        if missing_adapter or load_result.unexpected_keys:
            raise ValueError(
                f"checkpoint425 adapter load mismatch: {missing_adapter}, "
                f"{load_result.unexpected_keys}"
            )
        policy.eval()
        factor_dtypes = sorted({str(p.dtype) for n, p in policy.named_parameters() if "lora_" in n})
        if factor_dtypes != ["torch.float32"]:
            raise ValueError("live adapter factors are not originalFP32")
        with torch.no_grad():
            tensors = {
                name: value.detach().cpu().contiguous().clone()
                for name, value in grpo.merged_named_params(policy, multimodal=True)
            }
        names = sorted(tensors)
        inventory = tensor_inventory(cpu_model)
        unknown = set(tensors) - set(inventory)
        if unknown:
            raise ValueError(
                f"live exported names do not map to canonical checkpoint: {sorted(unknown)[:8]}"
            )
        # Unchanged tensors outside the text policy, e.g. vision, remain from the same base.
        untouched = sorted(set(inventory) - set(tensors))
        for name in untouched:
            tensors[name] = read_tensor(inventory[name], name)
        model_dir.mkdir()
        save_file(tensors, model_dir / "model.safetensors", metadata={"format": "pt"})
        for path in cpu_model.iterdir():
            if (
                path.is_file()
                and path.suffix != ".safetensors"
                and path.name not in ["model.safetensors.index.json", "merged_from.json"]
            ):
                shutil.copy2(path, model_dir / path.name)
        total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
        atomic(
            model_dir / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_bytes},
                "weight_map": {name: "model.safetensors" for name in sorted(tensors)},
            },
        )
        extra = {
            "construction": "C CPU SFTmerge→freshPEFT→GPU→load425→grpo.merged_named_params",
            "factor_dtypes": factor_dtypes,
            "serving_tensor_names": names,
            "adapter_configuration": frozen_adapter,
            "adapter_load_missing_lora_keys": missing_adapter,
            "adapter_load_unexpected_keys": list(load_result.unexpected_keys),
            "copied_nonpolicy_names": untouched,
            "runtime_before_vllm_precision": before,
            "vllm_float32_matmul_precision": envs.VLLM_FLOAT32_MATMUL_PRECISION,
        }
        del policy, base, tensors
    files = [fingerprint(p) for p in sorted(model_dir.iterdir()) if p.is_file()]
    inventory = tensor_inventory(model_dir)
    tensor_records = []
    for name, path in sorted(inventory.items()):
        t = read_tensor(path, name)
        tensor_records.append(
            {"name": name, "shape": list(t.shape), "dtype": str(t.dtype), "sha256": tensor_sha(t)}
        )
    atomic(folder / "tensor_hashes.json", tensor_records)
    atomic(
        folder / "result.json",
        {
            "binding_sha256": m["binding_sha256"],
            "status": "complete",
            "kind": kind,
            "model_directory": str(model_dir),
            "files": [*files, fingerprint(folder / "tensor_hashes.json")],
            "tensor_hashes": fingerprint(folder / "tensor_hashes.json"),
            "runtime": runtime_info(),
            **extra,
        },
    )


def compare_weights(out, m):
    import torch

    cpu, gpu = [weight_result(out, kind, m) for kind in ["cpu", "gpu"]]
    ai, bi = [tensor_inventory(Path(r["model_directory"])) for r in [cpu, gpu]]
    if set(ai) != set(bi):
        raise ValueError("CPU/GPUweight inventories differ")
    rows = []
    for name in sorted(ai):
        a, b = read_tensor(ai[name], name), read_tensor(bi[name], name)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"CPU/GPUweight shape/dtype mismatch:{name}")
        same = torch.equal(a, b)
        diff = a.float() - b.float()
        rows.append(
            {
                "name": name,
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "cpu_sha256": tensor_sha(a),
                "gpu_sha256": tensor_sha(b),
                "equal": same,
                "different_elements": int((a != b).sum()),
                "elements": a.numel(),
                "max_abs_difference": float(diff.abs().max()),
                "sum_squared_difference": float(diff.double().square().sum()),
                "text_policy_tensor": name in gpu["serving_tensor_names"],
            }
        )
    historical = read(
        Path(m["source_trajectory"]["cached_models"]["step425"]["raw"]["path"]).with_name(
            "result.json"
        )
    )
    old_files = {Path(p["path"]).name: p for p in historical["provenance"]["merged_weight_files"]}
    new_files = {Path(p["path"]).name: p for p in cpu["files"]}
    historic_bytes = all(
        name in new_files and item["sha256"] == new_files[name]["sha256"]
        for name, item in old_files.items()
    )
    atomic(
        out / "weights/comparison.json",
        {
            "binding_sha256": m["binding_sha256"],
            "status": "complete",
            "tensors": rows,
            "all_tensors_equal": all(r["equal"] for r in rows),
            "different_tensor_count": sum(not r["equal"] for r in rows),
            "canonical_historical_weight_file_bytes_match": historic_bytes,
            "note": "File-byte mismatch alone may include serialization changes; "
            "per-tensor comparison concerns the two newly constructed artifacts.",
        },
    )


def check_request_outputs(outputs, expected):
    """Check the actual vLLM input, not only an independent tokenizer encode."""
    if len(outputs) != len(expected):
        raise ValueError("vLLM returned a different number of requests")
    records, differences = [], []
    for result, row in zip(outputs, expected, strict=True):
        token_ids = list(result.prompt_token_ids) if result.prompt_token_ids is not None else None
        text_equal = result.prompt == row["wrapped_prompt"]
        ids_equal = token_ids == list(row["input_ids"])
        if not text_equal or not ids_equal:
            differences.append(
                {
                    "prompt_id": row["prompt_id"],
                    "text_equal": text_equal,
                    "token_ids_equal": ids_equal,
                    "actual_token_ids": token_ids,
                }
            )
        if len(result.outputs) != 1:
            raise ValueError("expected exactly one greedy completion")
        records.append(
            {
                "prompt_id": row["prompt_id"],
                "raw_output": result.outputs[0].text,
                "actual_input_ids": token_ids,
                "generated_token_ids": list(result.outputs[0].token_ids),
                "finish_reason": result.outputs[0].finish_reason,
            }
        )
    return records, differences


def token_rows(out, m):
    import pandas as pd

    result = bound_result(out / "tokenization/result.json", m)
    if not result:
        raise ValueError("tokenization gate has no success receipt")
    return pd.read_parquet(out / "tokenization/inputs.parquet").to_dict("records")


def write_arm(out, m, arm, records, evidence):
    import pandas as pd

    from hiring_bias_mitigation.eval.parsing import parse_frame

    rows = pd.read_parquet(m["rows"]["path"])
    records = unique_order(records, rows.prompt_id.tolist())
    raw = parse_frame(
        rows.drop(columns="prompt").assign(raw_output=[r["raw_output"] for r in records])
    )
    metrics = module("checkpoint_benchmark_trajectory").metrics(raw)
    folder = out / "arms" / arm
    folder.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(folder / "raw.parquet", index=False)
    pd.DataFrame(records).to_parquet(folder / "request_outputs.parquet", index=False)
    result = {
        "binding_sha256": m["binding_sha256"],
        "arm": arm,
        "status": "complete" if metrics["invalid_n"] == 0 and len(raw) == 31050 else "failed",
        "metrics": metrics,
        "raw": fingerprint(folder / "raw.parquet"),
        "actual_input_ids_ordered_sha256": digest([r["actual_input_ids"] for r in records]),
        "files": [
            fingerprint(folder / "raw.parquet"),
            fingerprint(folder / "request_outputs.parquet"),
        ],
        **evidence,
    }
    atomic(folder / "result.json", result)
    if result["status"] != "complete":
        raise ValueError("arm failed complete-bank/zero-invalid validation")


def score_historical(out, m, kind):
    import pandas as pd

    from hiring_bias_mitigation.eval.backends import _chat_wrap, build_backend

    arm = kind + "_historical"
    folder = out / "arms" / arm
    if bound_result(folder / "result.json", m):
        return
    weights = weight_result(out, kind, m)
    rows = pd.read_parquet(m["rows"]["path"])
    expected = token_rows(out, m)
    cfg = {
        **m["source_trajectory"]["audit_config"],
        "model": weights["model_directory"],
        "tensor_parallel_size": 1,
        "max_num_seqs": 256,
        "engine_kwargs": {"gdn_prefill_backend": "triton", "max_num_batched_tokens": 16384},
    }
    cfg.pop("mitigation", None)
    backend = build_backend(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, chunks = [], []
    try:
        backend._load()
        engine = engine_info(backend._llm)
        atomic(
            folder / "engine.json",
            {
                "requested_config": cfg,
                "effective": engine,
                "runtime": runtime_info(),
                "sampling_seed": None,
                "weights": fingerprint(out / "weights" / kind / "result.json"),
            },
        )
        for start in range(0, len(rows), 4000):
            stop = min(start + 4000, len(rows))
            raw_prompts = rows.prompt.tolist()[start:stop]
            wrapped = [
                _chat_wrap(backend._tokenizer, p, backend.chat_template_kwargs) for p in raw_prompts
            ]
            if wrapped != [r["wrapped_prompt"] for r in expected[start:stop]]:
                raise ValueError("historical engine tokenizer changed wrapped input")
            # Same backend constructor, wrapper and SamplingParams as backend.generate;
            # retain RequestOutput so actual input IDs can be independently checked.
            params = backend._sampling_params(raw_prompts, n=1)
            requests = backend._llm.generate(wrapped, params)
            batch, differences = check_request_outputs(requests, expected[start:stop])
            receipt = {
                "binding_sha256": m["binding_sha256"],
                "arm": arm,
                "start": start,
                "stop": stop,
                "weight_result": fingerprint(out / "weights" / kind / "result.json"),
                "sampling_params": repr(params),
                "records": batch,
                "input_differences": differences,
            }
            path = folder / f"chunk-{start:05d}.json"
            atomic(path, receipt)
            chunks.append(fingerprint(path))
            if differences:
                raise ValueError("actual vLLM historical input differs from frozen token IDs")
            records.extend(batch)
            atomic(
                folder / "progress.json",
                {"status": "generating", "rows_done": stop, "rows_total": 31050},
            )
        write_arm(
            out,
            m,
            arm,
            records,
            {
                "engine_receipt": fingerprint(folder / "engine.json"),
                "chunks": chunks,
                "wall_seconds": time.monotonic() - started,
                "weights_receipt": fingerprint(out / "weights" / kind / "result.json"),
            },
        )
    finally:
        backend.close()


def push_weights(llm, model_dir, names):
    """Mirror TRL's per-tensor load_weights path, preserving its supplied name set."""
    inventory = tensor_inventory(Path(model_dir))
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    loaded, observations = set(), []
    for name in names:
        value = read_tensor(inventory[name], name).to("cuda")
        result = model.load_weights([(name, value)])
        if not result:
            raise ValueError(f"live loader did not acknowledge source tensor: {name}")
        observations.append(
            {"source_name": name, "loader_return": sorted(result) if result is not None else None}
        )
        if result is not None:
            loaded.update(result)
        del value
    cache_reset = llm.reset_prefix_cache()
    if cache_reset is not True:
        raise RuntimeError("live prefix-cache reset did not acknowledge success")
    return {
        "source_names": names,
        "source_names_sha256": digest(names),
        "loaded_names": sorted(loaded),
        "per_tensor_loader_returns": observations,
        "reset_prefix_cache": cache_reset,
        "name_note": "Loader returns may be packed internal names; "
        "no HF-to-packed byte equality is asserted.",
    }


def served_parameters(llm):
    """Comparable within live arms; these packed names are not HF parameter names."""
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    return [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha(value),
        }
        for name, value in model.named_parameters()
    ]


def score_live(out, m, kind, smoke=False):
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from hiring_bias_mitigation.eval.backends import _chat_wrap
    from hiring_bias_mitigation.eval.margins import PREFIX, WORDS, decision_token_ids

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 4:
        raise ValueError("live engine requires the frozen four-rank context")
    arm = kind + "_live"
    if smoke and (kind != "cpu" or "retry" not in m):
        raise ValueError("smoke is authorized only for the repaired retry and CPU weights")
    folder = (out / "smoke" / "live" if smoke else out / "arms" / arm) / f"rank-{rank}"
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("live rank already has evidence; do not overwrite or repeat inference")
    folder.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    started = time.monotonic()

    def stage(name):
        atomic(
            folder / "stage.json",
            {
                "stage": name,
                "rank": rank,
                "smoke": smoke,
                "binding_sha256": m["binding_sha256"],
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    stage("process_group_ready")
    weights = weight_result(out, kind, m)
    live_weights = weight_result(out, "gpu", m)
    names = live_weights["serving_tensor_names"]
    expected = token_rows(out, m)[rank::4]
    if smoke:
        expected = expected[:1]
    tokenizer = AutoTokenizer.from_pretrained(
        m["sft_directory"], trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args = {
        "model": m["base_model"],
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.3,
        "max_model_len": 4096,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 4096,
        "distributed_executor_backend": "external_launcher",
        "seed": rank,
        "logprobs_mode": "processed_logprobs",
        "model_impl": "vllm",
        "gdn_prefill_backend": "triton",
        "language_model_only": True,
        "trust_remote_code": False,
        "dtype": "auto",
        "enable_sleep_mode": False,
        "quantization": None,
    }
    stage("constructing_engine")
    llm = LLM(**args)
    atomic(
        folder / "engine.json",
        {"requested": args, "effective": engine_info(llm), "runtime": runtime_info()},
    )
    stage("first_weight_push")
    before = push_weights(llm, weights["model_directory"], names)
    atomic(folder / "weight_push1.json", before)
    warmup = pd.read_parquet(m["warmup"]["path"])
    warm_texts = [
        _chat_wrap(tokenizer, p, {"enable_thinking": False}) + PREFIX for p in warmup.prompt
    ]
    ids = decision_token_ids(tokenizer, WORDS["en"])
    warm_params = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=list(ids))
    stage("warmup")
    warm = llm.generate(warm_texts, warm_params, use_tqdm=False)
    warm_margins = [
        o.outputs[0].logprobs[0][ids[0]].logprob - o.outputs[0].logprobs[0][ids[1]].logprob
        for o in warm
    ]
    atomic(
        folder / "warmup.json",
        {
            "rows": len(warm),
            "sampling_params": repr(warm_params),
            "text_sha256": digest(warm_texts),
            "margins": warm_margins,
            "limitation": "vLLM margin warmup only; no HF reference forward or optimizer memory.",
        },
    )
    stage("second_weight_push")
    after = push_weights(llm, weights["model_directory"], names)
    atomic(folder / "weight_push2.json", after)
    stage("served_parameter_receipt")
    atomic(
        folder / "served_tensor_hashes.json",
        {
            "parameters": served_parameters(llm),
            "name_space": "vLLM internal packed named_parameters; duplicate aliases omitted",
        },
    )
    rows = pd.read_parquet(m["rows"]["path"]).iloc[rank::4]
    if smoke:
        rows = rows.iloc[:1]
    wrapped = [_chat_wrap(tokenizer, p, {"enable_thinking": False}) for p in rows.prompt]
    if wrapped != [r["wrapped_prompt"] for r in expected]:
        raise ValueError("live wrapping differs from frozen input")
    params = SamplingParams(max_tokens=512, temperature=0.0, top_p=1.0, n=1, seed=42)
    stage("generating_frozen_requests")
    requests = llm.generate(wrapped, params, use_tqdm=False)
    records, differences = check_request_outputs(requests, expected)
    pd.DataFrame(records).to_parquet(folder / "request_outputs.parquet", index=False)
    atomic(folder / "input_differences.json", differences)
    stage("generation_saved")
    receipt = {
        "binding_sha256": m["binding_sha256"],
        "arm": arm,
        "rank": rank,
        "smoke": smoke,
        "status": "failed" if differences else "complete",
        "rows": len(records),
        "sampling_params": repr(params),
        "wall_seconds": time.monotonic() - started,
        "weights_receipt": fingerprint(out / "weights" / kind / "result.json"),
        "actual_input_ids_ordered_sha256": digest([r["actual_input_ids"] for r in records]),
        "files": [fingerprint(p) for p in sorted(folder.iterdir()) if p.is_file()],
    }
    atomic(folder / "result.json", receipt)
    if differences:
        raise ValueError("actual vLLM live input differs from frozen token IDs")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def collect_smoke(out, m):
    """Validate disposable GPU smoke output without promoting it to benchmark evidence."""
    import pandas as pd

    from hiring_bias_mitigation.eval.parsing import parse_frame

    if "retry" not in m:
        raise ValueError("smoke requires the explicit repaired retry manifest")
    expected = token_rows(out, m)
    rows = pd.read_parquet(m["rows"]["path"])
    receipts = []
    for rank in range(4):
        path = out / "smoke/live" / f"rank-{rank}" / "result.json"
        result = bound_result(path, m)
        if not result or result.get("rank") != rank or result.get("smoke") is not True:
            raise ValueError("missing matching completed smoke rank")
        records = pd.read_parquet(path.parent / "request_outputs.parquet").to_dict("records")
        if len(records) != 1 or records[0]["prompt_id"] != expected[rank]["prompt_id"]:
            raise ValueError("smoke must use exactly the predeclared first row per rank")
        parsed = parse_frame(rows.iloc[[rank]].assign(raw_output=[records[0]["raw_output"]]))
        if parsed.iloc[0].decision not in {"hire", "reject"}:
            raise ValueError("smoke produced an invalid decision")
        receipts.append(fingerprint(path))
    atomic(
        out / "smoke/result.json",
        {
            "status": "complete",
            "binding_sha256": m["binding_sha256"],
            "rows": 4,
            "training_authorized": False,
            "included_in_benchmark_metrics": False,
            "files": receipts,
        },
    )


def collect_live(out, m, kind):
    import pandas as pd

    arm = kind + "_live"
    folder = out / "arms" / arm
    records, receipts = [], []
    for rank in range(4):
        path = folder / f"rank-{rank}" / "result.json"
        receipt = bound_result(path, m)
        if receipt is None or receipt["rank"] != rank or receipt["arm"] != arm:
            raise ValueError("live arm is missing a matching completed rank")
        records.extend(pd.read_parquet(path.parent / "request_outputs.parquet").to_dict("records"))
        receipts.append(fingerprint(path))
    write_arm(
        out,
        m,
        arm,
        records,
        {
            "rank_receipts": receipts,
            "weights_receipt": fingerprint(out / "weights" / kind / "result.json"),
        },
    )


def comparison(a, b, labels):
    if len(a) != len(b) or len(a) != len(labels):
        raise ValueError("comparison coverage differs")
    counts = {label: {"hire_to_reject": 0, "reject_to_hire": 0} for label in ["hire", "reject"]}
    for old, new, label in zip(a, b, labels, strict=True):
        if old not in ["hire", "reject"] or new not in ["hire", "reject"] or label not in counts:
            raise ValueError("comparison has invalid decision/reference")
        if old != new:
            counts[label][f"{old}_to_{new}"] += 1
    return {
        "rows": len(a),
        "decision_differences": sum(x != y for x, y in zip(a, b, strict=True)),
        "by_reference": counts,
    }


def summarize(out, m):
    import pandas as pd

    rows = pd.read_parquet(m["rows"]["path"])
    frames, receipts = {}, {}
    for arm in ARMS:
        receipt = bound_result(out / "arms" / arm / "result.json", m)
        if receipt is None or receipt["arm"] != arm or receipt["metrics"]["n"] != 31050:
            raise ValueError("all four complete full-bank arms are required")
        raw = pd.read_parquet(receipt["raw"]["path"])
        frames[arm] = pd.DataFrame(unique_order(raw.to_dict("records"), rows.prompt_id.tolist()))
        receipts[arm] = receipt
    cached = pd.read_parquet(m["source_trajectory"]["cached_models"]["step425"]["raw"]["path"])
    cached = cached.set_index(KEY).loc[pd.MultiIndex.from_frame(rows[KEY])].reset_index()
    failed = (
        pd.read_parquet(m["failed_startup_raw"]["path"])
        .set_index("prompt_id")
        .loc[rows.prompt_id]
        .reset_index()
    )
    frames.update(historical_cache=cached, failed_startup=failed)
    contrasts = [
        ("historical_cache", "cpu_historical"),
        ("failed_startup", "gpu_live"),
        ("cpu_historical", "gpu_historical"),
        ("cpu_live", "gpu_live"),
        ("cpu_historical", "cpu_live"),
        ("gpu_historical", "gpu_live"),
    ]
    comparisons = {}
    for a, b in contrasts:
        metric = comparison(
            frames[a].decision.tolist(),
            frames[b].decision.tolist(),
            rows.reference_decision.tolist(),
        )
        metric["raw_output_differences"] = int(
            (frames[a].raw_output.to_numpy() != frames[b].raw_output.to_numpy()).sum()
        )
        metric["different_prompt_ids"] = rows.loc[
            frames[a].decision.to_numpy() != frames[b].decision.to_numpy(), "prompt_id"
        ].tolist()
        comparisons[f"{a}__to__{b}"] = metric
    detail = rows.drop(columns="prompt").copy()
    mismatch = set(m["mismatch_prompt_ids"])
    controls = {r["control_prompt_id"] for r in read(m["matched_controls"]["path"])}
    detail["original_mismatch"] = detail.prompt_id.isin(mismatch)
    detail["matched_control"] = detail.prompt_id.isin(controls)
    for name, frame in frames.items():
        detail[name + "_decision"] = frame.decision.to_numpy()
        detail[name + "_raw_output"] = frame.raw_output.to_numpy()
    detail[detail.original_mismatch | detail.matched_control].to_csv(
        out / "mismatch52_and_controls.csv", index=False
    )
    weights = read(out / "weights/comparison.json")
    served_comparisons = []
    for rank in range(4):
        cpu_served, gpu_served = [
            read(out / "arms" / (kind + "_live") / f"rank-{rank}" / "served_tensor_hashes.json")[
                "parameters"
            ]
            for kind in ["cpu", "gpu"]
        ]
        left, right = [
            {item["name"]: item for item in records} for records in [cpu_served, gpu_served]
        ]
        if left.keys() != right.keys():
            raise ValueError("live packed parameter name coverage changed between weight arms")
        served_comparisons.append(
            {
                "rank": rank,
                "parameter_count": len(left),
                "different_parameters": [name for name in left if left[name] != right[name]],
            }
        )
    replay_exact = comparisons["historical_cache__to__cpu_historical"]["decision_differences"] == 0
    result = {
        "schema": SCHEMA,
        "binding_sha256": m["binding_sha256"],
        "status": "complete",
        "training_authorized": False,
        "rows_per_arm": 31050,
        "all_four_arms_completed": True,
        "metrics": {arm: receipts[arm]["metrics"] for arm in ARMS},
        "comparisons": comparisons,
        "weight_comparison": fingerprint(out / "weights/comparison.json"),
        "all_constructed_tensors_equal": weights["all_tensors_equal"],
        "live_served_parameter_comparisons": served_comparisons,
        "canonical_historical_decisions_reproduced": replay_exact,
        "historical_context_nonreproduction": not replay_exact,
        "specific_numeric_or_kernel_cause_proven": False,
        "interpretation": "Same-weight contrasts isolate the frozen engine-context bundle; "
        "same-context contrasts isolate exported weight artifacts. A failed historical replay "
        "leaves additional context/replay differences unresolved.",
        "strict_gate_unchanged": True,
        "limitations": m["limitations"],
        "files": [
            fingerprint(out / "mismatch52_and_controls.csv"),
            *[fingerprint(out / "arms" / arm / "result.json") for arm in ARMS],
        ],
    }
    atomic(out / "result.json", result)


def run(out, m):
    import fcntl

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise ValueError("run requires an existing Slurm allocation; it never submits one")
    lock = (out / ".run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    execution = out / "execution.json"
    if execution.exists() and read(execution)["job_id"] != job_id:
        raise ValueError("another allocation already claimed this one-job experiment")
    atomic(
        execution,
        {
            "binding_sha256": m["binding_sha256"],
            "job_id": job_id,
            "training_authorized": False,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    script = str(Path(__file__).resolve())
    common = ["--output-dir", str(out)]
    phases = [
        ("tokens", "cpu", ["token-check"]),
        ("weights_cpu", "cpu", ["make-weights", "--kind", "cpu"]),
        ("weights_gpu", "gpu", ["make-weights", "--kind", "gpu"]),
        ("compare_weights", "cpu", ["compare-weights"]),
        ("cpu_historical", "gpu", ["score-historical", "--kind", "cpu"]),
        ("gpu_historical", "gpu", ["score-historical", "--kind", "gpu"]),
        ("cpu_live", "all", ["score-live", "--kind", "cpu"]),
        ("collect_cpu_live", "cpu", ["collect-live", "--kind", "cpu"]),
        ("gpu_live", "all", ["score-live", "--kind", "gpu"]),
        ("collect_gpu_live", "cpu", ["collect-live", "--kind", "gpu"]),
        ("summarize", "cpu", ["summarize"]),
    ]
    if "retry" in m:
        phases[4:4] = [
            ("live_smoke", "all", ["smoke-live", "--kind", "cpu"]),
            ("collect_live_smoke", "cpu", ["collect-smoke"]),
        ]
    completed = []
    for name, devices, arguments in phases:
        env = os.environ.copy()
        if devices != "all":
            env["CUDA_VISIBLE_DEVICES"] = (
                ""
                if devices == "cpu"
                else os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[0]
            )
        command = [sys.executable]
        if devices == "all":
            command += [
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                "--max_restarts=0",
            ]
        command += [script, *arguments, *common]
        phase = {
            "name": name,
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "binding_sha256": m["binding_sha256"],
            "job_id": job_id,
        }
        atomic(out / "progress.json", {"status": "running", "phase": phase, "completed": completed})
        path = out / "phase_logs" / (name + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with path.open("w") as log:
            process = subprocess.run(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT
            )
        phase.update(
            status="complete" if process.returncode == 0 else "failed",
            returncode=process.returncode,
            elapsed_seconds=time.monotonic() - started,
            log=fingerprint(path),
        )
        atomic(out / "phases" / (name + ".json"), phase)
        if process.returncode:
            atomic(
                out / "progress.json", {"status": "failed", "phase": phase, "completed": completed}
            )
            raise RuntimeError(
                f"diagnostic phase failed: {name}; "
                "preserve evidence and do not resubmit automatically"
            )
        completed.append(phase)
    atomic(
        out / "progress.json",
        {"status": "complete", "completed": completed, "training_authorized": False},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "run",
            "token-check",
            "make-weights",
            "compare-weights",
            "score-historical",
            "score-live",
            "smoke-live",
            "collect-smoke",
            "collect-live",
            "summarize",
        ],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extra-code", action="append", default=[])
    parser.add_argument("--kind", choices=["cpu", "gpu"])
    parser.add_argument("--retry-of")
    args = parser.parse_args()
    from hiring_bias_mitigation.utils.config import resolve_output_path

    out = Path(resolve_output_path(args.output_dir)).resolve()
    if args.command == "prepare":
        prepare(out, args.extra_code, args.retry_of)
        return
    m = load_manifest(out, full_verify=args.command == "run")
    functions = {
        "run": run,
        "token-check": token_check,
        "make-weights": make_weights,
        "compare-weights": compare_weights,
        "score-historical": score_historical,
        "score-live": score_live,
        "collect-live": collect_live,
        "summarize": summarize,
        "collect-smoke": collect_smoke,
    }
    if args.command == "smoke-live":
        if args.kind != "cpu":
            parser.error("smoke-live requires --kind cpu")
        score_live(out, m, "cpu", smoke=True)
        return
    if args.command in ["make-weights", "score-historical", "score-live", "collect-live"]:
        if not args.kind:
            parser.error("--kind is required for this phase")
        functions[args.command](out, m, args.kind)
    else:
        functions[args.command](out, m)


if __name__ == "__main__":
    main()
