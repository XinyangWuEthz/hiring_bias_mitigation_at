#!/usr/bin/env python3
"""Freeze a separate full-benchmark monitor bank for the authorized 425→500 LR pilot."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hiring_bias_mitigation.mitigation.full_benchmark_monitor import (
    SCHEMA,
    atomic_json,
    digest,
    fingerprint,
    verify_file,
)
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path

KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
SOURCE_DIR = "outputs/checkpoint_benchmark/grpo_v3_all_20261023"
SOURCE_BINDING = "d66a7315927d9b7b079a0529396b73b76d7aed61982f6f8f897c2f6ea326ce58"
SFT = "outputs/sft/qwen3.5-9b_en_only_sft_v3_additional"
TRAIN_DIR = "artifacts/semisynthetic-v3/grpo-v3-all"
DEFAULT_OUTPUT = "outputs/grpo_low_lr_pilot/grpo_v3_all_lr3e6_from425_20261023/monitor_bank"
SOURCE_FILES = ["adapter_model.safetensors", "adapter_config.json", "optimizer.pt", "scheduler.pt",
                "trainer_state.json", "training_args.bin", *[f"rng_state_{i}.pth" for i in range(4)]]
RUNTIME_CODE = ["src/hiring_bias_mitigation/mitigation/full_benchmark_monitor.py",
    "src/hiring_bias_mitigation/mitigation/benchmark_dev_monitor.py",
    "src/hiring_bias_mitigation/mitigation/grpo.py",
    "src/hiring_bias_mitigation/mitigation/grpo_continuation.py",
    "src/hiring_bias_mitigation/mitigation/training_common.py",
    "src/hiring_bias_mitigation/mitigation/merge.py",
    "src/hiring_bias_mitigation/eval/backends.py", "src/hiring_bias_mitigation/eval/parsing.py",
    "src/hiring_bias_mitigation/analysis/reference_metrics.py",
    "scripts/grpo_low_lr_pilot_results.py", "scripts/checkpoint_benchmark_trajectory.py",
    "scripts/analyze_checkpoint_average_sdt.py", "scripts/operating_point_control.py",
    "scripts/freeze_grpo_low_lr_full_monitor.py"]


def align(raw, expected):
    if len(raw) != len(expected) or raw.duplicated(KEY).any():
        raise ValueError("cached raw coverage differs from frozen prompts")
    order = expected[KEY].copy().assign(_order=range(len(expected)))
    joined = order.merge(raw, on=KEY, how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("cached raw identities differ")
    joined = joined.sort_values("_order")[raw.columns].reset_index(drop=True)
    for key in ["candidate_id", "job_id", "reference_decision"]:
        if not joined[key].equals(expected[key]):
            raise ValueError(f"cached raw {key} differs")
    if not joined.decision.isin(["hire", "reject"]).all() or not joined.outcome.eq("decided").all():
        raise ValueError("cached raw contains unusable decisions")
    return joined


def check_isolation(bank, train):
    if len(bank) != 31050 or len(train) != 9140:
        raise ValueError("the frozen 31050 benchmark / 9140 training rows are required")
    report = {}
    for key in ["candidate_id", "pair_id", "job_id"]:
        left, right = set(bank[key]), set(train[key])
        report[key] = {"benchmark_unique": len(left), "training_unique": len(right),
                       "overlap": len(left & right)}
        if left & right:
            raise ValueError(f"training leakage: {key}")
    return report


def tokenize_lengths(frame, tokenizer):
    from hiring_bias_mitigation.eval.backends import _chat_wrap

    lengths = []
    for start in range(0, len(frame), 128):
        texts = [_chat_wrap(tokenizer, text, {"enable_thinking": False})
                 for text in frame.prompt.iloc[start:start + 128]]
        encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    if max(lengths) + 512 > 4096:
        raise ValueError(f"benchmark input length {max(lengths)} + 512 exceeds colocated 4096 limit")
    return lengths


def freeze(output, extra_code=()):
    from transformers import AutoTokenizer

    output = Path(resolve_output_path(str(output))).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen monitor bank: {output}")
    source_dir = Path(resolve_output_path(SOURCE_DIR))
    source_manifest_path = source_dir / "manifest.json"
    source = json.loads(source_manifest_path.read_text())
    if (source["binding_sha256"] != SOURCE_BINDING or
            digest({k: v for k, v in source.items() if k != "binding_sha256"}) != SOURCE_BINDING):
        raise ValueError("source trajectory identity differs")
    verify_file(source["prompts"])
    frame = pd.read_parquet(source["prompts"]["path"])
    if frame.duplicated(KEY).any() or digest(frame.prompt.tolist()) != source["prompt_text_sha256"]:
        raise ValueError("source prompts differ")
    frame["prompt_id"] = [digest(list(values)) for values in frame[KEY].itertuples(index=False, name=None)]
    train_path = Path(resolve_output_path(f"{TRAIN_DIR}/grpo_train.parquet"))
    validation_path = Path(resolve_output_path(f"{TRAIN_DIR}/grpo_validation.parquet"))
    train_fp, validation_fp = fingerprint(train_path), fingerprint(validation_path)
    isolation = check_isolation(frame, pd.read_parquet(train_path))
    if int(frame.reference_decision.eq("hire").sum()) != 10350:
        raise ValueError("positive bank differs")
    steps = {item["step"]: item for item in source["steps"]}
    source_checkpoint = Path(steps[425]["adapter"])
    source_files = {name: fingerprint(source_checkpoint / name) for name in SOURCE_FILES}
    sft_dir = Path(resolve_output_path(SFT))
    if Path(steps[0]["adapter"]).resolve() != sft_dir.resolve():
        raise ValueError("source SFT adapter differs")
    source_state = json.loads((source_checkpoint / "trainer_state.json").read_text())
    if source_state["global_step"] != 425:
        raise ValueError("source checkpoint global step differs")
    inputs = [fingerprint(source_manifest_path), source["prompts"], train_fp, validation_fp,
              *source_files.values()]
    cached, raw_frames = {}, {}
    for step in [0, 425, 450, 475, 500]:
        result_path = source_dir / "steps" / f"step-{step:04d}" / "result.json"
        result = json.loads(result_path.read_text())
        if result["binding_sha256"] != SOURCE_BINDING:
            raise ValueError("cached result source binding differs")
        verify_file(result["raw"])
        label = "SFT-v3" if step == 0 else f"step{step}"
        cached[label] = {"step": step, "raw": result["raw"], "adapter": steps[step]["adapter"],
                         "adapter_chain": steps[step]["chain"]}
        raw_frames[label] = align(pd.read_parquet(result["raw"]["path"]), frame)
        inputs.extend([fingerprint(result_path), result["raw"]])
        for adapter in steps[step]["chain"]:
            for item in adapter["files"]:
                verify_file(item)
                inputs.append(item)
    runtime = [fingerprint(REPO_ROOT / name) for name in [*RUNTIME_CODE, *extra_code]]
    inputs.extend(runtime)
    for item in source["benchmark_partition"].values():
        verify_file(item)
        inputs.append(item)
    software = {name: importlib.metadata.version(name) for name in source["software"]}
    if software != source["software"]:
        raise ValueError("software differs from source benchmark")
    tokenizer_files = [fingerprint(sft_dir / name) for name in
        ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"] if (sft_dir / name).exists()]
    if not tokenizer_files:
        raise ValueError("frozen SFT tokenizer files are missing")
    inputs.extend(tokenizer_files)
    tokenizer = AutoTokenizer.from_pretrained(sft_dir, trust_remote_code=True, local_files_only=True)
    lengths = tokenize_lengths(frame, tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.with_name(output.name + ".building-" + uuid.uuid4().hex)
    stage.mkdir()
    try:
        frame.to_parquet(stage / "rows.parquet", index=False)
        pd.DataFrame({"prompt_id": frame.prompt_id, "input_tokens": lengths}).to_parquet(
            stage / "token_lengths.parquet", index=False)
        for label, filename in [("SFT-v3", "parent_decisions.parquet"),
                                ("step425", "start_decisions.parquet")]:
            raw = raw_frames[label]
            pd.DataFrame({"prompt_id": frame.prompt_id, "decision": raw.decision,
                          "raw_output": raw.raw_output}).to_parquet(stage / filename, index=False)

        def staged_fp(name):
            item = fingerprint(stage / name)
            item["path"] = str(output / name)
            return item

        continuation = {"source_checkpoint": str(source_checkpoint), "source_step": 425,
            "target_learning_rate": 3e-6, "source_learning_rate": 3e-5, "max_steps": 500,
            "world_size": 4, "source_files_sha256": {name: item["sha256"] for name, item in source_files.items()}}
        manifest = {"schema": SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
            "training_sft_checkpoint": SFT, "benchmark_previously_exposed": True,
            "independent_test_set": False,
            "policy": {"training_authorized": True, "scope": "this_pilot_only", "source_step": 425,
                "target_step": 500, "check_every_steps": 5, "save_every_steps": 5,
                "full_steps": [425, 450, 475, 500], "stop_delta_tpr_pp": -8.0,
                "original_final_tpr_guardrail_pp": -10.0, "first_stop_irreversible": True,
                "startup_exact_match_rows": 31050, "periodic_positive_rows": 10350,
                "trigger_checkpoint_is_safe": False, "gpu_startup_validation_required": True},
            "scorer": {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 512, "seed": 42,
                       "chat_template_kwargs": {"enable_thinking": False}},
            "rows_file": staged_fp("rows.parquet"),
            "parent_decisions_file": staged_fp("parent_decisions.parquet"),
            "start_decisions_file": staged_fp("start_decisions.parquet"),
            "tokenization": {"rows": len(frame), "max_input_tokens": max(lengths),
                "max_model_len": 4096, "max_new_tokens": 512, "add_special_tokens": True,
                "tokenizer_directory": str(sft_dir), "tokenizer_files": tokenizer_files,
                "lengths_file": staged_fp("token_lengths.parquet")},
            "training_data": {"train": train_fp, "validation": validation_fp, "n_train": 9140,
                              "overlap_checks": isolation},
            "continuation_source": continuation,
            "source_trajectory": {"directory": str(source_dir), "binding_sha256": SOURCE_BINDING,
                "manifest": fingerprint(source_manifest_path), "prompts": source["prompts"],
                "prompt_text_sha256": source["prompt_text_sha256"], "audit_config": source["audit_config"],
                "benchmark_partition": source["benchmark_partition"], "cached_models": cached},
            "runtime_code": runtime, "software": software,
            "inputs": list({item["path"]: item for item in inputs}.values()),
            "interpretation": ["Full exposed benchmark is development data for this pilot.",
                "An 8 pp sampled alarm cannot certify no crossing between five-step checks.",
                "Starting from recovered checkpoint 425 does not retroactively undo earlier crossings in the original run.",
                "The 10 pp original final standard is unchanged; final standalone merged audit is separate."]}
        manifest["binding_sha256"] = digest(manifest)
        atomic_json(stage / "manifest.json", manifest)
        stage.rename(output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    verify_file(train_fp)
    verify_file(validation_fp)
    receipt = {"schema": SCHEMA, "manifest": fingerprint(output / "manifest.json"),
               "continuation": continuation, "tokenization_max_input_tokens": max(lengths),
               "overlap_checks": isolation, "gpu_startup_validated": False,
               "no_training_or_inference_performed_by_freeze": True}
    atomic_json(output / "freeze_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--extra-code", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(freeze(args.output_dir, args.extra_code), indent=2))
