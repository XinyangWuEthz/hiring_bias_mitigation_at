#!/usr/bin/env python3
"""Freeze a label-only candidate split, then join the historical SFT parent.

Run `split` and persist its output before `materialize`. The split implementation
does not accept model outcomes. Materialization refuses a changed source or IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

VERSION = "benchmark-dev-monitor-v1"
SPLIT_VERSION = "benchmark-dev-v1"
SEED = "20261023"
KEY = ("pair_id", "protected_group", "protected_attr", "condition")
SFT_RUN = "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refuse to replace frozen file: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(canonical(row) + "\n")


def select_split(metadata):
    """Only the four allowed metadata columns enter this deterministic rule."""
    by_candidate = defaultdict(list)
    for original in metadata:
        row = {
            k: str(original[k])
            for k in ("pair_id", "candidate_id", "job_position", "reference_decision")
        }
        if row["reference_decision"] not in ("hire", "reject"):
            raise ValueError("Nonbinary reference label")
        by_candidate[row["candidate_id"]].append(row)
    if len(by_candidate) != 150:
        raise ValueError("Expected exactly 150 benchmark candidates")
    by_role = defaultdict(list)
    for candidate, rows in by_candidate.items():
        roles = {r["job_position"] for r in rows}
        if len(rows) != 3 or len(roles) != 1 or len({r["pair_id"] for r in rows}) != 3:
            raise ValueError("Each candidate must have three distinct pairs in one role")
        by_role[next(iter(roles))].append(
            (candidate, sum(r["reference_decision"] == "hire" for r in rows))
        )
    if len(by_role) != 30 or any(len(v) != 5 for v in by_role.values()):
        raise ValueError("Expected 30 roles with five candidates each")
    selected, role_counts = [], []
    for role, candidates in sorted(by_role.items()):
        hires = sum(n for _, n in candidates)

        def rank(combo, role=role, hires=hires):
            tie = f"{SPLIT_VERSION}|{SEED}|{role}|" + "|".join(c for c, _ in combo)
            return (
                abs(5 * sum(n for _, n in combo) - 3 * hires),
                hashlib.sha256(tie.encode()).hexdigest(),
            )

        choice = min(itertools.combinations(sorted(candidates), 3), key=rank)
        selected.extend(c for c, _ in choice)
        role_counts.append(
            {
                "role": role,
                "all_hire_pairs": hires,
                "dev_hire_pairs": sum(n for _, n in choice),
                "dev_candidate_ids": [c for c, _ in choice],
            }
        )
    dev = set(selected)
    result = {}
    for name, ids in (("dev", dev), ("historical_remainder", set(by_candidate) - dev)):
        rows = [r for c in sorted(ids) for r in by_candidate[c]]
        labels = Counter(r["reference_decision"] for r in rows)
        result[name] = {
            "candidate_ids": sorted(ids),
            "pair_ids": sorted(r["pair_id"] for r in rows),
            "candidate_count": len(ids),
            "pair_count": len(rows),
            "label_counts": dict(sorted(labels.items())),
            "positive_candidate_count": sum(
                any(r["reference_decision"] == "hire" for r in by_candidate[c]) for c in ids
            ),
            "prompt_count": len(rows) * 69,
            "positive_prompt_count": labels["hire"] * 69,
        }
    return {"strata": role_counts, "splits": result}


def source_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path)}


def freeze_split(repo, out):
    from hiring_bias_mitigation.data.benchmark import load_benchmark

    source = repo / "data/benchmark/reference_feedback_en.csv"
    metadata = load_benchmark("en")
    source_before = source_record(source)
    result = select_split(metadata.to_dict("records"))
    if source_record(source) != source_before:
        raise RuntimeError("Benchmark source changed during split")
    candidate_to_split = {
        c: name for name, part in result["splits"].items() for c in part["candidate_ids"]
    }
    jobs = {
        name: set(metadata.loc[metadata.candidate_id.isin(part["candidate_ids"]), "job_id"])
        for name, part in result["splits"].items()
    }
    result.update(
        {
            "schema": SPLIT_VERSION,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_before,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "seed": SEED,
            "dev_fraction": 0.6,
            "split_unit": "candidate_id",
            "allowed_selection_columns": [
                "pair_id",
                "candidate_id",
                "job_position",
                "reference_decision",
            ],
            "model_outcomes_used_for_selection": False,
            "rule": (
                "Each role: choose 3 of 5 candidates minimizing "
                "abs(5*dev_hire_pairs-3*role_hire_pairs); tie by "
                "SHA256('benchmark-dev-v1|20261023|'+role+'|'+'|'.join(sorted_candidate_ids))."
            ),
            "benchmark_previously_exposed": True,
            "remaining_split_name": "historical_remainder",
            "shared_job_ids": sorted(jobs["dev"] & jobs["historical_remainder"]),
            "parent_decisions_read": False,
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "split.json", result)
    write_jsonl(
        out / "pair_assignment.jsonl",
        (
            {
                "pair_id": r.pair_id,
                "candidate_id": r.candidate_id,
                "job_id": r.job_id,
                "job_position": r.job_position,
                "reference_decision": r.reference_decision,
                "split": candidate_to_split[r.candidate_id],
            }
            for r in metadata.itertuples()
        ),
    )
    return {
        "split_file": str(out / "split.json"),
        "split_sha256": sha(out / "split.json"),
        "counts": {
            k: {
                q: v[q]
                for q in (
                    "candidate_count",
                    "pair_count",
                    "label_counts",
                    "positive_candidate_count",
                    "prompt_count",
                    "positive_prompt_count",
                )
            }
            for k, v in result["splits"].items()
        },
    }


def materialize(repo, out, output_root, model_identity):
    import pandas as pd

    from hiring_bias_mitigation.data.benchmark import load_benchmark
    from hiring_bias_mitigation.eval.parsing import parse_frame
    from hiring_bias_mitigation.eval.runner import _build_prompts, build_frame
    from hiring_bias_mitigation.utils.config import load_config

    split_path = out / "split.json"
    split_bytes_hash = sha(split_path)
    split = json.loads(split_path.read_text())
    source = Path(split["source"]["path"])
    if source_record(source) != split["source"]:
        raise RuntimeError("Frozen source changed")
    recomputed = select_split(load_benchmark("en").to_dict("records"))
    if recomputed["splits"] != split["splits"] or recomputed["strata"] != split["strata"]:
        raise RuntimeError("Frozen split differs from declared deterministic rule")
    cfg_path = repo / "configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml"
    cfg = load_config(str(cfg_path))
    frame = build_frame(cfg)
    if len(frame) != 31050:
        raise ValueError("Expected all 31,050 original benchmark rows")
    frame["prompt"] = _build_prompts(frame, "baseline")

    def row_id(row):
        values = [None if pd.isna(row[k]) else str(row[k]) for k in KEY]
        return hashlib.sha256(canonical(values).encode()).hexdigest()

    frame["prompt_id"] = [row_id(r) for r in frame.to_dict("records")]
    if frame.prompt_id.duplicated().any():
        raise ValueError("Duplicate original benchmark key")
    # Only now read the parent's outcomes; split file is already frozen on disk.
    raw_path = output_root / "outputs/raw" / f"{SFT_RUN}.parquet"
    meta_path = raw_path.with_suffix(".meta.json")
    raw_source, meta_source = source_record(raw_path), source_record(meta_path)
    raw = pd.read_parquet(raw_path)
    reparsed = parse_frame(raw)
    if not raw.decision.equals(reparsed.decision) or not raw.outcome.equals(reparsed.outcome):
        raise ValueError("Current official parser differs from cached parent decisions")
    raw["prompt_id"] = [row_id(r) for r in raw.to_dict("records")]
    if len(raw) != len(frame) or raw.prompt_id.duplicated().any():
        raise ValueError("Incomplete or duplicate SFT parent rows")
    if set(raw.prompt_id) != set(frame.prompt_id):
        raise ValueError("Parent expected keys differ")
    if not raw.outcome.eq("decided").all() or not raw.decision.isin(["hire", "reject"]).all():
        raise ValueError("Parent has unusable decisions")
    raw["prompt"] = _build_prompts(raw, "baseline")
    aligned = raw.set_index("prompt_id").loc[frame.prompt_id]
    if frame.prompt.tolist() != aligned.prompt.tolist():
        raise ValueError("Parent prompt text differs from frozen original benchmark")
    if frame.reference_decision.tolist() != aligned.reference_decision.tolist():
        raise ValueError("Parent reference labels differ")
    if source_record(raw_path) != raw_source or source_record(meta_path) != meta_source:
        raise RuntimeError("Parent sources changed during materialization")
    decisions = dict(zip(raw.prompt_id, raw.decision))
    fields = [
        "prompt_id",
        "prompt",
        "candidate_id",
        "pair_id",
        "job_id",
        "job_position",
        "reference_decision",
        "lang",
        "protected_group",
        "protected_attr",
        "condition",
    ]
    dev_ids = split["splits"]["dev"]["candidate_ids"]
    dev = frame[frame.candidate_id.isin(dev_ids)]
    positives = dev[dev.reference_decision.eq("hire")]
    if len(dev) != 18630 or len(positives) != 6417:
        raise ValueError("Unexpected frozen dev row counts")
    write_jsonl(out / "dev_rows.jsonl", dev[fields].to_dict("records"))
    write_jsonl(out / "dev_positive_rows.jsonl", positives[fields].to_dict("records"))
    write_jsonl(
        out / "dev_parent_decisions.jsonl",
        ({"prompt_id": pid, "decision": decisions[pid]} for pid in dev.prompt_id),
    )
    write_jsonl(
        out / "dev_positive_parent_decisions.jsonl",
        ({"prompt_id": pid, "decision": decisions[pid]} for pid in positives.prompt_id),
    )
    model_inventory = json.loads(model_identity.read_text())
    parent = next(m for m in model_inventory["models"] if m["label"] == "SFT-v3")
    parent_identity = hashlib.sha256(canonical(parent).encode()).hexdigest()
    write_json(out / "parent_model_identity.json", parent)
    rows_path = out / "dev_positive_rows.jsonl"
    parent_path = out / "dev_positive_parent_decisions.jsonl"
    artifacts = {
        p.name: source_record(p)
        for p in (
            out / "dev_rows.jsonl",
            rows_path,
            out / "dev_parent_decisions.jsonl",
            parent_path,
            out / "parent_model_identity.json",
            split_path,
            out / "pair_assignment.jsonl",
        )
    }
    code_paths = [
        cfg_path,
        repo / "src/hiring_bias_mitigation/data/benchmark.py",
        repo / "src/hiring_bias_mitigation/data/prompts.py",
        repo / "src/hiring_bias_mitigation/data/injection.py",
        repo / "src/hiring_bias_mitigation/data/protected_groups.py",
        repo / "src/hiring_bias_mitigation/eval/parsing.py",
        repo / "src/hiring_bias_mitigation/eval/runner.py",
    ]
    summary = {}
    for name, part in split["splits"].items():
        ids = part["candidate_ids"]
        rows = frame[frame.candidate_id.isin(ids)]
        pos = rows[rows.reference_decision.eq("hire")]
        neg = rows[rows.reference_decision.eq("reject")]
        summary[name] = {
            "rows": len(rows),
            "positive_rows": len(pos),
            "negative_rows": len(neg),
            "sft_hire_on_positive": sum(decisions[pid] == "hire" for pid in pos.prompt_id),
            "sft_tpr_pct": 100 * sum(decisions[pid] == "hire" for pid in pos.prompt_id) / len(pos),
            "sft_fpr_pct": 100 * sum(decisions[pid] == "hire" for pid in neg.prompt_id) / len(neg),
        }
    manifest = {
        "schema": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows_file": rows_path.name,
        "rows_sha256": sha(rows_path),
        "parent_decisions_file": parent_path.name,
        "parent_decisions_sha256": sha(parent_path),
        "parent_model": {
            "run": SFT_RUN,
            "identity_sha256": parent_identity,
            "sft_checkpoint": "outputs/sft/qwen3.5-9b_en_only_sft_v3_additional",
            "checkpoint_step": 3923,
        },
        "scorer": {
            "max_new_tokens": 512,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "split": {
            "candidate_ids": dev_ids,
            "remaining_candidate_ids": split["splits"]["historical_remainder"]["candidate_ids"],
            "benchmark_previously_exposed": True,
            "split_file": split_path.name,
            "split_sha256": split_bytes_hash,
            "rows_subset": "reference_decision=hire",
            "positive_candidate_count": 50,
            "shared_job_count": len(split["shared_job_ids"]),
        },
        "source": split["source"],
        "parent_raw": raw_source,
        "parent_meta": meta_source,
        "model_identity_inventory": source_record(model_identity),
        "source_code": [source_record(p) for p in code_paths],
        "artifacts": artifacts,
        "parent_baselines": summary,
        "policy": {
            "check_every_optimizer_steps": 25,
            "stop_delta_tpr_pp": -8.0,
            "stop_comparison": "<=",
            "final_guardrail_delta_tpr_pp": -10.0,
            "integrity_failure": "save_and_stop",
            "bootstrap_intervals": "descriptive_only",
            "training_authorized": False,
        },
    }
    if sha(split_path) != split_bytes_hash:
        raise RuntimeError("Frozen split changed during outcome join")
    write_json(out / "manifest.json", manifest)
    return {
        "manifest": str(out / "manifest.json"),
        "manifest_sha256": sha(out / "manifest.json"),
        "parent_baselines": summary,
        "rows": len(positives),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["split", "materialize"])
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model-identity", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo / "src"))
    if args.stage == "split":
        result = freeze_split(args.repo, args.out)
    else:
        if args.output_root is None or args.model_identity is None:
            parser.error("materialize requires --output-root and --model-identity")
        result = materialize(args.repo, args.out, args.output_root, args.model_identity)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
