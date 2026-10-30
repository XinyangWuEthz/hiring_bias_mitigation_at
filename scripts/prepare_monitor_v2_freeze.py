"""Freeze monitor-v2 intake without annotation, inference or training.

The first lock fixes inputs and procedures. It cannot manufacture final labels,
parent margins or a passing transfer test. A different, reviewed release command
will be needed for the second lock; this command always forbids arm scoring and
training. All artifact paths are resolved relative to the config directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from pathlib import Path

SPLIT_SALT = "monitor-v2-split-20261023:42:"
INTAKE_SALT = "monitor-v2-intake-20261023:42:"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def text_digest(value):
    return digest(" ".join(unicodedata.normalize("NFKC", value).split()).casefold())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)
                                + "\n" for r in records))


def partition(candidate_id):
    """Independent of labels, parent confidence and trained-arm outcomes."""
    value = digest(SPLIT_SALT + candidate_id)
    return "monitor" if int(value, 16) < 2**255 else "validation"


def verify_inputs(config, base):
    paths = {}
    for key, entry in config["inputs"].items():
        path = (base / entry["path"]).resolve()
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Input fingerprint mismatch: {key}")
        paths[key] = path
    if config["annotation"]["labels"] != ["hire", "reject"]:
        raise ValueError("Only resolved binary labels may enter the banks")
    if config["partition"]["salt"] != SPLIT_SALT:
        raise ValueError("Partition salt changed; version the experiment")
    if config["partition"]["intake_salt"] != INTAKE_SALT:
        raise ValueError("Intake order changed; version the experiment")
    if config["release"]["arm_scoring_allowed"] or config["release"]["training_allowed"]:
        raise ValueError("An intake freeze cannot authorize scoring or training")
    return paths


def prepare(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output)
    config = read_json(config_path)
    paths = verify_inputs(config, config_path.parent)
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing freeze: {output}")
    registry = read_jsonl(paths["eligible_registry"])
    batch = read_jsonl(paths["intake_batch"])
    excluded = read_json(paths["excluded_candidates"])
    # Inventory exports one union alongside reason-specific lists.
    excluded_ids = set(excluded["all_candidate_ids"])
    excluded_jobs = set(excluded.get("benchmark_job_ids", []))
    ids = [r["candidate_id"] for r in registry]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ValueError("Registry must contain exactly one row per candidate")
    if set(ids) & excluded_ids:
        raise ValueError("An excluded candidate entered the registry")
    if {r["job_id"] for r in registry} & excluded_jobs:
        raise ValueError("A benchmark job entered the registry")
    if len({r["cv_sha256_norm"] for r in registry}) != len(registry):
        raise ValueError("Normalized duplicate CVs entered the registry")
    if len({r["pair_id"] for r in registry}) != len(registry):
        raise ValueError("Duplicate pairs entered the registry")
    expected = sorted(ids, key=lambda c: digest(INTAKE_SALT + c))
    if ids != expected:
        raise ValueError("Registry is not in the frozen hash order")
    if [r["candidate_id"] for r in batch] != ids[:config["intake"]["first_batch_size"]]:
        raise ValueError("First batch was selected using an unfrozen rule")
    by_id = {r["candidate_id"]: r for r in registry}
    for row in batch:
        for key in ("pair_id", "job_id", "cv_sha256_norm", "jd_sha256_norm"):
            if row[key] != by_id[row["candidate_id"]][key]:
                raise ValueError(f"Intake/registry mismatch: {key}")
        if row.get("label") is not None or row.get("label_status") != "unlabelled":
            raise ValueError("The first lock accepts no preselected labels")
        if not row.get("cv") or not row.get("job_description"):
            raise ValueError("Missing candidate/job text")
        if (text_digest(row["cv"]) != row["cv_sha256_norm"]
                or text_digest(row["job_description"]) != row["jd_sha256_norm"]):
            raise ValueError("Text does not match its registry digest")
        if row.get("source") != "djinni_new" or row.get("development_exposed") is not False:
            raise ValueError("Intake source or development exposure is invalid")
    weights = read_json(paths["legacy_target_weights"])
    expected_keys = {(b, g["protected_group"], g["condition"])
                     for b in range(8) for g in weights["grid"]}
    for label in ("hire", "reject"):
        cells = weights["cells"][label]
        actual_keys = [(c["bin"], c["protected_group"], c["condition"]) for c in cells]
        if len(actual_keys) != len(set(actual_keys)) or set(actual_keys) != expected_keys:
            raise ValueError("Missing or duplicated target cell")
        if any(not math.isfinite(c["weight"]) or isinstance(c["count"], bool)
               or not isinstance(c["count"], int) for c in cells):
            raise ValueError("Nonfinite weights or noninteger counts")
        if abs(sum(c["weight"] for c in cells) - 1.0) > 1e-12:
            raise ValueError(f"Target weights do not sum to one: {label}")
        if any(c["count"] < 0 or c["weight"] < 0 for c in cells):
            raise ValueError("Negative target weight")
        total = sum(c["count"] for c in cells)
        if total <= 0 or any(abs(c["weight"] - c["count"] / total) > 1e-12 for c in cells):
            raise ValueError("Target weights differ from count proportions")
    rows = [dict(r, bank=partition(r["candidate_id"])) for r in registry]
    queue = []
    for row in batch:
        # Only these fields may be exposed to the CV reviewers. No predictions,
        # old anchors, scores, eventual bank, or intake propensity are present.
        queue.append({"candidate_id": row["candidate_id"], "pair_id": row["pair_id"],
                      "job_id": row["job_id"], "job_description": row["job_description"],
                      "cv": row["cv"], "source_as_of": config["annotation"]["source_as_of"],
                      "label": None, "eligibility": "awaiting_jd_rubric"})
    jd_by_id = {}
    for row in batch:
        old = jd_by_id.get(row["job_id"])
        if old is not None and old != row["job_description"]:
            raise ValueError("One job ID has conflicting text")
        jd_by_id[row["job_id"]] = row["job_description"]
    output.mkdir(parents=True)
    write_jsonl(output / "candidate_partitions.jsonl", rows)
    write_jsonl(output / "annotation_queue.jsonl", queue)
    write_jsonl(output / "jd_only_queue.jsonl", [
        {"job_id": key, "job_description": value}
        for key, value in sorted(jd_by_id.items())])
    manifest = {"experiment_id": config["experiment_id"], "stage": "intake_frozen",
                "config_sha256": sha256(config_path), "inputs": config["inputs"],
                "eligible_candidates": len(rows), "first_batch_candidates": len(batch),
                "first_batch_jobs": len(jd_by_id),
                "first_batch_by_bank": {bank: sum(partition(r["candidate_id"]) == bank
                                                     for r in batch)
                                        for bank in ("monitor", "validation")},
                "confirmed_hire_candidates": 0, "confirmed_reject_candidates": 0,
                "arm_scoring_allowed": False, "training_allowed": False,
                "pending_second_lock": config["release"]["pending_second_lock"],
                "outputs": {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "manifest.json", manifest)
    return manifest


def verify_freeze(config_path, output, require_stage="intake"):
    config_path, output = Path(config_path).resolve(), Path(output)
    config = read_json(config_path)
    verify_inputs(config, config_path.parent)
    manifest = read_json(output / "manifest.json")
    if manifest["config_sha256"] != sha256(config_path):
        raise ValueError("Config changed after freeze")
    if set(manifest["outputs"]) != {"candidate_partitions.jsonl", "annotation_queue.jsonl",
                                   "jd_only_queue.jsonl"}:
        raise ValueError("Frozen output manifest is incomplete")
    for name, expected in manifest["outputs"].items():
        if sha256(output / name) != expected:
            raise ValueError(f"Frozen output changed: {name}")
    if manifest["arm_scoring_allowed"] or manifest["training_allowed"]:
        raise ValueError("Intake manifest has invalid release flags")
    if manifest["stage"] != "intake_frozen" or manifest["inputs"] != config["inputs"]:
        raise ValueError("Freeze identity or inputs changed")
    if manifest["confirmed_hire_candidates"] != 0 or manifest["confirmed_reject_candidates"] != 0:
        raise ValueError("Intake lock cannot claim confirmed labels")
    if require_stage != "intake":
        raise ValueError("Blocked: second lock, labels, parent margins, "
                         "bridge and validation missing")
    return {"intake_integrity_passed": True, "arm_scoring_allowed": False,
            "training_allowed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-stage", choices=["intake", "arm-scoring", "training"],
                        default="intake")
    args = parser.parse_args()
    try:
        if args.verify:
            result = verify_freeze(args.config, args.output, args.require_stage)
        else:
            if args.require_stage != "intake":
                raise ValueError("This command creates only an intake freeze")
            result = prepare(args.config, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"monitor-v2 freeze refused: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
