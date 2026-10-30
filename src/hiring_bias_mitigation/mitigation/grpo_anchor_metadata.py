"""Read-only anchor metadata for the exact frozen GRPO-v3-all splits.

Call before chat-template rendering. This module never edits parquet files, labels,
prompts, or row order. Other datasets need their own explicit provenance contract;
they must not silently borrow these anchor files.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

ROOT = "artifacts/semisynthetic-v3"
ROW_KEY = ("pair_id", "protected_attr", "condition")
METADATA_COLUMNS = ("prompt_id", "winning_votes", "decided_draws", "vote_stratum")
FROZEN_FILES = {
    "train": {
        "path": f"{ROOT}/grpo-v3-all/grpo_train.parquet",
        "sha256": "a2204df3c0ba784b5205b3bc27757a1396c4da125cf509949594ec5f89a69e5c",
        "bytes": 14272393,
    },
    "validation": {
        "path": f"{ROOT}/grpo-v3-all/grpo_validation.parquet",
        "sha256": "873cc509527544d17211e8c6928a4fde6041a4ae2cc9f070df00c69bbeb48a8c",
        "bytes": 352777,
    },
    "new_rows": {
        "path": f"{ROOT}/onpolicy-new/rows.parquet",
        "sha256": "80e4d90d48bd3e740a0010613a23c3f2522e6bfe513a5c247206c19db08281aa",
        "bytes": 12443240,
    },
    "new_prompts": {
        "path": f"{ROOT}/onpolicy-new/prompts.parquet",
        "sha256": "4d0d0c6e0fbcc0a309ebfdba47082e35a83cdc8b0cd0224f5f5726ae08449b02",
        "bytes": 16164936,
    },
    "new_anchors": {
        "path": f"{ROOT}/new-pairs/anchors_en.parquet",
        "sha256": "9b12ec8895e3b432b3cfd2cd0a6bff63f33537f595c63c17113d616dc4cf4ae2",
        "bytes": 4512572,
    },
    "new2_rows": {
        "path": f"{ROOT}/onpolicy-new2/rows.parquet",
        "sha256": "8ad06ef808333e8c1bf44a27e0f2d7a86d8b41533733fa6e0da7ad358a36ee9f",
        "bytes": 33932997,
    },
    "new2_prompts": {
        "path": f"{ROOT}/onpolicy-new2/prompts.parquet",
        "sha256": "ef13f531c3f205bd54d6fff5cc44d6452f9e3281708cc6bf4d8c45b49976b9e2",
        "bytes": 40536682,
    },
    "new2_anchors": {
        "path": f"{ROOT}/new-pairs-2/anchors_en.parquet",
        "sha256": "3687208ca97673555ed596e03ce6446754a015d468e0ff68e12d9dc5ae5d81f6",
        "bytes": 11299333,
    },
}
SPLIT_ROWS = {"train": 9140, "validation": 256}
SOURCE_PREFIXES = ("new", "new2")


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def _fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _verify_sources(resolve_path: Callable[[str], str | Path]) -> dict[str, dict]:
    output = {}
    for name, spec in FROZEN_FILES.items():
        path = Path(resolve_path(spec["path"])).resolve()
        actual = _fingerprint(path)
        if any(actual[field] != spec[field] for field in ("sha256", "bytes")):
            raise ValueError(f"Frozen GRPO source fingerprint changed: {name}: {path}")
        output[name] = actual
    return output


def _read_records(path: Path, columns: list[str] | None = None) -> Iterator[dict]:
    import pyarrow.parquet as pq

    # Streaming also avoids materializing every long source prompt in memory.
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns,
                                                  use_threads=False):
        yield from batch.to_pylist()


def _key(row: Mapping) -> tuple[str, str, str]:
    values = tuple(row.get(name) for name in ROW_KEY)
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("Missing or invalid GRPO row key")
    return values


def prompt_id(row: Mapping) -> str:
    """Stable training-row identity, including the unchanged raw prompt's digest."""
    _key(row)
    if not isinstance(row.get("prompt"), str) or not isinstance(row.get("lang"), str):
        raise ValueError("Raw prompt and language are required for prompt_id")
    return _digest({"schema": "grpo-training-prompt-v1",
                    **{name: row[name] for name in (*ROW_KEY, "lang")},
                    "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest()})


def _checked_vote(anchor: Mapping, reference: str) -> tuple[int, int, str]:
    if reference not in ("hire", "reject"):
        raise ValueError("Unknown reference decision")
    if (anchor.get("reference_decision") != reference or anchor.get("majority") != reference
            or anchor.get("anchored") is not True or anchor.get("anchor_source") != "vote"):
        raise ValueError("Conflicting, missing, or fallback anchor label")
    hire, decided = anchor.get("hire_votes"), anchor.get("decided")
    if (type(hire) is not int or type(decided) is not int
            or decided <= 0 or not 0 <= hire <= decided):
        raise ValueError("Anchor vote counts must be valid integers")
    winning = hire if reference == "hire" else decided - hire
    if 2 * winning <= decided:
        raise ValueError("Reference label does not have a strict vote majority")
    share = anchor.get("majority_share")
    if (not isinstance(share, (float, int)) or isinstance(share, bool)
            or not math.isclose(share, winning / decided, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Stored anchor majority share conflicts with vote counts")
    return winning, decided, f"{winning}_of_{decided}"


def _join_metadata(rows: list[dict], files: dict[str, dict]) -> dict[str, list]:
    wanted = {}
    for row in rows:
        key = _key(row)
        if key in wanted:
            raise ValueError("Duplicate row key in frozen split")
        if row.get("reference_decision") not in ("hire", "reject"):
            raise ValueError("Unknown reference decision in frozen split")
        wanted[key] = row
    pair_ids = {row["pair_id"] for row in rows}
    anchors, matched = {}, set()
    source_columns = [*ROW_KEY, "candidate_id", "job_id", "lang", "reference_decision"]
    anchor_columns = ["pair_id", "reference_decision", "majority", "hire_votes", "decided",
                      "majority_share", "anchor_source", "anchored"]
    for prefix in SOURCE_PREFIXES:
        source_indexes = {}
        seen_keys = set()
        for index, source in enumerate(_read_records(Path(files[f"{prefix}_rows"]["path"]),
                                                    source_columns)):
            key = _key(source)
            if key in seen_keys:
                raise ValueError("Duplicate source row key")
            seen_keys.add(key)
            if key not in wanted:
                continue
            if key in matched:
                raise ValueError("Row belongs to more than one source pool")
            row = wanted[key]
            if any(source[name] != row[name] for name in source_columns):
                raise ValueError("Frozen row identity or label conflicts with its source")
            source_indexes[str(index)] = row["prompt"]
            matched.add(key)
        seen_prompts = set()
        for source in _read_records(Path(files[f"{prefix}_prompts"]["path"]),
                                    ["kind", "key", "prompt"]):
            index = str(source["key"])
            if source["kind"] != "attributed" or index not in source_indexes:
                continue
            if index in seen_prompts or source["prompt"] != source_indexes[index]:
                raise ValueError("Duplicate or changed source prompt")
            seen_prompts.add(index)
        if seen_prompts != set(source_indexes):
            raise ValueError("Missing source prompt")
        for anchor in _read_records(Path(files[f"{prefix}_anchors"]["path"]), anchor_columns):
            pair = anchor["pair_id"]
            if pair not in pair_ids:
                continue
            if pair in anchors:
                raise ValueError("Duplicate anchor pair")
            anchors[pair] = anchor
    if matched != set(wanted):
        raise ValueError("Frozen split contains rows missing from its source pools")
    if set(anchors) != pair_ids:
        raise ValueError("Frozen split contains a pair missing its anchor")
    metadata = {name: [] for name in METADATA_COLUMNS}
    pair_labels = {}
    for row in rows:
        pair, reference = row["pair_id"], row["reference_decision"]
        if pair in pair_labels and pair_labels[pair] != reference:
            raise ValueError("Conflicting reference labels within one pair")
        pair_labels[pair] = reference
        winning, decided, stratum = _checked_vote(anchors[pair], reference)
        for name, value in zip(METADATA_COLUMNS, (prompt_id(row), winning, decided, stratum)):
            metadata[name].append(value)
    if len(set(metadata["prompt_id"])) != len(rows):
        raise ValueError("Duplicate prompt_id")
    return metadata


def attach_anchor_metadata(dataset, *, split: str,
                           resolve_path: Callable[[str], str | Path]):
    """Return (Dataset plus metadata, receipt), before rendering raw input prompts.

    `resolve_path` has the same string-in/path-out contract as resolve_output_path.
    The caller may persist the receipt. Existing metadata must agree exactly; no
    column is silently overwritten. Filtering or reordering a frozen split fails.
    """
    if split not in SPLIT_ROWS:
        raise ValueError("Only the frozen GRPO-v3-all train or validation split is supported")
    before = _verify_sources(resolve_path)
    frozen = list(_read_records(Path(before[split]["path"])))
    if len(frozen) != SPLIT_ROWS[split] or len(dataset) != len(frozen):
        raise ValueError("Dataset is not the complete frozen split")
    original_columns = list(frozen[0])
    extra = set(dataset.column_names) - set(original_columns) - set(METADATA_COLUMNS)
    if extra or not set(original_columns) <= set(dataset.column_names):
        raise ValueError("Dataset columns differ from the frozen split")
    actual = list(dataset)
    for index, (row, expected) in enumerate(zip(actual, frozen)):
        if any(row[name] != expected[name] for name in original_columns):
            raise ValueError(f"Raw dataset row {index} differs from the frozen ordered split")
    metadata = _join_metadata(frozen, before)
    result = dataset
    for name in METADATA_COLUMNS:
        if name in dataset.column_names:
            if [row[name] for row in actual] != metadata[name]:
                raise ValueError(f"Existing metadata conflicts with frozen provenance: {name}")
        else:
            result = result.add_column(name, metadata[name])
    after = _verify_sources(resolve_path)
    if before != after:
        raise ValueError("Source files changed while joining anchor metadata")
    strata = Counter((row["reference_decision"], votes)
                     for row, votes in zip(frozen, metadata["vote_stratum"]))
    receipt = {
        "schema": "grpo-v3-all-anchor-metadata-v1", "split": split, "rows": len(frozen),
        "added_columns": list(METADATA_COLUMNS), "original_columns": original_columns,
        "raw_prompt_join_before_chat_rendering": True, "row_order_preserved": True,
        "original_fields_preserved": True, "source_hashes_unchanged": True,
        "files_before": before, "files_after": after,
        "ordered_prompt_ids_sha256": _digest(metadata["prompt_id"]),
        "metadata_sha256": _digest(metadata),
        "strata": [{"reference_decision": side, "vote_stratum": vote, "rows": count}
                   for (side, vote), count in sorted(strata.items())],
        "vote_definition": "winning votes / decided draws; invalid/refused draws excluded",
    }
    receipt["binding_sha256"] = _digest(receipt)
    return result, receipt
