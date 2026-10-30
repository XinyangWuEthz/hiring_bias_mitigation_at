"""Extract exact target weights from frozen parent margins, without arm scores."""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from prepare_monitor_v2_freeze import sha256, write_json

EDGES = [-4.595, -2.197, -0.847, 0.0, 0.847, 2.197, 4.595]


def extract(source):
    counts = {"hire": Counter(), "reject": Counter()}
    parent_hires = Counter()
    grids = defaultdict(set)
    pair_labels = {}
    seen = set()
    n = 0
    with gzip.open(source, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["cohort"] != "benchmark":
                continue
            # Deliberately read no trained-arm margin columns.
            label, margin = row["reference_decision"], row["margin__SFT-v3"]
            if label not in counts or not math.isfinite(margin):
                raise ValueError("Invalid reference label or parent margin")
            if pair_labels.setdefault(row["pair_id"], label) != label:
                raise ValueError("Reference label changes across a pair's variants")
            key = row["protected_group"], row["condition"], row["protected_attr"]
            identity = row["pair_id"], key
            if identity in seen:
                raise ValueError("Duplicate benchmark key")
            seen.add(identity)
            grids[row["pair_id"]].add(key)
            counts[label][(bisect.bisect_right(EDGES, margin), *key[:2])] += 1
            parent_hires[label] += margin > 0
            n += 1
    if n != 30600 or len(grids) != 450:
        raise ValueError("Expected the frozen 450-pair attributed benchmark")
    grid = next(iter(grids.values()))
    if len(grid) != 68 or any(values != grid for values in grids.values()):
        raise ValueError("Expected the same 68 attributed variants on every pair")
    cells = {}
    for label, counter in counts.items():
        denominator = sum(counter.values())
        if not denominator:
            raise ValueError("Both reference labels must have target support")
        cells[label] = [
            {"bin": b, "protected_group": g, "condition": c,
             "count": counter[(b, g, c)], "denominator": denominator,
             "weight": counter[(b, g, c)] / denominator}
            for b in range(8) for g, c in sorted({key[:2] for key in grid})]
    return {"target_id": "legacy-gpt4o-attributed-parent-margin-v2",
            "source_path": str(Path(source).resolve()), "source_sha256": sha256(source),
            "used_columns": ["cohort", "reference_decision", "margin__SFT-v3", "pair_id",
                             "protected_group", "condition", "protected_attr"],
            "rows": n, "pairs": len(grids), "edges": EDGES, "boundary": "searchsorted-right",
            "decision_rule": "margin > 0; exact zero is reject",
            "label_authority": "released benchmark GPT-4o, unchanged",
            "not_validated_as_unified_protocol_labels": True,
            "parent_rates": {label: parent_hires[label] / sum(counts[label].values())
                             for label in counts}, "cells": cells,
            "grid": [{"protected_group": g, "condition": c, "protected_attr": a}
                     for g, c, a in sorted(grid)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    write_json(args.output, extract(args.source))


if __name__ == "__main__":
    main()
