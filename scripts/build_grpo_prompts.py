"""Builds the GRPO pilot's prompt set (CPU, seconds). What it keeps and why:
`src/hiring_bias_mitigation/generation/grpo_prompts.py`.

    python scripts/build_grpo_prompts.py                     # grpo-02: both sides the same pull
    python scripts/build_grpo_prompts.py --balance tipped --tipped-share 0.5 \
        --out artifacts/semisynthetic-v2/grpo-01            # grpo-01: half of each side tipped

`--pool` and `--hire-topup` repeat for several runs whose rows are disjoint (V3 round 3's
pilot and its second batch). `--fill pull` fills each half with the rows that carry signal
first, and `--n-train auto` makes each half as large as both verdicts' rows with a sampled
error allow (`grpo_prompts.signal_balanced_train`, `informative_half`).

Inputs are the v1 English pool as the SFT student was sampled on it (`onpolicy-full`: rows,
their attributed prompts, the k=8 selection and samples) and the k=32 hire top-up's selection
and samples (`onpolicy-hire32`); `--hire-topup none` for a pool sampled at k=8 only. Every
candidate in a `*validation.parquet` under semisynthetic-v2 is held out of training. Output, read
by a training config with `data_kind: grpo`:

    grpo_train.parquet, grpo_validation.parquet, report.json, samples.md

`--monitor-per-bin N` also writes `grpo_monitor.parquet` and `monitor.json`: held-out rows for
the margin monitor, N per bin of the parent's sampled hire rate (`grpo_prompts.monitor_rows`).
With `--monitor-only` and the arguments an existing build ran with, only those two are written,
next to that build's untouched splits:

    python scripts/build_grpo_prompts.py <the build's arguments> --monitor-per-bin 96 --monitor-only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.generation import grpo_prompts as G  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("build_grpo_prompts")


def write_monitor(held_pool: pd.DataFrame, args, out: Path, n_heldout: int) -> None:
    """grpo_monitor.parquet and monitor.json. Against an existing build (--monitor-only), no
    monitor candidate may be one its training split holds."""
    monitor, plan = G.monitor_rows(held_pool, args.monitor_per_bin, args.seed)
    assert_no_leakage(monitor, "grpo_monitor")
    train_path = out / "grpo_train.parquet"
    if train_path.is_file():
        trained = set(G.read(train_path, ["candidate_id"])["candidate_id"].astype(str))
        overlap = trained & set(monitor["candidate_id"].astype(str))
        if overlap:
            raise AssertionError(f"{len(overlap)} monitor candidates are in {train_path}")
    out.mkdir(parents=True, exist_ok=True)
    monitor[G.COLUMNS].to_parquet(out / "grpo_monitor.parquet", index=False)
    report = {"per_bin": args.monitor_per_bin, "seed": args.seed, "bins": plan,
              "heldout": {"candidates": n_heldout, "rows": len(held_pool)},
              "monitor": G.describe(monitor),
              "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "monitor.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s: %d monitor rows", out / "grpo_monitor.parquet", len(monitor))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", action="append",
                        help="an on-policy run directory; repeat for several "
                             "(default artifacts/semisynthetic-v2/onpolicy-full)")
    parser.add_argument("--hire-topup", action="append",
                        help="a k=32 top-up's run directory, repeatable, or `none` "
                             "(default artifacts/semisynthetic-v2/onpolicy-hire32)")
    parser.add_argument("--validation-root", default="artifacts/semisynthetic-v2",
                        help="every *validation.parquet below it names held-out candidates")
    parser.add_argument("--out", default="artifacts/semisynthetic-v2/grpo-02")
    parser.add_argument("--n-train", default="3000",
                        help="training prompts, or `auto`: twice informative_half")
    parser.add_argument("--fill", choices=("random", "pull"), default="random",
                        help="how untipped rows fill a half with --balance signal")
    parser.add_argument("--balance", choices=("signal", "tipped"), default="signal",
                        help="signal: both halves the same summed pull (grpo-02); tipped: "
                             "--tipped-share of each half from tipped rows (grpo-01)")
    parser.add_argument("--tipped-share", type=float, default=0.5)
    parser.add_argument("--n-validation", type=int, default=256)
    parser.add_argument("--monitor-per-bin", type=int, default=0,
                        help="held-out rows per parent hire-rate bin for the margin monitor")
    parser.add_argument("--monitor-only", action="store_true",
                        help="write only grpo_monitor.parquet next to an existing build")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.monitor_only and args.monitor_per_bin <= 0:
        parser.error("--monitor-only needs --monitor-per-bin")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    require_output_root()
    args.pool = args.pool or ["artifacts/semisynthetic-v2/onpolicy-full"]
    args.hire_topup = args.hire_topup or ["artifacts/semisynthetic-v2/onpolicy-hire32"]
    pools = [Path(resolve_output_path(p)) for p in args.pool]
    topups = [Path(resolve_output_path(t)) for t in args.hire_topup if t.lower() != "none"]
    out = Path(resolve_output_path(args.out))

    rows = pd.concat([G.attach_prompts(G.read(p / "rows.parquet"), G.read(p / "prompts.parquet"))
                      for p in pools], ignore_index=True)
    rows = rows[rows["lang"] == "en"]
    if rows["prompt"].duplicated().any():
        raise ValueError(f"{int(rows['prompt'].duplicated().sum())} prompts in more than one "
                         "pool: the pools must hold disjoint rows")
    keys = G.tipped_keys(*[G.read(r / "selected.parquet") for r in [*pools, *topups]])
    rows = G.mark_tipped(rows, keys)

    def counts(root: Path) -> dict:
        chunks = sorted((root / "raw" / "samples").glob("chunk-*.parquet"))
        frame = pd.concat([G.read(c, ["prompt", "raw_output"]) for c in chunks])
        return G.sample_hire_counts(frame[frame["prompt"].isin(set(rows["prompt"]))])

    # The k=32 top-up first: its hire rows' rates rest on four times the samples.
    rows = G.attach_sft_rates(rows, *[counts(r) for r in topups], *[counts(p) for p in pools])

    root = Path(resolve_output_path(args.validation_root))
    split_files = sorted(p for p in root.rglob("*validation.parquet") if out not in p.parents)
    heldout = G.heldout_candidates([G.read(p, ["candidate_id"]) for p in split_files])
    train_pool, held_pool = G.split_by_candidate(rows, heldout)
    log.info("%d held-out candidates from %d validation splits: %d rows held out, %d left",
             len(heldout), len(split_files), len(held_pool), len(train_pool))

    if args.monitor_per_bin > 0:
        write_monitor(held_pool, args, out, len(heldout))
        if args.monitor_only:
            return

    n_train = (2 * G.informative_half(train_pool) if args.n_train == "auto"
               else int(args.n_train))
    if args.balance == "signal":
        train, plan = G.signal_balanced_train(train_pool, n_train, args.seed, fill=args.fill)
    else:
        train, plan = G.balanced_train(train_pool, n_train, args.tipped_share, args.seed)
    validation = G.balanced_validation(held_pool, args.n_validation, args.seed)
    assert_no_leakage(train, "grpo_train")
    assert_no_leakage(validation, "grpo_validation")
    overlap = set(train["candidate_id"].astype(str)) & set(validation["candidate_id"].astype(str))
    if overlap:
        raise AssertionError(f"{len(overlap)} candidates in both GRPO splits")

    out.mkdir(parents=True, exist_ok=True)
    train[G.COLUMNS].to_parquet(out / "grpo_train.parquet", index=False)
    validation[G.COLUMNS].to_parquet(out / "grpo_validation.parquet", index=False)
    report = {
        "inputs": {"pool": args.pool, "hire_topup": args.hire_topup,
                   "validation_splits": [str(p.relative_to(root)) for p in split_files]},
        "settings": {"n_train": n_train, "n_train_arg": args.n_train, "balance": args.balance,
                     "fill": args.fill if args.balance == "signal" else None,
                     "tipped_share": args.tipped_share if args.balance == "tipped" else None,
                     "n_validation": args.n_validation, "seed": args.seed},
        "pool": G.describe(rows),
        "heldout": {"candidates": len(heldout), "rows": len(held_pool)},
        "draw": plan,
        "train": G.describe(train),
        "train_pull": G.side_pull(train),
        "validation": G.describe(validation),
        "leakage": "assert_no_leakage passed on both splits; no candidate in both",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "samples.md").write_text(G.samples_markdown(train, seed=args.seed))
    log.info("wrote %s: %d train, %d validation prompts", out, len(train), len(validation))
    print(json.dumps({k: report[k] for k in ("draw", "train_pull", "train")}, indent=2))


if __name__ == "__main__":
    main()
