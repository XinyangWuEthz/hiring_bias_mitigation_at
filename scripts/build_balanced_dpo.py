"""Builds the round-2 DPO set: covert-negative pools merged and balanced by error mass.

Four stages, each cached on the model drive, so one can be re-run without paying for the
ones before it:

    # 1. CPU, seconds -- gather and filter the pools
    python scripts/build_balanced_dpo.py candidates \\
        --round1-train <round1_train.parquet> --round1-validation <round1_validation.parquet> \\
        --extra <hire_surplus.parquet>
    # 2. GPU -- LLM judge for paraphrased attribute references, on the legacy pairs
    python scripts/build_balanced_dpo.py judge
    # 3. GPU -- the SFT student's decision margin on every prompt still in play
    python scripts/build_balanced_dpo.py score --student outputs/sft/qwen3.5-9b_en_only
    # 4. CPU, seconds -- merge, split by candidate, balance, write
    python scripts/build_balanced_dpo.py build

Inputs. Round-1 and extra pools are parquet, JSON Lines, JSON or CSV with at least `prompt`,
`chosen`, `rejected`, `candidate_id` and `job_id`; `chosen`/`rejected` are the JSON responses
(`{"decision": ..., "feedback": ...}`). Group, attribute and condition are read off the prompt
when a pool does not carry them. The legacy pool is the `semisynthetic-v1` `dpo` split, from
the artifacts directory or `--legacy-hub`.

Output, next to the other splits so a training config reads it with `data_kind: dpo_round2`:

    dpo_round2_train.parquet, dpo_round2_validation.parquet, dpo_round2_report.json

What each stage keeps and drops, and why the balance is on error mass rather than on count
alone: `src/hiring_bias_mitigation/generation/balanced_dpo.py`.

V3 balances each pool source on its own (`build --balance-by pool_source`, docs/V3_NEW_DATA.md
round 2); its pools carry `pool_source`, which `normalise_pairs` keeps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.generation import balanced_dpo as BD  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    REPO_ROOT,
    load_yaml,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("build_balanced_dpo")

TEACHER_CONFIG = REPO_ROOT / "configs" / "generation" / "teacher.yaml"


def stage_paths(root: Path, name: str) -> dict[str, Path]:
    return {
        "candidates": root / f"{name}_candidates.parquet",
        "candidates_report": root / f"{name}_candidates_report.json",
        "judge": root / f"{name}_judge.parquet",
        "judge_sample": root / f"{name}_judge_sample.csv",
        "scores": root / f"{name}_scores.parquet",
        "train": root / f"{name}_train.parquet",
        "validation": root / f"{name}_validation.parquet",
        "report": root / f"{name}_report.json",
    }


def _json(payload: dict) -> str:
    def native(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"not JSON serialisable: {type(value).__name__}")

    return json.dumps(payload, indent=2, ensure_ascii=False, default=native)


def _require(path: Path, stage: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} is missing -- run the `{stage}` stage first.")
    return pd.read_parquet(path)


# ---- 1. candidates ------------------------------------------------------------------------

def load_legacy(args, root: Path) -> dict[str, pd.DataFrame]:
    if args.legacy_hub:
        from datasets import load_dataset

        log.info("loading the dpo subset of %s", args.legacy_hub)
        ds = load_dataset(args.legacy_hub, "dpo")
        return {split: ds[split].to_pandas() for split in ("train", "validation") if split in ds}
    folder = Path(resolve_output_path(args.legacy_dir)) if args.legacy_dir else root
    out = {}
    for split in ("train", "validation"):
        path = folder / f"dpo_{split}.parquet"
        if path.exists():
            out[split] = pd.read_parquet(path)
        elif split == "train":
            raise SystemExit(f"{path} not found. Point --legacy-dir at the semisynthetic-v1 "
                             "artifacts, use --legacy-hub, or pass --no-legacy.")
    return out


def cmd_candidates(args, paths: dict[str, Path], root: Path) -> None:
    frames, report = [], {}

    def add(frame: pd.DataFrame, source: str, split: str) -> None:
        normalised, report[f"{source}/{split}"] = BD.normalise_pairs(frame, source, split,
                                                                     args.lang)
        frames.append(normalised)

    if args.round1_train:
        add(BD.read_table(args.round1_train), BD.ROUND1, "train")
    if args.round1_validation:
        add(BD.read_table(args.round1_validation), BD.ROUND1, "validation")
    if not args.round1_train:
        log.warning("no --round1-train: building without the round-1 pairs")
    for extra in args.extra:
        add(BD.read_table(extra), f"{BD.EXTRA_PREFIX}{Path(extra).stem}", "train")
    if not args.no_legacy:
        for split, frame in load_legacy(args, root).items():
            kept, report[f"{BD.LEGACY}/{split}/filter"] = BD.filter_legacy(frame, args.lang)
            add(kept, BD.LEGACY, split)
    if not frames:
        raise SystemExit("no pools given")

    candidates = BD.add_keys(pd.concat(frames, ignore_index=True))
    candidates["needs_judge"] = candidates["source"] == BD.LEGACY
    candidates.to_parquet(paths["candidates"], index=False)
    summary = (candidates.groupby(["source", "split", "decision"]).size()
               .unstack(fill_value=0).reset_index())
    report["candidates"] = summary.to_dict(orient="records")
    paths["candidates_report"].write_text(_json(report), encoding="utf-8")
    print(_json(report))
    print(f"\nwrote {paths['candidates']} ({len(candidates):,} pairs)")


# ---- 2. judge -----------------------------------------------------------------------------

def cmd_judge(args, paths: dict[str, Path]) -> None:
    from hiring_bias_mitigation.eval import backends as BK

    candidates = _require(paths["candidates"], "candidates")
    todo = candidates if args.all_sources else candidates[candidates["needs_judge"]]
    todo = todo.drop_duplicates("pair_sha1")

    cache = pd.read_parquet(paths["judge"]) if paths["judge"].exists() else pd.DataFrame()
    if not cache.empty:
        others = set(cache["judge_model"]) - {args.judge_model}
        if others and not args.force:
            raise SystemExit(f"{paths['judge']} holds verdicts from {sorted(others)}; pass "
                             "--force to discard them and judge again with this model.")
        cache = cache[cache["judge_model"] == args.judge_model]
        todo = todo[~todo["pair_sha1"].isin(cache["pair_sha1"])]
    if todo.empty:
        print("every pair that needs a verdict has one")
        return

    prompts = []
    for row in todo.itertuples(index=False):
        for side in ("chosen", "rejected"):
            feedback = parse_output(getattr(row, side), row.lang).feedback
            prompts.append(BD.judge_prompt(feedback, row.protected_group, row.protected_attr))
    log.info("judging %d rationales (%d pairs) with %s", len(prompts), len(todo),
             args.judge_model)

    backend = BK.build_backend({
        "backend": "vllm", "model": args.judge_model,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs,
        "tensor_parallel_size": args.tensor_parallel_size,
        "engine_kwargs": ({"gdn_prefill_backend": args.gdn_prefill_backend}
                          if args.gdn_prefill_backend else None),
        "chat_template_kwargs": {"enable_thinking": False},
        "generation": {"max_new_tokens": 32, "temperature": 0.0, "top_p": 1.0,
                       "seed": args.seed},
    })
    try:
        raw = backend.generate(prompts)
    finally:
        backend.close()

    verdicts = [BD.parse_judge(r) for r in raw]
    fresh = pd.DataFrame({
        "pair_sha1": todo["pair_sha1"].to_numpy(),
        "source": todo["source"].to_numpy(),
        "verdict_chosen": verdicts[0::2], "verdict_rejected": verdicts[1::2],
        "raw_chosen": raw[0::2], "raw_rejected": raw[1::2],
        "judge_model": args.judge_model,
    })
    judged = pd.concat([cache, fresh], ignore_index=True)
    judged.to_parquet(paths["judge"], index=False)

    # A sample to read by hand: the judge is new, and its flag rate means little until
    # someone has checked what it flags.
    shown = fresh.merge(todo[["pair_sha1", "protected_group", "protected_attr", "chosen",
                              "rejected"]], on="pair_sha1")
    shown["flagged"] = ~(shown["verdict_chosen"].eq("clean")
                         & shown["verdict_rejected"].eq("clean"))
    sample = pd.concat([
        part.sample(n=min(len(part), 50), random_state=args.seed)
        for _, part in shown.groupby("flagged")
    ])
    sample[["flagged", "verdict_chosen", "verdict_rejected", "protected_group",
            "protected_attr", "chosen", "rejected"]].to_csv(paths["judge_sample"], index=False)

    flagged = judged.assign(flagged=~(judged["verdict_chosen"].eq("clean")
                                      & judged["verdict_rejected"].eq("clean")))
    print(_json({
        "pairs_judged": len(judged),
        "flagged_share_by_source": flagged.groupby("source")["flagged"].mean().round(4).to_dict(),
        "unreadable_verdicts": int((judged[["verdict_chosen", "verdict_rejected"]]
                                    == "unreadable").sum().sum()),
    }))
    print(f"\nwrote {paths['judge']}; a sample to check by hand: {paths['judge_sample']}")


# ---- 3. score -----------------------------------------------------------------------------

def cmd_score(args, paths: dict[str, Path]) -> None:
    import torch

    from hiring_bias_mitigation.eval.margins import decision_margins
    from hiring_bias_mitigation.mitigation.training_common import load_model_and_tokenizer

    candidates = _require(paths["candidates"], "candidates")
    if paths["judge"].exists():
        verdicts = pd.read_parquet(paths["judge"])
        flagged = verdicts.loc[~(verdicts["verdict_chosen"].eq("clean")
                                 & verdicts["verdict_rejected"].eq("clean")), "pair_sha1"]
        candidates = candidates[~candidates["pair_sha1"].isin(set(flagged))]

    local = Path(resolve_output_path(args.student))
    student = str(local) if local.exists() else args.student
    cache = pd.read_parquet(paths["scores"]) if paths["scores"].exists() else pd.DataFrame()
    if not cache.empty:
        others = set(cache["student"]) - {student}
        if others and not args.force:
            raise SystemExit(f"{paths['scores']} holds scores from {sorted(others)}; pass "
                             "--force to discard them and score with this student.")
        cache = cache[cache["student"] == student]
    todo = candidates.drop_duplicates("prompt_sha1")
    if not cache.empty:
        todo = todo[~todo["prompt_sha1"].isin(cache["prompt_sha1"])]

    if todo.empty:
        print("every candidate prompt already has a score")
    else:
        log.info("scoring %d prompts with %s", len(todo), student)
        model, tokenizer = load_model_and_tokenizer(
            {"base_model": student, "dtype": args.dtype}, for_training=False
        )
        if torch.cuda.is_available():
            model.to("cuda")
        margins = decision_margins(model, tokenizer, todo["prompt"].tolist(), lang=args.lang,
                                   batch_size=args.batch_size)
        fresh = pd.DataFrame({"prompt_sha1": todo["prompt_sha1"].to_numpy(),
                              "student_margin": margins, "student": student})
        cache = pd.concat([cache, fresh], ignore_index=True)
        cache.to_parquet(paths["scores"], index=False)
        print(f"wrote {paths['scores']} ({len(cache):,} prompts)")

    if cache.empty:
        print("no candidate prompts left to score")
        return
    scored = BD.with_student_error(candidates.merge(
        cache[["prompt_sha1", "student_margin"]], on="prompt_sha1"))
    look = scored.groupby(["source", "split", "decision"]).agg(
        pairs=("p_wrong", "size"), error_mass=("p_wrong", "sum"),
        student_wrong=("p_wrong", lambda p: int((p > 0.5).sum())))
    print(look.round(1).to_string())


# ---- 4. build -----------------------------------------------------------------------------

def cmd_build(args, paths: dict[str, Path]) -> None:
    candidates = _require(paths["candidates"], "candidates")
    scores = _require(paths["scores"], "score")
    judge = pd.read_parquet(paths["judge"]) if paths["judge"].exists() else None

    train, validation, report = BD.assemble(
        candidates, scores, judge, tolerance=args.tolerance, min_keep=args.min_keep,
        seed=args.seed, require_judge=not args.skip_judge, balance_by=args.balance_by,
    )
    train.to_parquet(paths["train"], index=False)
    validation.to_parquet(paths["validation"], index=False)
    report["parameters"] = {"tolerance": args.tolerance, "min_keep": args.min_keep,
                            "seed": args.seed, "balance_by": args.balance_by,
                            "students": sorted(set(scores["student"]))}
    paths["report"].write_text(_json(report), encoding="utf-8")

    balance = report["balance"]
    groups = balance["groups"] if args.balance_by else {"all": balance}
    for name, group in groups.items():
        print(f"\n{args.balance_by or 'pairs'} {name}")
        if "left_out" in group:
            print(f"  LEFT OUT: {group['left_out']}")
            continue
        print("            pairs   error mass   student wrong")
        for stage in ("before", "after"):
            for side, row in group[stage].items():
                print(f"{stage:>7} {side:<7} {row['pairs']:>6,}   {row['error_mass']:>9.1f}"
                      f"   {row['student_wrong']:>8,}")
        trimmed = group["trimmed_most_confidently_wrong"]
        print(f"trimmed {trimmed['pairs']} {trimmed['side']}-side pairs the student was most "
              f"confidently wrong on (p_wrong >= {trimmed['p_wrong_at_least']})")
    print(f"\ntrain {len(train):,} pairs ({report['steps_per_epoch_at_effective_batch_16']} "
          f"steps per epoch at batch 16), validation {len(validation):,}")
    print(f"wrote {paths['train']}, {paths['validation']}, {paths['report']}")


def main() -> None:
    teacher = load_yaml(TEACHER_CONFIG) if TEACHER_CONFIG.exists() else {}

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--artifacts-dir", default="artifacts/semisynthetic-v1")
    common.add_argument("--name", default="dpo_round2",
                        help="file prefix; the training config's data_kind")
    common.add_argument("--lang", default="en")
    common.add_argument("--seed", type=int, default=42)

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("candidates", parents=[common], help="gather and filter the pools")
    c.add_argument("--round1-train")
    c.add_argument("--round1-validation")
    c.add_argument("--extra", action="append", default=[],
                   help="another training pool, e.g. the hire-resampling surplus; repeatable")
    c.add_argument("--legacy-dir", help="folder holding dpo_train/dpo_validation.parquet; "
                                        "defaults to --artifacts-dir")
    c.add_argument("--legacy-hub", help="read the legacy dpo subset from this Hub dataset")
    c.add_argument("--no-legacy", action="store_true")

    j = sub.add_parser("judge", parents=[common], help="LLM judge for paraphrased references")
    j.add_argument("--judge-model", default=teacher.get("teacher_model", "Qwen/Qwen3.5-9B"))
    j.add_argument("--gpu-memory-utilization", type=float,
                   default=float(teacher.get("gpu_memory_utilization", 0.8)))
    j.add_argument("--max-model-len", type=int, default=4096)
    j.add_argument("--max-num-seqs", type=int, default=teacher.get("max_num_seqs"))
    j.add_argument("--tensor-parallel-size", type=int,
                   default=int(teacher.get("tensor_parallel_size", 1)))
    j.add_argument("--gdn-prefill-backend",
                   help="vLLM's Qwen3.5 linear-attention prefill kernel; `triton` on nodes "
                        "without a CUDA toolkit (Daint), where FlashInfer cannot JIT-compile")
    j.add_argument("--all-sources", action="store_true",
                   help="judge every pool, not only the legacy pairs")
    j.add_argument("--force", action="store_true")

    s = sub.add_parser("score", parents=[common], help="student decision margins")
    s.add_argument("--student", default="outputs/sft/qwen3.5-9b_en_only",
                   help="the DPO starting point: an SFT adapter directory or a model id")
    s.add_argument("--batch-size", type=int, default=4,
                   help="1 avoids padding altogether, at about half the speed")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--force", action="store_true")

    b = sub.add_parser("build", parents=[common], help="merge, split, balance, write")
    b.add_argument("--tolerance", type=float, default=0.05,
                   help="largest relative gap allowed between the two sides' error mass")
    b.add_argument("--min-keep", type=float, default=0.5,
                   help="fail rather than keep less than this share of the smaller side")
    b.add_argument("--skip-judge", action="store_true",
                   help="build without LLM-judge verdicts on the legacy pairs")
    b.add_argument("--balance-by",
                   help="balance within each value of this column, e.g. V3's pool_source; "
                        "validation is then balanced by count per value and protected group")

    args = parser.parse_args()
    require_output_root()
    root = Path(resolve_output_path(args.artifacts_dir))
    root.mkdir(parents=True, exist_ok=True)
    paths = stage_paths(root, args.name)
    if args.stage == "candidates":
        cmd_candidates(args, paths, root)
    elif args.stage == "judge":
        cmd_judge(args, paths)
    elif args.stage == "score":
        cmd_score(args, paths)
    else:
        cmd_build(args, paths)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
