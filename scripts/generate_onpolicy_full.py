"""On-policy v2 negatives over v1's whole English pool, sampled on all four GPUs of a node.

    CFG=configs/generation/negatives_v2_onpolicy_full_gh200.yaml
    python scripts/generate_onpolicy_full.py --config $CFG --stage prepare
    # then scripts/sample_student.py once per GPU (scripts/cscs/onpolicy_full.sbatch does both)
    python scripts/generate_onpolicy_full.py --config $CFG --stage collect
    python scripts/generate_onpolicy_full.py --config $CFG --stage judge
    python scripts/generate_onpolicy_full.py --config $CFG --stage balance   # or merge (V3)

  prepare  every pool row with v1's chosen side (v1's own filter), one attribute-free prompt per
           pair and one attribute-bearing prompt per row in prompts.parquet, and the merged
           student checkpoint (reused when already merged)
  collect  reads every sample chunk, refuses to continue while one is missing, and selects one
           covert, attribute-tipped mistake per row (generation/onpolicy.py)
  judge    the teacher judges the selected negatives and their chosen side; the survivors
           become candidates, a candidate-level train/validation split (v1's held-out
           candidates stay held out), report.json and samples.md
  balance  equal hire and reject reference verdicts per protected group inside each split,
           written to balanced/ with balance.json (the pool is 86% reject; spec D6). The DPO
           data config reads balanced/
  merge    a top-up run, or a run with `merge_with`: its pairs merged into the base run's (or
           `merge_with`'s) inside each split, not balanced, written to merged/ with
           merge.json. V3 balances them by error mass per pool source afterwards
           (scripts/build_balanced_dpo.py), which is r2em's candidate set: the count balance
           above would drop pairs that balance should choose from

A top-up run (`row_filter` + `topup_of` in its config) samples one slice of the pool again
with more samples per prompt, e.g. the hire rows at k=32 because the pool is 86% reject. Its
train/validation split comes from the base run's rows, so a candidate is on the same side in
both runs, and its balance stage merges the base run's pairs with its own before balancing.

A run on new pairs of the same pool (`merge_with` in its config, V3's round 3) samples rows no
earlier run has, from its own source directory, and its merge joins its pairs to the merged
pairs `merge_with` names. Both must come from the same merged student, and no candidate may
end up on both sides of the split.

An attribute-coverage run (`add_attributes: {group, k, seed}`) samples only new rows: `k` more
attributes of one group per pair, with the pair's own chosen text reused
(onpolicy.add_attribute_rows). Its `split_of` names the run whose rows.parquet (the whole pool)
sets the train/validation split, so its candidates sit on the same side as in that run.

`val_fraction: 0` holds out exactly the candidates of the source's validation files, with no
seeded draw on top (`validation_ids`). On v1's pool the draw is v1's own split and adds nothing;
a pool split some other way, such as V3's new pairs, needs the 0.

Only `judge_strategies` (default: student_tipped) are judged and kept. Rows the student also
gets wrong without the attribute are counted in sample_report.json but are not
attribute-driven, so they do not train invariance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.generation import negatives_v2 as N  # noqa: E402
from hiring_bias_mitigation.generation import onpolicy as OP  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    load_config,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("generate_onpolicy_full")

V1_VALIDATION_FILES = ("sft_validation.parquet", "dpo_validation.parquet")


def _json(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=N._json_default)


def _source(cfg: dict) -> Path:
    return Path(resolve_output_path(cfg["source_artifacts_dir"]))


def _topup_base(cfg: dict) -> Path | None:
    return Path(resolve_output_path(cfg["topup_of"])) if cfg.get("topup_of") else None


def _merge_base(cfg: dict) -> Path | None:
    """The pairs this run's are merged into: its top-up base run's, or `merge_with`'s."""
    if cfg.get("topup_of") and cfg.get("merge_with"):
        raise SystemExit("merge: a config takes `topup_of` or `merge_with`, not both")
    if cfg.get("merge_with"):
        return Path(resolve_output_path(cfg["merge_with"]))
    return _topup_base(cfg)


def _student_of(run: Path) -> str | None:
    """The merged student a run sampled, from its prepare.json or its parent run's."""
    for path in (run / "prepare.json", run.parent / "prepare.json"):
        if path.exists():
            return json.loads(path.read_text()).get("merged_checkpoint")
    return None


def _source_label(cfg: dict) -> str:
    added = cfg.get("add_attributes")
    if added:
        return (f"v1 en pool + {added['k']} more {added['group']} attributes per pair "
                "(chosen text reused), student samples")
    return "v1 en pool, student samples"


def _split_rows(cfg: dict, rows: pd.DataFrame) -> pd.DataFrame:
    """The rows the train/validation split is drawn from: `split_of`'s, else the top-up base's,
    else this run's own. A run over a slice or an extension of the pool must use the pool's."""
    base = cfg.get("split_of") or cfg.get("topup_of")
    return pd.read_parquet(Path(resolve_output_path(base)) / "rows.parquet") if base else rows


def stage_prepare(cfg: dict, out: Path) -> None:
    lang = str(cfg.get("lang", "en"))
    invariant = pd.read_parquet(_source(cfg) / "raw" / f"{lang}_invariant.parquet")
    rows, invariant_report = OP.pool_rows(invariant, lang)
    rows = OP.filter_rows(rows, cfg.get("row_filter")) # 35,826 rows, 2,992 pairs, 85.8% reject -> ITERATIVE_DPO.md
    added = None
    if cfg.get("add_attributes"):
        rows, added = OP.add_attribute_rows(rows, lang=lang, **cfg["add_attributes"])
    free = OP.pair_free_prompts(rows)
    prompts = pd.DataFrame({
        "kind": ["attr_free"] * len(free) + ["attributed"] * len(rows),
        "key": [str(p) for p in free["pair_id"]] + [str(i) for i in range(len(rows))],
        "prompt": free["prompt"].tolist() + OP.attributed_prompts(rows),
    })
    rows.to_parquet(out / "rows.parquet", index=False)
    existing = out / "prompts.parquet"
    unchanged = (existing.exists()
                 and pd.read_parquet(existing)["prompt"].tolist() == prompts["prompt"].tolist())
    if not unchanged: # Resume from checkpoint
        prompts.to_parquet(existing, index=False) 
    model = OP.merged_student(cfg)
    info = {"merged_checkpoint": str(model), "row_filter": cfg.get("row_filter"),
            "rows": len(rows), "pairs": len(free),
            "prompts": len(prompts), "invariant_filter": invariant_report,
            "add_attributes": added}
    (out / "prepare.json").write_text(_json(info), encoding="utf-8")
    log.info("prepared %d rows over %d pairs: %d prompts", len(rows), len(free), len(prompts))


def stage_collect(cfg: dict, out: Path) -> None:
    rows = pd.read_parquet(out / "rows.parquet")
    prompts = pd.read_parquet(out / "prompts.parquet")
    k = int(cfg.get("samples_per_prompt", 8))
    samples = OP.read_samples(out / "raw" / "samples", prompts["prompt"].tolist(),
                              int(cfg.get("chunk_size", 512)), k)
    is_free = (prompts["kind"] == "attr_free").tolist()
    free_by_pair = {key: s for key, s, f in zip(prompts["key"], samples, is_free) if f}
    attributed = [s for s, f in zip(samples, is_free) if not f]
    selected, selection = OP.select_negatives(
        rows, attributed, OP.expand_free(rows, free_by_pair), lang=str(cfg.get("lang", "en")),
        min_free_right=float(cfg.get("min_free_right", 0.5)),
        max_length_delta=int(cfg.get("max_length_delta_words", 10)))
    selected.to_parquet(out / "selected.parquet", index=False)
    info = json.loads((out / "prepare.json").read_text())
    report = {"student": f"{cfg['student_base']} + {cfg['student_adapter']} (merged)",
              "merged_checkpoint": info["merged_checkpoint"], "samples_per_prompt": k,
              "student_generation": cfg.get("student_generation", {}), "selection": selection}
    if "pool_source" in rows.columns:   # V3: each pool source is balanced and checked alone
        report["by_pool_source"] = OP.selection_by(rows, attributed, free_by_pair, selected,
                                                   "pool_source", str(cfg.get("lang", "en")))
    (out / "sample_report.json").write_text(_json(report), encoding="utf-8")
    print(_json(selection))
    if "by_pool_source" in report:
        print(_json(report["by_pool_source"]))


def v1_validation_candidates(source: Path) -> set[str]:
    ids: set[str] = set()
    for name in V1_VALIDATION_FILES:
        path = source / name
        if path.exists():
            ids |= set(pd.read_parquet(path, columns=["candidate_id"])["candidate_id"].astype(str))
    return ids


#: Row columns a judged pair keeps beside build_pairs' own: the sampler's counts, the attribute
#: an attribute-coverage row's chosen text was written for, and V3's pool source, by which V3's
#: pairs are balanced and checked (docs/V3_NEW_DATA.md).
CARRIED_COLUMNS = ("free_right_rate", "sample_index", "n_wrong_samples", "source_attr",
                   "pool_source")


def carry_row_columns(pairs: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    """Copies CARRIED_COLUMNS from the judged rows onto their pairs, row for row."""
    for column in CARRIED_COLUMNS:
        if column in kept.columns:
            pairs[column] = kept[column].values
    return pairs


def judged_by(selected: pd.DataFrame, candidates: pd.DataFrame, is_val: pd.Series,
              by: str) -> dict:
    """Per value of `by`: negatives sent to the judge, pairs it kept, their reference hire
    share and their split. V3 reads each pool source's judge stage on its own (D7)."""
    out = {}
    for value, sent in selected.groupby(by, sort=True):
        mine = (candidates[by] == value).to_numpy()
        kept, val = candidates[mine], is_val.to_numpy()[mine]
        out[str(value)] = {
            "selected": len(sent), "kept": len(kept),
            "keep_rate": round(len(kept) / len(sent), 4),
            "kept_reference_hire_share": (round(float((kept["reference_decision"] == "hire")
                                                      .mean()), 4) if len(kept) else None),
            "train": int((~val).sum()), "validation": int(val.sum())}
    return out


def validation_ids(cfg: dict, rows: pd.DataFrame) -> set[str]:
    """The candidates the judged pairs hold out.

    Every candidate the source's validation files hold out, plus a seeded `val_fraction` of
    the split rows' candidates. On v1's pool that draw is v1's own split (the same function,
    seed and rows), so it adds nothing. V3's new pairs were split per pool source, so the same
    draw would add about 5% of the candidates SFT-v3 trained on and take them out of DPO's
    training split. `val_fraction: 0` skips the draw: DPO then holds out exactly the
    candidates SFT held out.
    """
    pinned = v1_validation_candidates(_source(cfg))
    fraction = float(cfg.get("val_fraction", 0.05))
    if fraction <= 0:
        if not pinned:
            raise ValueError(f"val_fraction 0 but {_source(cfg)} has none of "
                             f"{V1_VALIDATION_FILES}: no candidate would be held out")
        return pinned
    return N.validation_candidates(_split_rows(cfg, rows), pinned, fraction,
                                   int(cfg.get("seed", 42)))


def stage_judge(cfg: dict, out: Path) -> None:
    rows = pd.read_parquet(out / "rows.parquet")
    sample_report = json.loads((out / "sample_report.json").read_text())
    strategies = list(cfg.get("judge_strategies", [OP.TIPPED]))
    selected = pd.read_parquet(out / "selected.parquet")
    selected = selected[selected["rejected_strategy"].isin(strategies)].reset_index(drop=True)
    min_plausibility = int(cfg.get("min_plausibility", 3))
    judge_generation = cfg.get("judge_generation")
    log.info("judging %d selected negatives and their chosen side", len(selected))

    # teacher generate choosen,chosen 通过率（teacher 自己写的）0.9, rejected 通过率（学生写的）hire ~0.6
    # teacher model not biased towards 自己写的 -> JUDGE_AUTHOR_CONTROL.md
    teacher = BK.build_backend(N.teacher_backend_config(cfg))
    try:
        rejected_raw = N._generate_cached(out / "raw" / "student_judge.parquet",  # 断点续跑
                                          N.judge_prompts(selected, "rejected"), teacher,
                                          judge_generation, selected)
        chosen_raw = N._generate_cached(out / "raw" / "chosen_judge.parquet",
                                        N.judge_prompts(selected, "chosen"), teacher,
                                        judge_generation, selected)
    finally:
        teacher.close()
    judged = N.attach_judgements(selected, rejected_raw, "rejected")
    judged = N.attach_judgements(judged, chosen_raw, "chosen")

    report: dict = {"generator": sample_report["student"], "judge": cfg["teacher_model"],
                    "sampling": sample_report, "strategies": {}}
    all_pairs = []
    for strategy in strategies:
        part = judged[judged["rejected_strategy"] == strategy].reset_index(drop=True)
        entry: dict = {"n_prompts": len(rows), "n_selected": len(part),
                       "covertness": sample_report["selection"].get("wrong_samples_raw", {}),
                       "filters": {"negatives": sample_report["selection"].get("filters", {})},
                       "judge": N._judge_summary(part) if len(part) else {}}
        kept, entry["filters"]["judge"] = N.filter_by_judge(part, "rejected", strategy,
                                                            min_plausibility)
        kept, entry["filters"]["chosen_judge"] = N.filter_by_judge(kept, "chosen", None,
                                                                   min_plausibility)
        entry["filters"]["chosen_judge_dropped"] = entry["filters"]["chosen_judge"]["n_input"] \
            - entry["filters"]["chosen_judge"]["n_kept"]
        pairs = N.build_pairs(kept, cfg["teacher_model"], _source_label(cfg))
        if len(pairs):
            pairs["rejected_generator"] = sample_report["student"]
            carry_row_columns(pairs, kept)
        pairs.to_parquet(out / f"candidates_{strategy}.parquet", index=False)
        entry.update({"n_kept": len(pairs), "yield": len(pairs) / len(rows) if len(rows) else 0.0,
                      "separability": N.separability(pairs), "pairs": N._pair_summary(pairs)})
        report["strategies"][strategy] = entry
        all_pairs.append(pairs)

    candidates = pd.concat(all_pairs, ignore_index=True)
    candidates.to_parquet(out / "dpo_v2_candidates.parquet", index=False)
    val_ids = validation_ids(cfg, rows)
    is_val = candidates["candidate_id"].astype(str).isin(val_ids)
    candidates[~is_val].to_parquet(out / "dpo_v2_train.parquet", index=False)
    candidates[is_val].to_parquet(out / "dpo_v2_validation.parquet", index=False)
    report["splits"] = {"dpo_v2_train": int((~is_val).sum()),
                        "dpo_v2_validation": int(is_val.sum()),
                        "validation_candidates": len(val_ids)}
    if "pool_source" in candidates.columns:
        report["by_pool_source"] = judged_by(selected, candidates, is_val, "pool_source")
    (out / "report.json").write_text(_json(report), encoding="utf-8")
    N.write_samples(out / "samples.md", candidates, int(cfg.get("n_samples", 20)))
    for strategy, entry in report["strategies"].items():
        print(f"  {strategy:15s} kept {entry['n_kept']:5d} of {entry['n_selected']} selected | "
              f"balanced AUC {entry['separability']['auc_balanced']}")
    print(f"  splits: {report['splits']}")
    for source, entry in report.get("by_pool_source", {}).items():
        print(f"  {source:15s} {entry}")


def stage_balance(cfg: dict, out: Path) -> None:
    target = out / "balanced"
    target.mkdir(exist_ok=True)
    seed = int(cfg.get("seed", 42))
    base = _topup_base(cfg)
    info, parts = {}, []
    for split in ("train", "validation"):
        frame = pd.read_parquet(out / f"dpo_v2_{split}.parquet")
        merged = None
        if base:
            frame, merged = OP.merge_topup(pd.read_parquet(base / f"dpo_v2_{split}.parquet"),
                                           frame)
        balanced, info[split] = OP.balance_by_reference(frame, seed)
        if merged:
            info[split]["topup"] = merged
        balanced.to_parquet(target / f"dpo_v2_{split}.parquet", index=False)
        parts.append(balanced)
    candidates = pd.concat(parts, ignore_index=True)
    candidates.to_parquet(target / "dpo_v2_candidates.parquet", index=False)
    info["candidates"] = {"n": len(candidates),
                          "hire_share": float((candidates["reference_decision"] == "hire").mean())}
    (target / "balance.json").write_text(_json(info), encoding="utf-8")
    print(_json(info))


def stage_merge(cfg: dict, out: Path) -> None:
    """The top-up's pairs merged into its base run's inside each split, not balanced.

    r2em balanced by error mass over r1 and the surplus r1's count balance dropped, which
    together are every merged pair; V3 hands the merged pairs to scripts/build_balanced_dpo.py
    directly (docs/V3_NEW_DATA.md, round 2). Where both runs have a pair for a row, the base
    run's stays (`onpolicy.merge_topup`). With `merge_with` (round 3) the rows differ, so the
    merge is the union; it refuses pairs of another student and a candidate on both sides.
    """
    base = _merge_base(cfg)
    if base is None:
        raise SystemExit("merge: this config has neither `topup_of` nor `merge_with`; there is "
                         "nothing to merge")
    if cfg.get("merge_with"):
        mine, theirs = _student_of(out), _student_of(base)
        if not mine or mine != theirs:
            raise SystemExit(f"merge: {base} was sampled from {theirs}, this run from {mine}; "
                             "pairs of two students are not merged")
    target = out / "merged"
    target.mkdir(exist_ok=True)
    info: dict = {"base": str(base), "topup": str(out)}
    merged_splits = {}
    for split in ("train", "validation"):
        merged, info[split] = OP.merge_topup(pd.read_parquet(base / f"dpo_v2_{split}.parquet"),
                                             pd.read_parquet(out / f"dpo_v2_{split}.parquet"))
        merged_splits[split] = merged
        info[split]["n_merged"] = len(merged)
        if "pool_source" in merged.columns:
            info[split]["by_pool_source_and_reference"] = {
                f"{s} {d}": int(n) for (s, d), n in
                merged.groupby(["pool_source", "reference_decision"]).size().items()}
    if all("candidate_id" in m.columns for m in merged_splits.values()):
        both = set(merged_splits["train"]["candidate_id"]) & set(
            merged_splits["validation"]["candidate_id"])
        if both:
            raise SystemExit(f"merge: {len(both)} candidate(s) on both sides of the split, "
                             f"e.g. {sorted(both)[:3]}")
    for split, merged in merged_splits.items():
        merged.to_parquet(target / f"dpo_v2_{split}.parquet", index=False)
    (target / "merge.json").write_text(_json(info), encoding="utf-8")
    print(_json(info))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("prepare", "collect", "judge", "balance", "merge"),
                        required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_output_root()
    out = Path(resolve_output_path(cfg["artifacts_dir"]))
    (out / "raw").mkdir(parents=True, exist_ok=True)
    stages = {"prepare": stage_prepare, "collect": stage_collect, "judge": stage_judge,
              "balance": stage_balance, "merge": stage_merge}
    stages[args.stage](cfg, out)


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
