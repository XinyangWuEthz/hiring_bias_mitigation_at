"""On-policy negatives: the student's own attribute-tipped mistakes as `rejected`.

Why. The teacher-written v2 negatives (batch-01) are covert -- none names the attribute -- but
a text classifier still tells them from `chosen` at AUC 0.88, because a rationale argued for
the wrong verdict is weaker: it hedges a weak candidate into a hire or nitpicks a strong one.
The audited 9B's OWN wrong-verdict rationales in attribute-tipped sets are not separable from
its right ones (AUC 0.47 over 156 sets): its real covert bias is a confident, grounded
rationale with the decision moved. A negative DPO can learn from has to look like that, and
the one source of exactly that is the student.

Construction, per batch row (one v1 prompt; `chosen` stays v1's teacher text):
  1. k samples from the SFT student on the attribute-bearing audit prompt, and k on the same
     pair's attribute-free prompt;
  2. the row is *attribute-tipped* when the student is mostly right without the attribute
     (free_right_rate > min_free_right), so a wrong answer with it is the attribute's doing;
  3. its wrong samples go through the same hard filters as the teacher's negatives -- flipped,
     no attribute or group named, length within +-N words of chosen, right language, no
     meta-commentary -- and the first survivor in sample order becomes the row's `rejected`;
  4. the teacher then judges it exactly as it judged the teacher-written negatives.

Rows that are wrong but not tipped keep their own strategy name (`student_untipped`), so the
comparison shows whether the tipping requirement matters. Sampling order, not a score, picks
the survivor: the first valid sample is a draw from the student's distribution over its own
covert mistakes, not the most extreme one (cf. "Finding the Sweet Spot", 2025).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from ..data.injection import ATTR_FREE
from ..eval.metrics import mentions_attribute
from ..eval.parsing import parse_output
from ..utils.logging import get_logger
from .dataset import _audit_prompt
from .negatives_v2 import covertness_of_raw, filter_negatives, flip, mentions_group

log = get_logger(__name__)

TIPPED, UNTIPPED = "student_tipped", "student_untipped"
# One pool row: a pair's CV and job with one attribute in one condition.
ROW_KEY = ("pair_id", "protected_attr", "condition")


def attr_free_prompts(batch: pd.DataFrame) -> list[str]:
    """The audit prompt for each row's job and CV with no attribute anywhere."""
    out = []
    for row in batch.to_dict("records"):
        out.append(_audit_prompt({**row, "condition": "attr_free", "protected_attr": ATTR_FREE,
                                  "group_label": "", "implicit_injection": ""}))
    return out


def attributed_prompts(batch: pd.DataFrame) -> list[str]:
    """The audit prompt each row was built from: the CV with its attribute injected."""
    from .dataset import _with_implicit

    return [_audit_prompt(row) for row in _with_implicit(batch).to_dict("records")]


def _decisions(texts: list[str], lang: str) -> list[str]:
    """ returns the decisions of the texts, e.g. ["hire", "reject", ""model refuses to decide","invalid output"] """
    return [parse_output(t, lang).decision for t in texts]


def select_negatives(
    batch: pd.DataFrame,
    attributed: list[list[str]],
    free: list[list[str]],
    lang: str = "en",
    min_free_right: float = 0.5,
    max_length_delta: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """One `rejected` per row from the student's own wrong samples; see the module docstring."""
    rows = batch.reset_index(drop=True).to_dict("records")
    candidates, raw = [], []
    tipped_rows = rows_with_wrong = tipped_with_wrong = 0
    attributed_right, free_right_all = [], []
    for index, (row, samples, free_samples) in enumerate(zip(rows, attributed, free)):
        reference = row["reference_decision"]
        free_decisions = _decisions(free_samples, lang)
        free_right = (sum(d == reference for d in free_decisions) / len(free_decisions)
                      if free_decisions else 0.0)
        tipped = free_right > min_free_right
        decisions = _decisions(samples, lang)
        attributed_right.append(sum(d == reference for d in decisions) / max(1, len(decisions)))
        free_right_all.append(free_right)
        # NOTE might have sampling noise, improve once we have enough data
        wrong = [j for j, d in enumerate(decisions) if d == flip(reference)]
        tipped_rows += tipped
        rows_with_wrong += bool(wrong)
        tipped_with_wrong += bool(wrong) and tipped
        for j in wrong:
            candidates.append({**row, "_row": index, "sample_index": j,
                               "free_right_rate": free_right, "tipped": tipped,
                               "n_wrong_samples": len(wrong), "n_samples": len(samples)})
            raw.append(samples[j])

    report: dict = {
        "rows": len(rows),
        "samples_per_row": len(attributed[0]) if attributed else 0,
        "student_right_rate_with_attribute": (sum(attributed_right) / len(rows)) if rows else 0.0,
        "student_right_rate_without_attribute": (sum(free_right_all) / len(rows)) if rows else 0.0,
        "rows_tipped": tipped_rows,
        "rows_with_a_wrong_sample": rows_with_wrong,
        "rows_tipped_with_a_wrong_sample": tipped_with_wrong,
        "wrong_samples": len(raw),
        "min_free_right": min_free_right,
    }
    if not candidates:
        report.update({"wrong_samples_raw": {}, "filters": {}, "selected": {}})
        return pd.DataFrame(), report

    frame = pd.DataFrame(candidates)
    report["wrong_samples_raw"] = covertness_of_raw(raw, frame)
    feedback = [parse_output(t, lang).feedback for t in raw]
    overt = [mentions_attribute(f, a) or mentions_group(f, g) for f, a, g in
             zip(feedback, frame["protected_attr"], frame["protected_group"])]
    overt_by_cell = (frame.assign(_overt=overt)
                     .groupby(["protected_group", "condition"])["_overt"].mean())
    report["wrong_samples_overt_by_cell"] = {
        f"{g} | {c}": round(float(v), 4) for (g, c), v in overt_by_cell.items()
    }

    kept, report["filters"] = filter_negatives(raw, frame, lang, "student", max_length_delta)
    if kept.empty:
        report["selected"] = {}
        return kept, report
    kept = (kept.sort_values(["_row", "sample_index"])
                .drop_duplicates("_row", keep="first") # NOTE we can try to keep more samples from hire, not 1 for each row
                .reset_index(drop=True))
    kept["rejected_strategy"] = [TIPPED if t else UNTIPPED for t in kept["tipped"]]
    report["selected"] = {k: int(v) for k, v in kept["rejected_strategy"].value_counts().items()}
    log.info("on-policy negatives: %s from %d wrong samples over %d rows",
             report["selected"], len(raw), len(rows))
    return kept.drop(columns=["_row"]), report


def _share(flags: list[bool]) -> float | None:
    return round(sum(flags) / len(flags), 4) if flags else None


def selection_by(rows: pd.DataFrame, attributed: list[list[str]],
                 free_by_pair: dict[str, list[str]], selected: pd.DataFrame, by: str,
                 lang: str = "en") -> dict:
    """The student's decisions and the negatives selected from them, per value of `by`.

    V3 balances and checks each pool source on its own (docs/V3_NEW_DATA.md, round 2). Which
    side a student errs on follows from its operating point, so each source's attribute-free
    hire rate is read beside its negatives; a source whose tipped negatives fall under 2% of
    its rows is reported before the judge runs. `attributed` lines up with `rows`, and
    `free_by_pair` holds each pair's attribute-free samples under its `pair_id` as a string.
    """
    rows = rows.reset_index(drop=True)
    out = {}
    for value, part in rows.groupby(by, sort=True):
        references = part["reference_decision"].tolist()
        with_attr = [(d, ref) for i, ref in zip(part.index, references)
                     for d in _decisions(attributed[i], lang)]
        pair_ref = part.drop_duplicates("pair_id").set_index("pair_id")["reference_decision"]
        free = [(d, ref) for pair, ref in pair_ref.items()
                for d in _decisions(free_by_pair[str(pair)], lang)]
        picked = selected[selected[by] == value] if len(selected) else selected
        strategies = picked["rejected_strategy"].value_counts() if len(picked) else {}
        tipped = picked[picked["rejected_strategy"] == TIPPED] if len(picked) else picked
        out[str(value)] = {
            "rows": len(part), "pairs": len(pair_ref),
            "reference_hire_share": _share([r == "hire" for r in references]),
            "student_hire_rate_attr_free": _share([d == "hire" for d, _ in free]),
            "student_right_rate_attr_free": _share([d == r for d, r in free]),
            "student_hire_rate_with_attribute": _share([d == "hire" for d, _ in with_attr]),
            "student_right_rate_with_attribute": _share([d == r for d, r in with_attr]),
            "student_right_rate_with_attribute_by_reference": {
                side: _share([d == r for d, r in with_attr if r == side])
                for side in ("hire", "reject")},
            "selected": {str(k): int(v) for k, v in dict(strategies).items()},
            "tipped_per_row": round(int(dict(strategies).get(TIPPED, 0)) / len(part), 4),
            # The side each source's error-mass balance will be short of.
            "tipped_by_reference": {side: int((tipped["reference_decision"] == side).sum())
                                    if len(tipped) else 0 for side in ("hire", "reject")},
        }
    return out


def filter_rows(rows: pd.DataFrame, row_filter: dict | list[dict] | None) -> pd.DataFrame:
    """The rows `row_filter` keeps, re-indexed; None keeps all.

    A dict keeps the rows matching every `column: value` of it, and a list value matches any of
    its items. A list of dicts keeps the rows matching any one of them: V3's top-up takes the
    hire rows of djinni_new and kaggle_linkedin and the reject rows of RJDB, each source's short
    side. A top-up run samples one slice of the pool harder (k=32); its prompt keys are positions
    in the filtered rows, which is why the index is reset.
    """
    if not row_filter:
        return rows
    keep = pd.Series(False, index=rows.index)
    for one in (row_filter if isinstance(row_filter, list) else [row_filter]):
        mask = pd.Series(True, index=rows.index)
        for column, value in one.items():
            if isinstance(value, (list, tuple)):
                mask &= rows[column].isin(list(value))
            else:
                mask &= rows[column] == value
        keep |= mask
    return rows[keep].reset_index(drop=True)


#: A rationale with one of these addresses one gender. Reused under another attribute it
#: would contradict the profile, so it is never reused.
GENDERED_PRONOUNS = re.compile(r"\b(?:she|her|hers|herself|he|him|his|himself)\b", re.IGNORECASE)
#: The columns that describe one variant of a pair; every other pool column is the pair's.
VARIANT_COLUMNS = ("protected_group", "protected_attr", "condition", "group_label", "profile",
                   "chosen_decision", "chosen_feedback")


def add_attribute_rows(
    rows: pd.DataFrame,
    group: str,
    k: int,
    seed: int,
    lang: str = "en",
    conditions: tuple[str, ...] = ("explicit", "implicit"),
) -> tuple[pd.DataFrame, dict]:
    """New pool rows for `k` more attributes of `group` per pair, reusing the pair's chosen text.

    v1 draws the same 2 attributes per group for every pair (synth.build_variant_frame), so an
    English gender attribute (1 of 20) has about 600 rows where a military one (1 of 5) has
    about 2,400. The docstring there assumes every attribute "still appears thousands of
    times"; for gender it does not.

    The chosen side needs no new teacher call. The invariant pass pins its verdict to the
    pair's reference and drops any text that names its own attribute (filters.filter_invariant),
    so the pair's existing text for the same group and condition is reused. A text with a
    gendered pronoun is never reused. The reused text then goes through filter_invariant again,
    under the new attribute, exactly as v1's own rows did.

    Draws use their own rng, not v1's seed-42 stream: pairs in sorted order, `k` new
    attributes uniformly among those the pair lacks (sorted by list position), then one source
    text per new attribute and condition. Returns only the rows that pass the filter, with the
    pool's columns plus `source_attr` (whose text was reused), and a report.
    """
    import json

    import numpy as np

    from ..data.protected_groups import GROUP_LABELS, load_group
    from .filters import filter_invariant
    from .synth import build_profile

    attributes = load_group(group, lang).attributes
    rng = np.random.default_rng(seed)
    in_group = rows[rows["protected_group"] == group]
    used = in_group.groupby("pair_id")["protected_attr"].agg(set).to_dict()
    sources = {key: part for key, part in in_group.groupby(["pair_id", "condition"])}
    pair_fields = [c for c in rows.columns if c not in VARIANT_COLUMNS]
    pairs = rows.drop_duplicates("pair_id").set_index("pair_id")

    built, lost = [], {}
    for pid in sorted(pairs.index):
        free = [a for a in attributes if a not in used.get(pid, set())]
        picked = sorted(int(i) for i in rng.choice(len(free), size=min(k, len(free)),
                                                   replace=False))
        base = pairs.loc[pid]
        for index in picked:
            attr = free[index]
            for condition in conditions:
                source = sources.get((pid, condition))
                if source is None:
                    lost["no_source_row_same_condition"] = lost.get(
                        "no_source_row_same_condition", 0) + 1
                    continue
                usable = source[~source["chosen_feedback"].str.contains(GENDERED_PRONOUNS)]
                if usable.empty:
                    lost["source_has_gendered_pronoun"] = lost.get(
                        "source_has_gendered_pronoun", 0) + 1
                    continue
                pick = usable.iloc[int(rng.integers(len(usable)))]
                built.append({
                    **{f: (pid if f == "pair_id" else base[f]) for f in pair_fields},
                    "protected_group": group, "protected_attr": attr, "condition": condition,
                    "group_label": GROUP_LABELS[group][lang],
                    "profile": build_profile(base["cv"], group, attr, condition, lang),
                    "source_attr": pick["protected_attr"],
                    "source_feedback": pick["chosen_feedback"],
                })

    variants = pd.DataFrame(built)
    report: dict = {"group": group, "k": k, "seed": seed,
                    "planned": len(variants) + sum(lost.values()), "lost_before_filter": lost,
                    "built": len(variants)}
    if variants.empty:
        report.update({"kept": 0, "filter": {}})
        return variants, report
    raw = [json.dumps({"decision": d, "feedback": f}, ensure_ascii=False)
           for d, f in zip(variants["reference_decision"], variants["source_feedback"])]
    kept, report["filter"] = filter_invariant(raw, variants, lang)
    kept = kept.drop(columns=["source_feedback"], errors="ignore")
    by_attr = variants.groupby("protected_attr").size()
    kept_by_attr = kept.groupby("protected_attr").size().reindex(by_attr.index, fill_value=0)
    report["kept"] = len(kept)
    report["yield_by_attribute"] = {a: round(float(kept_by_attr[a] / n), 4)
                                    for a, n in by_attr.items()}
    log.info("%s: %d new rows kept of %d planned (k=%d)", group, len(kept), report["planned"], k)
    return kept.reset_index(drop=True), report


def merge_topup(base: pd.DataFrame, topup: pd.DataFrame,
                key: tuple[str, ...] = ROW_KEY) -> tuple[pd.DataFrame, dict]:
    """The base run's pairs, then the top-up run's pairs for rows the base run lacks.

    When both runs kept a pair for the same row, the base run's stays: a row is never counted
    twice, and pairs that were already judged and checked do not change.
    """
    def keys(frame: pd.DataFrame) -> list[tuple]:
        return list(frame[list(key)].astype(str).itertuples(index=False, name=None))

    seen = set(keys(base))
    new = topup[[k not in seen for k in keys(topup)]]
    merged = pd.concat([base, new], ignore_index=True)
    return merged, {"n_base": len(base), "n_topup": len(topup), "n_new": len(new),
                    "n_already_in_base": len(topup) - len(new)}


def balance_by_reference(pairs: pd.DataFrame, seed: int = 42,
                         by: str = "protected_group") -> tuple[pd.DataFrame, dict]:
    """Equal hire and reject reference verdicts within each `by` group: every pair of the
    scarcer verdict, and a seeded sample of the same size from the other.

    v1's English pool is 86% reject, so the full-pool pairs are too: 86% of them teach "reject
    beats hire". DPO can follow that by lowering the hire rate overall, which also shrinks the
    audit's group gaps without teaching invariance (spec, D6). Balancing inside each group keeps
    the group mix the same on both sides. A group lacking one verdict drops out.
    """
    parts, groups = [], {}
    for name, part in pairs.groupby(by, sort=True):
        hire = part[part["reference_decision"] == "hire"]
        reject = part[part["reference_decision"] == "reject"]
        n = min(len(hire), len(reject))
        parts += [hire.sample(n=n, random_state=seed), reject.sample(n=n, random_state=seed)]
        groups[str(name)] = {"hire": n, "reject": n, "dropped": len(part) - 2 * n}
    balanced = pd.concat(parts).sort_index() if parts else pairs.iloc[:0]
    info = {"n_input": len(pairs), "n_kept": len(balanced), "by": by, "seed": seed,
            "groups": groups}
    return balanced.reset_index(drop=True), info


def balance_by_margin(pairs: pd.DataFrame, column: str, bins: int = 10, seed: int = 42,
                      by: str = "protected_group") -> tuple[pd.DataFrame, dict]:
    """Equal hire and reject reference verdicts with the same distribution of `column`, the
    SFT model's log-prob margin log p(chosen) - log p(rejected), within each `by` group.

    Balancing counts was not enough in round 1. On reject-reference pairs the rejected answer
    is the SFT model's usual one (greedy hires 18% of them), on hire-reference pairs a rare
    sample (greedy hires all of them); the SFT model preferred chosen on 43% vs 82%. DPO then
    changed behaviour mostly on the reject side, and the model became stricter (hire rate
    28% -> 19% on the benchmark) with no gain in invariance at matched hire rate. Matching the
    margin distribution makes each side ask for the same amount of change.

    Bins are quantiles of the hire side's margins, bounded by its range: reject pairs with a
    margin outside it (the most negative ones, where the rejected answer is the model's usual
    one) are left out rather than lumped into an edge bin. Each bin keeps min(hire, reject)
    pairs of both sides, a seeded draw.
    """
    import numpy as np

    parts, groups = [], {}
    for name, part in pairs.groupby(by, sort=True):
        hire = part[part["reference_decision"] == "hire"]
        reject = part[part["reference_decision"] == "reject"]
        if hire.empty or reject.empty:
            groups[str(name)] = {"hire": 0, "reject": 0, "dropped": len(part)}
            continue
        edges = np.unique(np.quantile(hire[column], np.linspace(0, 1, bins + 1)))
        hire_bin = pd.cut(hire[column], edges, labels=False, include_lowest=True)
        reject_bin = pd.cut(reject[column], edges, labels=False, include_lowest=True)
        kept = 0
        for b in sorted(set(hire_bin.dropna())):
            h, r = hire[hire_bin == b], reject[reject_bin == b]
            k = min(len(h), len(r))
            parts += [h.sample(n=k, random_state=seed), r.sample(n=k, random_state=seed)]
            kept += k
        groups[str(name)] = {"hire": kept, "reject": kept, "dropped": len(part) - 2 * kept}
    balanced = pd.concat(parts).sort_index() if parts else pairs.iloc[:0]
    info = {"n_input": len(pairs), "n_kept": len(balanced), "by": by, "column": column,
            "bins": bins, "seed": seed, "groups": groups}
    if len(balanced):
        info["mean_margin"] = {d: float(m) for d, m in
                               balanced.groupby("reference_decision")[column].mean().items()}
    return balanced.reset_index(drop=True), info


# ---------------------------------------------------------------------------------------
# Full-pool sampling: the pool, chunks spread over GPUs, and resumable chunk files
# ---------------------------------------------------------------------------------------

def pool_rows(invariant_raw: pd.DataFrame, lang: str = "en") -> tuple[pd.DataFrame, dict]:
    """Every row of v1's pool with its `chosen` side, rebuilt by v1's own invariant filter."""
    from .filters import filter_invariant

    raw = invariant_raw["raw_output"].tolist()
    rows, report = filter_invariant(raw, invariant_raw.drop(columns=["raw_output"]), lang)
    return rows.reset_index(drop=True), report


def pair_free_prompts(rows: pd.DataFrame) -> pd.DataFrame:
    """One attribute-free prompt per pair. Every variant of a pair shares its job and CV, so the
    attribute-free prompt is the same for all of them and is sampled once."""
    first = rows.drop_duplicates("pair_id").reset_index(drop=True)
    return pd.DataFrame({"pair_id": first["pair_id"], "prompt": attr_free_prompts(first)})


def expand_free(rows: pd.DataFrame, free_by_pair: dict[str, list[str]]) -> list[list[str]]:
    """The pair-level attribute-free samples, repeated for each of the pair's rows."""
    return [free_by_pair[pid] for pid in rows["pair_id"]]


def chunk_plan(n_items: int, chunk_size: int, shards: int) -> list[list[tuple[int, int, int]]]:
    """(chunk id, start, end) per shard. Chunks go round-robin, so shards finish together."""
    chunks = [(cid, start, min(start + chunk_size, n_items))
              for cid, start in enumerate(range(0, n_items, chunk_size))]
    return [chunks[shard::shards] for shard in range(shards)]


def chunk_path(directory: str | Path, chunk_id: int) -> Path:
    return Path(directory) / f"chunk-{chunk_id:05d}.parquet"


def chunk_done(path: Path, prompts: list[str], k: int) -> bool:
    """A chunk file counts only if it holds exactly these prompts, k samples each."""
    if not Path(path).exists():
        return False
    frame = pd.read_parquet(path, columns=["item", "sample_index", "prompt"])
    per_item = frame.sort_values(["item", "sample_index"]).groupby("item", sort=True)
    return (per_item["prompt"].first().tolist() == list(prompts)
            and bool(per_item.size().eq(k).all()))


def write_chunk(path: Path, start: int, prompts: list[str], samples: list[list[str]]) -> None:
    """Writes atomically, so a job killed mid-write leaves no half chunk to be trusted."""
    rows = [{"item": start + i, "sample_index": j, "prompt": prompt, "raw_output": text}
            for i, (prompt, outs) in enumerate(zip(prompts, samples))
            for j, text in enumerate(outs)]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, path)


def read_samples(directory: str | Path, prompts: list[str], chunk_size: int,
                 k: int) -> list[list[str]]:
    """All samples in prompt order; refuses to return anything while a chunk is missing.
        return e.g. :
            '{"decision": "reject", "feedback": "Team-lead scope is unverified and ..."}'
    """
    out: list[list[str]] = []
    missing = []
    for chunk_id, start in enumerate(range(0, len(prompts), chunk_size)):
        end = min(start + chunk_size, len(prompts))
        path = chunk_path(directory, chunk_id)
        if not chunk_done(path, prompts[start:end], k):
            missing.append(chunk_id)
            continue
        frame = pd.read_parquet(path).sort_values(["item", "sample_index"])
        out.extend(group["raw_output"].tolist() for _, group in frame.groupby("item", sort=True))
    if missing:
        raise FileNotFoundError(f"{len(missing)} chunk(s) not sampled yet: {missing[:10]}")
    return out


def merged_student(cfg: dict) -> Path:
    """The student folded into its base, under outputs/merged/<adapter name>.

    `student_adapter` is the published SFT adapter (a Hub id), or a run or checkpoint under the
    output root, e.g. a DPO checkpoint whose own mistakes make the next round's negatives.
    A local adapter is merged together with the adapters it was trained on top of, SFT first
    (merge.adapter_chain): linked into outputs/sft/ like a Hub snapshot, a DPO checkpoint would
    lose its run's run_config.json and be merged without SFT.

    vLLM's LoRA path does not reproduce these adapters, so the student is always served as a
    merged checkpoint. Merging is skipped when an up-to-date merge already exists.
    """
    from ..mitigation import merge as M
    from ..mitigation import training_common as C
    from ..utils.config import resolve_output_path

    if resolve_output_path(cfg["student_adapter"]).exists():
        return M.merge(cfg["student_adapter"])
    snapshot = Path(C.resolve_sft_checkpoint(cfg["student_adapter"]))
    name = cfg["student_adapter"].rstrip("/").split("/")[-1]
    link = Path(resolve_output_path(f"outputs/sft/{name}"))
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(snapshot, target_is_directory=True)
    return M.merge(f"outputs/sft/{name}")
