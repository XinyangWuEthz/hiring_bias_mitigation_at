"""One preference set from several covert-negative pools, balanced by gradient, not only count.

What goes in
------------
Pairs of (prompt, chosen, rejected) whose negative is covert: the rejected response takes the
opposite decision under a rationale that never names the attribute.

``round1``
    The round-1 pairs, train and validation as they were used.
``extra:<name>``
    More pairs from the same construction -- e.g. the hire-side surplus that the round-1 hire
    resampling drew beyond what its 50/50 split could use. Training candidates only.
``legacy``
    The ``semisynthetic-v1`` DPO pairs (``dpo`` subset), kept only where the rejected response
    does not name the attribute. The biased-pass filter keeps a negative only if it names the
    attribute or flips the decision, so every negative kept here flips it. The prefix-matching
    detector misses paraphrases ("your gender identity suggests..."), so these pairs also pass
    an LLM judge before they are used.

Why balance the gradient, not only the count
--------------------------------------------
DPO scales each pair's update by ``beta * sigmoid(-beta * h)``, with ``h`` the pair's
implicit-reward margin. At the first step ``h = 0`` everywhere, so every pair starts with the
same weight and a set that is 50/50 by count starts with no net push toward either decision.
After that the weight falls fastest on pairs the model can satisfy with features it already
has, and stays high longest on pairs that run against them -- typically the ones where the
starting model is confident in the *rejected* decision. Those keep pulling after the rest have
gone quiet. If they sit mostly on one side, the late gradient is one-sided and moves the whole
operating point: a reject-heavy pull shows up as a falling hire rate on the items whose
reference is hire, even though the counts were equal.

Per pair, the proxy for that persistent pull is the starting model's probability of the wrong
decision, ``p_wrong = sigmoid(-y * margin)`` (``margin`` = logit(hire) - logit(reject) at the
decision token, ``y`` = +1 when chosen is hire, -1 when reject). It is also the size of the
cross-entropy gradient on the decision restricted to the two words, and a pair whose chosen
answer the model finds unlikely also carries the larger gradients on its rationale tokens.
Summed over a side it is that side's expected number of student errors -- its error mass.
:func:`balance_error_mass` makes both the counts and the error masses equal.

The starting model is the DPO starting point and reference: the SFT model the run continues
from. Scores from any other model describe a different run.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.benchmark import assert_no_leakage
from ..data.injection import implicit_sentence
from ..data.prompts import _explicit_line
from ..data.protected_groups import GROUP_LABELS, load_group
from ..eval.metrics import mentions_attribute
from ..eval.parsing import HIRE, REJECT, _extract_json, parse_output
from ..utils.logging import get_logger

log = get_logger(__name__)

SIDES = (HIRE, REJECT)
ROUND1 = "round1"
LEGACY = "legacy"
EXTRA_PREFIX = "extra:"

#: Every pool must carry these. The ids feed the benchmark-leakage guard and the
#: candidate-level split, and neither can be recovered from the prompt text.
REQUIRED = ("prompt", "chosen", "rejected", "candidate_id", "job_id")
#: Read off the prompt when a pool does not carry them (see `infer_variant`).
VARIANT = ("protected_group", "protected_attr", "condition")
COLUMNS = (
    "prompt", "chosen", "rejected", "decision", "rejected_decision", "lang",
    "protected_group", "protected_attr", "condition", "candidate_id", "job_id", "pair_id",
    "source", "split",
)
#: Kept from a pool that carries them. V3's pairs are balanced per `pool_source`
#: (`assemble(balance_by=...)`), and a column the schema dropped could not be balanced by.
PASSTHROUGH = ("pool_source",)


# ---------------------------------------------------------------------------------------
# Reading and normalising the pools
# ---------------------------------------------------------------------------------------

def read_table(path: str | Path) -> pd.DataFrame:
    """Reads a pool from parquet, JSON Lines, JSON records or CSV, keeping ids as written."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    if suffix == ".json":
        return pd.read_json(path, dtype=False, convert_dates=False)
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    raise ValueError(f"{path}: unsupported format {suffix!r}; use parquet, jsonl, json or csv")


def source_rank(source: str) -> int:
    """Whose pair wins a tie: round 1, then extras from its construction, then legacy."""
    if source == ROUND1:
        return 0
    return 2 if source == LEGACY else 1


def normalise_pairs(
    frame: pd.DataFrame, source: str, split: str, lang: str
) -> tuple[pd.DataFrame, dict]:
    """One pool in the common schema, with the chosen and rejected decisions parsed out.

    A pair is dropped, never repaired, if either side has no readable decision or both sides
    decide the same way -- a covert negative is defined by its flipped decision.
    """
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} ({split}) is missing column(s) {missing}. The ids are needed for the "
            "benchmark-leakage check and the candidate-level split, and cannot be recovered "
            "from the prompt; export them with the pairs."
        )
    df = frame.copy()
    report: dict = {"rows_in": len(df)}
    if "lang" not in df.columns:
        df["lang"] = lang
        report["lang_filled_with"] = lang
    other = df["lang"].astype(str) != lang
    if other.any():
        report["dropped_other_language"] = int(other.sum())
        df = df[~other].copy()
    for column in ("candidate_id", "job_id"):
        df[column] = df[column].astype(str)

    df["decision"] = [parse_output(t, lang).decision for t in df["chosen"]]
    df["rejected_decision"] = [parse_output(t, lang).decision for t in df["rejected"]]
    unreadable = ~df["decision"].isin(SIDES) | ~df["rejected_decision"].isin(SIDES)
    if unreadable.any():
        report["dropped_unreadable_decision"] = int(unreadable.sum())
        df = df[~unreadable].copy()
    same = df["decision"] == df["rejected_decision"]
    if same.any():
        report["dropped_same_decision_both_sides"] = int(same.sum())
        df = df[~same].copy()

    for column in VARIANT:
        if column not in df.columns:
            df[column] = None
    need = df[list(VARIANT)].isna().any(axis=1)
    if need.any():
        found, lost = 0, []
        for index in df.index[need]:
            variant = infer_variant(df.at[index, "prompt"], lang)
            if variant is None:
                lost.append(index)
                continue
            for column, value in zip(VARIANT, variant):
                df.at[index, column] = value
            found += 1
        report["variant_read_from_prompt"] = found
        if lost and not found:
            raise ValueError(
                f"{source} ({split}): none of its prompts carries the audit's attribute field "
                "or sentence, so group, attribute and condition cannot be read off them. "
                "Export protected_group, protected_attr and condition with the pairs."
            )
        if lost:
            report["dropped_variant_not_in_prompt"] = len(lost)
            df = df.drop(index=lost)

    if "pair_id" not in df.columns:
        df["pair_id"] = None
    df["pair_id"] = df["pair_id"].where(df["pair_id"].notna(),
                                        df["candidate_id"] + "__" + df["job_id"])
    df = df.assign(source=source, split=split)
    report["rows_out"] = len(df)
    for reason, n in report.items():
        if reason.startswith("dropped") and n:
            log.warning("%s (%s): %s = %d", source, split, reason, n)
    kept = list(COLUMNS) + [c for c in PASSTHROUGH if c in df.columns]
    return df[kept].reset_index(drop=True), report


@cache
def _variant_markers(lang: str) -> tuple[tuple[str, str, str, str], ...]:
    """(text, group, attribute, condition) for every single-group variant the prompt injects."""
    markers = []
    for group, labels in GROUP_LABELS.items():
        try:
            attributes = load_group(group, lang).attributes
        except FileNotFoundError:
            continue
        for attr in attributes:
            line = _explicit_line(
                {"protected_group_label": labels[lang], "protected_attr": attr, "lang": lang}
            )
            markers.append((f"{line}\n", group, attr, "explicit"))
            try:
                markers.append((f"{implicit_sentence(group, attr, lang)}\n", group, attr,
                                "implicit"))
            except KeyError:
                continue
    return tuple(markers)


def infer_variant(prompt: str, lang: str) -> tuple[str, str, str] | None:
    """The (group, attribute, condition) a prompt was rendered with, read off its text.

    Matches the exact strings the audit prompt injects -- the labelled field for `explicit`,
    the first-person sentence for `implicit` -- each with the newline that starts the CV, so
    `Male` cannot match inside `Female`. None when nothing matches or more than one does.
    """
    hits = {(g, a, c) for marker, g, a, c in _variant_markers(lang) if marker in prompt}
    return hits.pop() if len(hits) == 1 else None


def filter_legacy(frame: pd.DataFrame, lang: str) -> tuple[pd.DataFrame, dict]:
    """The `semisynthetic-v1` pairs whose negative does not name the attribute.

    Uses the flag the generator stored, then re-checks both rationales with the current
    detector, so a pair cannot pass on a flag computed by an older version of it.
    """
    report: dict = {"rows_in": len(frame)}
    df = frame[frame["lang"].astype(str) == lang].copy()
    report["in_language"] = len(df)
    if "rejected_mentions_attribute" in df.columns:
        named = df["rejected_mentions_attribute"].astype(bool)
        report["dropped_rejected_names_attribute"] = int(named.sum())
        df = df[~named]
    leaks = [
        mentions_attribute(parse_output(chosen, lang).feedback, attr)
        or mentions_attribute(parse_output(rejected, lang).feedback, attr)
        for chosen, rejected, attr in zip(df["chosen"], df["rejected"], df["protected_attr"])
    ]
    leaks = np.asarray(leaks, dtype=bool)
    report["dropped_detector_recheck"] = int(leaks.sum())
    df = df[~leaks]
    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def add_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Stable keys for caching the expensive stages: the prompt alone, and the whole pair."""
    return df.assign(
        prompt_sha1=[_sha1(p) for p in df["prompt"]],
        pair_sha1=[_sha1(f"{p}\x1f{c}\x1f{r}")
                   for p, c, r in zip(df["prompt"], df["chosen"], df["rejected"])],
    )


def _sha1(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------
# The LLM judge for paraphrased references
# ---------------------------------------------------------------------------------------

JUDGE_PROMPT = (
    "You check hiring feedback for references to a protected characteristic of the "
    "candidate.\n\n"
    "The candidate's profile stated their {label}: {attr}.\n\n"
    "Feedback: ```\n{feedback}\n```\n\n"
    "Does the feedback refer to the candidate's {label} in any way -- naming it, "
    "paraphrasing it, or alluding to it ({hints})? Professional experience, skills, "
    "education and job history are not references. Judge only whether the characteristic "
    "is referred to, not whether the decision is fair.\n"
    'Return only a JSON object: {{"refers": true}} or {{"refers": false}}'
)

JUDGE_HINTS = {
    "military_status": "military service, veteran or reservist status, the armed forces, "
                       "combat, being a civilian",
    "gender": "gender, sex, gender identity, pronouns, being a man, a woman, non-binary or "
              "transgender",
    "religion": "religion, faith, belief or non-belief, worship, religious practice or "
                "holidays",
    "marital_status": "marriage, divorce, a partner or spouse, being single or widowed",
}


def judge_prompt(feedback: str, group: str, attr: str) -> str:
    return JUDGE_PROMPT.format(
        label=GROUP_LABELS.get(group, {}).get("en", group.replace("_", " ")),
        attr=attr, feedback=feedback, hints=JUDGE_HINTS.get(group, group.replace("_", " ")),
    )


def parse_judge(raw: str) -> str:
    """The judge's verdict on one rationale: "refers", "clean" or "unreadable".

    The build stage drops a pair unless both of its rationales are "clean". An unreadable
    answer counts against the pair: dropping a clean pair costs one pair, keeping a leaking
    one brings back the wording shortcut this construction exists to remove.
    """
    obj = _extract_json(raw)
    value = obj.get("refers") if isinstance(obj, dict) else None
    if value is None:
        match = re.search(r'"?refers"?\s*:\s*"?(true|false)', str(raw), flags=re.IGNORECASE)
        value = match.group(1) if match else None
    text = str(value).strip().lower()
    if value is True or text in ("true", "yes"):
        return "refers"
    if value is False or text in ("false", "no"):
        return "clean"
    return "unreadable"


# ---------------------------------------------------------------------------------------
# Merging: duplicates, contradictory anchors, the candidate split
# ---------------------------------------------------------------------------------------

def drop_duplicate_prompts(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One pair per prompt. The same variant drawn into two pools would count twice."""
    ranked = df.assign(_rank=df["source"].map(source_rank)).sort_values("_rank", kind="stable")
    duplicate = ranked.duplicated("prompt", keep="first")
    report = {
        "dropped": int(duplicate.sum()),
        "dropped_by_source": ranked.loc[duplicate, "source"].value_counts().to_dict(),
    }
    return ranked[~duplicate].drop(columns="_rank").sort_index(), report


def drop_conflicting_anchors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Every variant of one job-CV pair must carry the same decision.

    That invariance is what the training installs. Pools anchored separately can disagree on
    a pair -- the `semisynthetic-v1` anchor is one sample at temperature 0.7 -- and keeping
    both would teach the model to decide one CV differently depending on the attribute. The
    highest-priority pool holding the pair sets its decision; a pair that pool disagrees with
    itself on is dropped whole.
    """
    key = ["candidate_id", "job_id"]
    ranked = df.assign(_rank=df["source"].map(source_rank))
    top = ranked.groupby(key)["_rank"].transform("min")
    authority = ranked[ranked["_rank"] == top]
    n_decisions = authority.groupby(key)["decision"].nunique()
    anchor = authority.groupby(key)["decision"].first().rename("_anchor")
    ranked = ranked.join(anchor, on=key).join((n_decisions > 1).rename("_split_anchor"), on=key)
    self_conflict = ranked["_split_anchor"]
    against = ~self_conflict & (ranked["decision"] != ranked["_anchor"])
    report = {
        "pairs_dropped_disagreeing_with_themselves": int((n_decisions > 1).sum()),
        "rows_dropped_with_them": int(self_conflict.sum()),
        "rows_dropped_against_higher_priority_anchor": int(against.sum()),
        "against_anchor_by_source": ranked.loc[against, "source"].value_counts().to_dict(),
    }
    keep = ~(self_conflict | against)
    return ranked[keep].drop(columns=["_rank", "_anchor", "_split_anchor"]), report


def split_by_candidate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Validation keeps every candidate any pool held out; training loses their rows.

    Each pool was split by candidate on its own, so a candidate one pool held out can sit in
    another pool's training split -- and validation would then measure memory. Every
    variant of a CV is a near-duplicate of the others, so this is checked by candidate, not
    by row.
    """
    in_validation = df["split"] == "validation"
    held_out = set(df.loc[in_validation, "candidate_id"])
    leaked = ~in_validation & df["candidate_id"].isin(held_out)
    report = {
        "train_rows_dropped_candidate_held_out": int(leaked.sum()),
        "by_source": df.loc[leaked, "source"].value_counts().to_dict(),
    }
    train = df[~in_validation & ~leaked].reset_index(drop=True)
    return train, df[in_validation].reset_index(drop=True), report


# ---------------------------------------------------------------------------------------
# Balancing by error mass
# ---------------------------------------------------------------------------------------

def with_student_error(df: pd.DataFrame, margin: str = "student_margin") -> pd.DataFrame:
    """Adds `wrong_logit` (the margin toward the rejected decision) and `p_wrong`."""
    sign = np.where(df["decision"] == HIRE, -1.0, 1.0)
    wrong = sign * df[margin].to_numpy(dtype=float)
    return df.assign(wrong_logit=wrong, p_wrong=1.0 / (1.0 + np.exp(-wrong)))


def side_summary(df: pd.DataFrame) -> dict:
    out = {}
    for side in SIDES:
        part = df[df["decision"] == side]
        out[side] = {
            "pairs": len(part),
            "error_mass": round(float(part["p_wrong"].sum()), 2),
            "mean_p_wrong": round(float(part["p_wrong"].mean()), 4) if len(part) else None,
            "student_wrong": int((part["p_wrong"] > 0.5).sum()),
            "by_source": part["source"].value_counts().to_dict(),
        }
    return out


def balance_error_mass(
    df: pd.DataFrame, tolerance: float = 0.05, min_keep: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """Equal counts and equal error mass on the hire and reject sides.

    Each side is ranked by how confidently the student prefers the rejected decision. The
    lighter side keeps its hardest pairs -- harder pairs added to it raise its mass first.
    The heavier side drops its *most confidently wrong* pairs until its mass is within
    `tolerance` of the lighter side's, and both sides are then cut to the same count by
    dropping their easiest pairs, which carry almost no error mass. The largest count that
    allows this wins, so trimming is the least the balance needs.

    Fails rather than return a set either side of which keeps under `min_keep` of the
    smaller side's pool: at that point the lighter side lacks hard pairs, and the remedy is
    to add some, not to train on what is left.
    """
    if not {"p_wrong", "wrong_logit"} <= set(df.columns):
        df = with_student_error(df)
    rng = np.random.default_rng(seed)
    ranked, cum = {}, {}
    for side in SIDES:
        part = df[df["decision"] == side]
        if part.empty:
            raise ValueError(f"no {side}-side pairs to balance against")
        order = np.lexsort((rng.random(len(part)), -part["wrong_logit"].to_numpy()))
        ranked[side] = part.iloc[order]
        cum[side] = np.concatenate([[0.0], np.cumsum(ranked[side]["p_wrong"].to_numpy())])

    pool = {side: len(ranked[side]) for side in SIDES}
    n_max = min(pool.values())
    n_min = max(1, math.ceil(min_keep * n_max))
    for n in range(n_max, n_min - 1, -1):
        hardest = {side: cum[side][n] for side in SIDES}
        heavy = max(SIDES, key=lambda s: hardest[s])
        light = REJECT if heavy == HIRE else HIRE
        ceiling = hardest[light] * (1.0 + tolerance)
        window = cum[heavy][n:] - cum[heavy][: len(cum[heavy]) - n]
        fits = np.flatnonzero(window <= ceiling + 1e-9)
        if fits.size:
            skip = int(fits[0])
            break
    else:
        raise ValueError(
            f"cannot balance: keeping at least {n_min} pairs per side, the lightest the "
            f"heavier side gets is still more than {1 + tolerance:.2f}x the error mass of the "
            f"lighter one. Add harder pairs to the lighter side (e.g. the hire-resampling "
            f"surplus as an extra pool), or lower min_keep. Pools: "
            f"{ {s: round(float(cum[s][-1]), 1) for s in SIDES} } error mass over {pool} pairs."
        )

    kept_heavy = ranked[heavy].iloc[skip: skip + n]
    kept_light = ranked[light].iloc[:n]
    trimmed = ranked[heavy].iloc[:skip]
    selected = pd.concat([kept_heavy, kept_light]).sort_index()

    mass = {heavy: float(kept_heavy["p_wrong"].sum()), light: float(kept_light["p_wrong"].sum())}
    report = {
        "tolerance": tolerance,
        "pairs_per_side": n,
        "heavier_side": heavy,
        "trimmed_most_confidently_wrong": {
            "side": heavy,
            "pairs": skip,
            "p_wrong_at_least": round(float(trimmed["p_wrong"].min()), 4) if skip else None,
            "by_source": trimmed["source"].value_counts().to_dict(),
        },
        "dropped_easiest_for_count": {
            heavy: int(pool[heavy] - skip - n),
            light: int(pool[light] - n),
        },
        "error_mass_ratio": round(mass[heavy] / mass[light], 4) if mass[light] else None,
        "before": side_summary(df),
        "after": side_summary(selected),
    }
    return selected.reset_index(drop=True), report


def balance_error_mass_by(
    df: pd.DataFrame, by: str, tolerance: float = 0.05, min_keep: float = 0.5, seed: int = 42
) -> tuple[pd.DataFrame, dict]:
    """`balance_error_mass` within each value of `by`, each value on its own.

    V3's pool sources differ in hire share (22%, 56% and 15% of anchors), so a pooled balance
    would let one source supply a side for another; each source is balanced by count and by
    error mass on its own. A value that cannot be balanced -- one side missing, or too few hard
    pairs on the lighter side to keep `min_keep` -- stays out, with the reason in the report,
    rather than failing the whole build.
    """
    if df[by].isna().any():
        raise ValueError(f"{int(df[by].isna().sum())} pairs have no {by}; a per-{by} balance "
                         "needs it on every pair")
    parts, groups = [], {}
    for value, part in df.groupby(by, sort=True):
        try:
            kept, groups[str(value)] = balance_error_mass(part, tolerance, min_keep, seed)
        except ValueError as exc:
            groups[str(value)] = {"left_out": str(exc), "before": side_summary(
                part if "p_wrong" in part else with_student_error(part))}
            log.warning("%s=%s left out: %s", by, value, exc)
            continue
        parts.append(kept)
    if not parts:
        raise ValueError(f"no {by} could be balanced: {groups}")
    report = {"by": by, "groups": groups,
              "left_out": sorted(k for k, v in groups.items() if "left_out" in v),
              "after": side_summary(pd.concat(parts, ignore_index=True))}
    return pd.concat(parts, ignore_index=True), report


def balance_count_by(df: pd.DataFrame, by: list[str], seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Equal hire and reject pairs within each cell of `by`: every pair of the scarcer verdict
    and a seeded sample of the same size from the other. A cell lacking one verdict drops out.

    How r1's validation was balanced (`onpolicy.balance_by_reference`, per protected group),
    which r2em kept; V3 does it within each pool source as well.
    """
    parts, cells = [], {}
    for key, part in df.groupby(by, sort=True):
        hire, reject = part[part["decision"] == HIRE], part[part["decision"] == REJECT]
        n = min(len(hire), len(reject))
        parts += [hire.sample(n=n, random_state=seed), reject.sample(n=n, random_state=seed)]
        name = " / ".join(map(str, key)) if isinstance(key, tuple) else str(key)
        cells[name] = {"hire": n, "reject": n, "dropped": len(part) - 2 * n}
    kept = pd.concat(parts).sort_index() if parts else df.iloc[:0]
    return kept.reset_index(drop=True), {"by": list(by), "seed": seed, "n_input": len(df),
                                         "n_kept": len(kept), "cells": cells}


# ---------------------------------------------------------------------------------------
# The whole build
# ---------------------------------------------------------------------------------------

def assemble(
    candidates: pd.DataFrame,
    scores: pd.DataFrame,
    judge: pd.DataFrame | None,
    tolerance: float = 0.05,
    min_keep: float = 0.5,
    seed: int = 42,
    require_judge: bool = True,
    balance_by: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Judge filter -> duplicates -> anchors -> candidate split -> leakage -> balance.

    Returns (train, validation, report). Only the training split is balanced by error mass:
    validation is a measurement, and its pairs never reach the gradient.

    With `balance_by` (V3: `pool_source`), training is balanced within each value of that
    column (`balance_error_mass_by`), and validation by count within each value and protected
    group (`balance_count_by`), as r1's validation was before r2em kept it. A value left out of
    training is left out of validation too.
    """
    report: dict = {"candidates": candidates["source"].value_counts().to_dict()}
    df = candidates

    verdicts = ["verdict_chosen", "verdict_rejected"]
    if judge is not None and not judge.empty:
        latest = judge.drop_duplicates("pair_sha1", keep="last")
        df = df.merge(latest[["pair_sha1", *verdicts]], on="pair_sha1", how="left")
    else:
        df = df.assign(verdict_chosen=None, verdict_rejected=None)
    judged = df["verdict_chosen"].notna() & df["verdict_rejected"].notna()
    unjudged = df["needs_judge"].astype(bool) & ~judged
    if unjudged.any() and require_judge:
        raise ValueError(
            f"{int(unjudged.sum())} pairs need an LLM-judge verdict and have none -- run the "
            "judge stage first, or pass --skip-judge to build without it."
        )
    clean = df["verdict_chosen"].eq("clean") & df["verdict_rejected"].eq("clean")
    flagged = judged & ~clean
    report["judge"] = {
        "judged": int(judged.sum()),
        "dropped_refers_or_unreadable": int(flagged.sum()),
        "dropped_by_source": df.loc[flagged, "source"].value_counts().to_dict(),
        "built_without_judge": int(unjudged.sum()),
    }
    df = df[~flagged].drop(columns=verdicts)

    df, report["duplicates"] = drop_duplicate_prompts(df)
    df, report["anchors"] = drop_conflicting_anchors(df)
    train, validation, report["split"] = split_by_candidate(df)
    assert_no_leakage(train, "balanced_dpo.train")
    assert_no_leakage(validation, "balanced_dpo.validation")

    score_cols = scores[["prompt_sha1", "student_margin"]].drop_duplicates("prompt_sha1")
    train = train.merge(score_cols, on="prompt_sha1", how="left")
    unscored = train["student_margin"].isna()
    if unscored.any():
        raise ValueError(
            f"{int(unscored.sum())} training pairs have no student score -- run the score "
            "stage on the current candidates first."
        )
    validation = validation.merge(score_cols, on="prompt_sha1", how="left")

    if balance_by:
        train, report["balance"] = balance_error_mass_by(
            with_student_error(train), balance_by, tolerance=tolerance, min_keep=min_keep,
            seed=seed)
        left_out = validation[balance_by].astype(str).isin(report["balance"]["left_out"])
        validation, report["validation_balance"] = balance_count_by(
            validation[~left_out], [balance_by, "protected_group"], seed=seed)
        report["validation_balance"]["left_out_with_training"] = int(left_out.sum())
    else:
        train, report["balance"] = balance_error_mass(
            with_student_error(train), tolerance=tolerance, min_keep=min_keep, seed=seed
        )
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    scored_val = validation.dropna(subset=["student_margin"])
    report["validation"] = side_summary(with_student_error(scored_val)) if len(scored_val) else {}
    report["validation_pairs"] = len(validation)
    report["train_pairs"] = len(train)
    report["steps_per_epoch_at_effective_batch_16"] = math.ceil(len(train) / 16)
    # The chosen decision is the attribute-free reference: one per CV and job after the anchor
    # check. The trainer's decision eval (mitigation/decision_eval.py) reads it by this name.
    train, validation = (part.assign(reference_decision=part["decision"])
                         for part in (train, validation))
    drop = ["needs_judge", "wrong_logit"]
    return (train.drop(columns=[c for c in drop if c in train.columns]),
            validation.drop(columns=[c for c in drop if c in validation.columns]), report)
