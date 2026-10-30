"""The 8-draw vote anchor, for any pool.

docs/SFT_V3_ANCHORS.md made the vote's majority the anchor of record on 2026-10-20. v1's anchor
was one draw per pair under one seed shared by every prompt, which on vLLM's V2 runner acts as
a common hire threshold near p = 0.87 rather than a sample of the teacher's decision (job
5103584). The vote replaces it with the teacher's own majority:

* `synth.REFERENCE_PROMPT`, unchanged, on the attribute-free pair;
* 8 draws per pair from one request (`VLLMBackend.generate_samples`), at v1's decoding --
  temperature 0.7, top_p 0.9, 320 new tokens, thinking off -- with the request seeded by
  `per_prompt_seed(base, prompt)`, so no two pairs share sampling noise;
* the majority over *decided* draws (hire or reject) is the decision;
* its feedback is the lowest-index majority draw that passes `filters.filter_reference`.

The rule is `new_anchors` in scripts/sft_v3_anchors.py (branch xinyang/sft-v3-vote-anchors,
e4a7c55), ported here so a pool other than v1's can use it. One thing has to differ. On v1's
pool, a tie, a pair with no decided draw and a majority with no usable feedback kept the v1
anchor. A pair from a new source has no v1 anchor, and inventing one would put an arbitrary
verdict into training, so `vote_anchors` leaves those pairs without an anchor (`anchored`
False) unless the caller passes a fallback. With a fallback it reproduces `new_anchors`
exactly, which `tests/test_vote.py` pins. Its `source` column is called `anchor_source` here,
so it cannot collide with a pool's `source` (the corpus a pair came from).

`unstable` is the pre-registered definition of docs/SFT_V3_ANCHORS.md: a majority of at most
5 of the decided draws, no decided draw, or -- when a fallback exists -- a verdict the vote
overturns. On v1's pool it covered 11.2% of pairs. The flag travels with the anchor so a
training set can be built with or without those pairs; nothing here drops them.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..eval.parsing import parse_output
from . import filters as F

#: Draws per pair, and v1's anchor decoding with the vote's base seed (43, as job 5073453).
VOTE_DRAWS = 8
VOTE_GENERATION = {"max_new_tokens": 320, "temperature": 0.7, "top_p": 0.9, "seed": 43}
DECIDED = ("hire", "reject")
#: A majority at or below this share of decided draws is unstable (5 of 8, or 4 to 4).
UNSTABLE_MAX_SHARE = 5 / 8

ANCHOR_COLUMNS = [
    "pair_id", "hire_votes", "decided", "majority", "majority_share", "vote_feedback",
    "vote_sample_index", "anchor_source", "anchored", "reference_decision", "reference_feedback",
    "unstable",
]


def prompts_sha256(prompts: list[str]) -> str:
    """A fingerprint of exactly what was sent, in order."""
    return hashlib.sha256("\x1e".join(prompts).encode("utf-8")).hexdigest()


def draw_frame(pair_ids: list[str], draws: list[list[str]], lang: str) -> pd.DataFrame:
    """One row per draw: pair, index, raw text and its parse."""
    rows = [{"pair_id": pair, "sample_index": j, "raw_output": text}
            for pair, texts in zip(pair_ids, draws) for j, text in enumerate(texts)]
    frame = pd.DataFrame(rows, columns=["pair_id", "sample_index", "raw_output"])
    parsed = [parse_output(text, lang) for text in frame["raw_output"]]
    return frame.assign(decision=[p.decision for p in parsed],
                        outcome=[p.outcome for p in parsed],
                        feedback=[p.feedback for p in parsed])


def vote_anchors(draws: pd.DataFrame, pair_ids: pd.Series | list[str], lang: str,
                 fallback: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per pair: the vote and the anchor it gives.

    `draws` is `draw_frame`'s output. `fallback`, when given, has `pair_id`,
    `fallback_decision` and `fallback_feedback` (v1's anchor on v1's pool) and is used exactly
    where `new_anchors` used v1's. Without it, those pairs get no anchor.
    """
    out = pd.DataFrame({"pair_id": pd.Series(list(pair_ids), dtype=object)})
    if out["pair_id"].duplicated().any():
        raise ValueError("vote_anchors: duplicate pair_ids")
    decided = draws[draws["decision"].isin(DECIDED)]
    hire = decided.assign(h=decided["decision"] == "hire").groupby("pair_id")["h"].sum()
    count = decided.groupby("pair_id").size()
    out["hire_votes"] = out["pair_id"].map(hire).fillna(0).astype(int)
    out["decided"] = out["pair_id"].map(count).fillna(0).astype(int)
    out["majority"] = np.select([2 * out["hire_votes"] > out["decided"],
                                 2 * out["hire_votes"] < out["decided"]],
                                ["hire", "reject"], "tie")
    winning = np.where(out["majority"] == "hire", out["hire_votes"],
                       out["decided"] - out["hire_votes"])
    out["majority_share"] = np.where(out["decided"] > 0,
                                     winning / out["decided"].clip(lower=1), np.nan)

    candidates = draws.merge(out[["pair_id", "majority"]], on="pair_id")
    candidates = candidates[candidates["decision"] == candidates["majority"]]
    candidates = candidates.sort_values(["pair_id", "sample_index"]).reset_index(drop=True)
    kept = (F.filter_reference(candidates["raw_output"].tolist(),
                               candidates[["pair_id", "sample_index"]], lang)[0]
            if len(candidates) else pd.DataFrame())
    first = (kept.drop_duplicates("pair_id").set_index("pair_id") if len(kept)
             else pd.DataFrame(columns=["reference_feedback", "sample_index"]))
    out["vote_feedback"] = out["pair_id"].map(first["reference_feedback"])
    out["vote_sample_index"] = out["pair_id"].map(first["sample_index"])
    out["anchor_source"] = np.select(
        [out["decided"] == 0, out["majority"] == "tie", out["vote_feedback"].isna()],
        ["no_decided_draw", "tie", "no_usable_feedback"], "vote")
    vote = out["anchor_source"] == "vote"

    if fallback is None:
        out["anchored"] = vote
        out["reference_decision"] = out["majority"].where(vote, None)
        out["reference_feedback"] = out["vote_feedback"].where(vote, None)
        overturned = pd.Series(False, index=out.index)
    else:
        fb = out[["pair_id"]].merge(fallback, on="pair_id", how="left", validate="one_to_one")
        if fb["fallback_decision"].isna().any():
            raise ValueError("vote_anchors: a pair has no fallback anchor")
        out["anchored"] = True
        out["reference_decision"] = out["majority"].where(vote, fb["fallback_decision"])
        out["reference_feedback"] = out["vote_feedback"].where(vote, fb["fallback_feedback"])
        overturned = out["reference_decision"] != fb["fallback_decision"]
    out["unstable"] = ((out["decided"] == 0)
                       | (out["majority_share"].fillna(0) <= UNSTABLE_MAX_SHARE)
                       | overturned)
    return out[ANCHOR_COLUMNS]


def vote_report(anchors: pd.DataFrame, draws: pd.DataFrame,
                by: pd.Series | None = None) -> dict:
    """Descriptive readouts of one vote, optionally per stratum (e.g. pool source)."""

    def one(a: pd.DataFrame, d: pd.DataFrame) -> dict:
        anchored = a[a["anchored"]]
        fenced = d["raw_output"].astype(str).str.lstrip().str.startswith("```")
        return {
            "pairs": len(a),
            "anchored": len(anchored),
            "anchor_sources": {k: int(v) for k, v in a["anchor_source"].value_counts().items()},
            "hire_votes": {f"{k} of {VOTE_DRAWS}": int(v)
                           for k, v in a["hire_votes"].value_counts().sort_index().items()},
            "per_draw_hire_share": float((d["decision"] == "hire").mean()) if len(d) else None,
            "anchor_hire_share": (float((anchored["reference_decision"] == "hire").mean())
                                  if len(anchored) else None),
            "unanimous_share": (float((a["majority_share"] == 1.0).mean()) if len(a) else None),
            "unstable_share": float(a["unstable"].mean()) if len(a) else None,
            "undecided_draw_share": (float((~d["decision"].isin(DECIDED)).mean())
                                     if len(d) else None),
            "fenced_draw_share": float(fenced.mean()) if len(d) else None,
            "fenced_hire_share": (float((d.loc[fenced, "decision"] == "hire").mean())
                                  if fenced.any() else None),
            "bare_hire_share": (float((d.loc[~fenced, "decision"] == "hire").mean())
                                if (~fenced).any() else None),
        }

    report = {"all": one(anchors, draws)}
    if by is not None:
        strata = pd.Series(list(by), index=anchors.index)
        pair_stratum = dict(zip(anchors["pair_id"], strata))
        draw_stratum = draws["pair_id"].map(pair_stratum)
        report["by"] = {str(s): one(anchors[strata == s], draws[draw_stratum == s])
                        for s in sorted(strata.unique())}
    return report
