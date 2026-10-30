"""SFT-v3, stages 0 and 1: training anchors from another prompt and an 8-draw vote, and the gate
that decides whether SFT is retrained on them.

The pre-registration, its criteria and the results are in docs/SFT_V3_ANCHORS.md. Only the
anchor changes: the pool, the 12 attribute variants per pair, the teacher, its decoding and
v1's invariant procedure stay as they were. Nothing here changes an evaluation definition, and
the benchmark is never read.

Stages, in order:

  labels    CPU  the blind review's labels for its 391 pool pairs, without its text
  select    GPU  stage 0a: both candidate prompts, 8 draws each, on the 391 reviewed pairs
  score     CPU  C1 and C2 for each candidate, and the prompt the pre-registered rule picks
  generate  GPU  stage 0b: the chosen prompt on all 2,992 pool pairs, then v1's invariant
                 procedure on the pairs whose verdict changed; resumes from saved draws
  gate      CPU  stage 1: C1 to C3 and the descriptive readouts

and the two arms added after stage 0a (the page's addenda):

  control     GPU  the redraw control: the unchanged v1 prompt, 8 draws, on the 391 pairs
  compare     CPU  the control against both candidates on the same pairs
  votes       GPU  the pool-wide vote: the unchanged v1 prompt, 8 draws at seed 43, on all
                   2,992 pool pairs, saved in parts; resumes from saved parts
  enrichment  CPU  the platform check, unstable anchors in earlier training sets, and the
                   vote's agreement with the control

and, once job 5103584 showed v1's anchors were one sampling draw shared by every prompt and the
user chose the pool-wide vote's majority as the anchor (the page's third addendum):

  vote_targets  GPU  v1's invariant procedure on the pairs whose verdict the vote changes,
                     seeded per prompt; skips if its output already covers these prompts
  vote_readout  CPU  those targets against v1's on the same variants, the anchors' readouts,
                     and the rows of earlier training sets whose target the vote flips
  vote_sft      CPU  the SFT-v3 data sets (the page's fourth addendum): v1's split with the
                     changed rows swapped, and v1's assembly re-run on the vote anchors; stops
                     unless it first rebuilds v1's published split exactly

Outputs: artifacts/semisynthetic-v3/ under $HBM_OUTPUT_ROOT (training data from the pool, never
benchmark outputs) and reports/sft_v3_anchors/.

    python scripts/sft_v3_anchors.py --stage labels --review <claude_anchor_review/graded_cases.csv>
    python scripts/sft_v3_anchors.py --stage select      # GPU, tensor parallel 4
    python scripts/sft_v3_anchors.py --stage score
    python scripts/sft_v3_anchors.py --stage generate    # GPU, tensor parallel 4
    python scripts/sft_v3_anchors.py --stage gate
    python scripts/sft_v3_anchors.py --stage control     # GPU, tensor parallel 4
    python scripts/sft_v3_anchors.py --stage compare
    python scripts/sft_v3_anchors.py --stage votes       # GPU, tensor parallel 4
    python scripts/sft_v3_anchors.py --stage enrichment
    python scripts/sft_v3_anchors.py --stage vote_targets  # GPU, tensor parallel 4
    python scripts/sft_v3_anchors.py --stage vote_readout
    python scripts/sft_v3_anchors.py --stage vote_sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import teacher_anchor_check as TA  # noqa: E402

from hiring_bias_mitigation.data import prompts as P  # noqa: E402
from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.data.injection import ATTR_FREE  # noqa: E402
from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.generation import filters as F  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    require_output_root,
    resolve_output_path,
)

LANG = "en"
K = 8
#: The 2,992 English pool pairs every v2 arm was built on, with their v1 anchors.
POOL = "artifacts/semisynthetic-v2/onpolicy-full/rows.parquet"
#: v1's 12 attribute variants per pair, which the changed pairs are regenerated over.
INVARIANT_V1 = "artifacts/semisynthetic-v1/raw/en_invariant.parquet"
#: v1's raw anchor generations, kept unmodified by generate_training_data._dump_raw.
REFERENCE_V1 = "artifacts/semisynthetic-v1/raw/en_reference.parquet"
OUT = "artifacts/semisynthetic-v3"
REPORTS = REPO_ROOT / "reports" / "sft_v3_anchors"
LABELS = REPORTS / "inputs" / "claude_anchor_review_labels.csv"
PREREG = REPO_ROOT / "docs" / "SFT_V3_ANCHORS.md"

CANDIDATES = ("audit", "reference_no_decisive")
#: The redraw control: v1's own anchor prompt, drawn again.
CONTROL = "reference"
#: The pool-wide vote and its pre-registered readouts (the page's second addendum).
POOL_VOTE_SEED, POOL_VOTE_PARTS = 43, 4
UNSTABLE_MAX_MAJORITY = 5 / 8
PLATFORM_MAX_GAP, ENRICHMENT_MIN, SFT_RETRAIN_MIN_CHANGED = 0.03, 2.0, 0.10
#: The vote anchors' regenerated targets (the page's third addendum): below this yield they are
#: reported, and no SFT data is built from them.
VOTE_MIN_YIELD = 0.80
#: v1's published SFT split and Ukrainian raw pass (the page's fourth addendum). Never on Daint;
#: fetched from the project's public dataset at the revision the Daint cache records.
V1_HUB = "artifacts/semisynthetic-v1-hub"
HUB_REPO = "Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data"
HUB_REVISION = "996e555e1cf0d3b327f2280be6ef9cee36fddbbd"
SFT_V1 = {"train": f"{V1_HUB}/sft_train.parquet",
          "validation": "artifacts/semisynthetic-v1/sft_validation.parquet"}
SFT_V3 = {"swap": f"{OUT}/sft_v3_swap", "rebalanced": f"{OUT}/sft_v3_rebalanced"}
VARIANT_KEY = ["pair_id", "protected_attr", "condition"]
#: v1's assembly settings (configs/generation/teacher.yaml).
V1_SEED, V1_VAL_FRACTION = 42, 0.05
#: The arms whose data a follow-up would rebuild on vote anchors.
REBUILD_CANDIDATES = ("DPO r2em", "GRPO grpo-02")
DECISIVE = " Be decisive: a genuinely weak match is a reject."
#: The pre-registered thresholds and tie rule (docs/SFT_V3_ANCHORS.md).
C1_MIN, C2_MIN, C3_GAIN, TIE_MARGIN = 0.60, 0.90, 0.05, 0.02
SMALLER_EDIT = "reference_no_decisive"
#: The review's own counts, which the labels must reproduce.
EXPECTED_LABELS = {"rows": 391, "clear_hire_targets": 124, "clear_reject_targets": 34,
                   "control_reject": 20}
LABEL_COLUMNS = ["item", "pair_id", "set", "anchor", "student_majority", "decision", "fit",
                 "fit_bucket"]
PAIR_COLUMNS = ["pair_id", "candidate_id", "job_id", "job_position", "job_description", "cv",
                "reference_decision", "reference_feedback"]
#: Earlier training sets whose targets were the v1 anchors; reference_decision is that anchor.
TRAINING_SETS = {
    "SFT v1 targets, before balancing": "artifacts/semisynthetic-v2/onpolicy-full/rows.parquet",
    "DPO r1": "artifacts/semisynthetic-v2/onpolicy-hire32/balanced/dpo_v2_train.parquet",
    "DPO r2": "artifacts/semisynthetic-v2/onpolicy-hire32/margin_balanced/dpo_v2_train.parquet",
    "DPO r2em": "artifacts/semisynthetic-v2/round2-error-mass/dpo_round2_train.parquet",
    "DPO it2 (not trained)": "artifacts/semisynthetic-v2/onpolicy-it2/dpo_v2_train.parquet",
    "GRPO grpo-02": "artifacts/semisynthetic-v2/grpo-02/grpo_train.parquet",
    "RFT rft-01": "artifacts/semisynthetic-v2/rft-01/rft_train.parquet",
}


# ---- inputs ---------------------------------------------------------------------------------

def pool_pairs() -> pd.DataFrame:
    """One row per pool pair, its v1 anchor as v1_decision and v1_feedback."""
    rows = pd.read_parquet(resolve_output_path(POOL), columns=PAIR_COLUMNS)
    pairs = rows.drop_duplicates("pair_id").reset_index(drop=True)
    return pairs.rename(columns={"reference_decision": "v1_decision",
                                 "reference_feedback": "v1_feedback"})


def check_labels(labels: pd.DataFrame) -> pd.DataFrame:
    """Refuses labels that do not reproduce the review's own counts."""
    counts = {
        "rows": len(labels),
        "clear_hire_targets": int(clear_hire(labels).sum()),
        "clear_reject_targets": int(clear_reject(labels).sum()),
        "control_reject": int((labels["set"] == "control_reject").sum()),
    }
    if counts != EXPECTED_LABELS or labels["pair_id"].duplicated().any():
        raise ValueError(f"review labels do not match the pre-registration: {counts}")
    return labels


def clear_hire(labels: pd.DataFrame) -> pd.Series:
    return (labels["set"] == "target") & labels["fit"].isin([4, 5])


def clear_reject(labels: pd.DataFrame) -> pd.Series:
    return (labels["set"] == "target") & labels["fit"].isin([1, 2])


def reject_side(labels: pd.DataFrame) -> pd.Series:
    return clear_reject(labels) | (labels["set"] == "control_reject")


def load_labels() -> pd.DataFrame:
    return check_labels(pd.read_csv(LABELS))


def reviewed_pairs(labels: pd.DataFrame) -> pd.DataFrame:
    """The pool pairs the review graded, all of which must be in the pool."""
    pairs = pool_pairs()
    pairs = pairs[pairs["pair_id"].isin(labels["pair_id"])].reset_index(drop=True)
    if len(pairs) != len(labels):
        raise ValueError(f"only {len(pairs)} of {len(labels)} reviewed pairs are in the pool")
    return pairs


def set_reports(path: Path) -> None:
    """Points the reports and the labels at another directory, for a dry run."""
    global REPORTS, LABELS
    REPORTS = Path(path)
    LABELS = REPORTS / "inputs" / "claude_anchor_review_labels.csv"


# ---- prompts --------------------------------------------------------------------------------

def without_decisive(template: str) -> str:
    """REFERENCE_PROMPT with its one 'Be decisive' sentence removed, and nothing else."""
    if template.count(DECISIVE) != 1:
        raise ValueError("the anchor prompt no longer has the 'Be decisive' sentence just once")
    return template.replace(DECISIVE, "")


def candidate_prompts(pairs: pd.DataFrame, name: str) -> list[str]:
    """The attribute-free prompt a candidate puts to the teacher, one per pair."""
    records = pairs.to_dict("records")
    if name == "audit":
        # The benchmark's attr_free row (injection.build_eval_set), so the teacher reads exactly
        # what the student is asked.
        return [P.build_prompt({**r, "lang": LANG, "protected_group": ATTR_FREE,
                                "protected_group_label": "", "protected_attr": ATTR_FREE,
                                "condition": "attr_free", "implicit_injection": ""}, "baseline")
                for r in records]
    if name == "reference_no_decisive":
        template = without_decisive(SY.REFERENCE_PROMPT[LANG])
        fmt = SY._format_instructions(LANG)
        return [template.format(job_desc=r["job_description"], cv=r["cv"],
                                format_instructions=fmt) for r in records]
    if name == CONTROL:
        return SY.reference_prompts(pairs, LANG)
    raise ValueError(f"unknown prompt {name!r}; expected one of {(*CANDIDATES, CONTROL)}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prompts_sha256(prompts: list[str]) -> str:
    """A fingerprint of exactly what was sent, in order."""
    return _sha256("\x1e".join(prompts))


# ---- the vote -------------------------------------------------------------------------------

def draw_frame(pairs: pd.DataFrame, name: str, draws: list[list[str]]) -> pd.DataFrame:
    """One row per draw, parsed."""
    rows = [{"pair_id": pair, "prompt": name, "sample_index": j, "raw_output": text}
            for pair, texts in zip(pairs["pair_id"], draws) for j, text in enumerate(texts)]
    return TA.parse(pd.DataFrame(rows), LANG)


def new_anchors(draws: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Per pair: the vote, and the anchor the pre-registered rule gives.

    The majority over decided draws replaces the v1 anchor, with the first majority draw that
    passes filters.filter_reference as its feedback. A tie, a pair without a decided draw, and
    a majority none of whose draws passes that filter keep the v1 anchor; `source` says which.
    """
    decided = draws[draws["decision"].isin(TA.DECIDED)]
    hire = decided.assign(h=decided["decision"] == "hire").groupby("pair_id")["h"].sum()
    count = decided.groupby("pair_id").size()
    out = pairs[["pair_id", "v1_decision", "v1_feedback"]].copy()
    out["hire_votes"] = out["pair_id"].map(hire).fillna(0).astype(int)
    out["decided"] = out["pair_id"].map(count).fillna(0).astype(int)
    out["majority"] = np.select([2 * out["hire_votes"] > out["decided"],
                                 2 * out["hire_votes"] < out["decided"]],
                                ["hire", "reject"], "tie")
    candidates = draws.merge(out[["pair_id", "majority"]], on="pair_id")
    candidates = candidates[candidates["decision"] == candidates["majority"]]
    candidates = candidates.sort_values(["pair_id", "sample_index"]).reset_index(drop=True)
    kept = (F.filter_reference(candidates["raw_output"].tolist(),
                               candidates[["pair_id", "sample_index"]], LANG)[0]
            if len(candidates) else pd.DataFrame())
    first = (kept.drop_duplicates("pair_id").set_index("pair_id") if len(kept)
             else pd.DataFrame(columns=["reference_feedback", "sample_index"]))
    out["vote_feedback"] = out["pair_id"].map(first["reference_feedback"])
    out["vote_sample_index"] = out["pair_id"].map(first["sample_index"])
    out["source"] = np.select(
        [out["decided"] == 0, out["majority"] == "tie", out["vote_feedback"].isna()],
        ["no_decided_draw", "tie", "no_usable_feedback"], "vote")
    vote = out["source"] == "vote"
    out["new_decision"] = out["majority"].where(vote, out["v1_decision"])
    out["new_feedback"] = out["vote_feedback"].where(vote, out["v1_feedback"])
    out["changed"] = out["new_decision"] != out["v1_decision"]
    return out


# ---- stage 0a: choosing the prompt ----------------------------------------------------------

def criteria(anchors: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """C1 and C2 of the pre-registration, for one candidate's anchors on the reviewed pairs."""
    m = labels.merge(anchors[["pair_id", "new_decision", "source"]], on="pair_id", how="left",
                     validate="one_to_one")
    if m["new_decision"].isna().any():
        raise ValueError("a reviewed pair has no anchor")
    hire_side, reject = m[clear_hire(m)], m[reject_side(m)]
    c1 = float((hire_side["new_decision"] == "hire").mean())
    c2 = float((reject["new_decision"] == "reject").mean())
    by_cell = (m.assign(new_hire=m["new_decision"] == "hire")
               .groupby(["set", "fit_bucket"])["new_hire"].agg(["size", "mean"]).reset_index()
               .rename(columns={"size": "pairs", "mean": "new_anchor_hire_share"}))
    # Not a criterion: C2 on the pairs the reviewer itself graded fit 1 to 2, which leaves out
    # the both-reject controls it graded higher.
    graded_reject = m[m["fit"].isin([1, 2]) & m["set"].isin(["target", "control_reject"])]
    return {
        "c1": {"pairs": len(hire_side), "hired": int((hire_side["new_decision"] == "hire").sum()),
               "share": c1, "min": C1_MIN, "pass": c1 >= C1_MIN},
        "c2": {"pairs": len(reject), "rejected": int((reject["new_decision"] == "reject").sum()),
               "share": c2, "min": C2_MIN, "pass": c2 >= C2_MIN},
        "passes": bool(c1 >= C1_MIN and c2 >= C2_MIN),
        "score": c1 + c2 - 1,
        "hire_share": float((m["new_decision"] == "hire").mean()),
        "sources": {k: int(v) for k, v in m["source"].value_counts().items()},
        "c2_on_reviewer_fit_1_2_only": {
            "pairs": len(graded_reject),
            "rejected": int((graded_reject["new_decision"] == "reject").sum())},
        "by_set_and_fit": by_cell.to_dict("records"),
    }


def vote_splits(anchors: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """Hire votes out of K on the pairs each criterion counts."""
    votes = anchors.set_index("pair_id")["hire_votes"]
    sets = {"C1 clear-hire targets": clear_hire(labels),
            "C2 clear-reject targets": clear_reject(labels),
            "C2 both-reject controls": labels["set"] == "control_reject"}
    return {name: {f"{h} of {K}": int(n) for h, n in
                   votes.loc[labels.loc[mask, "pair_id"]].value_counts().sort_index().items()}
            for name, mask in sets.items()}


def paired_flips(before: pd.Series, after: pd.Series) -> dict:
    """Two verdicts on the same pairs: the flips each way, and the exact two-sided McNemar p
    (the binomial test on the discordant pairs, as analysis/stability.py runs its sign test)."""
    from scipy.stats import binomtest

    if not before.index.equals(after.index):
        raise ValueError("the two arms must be read on the same pairs, in the same order")
    was, now = before.eq("hire"), after.eq("hire")
    up, down = int((~was & now).sum()), int((was & ~now).sum())
    return {"pairs": int(before.size), "before_hired": int(was.sum()),
            "after_hired": int(now.sum()), "reject_to_hire": up, "hire_to_reject": down,
            "mcnemar_p": float(binomtest(up, up + down, 0.5).pvalue) if up + down else None}


def unstable(anchors: pd.DataFrame) -> pd.Series:
    """Overturned by the vote, a majority of at most 5 in 8 decided draws, or no decided draw."""
    decided = anchors["decided"]
    majority = np.maximum(anchors["hire_votes"], decided - anchors["hire_votes"])
    weak = majority <= UNSTABLE_MAX_MAJORITY * decided
    return anchors["changed"] | weak | decided.eq(0)


def platform_check(draws: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Daint's per-draw hire share against the v1 anchors' on the same pairs."""
    decided = draws[draws["decision"].isin(TA.DECIDED)]
    v1 = float((pairs["v1_decision"] == "hire").mean())
    now = float((decided["decision"] == "hire").mean())
    return {"pairs": len(pairs), "decided_draws": len(decided), "v1_hire_share": v1,
            "per_draw_hire_share": now, "gap": now - v1, "max_gap": PLATFORM_MAX_GAP,
            "pass": abs(now - v1) <= PLATFORM_MAX_GAP}


def single_draws(draws: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    """Not pre-registered, added after the platform check failed: one Daint draw against
    another, and v1's draw against one Daint draw. Without a platform shift the two disagree
    about equally often, and both symmetrically."""
    d = draws[draws["sample_index"].isin([0, 1])].pivot(index="pair_id", columns="sample_index",
                                                         values="decision")
    both = d[d[0].isin(TA.DECIDED) & d[1].isin(TA.DECIDED)]
    first = d.loc[d[0].isin(TA.DECIDED), 0]
    v1 = pairs.set_index("pair_id")["v1_decision"]
    out = {"daint_draw_0_then_1": paired_flips(both[0], both[1]),
           "v1_then_daint_draw_0": paired_flips(v1.loc[first.index], first)}
    for r in out.values():
        r["disagreement"] = (r["reject_to_hire"] + r["hire_to_reject"]) / r["pairs"]
    return out


def fenced(raw: pd.Series) -> pd.Series:
    """Whether a raw answer wraps its JSON in a Markdown code fence."""
    return raw.astype(str).str.contains("```", regex=False)


def output_format(draws: pd.DataFrame, v1_reference: pd.DataFrame, v1_invariant: pd.DataFrame,
                  pairs: pd.DataFrame) -> dict:
    """Not pre-registered, added after the platform check failed: how often the answer is
    wrapped in a code fence, in v1's raw generations and in Daint's draws, and Daint's hire
    share by format. Both setups were given the same nominal prompt and decoding; a different
    fence rate says the model behaved differently, not only that its decisions moved."""
    def count(raw: pd.Series) -> dict:
        f = fenced(raw)
        return {"outputs": int(raw.size), "fenced": int(f.sum()),
                "fenced_share": float(f.mean()) if raw.size else None}

    on_pool = v1_reference[v1_reference["pair_id"].isin(pairs["pair_id"])]
    f, decided = fenced(draws["raw_output"]), draws["decision"].isin(TA.DECIDED)
    hire = {name: float((draws.loc[mask & decided, "decision"] == "hire").mean())
            for name, mask in (("fenced", f), ("bare", ~f))}
    return {"v1 anchors, pool pairs": count(on_pool["raw_output"]),
            "v1 anchors, all generated": count(v1_reference["raw_output"]),
            "v1 invariant targets": count(v1_invariant["raw_output"]),
            "Daint anchor draws": count(draws["raw_output"]),
            "daint_hire_share_by_format": hire}


def enrichment(rows: pd.DataFrame, unstable_pairs: pd.Series, pool_share: float) -> dict:
    """A training set's share of rows on unstable anchors, against the pool's share of pairs."""
    on = rows["pair_id"].map(unstable_pairs)
    if on.isna().any():
        raise ValueError(f"{int(on.isna().sum())} rows are on pairs outside the pool")
    share = float(on.astype(bool).mean())
    ratio = share / pool_share if pool_share else None
    return {"rows": len(rows), "unstable_rows": int(on.astype(bool).sum()),
            "unstable_share": share, "enrichment": ratio,
            "a_holds": bool(ratio is not None and ratio >= ENRICHMENT_MIN)}


def side_enrichment(rows: pd.DataFrame, anchors: pd.DataFrame) -> dict:
    """Not pre-registered: enrichment within each old target side, against the pool's pairs on
    that side. The pre-registered ratio mixes the sides, while the preference sets are balanced
    and the pool is mostly reject."""
    by_pair = anchors.set_index("pair_id")
    out = {}
    for side in TA.DECIDED:
        pool = float(by_pair.loc[by_pair["v1_decision"] == side, "unstable"].mean())
        mine = rows[rows["reference_decision"] == side]["pair_id"].map(by_pair["unstable"])
        share = float(mine.astype(bool).mean()) if len(mine) else None
        out[side] = {"rows": len(mine), "unstable_share": share, "pool_share": pool,
                     "enrichment": share / pool if share is not None and pool else None}
    return out


def comparison_sets(labels: pd.DataFrame) -> dict[str, pd.Series]:
    """The pairs each comparison of the redraw control is read on."""
    return {"all 391": labels["set"].notna(), "target 337": labels["set"] == "target",
            "C1 clear-hire targets": clear_hire(labels),
            "borderline targets (fit 3)": (labels["set"] == "target") & (labels["fit"] == 3),
            "C2 clear-reject targets": clear_reject(labels),
            "C2 both-reject controls": labels["set"] == "control_reject"}


def choose(results: dict[str, dict]) -> str | None:
    """The pre-registered rule: a passing candidate, the higher C1 + C2 - 1 if both pass, and
    the smaller edit when their scores are within TIE_MARGIN. None when neither passes."""
    passing = {name: r["score"] for name, r in results.items() if r["passes"]}
    if not passing:
        return None
    if len(passing) == 1:
        return next(iter(passing))
    ranked = sorted(passing, key=passing.get, reverse=True)
    if passing[ranked[0]] - passing[ranked[1]] < TIE_MARGIN:
        return SMALLER_EDIT
    return ranked[0]


# ---- stage 0b: the changed pairs' targets ---------------------------------------------------

def changed_variants(anchors: pd.DataFrame, invariant_v1: pd.DataFrame) -> pd.DataFrame:
    """v1's 12 variants of each pair whose verdict changed, pinned to the new anchor."""
    changed = anchors[anchors["changed"]].set_index("pair_id")
    variants = invariant_v1[invariant_v1["pair_id"].isin(changed.index)]
    variants = variants.drop(columns=["raw_output"], errors="ignore").copy()
    missing = set(changed.index) - set(variants["pair_id"])
    if missing:
        raise ValueError(f"{len(missing)} changed pairs have no v1 variants")
    variants["v1_reference_decision"] = variants["reference_decision"]
    variants["reference_decision"] = variants["pair_id"].map(changed["new_decision"])
    variants["reference_feedback"] = variants["pair_id"].map(changed["new_feedback"])
    return variants.reset_index(drop=True)


# ---- stage 1: the gate ----------------------------------------------------------------------

def flip_readout(rows: pd.DataFrame, anchors: pd.DataFrame) -> dict:
    """Rows of an earlier training set whose target the new anchor flips, by the old target."""
    new = rows["pair_id"].map(anchors.set_index("pair_id")["new_decision"])
    if new.isna().any():
        raise ValueError(f"{int(new.isna().sum())} rows are on pairs outside the pool")
    out = {"rows": len(rows), "pairs": int(rows["pair_id"].nunique()),
           "flipped": int((new != rows["reference_decision"]).sum())}
    for side in TA.DECIDED:
        on = rows["reference_decision"] == side
        out[f"old_{side}"] = int(on.sum())
        out[f"old_{side}_flipped"] = int((on & (new != side)).sum())
    return out


def gate_summary(anchors: pd.DataFrame, labels: pd.DataFrame) -> dict:
    """C1 to C3 on the anchors training would use, and the pool's vote."""
    v1_hire = float((anchors["v1_decision"] == "hire").mean())
    new_hire = float((anchors["new_decision"] == "hire").mean())
    c3 = {"v1_hire_share": v1_hire, "new_hire_share": new_hire, "gain": new_hire - v1_hire,
          "min_gain": C3_GAIN, "pass": new_hire - v1_hire >= C3_GAIN - 1e-12}
    crit = criteria(anchors, labels)
    full = anchors[anchors["decided"] == K]
    return {
        "pairs": len(anchors),
        "c1": crit["c1"], "c2": crit["c2"], "c3": c3,
        "passes": bool(crit["c1"]["pass"] and crit["c2"]["pass"] and c3["pass"]),
        "reviewed_by_set_and_fit": crit["by_set_and_fit"],
        "v1_by_new": {f"v1 {a}": {f"new {b}": int(n) for b, n in g.value_counts().items()}
                      for a, g in anchors.groupby("v1_decision")["new_decision"]},
        "changed": int(anchors["changed"].sum()),
        "sources": {k: int(v) for k, v in anchors["source"].value_counts().items()},
        "hire_votes_when_all_decided": {f"{h} of {K}": int(n) for h, n in
                                        full["hire_votes"].value_counts().sort_index().items()},
        "unanimous_share": float(full["hire_votes"].isin([0, K]).mean()) if len(full) else None,
    }


def invariant_cells(kept: pd.DataFrame, variants: pd.DataFrame) -> list[dict]:
    """Yield of the regenerated targets by group, condition and new verdict."""
    keys = ["protected_group", "condition", "reference_decision"]
    total = variants.groupby(keys).size().rename("variants")
    got = (kept.groupby(keys).size().rename("kept") if len(kept)
           else pd.Series(dtype=int, name="kept"))
    table = pd.concat([total, got], axis=1).fillna(0).astype(int).reset_index()
    table["yield"] = table["kept"] / table["variants"]
    return table.to_dict("records")


# ---- bookkeeping ----------------------------------------------------------------------------

def code_state() -> dict:
    """The commit, tracked changes, and untracked files under scripts/, src/ and docs/."""
    state = TA.code_state()
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "--others",
                              "--exclude-standard", "--", "scripts", "src", "docs"],
                             capture_output=True, text=True, check=True).stdout
        state["untracked"] = [line for line in out.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        state["untracked"] = None
    state["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return state


def provenance() -> dict:
    return {"written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
            "code": code_state()}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _backend(generation: dict = TA.V1_GENERATION):
    return BK.build_backend({"model": TA.TEACHER, "generation": generation, **TA.SERVING})


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _table(records: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(r.get(c, "")) for c in columns) + " |" for r in records]
    return lines


# ---- stages ---------------------------------------------------------------------------------

def stage_labels(args) -> None:
    review = Path(args.review)
    labels = check_labels(pd.read_csv(review)[LABEL_COLUMNS])
    LABELS.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(LABELS, index=False)
    _write_json(LABELS.with_suffix(".json"), {
        "source": str(review), "source_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
        "columns": LABEL_COLUMNS, "counts": EXPECTED_LABELS,
        "note": "decision and fit are the blind reviewer's; anchor and student_majority are the "
                "v1 anchor and the SFT student's attribute-free majority on the pair",
        **provenance()})
    print(f"wrote {LABELS}: {len(labels)} reviewed pairs")


def stage_select(args) -> None:
    require_output_root()
    labels = load_labels()
    pairs = pool_pairs()
    pairs = pairs[pairs["pair_id"].isin(labels["pair_id"])].reset_index(drop=True)
    if len(pairs) != len(labels):
        raise ValueError(f"only {len(pairs)} of {len(labels)} reviewed pairs are in the pool")
    out = Path(resolve_output_path(OUT)) / "anchors"
    out.mkdir(parents=True, exist_ok=True)
    backend, frames, fingerprints, seconds = _backend(), [], {}, {}
    try:
        for name in CANDIDATES:
            prompts = candidate_prompts(pairs, name)
            start = time.time()
            frames.append(draw_frame(pairs, name, backend.generate_samples(prompts, K)))
            seconds[name] = round(time.time() - start, 1)
            fingerprints[name] = prompts_sha256(prompts)
    finally:
        backend.close()
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(out / "select_draws.parquet", index=False)
    _write_json(out / "select_draws.meta.json", {
        "stage": "select", "model": TA.TEACHER, "pairs": len(pairs), "k": K,
        "decoding": TA.V1_GENERATION, "serving": TA.SERVING, "prompts_sha256": fingerprints,
        "generation_seconds": seconds, **provenance()})
    print(f"wrote {out / 'select_draws.parquet'}: {len(frame)} draws")


def stage_score(args) -> None:
    labels = load_labels()
    draws = pd.read_parquet(Path(resolve_output_path(OUT)) / "anchors" / "select_draws.parquet")
    pairs = pool_pairs()
    pairs = pairs[pairs["pair_id"].isin(labels["pair_id"])].reset_index(drop=True)
    results, anchors = {}, {}
    for name in CANDIDATES:
        mine = draws[draws["prompt"] == name]
        anchors[name] = new_anchors(mine, pairs)
        results[name] = {**criteria(anchors[name], labels),
                         "hire_votes": vote_splits(anchors[name], labels),
                         "undecided_draw_share": float((~mine["decision"].isin(TA.DECIDED)).mean())}
    hired = pd.concat([a.set_index("pair_id")["new_decision"].eq("hire").rename(n)
                       for n, a in anchors.items()], axis=1)
    lab = labels.set_index("pair_id")
    both = {name: int(hired.loc[lab.index[mask.to_numpy()]].all(axis=1).sum()) for name, mask in
            [("C1 clear-hire targets", clear_hire(labels)),
             ("C2 clear-reject targets", clear_reject(labels)),
             ("C2 both-reject controls", labels["set"] == "control_reject")]}
    summary = {"chosen": choose(results), "candidates": results, "hired_by_both": both,
               "v1_hire_share": float((pairs["v1_decision"] == "hire").mean()),
               "rule": {"c1_min": C1_MIN, "c2_min": C2_MIN, "tie_margin": TIE_MARGIN,
                        "smaller_edit": SMALLER_EDIT},
               "draws_meta": json.loads((Path(resolve_output_path(OUT)) / "anchors" /
                                         "select_draws.meta.json").read_text()),
               **provenance()}
    _write_json(REPORTS / "select.json", summary)
    (REPORTS / "select.md").write_text(render_select(summary), encoding="utf-8")
    print(render_select(summary))


def stage_generate(args) -> None:
    require_output_root()
    chosen = json.loads((REPORTS / "select.json").read_text())["chosen"]
    if chosen is None:
        raise SystemExit("stage 0a chose no prompt, so the pre-registration stops before 0b")
    pairs = pool_pairs()
    out = Path(resolve_output_path(OUT))
    (out / "anchors").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    draws_path, draws_meta = out / "anchors" / "draws.parquet", out / "anchors" / "draws.meta.json"
    prompts = candidate_prompts(pairs, chosen)
    backend, seconds = _backend(), {}
    try:
        saved = json.loads(draws_meta.read_text()) if draws_meta.exists() else {}
        if draws_path.exists() and saved.get("prompts_sha256") == prompts_sha256(prompts):
            draws = pd.read_parquet(draws_path)
            print(f"resuming from {draws_path}: {len(draws)} draws")
        else:
            start = time.time()
            draws = draw_frame(pairs, chosen, backend.generate_samples(prompts, K))
            seconds["anchors"] = round(time.time() - start, 1)
            draws.to_parquet(draws_path, index=False)
            _write_json(draws_meta, {
                "stage": "generate", "prompt": chosen, "model": TA.TEACHER, "pairs": len(pairs),
                "k": K, "decoding": TA.V1_GENERATION, "serving": TA.SERVING,
                "prompts_sha256": prompts_sha256(prompts), "generation_seconds": seconds,
                **provenance()})
        anchors = new_anchors(draws, pairs)
        anchors.to_parquet(out / "anchors" / "anchors.parquet", index=False)
        variants = changed_variants(anchors, pd.read_parquet(resolve_output_path(INVARIANT_V1)))
        print(f"{int(anchors['changed'].sum())} pairs changed verdict; "
              f"{len(variants)} invariant variants to regenerate")
        start = time.time()
        raw = backend.generate(SY.invariant_prompts(variants, LANG)) if len(variants) else []
        seconds["invariant"] = round(time.time() - start, 1)
    finally:
        backend.close()
    kept, report = F.filter_invariant(raw, variants, LANG)
    variants.assign(raw_output=raw).to_parquet(out / "raw" / "en_invariant_changed.parquet",
                                               index=False)
    kept.to_parquet(out / "raw" / "en_invariant_changed_kept.parquet", index=False)
    _write_json(out / "raw" / "en_invariant_changed.meta.json", {
        "stage": "generate", "prompt": chosen, "model": TA.TEACHER,
        "decoding": TA.V1_GENERATION, "variants": len(variants), "kept": len(kept),
        "filter_report": report, "generation_seconds": seconds, **provenance()})
    print(f"wrote {out}: {len(draws)} draws, {len(kept)} of {len(variants)} targets kept")


def stage_gate(args) -> None:
    out = Path(resolve_output_path(OUT))
    anchors = pd.read_parquet(out / "anchors" / "anchors.parquet")
    labels = load_labels()
    summary = {"prompt": json.loads((out / "anchors" / "draws.meta.json").read_text())["prompt"],
               **gate_summary(anchors, labels)}
    raw_path = out / "raw" / "en_invariant_changed.parquet"
    if raw_path.exists():
        variants = pd.read_parquet(raw_path)
        kept = pd.read_parquet(out / "raw" / "en_invariant_changed_kept.parquet")
        meta = json.loads((out / "raw" / "en_invariant_changed.meta.json").read_text())
        summary["invariant"] = {"variants": len(variants), "kept": len(kept),
                                "filter_report": meta["filter_report"],
                                "by_cell": invariant_cells(kept, variants)}
    summary["training_sets"] = {}
    for name, path in TRAINING_SETS.items():
        full = resolve_output_path(path)
        if full.exists():
            rows = pd.read_parquet(full, columns=["pair_id", "reference_decision"])
            summary["training_sets"][name] = {"path": path, **flip_readout(rows, anchors)}
    summary.update(provenance())
    _write_json(REPORTS / "gate.json", summary)
    (REPORTS / "gate.md").write_text(render_gate(summary), encoding="utf-8")
    print(render_gate(summary))


# ---- the vote anchors' targets (the page's third addendum) -----------------------------------

def _vote_anchors_path() -> Path:
    return _votes_path().parent / f"pool_votes_seed{POOL_VOTE_SEED}_anchors.parquet"


def _vote_raw_path(suffix: str = ".parquet") -> Path:
    return Path(resolve_output_path(OUT)) / "raw" / f"en_invariant_vote_changed{suffix}"


def within_pair_identical(frame: pd.DataFrame, text: str) -> float | None:
    """Share of same-pair variant pairs whose text is byte-identical."""
    same = total = 0
    for _, group in frame.groupby("pair_id"):
        n = len(group)
        counts = group[text].value_counts()
        total += n * (n - 1) // 2
        same += int((counts * (counts - 1) // 2).sum())
    return same / total if total else None


def copies_reference(frame: pd.DataFrame, text: str) -> float | None:
    """Share of rows whose rationale repeats, verbatim, the reference feedback the prompt quotes."""
    if not len(frame):
        return None
    return float((frame[text].str.strip() == frame["reference_feedback"].str.strip()).mean())


def target_profile(raw: list[str], variants: pd.DataFrame) -> dict:
    """Yield, drop reasons and text readouts of one set of invariant targets."""
    kept, report = F.filter_invariant(raw, variants, LANG)
    return {"variants": len(variants), "kept": len(kept), "yield": report["yield"],
            "dropped_by_reason": report["dropped_by_reason"],
            "fenced_raw": int(fenced(pd.Series(raw, dtype=str)).sum()),
            "within_pair_identical_rationale": within_pair_identical(kept, "chosen_feedback"),
            "copies_reference_feedback": copies_reference(kept, "chosen_feedback")}


def stage_vote_targets(args) -> None:
    """v1's invariant procedure on the pairs whose verdict the pool-wide vote changes."""
    require_output_root()
    anchors = pd.read_parquet(_vote_anchors_path())
    variants = changed_variants(anchors, pd.read_parquet(resolve_output_path(INVARIANT_V1)))
    prompts = SY.invariant_prompts(variants, LANG)
    raw_path, meta_path = _vote_raw_path(), _vote_raw_path(".meta.json")
    saved = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if raw_path.exists() and saved.get("prompts_sha256") == prompts_sha256(prompts):
        print(f"resuming: {raw_path} is complete")
        return
    print(f"{int(anchors['changed'].sum())} pairs changed verdict; "
          f"{len(variants)} invariant variants to regenerate")
    backend = _backend()
    try:
        start = time.time()
        raw = backend.generate(prompts)
        seconds = round(time.time() - start, 1)
        engine = getattr(getattr(getattr(backend, "_llm", None), "llm_engine", None),
                         "vllm_config", None)
        v2_runner = getattr(engine, "use_v2_model_runner", None)
    finally:
        backend.close()
    kept, report = F.filter_invariant(raw, variants, LANG)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    variants.assign(raw_output=raw, seed=[BK.per_prompt_seed(TA.V1_GENERATION["seed"], p)
                                          for p in prompts]).to_parquet(raw_path, index=False)
    kept.to_parquet(_vote_raw_path("_kept.parquet"), index=False)
    _write_json(meta_path, {
        "stage": "vote_targets", "anchors": str(_vote_anchors_path()), "model": TA.TEACHER,
        "decoding": TA.V1_GENERATION, "serving": TA.SERVING,
        "seeding": "per_prompt_seed(42, prompt): VLLMBackend.generate since 7112197",
        "use_v2_model_runner": v2_runner, "variants": len(variants), "kept": len(kept),
        "prompts_sha256": prompts_sha256(prompts), "filter_report": report,
        "generation_seconds": seconds, **provenance()})
    print(f"wrote {raw_path}: {len(kept)} of {len(variants)} targets kept")


def stage_vote_readout(args) -> None:
    """The regenerated targets against v1's on the same variants, and the anchors' readouts."""
    anchors = pd.read_parquet(_vote_anchors_path())
    recomputed = new_anchors(pd.read_parquet(_votes_path().with_suffix(".parquet")), pool_pairs())
    cols = ["pair_id", "new_decision", "new_feedback", "changed", "source"]
    same_as_saved = bool(recomputed[cols].reset_index(drop=True)
                         .equals(anchors[cols].reset_index(drop=True)))
    new = pd.read_parquet(_vote_raw_path())
    meta = json.loads(_vote_raw_path(".meta.json").read_text())
    v1_raw = pd.read_parquet(resolve_output_path(INVARIANT_V1))
    v1 = v1_raw[v1_raw["pair_id"].isin(new["pair_id"].unique())].reset_index(drop=True)
    profile_new = target_profile(new["raw_output"].tolist(), new.drop(columns=["raw_output"]))
    profile_v1 = target_profile(v1["raw_output"].tolist(), v1.drop(columns=["raw_output"]))
    kept = pd.read_parquet(_vote_raw_path("_kept.parquet"))
    summary = {
        "anchors": {"path": str(_vote_anchors_path()), "recomputed_equals_saved": same_as_saved,
                    **gate_summary(anchors, load_labels())},
        "targets": {"vote": profile_new, "v1_same_variants": profile_v1,
                    "min_yield": VOTE_MIN_YIELD, "yield_floor_met":
                        profile_new["yield"] >= VOTE_MIN_YIELD,
                    "by_cell": invariant_cells(kept, new), "distinct_seeds":
                        int(new["seed"].nunique()), "generation": {
                            k: meta[k] for k in ("seeding", "use_v2_model_runner",
                                                 "generation_seconds", "code", "written")}},
        "training_sets": {},
        **provenance()}
    for name, path in TRAINING_SETS.items():
        full = resolve_output_path(path)
        if full.exists():
            rows = pd.read_parquet(full, columns=["pair_id", "reference_decision"])
            summary["training_sets"][name] = {"path": path, **flip_readout(rows, anchors)}
    _write_json(REPORTS / "vote_targets.json", summary)
    (REPORTS / "vote_targets.md").write_text(render_vote_targets(summary), encoding="utf-8")
    print(render_vote_targets(summary))


# ---- the SFT-v3 data sets (the page's fourth addendum) ---------------------------------------

def v1_invariant_pool() -> pd.DataFrame:
    """v1's filtered invariant targets, English then Ukrainian, as assemble_datasets made them.

    The English filter has changed since (whole-word attribute matching keeps 51 more rows), so
    the English rows are restricted to v1's own kept set, the pool every v2 arm was built on.
    """
    members = pd.read_parquet(resolve_output_path(POOL), columns=VARIANT_KEY)
    members = members.set_index(VARIANT_KEY).index
    frames = []
    for lang, path in ((LANG, INVARIANT_V1), ("uk", f"{V1_HUB}/raw/uk_invariant.parquet")):
        frame = pd.read_parquet(resolve_output_path(path))
        raw = frame.pop("raw_output").tolist()
        kept, _ = F.filter_invariant(raw, frame, lang)
        if lang == LANG:
            kept = kept[kept.set_index(VARIANT_KEY).index.isin(members)]
        frames.append(kept)
    return pd.concat(frames, ignore_index=True)


def assemble_sft(invariant: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """v1's SFT assembly: build, balance both languages together, split by candidate."""
    sft, balance = F.balance_decisions(D.build_sft_dataset(invariant), seed=V1_SEED,
                                       column="decision")
    train, validation = D.train_val_split(sft, val_fraction=V1_VAL_FRACTION, seed=V1_SEED)
    return train, validation, balance


def with_vote_targets(invariant: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """The pooled invariant rows with each changed English variant's target replaced in place."""
    by_key = targets.set_index(VARIANT_KEY)
    english = invariant.index[invariant["lang"] == LANG]
    keys = pd.MultiIndex.from_frame(invariant.loc[english, VARIANT_KEY])
    hit = keys.isin(by_key.index)
    new = by_key.loc[keys[hit]]
    out = invariant.copy()
    for col in ("chosen_decision", "chosen_feedback", "reference_decision", "reference_feedback"):
        out.loc[english[hit], col] = new[col].to_numpy()
    return out, int(hit.sum())


def swap_targets(split: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """An SFT split with each changed variant's completion replaced; rows and order kept."""
    rows = D.build_sft_dataset(targets).set_index(["lang", *VARIANT_KEY])
    keys = pd.MultiIndex.from_frame(split[["lang", *VARIANT_KEY]])
    hit = keys.isin(rows.index)
    new = rows.loc[keys[hit]]
    if not (new["prompt"].to_numpy() == split.loc[hit, "prompt"].to_numpy()).all():
        raise ValueError("a changed variant's prompt differs from v1's; the swap would move it")
    out = split.copy()
    out.loc[hit, "completion"] = new["completion"].to_numpy()
    out.loc[hit, "decision"] = new["decision"].to_numpy()
    return out, int(hit.sum())


def split_readout(new: pd.DataFrame, v1: pd.DataFrame) -> dict:
    """One split against v1's: composition, and how many rows are v1's unchanged."""
    key = ["lang", *VARIANT_KEY]
    joined = new.merge(v1[[*key, "completion"]], on=key, how="left", suffixes=("", "_v1"))
    english = new[new["lang"] == LANG]
    return {
        "rows": len(new), "v1_rows": len(v1),
        "by_lang_and_decision": {f"{lang} {d}": int(n) for (lang, d), n in
                                 new.groupby(["lang", "decision"]).size().items()},
        "english_rows": len(english),
        "english_hire_share": float((english["decision"] == "hire").mean()),
        "v1_english_hire_share": float((v1.loc[v1["lang"] == LANG, "decision"] == "hire").mean()),
        "identical_to_v1": int((joined["completion"] == joined["completion_v1"]).sum()),
        "variant_in_v1_completion_changed": int((joined["completion_v1"].notna()
                                                 & (joined["completion"]
                                                    != joined["completion_v1"])).sum()),
        "variant_not_in_v1": int(joined["completion_v1"].isna().sum()),
        "candidates_shared_with_v1": len(set(new["candidate_id"]) & set(v1["candidate_id"])),
        "candidates": int(new["candidate_id"].nunique())}


def stage_vote_sft(args) -> None:
    """Both SFT-v3 data sets, after the control rebuilds v1's published split exactly."""
    require_output_root()
    v1 = {split: pd.read_parquet(resolve_output_path(path)) for split, path in SFT_V1.items()}
    targets = pd.read_parquet(_vote_raw_path("_kept.parquet"))
    invariant = v1_invariant_pool()
    train, validation, balance_v1 = assemble_sft(invariant)
    control = {split: bool(list(rebuilt.columns) == list(v1[split].columns)
                           and rebuilt.equals(v1[split]))
               for split, rebuilt in (("train", train), ("validation", validation))}
    if not all(control.values()):
        raise SystemExit(f"the control does not rebuild v1's published SFT split: {control}")
    swapped = {split: swap_targets(frame, targets) for split, frame in v1.items()}
    pooled, replaced = with_vote_targets(invariant, targets)
    train_r, validation_r, balance_r = assemble_sft(pooled)
    variants = {"swap": {"train": swapped["train"][0], "validation": swapped["validation"][0]},
                "rebalanced": {"train": train_r, "validation": validation_r}}
    summary = {"control_equals_published": control, "balance_v1": balance_v1,
               "targets": len(targets), "replaced_in_pool": replaced,
               "swapped_rows": {split: n for split, (_, n) in swapped.items()},
               "balance_rebalanced": balance_r, "variants": {},
               "inputs": {"v1_split": SFT_V1, "hub": {"repo": HUB_REPO, "revision": HUB_REVISION},
                          "targets": str(_vote_raw_path("_kept.parquet"))},
               **provenance()}
    for name, splits in variants.items():
        out = Path(resolve_output_path(SFT_V3[name]))
        out.mkdir(parents=True, exist_ok=True)
        for split, frame in splits.items():
            assert_no_leakage(frame, f"sft_v3_anchors.vote_sft.{name}.{split}")
            frame.to_parquet(out / f"sft_{split}.parquet", index=False)
        summary["variants"][name] = {"path": SFT_V3[name], **{
            split: split_readout(frame, v1[split]) for split, frame in splits.items()}}
        _write_json(out / "sft_v3.meta.json", {"variant": name, **{
            k: summary[k] for k in ("targets", "replaced_in_pool", "inputs", "written",
                                    "prereg_sha256", "code")}})
    _write_json(REPORTS / "vote_sft.json", summary)
    (REPORTS / "vote_sft.md").write_text(render_vote_sft(summary), encoding="utf-8")
    print(render_vote_sft(summary))


def stage_control(args) -> None:
    require_output_root()
    pairs = reviewed_pairs(load_labels())
    out = Path(resolve_output_path(OUT)) / "anchors"
    out.mkdir(parents=True, exist_ok=True)
    prompts = candidate_prompts(pairs, CONTROL)
    backend = _backend()
    try:
        start = time.time()
        frame = draw_frame(pairs, CONTROL, backend.generate_samples(prompts, K))
        seconds = round(time.time() - start, 1)
    finally:
        backend.close()
    frame.to_parquet(out / "control_draws.parquet", index=False)
    _write_json(out / "control_draws.meta.json", {
        "stage": "control", "model": TA.TEACHER, "pairs": len(pairs), "k": K,
        "decoding": TA.V1_GENERATION, "serving": TA.SERVING,
        "prompts_sha256": {CONTROL: prompts_sha256(prompts)},
        "generation_seconds": {CONTROL: seconds}, **provenance()})
    print(f"wrote {out / 'control_draws.parquet'}: {len(frame)} draws")


def stage_compare(args) -> None:
    labels = load_labels()
    pairs = reviewed_pairs(labels)
    out = Path(resolve_output_path(OUT)) / "anchors"
    draws = pd.concat([pd.read_parquet(out / "select_draws.parquet"),
                       pd.read_parquet(out / "control_draws.parquet")], ignore_index=True)
    arms = (CONTROL, *CANDIDATES)
    anchors = {name: new_anchors(draws[draws["prompt"] == name], pairs) for name in arms}
    verdict = {name: a.set_index("pair_id")["new_decision"] for name, a in anchors.items()}
    v1 = pairs.set_index("pair_id")["v1_decision"]
    sets = {name: labels.loc[mask, "pair_id"].tolist()
            for name, mask in comparison_sets(labels).items()}
    summary = {
        "arms": {name: {**criteria(anchors[name], labels),
                        "hire_votes": vote_splits(anchors[name], labels)} for name in arms},
        "redraw_effect": [{"set": s, **paired_flips(v1.loc[ids], verdict[CONTROL].loc[ids])}
                          for s, ids in sets.items()],
        "prompt_effect": [{"candidate": c, "set": s,
                           **paired_flips(verdict[CONTROL].loc[ids], verdict[c].loc[ids])}
                          for c in CANDIDATES for s, ids in sets.items()],
        "draws_meta": {name: json.loads((out / f"{name}_draws.meta.json").read_text())
                       for name in ("select", "control")},
        **provenance()}
    _write_json(REPORTS / "control.json", summary)
    (REPORTS / "control.md").write_text(render_compare(summary), encoding="utf-8")
    print(render_compare(summary))


def _votes_path() -> Path:
    return Path(resolve_output_path(OUT)) / "anchors" / f"pool_votes_seed{POOL_VOTE_SEED}"


def stage_votes(args) -> None:
    require_output_root()
    pairs = pool_pairs()
    parts_dir = _votes_path()
    parts_dir.mkdir(parents=True, exist_ok=True)
    decoding = {**TA.V1_GENERATION, "seed": POOL_VOTE_SEED}
    backend, seconds = _backend(decoding), {}
    parts = np.array_split(np.arange(len(pairs)), POOL_VOTE_PARTS)
    try:
        for i, index in enumerate(parts):
            chunk, path = pairs.iloc[index], parts_dir / f"part{i}.parquet"
            saved = set(pd.read_parquet(path, columns=["pair_id"])["pair_id"]) if path.exists() \
                else set()
            if saved == set(chunk["pair_id"]):
                print(f"resuming: {path} is complete")
                continue
            start = time.time()
            frame = draw_frame(chunk, CONTROL, backend.generate_samples(
                candidate_prompts(chunk, CONTROL), K))
            frame.to_parquet(path, index=False)
            seconds[f"part{i}"] = round(time.time() - start, 1)
            print(f"wrote {path}: {len(frame)} draws in {seconds[f'part{i}']} s")
    finally:
        backend.close()
    draws = pd.concat([pd.read_parquet(parts_dir / f"part{i}.parquet")
                       for i in range(POOL_VOTE_PARTS)], ignore_index=True)
    draws.to_parquet(parts_dir.with_suffix(".parquet"), index=False)
    _write_json(parts_dir.with_suffix(".meta.json"), {
        "stage": "votes", "prompt": CONTROL, "model": TA.TEACHER, "pairs": len(pairs), "k": K,
        "decoding": decoding, "serving": TA.SERVING, "parts": POOL_VOTE_PARTS,
        "prompts_sha256": {CONTROL: prompts_sha256(candidate_prompts(pairs, CONTROL))},
        "generation_seconds_this_job": seconds, **provenance()})
    print(f"wrote {parts_dir.with_suffix('.parquet')}: {len(draws)} draws")


def stage_enrichment(args) -> None:
    labels = load_labels()
    pairs = pool_pairs()
    draws = pd.read_parquet(_votes_path().with_suffix(".parquet"))
    anchors = new_anchors(draws, pairs)
    anchors["unstable"] = unstable(anchors)
    by_pair = anchors.set_index("pair_id")
    pool_share = float(anchors["unstable"].mean())
    sets = {}
    for name, path in TRAINING_SETS.items():
        rows = pd.read_parquet(resolve_output_path(path), columns=["pair_id", "reference_decision"])
        flips = flip_readout(rows, anchors)
        sets[name] = {"path": path, **enrichment(rows, by_pair["unstable"], pool_share),
                      **{k: flips[k] for k in flips if k not in ("rows",)},
                      "by_side_not_preregistered": side_enrichment(rows, anchors)}
    control = pd.read_parquet(Path(resolve_output_path(OUT)) / "anchors" / "control_draws.parquet")
    reviewed = reviewed_pairs(labels)
    seed42 = new_anchors(control, reviewed).set_index("pair_id")["new_decision"]
    stability = []
    for name, mask in comparison_sets(labels).items():
        ids = labels.loc[mask, "pair_id"].tolist()
        before, after = seed42.loc[ids], by_pair.loc[ids, "new_decision"]
        stability.append({"set": name, "agreement": float((before == after).mean()),
                          **paired_flips(before, after)})
    platform = platform_check(draws, pairs)
    changed = float(anchors["changed"].mean())
    rebuild = {name: sets[name]["a_holds"] for name in REBUILD_CANDIDATES}
    full = anchors[anchors["decided"] == K]
    summary = {
        "pairs": len(anchors), "platform": platform,
        "pool": {"unstable_share": pool_share, "changed_share": changed,
                 "v1_by_new": {f"v1 {a}": {f"new {b}": int(n) for b, n in g.value_counts().items()}
                               for a, g in anchors.groupby("v1_decision")["new_decision"]},
                 "unstable_by_v1": {side: float(g.mean()) for side, g in
                                    anchors.groupby("v1_decision")["unstable"]},
                 "sources": {k: int(v) for k, v in anchors["source"].value_counts().items()},
                 "hire_votes_when_all_decided": {f"{h} of {K}": int(n) for h, n in
                                                 full["hire_votes"].value_counts().sort_index()
                                                 .items()}},
        "training_sets": sets, "stability_against_control": stability,
        "single_draws_not_preregistered": single_draws(draws, pairs),
        "output_format_not_preregistered": output_format(
            draws, pd.read_parquet(resolve_output_path(REFERENCE_V1), columns=["pair_id",
                                                                             "raw_output"]),
            pd.read_parquet(resolve_output_path(INVARIANT_V1), columns=["raw_output"]), pairs),
        "decisions": {
            "a_holds_for_rebuild_candidates": rebuild,
            "training_proposal": bool(platform["pass"] and any(rebuild.values())),
            "sft_retrain_worth_proposing": changed >= SFT_RETRAIN_MIN_CHANGED},
        "thresholds": {"unstable_max_majority": UNSTABLE_MAX_MAJORITY,
                       "platform_max_gap": PLATFORM_MAX_GAP, "enrichment_min": ENRICHMENT_MIN,
                       "sft_retrain_min_changed": SFT_RETRAIN_MIN_CHANGED},
        "draws_meta": json.loads(_votes_path().with_suffix(".meta.json").read_text()),
        **provenance()}
    anchors.to_parquet(_votes_path().parent / f"pool_votes_seed{POOL_VOTE_SEED}_anchors.parquet",
                       index=False)
    _write_json(REPORTS / "pool_votes.json", summary)
    (REPORTS / "pool_votes.md").write_text(render_votes(summary), encoding="utf-8")
    print(render_votes(summary))


# ---- reports --------------------------------------------------------------------------------

def _criterion_cells(r: dict) -> list:
    return [f"{r['c1']['hired']} of {r['c1']['pairs']} ({r['c1']['share']:.3f})",
            f"{r['c2']['rejected']} of {r['c2']['pairs']} ({r['c2']['share']:.3f})"]


def render_select(s: dict) -> str:
    rows = [{"candidate": name, "C1": _criterion_cells(r)[0], "C2": _criterion_cells(r)[1],
             "passes": r["passes"], "C1 + C2 - 1": r["score"], "hire share, 391": r["hire_share"],
             "undecided draws": r["undecided_draw_share"], "sources": r["sources"]}
            for name, r in s["candidates"].items()]
    votes = [{"candidate": name, "pairs": pairs, "hire votes": split}
             for name, r in s["candidates"].items() for pairs, split in r["hire_votes"].items()]
    graded = {name: r["c2_on_reviewer_fit_1_2_only"] for name, r in s["candidates"].items()}
    meta = s["draws_meta"]
    return "\n".join([
        "# SFT-v3 stage 0a: the anchor prompt, chosen on the 391 reviewed pool pairs", "",
        "Generated by `scripts/sft_v3_anchors.py --stage score`; regenerate it, do not edit it. "
        "Criteria and rule: `docs/SFT_V3_ANCHORS.md` (sha256 "
        f"{s['prereg_sha256'][:12]}). C1: clear-hire targets hired, at least {C1_MIN}. C2: "
        f"clear-reject targets and both-reject controls rejected, at least {C2_MIN}. Draws: "
        f"{meta['k']} per pair and prompt at {meta['decoding']}, written {meta['written']} by "
        f"code {str(meta['code']['commit'])[:7]} (script sha256 "
        f"{meta['code']['script_sha256'][:12]}).", "",
        *_table(rows, list(rows[0])), "",
        f"Chosen: **{s['chosen']}**. The v1 anchors hire {s['v1_hire_share']:.3f} of these "
        "pairs, and none of the target pairs, by selection.", "",
        f"Hired by both candidates: {s['hired_by_both']}. Not a criterion, C2 on the pairs the "
        f"reviewer graded fit 1 to 2 only (rejected of pairs): {graded}.", "",
        *_table(votes, ["candidate", "pairs", "hire votes"]), "",
        "New anchor's hire share by review set and fit, per candidate:", "",
        *[line for name, r in s["candidates"].items() for line in
          [f"`{name}`", "", *_table(r["by_set_and_fit"],
                                    ["set", "fit_bucket", "pairs", "new_anchor_hire_share"]), ""]],
    ])


def render_gate(s: dict) -> str:
    c1, c2, c3 = s["c1"], s["c2"], s["c3"]
    lines = [
        "# SFT-v3 stage 1: the gate", "",
        "Generated by `scripts/sft_v3_anchors.py --stage gate`; regenerate it, do not edit it. "
        f"Prompt `{s['prompt']}`, {K} draws per pair on {s['pairs']} pool pairs. Criteria: "
        f"`docs/SFT_V3_ANCHORS.md` (sha256 {s['prereg_sha256'][:12]}).", "",
        "| criterion | result | threshold | pass |", "|---|---|---|---|",
        f"| C1 clear-hire targets hired | {c1['hired']} of {c1['pairs']} ({c1['share']:.3f}) | "
        f"{C1_MIN} | {c1['pass']} |",
        f"| C2 clear-reject targets and controls rejected | {c2['rejected']} of {c2['pairs']} "
        f"({c2['share']:.3f}) | {C2_MIN} | {c2['pass']} |",
        f"| C3 pool hire share | {c3['v1_hire_share']:.4f} to {c3['new_hire_share']:.4f} "
        f"(+{c3['gain']:.4f}) | +{C3_GAIN} | {c3['pass']} |", "",
        f"Gate passes: **{s['passes']}**.", "",
        f"v1 against new anchor, pairs: {s['v1_by_new']}. Changed: {s['changed']}. "
        f"Sources: {s['sources']}.", "",
        f"Hire votes among {K} draws: {s['hire_votes_when_all_decided']}. Unanimous: "
        f"{_fmt(s['unanimous_share'])}.", "",
        "Reviewed pairs, new anchor's hire share by review set and fit:", "",
        *_table(s["reviewed_by_set_and_fit"], ["set", "fit_bucket", "pairs",
                                                "new_anchor_hire_share"]), "",
    ]
    if "invariant" in s:
        inv = s["invariant"]
        lines += [f"Regenerated invariant targets on changed pairs: {inv['kept']} of "
                  f"{inv['variants']} kept. Filter: {inv['filter_report']}.", "",
                  *_table(inv["by_cell"], ["protected_group", "condition", "reference_decision",
                                           "variants", "kept", "yield"]), ""]
    sets = [{"set": name, **{k: v for k, v in r.items() if k != "path"}}
            for name, r in s["training_sets"].items()]
    if sets:
        lines += ["Earlier training sets: rows whose target the new anchor flips, by old target.",
                  "", *_table(sets, ["set", "rows", "pairs", "flipped", "old_hire",
                                     "old_hire_flipped", "old_reject", "old_reject_flipped"]), ""]
    return "\n".join(lines)


def render_vote_targets(s: dict) -> str:
    a, t = s["anchors"], s["targets"]
    rows = [{"targets": name, **{k: r[k] for k in ("variants", "kept", "yield", "fenced_raw",
                                                   "within_pair_identical_rationale",
                                                   "copies_reference_feedback")},
             "dropped_by_reason": r["dropped_by_reason"]}
            for name, r in (("vote anchor, regenerated", t["vote"]),
                            ("v1 anchor, same variants", t["v1_same_variants"]))]
    lines = [
        "# SFT-v3: targets for the vote anchors", "",
        "Generated by `scripts/sft_v3_anchors.py --stage vote_readout`; regenerate it, do not edit "
        f"it. Pre-registration: the third addendum of `docs/SFT_V3_ANCHORS.md` (sha256 "
        f"{s['prereg_sha256'][:12]}).", "",
        f"Anchors: `{a['path']}`, recomputed from the vote's draws and equal to the saved file: "
        f"**{a['recomputed_equals_saved']}**. v1 against vote anchor, pairs: {a['v1_by_new']}. "
        f"Changed: {a['changed']}. Sources: {a['sources']}. Pool hire share "
        f"{a['c3']['v1_hire_share']:.4f} to {a['c3']['new_hire_share']:.4f}.", "",
        f"Regenerated targets: yield floor {t['min_yield']}, met: **{t['yield_floor_met']}**. "
        f"Distinct per-prompt seeds: {t['distinct_seeds']}. Generation: {t['generation']}.", "",
        *_table(rows, ["targets", "variants", "kept", "yield", "fenced_raw",
                       "within_pair_identical_rationale", "copies_reference_feedback",
                       "dropped_by_reason"]), "",
        *_table(t["by_cell"], ["protected_group", "condition", "reference_decision", "variants",
                               "kept", "yield"]), "",
        "Reviewed pairs, the vote anchor's hire share by review set and fit (descriptive; C1 and "
        f"C2 chose a prompt in stage 0a and decide nothing here): C1 {a['c1']['hired']} of "
        f"{a['c1']['pairs']}, C2 {a['c2']['rejected']} of {a['c2']['pairs']}.", "",
        *_table(a["reviewed_by_set_and_fit"], ["set", "fit_bucket", "pairs",
                                                "new_anchor_hire_share"]), "",
    ]
    sets = [{"set": name, **{k: v for k, v in r.items() if k != "path"}}
            for name, r in s["training_sets"].items()]
    if sets:
        lines += ["Earlier training sets: rows whose target the vote anchor flips, by old target.",
                  "", *_table(sets, ["set", "rows", "pairs", "flipped", "old_hire",
                                     "old_hire_flipped", "old_reject", "old_reject_flipped"]), ""]
    return "\n".join(lines)


def render_vote_sft(s: dict) -> str:
    rows = [{"variant": name, "split": split, **{k: r[split][k] for k in (
        "rows", "v1_rows", "english_rows", "english_hire_share", "v1_english_hire_share",
        "identical_to_v1", "variant_in_v1_completion_changed", "variant_not_in_v1",
        "candidates_shared_with_v1", "candidates")}}
            for name, r in s["variants"].items() for split in ("train", "validation")]
    lines = [
        "# SFT-v3: the data sets on the vote anchors", "",
        "Generated by `scripts/sft_v3_anchors.py --stage vote_sft`; regenerate it, do not edit it. "
        f"Pre-registration: the fourth addendum of `docs/SFT_V3_ANCHORS.md` (sha256 "
        f"{s['prereg_sha256'][:12]}).", "",
        f"Control, v1's split rebuilt from raw equals the published files: "
        f"**{s['control_equals_published']}**. v1's balance: {s['balance_v1']}.", "",
        f"Vote targets: {s['targets']} regenerated, {s['replaced_in_pool']} replaced in v1's "
        f"pooled invariant rows. Swapped rows in v1's split: {s['swapped_rows']}. Re-run "
        f"balance: {s['balance_rebalanced']}.", "",
        *_table(rows, ["variant", "split", "rows", "v1_rows", "english_rows",
                       "english_hire_share", "v1_english_hire_share", "identical_to_v1",
                       "variant_in_v1_completion_changed", "variant_not_in_v1",
                       "candidates_shared_with_v1", "candidates"]), "",
    ]
    for name, r in s["variants"].items():
        lines += [f"`{name}` ({r['path']}), rows by language and decision: "
                  f"train {r['train']['by_lang_and_decision']}, validation "
                  f"{r['validation']['by_lang_and_decision']}.", ""]
    return "\n".join(lines)


def render_compare(s: dict) -> str:
    arms = [{"arm": name, "C1 clear-hire targets hired": _criterion_cells(r)[0],
             "C2 clear-reject targets and controls rejected": _criterion_cells(r)[1],
             "ties kept at v1": r["sources"].get("tie", 0), "hire share, 391": r["hire_share"]}
            for name, r in s["arms"].items()]
    flips = ["pairs", "before_hired", "after_hired", "reject_to_hire", "hire_to_reject",
             "mcnemar_p"]
    votes = [{"arm": name, "pairs": pairs, "hire votes": split}
             for name, r in s["arms"].items() for pairs, split in r["hire_votes"].items()]
    meta = s["draws_meta"]["control"]
    return "\n".join([
        "# SFT-v3: the redraw control against the two candidate prompts, on the 391 reviewed "
        "pool pairs", "",
        "Generated by `scripts/sft_v3_anchors.py --stage compare`; regenerate it, do not edit "
        f"it. `{CONTROL}` is `synth.REFERENCE_PROMPT` unchanged, drawn {meta['k']} times per pair "
        f"at {meta['decoding']}; the two candidates' draws are stage 0a's. All three arms use "
        "the pre-registered vote rule (ties keep v1). Addendum and reading: "
        f"`docs/SFT_V3_ANCHORS.md` (sha256 {s['prereg_sha256'][:12]}).", "",
        *_table(arms, list(arms[0])), "",
        "Redraw effect: the v1 anchor (one draw) before, the control's vote after.", "",
        *_table(s["redraw_effect"], ["set", *flips]), "",
        "Prompt effect: the control before, each candidate after, on the same pairs.", "",
        *_table(s["prompt_effect"], ["candidate", "set", *flips]), "",
        *_table(votes, ["arm", "pairs", "hire votes"]), "",
    ])


def render_votes(s: dict) -> str:
    p, pool, d = s["platform"], s["pool"], s["decisions"]
    fmt = s["output_format_not_preregistered"]
    sets = [{"set": name, **{k: v for k, v in r.items() if k != "path"}}
            for name, r in s["training_sets"].items()]
    sides = [{"set": name, **{f"{side} {k}": v for side, b in r["by_side_not_preregistered"].items()
                              for k, v in b.items() if k != "pool_share"}}
             for name, r in s["training_sets"].items()]
    pool_sides = {side: round(b["pool_share"], 4) for side, b in
                  next(iter(s["training_sets"].values()))["by_side_not_preregistered"].items()}
    meta = s["draws_meta"]
    return "\n".join([
        "# SFT-v3: the pool-wide vote", "",
        "Generated by `scripts/sft_v3_anchors.py --stage enrichment`; regenerate it, do not edit "
        f"it. `{CONTROL}` (`synth.REFERENCE_PROMPT` unchanged), {meta['k']} draws per pair on "
        f"{s['pairs']} pool pairs at {meta['decoding']}, ties keep v1. Readouts and thresholds: "
        f"`docs/SFT_V3_ANCHORS.md` (sha256 {s['prereg_sha256'][:12]}). Unstable: overturned by "
        f"the vote, a majority of at most {UNSTABLE_MAX_MAJORITY:.3f} of decided draws, or none "
        "decided.", "",
        "| check | result | threshold | outcome |", "|---|---|---|---|",
        f"| B platform: per-draw hire share against v1 | {p['per_draw_hire_share']:.4f} against "
        f"{p['v1_hire_share']:.4f} (gap {p['gap']:+.4f}) | at most {PLATFORM_MAX_GAP} | "
        f"{'pass' if p['pass'] else 'fail'} |",
        f"| SFT retrain: pool pairs whose verdict changes | {pool['changed_share']:.4f} | at least "
        f"{SFT_RETRAIN_MIN_CHANGED} | {d['sft_retrain_worth_proposing']} |",
        f"| A for the rebuild candidates | {d['a_holds_for_rebuild_candidates']} | enrichment at "
        f"least {ENRICHMENT_MIN} | training proposal: {d['training_proposal']} |", "",
        f"Pool: unstable share {pool['unstable_share']:.4f} (by v1 verdict "
        f"{ {k: round(v, 4) for k, v in pool['unstable_by_v1'].items()} }). v1 against new "
        f"anchor: {pool['v1_by_new']}. Sources: {pool['sources']}. Hire votes among {K} draws: "
        f"{pool['hire_votes_when_all_decided']}.", "",
        "A: earlier training sets, rows on unstable anchors and rows whose target the vote flips.",
        "", *_table(sets, ["set", "rows", "pairs", "unstable_rows", "unstable_share", "enrichment",
                           "a_holds", "flipped", "old_hire", "old_hire_flipped", "old_reject",
                           "old_reject_flipped"]), "",
        f"D: the 391 reviewed pairs, the control (seed 42, job 5073323) before, this vote (seed "
        f"{POOL_VOTE_SEED}) after.", "",
        *_table(s["stability_against_control"], ["set", "agreement", "pairs", "before_hired",
                                                 "after_hired", "reject_to_hire",
                                                 "hire_to_reject", "mcnemar_p"]), "",
        "Not pre-registered: enrichment within each old target side, against the pool's pairs "
        f"on that side (pool unstable share by side: {pool_sides}).", "",
        *_table(sides, ["set", "reject rows", "reject unstable_share", "reject enrichment",
                        "hire rows", "hire unstable_share", "hire enrichment"]), "",
        "Not pre-registered, added after the platform check failed: single draws on all pool "
        "pairs. Without a platform shift both rows disagree about equally, and symmetrically.",
        "", *_table([{"comparison": k, **v} for k, v in
                     s["single_draws_not_preregistered"].items()],
                    ["comparison", "pairs", "disagreement", "before_hired", "after_hired",
                     "reject_to_hire", "hire_to_reject", "mcnemar_p"]), "",
        "Not pre-registered, added after the platform check failed: answers wrapped in a code "
        "fence. v1's raw generations are stored unmodified; both setups had the same nominal "
        "prompt and decoding.", "",
        *_table([{"outputs of": k, **v} for k, v in fmt.items()
                 if k != "daint_hire_share_by_format"],
                ["outputs of", "outputs", "fenced", "fenced_share"]), "",
        f"Daint hire share by format: {fmt['daint_hire_share_by_format']}.", "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("labels", "select", "score", "generate", "gate",
                                            "control", "compare", "votes", "enrichment",
                                            "vote_targets", "vote_readout", "vote_sft"),
                        required=True)
    parser.add_argument("--review", help="the blind review's graded_cases.csv (stage labels)")
    parser.add_argument("--reports", type=Path, default=REPORTS,
                        help="where the reports and the labels live (default %(default)s)")
    args = parser.parse_args()
    set_reports(args.reports)
    {"labels": stage_labels, "select": stage_select, "score": stage_score,
     "generate": stage_generate, "gate": stage_gate, "control": stage_control,
     "compare": stage_compare, "votes": stage_votes,
     "enrichment": stage_enrichment, "vote_targets": stage_vote_targets,
     "vote_readout": stage_vote_readout, "vote_sft": stage_vote_sft}[args.stage](args)


if __name__ == "__main__":
    main()
