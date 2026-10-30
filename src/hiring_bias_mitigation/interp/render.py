"""The prompts the analysis runs on: the audit's own grid, and a disjoint discovery grid.

Two datasets, for the reason any held-out evaluation needs two:

`grid`
    The benchmark evaluation grid exactly as the audit built it (`eval/runner.build_frame`),
    single protected groups only. Circuits are *confirmed* here, and SFT/DPO are compared here,
    because these are the rows the behavioural results were measured on.

`pool`
    The same counterfactual grid rendered on job-CV pairs from the synthetic training pool,
    which is disjoint from the benchmark by construction (`assert_no_leakage`). Circuits are
    *discovered* here, on the base model only -- choosing components on the benchmark and then
    reporting their effect on the benchmark would fit the circuit to its own test set. The pool
    is SFT's training data, so it is never used to compare the fine-tuned models.

Only the pool's (job, CV) pairs are used, not its teacher outputs: the grid is re-rendered with
every attribute of every group, so every contrast exists for every pair. The teacher pass held
twelve variants per pair, which would leave most fixed contrasts unpaired.

Prompts are rendered by the audit's own `build_prompt` and chat-wrapped by the audit's own
`_chat_wrap`, then the decision prefix is appended -- byte-for-byte what the margin scripts feed
the model, so the analysis runs on the inputs the behavioural numbers came from.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ..data import benchmark as B
from ..data.injection import build_eval_set
from ..data.prompts import build_prompt
from ..eval import runner as R
from ..eval.backends import _chat_wrap
from ..eval.decision import PREFIX
from ..utils.config import resolve_output_path

SINGLE_GROUPS = ("military_status", "gender", "religion")
POOL_COLUMNS = ["pair_id", "candidate_id", "job_id", "cv", "job_description", "job_position",
                "lang", "reference_decision", "reference_feedback"]


def pool_path(lang: str) -> Path:
    return Path(resolve_output_path("artifacts/semisynthetic-v1/raw")) / f"{lang}_invariant.parquet"


def pool_pairs(lang: str, n_pairs: int, seed: int) -> pd.DataFrame:
    """`n_pairs` job-CV pairs from the training pool, one per candidate, benchmark-disjoint.

    One pair per candidate because the analysis bootstraps over candidates: two pairs of the
    same CV are not two independent observations of the attribute effect.
    """
    path = pool_path(lang)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/interp/01_fetch_data.py first")
    pool = pd.read_parquet(path, columns=POOL_COLUMNS).drop_duplicates("pair_id")
    holdout = B.holdout_ids()
    pool = pool[[not holdout.excludes(c, j) for c, j in zip(pool["candidate_id"], pool["job_id"])]]
    per_candidate = pool.sort_values("pair_id").groupby("candidate_id", sort=True).head(1)
    chosen = per_candidate.sample(n=min(n_pairs, len(per_candidate)), random_state=seed)
    chosen = chosen.sort_values("pair_id").reset_index(drop=True)
    B.assert_no_leakage(chosen, "interp discovery pool")
    return chosen


def build_frame(dataset: str, cfg: dict) -> pd.DataFrame:
    """The counterfactual grid for one dataset, with a `dataset` column."""
    lang = cfg["lang"]
    groups = list(cfg.get("protected_groups", SINGLE_GROUPS))
    conditions = tuple(cfg.get("conditions", ("explicit", "implicit", "attr_free")))
    if dataset == "grid":
        frame = R.build_frame({"lang": lang, "protected_groups": groups, "conditions": conditions,
                               "limit_pairs": cfg.get("limit_pairs")})
    elif dataset == "pool":
        pairs = pool_pairs(lang, int(cfg["pool_pairs"]), int(cfg.get("seed", 42)))
        frame = build_eval_set(pairs, groups, lang, conditions)
    else:
        raise ValueError(f"unknown dataset {dataset!r}; expected 'grid' or 'pool'")
    return frame.assign(dataset=dataset)


def chat_text(tokenizer, row: dict, chat_template_kwargs: dict | None = None) -> str:
    """The exact model input for one row: chat-wrapped baseline prompt + decision prefix."""
    return _chat_wrap(tokenizer, build_prompt(row, "baseline"), chat_template_kwargs) + PREFIX


def prompt_uid(text: str) -> str:
    """A content address for a model input, so every artifact joins on what was run."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def render(frame: pd.DataFrame, tokenizer, chat_template_kwargs: dict | None = None
           ) -> pd.DataFrame:
    """Adds `text` and `prompt_uid` columns."""
    texts = [chat_text(tokenizer, row, chat_template_kwargs) for row in frame.to_dict("records")]
    return frame.assign(text=texts, prompt_uid=[prompt_uid(t) for t in texts])
