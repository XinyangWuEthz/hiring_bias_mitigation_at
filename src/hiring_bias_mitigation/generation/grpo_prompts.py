"""The GRPO pilot's prompt set: attributed audit prompts, balanced by reference verdict.

GRPO needs no responses, only prompts and what a right answer is. Each row here is one v1 pool
row (a pair's CV and job with one attribute in one condition) as the audit prompt the SFT
student was sampled on, with the pair's attribute-free reference verdict as the target. The
reward (mitigation/grpo.py) reads `reference_decision` next to each completion.

Three choices, each against a failure seen in the DPO rounds:

  Balance. The pool's reference is 86% reject. DPO on pool-shaped data taught "reject" and
  moved the hire rate from 28% to 19% (docs/ITERATIVE_DPO.md). Here both verdicts get exactly
  half the prompts, so a policy that drifts to one verdict loses reward on half of them.

  Tipped rows. A GRPO group whose completions all earn the same reward has zero advantage and
  teaches nothing. Rows where the student is right without the attribute and wrong with it in
  at least one sample (`tipped` in the on-policy selection) are where its samples disagree,
  so `tipped_share` of each side comes from them -- far above their natural rate. Hire rows
  read the k=32 hire top-up's selection, reject rows the k=8 pool's.

  Hold-out. Every candidate in an existing validation split stays out of training, so those
  splits stay comparable across arms, and the decision monitor reads its rows from them. Both
  frames pass `assert_no_leakage` against the benchmark, as every data builder here does.

  Signal balance (`signal_balanced_train`, the default since grpo-02). Equal counts are not
  an equal pull. With group-normalised advantages, a prompt on which the student errs with
  probability p pushes its decision by sqrt(p(1-p)) per completion in expectation. The
  student's own samples at temperature 1 (8 per row, 32 for the hire top-up) give p for every
  row. In grpo-01 (half of each side tipped) the reject side carried 3x the hire side's error
  mass and 1.5x its pull, and the 20-step smoke (job 5065891) moved accordingly: sampled hires
  on reject-reference prompts fell from 29% to ~5%, on hire-reference prompts not at all. So
  each side's tipped share is now chosen to give both sides the same summed pull: the weaker
  side takes all the tipped rows its half allows, the stronger one as many as match it. This
  is r2em's error-mass balance (docs/ITERATIVE_DPO.md) in the quantity GRPO responds to.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

#: One pool row: a pair's CV and job with one attribute in one condition.
ROW_KEY = ["pair_id", "protected_attr", "condition"]
#: Carried into the parquet next to `prompt`; the reward functions receive every one of them.
COLUMNS = ["prompt", "reference_decision", "pair_id", "candidate_id", "job_id", "lang",
           "protected_group", "protected_attr", "condition", "tipped", "sft_hire_rate",
           "sft_samples", "pull"]
SIDES = ("hire", "reject")


def attach_prompts(rows: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    """Each pool row with the attributed audit prompt it was sampled on.

    prompts.parquet keys an attributed prompt by its row's position in rows.parquet, so a row
    count that differs means the two files are not from the same run.
    """
    attributed = prompts[prompts["kind"] == "attributed"]
    if len(attributed) != len(rows):
        raise ValueError(f"{len(attributed)} attributed prompts for {len(rows)} rows: "
                         "rows.parquet and prompts.parquet are from different runs")
    text = dict(zip(attributed["key"].astype(int), attributed["prompt"]))
    return rows.reset_index(drop=True).assign(prompt=[text[i] for i in range(len(rows))])


def tipped_keys(*selections: pd.DataFrame) -> set[tuple]:
    """Row keys the on-policy selection marked attribute-tipped, over any number of runs."""
    keys: set[tuple] = set()
    for frame in selections:
        tipped = frame[frame["tipped"].astype(bool)]
        keys.update(tipped[ROW_KEY].itertuples(index=False, name=None))
    return keys


def mark_tipped(rows: pd.DataFrame, keys: set[tuple]) -> pd.DataFrame:
    flags = [k in keys for k in rows[ROW_KEY].itertuples(index=False, name=None)]
    return rows.assign(tipped=flags)


def heldout_candidates(frames: list[pd.DataFrame]) -> set[str]:
    return {str(c) for f in frames for c in f["candidate_id"]}


def split_by_candidate(rows: pd.DataFrame, heldout: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    held = rows["candidate_id"].astype(str).isin(heldout)
    return rows[~held].reset_index(drop=True), rows[held].reset_index(drop=True)


def _draw(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[:0]
    index = rng.choice(len(frame), size=min(n, len(frame)), replace=False)
    return frame.iloc[np.sort(index)]


def balanced_train(rows: pd.DataFrame, n: int, tipped_share: float,
                   seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """`n` prompts, exactly half per reference verdict; within a side, `tipped_share` of the
    half from tipped rows and the rest from the others, each drawn without replacement. A side
    short of tipped rows fills up with untipped ones, so the halves stay exact."""
    if n % 2:
        raise ValueError(f"n={n} cannot split exactly in half")
    rng = np.random.default_rng(seed)
    half, parts, plan = n // 2, [], {}
    for side in SIDES:
        pool = rows[rows["reference_decision"] == side]
        tipped, rest = pool[pool["tipped"]], pool[~pool["tipped"]]
        want = round(tipped_share * half)
        a = _draw(tipped, want, rng)
        b = _draw(rest, half - len(a), rng)
        if len(a) + len(b) < half:
            raise ValueError(f"only {len(pool)} {side}-reference rows for a half of {half}")
        parts += [a, b]
        plan[side] = {"available": len(pool), "available_tipped": len(tipped),
                      "drawn_tipped": len(a), "drawn_untipped": len(b)}
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, plan


def balanced_validation(rows: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """`n` held-out rows, half per reference verdict, one seeded draw."""
    rng = np.random.default_rng(seed)
    parts = [_draw(rows[rows["reference_decision"] == side], n // 2, rng) for side in SIDES]
    return pd.concat(parts).sort_index().reset_index(drop=True)


#: Bins of the SFT parent's sampled hire rate for the margin monitor's rows (the last bin
#: includes 1.0). The benchmark's TPR moved where the parent hires 50-99% of the time: GRPO-v3-all
#: and DAPO-v3-all lost 59-71 pp of those hires and 2-5 pp where it hires over 99%. Of the 128
#: hire rows of grpo-v3-all's monitor, 74% sat in that last bin (docs/DAPO_V3.md, monitor).
MONITOR_RATE_BINS = (0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0)


def monitor_rows(rows: pd.DataFrame, per_bin: int, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Held-out rows for the margin monitor (mitigation/grpo.py, MarginMonitorCallback): up to
    `per_bin` from each MONITOR_RATE_BINS bin of the parent's sampled hire rate, whatever their
    reference verdict. The monitor reads how decisions move at each level of the parent's
    confidence and weights the levels as the benchmark has them, so it needs every level
    covered, not the pool's mix."""
    rng = np.random.default_rng(seed)
    edges = np.asarray(MONITOR_RATE_BINS)
    index = np.clip(np.searchsorted(edges, rows["sft_hire_rate"].to_numpy(), side="right") - 1,
                    0, len(edges) - 2)
    parts, plan = [], {}
    for b in range(len(edges) - 1):
        pool = rows[index == b]
        parts.append(_draw(pool, per_bin, rng))
        close = "]" if b == len(edges) - 2 else ")"
        plan[f"[{edges[b]:g}, {edges[b + 1]:g}{close}"] = {"available": len(pool),
                                                           "drawn": len(parts[-1])}
    return pd.concat(parts).sort_index().reset_index(drop=True), plan


def sample_hire_counts(samples: pd.DataFrame, lang: str = "en") -> dict[str, tuple[int, int]]:
    """prompt -> (hire decisions, samples) over the student's stored samples
    (`raw/samples/chunk-*.parquet`: one row per prompt and sample)."""
    from ..eval.parsing import HIRE, parse_output

    hires = [parse_output(o, lang).decision == HIRE for o in samples["raw_output"]]
    grouped = pd.DataFrame({"prompt": samples["prompt"], "hire": hires}).groupby("prompt")["hire"]
    return {p: (int(h), int(n)) for p, h, n in zip(grouped.sum().index, grouped.sum(),
                                                    grouped.count())}


def attach_sft_rates(rows: pd.DataFrame, *counts: dict) -> pd.DataFrame:
    """The student's sampled hire rate per row, from the first count table that has its prompt
    (pass the k=32 top-up before the k=8 pool), and the row's expected pull
    sqrt(p(1-p)), p the rate of the decision that disagrees with the reference."""
    hire, n = [], []
    for prompt in rows["prompt"]:
        found = next((c[prompt] for c in counts if prompt in c), None)
        if found is None:
            raise KeyError("a pool row has no stored student samples: the samples are from "
                           "another run than rows.parquet")
        hire.append(found[0] / found[1])
        n.append(found[1])
    rate = np.asarray(hire)
    wrong = np.where(rows["reference_decision"].to_numpy() == "hire", 1 - rate, rate)
    return rows.assign(sft_hire_rate=rate, sft_samples=n, pull=np.sqrt(wrong * (1 - wrong)))


#: `fill: pull` takes, within this share of the target, the k with the most tipped rows.
PULL_TOLERANCE = 0.01


def informative_half(rows: pd.DataFrame, column: str = "pull") -> int:
    """The largest half both verdicts can fill with rows of nonzero `column`: `--n-train auto`."""
    return int(min((rows[rows["reference_decision"] == side][column] > 0).sum()
                   for side in SIDES))


def signal_balanced_train(rows: pd.DataFrame, n: int, seed: int = 42, column: str = "pull",
                          fill: str = "random") -> tuple[pd.DataFrame, dict]:
    """`n` prompts, exactly half per reference verdict, the two halves carrying the same summed
    `column` (by default GRPO's `pull`) as nearly as the rows allow.

    Within a side, tipped and untipped rows are each shuffled once (seeded); the half is the
    first k tipped and the first half-k untipped. Every side's largest pull (k as large as the
    half and its tipped rows allow) is computed; the smaller of those is the target, and each
    side takes the k whose pull comes closest to it.

    `fill="random"` (grpo-02, grpo-v3) keeps the shuffled order, so the untipped rows that fill
    a half are a random draw, most of them without a sampled error. `fill="pull"` orders tipped
    and untipped rows by `column`, highest first (ties in the shuffled order), so a half fills
    with the rows that carry signal before any that carry none; among the k within
    PULL_TOLERANCE of the target it takes the largest, keeping attribute-tipped rows first.
    """
    if n % 2:
        raise ValueError(f"n={n} cannot split exactly in half")
    if fill not in ("random", "pull"):
        raise ValueError(f"fill must be random or pull, not {fill!r}")
    rng = np.random.default_rng(seed)
    half, options = n // 2, {}
    for side in SIDES:
        pool = rows[rows["reference_decision"] == side]
        tipped = pool[pool["tipped"]].iloc[rng.permutation(int(pool["tipped"].sum()))]
        rest = pool[~pool["tipped"]].iloc[rng.permutation(int((~pool["tipped"]).sum()))]
        if fill == "pull":
            tipped = tipped.sort_values(column, ascending=False, kind="stable")
            rest = rest.sort_values(column, ascending=False, kind="stable")
        low, high = max(0, half - len(rest)), min(half, len(tipped))
        if low > high:
            raise ValueError(f"only {len(pool)} {side}-reference rows for a half of {half}")
        a = np.concatenate([[0.0], np.cumsum(tipped[column].to_numpy())])
        b = np.concatenate([[0.0], np.cumsum(rest[column].to_numpy())])
        ks = np.arange(low, high + 1)
        options[side] = (tipped, rest, ks, a[ks] + b[half - ks])
    target = min(pulls.max() for *_, pulls in options.values())
    parts, plan = [], {f"target_{column}": round(float(target), 2)}
    for side, (tipped, rest, ks, pulls) in options.items():
        distance = np.abs(pulls - target)
        i = int(np.argmin(distance))
        if fill == "pull":
            i = int(np.flatnonzero(distance <= max(distance[i], PULL_TOLERANCE * target))[-1])
        k = int(ks[i])
        chosen = pd.concat([tipped.iloc[:k], rest.iloc[:half - k]])
        parts.append(chosen)
        plan[side] = {"available": len(tipped) + len(rest), "available_tipped": len(tipped),
                      "drawn_tipped": k, "drawn_untipped": half - k,
                      "available_informative": int((tipped[column] > 0).sum()
                                                   + (rest[column] > 0).sum()),
                      "drawn_informative": int((chosen[column] > 0).sum()),
                      column: round(float(pulls[i]), 2),
                      "error_mass": round(float(np.where(side == "hire",
                                                         1 - chosen["sft_hire_rate"],
                                                         chosen["sft_hire_rate"]).sum()), 2)}
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, plan


def side_pull(frame: pd.DataFrame) -> dict:
    """Summed pull and error mass per reference verdict, for the report."""
    out = {}
    for side, g in frame.groupby("reference_decision"):
        wrong = 1 - g["sft_hire_rate"] if side == "hire" else g["sft_hire_rate"]
        out[str(side)] = {"pull": round(float(g["pull"].sum()), 2),
                          "error_mass": round(float(wrong.sum()), 2),
                          "informative_share": round(float(((wrong > 0) & (wrong < 1)).mean()), 4)}
    return out


def describe(frame: pd.DataFrame) -> dict:
    """Counts a reader checks first: per verdict, tipped, group and condition."""
    def counts(column: str) -> dict:
        return {str(k): int(v) for k, v in frame[column].value_counts().sort_index().items()}

    by_side = frame.groupby("reference_decision")
    return {
        "rows": len(frame),
        "pairs": int(frame["pair_id"].nunique()),
        "candidates": int(frame["candidate_id"].nunique()),
        "reference_decision": counts("reference_decision"),
        "tipped_share_by_reference": {str(k): round(float(g["tipped"].mean()), 4)
                                      for k, g in by_side},
        "protected_group": counts("protected_group"),
        "condition": counts("condition"),
        "prompt_chars": {q: int(frame["prompt"].str.len().quantile(p))
                         for q, p in (("p50", 0.5), ("p99", 0.99), ("max", 1.0))},
    }


def samples_markdown(frame: pd.DataFrame, n: int = 12, seed: int = 42,
                     head_chars: int = 600) -> str:
    """A few rows per verdict as they reach the trainer, for reading before training."""
    lines = ["# GRPO prompt samples", ""]
    for side in SIDES:
        side_rows = frame[frame["reference_decision"] == side]
        pick = side_rows.sample(n=min(n // 2, len(side_rows)), random_state=seed)
        for row in pick.to_dict("records"):
            lines += [f"## {side} | {row['protected_group']} | {row['protected_attr']} | "
                      f"{row['condition']} | tipped={row['tipped']}",
                      f"pair `{row['pair_id']}`", "", "```text",
                      row["prompt"][:head_chars] + (" ..." if len(row["prompt"]) > head_chars
                                                     else ""),
                      "...", row["prompt"][-head_chars:], "```", ""]
    return "\n".join(lines)


def read(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Single-threaded parquet read: Daint login nodes refuse pyarrow's worker threads."""
    return pd.read_parquet(path, columns=columns, use_threads=False)
