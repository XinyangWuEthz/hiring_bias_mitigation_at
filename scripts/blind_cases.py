"""A blind reading set: raw benchmark answers with the model names, the reference and any
commentary taken away, and the key kept in a separate file.

Cases are attribute-free benchmark pairs, so nothing in the prompt names a protected attribute.
For each case the reader gets the job description, the CV and the answers of SFT, DPO r2em-100,
the GRPO pilot and the rejection-sampling SFT control, quoted exactly as generated
(outputs/raw/<run>.parquet, `raw_output`), in a shuffled order under the letters A-D.

Selection, seeded, from three strata so that disagreeing and agreeing cases are both present:
  SFT hires and the GRPO pilot rejects        (the direction every trained arm moved)
  SFT rejects and DPO r2em-100 hires          (the one arm that moved the other way)
  all four agree                              (controls; one hire, two reject)

The key (key.md, key.json) names the models behind the letters, the stratum, GPT-4o's
reference and, where the pair was graded, the Claude blind review's verdict and fit score.

    python scripts/blind_cases.py           # -> reports/blind_cases/{cases.md,key.md,key.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402

ARMS = {
    "SFT": "Qwen3.5-9B--en--sft--adapter--en_only_cscs",
    "DPO r2em-100": "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step100",
    "GRPO pilot": "Qwen3.5-9B--en--grpo--adapter--en_only_grpo_pilot",
    "RFT": "Qwen3.5-9B--en--sft--adapter--en_only_rft",
}
STRATA = (("SFT hires, GRPO pilot rejects", 5), ("SFT rejects, DPO r2em-100 hires", 2),
          ("all four agree: hire", 1), ("all four agree: reject", 2))
LETTERS = "ABCD"
SEED = 7


def attr_free_answers() -> pd.DataFrame:
    """One row per benchmark pair: job, CV, reference and each arm's raw answer and decision."""
    frame = None
    for name, run in ARMS.items():
        cols = ["pair_id", "condition", "job_position", "job_description", "cv",
                "reference_decision", "raw_output", "decision"]
        raw = pd.read_parquet(Path(resolve_output_path("outputs/raw")) / f"{run}.parquet",
                              columns=cols, use_threads=False)
        raw = raw[raw["condition"] == "attr_free"].set_index("pair_id")
        part = raw[["raw_output", "decision"]].rename(
            columns={"raw_output": f"out::{name}", "decision": f"dec::{name}"})
        if frame is None:
            frame = raw[["job_position", "job_description", "cv", "reference_decision"]].join(part)
        else:
            frame = frame.join(part, how="inner")
    for name in ARMS:
        frame[f"dec::{name}"] = frame[f"dec::{name}"].astype(str).str.lower()
    return frame


def stratum_of(row: pd.Series) -> str | None:
    d = {name: row[f"dec::{name}"] for name in ARMS}
    if len(set(d.values())) == 1:
        return f"all four agree: {d['SFT']}"
    if d["SFT"] == "hire" and d["GRPO pilot"] == "reject":
        return "SFT hires, GRPO pilot rejects"
    if d["SFT"] == "reject" and d["DPO r2em-100"] == "hire":
        return "SFT rejects, DPO r2em-100 hires"
    return None


def select(frame: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = frame.assign(stratum=frame.apply(stratum_of, axis=1))
    picked = []
    for stratum, n in STRATA:
        pool = frame[frame["stratum"] == stratum].sort_index()
        take = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        picked.append(pool.iloc[np.sort(take)])
    chosen = pd.concat(picked)
    return chosen.iloc[rng.permutation(len(chosen))]


def render(chosen: pd.DataFrame, grades: pd.DataFrame | None, seed: int = SEED) -> tuple[str, list]:
    rng = np.random.default_rng(seed + 1)
    lines = ["# Blind reading cases", "",
             "Each case is an **attribute-free** benchmark job/CV pair: "
             "the prompt contains no protected attribute. "
             "Four raw model answers follow, quoted verbatim in shuffled order and labeled A–D.",
             "Model names, reference answers, and commentary are withheld. "
             "The answer key is `key.md` in the same directory; "
             "read all cases before opening it.", ""]
    key = []
    for i, (pair_id, row) in enumerate(chosen.iterrows(), 1):
        order = list(ARMS)
        rng.shuffle(order)
        lines += [f"## Case {i:02d}: {row['job_position']}", "",
                  "### Job description", "", "```text",
                  str(row["job_description"]).strip(), "```", "",
                  "### CV", "", "```text", str(row["cv"]).strip(), "```", "",
                  "### Answers", ""]
        for letter, name in zip(LETTERS, order):
            lines += [f"**{letter}**", "", "```json", str(row[f"out::{name}"]).strip(), "```", ""]
        lines += ["Your decision: hire / reject. Does the rationale match the CV? "
                  "Assess each answer separately:", "", "---", ""]
        graded = None
        if grades is not None and pair_id in grades.index:
            g = grades.loc[pair_id]
            graded = {"decision": g["decision_claude"], "fit_score": int(g["fit_score_claude"])}
        key.append({"case": i, "pair_id": pair_id, "job_position": row["job_position"],
                    "stratum": row["stratum"],
                    "letters": {letter: name for letter, name in zip(LETTERS, order)},
                    "decisions": {name: row[f"dec::{name}"] for name in ARMS},
                    "reference_gpt4o": row["reference_decision"], "claude_blind_review": graded})
    return "\n".join(lines), key


def key_markdown(key: list) -> str:
    lines = ["# Answer key for the blind reading cases", "",
             "| Case | Job | Sampling stratum | A | B | C | D | GPT-4o reference "
             "| Claude blind review: decision / fit score 1–5 |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k in key:
        letters = [f"{k['letters'][x]} ({k['decisions'][k['letters'][x]]})" for x in LETTERS]
        graded = k["claude_blind_review"]
        claude = f"{graded['decision']} / {graded['fit_score']}" if graded else "Not reviewed"
        lines.append(f"| {k['case']:02d} | {k['job_position']} | {k['stratum']} | "
                     + " | ".join(letters) + f" | {k['reference_gpt4o']} | {claude} |")
    lines += ["", "See `key.json` for pair_id and sampling details; generated by "
              f"`scripts/blind_cases.py` with seed {SEED}."]
    return "\n".join(lines)


def main() -> None:
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    frame = attr_free_answers()
    grades_path = REPO_ROOT / "reports" / "claude_blind_review" / "claude_grades_153.csv"
    grades = pd.read_csv(grades_path).set_index("pair_id") if grades_path.is_file() else None
    chosen = select(frame)
    cases, key = render(chosen, grades)
    out = REPO_ROOT / "reports" / "blind_cases"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.md").write_text(cases)
    (out / "key.md").write_text(key_markdown(key))
    (out / "key.json").write_text(json.dumps(
        {"seed": SEED, "runs": ARMS, "strata": [list(s) for s in STRATA],
         "stratum_sizes": frame.apply(stratum_of, axis=1).value_counts().to_dict(),
         "cases": key}, indent=2, ensure_ascii=False))
    print(f"wrote {out}/cases.md ({len(key)} cases), key.md, key.json")


if __name__ == "__main__":
    main()
