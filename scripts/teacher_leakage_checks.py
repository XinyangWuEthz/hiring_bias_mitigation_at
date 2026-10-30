"""Does the teacher's own attribute bias reach the training data through the two stages that
show it the attribute? CPU only.

Why. The 122B teacher decides very differently once it sees an attribute: on the benchmark its
instability over 900 sets is 28.0% against SFT's 6.89%, and an implicit religious-holiday
sentence drops its hire rate to 7.5% (eval/results/Qwen3.5-122B-A10B-GPTQ-Int4--en--baseline--
cscs.json, reports/decision_diff/). Its verdicts never become training targets, which are pinned
to its attribute-free anchor, but two stages show it the attribute:

  invariant  v1's SFT targets (synth.INVARIANT_PROMPT): the teacher writes the rationale for the
             attributed profile with the verdict pinned. artifacts/semisynthetic-v1/raw/
             en_invariant.parquet, 3,000 pairs x 12 variants.
  judge      v2's judge (negatives_v2.JUDGE_PROMPT): told the attribute so it can spot an
             allusion, it also rates groundedness, decision support and plausibility.
             <round>/raw/{student,chosen}_judge.parquet.

Each check reads a cell (group x condition, split by the reference verdict) against the rest:

  invariant  filters.filter_invariant run per cell (what it dropped and why); on kept rows the
             within-pair deviation of length, hedges, intensity and proxy words, with a
             pair-clustered bootstrap; and a TF-IDF classifier that tries to tell each cell
             from the others with folds grouped by pair, against a null in which the cell
             labels are shuffled within each pair. Every pair holds every cell, so above-null
             accuracy can only come from how the variants are worded.
  judge      negatives_v2.filter_by_judge run per cell and side: the pass rate and the first
             failing reason. The judge sees the attribute-free CV and the attribute's name, not
             the implicit sentence, so a condition effect can only come from the rationale text.

HEDGES and INTENSITY are this script's own word lists; PROXY_LEXICON is negatives_v2's.

    python scripts/teacher_leakage_checks.py --check invariant
    python scripts/teacher_leakage_checks.py --check judge
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiring_bias_mitigation.generation import negatives_v2 as N  # noqa: E402
from hiring_bias_mitigation.generation.filters import filter_invariant  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

CELL = ["protected_group", "condition"]
SPLIT = [*CELL, "reference_decision"]
HEDGES = ("however", "although", "but", "though", "despite", "while", "yet", "may", "might",
          "could", "somewhat", "partially", "potential", "potentially")
INTENSITY = ("strong", "strongly", "excellent", "extensive", "proven", "perfectly", "exceeds",
             "solid", "robust", "outstanding", "highly", "deep", "impressive")
#: Addressing the candidate ("your background", "we require") instead of describing them.
ADDRESS = ("you", "your", "yours", "we", "our", "us")
INVARIANT = "artifacts/semisynthetic-v1/raw/en_invariant.parquet"
ROUNDS = ("artifacts/semisynthetic-v2/onpolicy-full", "artifacts/semisynthetic-v2/onpolicy-hire32")


def count_terms(text: str, terms: tuple[str, ...]) -> int:
    """Whole-word occurrences of any term, case-insensitive."""
    lowered = str(text).lower()
    return sum(len(re.findall(r"\b" + re.escape(t) + r"\b", lowered)) for t in terms)


def features(feedback: pd.Series, groups: pd.Series | None = None) -> pd.DataFrame:
    out = pd.DataFrame({
        "words": feedback.map(lambda t: len(str(t).split())),
        "hedges": feedback.map(lambda t: count_terms(t, HEDGES)),
        "intensity": feedback.map(lambda t: count_terms(t, INTENSITY)),
        "proxies": feedback.map(lambda t: len(N.proxy_words(str(t)))),
        "addresses": feedback.map(lambda t: int(count_terms(t, ADDRESS) > 0)),
    }, index=feedback.index)
    if groups is not None:
        # Words naming the row's own group (negatives_v2.GROUP_LEXICON), which the invariant
        # filter does not check: it drops only the attribute value itself.
        out["group_words"] = [int(N.mentions_group(str(t), g)) for t, g in zip(feedback, groups)]
    return out


# ---- invariant targets ----------------------------------------------------------------------

def invariant_filter_by_cell(raw: pd.DataFrame, lang: str = "en") -> tuple[pd.DataFrame,
                                                                            pd.DataFrame]:
    """filter_invariant per cell: the kept rows, and a table of yields and drop reasons."""
    kept_parts, rows = [], []
    for key, part in raw.groupby(SPLIT, sort=True):
        kept, report = filter_invariant(part["raw_output"].tolist(),
                                        part.drop(columns=["raw_output"]), lang)
        kept_parts.append(kept)
        rows.append({**dict(zip(SPLIT, key)), "n": report["n_input"], "kept": report["n_kept"],
                     "yield": report["yield"],
                     **{f"drop_{k}": v for k, v in report["dropped_by_reason"].items()}})
    table = pd.DataFrame(rows)
    drops = [c for c in table.columns if c.startswith("drop_")]
    table[drops] = table[drops].fillna(0).astype(int)
    return pd.concat(kept_parts, ignore_index=True), table


def within_pair_deviation(kept: pd.DataFrame, n_boot: int = 1000,
                          seed: int = 42) -> pd.DataFrame:
    """Each cell's mean feature and its mean deviation from its own pair's mean, with a
    bootstrap over pairs for the deviation."""
    feats = features(kept["chosen_feedback"], kept["protected_group"])
    cols = list(feats.columns)
    data = pd.concat([kept[["pair_id", *SPLIT]], feats], axis=1)
    dev = data[cols] - data.groupby("pair_id")[cols].transform("mean")
    data = data.join(dev.add_prefix("dev_"))
    rng = np.random.default_rng(seed)
    rows = []
    for key, part in data.groupby(SPLIT, sort=True):
        per_pair = part.groupby("pair_id")[[f"dev_{c}" for c in cols]].mean()
        idx = rng.integers(0, len(per_pair), size=(n_boot, len(per_pair)))
        row = {**dict(zip(SPLIT, key)), "n": len(part), "pairs": len(per_pair)}
        for c in cols:
            values = per_pair[f"dev_{c}"].to_numpy()
            boots = values[idx].mean(axis=1)
            row[c] = float(part[c].mean())
            row[f"dev_{c}"] = float(values.mean())
            row[f"dev_{c}_lo"] = float(np.percentile(boots, 2.5))
            row[f"dev_{c}_hi"] = float(np.percentile(boots, 97.5))
        rows.append(row)
    return pd.DataFrame(rows)


def _shuffle_within(labels: np.ndarray, groups: np.ndarray, rng) -> np.ndarray:
    out = labels.copy()
    for idx in pd.Series(np.arange(len(labels))).groupby(groups).indices.values():
        out[idx] = labels[rng.permutation(idx)]
    return out


def _cell_aucs(texts: list[str], labels: np.ndarray, groups: np.ndarray,
               classes: list[str], folds: int) -> dict[str, float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    proba = np.zeros((len(labels), len(classes)))
    for train, test in GroupKFold(n_splits=folds).split(texts, labels, groups):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        x_train = vec.fit_transform([texts[i] for i in train])
        model = LogisticRegression(max_iter=2000)
        model.fit(x_train, labels[train])
        order = [list(model.classes_).index(c) for c in classes]
        proba[test] = model.predict_proba(vec.transform([texts[i] for i in test]))[:, order]
    return {c: float(roc_auc_score(labels == c, proba[:, k])) for k, c in enumerate(classes)}


def cell_classifier(kept: pd.DataFrame, n_perm: int = 10, folds: int = 5,
                    seed: int = 42) -> pd.DataFrame:
    """One-vs-rest AUC per cell from a TF-IDF logistic regression over the rationale text, folds
    grouped by pair, against `n_perm` fits with the labels shuffled within each pair."""
    texts = kept["chosen_feedback"].astype(str).tolist()
    labels = (kept["protected_group"] + " / " + kept["condition"]).to_numpy()
    groups = kept["pair_id"].to_numpy()
    classes = sorted(set(labels))
    observed = _cell_aucs(texts, labels, groups, classes, folds)
    rng = np.random.default_rng(seed)
    null = {c: [] for c in classes}
    for _ in range(n_perm):
        for c, auc in _cell_aucs(texts, _shuffle_within(labels, groups, rng), groups,
                                 classes, folds).items():
            null[c].append(auc)
    return pd.DataFrame([{"cell": c, "n": int((labels == c).sum()), "auc": observed[c],
                          "null_mean": float(np.mean(null[c])) if null[c] else None,
                          "null_max": float(np.max(null[c])) if null[c] else None,
                          "n_perm": n_perm} for c in classes])


def cell_top_features(kept: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """What the cell classifier reads: the n-grams with the largest coefficients per cell when
    one model is fitted on every kept rationale."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    texts = kept["chosen_feedback"].astype(str).tolist()
    labels = (kept["protected_group"] + " / " + kept["condition"]).to_numpy()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000).fit(vec.fit_transform(texts), labels)
    terms = np.asarray(vec.get_feature_names_out())
    coef = model.coef_
    if coef.shape[0] == 1:  # two cells: one row of coefficients, for classes_[1]
        coef = np.vstack([-coef[0], coef[0]])
    rows = []
    for i, cell in enumerate(model.classes_):
        top = np.argsort(coef[i])[::-1][:k]
        rows += [{"cell": cell, "rank": r + 1, "ngram": terms[j],
                  "coefficient": float(coef[i][j]),
                  "rationales_with_it": int(sum(terms[j] in t.lower() for t in
                                                np.asarray(texts)[labels == cell]))}
                 for r, j in enumerate(top)]
    return pd.DataFrame(rows)


# ---- judge ---------------------------------------------------------------------------------

def judged_round(root: Path) -> pd.DataFrame:
    """The round's judged rows with both sides' judgements attached."""
    student = pd.read_parquet(root / "raw" / "student_judge.parquet")
    chosen = pd.read_parquet(root / "raw" / "chosen_judge.parquet")
    key = ["pair_id", "protected_attr", "condition"]
    if not student[key].reset_index(drop=True).equals(chosen[key].reset_index(drop=True)):
        raise ValueError(f"{root}: the two judge files are not row-aligned")
    frame = student.drop(columns=["prompt", "raw_output"], errors="ignore")
    frame = N.attach_judgements(frame, student["raw_output"].tolist(), "rejected")
    return N.attach_judgements(frame, chosen["raw_output"].tolist(), "chosen")


def judge_by_cell(judged: pd.DataFrame, min_plausibility: int = 3) -> pd.DataFrame:
    """filter_by_judge per cell: each side's pass rate and drop reasons, and the rows that pass
    both as the pipeline applies them (rejected side first)."""
    rows = []
    for key, part in judged.groupby(SPLIT, sort=True):
        strategy = str(part["rejected_strategy"].iloc[0])
        kept_r, stats_r = N.filter_by_judge(part, "rejected", strategy, min_plausibility)
        kept_c, stats_c = N.filter_by_judge(part, "chosen", None, min_plausibility)
        kept_both, _ = N.filter_by_judge(kept_r, "chosen", None, min_plausibility)
        n = len(part)
        rows.append({**dict(zip(SPLIT, key)), "n": n,
                     "rejected_pass": len(kept_r) / n, "chosen_pass": len(kept_c) / n,
                     "both_pass": len(kept_both) / n,
                     # reasons arrive as judge_* (rejected side) and chosen_judge_* (chosen)
                     **{f"rejected_{k}": v for k, v in stats_r["dropped_by_reason"].items()},
                     **stats_c["dropped_by_reason"]})
    table = pd.DataFrame(rows)
    reasons = [c for c in table.columns if "judge_" in c]
    table[reasons] = table[reasons].fillna(0).astype(int)
    for side in ("rejected", "chosen", "both"):
        lo, hi = wilson(table[f"{side}_pass"] * table["n"], table["n"])
        table[f"{side}_pass_lo"], table[f"{side}_pass_hi"] = lo, hi
    return table


def wilson(successes, n, z: float = 1.96):
    """Wilson 95% interval for a proportion, elementwise."""
    successes, n = np.asarray(successes, dtype=float), np.asarray(n, dtype=float)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def judge_homogeneity(table: pd.DataFrame) -> pd.DataFrame:
    """Chi-square test across the group x condition cells, per round, reference verdict and
    side: does the pass rate depend on the cell at all?"""
    from scipy.stats import chi2_contingency

    rows = []
    for (round_, reference), part in table.groupby(["round", "reference_decision"]):
        for side in ("rejected", "chosen", "both"):
            passed = (part[f"{side}_pass"] * part["n"]).round().astype(int)
            counts = np.stack([passed, part["n"] - passed])
            chi2, p, dof, _ = chi2_contingency(counts)
            rows.append({"round": round_, "reference_decision": reference, "side": side,
                         "cells": len(part), "chi2": float(chi2), "dof": int(dof),
                         "p": float(p), "min_pass": float(part[f"{side}_pass"].min()),
                         "max_pass": float(part[f"{side}_pass"].max())})
    return pd.DataFrame(rows)


# ---- output --------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout

    try:
        status = git("status", "--porcelain", "--untracked-files=no", "--", "scripts", "src")
        return {"commit": git("rev-parse", "HEAD").strip(),
                "uncommitted": [line[3:] for line in status.splitlines() if line.strip()]}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", choices=("invariant", "judge", "both"), default="both")
    parser.add_argument("--invariant", default=INVARIANT)
    parser.add_argument("--rounds", nargs="*", default=list(ROUNDS))
    parser.add_argument("--n-perm", type=int, default=10)
    parser.add_argument("--min-plausibility", type=int, default=3)
    parser.add_argument("--out", default=str(REPO_ROOT / "reports" / "teacher_leakage"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}

    if args.check in ("invariant", "both"):
        path = Path(resolve_output_path(args.invariant))
        inputs[args.invariant] = _sha256(path)
        raw = pd.read_parquet(path)
        kept, filt = invariant_filter_by_cell(raw)
        filt.to_csv(out / "invariant_filter_by_cell.csv", index=False)
        within_pair_deviation(kept).to_csv(out / "invariant_features_by_cell.csv", index=False)
        cell_classifier(kept, n_perm=args.n_perm).to_csv(out / "invariant_cell_classifier.csv",
                                                          index=False)
        cell_top_features(kept).to_csv(out / "invariant_cell_top_features.csv", index=False)
        print(f"invariant: {len(kept)} of {len(raw)} kept")

    if args.check in ("judge", "both"):
        tables = []
        for directory in args.rounds:
            root = Path(resolve_output_path(directory))
            for name in ("student_judge.parquet", "chosen_judge.parquet"):
                inputs[f"{directory}/raw/{name}"] = _sha256(root / "raw" / name)
            tables.append(judge_by_cell(judged_round(root), args.min_plausibility)
                          .assign(round=Path(directory).name))
        judged = pd.concat(tables, ignore_index=True)
        judged.to_csv(out / "judge_by_cell.csv", index=False)
        judge_homogeneity(judged).to_csv(out / "judge_homogeneity.csv", index=False)
        print(f"judge: {len(args.rounds)} rounds")

    meta = {"check": args.check, "inputs_sha256": inputs, "n_perm": args.n_perm,
            "min_plausibility": args.min_plausibility, "hedges": HEDGES, "intensity": INTENSITY,
            "address": ADDRESS,
            "code": code_state()}
    (out / f"meta_{args.check}.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
