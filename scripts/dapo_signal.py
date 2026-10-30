"""How much of a GRPO batch DAPO's dynamic sampling would keep: the evidence in docs/DAPO_V3.md.

DAPO trains only on groups whose decisions disagree (0 < correct < G). Two readings say how many
such groups a prompt set holds:

  log      a GRPO job's own training log: per stretch of steps, the share of groups with zero
           reward variance (TRL's frac_reward_zero_std, which dynamic sampling would drop), the
           decision accuracy, the sampled hire rate per reference side, the token entropy and
           the clip ratio. These are under the policy as it trained. On a DAPO job the same
           readings come from its `dapo/probe_*` metrics, the first probe of each rollout:
           TRL's own reward metrics there describe the filtered, balanced batch, not the draw.
  prompts  a prompt set built by scripts/build_grpo_prompts.py: each row's chance that a group
           of G samples disagrees, 1 - p^G - (1-p)^G, with p the SFT parent's sampled error rate
           on the row (`sft_hire_rate` against `reference_decision`). This is under the parent,
           before any training.

    python scripts/dapo_signal.py log $XW/logs/xw-grpo-5116822.out --parts 8
    python scripts/dapo_signal.py prompts $XW/artifacts/semisynthetic-v3/grpo-v3-all \
        --generations 8 16 32
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

LOG_KEYS = {
    "frac_reward_zero_std": "zero-variance groups",
    "rewards/decision_reward/mean": "decision accuracy",
    "sampled_hire_rate_ref_hire": "hire rate | ref hire",
    "sampled_hire_rate_ref_reject": "hire rate | ref reject",
    "entropy": "token entropy",
    "clip_ratio/region_mean": "clip ratio",
}
#: The same readings on a DAPO job (mitigation/dapo.py), and what only DAPO has.
DAPO_KEYS = {
    "dapo/probe_zero_variance": "zero-variance groups",
    "dapo/probe_accuracy": "decision accuracy",
    "dapo/probe_hire_rate_ref_hire": "hire rate | ref hire",
    "dapo/probe_hire_rate_ref_reject": "hire rate | ref reject",
    "entropy": "token entropy",
    "clip_ratio/region_mean": "clip ratio",
    "dapo/rounds": "probe rounds",
    "dapo/fill": "fill",
}
STEP_LINE = re.compile(r"\{'loss': [^}]*\}")


def log_steps(path: Path) -> list[dict]:
    """Every per-step metrics dict TRL printed to the job log, values as floats."""
    steps = []
    for match in STEP_LINE.finditer(path.read_text(errors="replace")):
        row = ast.literal_eval(match.group(0))
        steps.append({k: float(v) for k, v in row.items() if k in LOG_KEYS or k in DAPO_KEYS})
    return steps


def keys_of(steps: list[dict]) -> dict[str, str]:
    """DAPO_KEYS for a DAPO job's log, LOG_KEYS for a GRPO job's."""
    return DAPO_KEYS if any("dapo/fill" in s for s in steps) else LOG_KEYS


def by_part(steps: list[dict], parts: int) -> list[dict]:
    """Means over `parts` consecutive, equal stretches of steps (the last takes the remainder)."""
    keys = keys_of(steps)
    size = max(1, len(steps) // parts)
    out = []
    for i in range(parts):
        chunk = steps[i * size:(i + 1) * size if i < parts - 1 else len(steps)]
        if not chunk:
            break
        row = {"steps": f"{i * size + 1}-{i * size + len(chunk)}"}
        for key in keys:
            values = [s[key] for s in chunk if key in s]
            row[key] = sum(values) / len(values) if values else float("nan")
        out.append(row)
    return out


def informative_share(p, generations: int):
    """Chance that `generations` samples at error rate p are neither all right nor all wrong."""
    return 1 - p ** generations - (1 - p) ** generations


def prompt_table(artifacts: Path, generations: list[int]):
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(artifacts / "grpo_train.parquet")
    hire = frame["reference_decision"].str.lower() == "hire"
    p = np.where(hire, 1 - frame["sft_hire_rate"], frame["sft_hire_rate"])
    rows = []
    for side, mask in (("all", np.ones(len(frame), bool)), ("hire", hire.to_numpy()),
                       ("reject", ~hire.to_numpy())):
        row = {"rows": int(mask.sum()), "side": side, "mean error rate": float(p[mask].mean()),
               "error rate 0": float((p[mask] == 0).mean())}
        for g in generations:
            row[f"informative at G={g}"] = float(informative_share(p[mask], g).mean())
        rows.append(row)
    return rows


def _print(rows: list[dict]) -> None:
    keys = list(rows[0])
    labels = {**LOG_KEYS, **DAPO_KEYS}
    print(" | ".join(labels.get(k, k) for k in keys))
    for row in rows:
        print(" | ".join(f"{row[k]:.3f}" if isinstance(row[k], float) else str(row[k])
                         for k in keys))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    log_cmd = sub.add_parser("log")
    log_cmd.add_argument("path", type=Path)
    log_cmd.add_argument("--parts", type=int, default=8)
    prompts_cmd = sub.add_parser("prompts")
    prompts_cmd.add_argument("artifacts", type=Path)
    prompts_cmd.add_argument("--generations", type=int, nargs="+", default=[16])
    args = parser.parse_args()
    if args.command == "log":
        steps = log_steps(args.path)
        print(f"{args.path.name}: {len(steps)} steps")
        _print(by_part(steps, args.parts))
    else:
        _print(prompt_table(args.artifacts, args.generations))


if __name__ == "__main__":
    main()
