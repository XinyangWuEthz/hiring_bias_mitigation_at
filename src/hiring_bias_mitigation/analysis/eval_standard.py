"""The comparison standard of `docs/EVAL_STANDARD.md`, as code.

`judge_rows` takes the per-run reference rows (`analysis/reference_metrics.py`) and adds, for
every run, which guardrails it passes and whether it beats its reference arm. The thresholds
are the ones the user fixed on 2026-10-20. They change here and in the document together, and
only on the user's decision.

The reference arm of a cell is declared in `REFERENCE_ARMS`: every method trained after SFT is
compared with that SFT run, and the SFT run with its own base. A row is judged against the arm
its primary op-margin-v2 report compares it with.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

STANDARD = "eval-standard-v1 (2026-10-20)"

PASS, FAIL, NOT_MEASURED, NO_REFERENCE, NOT_APPLICABLE = (
    "pass", "fail", "not measured", "no reference row", "n/a")


@dataclass(frozen=True)
class Thresholds:
    #: The arm's own adapter, the last of its merge chain, keeps at least this share of its
    #: update through the bf16 merge (`mitigation/merge.py`).
    min_merge_kept_pct: float = 80.0
    #: Hire rate on prompts GPT-4o hires falls by at most this much against the reference arm.
    max_tpr_drop_pp: float = 10.0
    #: Agreement with GPT-4o falls by at most this much against the reference arm.
    max_agree_drop_pp: float = 5.0
    #: Expected prompts without a usable decision: parse failures, refusals and prompts the audit
    #: did not produce, against the grid its config asks for.
    max_undecided_pct: float = 1.0
    #: A "better" within this distance of zero needs a second seed with the same sign.
    second_seed_within_pp: float = 1.0


_SFT_V1 = "Qwen3.5-9B--en--sft--adapter--en_only_cscs"
_SFT_V3 = "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional"

#: The reference arm of each (model, language, serving stack) cell: the SFT run every later
#: method is compared with. The SFT run itself is compared with its base. In the 9B English
#: cell SFT-v3 replaced SFT v1 on 2026-10-20 (user: "V3 as new SFT comparison"), after it beat
#: SFT v1 under this standard (docs/V3_NEW_DATA.md, round 2).
REFERENCE_ARMS: dict[tuple[str, str, str], str] = {
    ("Qwen/Qwen3.5-9B", "en", "gh200"): _SFT_V3,
}

#: Runs that keep the comparison they were judged by when their cell's reference arm changed.
#: An arm trained from SFT v1 is an ablation against SFT v1, so it is still judged against it;
#: SFT v1 against its base; SFT-v3 against SFT v1, the comparison by which it became the
#: reference. Without this, refreshing their rows would look for a comparison with SFT-v3.
JUDGED_BEFORE: dict[tuple[str, str, str], dict[str, str]] = {
    ("Qwen/Qwen3.5-9B", "en", "gh200"): {
        _SFT_V1: "base",
        _SFT_V3: _SFT_V1,
        **{f"Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_{arm}": _SFT_V1 for arm in (
            "r1_step100", "r1_step300", "r2_step50", "r2_step102",
            "r2em_step50", "r2em_step100", "r2em_step148")},
        "Qwen3.5-9B--en--grpo--adapter--en_only_grpo_pilot": _SFT_V1,
        "Qwen3.5-9B--en--sft--adapter--en_only_rft": _SFT_V1,
    },
}


def declared_reference(run_name: str, model: str | None, lang: str | None,
                       stack: str | None) -> str | None:
    """The comparison the standard reads for this run: what `JUDGED_BEFORE` keeps for it, else
    "base" for the cell's reference arm and the reference arm for any other run of the cell;
    None outside a declared cell."""
    cell = (model, lang, stack)
    arm = REFERENCE_ARMS.get(cell)
    if arm is None:
        return None
    kept = JUDGED_BEFORE.get(cell, {})
    if run_name in kept:
        return kept[run_name]
    return "base" if run_name == arm else arm


def _num(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def reference_arm(row: dict) -> str | None:
    """The run this row is judged against: the op-margin report's `against`, or its baseline."""
    against = row.get("primary_against")
    if against == "base":
        return row.get("baseline_run")
    if against and not str(against).startswith("several"):
        return str(against)
    return None


def judge(row: dict, reference: dict | None, t: Thresholds = Thresholds()) -> dict:
    """The standard applied to one run. `reference` is its reference arm's row, if there is one."""
    out = {"std_version": STANDARD, "std_reference_arm": reference_arm(row),
           "std_tpr_vs_ref_pp": None, "std_agree_vs_ref_pp": None}
    if row.get("family", "none") == "none":
        out.update(guard_merge=NOT_APPLICABLE, guard_tpr=NOT_APPLICABLE,
                   guard_agree=NOT_APPLICABLE, guard_decided=NOT_APPLICABLE,
                   std_decision=NOT_APPLICABLE, std_verdict="baseline")
        return out

    kept = _num(row.get("merge_kept_pct"))
    guards = {"guard_merge": NOT_MEASURED if kept is None
              else PASS if kept >= t.min_merge_kept_pct else FAIL}
    decided = _num(row.get("decided_pct"))
    if decided is None:  # rows written without the expected grid: parse failures and refusals
        parts = [_num(row.get("parse_failure_pct")), _num(row.get("refusal_pct"))]
        undecided = None if None in parts else sum(parts)
    else:
        undecided = 100.0 - decided
    guards["guard_decided"] = (NOT_MEASURED if undecided is None
                               else PASS if undecided <= t.max_undecided_pct else FAIL)
    if reference is None:
        guards["guard_tpr"] = guards["guard_agree"] = NO_REFERENCE
    else:
        for name, column, limit in (("tpr", "tpr_vs_gpt4o_pct", t.max_tpr_drop_pp),
                                    ("agree", "agree_gpt4o_pct", t.max_agree_drop_pp)):
            mine, theirs = _num(row.get(column)), _num(reference.get(column))
            if mine is None or theirs is None:
                guards[f"guard_{name}"] = NOT_MEASURED
                continue
            out[f"std_{name}_vs_ref_pp"] = mine - theirs
            guards[f"guard_{name}"] = PASS if mine - theirs >= -limit else FAIL
    out.update(guards)

    delta, high = _num(row.get("primary_delta_pp")), _num(row.get("primary_ci_high_pp"))
    conforms = row.get("primary_conforms")
    if delta is None:
        decision = "no matched comparison"
    elif conforms is not True:
        decision = "comparison does not conform"
    elif high is None:
        decision = "no interval"
    else:
        decision = "better" if high < 0 else "not better"
    out["std_decision"] = decision

    failed = [name.removeprefix("guard_") for name, status in guards.items() if status == FAIL]
    open_ = [name.removeprefix("guard_") for name, status in guards.items()
             if status in (NOT_MEASURED, NO_REFERENCE)]
    if decision == "better":
        if failed:
            verdict = "not better: fails " + ", ".join(failed)
        elif open_:
            verdict = "undecided: not measured " + ", ".join(open_)
        elif abs(delta) <= t.second_seed_within_pp:
            verdict = "better, pending a second seed"
        else:
            verdict = "better"
    elif decision == "not better":
        verdict = "not better: the interval reaches 0" + (
            "; also fails " + ", ".join(failed) if failed else "")
    else:
        verdict = f"undecided: {decision}"
    out["std_verdict"] = verdict
    return out


def judge_rows(rows: list[dict], t: Thresholds = Thresholds()) -> list[dict]:
    """`judge` for every row, each against its reference arm's row when that row exists."""
    by_run = {row["run"]: row for row in rows}
    return [judge(row, by_run.get(reference_arm(row)), t) for row in rows]
