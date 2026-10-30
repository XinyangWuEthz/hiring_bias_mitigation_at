#!/usr/bin/env python3
"""Validate and classify a complete paired 16-sample anchor agreement probe.

Run with: python paired_probe_analysis.py RUN_DIR
Raw generations remain in samples/. Analysis loads only narrow sample columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MODELS = ("sft_v3", "dapo_final")
VIEWS = ("attributed", "attr_free")
OUTCOMES = ("decided", "invalid", "refused")
ROUTES = (
    "attribute_bias_AB",
    "anchor_disagreement_CD_pause_RL",
    "mixed_unresolved",
    "invalid_unresolved",
)
REQUIRED_METADATA = (
    "row_id",
    "reference_decision",
    "pair_id",
    "anchor_hire_votes",
    "anchor_reject_votes",
    "anchor_valid_votes",
    "anchor_total_votes",
    "anchor_winner_votes",
)
REQUIRED_SAMPLE = (
    "model",
    "row_id",
    "view",
    "sample_index",
    "decision",
    "outcome",
    "finish_reason",
    "raw_output",
    "seed",
    "output_tokens",
)
NARROW_SAMPLE = tuple(c for c in REQUIRED_SAMPLE if c != "raw_output")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label}: missing columns {missing}")


def _require_integer_range(values: pd.Series, low: int, high: int, label: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if (
        numeric.isna().any()
        or not numeric.between(low, high).all()
        or not numeric.eq(numeric.round()).all()
    ):
        raise ValueError(f"{label}: expected integers in [{low}, {high}]")
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"{label}: expected numeric integer IDs, not string/bool IDs")


def validate_metadata(rows: pd.DataFrame, expected_rows: int) -> None:
    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    _require_columns(rows, REQUIRED_METADATA, "rows.parquet")
    if len(rows) != expected_rows or rows.row_id.duplicated().any():
        raise ValueError(f"rows.parquet: expected {expected_rows} unique rows; got {len(rows)}")
    _require_integer_range(rows.row_id, 0, expected_rows - 1, "row_id")
    if set(rows.row_id) != set(range(expected_rows)):
        raise ValueError("rows.parquet: row_id must cover every expected row")
    if not rows.reference_decision.isin(["hire", "reject"]).all():
        raise ValueError("rows.parquet: reference_decision must be hire/reject")
    if rows.pair_id.isna().any():
        raise ValueError("rows.parquet: pair_id must be present for unique-pair denominators")
    vote_cols = [c for c in REQUIRED_METADATA if c.startswith("anchor_")]
    for col in vote_cols:
        _require_integer_range(rows[col], 0, 2**31 - 1, col)
    if not (rows.anchor_hire_votes + rows.anchor_reject_votes).eq(rows.anchor_valid_votes).all():
        raise ValueError("anchor hire + reject votes must equal valid votes")
    if (rows.anchor_valid_votes > rows.anchor_total_votes).any():
        raise ValueError("anchor valid votes cannot exceed total votes")
    winner = rows[["anchor_hire_votes", "anchor_reject_votes"]].max(axis=1)
    if not rows.anchor_winner_votes.eq(winner).all():
        raise ValueError("anchor_winner_votes must equal max(hire_votes, reject_votes)")


def validate_samples(samples: pd.DataFrame, model: str, expected_rows: int, n: int) -> None:
    if n <= 0:
        raise ValueError("n must be positive")
    _require_columns(samples, NARROW_SAMPLE, model)
    if not samples.model.eq(model).all():
        raise ValueError(f"{model}: mixed or incorrect model labels")
    _require_integer_range(samples.row_id, 0, expected_rows - 1, f"{model}.row_id")
    _require_integer_range(samples.sample_index, 0, n - 1, f"{model}.sample_index")
    if not samples.view.isin(VIEWS).all():
        raise ValueError(f"{model}: unknown view")
    key = ["row_id", "view", "sample_index"]
    duplicate_count = int(samples.duplicated(key).sum())
    if duplicate_count:
        raise ValueError(f"{model}: duplicate sample keys ({duplicate_count})")
    expected = expected_rows * len(VIEWS) * n
    if len(samples) != expected:
        raise ValueError(f"{model}: incomplete samples; expected {expected}, got {len(samples)}")
    # Unique keys in the exact finite domain plus matching cardinality prove completeness.
    if not samples.outcome.isin(OUTCOMES).all():
        unknown = samples.loc[~samples.outcome.isin(OUTCOMES), "outcome"].unique().tolist()
        raise ValueError(f"{model}: unknown parser outcomes {unknown}")
    decided = samples.outcome.eq("decided")
    if not samples.loc[decided, "decision"].isin(["hire", "reject"]).all():
        raise ValueError(f"{model}: decided outcome requires a hire/reject decision")


def build_model_rows(rows: pd.DataFrame, samples: pd.DataFrame, model: str, n: int) -> pd.DataFrame:
    """Build one record per input row; callers must first validate the inputs."""
    frame = samples.merge(
        rows[["row_id", "reference_decision"]], on="row_id", validate="many_to_one"
    )
    frame["valid"] = frame.outcome.eq("decided")
    frame["right"] = frame.valid & frame.decision.eq(frame.reference_decision)
    frame["wrong"] = frame.valid & ~frame.decision.eq(frame.reference_decision)
    frame["invalid"] = frame.outcome.eq("invalid")
    frame["refused"] = frame.outcome.eq("refused")
    # Truncation is orthogonal to parsing. A parseable decision remains decided.
    frame["truncated"] = frame.finish_reason.isin(["length", "max_tokens", "max_length"])
    metrics = ["right", "wrong", "valid", "invalid", "refused", "truncated"]
    counts = frame.groupby(["row_id", "view"], sort=True)[metrics].sum().astype("int64")
    result = rows.drop(columns=[c for c in ("prompt", "free_prompt") if c in rows]).copy()
    result["model"] = model
    for view in VIEWS:
        view_counts = counts.xs(view, level="view").add_prefix(view + "_").reset_index()
        result = result.merge(view_counts, on="row_id", validate="one_to_one")
        status = pd.Series("mixed", index=result.index, dtype="object")
        status.loc[result[f"{view}_right"].eq(n)] = "all_right"
        status.loc[result[f"{view}_wrong"].eq(n)] = "all_wrong"
        status.loc[result[f"{view}_valid"].lt(n)] = "invalid"
        result[view + "_status"] = status
        result[view + "_zero_agreement"] = result[view + "_right"].eq(0)
    all_wrong = result.attributed_status.eq("all_wrong")
    strict = pd.Series("not_applicable", index=result.index, dtype="object")
    strict.loc[all_wrong] = "mixed_unresolved"
    strict.loc[all_wrong & result.attr_free_right.eq(n)] = "attribute_bias_AB"
    strict.loc[all_wrong & result.attr_free_wrong.eq(n)] = "anchor_disagreement_CD_pause_RL"
    strict.loc[all_wrong & result.attr_free_valid.lt(n)] = "invalid_unresolved"
    result["strict_route"] = strict
    result["strict_classification"] = "attributed_" + result.attributed_status
    result.loc[all_wrong, "strict_classification"] = strict[all_wrong]
    majority = pd.Series("not_applicable", index=result.index, dtype="object")
    majority.loc[all_wrong] = "tie_unresolved"
    majority.loc[all_wrong & result.attr_free_right.gt(n / 2)] = "attribute_bias_AB"
    majority.loc[all_wrong & result.attr_free_wrong.gt(n / 2)] = "anchor_disagreement_CD_pause_RL"
    majority.loc[all_wrong & result.attr_free_valid.lt(n)] = "invalid_unresolved"
    result["majority_route"] = majority
    return result.sort_values("row_id").reset_index(drop=True)


def _percent(value: int, denominator: int) -> float | None:
    return 100.0 * value / denominator if denominator else None


def _counts(frame: pd.DataFrame, column: str, levels: tuple[str, ...]) -> dict[str, int]:
    found = frame[column].value_counts().to_dict()
    return {level: int(found.get(level, 0)) for level in levels}


def summarize_group(group: pd.DataFrame, n: int) -> dict[str, Any]:
    total = len(group)
    wrong = group[group.attributed_status.eq("all_wrong")]
    wrong_count = len(wrong)
    strict_counts = _counts(wrong, "strict_route", ROUTES)
    zero_agreement = group[group.attributed_zero_agreement]
    return {
        "all_rows": total,
        "unique_pairs": int(group.pair_id.nunique()),
        "attributed_all_wrong_rows": wrong_count,
        "attributed_all_wrong_unique_pairs": int(wrong.pair_id.nunique()),
        "attributed_all_wrong_pct_all_rows": _percent(wrong_count, total),
        "attributed_status_counts": _counts(
            group, "attributed_status", ("all_right", "all_wrong", "mixed", "invalid")
        ),
        "attr_free_status_counts": _counts(
            group, "attr_free_status", ("all_right", "all_wrong", "mixed", "invalid")
        ),
        "strict_routes": {
            route: {
                "rows": count,
                "unique_pairs": int(wrong.loc[wrong.strict_route.eq(route), "pair_id"].nunique()),
                "pct_all_wrong_rows": _percent(count, wrong_count),
                "pct_all_rows": _percent(count, total),
            }
            for route, count in strict_counts.items()
        },
        "majority_route_counts": _counts(
            wrong,
            "majority_route",
            (
                "attribute_bias_AB",
                "anchor_disagreement_CD_pause_RL",
                "tie_unresolved",
                "invalid_unresolved",
            ),
        ),
        "free_correct_histogram_among_all_wrong": {
            str(k): int(wrong.attr_free_right.eq(k).sum()) for k in range(n + 1)
        },
        "zero_agreement_sensitivity": {
            "definition": "attributed_right == 0; invalid/refused may be present; not semantic all-wrong",  # noqa: E501
            "rows": len(zero_agreement),
            "unique_pairs": int(zero_agreement.pair_id.nunique()),
            "strict_all_wrong_rows": wrong_count,
            "rows_with_invalid_or_refused": int(zero_agreement.attributed_valid.lt(n).sum()),
        },
        "sample_outcomes": {
            view: {
                metric: int(group[f"{view}_{metric}"].sum())
                for metric in ("right", "wrong", "valid", "invalid", "refused", "truncated")
            }
            for view in VIEWS
        },
    }


def _json_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _group_summary(table: pd.DataFrame, keys: list[str], n: int) -> list[dict[str, Any]]:
    output = []
    for values, group in table.groupby(keys, sort=True, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        record = {key: _json_scalar(value) for key, value in zip(keys, values)}
        record.update(summarize_group(group, n))
        output.append(record)
    return output


def build_summary(table: pd.DataFrame, n: int, expected_rows: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "complete": True,
        "models": list(MODELS),
        "expected_rows_per_model": expected_rows,
        "samples_per_row_per_view": n,
        "expected_raw_samples": expected_rows * len(MODELS) * len(VIEWS) * n,
        "interpretation": {
            "right_means": "agreement with reference_decision, not independently verified truth",
            "strict_all_wrong": f"attributed valid={n} and wrong={n}",
            "strict_AB": f"strict all-wrong attributed and attr_free right={n}",
            "strict_CD": f"strict all-wrong attributed and attr_free wrong={n}",
            "majority": f"sensitivity only, all attr_free samples must be valid: right>{n / 2:g} => AB; wrong>{n / 2:g} => CD; any invalid/refused remains unresolved",  # noqa: E501
            "truncated": "overlapping diagnostic from finish_reason; a parsed decided answer remains valid",  # noqa: E501
            "pair_dependence": "row percentages are descriptive; rows sharing pair_id need not be independent; category unique-pair counts overlap",  # noqa: E501
            "stability": f"all-wrong describes only these observed {n} stochastic generations",
            "RL": "pause_RL is a review route, not an automatic stop instruction; no numeric stopping threshold is assumed",  # noqa: E501
        },
    }
    axes = {
        "by_model": ["model"],
        "by_model_and_reference": ["model", "reference_decision"],
        "by_model_reference_anchor_hire_votes": [
            "model",
            "reference_decision",
            "anchor_hire_votes",
            "anchor_valid_votes",
        ],
        "by_model_reference_anchor_winner_votes": [
            "model",
            "reference_decision",
            "anchor_winner_votes",
            "anchor_valid_votes",
        ],
        "by_model_reference_full_anchor_votes": [
            "model",
            "reference_decision",
            "anchor_hire_votes",
            "anchor_reject_votes",
            "anchor_valid_votes",
            "anchor_total_votes",
            "anchor_winner_votes",
        ],
    }
    for name, keys in axes.items():
        summary[name] = _group_summary(table, keys, n)
    model_tables = {model: table.loc[table.model.eq(model)].set_index("row_id") for model in MODELS}
    left, right = (model_tables[model] for model in MODELS)
    if not left.index.equals(right.index):
        raise ValueError("model transition analysis requires exactly matched ordered row_id sets")
    transition = left[
        [
            "pair_id",
            "reference_decision",
            "anchor_hire_votes",
            "anchor_valid_votes",
            "anchor_winner_votes",
        ]
    ].copy()
    transition["sft_v3_classification"] = left.strict_classification
    transition["dapo_final_classification"] = right.strict_classification

    def transitions(keys: list[str]) -> list[dict[str, Any]]:
        records = []
        for values, group in transition.groupby(keys, sort=True, dropna=False):
            record = dict(zip(keys, map(_json_scalar, values)))
            record.update(
                rows=len(group),
                unique_pairs=int(group.pair_id.nunique()),
                pct_all_rows=_percent(len(group), expected_rows),
            )
            records.append(record)
        return records

    transition_keys = ["sft_v3_classification", "dapo_final_classification"]
    summary["model_transitions"] = transitions(transition_keys)
    left_wrong = left.attributed_status.eq("all_wrong")
    right_wrong = right.attributed_status.eq("all_wrong")
    summary["all_wrong_changes_sft_to_dapo"] = {
        "new_all_wrong_rows": int((~left_wrong & right_wrong).sum()),
        "persistent_all_wrong_rows": int((left_wrong & right_wrong).sum()),
        "resolved_all_wrong_rows": int((left_wrong & ~right_wrong).sum()),
        "neither_all_wrong_rows": int((~left_wrong & ~right_wrong).sum()),
        "resolved_meaning": "no longer strict all-wrong; may be mixed or invalid, not necessarily all-right",  # noqa: E501
    }
    summary["model_transitions_by_reference"] = transitions(
        ["reference_decision", *transition_keys]
    )
    summary["model_transitions_by_reference_and_anchor"] = transitions(
        [
            "reference_decision",
            "anchor_hire_votes",
            "anchor_valid_votes",
            "anchor_winner_votes",
            *transition_keys,
        ]
    )
    return summary


def render_report(summary: dict[str, Any]) -> str:
    n = summary["samples_per_row_per_view"]
    rows = summary["expected_rows_per_model"]
    lines = [
        "# DAPO 与 SFT-v3 的带属性、无属性采样对照",
        "",
        f"两种模型各覆盖 {rows:,} 行，每行两种输入各采样 {n} 次。已验证模型、行号、输入版本、采样序号全部齐全且无重复。",  # noqa: E501
        "",
        f"下文的答对表示与锚点 reference_decision 一致，并不表示已验证的事实正确。全错表示本次 {n} 次采样均能解析为 hire/reject 且全部与锚点不一致；不能据此断言以后每次都会错。",  # noqa: E501
        "",
        f"严格 A/B 路径要求带属性 {n} 次全错、无属性 {n} 次全对。严格 C+D 路径要求两个版本都 {n} 次全错。无属性对错混合保留为未解决；出现无法解析或拒答则单列。多数票结果仅作敏感性分析。",  # noqa: E501
        "",
        "| 模型 | 总行数 | 唯一 pair_id 数 | 带属性全错 | A/B | C+D，评估暂停 RL | 混合未解决 | 无效未解决 |",  # noqa: E501
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["by_model"]:
        counts = group["strict_routes"]
        lines.append(
            f"| {group['model']} | {group['all_rows']} | {group['unique_pairs']} | {group['attributed_all_wrong_rows']} | "  # noqa: E501
            + " | ".join(str(counts[route]["rows"]) for route in ROUTES)
            + " |"
        )
    lines += [
        "",
        "| 模型 | 锚点决策 | 总行数 | 带属性全错 | A/B | C+D | 混合 | 无效 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["by_model_and_reference"]:
        counts = group["strict_routes"]
        lines.append(
            f"| {group['model']} | {group['reference_decision']} | {group['all_rows']} | {group['attributed_all_wrong_rows']} | "  # noqa: E501
            + " | ".join(str(counts[route]["rows"]) for route in ROUTES)
            + " |"
        )
    lines += [
        "",
        "下表按锚点胜方票数与有效票数细分。A/B 和 C+D 的百分比分母是同组带属性全错行数；该组没有全错行时记为无。",  # noqa: E501
        "",
        "| 模型 | 锚点决策 | 胜方票/有效票 | 总行数 | 全错 | A/B | C+D | 混合 | 无效 | A/B 占全错 | C+D 占全错 |",  # noqa: E501
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in summary["by_model_reference_anchor_winner_votes"]:
        counts = group["strict_routes"]
        percentages = [counts[route]["pct_all_wrong_rows"] for route in ROUTES[:2]]
        formatted = ["无" if pct is None else f"{pct:.1f}%" for pct in percentages]
        lines.append(
            f"| {group['model']} | {group['reference_decision']} | {group['anchor_winner_votes']}/{group['anchor_valid_votes']} | {group['all_rows']} | {group['attributed_all_wrong_rows']} | "  # noqa: E501
            + " | ".join(str(counts[route]["rows"]) for route in ROUTES)
            + " | "
            + " | ".join(formatted)
            + " |"
        )
    changes = summary["all_wrong_changes_sft_to_dapo"]
    lines += [
        "",
        f"按同一 row_id 对齐，从 SFT-v3 到 DAPO，新增带属性全错 {changes['new_all_wrong_rows']} 行，持续全错 {changes['persistent_all_wrong_rows']} 行，退出全错 {changes['resolved_all_wrong_rows']} 行。退出全错只说明不再满足严格全错，也可能变为混合或无效，不能直接视为全对。",  # noqa: E501
    ]
    lines += [
        "",
        f"按锚点 hire 票数、有效票数，以及胜方票数、有效票数的精确分组在 summary.json。每组保留总行数、全错行数、pair_id 数，以及各路径占全错行和全部行的百分比。原始无属性答对次数也保留为 0 到 {n} 的直方图。分母没有只留下全错行后再反推。",  # noqa: E501
        "",
        "row_classifications.parquet 保留每个模型每行的两种输入计数、锚点票数及路径。all_wrong_rows.parquet 包含严格全错行。summary.json 的 model_transitions 按同一 row_id 连接两个模型，可追踪 SFT-v3 到 DAPO 的路径变化。",  # noqa: E501
        "",
        "如果 A/B 在全错行中占主导，优先检查属性导致的决策变化。如果 C+D 占主导，先检查锚点定义、票数和标签一致性，并认真评估是否应暂停 RL。如果混合或无效行较多，应先补充证据，不能把它们归入偏见或锚点分歧。本分析没有设定自动暂停阈值，也不会停止训练任务。",  # noqa: E501
        "",
        "A/B 路径是本实验对属性敏感的证据，仍需检查具体文本变化及配对。C+D 路径说明去掉属性后仍与锚点不一致，并不能单独证明模型或锚点哪一方正确。应同时查看 hire/reject 和各票数分组，避免总比例掩盖方向差异。",  # noqa: E501
        "",
        "zero_agreement_sensitivity 把零次锚点一致但含无法解析、拒答的行单列，近似检查训练中零奖励行的构成。它没有复现完整奖励函数，不能直接等同于实际训练奖励。",  # noqa: E501
        "",
        "truncated 由 finish_reason 判断，是与解析结果交叉的诊断项。已解析的 hire/reject 即使被截断，仍按 decided 计算；因此 truncated 不可与 right、wrong、invalid、refused 直接相加。相同 pair_id 的行可能相关，各类别的 pair_id 数也可能重叠，不适合按独立行计算显著性。",  # noqa: E501
        "",
    ]
    return "\n".join(lines)


def analyze(run_dir: Path, expected_rows: int = 9140, n: int = 16) -> dict[str, Any]:
    import pyarrow.parquet as pq

    run_dir = Path(run_dir)
    rows = pd.read_parquet(run_dir / "rows.parquet")
    validate_metadata(rows, expected_rows)
    model_tables = []
    for model in MODELS:
        paths = sorted((run_dir / "samples" / model).glob("shard-*/chunk-*.parquet"))
        if not paths:
            raise ValueError(f"{model}: no raw sample shards found")
        chunks = []
        for path in paths:
            schema_columns = set(pq.read_schema(path).names)
            missing = set(REQUIRED_SAMPLE) - schema_columns
            if missing:
                raise ValueError(f"{path}: missing raw sample columns {sorted(missing)}")
            chunks.append(pd.read_parquet(path, columns=list(NARROW_SAMPLE)))
        samples = pd.concat(chunks, ignore_index=True)
        validate_samples(samples, model, expected_rows, n)
        model_tables.append(build_model_rows(rows, samples, model, n))
        del samples, chunks
    table = pd.concat(model_tables, ignore_index=True)
    summary = build_summary(table, n, expected_rows)
    # No deliverable is written until every model passes complete-data validation.
    table.to_parquet(run_dir / "row_classifications.parquet", index=False)
    table.loc[table.attributed_status.eq("all_wrong")].to_parquet(
        run_dir / "all_wrong_rows.parquet", index=False
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (run_dir / "report.md").write_text(render_report(summary))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-rows", type=int, default=9140)
    parser.add_argument("--n", type=int, default=16)
    args = parser.parse_args()
    summary = analyze(args.run_dir, expected_rows=args.expected_rows, n=args.n)
    print(
        json.dumps(
            {
                "complete": True,
                "expected_raw_samples": summary["expected_raw_samples"],
                "by_model": summary["by_model"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
