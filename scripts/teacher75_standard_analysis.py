"""Validate two independent frozen-policy ledgers per pair and aggregate rows.

This module never chooses semantic evidence states or adjudicates a disagreement.
The canonical validator checks source quotations, checklist identity, counts, and
the decision gate. Natural-language evidence judgments remain the reviewers'.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "scripts" else HERE.parents[1]
STANDARD_PATH = Path(__file__).with_name("hiring_standard.py")
if not STANDARD_PATH.exists():
    STANDARD_PATH = ROOT / "outputs/hiring_standard_v1/hiring_standard.py"


def load_standard(path: Path):
    spec = importlib.util.spec_from_file_location("hiring_standard_full_audit", path)
    require_spec = spec is not None and spec.loader is not None
    if not require_spec:
        raise ValueError(f"Cannot load standard validator at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


standard = load_standard(STANDARD_PATH) if STANDARD_PATH.exists() else None

CATEGORIES = ("dapo_degradation", "anchor_problem", "insufficient_information")
CN = {
    "dapo_degradation": "DAPO 判断退化",
    "anchor_problem": "锚点有问题",
    "insufficient_information": "信息不足或审阅分歧",
}
LEGACY_FIELDS = (
    "audit_category", "teacher_correction_supported", "correction_evidence_basis",
    "correction_condition", "anchor_reason_issue", "blind_agreement",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def unique_by(records: list[dict], key: str, name: str) -> dict:
    require(isinstance(records, list), f"{name} must be an array")
    result = {}
    for record in records:
        require(isinstance(record, dict), f"{name} entries must be objects")
        require(key in record, f"{name} entry missing {key}")
        identifier = record[key]
        require(identifier not in result, f"{name} duplicate {key}={identifier}")
        result[identifier] = record
    return result


def classify(anchor: str, decisions: list[str]) -> tuple[str, str, bool]:
    require(anchor in {"hire", "reject"}, f"Invalid anchor {anchor}")
    require(len(decisions) == 2, "Exactly two reviewer decisions are required")
    require(all(x in {"hire", "reject", "review"} for x in decisions), "Invalid audit decision")
    if decisions[0] != decisions[1] or decisions[0] == "review":
        return "insufficient_information", "review", False
    if decisions[0] == anchor:
        return "dapo_degradation", decisions[0], True
    return "anchor_problem", decisions[0], False


def assert_selected_row(row: dict) -> None:
    """The selected 75 rows have fixed sampling invariants, independent of audit."""
    for prefix, right, wrong, status in (
        ("sft_attr_free", 16, 0, "all_right"),
        ("dapo_attributed", 0, 16, "all_wrong"),
        ("dapo_attr_free", 0, 16, "all_wrong"),
    ):
        for field, expected in (("right", right), ("wrong", wrong), ("valid", 16),
                                ("invalid", 0), ("refused", 0), ("truncated", 0),
                                ("status", status)):
            require(row.get(f"{prefix}_{field}") == expected,
                    f"row {row.get('row_id')}: {prefix}_{field} must equal {expected}")
    require(row.get("reference_decision") in {"hire", "reject"}, "Invalid row reference decision")
    require(row.get("anchor_reference") == row["reference_decision"], "Row anchor/reference mismatch")
    for field in ("anchor_hire_votes", "anchor_reject_votes", "anchor_valid_votes", "anchor_total_votes",
                  "anchor_winner_votes", "sft_attributed_right", "sft_attributed_wrong",
                  "sft_attributed_valid", "sft_attributed_invalid", "sft_attributed_refused"):
        require(type(row.get(field)) is int and row[field] >= 0, f"Invalid nonnegative integer {field}")
    require(row["anchor_hire_votes"] + row["anchor_reject_votes"] == row["anchor_valid_votes"],
            "Anchor vote totals do not reconcile")
    require(row["anchor_valid_votes"] <= row["anchor_total_votes"], "Valid anchor votes exceed total")
    expected_winner = row["anchor_hire_votes"] if row["reference_decision"] == "hire" else row["anchor_reject_votes"]
    require(row["anchor_winner_votes"] == expected_winner, "Anchor winner votes mismatch")
    require(expected_winner > row["anchor_valid_votes"] / 2, "Anchor is not a strict majority")
    require(row["sft_attributed_right"] + row["sft_attributed_wrong"] == row["sft_attributed_valid"],
            "SFT attributed valid counts do not reconcile")
    require(row["sft_attributed_valid"] + row["sft_attributed_invalid"] + row["sft_attributed_refused"] == 16,
            "SFT attributed draw counts do not total 16")


def quote_instances(ledger: dict) -> list[tuple[str, str]]:
    values = []
    for section in ("mandatory", "preferred"):
        for item in ledger[section]:
            values.append(("jd", item["jd_quote"]))
            values.extend(("cv", q) for q in item["cv_quotes"])
    if ledger["role_fit"]["jd_quote"]:
        values.append(("jd", ledger["role_fit"]["jd_quote"]))
    values.extend(("cv", q) for q in ledger["role_fit"]["cv_quotes"])
    return values


def unresolved_items(ledger: dict) -> list[dict]:
    return [{k: item[k] for k in ("id", "requirement", "state", "jd_quote", "cv_quotes", "basis")}
            for item in ledger["mandatory"] if item["state"] in {"unknown", "contradicted"}]


def state_disagreements(first: dict, second: dict) -> list[dict]:
    other = {x["id"]: x for x in second["mandatory"]}
    return [{"id": x["id"], "requirement": x["requirement"], "states": [x["state"], other[x["id"]]["state"]]}
            for x in first["mandatory"] if x["state"] != other[x["id"]]["state"]]


def aggregate(cases: list[dict], fields: tuple[str, ...]) -> list[dict]:
    groups = {}
    for case in cases:
        key = tuple(case[x] for x in fields)
        if key not in groups:
            groups[key] = {**dict(zip(fields, key)), "pairs": 0, "rows": 0,
                           "categories": {c: {"pairs": 0, "rows": 0} for c in CATEGORIES}}
        g = groups[key]
        g["pairs"] += 1
        g["rows"] += case["rows"]
        g["categories"][case["audit_category"]]["pairs"] += 1
        g["categories"][case["audit_category"]]["rows"] += case["rows"]
    return [groups[key] for key in sorted(groups)]


def analyze(policy: dict, sources: list[dict], rubrics: list[dict], review_files: list[dict],
            old_rows: list[dict], pair_metadata: list[dict], expected_pairs: int = 40,
            expected_rows: int = 75, standard_commit: str = "323d4f5") -> dict[str, Any]:
    """Produce aggregates only after every supplied ledger and row passes checks."""
    source_map = unique_by(sources, "case_id", "sources")
    require(standard is not None, "Validator unavailable; supply --standard-script")
    rubric_map = unique_by(rubrics, "case_id", "rubrics")
    pair_map = unique_by(pair_metadata, "case_id", "pair_metadata")
    row_map = unique_by(old_rows, "row_id", "old_rows")
    require(len(source_map) == expected_pairs, f"Expected {expected_pairs} source pairs")
    require(len(row_map) == expected_rows, f"Expected {expected_rows} rows")
    case_ids = set(source_map)
    require(case_ids == set(rubric_map) == set(pair_map), "Source/rubric/metadata case sets differ")
    require(len({x["pair_id"] for x in pair_metadata}) == expected_pairs, "Pair IDs are not unique")
    reviews_by_case = defaultdict(list)
    seen_reviewers = set()
    for review_file in review_files:
        reviewer = review_file["reviewer"]
        require(reviewer not in seen_reviewers, f"Duplicate reviewer identity {reviewer}")
        seen_reviewers.add(reviewer)
        records = review_file["cases"]
        require(isinstance(records, dict) and records, f"Empty or malformed reviews from {reviewer}")
        require(set(records).issubset(case_ids), f"Unknown review case in {reviewer}")
        for case_id, ledger in records.items():
            reviews_by_case[case_id].append((reviewer, review_file["file"], ledger))
    require(set(reviews_by_case) == case_ids, "Not all cases have reviews")
    for case_id in sorted(case_ids):
        require(len(reviews_by_case[case_id]) == 2, f"{case_id} must have exactly two ledgers")

    rows_by_case = defaultdict(list)
    for row in old_rows:
        assert_selected_row(row)
        require(row["case_id"] in case_ids, f"Unknown row case {row['case_id']}")
        require(row["audit_category"] in CATEGORIES, "Unknown legacy category")
        rows_by_case[row["case_id"]].append(row)
    require(set(rows_by_case) == case_ids, "Every case must have at least one selected row")

    cases, updated_rows = [], []
    all_quotes = []
    canonical_fingerprint = standard.canonical_policy_sha256(policy)
    for case_id in sorted(case_ids):
        source, rubric, meta = source_map[case_id], rubric_map[case_id], pair_map[case_id]
        rows = sorted(rows_by_case[case_id], key=lambda x: x["row_id"])
        require(meta["rows"] == len(rows), f"{case_id} pair row count mismatch")
        require(sorted(meta["row_ids"]) == [x["row_id"] for x in rows], f"{case_id} pair row IDs mismatch")
        for field in ("pair_id", "reference_decision", "anchor_winner_votes", "anchor_valid_votes"):
            require(all(row[field] == meta[field] for row in rows), f"{case_id} inconsistent {field}")
        require(len({r["audit_category"] for r in rows}) == 1, f"{case_id} inconsistent legacy category")
        validated = []
        for reviewer, path, ledger in sorted(reviews_by_case[case_id]):
            try:
                result = standard.validate_record(ledger, source["job_description"], source["candidate_profile"], policy, rubric)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{case_id}/{reviewer}: {error}") from error
            quotes = quote_instances(ledger)
            all_quotes.extend((case_id, kind, q) for kind, q in quotes)
            validated.append({"reviewer": reviewer, "file": path, "decision": result["decision"],
                              "counts": result["counts"], "role_fit": ledger["role_fit"],
                              "material_conflict": ledger["material_conflict"],
                              "data_quality_flags": ledger["data_quality_flags"],
                              "unresolved_mandatory": unresolved_items(ledger),
                              "feedback": ledger["feedback"], "source_quote_instances": len(quotes),
                              "ledger_sha256": canonical_hash(ledger)})
        ordered_ledgers = [x[2] for x in sorted(reviews_by_case[case_id])]
        decisions = [v["decision"] for v in validated]
        category, decision, supported = classify(meta["reference_decision"], decisions)
        disagreements = state_disagreements(*ordered_ledgers)
        case = {
            "case_id": case_id, "pair_id": meta["pair_id"], "row_ids": meta["row_ids"], "rows": len(rows),
            "reference_decision": meta["reference_decision"],
            "anchor_winner_votes": meta["anchor_winner_votes"], "anchor_valid_votes": meta["anchor_valid_votes"],
            "policy_id": policy["policy_id"], "policy_sha256": canonical_fingerprint,
            "standard_commit": standard_commit, "frozen_rubric_sha256": canonical_hash(rubric),
            "source_sha256": canonical_hash(source),
            "legacy_audit_category": rows[0]["audit_category"],
            "legacy_teacher_correction_supported": rows[0]["teacher_correction_supported"],
            "audit_category": category, "audit_decision": decision,
            "reviewer_decisions": decisions, "reviewer_decision_agreement": decisions[0] == decisions[1],
            "mandatory_state_disagreements": disagreements,
            "role_fit_disagreement": ordered_ledgers[0]["role_fit"]["state"] != ordered_ledgers[1]["role_fit"]["state"],
            "material_conflict_disagreement": ordered_ledgers[0]["material_conflict"] != ordered_ledgers[1]["material_conflict"],
            "teacher_correction_supported": supported,
            "teacher_correction_label": decision if supported else None,
            "target_reason": [{"reviewer": v["reviewer"], "feedback": ledger["feedback"],
                               "role_fit_basis": ledger["role_fit"]["basis"],
                               "mandatory_evidence": [{k: item[k] for k in ("id", "state", "basis", "date_precision", "cv_quotes")}
                                                      for item in ledger["mandatory"]]}
                              for v, ledger in zip(validated, ordered_ledgers)] if supported else None,
            "correction_conditions": {
                "scope": "Same candidate and JD evidence, frozen checklist, and hiring-evidence-v1 decision meaning",
                "reviewer_data_quality_flags": [ledger["data_quality_flags"] for ledger in ordered_ledgers],
                "year_precision_is_nominal_not_exact_completed_years": any(
                    item["date_precision"] == "year" for ledger in ordered_ledgers for item in ledger["mandatory"]),
            } if supported else None,
            "review_queue_reason": ("reviewer_decision_disagreement" if decisions[0] != decisions[1] else "both_review") if category == "insufficient_information" else None,
            "review_ledgers": validated,
        }
        cases.append(case)
        for old_row in rows:
            row = copy.deepcopy(old_row)
            for field in LEGACY_FIELDS:
                if field in row:
                    row[f"legacy_{field}"] = row.pop(field)
            row.update({key: copy.deepcopy(case[key]) for key in (
                "policy_id", "policy_sha256", "standard_commit", "audit_category", "audit_decision",
                "reviewer_decisions", "reviewer_decision_agreement", "mandatory_state_disagreements",
                "teacher_correction_supported", "teacher_correction_label", "review_queue_reason",
                "target_reason", "correction_conditions",
            )})
            row["audit_counts"] = [r["counts"] for r in validated]
            row["audit_review_files"] = [r["file"] for r in validated]
            row["correction_evidence_basis"] = "two_independent_same_policy_resolved_ledgers" if supported else None
            row["teacher_source"] = "SFT-v3 attribute-free k=16; label validated by two independent ledgers" if supported else None
            row["teacher_rationale_approved_for_reuse"] = False
            row["rationale_reuse_unchecked"] = True
            updated_rows.append(row)
    updated_rows.sort(key=lambda r: r["row_id"])
    targets = [r for r in updated_rows if r["teacher_correction_supported"]]
    queue = [c for c in cases if c["audit_category"] == "insufficient_information"]
    category_counts = {category: {"pairs": sum(c["audit_category"] == category for c in cases),
                                  "rows": sum(c["rows"] for c in cases if c["audit_category"] == category)}
                       for category in CATEGORIES}
    transitions = []
    for old in CATEGORIES:
        for new in CATEGORIES:
            subset = [c for c in cases if c["legacy_audit_category"] == old and c["audit_category"] == new]
            transitions.append({"legacy_category": old, "new_category": new, "pairs": len(subset),
                                "rows": sum(c["rows"] for c in subset), "case_ids": [c["case_id"] for c in subset]})
    prior_targets = [r for r in updated_rows if r["legacy_teacher_correction_supported"]]
    retained = [r for r in prior_targets if r["teacher_correction_supported"]]
    removed = [r for r in prior_targets if not r["teacher_correction_supported"]]
    added = [r for r in targets if not r["legacy_teacher_correction_supported"]]
    right = sum(r["sft_attributed_right"] for r in targets)
    valid = sum(r["sft_attributed_valid"] for r in targets)
    total = 16 * len(targets)
    summary = {
        "policy_id": policy["policy_id"], "policy_sha256": canonical_fingerprint,
        "standard_commit": standard_commit, "pairs": len(cases), "rows": len(updated_rows),
        "review_ledgers": 2 * len(cases), "review_file_count": len(review_files),
        "categories": category_counts, "by_anchor": aggregate(cases, ("reference_decision",)),
        "audit_decision_counts": {decision: {"pairs": sum(c["audit_decision"] == decision for c in cases),
                                             "rows": sum(c["rows"] for c in cases if c["audit_decision"] == decision)}
                                  for decision in ("hire", "reject", "review")},
        "resolved_coverage": {
            "pairs_resolved": sum(c["audit_decision"] != "review" for c in cases),
            "pairs_total": len(cases),
            "pair_fraction": sum(c["audit_decision"] != "review" for c in cases) / len(cases),
            "rows_resolved": sum(c["rows"] for c in cases if c["audit_decision"] != "review"),
            "rows_total": len(updated_rows),
            "row_fraction": sum(c["rows"] for c in cases if c["audit_decision"] != "review") / len(updated_rows),
        },
        "by_anchor_votes": aggregate(cases, ("reference_decision", "anchor_winner_votes", "anchor_valid_votes")),
        "legacy_transitions": transitions,
        "decision_disagreement": {"pairs": sum(not c["reviewer_decision_agreement"] for c in cases),
                                  "rows": sum(c["rows"] for c in cases if not c["reviewer_decision_agreement"]),
                                  "case_ids": [c["case_id"] for c in cases if not c["reviewer_decision_agreement"]]},
        "prior_seven_candidates": {"prior_rows": len(prior_targets), "prior_pairs": len({r["case_id"] for r in prior_targets}),
                                   "retained_row_ids": [r["row_id"] for r in retained],
                                   "removed_row_ids": [r["row_id"] for r in removed],
                                   "added_row_ids": [r["row_id"] for r in added]},
        "teacher_targets": {"pairs": len({r["case_id"] for r in targets}), "rows": len(targets),
                            "case_ids": sorted({r["case_id"] for r in targets}),
                            "row_ids": [r["row_id"] for r in targets],
                            "sft_attributed_agreement": right, "sft_attributed_valid": valid,
                            "sft_attributed_draws": total,
                            "sft_attributed_agreement_per_draw": right / total if total else None,
                            "sft_attributed_agreement_per_valid": right / valid if valid else None,
                            "sft_attributed_right_min": min((r["sft_attributed_right"] for r in targets), default=None),
                            "sft_attributed_right_max": max((r["sft_attributed_right"] for r in targets), default=None)},
        "source_quote_validation": {"ledger_quote_instances": len(all_quotes),
                                    "distinct_case_source_quotes": len(set(all_quotes)),
                                    "matching_instances": len(all_quotes), "failed_instances": 0,
                                    "scope": "all mandatory/preferred JD and CV quotes plus role-fit JD and CV quotes in every ledger"},
        "invariants": {"sft_attribute_free_anchor_agreement_per_row": "16/16",
                       "dapo_attributed_anchor_agreement_per_row": "0/16",
                       "dapo_attribute_free_anchor_agreement_per_row": "0/16",
                       "exactly_two_validated_ledgers_per_pair": True},
        "limitations": ["Independent model text audits, not human-adjudicated gold labels.",
                        "Source-quote and gate validation does not prove semantic entailment or complete JD extraction.",
                        "Results cover these 40 pairs and 75 selected rows only, not all 1,081 C+D rows or 9,140 prompts.",
                        "Teacher targets indicate supported correction labels; they do not establish future training benefit.",
                        "Original teacher feedback is not automatically approved for reuse.",
                        "No training or new sampling was run by this analysis."],
    }
    require(sum(x["pairs"] for x in category_counts.values()) == expected_pairs, "Final pair counts do not reconcile")
    require(sum(x["rows"] for x in category_counts.values()) == expected_rows, "Final row counts do not reconcile")
    return {"audit_cases_40": cases, "audit_rows_75": updated_rows,
            "teacher_correction_candidates": targets, "review_queue": queue, "summary": summary}


def cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def counts_text(counts: dict) -> str:
    return "/".join(str(counts[k]) for k in ("H", "D", "I", "U", "F"))


def report(data: dict) -> str:
    summary, cases, targets = data["summary"], data["audit_cases_40"], data["teacher_correction_candidates"]
    teacher = summary["teacher_targets"]
    lines = ["# 75 行、40 个 pair：统一标准全量复审", "",
             f"标准 `{summary['policy_id']}`，提交 `{summary['standard_commit']}`。本次每个 pair 由两名独立审阅者使用同一份冻结 JD 清单审阅，共 {summary['review_ledgers']} 份证据账本。",
             f"标准指纹 `{summary['policy_sha256']}`，采用 canonical-json-sha256。", "",
             "hire 要求所有明示必需项都有支持证据，且岗位匹配有正面证据；明确不满足必需项才判 reject；未知、关键矛盾或两名审阅者结论不同均进入复核队列。审阅意见分歧未由第三人裁决。", "",
             "## 更新后的分类", "", "| 分类 | Pair 数 | 行数 |", "|---|---:|---:|"]
    for category in CATEGORIES:
        n = summary["categories"][category]
        lines.append(f"| {CN[category]} | {n['pairs']} | {n['rows']} |")
    coverage = summary["resolved_coverage"]
    decision_counts = summary["audit_decision_counts"]
    lines += ["", "审计标签分母：" + "；".join(
        f"{decision} {decision_counts[decision]['pairs']} 个 pair / {decision_counts[decision]['rows']} 行"
        for decision in ("hire", "reject", "review")) + "。",
        f"二元结论覆盖率为 pair {coverage['pairs_resolved']}/{coverage['pairs_total']} = {coverage['pair_fraction']:.1%}，行 {coverage['rows_resolved']}/{coverage['rows_total']} = {coverage['row_fraction']:.1%}。review 未记为二元错误，也未加入已解决子集的二元比较分母。"]
    lines += ["", "两名审阅者一致给出 hire/reject，且该标签等于锚点，归为 DAPO 判断退化；一致标签与锚点相反，归为锚点有问题；任一 review 或结论分歧，归为信息不足。", "",
              "这一子集的原始采样不变：SFT-v3 无属性与锚点一致 16/16，DAPO 带属性及无属性均与锚点一致 0/16。历史上的“全错”表示与旧锚点不一致，本次分类表示 v1 标准下的证据判断。因此，只有审阅确认的第一类才给冻结 SFT 教师提供标签纠正目标。", "",
              "## 按锚点标签和票数分层", "",
              "| 锚点 | 胜出/有效票 | Pair/行 | DAPO 退化 pair/行 | 锚点问题 pair/行 | 信息不足 pair/行 |", "|---|---|---|---|---|---|"]
    for group in summary["by_anchor_votes"]:
        parts = [f"{group['categories'][c]['pairs']}/{group['categories'][c]['rows']}" for c in CATEGORIES]
        lines.append(f"| {group['reference_decision']} | {group['anchor_winner_votes']}/{group['anchor_valid_votes']} | {group['pairs']}/{group['rows']} | " + " | ".join(parts) + " |")
    lines += ["", "## 冻结 SFT 教师的纠正目标", "",
              f"保留 {teacher['pairs']} 个 pair、{teacher['rows']} 行。这里只确认标签方向，不表示训练后已经纠正，也不自动复用 SFT 原来的反馈文字。", "",
              "| Pair | 行 ID | 目标 | SFT 带属性与目标一致 | 审阅计数 H/D/I/U/F |", "|---|---:|---|---:|---|"]
    for row in targets:
        counts = "；".join(counts_text(x) for x in row["audit_counts"])
        lines.append(f"| {row['case_id']} | {row['row_id']} | {row['teacher_correction_label']} | {row['sft_attributed_right']}/16 | {counts} |")
    if targets:
        lines += ["", f"目标行中 SFT 带属性采样与目标一致 {teacher['sft_attributed_agreement']}/{teacher['sft_attributed_draws']}，即 {teacher['sft_attributed_agreement_per_draw']:.1%}。每行范围 {teacher['sft_attributed_right_min']} 至 {teacher['sft_attributed_right_max']}/16。",
                  "", "每个候选的 target_reason 保存本轮两份审阅的岗位匹配依据及必需项证据。correction_conditions 保存适用的固定输入和数据质量标记。仅有年份的年限按名义差值判断，不能解读为已核实精确满年。原始 SFT rationale 未逐条核验，rationale_reuse_unchecked=true。"]
    prior = summary["prior_seven_candidates"]
    lines += ["", f"原先 {prior['prior_pairs']} 个 pair、{prior['prior_rows']} 行候选：保留 {len(prior['retained_row_ids'])} 行，移出 {len(prior['removed_row_ids'])} 行；本次另新增 {len(prior['added_row_ids'])} 行。",
              f"保留行：{prior['retained_row_ids']}。移出行：{prior['removed_row_ids']}。新增行：{prior['added_row_ids']}。", "",
              "## 与原先 40-pair 审计的迁移", "", "| 旧分类 | 新分类 | Pair 数 | 行数 | Pair ID |", "|---|---|---:|---:|---|"]
    for t in summary["legacy_transitions"]:
        if t["pairs"]:
            lines.append(f"| {CN[t['legacy_category']]} | {CN[t['new_category']]} | {t['pairs']} | {t['rows']} | {', '.join(t['case_ids'])} |")
    lines += ["", "## 逐 pair 证据和待确认项", "",
              "计数顺序为 H/D/I/U/F：必需项总数、直接支持、有限推断支持、未知、明确不满足。完整引文和依据保留在各审阅 JSON 中。", ""]
    for case in cases:
        lines += [f"### {case['case_id']} · {CN[case['audit_category']]}", "",
                  f"行 {case['row_ids']}；锚点 {case['reference_decision']}，{case['anchor_winner_votes']}/{case['anchor_valid_votes']} 票。两份结论 {' / '.join(case['reviewer_decisions'])}。", ""]
        for index, ledger in enumerate(case["review_ledgers"], 1):
            lines += [f"审阅 {index}：{ledger['decision']}；H/D/I/U/F = {counts_text(ledger['counts'])}。{ledger['feedback']} [证据账本]({ledger['file']})", ""]
            if ledger["unresolved_mandatory"]:
                for item in ledger["unresolved_mandatory"]:
                    lines.append(f"- {item['id']} {item['state']}：{item['requirement']}。{item['basis']}")
                lines.append("")
            if ledger["material_conflict"]:
                lines += [f"该审阅标记关键冲突：{cell(ledger['data_quality_flags'])}", ""]
            if not ledger["unresolved_mandatory"]:
                lines += [f"岗位匹配依据：{ledger['role_fit']['basis']}", ""]
        if not case["reviewer_decision_agreement"]:
            lines += ["两名审阅者结论不同，保留分歧并进入复核队列，不形成硬纠正目标。", ""]
        if case["mandatory_state_disagreements"]:
            items = "; ".join(f"{x['id']}: {' / '.join(x['states'])}" for x in case["mandatory_state_disagreements"])
            lines += [f"逐项状态分歧：{items}。", ""]
    quotes = summary["source_quote_validation"]
    lines += ["## 校验和适用范围", "",
              f"{summary['review_ledgers']} 份账本全部通过政策指纹、冻结清单、计数、决策门槛和原文引文校验。共核对 {quotes['ledger_quote_instances']} 个引文实例，按 pair、来源、引文去重后为 {quotes['distinct_case_source_quotes']} 个，未发现原文不匹配。", "",
              "这些是独立模型文本审计，并非人工金标。程序校验不能证明自然语言推断正确，不能替代 JD 清单完整性和证据含义的审查。结论只覆盖这 40 个 pair、75 行，不外推到全部 1,081 条 C+D 或 9,140 道题。", "",
              "本轮未启动训练或新采样。先保留 review 队列和分歧，再据已确认目标设计后续实验。", "",
              "[逐 pair 结果](audit_cases_40.json) · [75 行分类](audit_rows_75.json) · [纠正目标](teacher_correction_candidates.json) · [复核队列](review_queue.json) · [统计](summary.json)", ""]
    return "\n".join(lines)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    global standard
    parser = argparse.ArgumentParser(description=__doc__)
    policy_default = ROOT / "configs/hiring_decision_standard_v1.json"
    if not policy_default.exists():
        policy_default = ROOT / "outputs/hiring_standard_v1/policy.json"
    parser.add_argument("--policy", type=Path, default=policy_default)
    parser.add_argument("--sources", type=Path, default=ROOT / "work/teacher75_v1_full/source_cases.json")
    parser.add_argument("--rubrics", type=Path, default=ROOT / "outputs/teacher75_v1_full/frozen_job_rubrics.json")
    parser.add_argument("--reviews", type=Path, default=ROOT / "outputs/teacher75_v1_full/reviews")
    parser.add_argument("--old-rows", type=Path, default=ROOT / "outputs/teacher75_audit/audit_rows_75.json")
    parser.add_argument("--pair-metadata", type=Path, default=ROOT / "outputs/teacher75_audit/pair_evidence.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/teacher75_v1_full")
    parser.add_argument("--standard-commit", default="323d4f5")
    parser.add_argument("--standard-script", type=Path, default=STANDARD_PATH,
                        help="Validator implementation; prefer same-directory hiring_standard.py, then local output copy")
    args = parser.parse_args(argv)
    standard = load_standard(args.standard_script)
    reviews = [{"reviewer": p.stem, "file": str(p.relative_to(args.output)), "cases": load_json(p)}
               for p in sorted(args.reviews.glob("*.json"))]
    data = analyze(load_json(args.policy), load_json(args.sources), load_json(args.rubrics), reviews,
                   load_json(args.old_rows), load_json(args.pair_metadata), standard_commit=args.standard_commit)
    args.output.mkdir(parents=True, exist_ok=True)
    # Validation precedes every write, so incomplete or invalid reviews cannot
    # silently replace a completed result with partial classifications.
    for name, value in data.items():
        (args.output / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(report(data), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "categories": data["summary"]["categories"],
                      "targets": data["summary"]["teacher_targets"],
                      "quotes": data["summary"]["source_quote_validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
