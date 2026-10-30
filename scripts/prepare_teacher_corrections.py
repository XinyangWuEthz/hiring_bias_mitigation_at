"""Prepare versioned, label-only correction candidates without model calls.

Recompute every result from the frozen JD/CV sources and both evidence ledgers.
Saved classifications are cross-checked, never used as target authorization.
This command cannot train a model and does not replace historical audit files.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path


WORKFLOW_ID = "hiring-evidence-v1-application-1"
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "teacher_correction_analysis", HERE / "teacher75_standard_analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)

CONFIG_KEYS = {"workflow_id", "review_prompt_id", "policy", "source_audit_dir",
               "expected_counts", "provenance"}
COMPARISON_KEYS = (
    "row_id", "case_id", "pair_id", "reference_decision", "policy_id",
    "policy_sha256", "audit_category", "audit_decision", "reviewer_decisions",
    "teacher_correction_supported", "teacher_correction_label", "audit_counts",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json(path, fingerprints):
    content = path.read_bytes()
    fingerprints[str(path.resolve())] = sha256(content)
    return json.loads(content)


def path_from_config(parent, value, name):
    require(isinstance(value, str) and bool(value.strip()), f"{name} must be a nonempty path")
    path = Path(value)
    return (parent / path).resolve() if not path.is_absolute() else path.resolve()


def compare_saved(saved, computed, name):
    old = analysis.unique_by(saved, "row_id", name)
    new = analysis.unique_by(computed, "row_id", "recomputed " + name)
    require(set(old) == set(new), f"{name}: stale or missing row IDs")
    for row_id in sorted(new):
        for key in COMPARISON_KEYS:
            require(key in old[row_id] and old[row_id][key] == new[row_id][key],
                    f"{name}: stale {key} for row {row_id}")


def candidate_status(row):
    if row["teacher_correction_supported"]:
        return "label_only_candidate"
    if row["audit_category"] == "anchor_problem":
        return "excluded_anchor_problem"
    return "excluded_pending_review"


def prepare(config_path, output_path):
    """Validate all inputs before writing a new, separate preparation bundle."""
    config_path, output_path = Path(config_path).resolve(), Path(output_path).absolute()
    require(not output_path.exists() and not output_path.is_symlink(),
            "Output must be a fresh directory; existing artifacts are never overwritten")
    output_path = output_path.resolve()
    fingerprints = {}
    config = read_json(config_path, fingerprints)
    require(isinstance(config, dict) and set(config) == CONFIG_KEYS,
            "Config must contain exactly workflow_id, review_prompt_id, policy, source_audit_dir, expected_counts, provenance")
    standard = analysis.standard
    require(config["workflow_id"] == WORKFLOW_ID, "Unsupported workflow_id")
    require(config["review_prompt_id"] == standard.REVIEW_PROMPT_ID, "Unexpected review_prompt_id")
    expected = config["expected_counts"]
    require(isinstance(expected, dict) and set(expected) == {"pairs", "rows"}
            and all(type(expected[k]) is int and expected[k] > 0 for k in expected),
            "expected_counts must contain positive integer pairs and rows")
    provenance = config["provenance"]
    require(isinstance(provenance, dict)
            and {"expected_policy_id", "expected_policy_sha256", "expected_source_file_sha256"} <= set(provenance)
            and set(provenance) <= {"expected_policy_id", "expected_policy_sha256", "standard_commit",
                                   "expected_source_file_sha256"},
            "provenance must declare expected_policy_id, expected_policy_sha256 and expected_source_file_sha256")
    policy_path = path_from_config(config_path.parent, config["policy"], "policy")
    source_dir = path_from_config(config_path.parent, config["source_audit_dir"], "source_audit_dir")
    require(source_dir.is_dir(), "source_audit_dir must be an existing directory")
    require(source_dir != output_path and source_dir not in output_path.parents,
            "Output must be outside the historical source audit directory")
    policy = read_json(policy_path, fingerprints)
    policy_hash = standard.canonical_policy_sha256(policy)
    require(policy["policy_id"] == provenance["expected_policy_id"], "Unexpected policy_id")
    require(policy_hash == provenance["expected_policy_sha256"], "Unexpected policy_sha256")
    source_names = ("source_cases.json", "frozen_job_rubrics.json", "legacy_audit_rows_75.json",
                    "pair_metadata.json", "audit_rows_75.json", "teacher_correction_candidates.json")
    review_paths = sorted((source_dir / "reviews").glob("*.json"))
    require(bool(review_paths), "No review ledgers found")
    consumed_paths = set(source_names) | {str(path.relative_to(source_dir)) for path in review_paths} | {"audit_prompt.txt"}
    pinned_hashes = provenance["expected_source_file_sha256"]
    require(isinstance(pinned_hashes, dict) and set(pinned_hashes) == consumed_paths,
            "expected_source_file_sha256 keys must match all consumed historical files exactly")
    pinned_content = {}
    for name in sorted(consumed_paths):
        expected_hash = pinned_hashes[name]
        require(isinstance(expected_hash, str) and len(expected_hash) == 64
                and all(char in "0123456789abcdef" for char in expected_hash),
                f"Malformed expected source SHA256 for {name}")
        content = (source_dir / name).read_bytes()
        actual_hash = sha256(content)
        require(actual_hash == expected_hash, f"Pinned historical source SHA256 mismatch: {name}")
        pinned_content[name] = content
        fingerprints[str((source_dir / name).resolve())] = actual_hash
    # Parse the exact bytes whose hashes were checked, avoiding a second read
    # that could assess a different source snapshot after validation.
    inputs = {name: json.loads(pinned_content[name]) for name in source_names}
    reviews = [{"reviewer": path.stem, "file": str(path.relative_to(source_dir)),
                "cases": json.loads(pinned_content[str(path.relative_to(source_dir))])} for path in review_paths]
    data = analysis.analyze(
        policy, inputs["source_cases.json"], inputs["frozen_job_rubrics.json"], reviews,
        inputs["legacy_audit_rows_75.json"], inputs["pair_metadata.json"],
        expected_pairs=expected["pairs"], expected_rows=expected["rows"],
        standard_commit=provenance.get("standard_commit", "not_recorded"),
    )
    compare_saved(inputs["audit_rows_75.json"], data["audit_rows_75"], "audit_rows_75.json")
    compare_saved(inputs["teacher_correction_candidates.json"],
                  data["teacher_correction_candidates"], "teacher_correction_candidates.json")

    sources = analysis.unique_by(inputs["source_cases.json"], "case_id", "sources")
    rubrics = analysis.unique_by(inputs["frozen_job_rubrics.json"], "case_id", "rubrics")
    requests = []
    for case_id in sorted(sources):
        source = sources[case_id]
        prompt = standard.build_review_prompt(
            policy, source["job_description"], source["candidate_profile"], rubrics[case_id])
        requests.append({"case_id": case_id, "policy_id": policy["policy_id"],
                         "policy_sha256": policy_hash, "prompt_id": standard.REVIEW_PROMPT_ID,
                         "prompt_sha256": sha256(prompt.encode("utf-8")), "prompt": prompt,
                         "request_executed": False})

    sample_provenance = {
        "source": "existing_paired_probe_legacy_prompt",
        "legacy_review_prompt_sha256": pinned_hashes["audit_prompt.txt"],
        "legacy_review_prompt_scope": "Actual prior reviewer template; not the SFT/DAPO sampling prompt.",
        "resampled_under_current_prompt": False,
        "existing_reviews_rerun_with_current_prompt": False,
        "scope": "Historical samples support direction only; they are not a new-policy model baseline.",
    }
    rows = copy.deepcopy(data["audit_rows_75"])
    eligibility = []
    for row in rows:
        row.update({"workflow_id": WORKFLOW_ID, "correction_status": candidate_status(row),
                    "training_ready": False, "teacher_rationale_approved_for_reuse": False,
                    "rationale_reuse_unchecked": True,
                    "sample_provenance": copy.deepcopy(sample_provenance)})
        eligibility.append({key: copy.deepcopy(row[key]) for key in (
            "row_id", "case_id", "pair_id", "policy_id", "policy_sha256", "audit_category",
            "audit_decision", "teacher_correction_supported", "teacher_correction_label",
            "correction_status", "training_ready", "teacher_rationale_approved_for_reuse",
            "rationale_reuse_unchecked", "review_queue_reason", "sample_provenance")})
    targets = [row for row in rows if row["teacher_correction_supported"]]
    evidence = {
        "source_cases.json": inputs["source_cases.json"],
        "frozen_job_rubrics.json": inputs["frozen_job_rubrics.json"],
    }
    artifacts = {"audit_rows_75.json": rows, "teacher_correction_candidates.json": targets,
                 "eligibility_decisions.json": eligibility,
                 "anchor_review_queue.json": [case for case in data["audit_cases_40"]
                                              if case["audit_category"] == "anchor_problem"],
                 "review_queue.json": data["review_queue"],
                 "audit_cases_40.json": data["audit_cases_40"],
                 "summary.json": data["summary"], "review_requests.json": requests,
                 "policy.json": policy, **evidence}
    for review in reviews:
        artifacts[review["file"]] = review["cases"]
    encoded = {name: json_bytes(value) for name, value in artifacts.items()}
    encoded["historical_review_prompt.txt"] = pinned_content["audit_prompt.txt"]
    report_note = (
        "本包按统一应用入口复验既有审阅账本，并重算分类和候选资格。"
        "下方是既有文本复审结果的重算报告；本次没有重新调用审阅模型。\n\n"
        "严格 prompt 的请求已准备，但尚未执行。原 SFT/DAPO 采样来自历史 prompt，"
        "没有按新 prompt 重采样。候选仅支持标签方向，training_ready=false，"
        "原教师理由未批准复用。\n\n"
    )
    encoded["report.md"] = (report_note + analysis.report(data)).encode("utf-8")
    manifest = {
        "workflow_id": WORKFLOW_ID, "policy_id": policy["policy_id"], "policy_sha256": policy_hash,
        "review_prompt_id": standard.REVIEW_PROMPT_ID, "source_audit_dir": str(source_dir),
        "legacy_review_prompt_sha256": pinned_hashes["audit_prompt.txt"],
        "expected_source_file_sha256": pinned_hashes,
        "config_path": str(config_path), "source_file_sha256": fingerprints,
        "script_sha256": {path.name: sha256(path.read_bytes()) for path in
                          (Path(__file__).resolve(), HERE / "teacher75_standard_analysis.py",
                           HERE / "hiring_standard.py")},
        "artifact_sha256": {name: sha256(content) for name, content in encoded.items()},
        "expected_counts": expected, "observed_counts": {
            "pairs": len(data["audit_cases_40"]), "rows": len(rows),
            "evidence_ledgers": data["summary"]["review_ledgers"],
            "label_only_candidates": len(targets), "review_requests_prepared": len(requests)},
        "categories": data["summary"]["categories"], "sample_provenance": sample_provenance,
        "training_started": False, "training_ready": False, "model_calls_made": 0,
        "new_sampling_started": False, "historical_artifacts_modified": False,
        "eligibility_method": "Revalidate both source ledgers and frozen rubric, recompute classifications, cross-check saved outputs.",
        "limitations": data["summary"]["limitations"],
    }
    encoded["manifest.json"] = json_bytes(manifest)
    output_path.mkdir(parents=True, exist_ok=False)
    for name, content in encoded.items():
        path = output_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare(args.config, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "counts": result["observed_counts"],
                      "training_started": False, "model_calls_made": 0}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
