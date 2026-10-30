"""The round-2 preference build: pool merging and balancing by error mass. No GPU: the student
scores and judge verdicts are supplied as frames, and the margin reader runs on a stub model.
"""

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from hiring_bias_mitigation.generation import balanced_dpo as BD
from hiring_bias_mitigation.generation import dataset as D

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _row(i, decision, margin, source="round1", split="train", candidate=None):
    """One scored pair. `margin` is the student's logit(hire) - logit(reject)."""
    return {
        "prompt": f"prompt {source} {i}", "chosen": "c", "rejected": "r",
        "decision": decision, "student_margin": margin, "source": source, "split": split,
        "candidate_id": candidate or f"cand-{source}-{i}", "job_id": f"job-{i}",
    }


def _pool(hard_reject_margins, n_easy_reject, n_hard_hire, n_easy_hire):
    """Reject side: graded confidently-wrong pairs plus easy ones; hire side likewise."""
    rows = [_row(i, "reject", m) for i, m in enumerate(hard_reject_margins)]
    rows += [_row(100 + i, "reject", -4.0) for i in range(n_easy_reject)]
    rows += [_row(200 + i, "hire", -4.0) for i in range(n_hard_hire)]
    rows += [_row(300 + i, "hire", 4.0) for i in range(n_easy_hire)]
    return pd.DataFrame(rows)


# ---- balancing by error mass ---------------------------------------------------------------

def test_balance_trims_the_heavier_sides_most_confidently_wrong_pairs_first():
    """Equal counts are not equal pull: confidently-wrong pairs keep pulling after the rest
    have gone quiet, so a reject side heavy with them drags the whole model toward reject."""
    hard = list(np.linspace(6.0, 2.0, 20))       # the student says hire; reference says reject
    pool = _pool(hard, n_easy_reject=80, n_hard_hire=5, n_easy_hire=95)

    selected, report = BD.balance_error_mass(BD.with_student_error(pool), tolerance=0.05)
    after = report["after"]

    assert after["hire"]["pairs"] == after["reject"]["pairs"]
    assert report["heavier_side"] == "reject"
    assert report["error_mass_ratio"] <= 1.05
    # The trimmed pairs are exactly the reject side's most confident errors, in order.
    trimmed = report["trimmed_most_confidently_wrong"]["pairs"]
    kept_hard = selected[(selected["decision"] == "reject") & (selected["student_margin"] > 0)]
    assert len(kept_hard) == len(hard) - trimmed
    assert kept_hard["student_margin"].max() == pytest.approx(sorted(hard)[len(hard) - trimmed - 1])
    # Every hard hire-side pair stays: the lighter side keeps its hardest pairs.
    assert ((selected["decision"] == "hire") & (selected["student_margin"] < 0)).sum() == 5


def test_harder_pairs_on_the_lighter_side_reduce_the_trimming():
    """The hire-resampling surplus raises the hire side's mass before anything is cut."""
    hard = list(np.linspace(6.0, 2.0, 20))
    base = _pool(hard, n_easy_reject=80, n_hard_hire=5, n_easy_hire=95)
    surplus = pd.DataFrame([_row(400 + i, "hire", -3.0, source="extra:hire_surplus")
                            for i in range(10)])

    _, alone = BD.balance_error_mass(BD.with_student_error(base))
    selected, with_surplus = BD.balance_error_mass(
        BD.with_student_error(pd.concat([base, surplus], ignore_index=True)))

    assert (with_surplus["trimmed_most_confidently_wrong"]["pairs"]
            < alone["trimmed_most_confidently_wrong"]["pairs"])
    assert (selected["source"] == "extra:hire_surplus").sum() == 10


def test_balance_is_symmetric_when_the_hire_side_is_heavier():
    pool = _pool([-0.5] * 5, n_easy_reject=95, n_hard_hire=20, n_easy_hire=80)
    pool.loc[pool["decision"] == "reject", "student_margin"] = np.r_[[3.0] * 5, [-4.0] * 95]
    _, report = BD.balance_error_mass(BD.with_student_error(pool))
    assert report["heavier_side"] == "hire"
    assert report["after"]["hire"]["pairs"] == report["after"]["reject"]["pairs"]
    assert report["error_mass_ratio"] <= 1.05


def test_balance_refuses_rather_than_gut_the_set():
    """With no hard pairs on the lighter side, matching its mass would leave almost nothing;
    the remedy is more hard pairs, not training on the remainder."""
    pool = _pool(list(np.linspace(6.0, 2.0, 60)), n_easy_reject=40, n_hard_hire=0,
                 n_easy_hire=100)
    with pytest.raises(ValueError, match="Add harder pairs"):
        BD.balance_error_mass(BD.with_student_error(pool), min_keep=0.9)


def test_counts_are_matched_by_dropping_the_easiest_pairs():
    """Easy pairs carry almost no error mass, so cutting them for the count keeps the
    mass balance intact."""
    pool = _pool([], n_easy_reject=50, n_hard_hire=10, n_easy_hire=200)
    pool = pd.concat([pool, pd.DataFrame([_row(500 + i, "reject", 3.0) for i in range(10)])],
                     ignore_index=True)
    selected, report = BD.balance_error_mass(BD.with_student_error(pool))
    assert report["after"]["hire"]["pairs"] == 60
    assert report["dropped_easiest_for_count"]["hire"] == 150
    kept_hire = selected[selected["decision"] == "hire"]
    assert (kept_hire["student_margin"] < 0).sum() == 10   # every hard hire pair survives


# ---- merging ------------------------------------------------------------------------------

def test_a_lower_priority_pool_cannot_contradict_round1s_decision():
    """Two decisions for one CV-job pair would train the attribute dependence away from and
    back into the model at the same time."""
    df = pd.DataFrame([
        _row(1, "hire", 0.0, source="round1", candidate="c1"),
        _row(2, "reject", 0.0, source="legacy", candidate="c1"),
        _row(3, "hire", 0.0, source="legacy", candidate="c1"),
    ])
    df["job_id"] = "j1"
    kept, report = BD.drop_conflicting_anchors(df)
    assert list(kept["source"]) == ["round1", "legacy"]
    assert set(kept["decision"]) == {"hire"}
    assert report["rows_dropped_against_higher_priority_anchor"] == 1


def test_a_pool_that_disagrees_with_itself_loses_the_pair():
    df = pd.DataFrame([_row(1, "hire", 0.0, source="legacy", candidate="c2"),
                       _row(2, "reject", 0.0, source="legacy", candidate="c2")])
    df["job_id"] = "j2"
    kept, report = BD.drop_conflicting_anchors(df)
    assert kept.empty
    assert report["pairs_dropped_disagreeing_with_themselves"] == 1


def test_a_candidate_held_out_by_any_pool_never_trains():
    df = pd.DataFrame([
        _row(1, "hire", 0.0, source="round1", split="validation", candidate="c9"),
        _row(2, "hire", 0.0, source="legacy", split="train", candidate="c9"),
        _row(3, "reject", 0.0, source="legacy", split="train", candidate="c8"),
    ])
    train, validation, report = BD.split_by_candidate(df)
    assert list(train["candidate_id"]) == ["c8"]
    assert list(validation["candidate_id"]) == ["c9"]
    assert report["train_rows_dropped_candidate_held_out"] == 1


def test_duplicate_prompts_keep_the_round1_pair():
    df = pd.DataFrame([_row(1, "hire", 0.0, source="legacy"), _row(1, "hire", 0.0)])
    df["prompt"] = "same prompt"
    kept, report = BD.drop_duplicate_prompts(df)
    assert list(kept["source"]) == ["round1"]
    assert report["dropped_by_source"] == {"legacy": 1}


# ---- reading the pools -------------------------------------------------------------------

def _variant(condition, attr="War veteran", group="military_status", decision="hire",
             candidate="cand-x", job="job-x", lang="en"):
    from hiring_bias_mitigation.data.protected_groups import GROUP_LABELS

    row = {"lang": lang, "cv": f"Eight years of Python and Django ({candidate}).",
           "job_description": "Senior Python engineer.", "condition": condition,
           "group_label": GROUP_LABELS[group][lang], "protected_group": group,
           "protected_attr": attr}
    row["prompt"] = D._audit_prompt(D._with_implicit(pd.DataFrame([row])).iloc[0].to_dict())
    other = "reject" if decision == "hire" else "hire"
    row["chosen"] = D._completion(decision, "Django and PostgreSQL match the stack.", lang)
    row["rejected"] = D._completion(other, "Too little production experience for this role.",
                                    lang)
    row.update(candidate_id=candidate, job_id=job)
    return row


@pytest.mark.parametrize("condition", ["explicit", "implicit"])
def test_variant_is_read_off_the_prompt(condition):
    row = _variant(condition, attr="Female", group="gender")
    assert BD.infer_variant(row["prompt"], "en") == ("gender", "Female", condition)


def test_male_is_not_found_inside_female():
    explicit = _variant("explicit", attr="Female", group="gender")["prompt"]
    assert BD.infer_variant(explicit, "en")[1] == "Female"
    assert BD.infer_variant("no attribute anywhere", "en") is None


def test_normalising_reads_decisions_and_drops_negatives_that_do_not_flip():
    rows = [_variant("explicit"), _variant("implicit", candidate="cand-y")]
    rows[1]["rejected"] = rows[1]["chosen"]          # not a negative at all
    frame = pd.DataFrame(rows)[["prompt", "chosen", "rejected", "candidate_id", "job_id"]]
    out, report = BD.normalise_pairs(frame, "round1", "train", "en")
    assert len(out) == 1
    assert report["dropped_same_decision_both_sides"] == 1
    assert report["variant_read_from_prompt"] == 1
    first = out.iloc[0]
    assert (first["decision"], first["rejected_decision"]) == ("hire", "reject")
    assert (first["protected_group"], first["condition"]) == ("military_status", "explicit")
    assert first["pair_id"] == "cand-x__job-x"


def test_missing_ids_are_refused_not_guessed():
    frame = pd.DataFrame([_variant("explicit")])[["prompt", "chosen", "rejected"]]
    with pytest.raises(ValueError, match="candidate_id"):
        BD.normalise_pairs(frame, "round1", "train", "en")


def test_legacy_pairs_keep_only_negatives_that_do_not_name_the_attribute():
    covert = _variant("explicit")
    overt = _variant("explicit", candidate="cand-y")
    overt["rejected"] = D._completion("reject", "We do not hire war veterans.", "en")
    unflagged_overt = dict(overt, candidate_id="cand-z")   # the stored flag missed it
    ukrainian = dict(covert, lang="uk", candidate_id="cand-w")
    frame = pd.DataFrame([covert, overt, unflagged_overt, ukrainian])
    frame["rejected_mentions_attribute"] = [False, True, False, False]
    kept, report = BD.filter_legacy(frame, "en")
    assert list(kept["candidate_id"]) == ["cand-x"]
    assert report["dropped_rejected_names_attribute"] == 1
    assert report["dropped_detector_recheck"] == 1


@pytest.mark.parametrize("raw, verdict", [
    ('{"refers": false}', "clean"),
    ('```json\n{"refers": true}\n```', "refers"),
    ('{"refers": "false"}', "clean"),
    ("refers: true", "refers"),
    ("I cannot tell.", "unreadable"),
])
def test_judge_verdicts_have_three_states(raw, verdict):
    assert BD.parse_judge(raw) == verdict


def test_judge_prompt_names_the_characteristic_in_english():
    text = BD.judge_prompt("Strong Go skills.", "military_status", "Ветеран війни")
    assert "military status: Ветеран війни" in text
    assert "veteran or reservist" in text


# ---- the whole build -----------------------------------------------------------------------

def _candidates():
    rows = []
    for i in range(12):
        decision = "hire" if i % 2 else "reject"
        row = _variant("explicit" if i % 3 else "implicit", decision=decision,
                       candidate=f"cand-{i}", job=f"job-{i}")
        rows.append((row, "round1", "train"))
    for i in range(12, 16):
        rows.append((_variant("explicit", decision="hire", candidate=f"cand-{i}",
                              job=f"job-{i}"), "legacy", "train"))
    rows.append((_variant("explicit", decision="hire", candidate="cand-v", job="job-v"),
                 "round1", "validation"))
    frames = [BD.normalise_pairs(pd.DataFrame([r]), s, sp, "en")[0] for r, s, sp in rows]
    out = BD.add_keys(pd.concat(frames, ignore_index=True))
    out["needs_judge"] = out["source"] == BD.LEGACY
    return out


def test_assemble_judges_scores_and_balances():
    candidates = _candidates()
    legacy = candidates[candidates["needs_judge"]]
    judge = pd.DataFrame({"pair_sha1": legacy["pair_sha1"],
                          "verdict_chosen": ["clean"] * len(legacy),
                          "verdict_rejected": ["clean", "refers", "unreadable", "clean"]})
    scores = pd.DataFrame({"prompt_sha1": candidates["prompt_sha1"],
                           "student_margin": np.linspace(-3, 3, len(candidates))})

    train, validation, report = BD.assemble(candidates, scores, judge)

    assert report["judge"]["dropped_refers_or_unreadable"] == 2
    assert (train["decision"] == "hire").sum() == (train["decision"] == "reject").sum()
    assert list(validation["candidate_id"]) == ["cand-v"]
    assert {"prompt", "chosen", "rejected", "lang", "protected_group", "condition",
            "student_margin", "p_wrong"} <= set(train.columns)
    # The trainer's decision eval samples and scores validation rows by reference_decision.
    for part in (train, validation):
        assert (part["reference_decision"] == part["decision"]).all()


def test_assemble_refuses_legacy_pairs_without_a_verdict():
    candidates = _candidates()
    scores = pd.DataFrame({"prompt_sha1": candidates["prompt_sha1"], "student_margin": 0.0})
    with pytest.raises(ValueError, match="judge stage"):
        BD.assemble(candidates, scores, None)
    _, _, report = BD.assemble(candidates, scores, None, require_judge=False)
    assert report["judge"]["built_without_judge"] == 4


def test_assemble_refuses_unscored_training_pairs():
    candidates = _candidates()
    scores = pd.DataFrame({"prompt_sha1": candidates["prompt_sha1"].iloc[:3],
                           "student_margin": 0.0})
    with pytest.raises(ValueError, match="score stage"):
        BD.assemble(candidates, scores, None, require_judge=False)


def test_the_cli_runs_candidates_and_build_end_to_end(tmp_path):
    """The GPU stages are replaced by their cache files; the glue around them is real."""
    pools = _candidates()
    round1 = pools[pools["source"] == "round1"]
    round1[round1["split"] == "train"][list(BD.REQUIRED)].to_json(
        tmp_path / "r1_train.jsonl", orient="records", lines=True, force_ascii=False)
    round1[round1["split"] == "validation"][list(BD.REQUIRED)].to_parquet(
        tmp_path / "r1_val.parquet")
    env = {**__import__("os").environ, "HBM_OUTPUT_ROOT": str(tmp_path),
           "HBM_ALLOW_LOCAL_OUTPUT": "1"}
    script = ["python", "scripts/build_balanced_dpo.py"]
    script[0] = sys.executable

    subprocess.run([*script, "candidates", "--artifacts-dir", "art", "--no-legacy",
                    "--round1-train", str(tmp_path / "r1_train.jsonl"),
                    "--round1-validation", str(tmp_path / "r1_val.parquet")],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    candidates = pd.read_parquet(tmp_path / "art" / "dpo_round2_candidates.parquet")
    assert len(candidates) == 13
    pd.DataFrame({"prompt_sha1": candidates["prompt_sha1"],
                  "student_margin": np.linspace(-2, 2, len(candidates)),
                  "student": "stub"}).to_parquet(tmp_path / "art" / "dpo_round2_scores.parquet")

    subprocess.run([*script, "build", "--artifacts-dir", "art"], cwd=ROOT, env=env,
                   check=True, capture_output=True, text=True)
    train = pd.read_parquet(tmp_path / "art" / "dpo_round2_train.parquet")
    report = json.loads((tmp_path / "art" / "dpo_round2_report.json").read_text())
    assert (train["decision"] == "hire").sum() == (train["decision"] == "reject").sum()
    assert report["parameters"]["students"] == ["stub"]


# ---- reading the student's decision margin -------------------------------------------------

class _Encoding(dict):
    def to(self, device):
        return self


class _CharTokenizer:
    """One token per character; no chat template, so prompts pass through unwrapped."""

    chat_template = None
    padding_side = "right"

    def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=False):
        import torch

        if isinstance(texts, str):
            return {"input_ids": [ord(c) % 256 for c in texts]}
        ids = [[ord(c) % 256 for c in t] for t in texts]
        width = max(len(i) for i in ids)
        assert self.padding_side == "left", "the decision position must be last in every row"
        pad = [[0] * (width - len(i)) + i for i in ids]
        mask = [[0] * (width - len(i)) + [1] * len(i) for i in ids]
        return _Encoding(input_ids=torch.tensor(pad), attention_mask=torch.tensor(mask))


class _LengthModel:
    """logit('h') = number of real tokens, logit('r') = 0: the margin is the row's length."""

    device = "cpu"
    training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, input_ids, attention_mask, logits_to_keep=None):
        import torch

        logits = torch.zeros(input_ids.shape[0], 1, 256)
        logits[:, -1, ord("h")] = attention_mask.sum(dim=1).float()
        return type("Out", (), {"logits": logits})()


def test_margins_come_back_in_the_order_the_prompts_were_given():
    """Prompts run longest first; reading them back unsorted would score the wrong pairs."""
    from hiring_bias_mitigation.eval.margins import PREFIX, decision_margins

    prompts = ["short", "a much longer prompt than the others", "mid length one"]
    model, tokenizer = _LengthModel(), _CharTokenizer()
    margins = decision_margins(model, tokenizer, prompts, batch_size=2)

    assert list(margins) == [len(p + PREFIX) for p in prompts]
    assert tokenizer.padding_side == "right", "the tokenizer is shared with the trainer"
    assert model.training, "a model handed over in train mode goes back in train mode"


def test_the_script_documents_every_stage():
    out = subprocess.run([sys.executable, "scripts/build_balanced_dpo.py", "--help"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    for stage in ("candidates", "judge", "score", "build"):
        assert stage in out


# ---- the GPU stages' glue, with the model replaced -------------------------------------------

def _script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_balanced_dpo", ROOT / "scripts" / "build_balanced_dpo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_judge_verdicts_line_up_with_their_own_pair_and_side(tmp_path, monkeypatch):
    """Two rationales per pair go out in one flat batch; a verdict landing on the wrong pair
    or side would keep a leaking negative and drop a clean one."""
    import argparse

    from hiring_bias_mitigation.eval import backends as BK

    S = _script()
    candidates = _candidates()
    leak = candidates.index[candidates["needs_judge"]][1]
    candidates.loc[leak, "rejected"] = D._completion("reject", "Your service history worries us.",
                                                     "en")
    candidates = BD.add_keys(candidates.drop(columns=["prompt_sha1", "pair_sha1"]))
    paths = S.stage_paths(tmp_path, "dpo_round2")
    candidates.to_parquet(paths["candidates"])

    class Judge:
        def generate(self, prompts):
            return ['{"refers": true}' if "service history" in p else '{"refers": false}'
                    for p in prompts]

        def close(self):
            pass

    monkeypatch.setattr(BK, "build_backend", lambda cfg: Judge())
    args = argparse.Namespace(all_sources=False, judge_model="stub", force=False, seed=42,
                              gpu_memory_utilization=0.5, max_model_len=4096, max_num_seqs=None,
                              tensor_parallel_size=1, gdn_prefill_backend=None)
    S.cmd_judge(args, paths)

    verdicts = pd.read_parquet(paths["judge"]).set_index("pair_sha1")
    assert len(verdicts) == 4, "only the legacy pairs are judged by default"
    leaking = verdicts.loc[candidates.at[leak, "pair_sha1"]]
    assert (leaking["verdict_chosen"], leaking["verdict_rejected"]) == ("clean", "refers")
    assert (verdicts["verdict_rejected"] == "refers").sum() == 1
    assert paths["judge_sample"].exists()


def test_scores_are_cached_by_prompt_and_tied_to_one_student(tmp_path, monkeypatch):
    import argparse

    from hiring_bias_mitigation.eval import margins as M
    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    S = _script()
    candidates = _candidates()
    paths = S.stage_paths(tmp_path, "dpo_round2")
    candidates.to_parquet(paths["candidates"])

    calls = []

    def margins(model, tokenizer, prompts, lang="en", batch_size=4):
        calls.append(list(prompts))
        return np.arange(len(prompts), dtype=float)

    monkeypatch.setattr(C, "load_model_and_tokenizer", lambda cfg, for_training=True: (None, None))
    monkeypatch.setattr(M, "decision_margins", margins)
    args = argparse.Namespace(student="stub-sft", dtype="bfloat16", lang="en", batch_size=4,
                              force=False)

    S.cmd_score(args, paths)
    scores = pd.read_parquet(paths["scores"])
    assert len(scores) == candidates["prompt_sha1"].nunique()
    assert set(scores["student"]) == {"stub-sft"}

    S.cmd_score(args, paths)
    assert len(calls) == 1, "a second run must reuse the cache, not reload the model"

    with pytest.raises(SystemExit, match="--force"):
        S.cmd_score(argparse.Namespace(**{**vars(args), "student": "another-model"}), paths)


def test_a_pool_in_another_prompt_format_is_refused_not_emptied():
    frame = pd.DataFrame([_variant("explicit")])[["prompt", "chosen", "rejected",
                                                  "candidate_id", "job_id"]]
    frame["prompt"] = "A different template that never states the attribute."
    with pytest.raises(ValueError, match="protected_group, protected_attr and condition"):
        BD.normalise_pairs(frame, "round1", "train", "en")


def test_a_repeated_verdict_does_not_duplicate_the_pair():
    candidates = _candidates()
    legacy = candidates[candidates["needs_judge"]]
    once = pd.DataFrame({"pair_sha1": legacy["pair_sha1"], "verdict_chosen": "clean",
                         "verdict_rejected": "clean"})
    scores = pd.DataFrame({"prompt_sha1": candidates["prompt_sha1"],
                           "student_margin": np.linspace(-3, 3, len(candidates))})
    single, _, _ = BD.assemble(candidates, scores, once)
    double, _, _ = BD.assemble(candidates, scores, pd.concat([once, once]))
    assert len(double) == len(single)


# ---- V3: each pool source on its own -------------------------------------------------------

def _sourced(source, i, decision, split="train", group_attr=("military_status", "War veteran")):
    group, attr = group_attr
    row = _variant("explicit" if i % 3 else "implicit", attr=attr, group=group,
                   decision=decision, candidate=f"{source}-{split}-{i}",
                   job=f"job-{source}-{split}-{i}")
    return {**row, "pool_source": source}


def test_normalising_keeps_the_pool_source_v3_balances_by():
    row = _sourced("rjdb", 0, "hire")
    kept, _ = BD.normalise_pairs(pd.DataFrame([row]), BD.ROUND1, "train", "en")
    assert kept["pool_source"].tolist() == ["rjdb"]
    plain, _ = BD.normalise_pairs(pd.DataFrame([_variant("explicit")]), BD.ROUND1, "train", "en")
    assert "pool_source" not in plain


def test_each_pool_source_is_balanced_on_its_own():
    """V3's sources hire at 22%, 56% and 15%; a pooled balance would let one source supply a
    side for another (docs/V3_NEW_DATA.md, round 2). A source without one side stays out."""
    djinni = _pool(list(np.linspace(6.0, 2.0, 10)), n_easy_reject=40, n_hard_hire=4,
                   n_easy_hire=46).assign(pool_source="djinni_new")
    rjdb = _pool([], n_easy_reject=10, n_hard_hire=10, n_easy_hire=30).assign(pool_source="rjdb")
    kaggle = pd.DataFrame([_row(500 + i, "reject", -4.0) for i in range(5)]).assign(
        pool_source="kaggle_linkedin")
    pool = BD.with_student_error(pd.concat([djinni, rjdb, kaggle], ignore_index=True))

    selected, report = BD.balance_error_mass_by(pool, "pool_source")

    assert report["left_out"] == ["kaggle_linkedin"]
    assert "no hire-side pairs" in report["groups"]["kaggle_linkedin"]["left_out"]
    assert report["groups"]["rjdb"]["heavier_side"] == "hire"
    assert report["groups"]["djinni_new"]["heavier_side"] == "reject"
    for source, part in selected.groupby("pool_source"):
        assert (part["decision"] == "hire").sum() == (part["decision"] == "reject").sum()
        assert report["groups"][source]["error_mass_ratio"] <= 1.05
    # Exactly the pairs that balancing the source alone keeps.
    alone, _ = BD.balance_error_mass(pool[pool["pool_source"] == "djinni_new"])
    assert (sorted(selected.loc[selected["pool_source"] == "djinni_new", "prompt"])
            == sorted(alone["prompt"]))


def test_a_per_source_balance_needs_the_source_on_every_pair():
    pool = BD.with_student_error(_pool([3.0], 3, 1, 3).assign(pool_source=None))
    with pytest.raises(ValueError, match="no pool_source"):
        BD.balance_error_mass_by(pool, "pool_source")


def test_validation_is_balanced_by_count_within_each_source_and_group():
    rows = []
    for source, n_hire, n_reject in (("djinni_new", 2, 5), ("rjdb", 4, 1),
                                     ("kaggle_linkedin", 0, 3)):
        rows += [{**_row(f"{source}-h{i}", "hire", 0.0), "pool_source": source,
                  "protected_group": "gender"} for i in range(n_hire)]
        rows += [{**_row(f"{source}-r{i}", "reject", 0.0), "pool_source": source,
                  "protected_group": "gender"} for i in range(n_reject)]
    kept, report = BD.balance_count_by(pd.DataFrame(rows), ["pool_source", "protected_group"],
                                       seed=1)
    assert report["cells"]["djinni_new / gender"] == {"hire": 2, "reject": 2, "dropped": 3}
    assert report["cells"]["rjdb / gender"] == {"hire": 1, "reject": 1, "dropped": 3}
    assert report["cells"]["kaggle_linkedin / gender"] == {"hire": 0, "reject": 0, "dropped": 3}
    assert len(kept) == 6


def _sourced_candidates():
    rows = []
    for source in ("djinni_new", "rjdb"):
        rows += [(_sourced(source, i, "hire" if i % 2 else "reject"), "train") for i in range(8)]
        rows += [(_sourced(source, i, d, split="validation"), "validation")
                 for i, d in enumerate(("hire", "reject", "reject"))]
    rows += [(_sourced("kaggle_linkedin", i, "reject"), "train") for i in range(3)]
    rows += [(_sourced("kaggle_linkedin", 0, "reject", split="validation"), "validation")]
    frames = [BD.normalise_pairs(pd.DataFrame([r]), BD.ROUND1, split, "en")[0]
              for r, split in rows]
    return BD.add_keys(pd.concat(frames, ignore_index=True)).assign(needs_judge=False)


def test_assemble_balances_per_source_and_drops_a_left_out_sources_validation():
    candidates = _sourced_candidates()
    scores = pd.DataFrame({"prompt_sha1": candidates["prompt_sha1"], "student_margin": 0.0})

    train, validation, report = BD.assemble(candidates, scores, None, balance_by="pool_source")

    assert report["balance"]["left_out"] == ["kaggle_linkedin"]
    assert set(train["pool_source"]) == {"djinni_new", "rjdb"}
    for _, part in train.groupby("pool_source"):
        assert (part["decision"] == "hire").sum() == (part["decision"] == "reject").sum() == 4
    # Validation: one hire and one reject per source and group; kaggle_linkedin's goes with
    # its training pairs.
    assert sorted(validation.groupby(["pool_source", "decision"]).size().tolist()) == [1] * 4
    assert report["validation_balance"]["left_out_with_training"] == 1
    assert (validation["reference_decision"] == validation["decision"]).all()


def test_the_cli_balances_per_source(tmp_path):
    candidates = _sourced_candidates()
    columns = [*BD.REQUIRED, "pool_source"]
    for split in ("train", "validation"):
        candidates[candidates["split"] == split][columns].to_parquet(tmp_path / f"{split}.parquet")
    env = {**__import__("os").environ, "HBM_OUTPUT_ROOT": str(tmp_path),
           "HBM_ALLOW_LOCAL_OUTPUT": "1"}
    script = [sys.executable, "scripts/build_balanced_dpo.py"]
    common = ["--artifacts-dir", "art", "--name", "dpo_v3_onpolicy"]
    subprocess.run([*script, "candidates", *common, "--no-legacy",
                    "--round1-train", str(tmp_path / "train.parquet"),
                    "--round1-validation", str(tmp_path / "validation.parquet")],
                   cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    art = tmp_path / "art"
    built = pd.read_parquet(art / "dpo_v3_onpolicy_candidates.parquet")
    pd.DataFrame({"prompt_sha1": built["prompt_sha1"], "student_margin": 0.0,
                  "student": "stub"}).to_parquet(art / "dpo_v3_onpolicy_scores.parquet")

    run = subprocess.run([*script, "build", *common, "--balance-by", "pool_source"], cwd=ROOT,
                         env=env, check=True, capture_output=True, text=True)

    assert "LEFT OUT" in run.stdout
    report = json.loads((tmp_path / "art" / "dpo_v3_onpolicy_report.json").read_text())
    assert report["parameters"]["balance_by"] == "pool_source"
    assert report["balance"]["left_out"] == ["kaggle_linkedin"]
    train = pd.read_parquet(tmp_path / "art" / "dpo_v3_onpolicy_train.parquet")
    assert set(train["pool_source"]) == {"djinni_new", "rjdb"}
