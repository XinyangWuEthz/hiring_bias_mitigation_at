"""SFT-v3 stages 0 and 1 (scripts/sft_v3_anchors.py): the candidate prompts, the vote rule, the
pre-registered criteria and the gate's readouts. No GPU, no network."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "sft_v3_anchors", Path(__file__).resolve().parents[1] / "scripts" / "sft_v3_anchors.py")
V3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V3)

from hiring_bias_mitigation.data import prompts as P  # noqa: E402
from hiring_bias_mitigation.data.benchmark import load_benchmark  # noqa: E402
from hiring_bias_mitigation.data.injection import build_eval_set  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402


def test_the_reference_candidate_drops_only_the_decisive_sentence():
    pairs = pd.DataFrame([{"job_description": "Backend role, Django.",
                           "cv": "Five years of Django."}])
    original = SY.reference_prompts(pairs, "en")[0]
    edited = V3.candidate_prompts(pairs, "reference_no_decisive")[0]
    assert "Be decisive" in original and "Be decisive" not in edited
    assert edited == original.replace(V3.DECISIVE, "")
    assert V3.candidate_prompts(pairs, V3.CONTROL) == [original]   # the redraw control
    with pytest.raises(ValueError):
        V3.without_decisive("a prompt without the sentence")


def test_the_audit_candidate_is_the_benchmarks_attribute_free_prompt():
    pairs = load_benchmark("en").head(3)
    rows = build_eval_set(pairs, [], "en", conditions=("attr_free",))
    assert V3.candidate_prompts(pairs, "audit") == [P.build_prompt(r, "baseline")
                                                    for r in rows.to_dict("records")]


def _draw(pair, i, decision, feedback):
    raw = json.dumps({"decision": decision, "feedback": feedback})
    return {"pair_id": pair, "prompt": "audit", "sample_index": i, "raw_output": raw}


GOOD = "Five years of Django and PostgreSQL meet the role's core requirements well."
SHORT = "Good fit."


def test_the_vote_keeps_v1_on_ties_missing_votes_and_unusable_majorities():
    pairs = pd.DataFrame({"pair_id": ["vote", "tie", "none", "short", "same"],
                          "v1_decision": "reject", "v1_feedback": "v1 reasoning here, kept"})
    draws = ([_draw("vote", 0, "hire", SHORT), _draw("vote", 1, "hire", GOOD)]
             + [_draw("vote", i, "hire", GOOD) for i in range(2, 6)]
             + [_draw("vote", i, "reject", GOOD) for i in range(6, 8)]
             + [_draw("tie", i, "hire" if i < 4 else "reject", GOOD) for i in range(8)]
             + [{"pair_id": "none", "prompt": "audit", "sample_index": i,
                 "raw_output": "I cannot decide."} for i in range(8)]
             + [_draw("short", i, "hire", SHORT) for i in range(5)]
             + [_draw("short", i, "reject", GOOD) for i in range(5, 8)]
             + [_draw("same", i, "reject", GOOD) for i in range(8)])
    anchors = V3.new_anchors(V3.TA.parse(pd.DataFrame(draws), "en"), pairs).set_index("pair_id")
    assert anchors.loc["vote", "new_decision"] == "hire" and anchors.loc["vote", "changed"]
    assert anchors.loc["vote", "hire_votes"] == 6 and anchors.loc["vote", "decided"] == 8
    assert anchors.loc["vote", "new_feedback"] == GOOD          # the first draw that passes
    assert anchors.loc["vote", "vote_sample_index"] == 1
    for pair, source in [("tie", "tie"), ("none", "no_decided_draw"),
                         ("short", "no_usable_feedback")]:
        assert anchors.loc[pair, "source"] == source
        assert anchors.loc[pair, "new_decision"] == "reject" and not anchors.loc[pair, "changed"]
        assert anchors.loc[pair, "new_feedback"] == "v1 reasoning here, kept"
    assert anchors.loc["same", "source"] == "vote" and not anchors.loc["same", "changed"]


def _labels():
    rows = ([("target", 4)] * 10 + [("target", 2)] * 5 + [("control_reject", 1)] * 5
            + [("target", 3)] * 3 + [("reverse", 4)] * 2)
    return pd.DataFrame([{"pair_id": f"p{i}", "set": s, "fit": f,
                          "fit_bucket": "4-5" if f >= 4 else ("3" if f == 3 else "1-2")}
                         for i, (s, f) in enumerate(rows)])


def test_criteria_and_the_choice_follow_the_preregistration():
    labels = _labels()
    hire = {f"p{i}" for i in range(6)} | {"p10"}          # 6 of 10 clear hires, 1 of 10 rejects
    anchors = pd.DataFrame({"pair_id": labels["pair_id"], "source": "vote",
                            "new_decision": ["hire" if p in hire else "reject"
                                             for p in labels["pair_id"]]})
    r = V3.criteria(anchors, labels)
    assert (r["c1"]["pairs"], r["c1"]["hired"], r["c1"]["pass"]) == (10, 6, True)
    assert (r["c2"]["pairs"], r["c2"]["rejected"], r["c2"]["pass"]) == (10, 9, True)
    assert r["passes"] and abs(r["score"] - 0.5) < 1e-9
    one_short = anchors.assign(new_decision=lambda d: d["new_decision"].mask(d["pair_id"] == "p5",
                                                                             "reject"))
    assert not V3.criteria(one_short, labels)["c1"]["pass"]   # 5 of 10 is below 0.60

    ok = {"passes": True}
    assert V3.choose({"audit": {**ok, "score": 0.51},
                      "reference_no_decisive": {**ok, "score": 0.50}}) == V3.SMALLER_EDIT
    assert V3.choose({"audit": {**ok, "score": 0.60},
                      "reference_no_decisive": {**ok, "score": 0.50}}) == "audit"
    assert V3.choose({"audit": {**ok, "score": 0.10},
                      "reference_no_decisive": {"passes": False, "score": 0.90}}) == "audit"
    assert V3.choose({name: {"passes": False, "score": 1.0} for name in V3.CANDIDATES}) is None


def test_labels_must_reproduce_the_reviews_counts():
    with pytest.raises(ValueError):
        V3.check_labels(_labels())


def test_changed_pairs_keep_v1s_variants_pinned_to_the_new_anchor():
    invariant = pd.DataFrame([{"pair_id": p, "protected_group": "religion", "protected_attr": a,
                               "condition": "explicit", "reference_decision": "reject",
                               "reference_feedback": "old", "raw_output": "{}"}
                              for p in ("a", "b") for a in ("Jain", "Atheist")])
    anchors = pd.DataFrame({"pair_id": ["a", "b"], "changed": [True, False],
                            "new_decision": ["hire", "reject"], "new_feedback": ["new", "old"]})
    v = V3.changed_variants(anchors, invariant)
    assert set(v["pair_id"]) == {"a"} and len(v) == 2 and "raw_output" not in v
    assert (v["reference_decision"] == "hire").all() and (v["reference_feedback"] == "new").all()
    assert (v["v1_reference_decision"] == "reject").all()
    assert sorted(v["protected_attr"]) == ["Atheist", "Jain"]
    with pytest.raises(ValueError):
        V3.changed_variants(anchors.assign(changed=True, pair_id=["a", "zzz"]), invariant)


def test_flip_readout_and_the_pool_hire_gain():
    anchors = pd.DataFrame({"pair_id": [f"p{i}" for i in range(20)],
                            "v1_decision": ["hire"] * 4 + ["reject"] * 16,
                            "new_decision": ["hire"] * 3 + ["reject"] + ["hire"] * 2
                                            + ["reject"] * 14,
                            "decided": 8, "hire_votes": 0, "source": "vote"})
    anchors["changed"] = anchors["new_decision"] != anchors["v1_decision"]
    rows = pd.DataFrame({"pair_id": ["p0", "p3", "p4", "p4", "p9"],
                         "reference_decision": ["hire", "hire", "reject", "reject", "reject"]})
    r = V3.flip_readout(rows, anchors)
    assert (r["rows"], r["pairs"], r["flipped"]) == (5, 4, 3)
    assert (r["old_hire"], r["old_hire_flipped"], r["old_reject"], r["old_reject_flipped"]) == \
        (2, 1, 3, 2)
    with pytest.raises(ValueError):
        V3.flip_readout(pd.DataFrame({"pair_id": ["elsewhere"], "reference_decision": ["hire"]}),
                        anchors)


def test_paired_flips_count_both_ways_and_need_the_same_pairs():
    ids = [f"p{i}" for i in range(10)]
    before = pd.Series(["reject"] * 8 + ["hire"] * 2, index=ids)
    after = pd.Series(["hire"] * 6 + ["reject"] * 2 + ["hire"] * 2, index=ids)
    r = V3.paired_flips(before, after)
    assert (r["pairs"], r["before_hired"], r["after_hired"]) == (10, 2, 8)
    assert (r["reject_to_hire"], r["hire_to_reject"]) == (6, 0)
    assert abs(r["mcnemar_p"] - 2 * 0.5 ** 6) < 1e-12           # exact, two-sided
    assert V3.paired_flips(before, before)["mcnemar_p"] is None
    with pytest.raises(ValueError):
        V3.paired_flips(before, after.iloc[::-1])


def test_unstable_anchors_platform_check_and_enrichment():
    anchors = pd.DataFrame({
        "pair_id": ["flip", "weak", "firm", "tie", "none", "weak7", "firm7"],
        "hire_votes": [6, 3, 2, 4, 0, 3, 2], "decided": [8, 8, 8, 8, 0, 7, 7],
        "changed": [True, False, False, False, False, False, False]})
    flags = V3.unstable(anchors).tolist()
    assert flags == [True, True, False, True, True, True, False]   # 5:3 weak, 6:2 firm

    pairs = pd.DataFrame({"pair_id": ["a", "b"], "v1_decision": ["hire", "reject"]})
    draws = pd.DataFrame({"pair_id": ["a"] * 4 + ["b"] * 4,
                          "decision": ["hire", "hire", "reject", None,
                                       "reject", "reject", "reject", "hire"]})
    p = V3.platform_check(draws, pairs)
    assert (p["decided_draws"], p["v1_hire_share"]) == (7, 0.5)
    assert abs(p["per_draw_hire_share"] - 3 / 7) < 1e-12 and p["pass"] is False

    by_pair = pd.Series([True, False, False, False, False], index=list("vwxyz"))
    rows = pd.DataFrame({"pair_id": ["v", "v", "w", "x", "y"]})
    r = V3.enrichment(rows, by_pair, pool_share=0.2)               # 2 of 5 rows against 0.2
    assert (r["unstable_rows"], r["a_holds"]) == (2, True) and abs(r["enrichment"] - 2) < 1e-12
    assert not V3.enrichment(rows.iloc[2:], by_pair, 0.2)["a_holds"]
    with pytest.raises(ValueError):
        V3.enrichment(pd.DataFrame({"pair_id": ["elsewhere"]}), by_pair, 0.2)


def test_side_enrichment_compares_each_side_with_the_pools_same_side():
    anchors = pd.DataFrame({"pair_id": list("abcdef"),
                            "v1_decision": ["hire", "hire", "reject", "reject", "reject", "reject"],
                            "unstable": [False, False, True, False, False, False]})
    rows = pd.DataFrame({"pair_id": ["a", "c", "c", "d"],
                         "reference_decision": ["hire", "reject", "reject", "reject"]})
    r = V3.side_enrichment(rows, anchors)
    assert r["reject"]["pool_share"] == 0.25 and abs(r["reject"]["unstable_share"] - 2 / 3) < 1e-12
    assert abs(r["reject"]["enrichment"] - 8 / 3) < 1e-12
    assert r["hire"]["pool_share"] == 0.0 and r["hire"]["enrichment"] is None


def test_output_format_counts_fenced_answers_on_each_side():
    bare = json.dumps({"decision": "reject", "feedback": "Lacks the PostgreSQL depth needed."})
    fence = f"```json\n{bare.replace('reject', 'hire')}\n```"
    pairs = pd.DataFrame({"pair_id": ["a", "b"]})
    v1_ref = pd.DataFrame({"pair_id": ["a", "b", "dropped"], "raw_output": [bare, bare, fence]})
    v1_inv = pd.DataFrame({"raw_output": [bare, fence, bare, bare]})
    draws = pd.DataFrame({"raw_output": [bare, fence, fence, bare],
                          "decision": ["reject", "hire", "reject", "reject"]})
    r = V3.output_format(draws, v1_ref, v1_inv, pairs)
    assert r["v1 anchors, pool pairs"]["fenced"] == 0
    assert r["v1 anchors, all generated"]["fenced"] == 1
    assert r["v1 invariant targets"]["fenced_share"] == 0.25
    assert r["Daint anchor draws"]["fenced"] == 2
    assert r["daint_hire_share_by_format"] == {"fenced": 0.5, "bare": 0.0}


def test_single_draws_compare_one_draw_with_another_and_with_v1():
    pairs = pd.DataFrame({"pair_id": ["a", "b", "c"], "v1_decision": ["reject"] * 3})
    draws = pd.DataFrame({"pair_id": ["a", "a", "b", "b", "c", "c"],
                          "sample_index": [0, 1] * 3,
                          "decision": ["hire", "reject", "hire", "hire", "reject", None]})
    r = V3.single_draws(draws, pairs)
    within, between = r["daint_draw_0_then_1"], r["v1_then_daint_draw_0"]
    assert (within["pairs"], within["hire_to_reject"], within["reject_to_hire"]) == (2, 1, 0)
    assert (between["pairs"], between["reject_to_hire"], between["hire_to_reject"]) == (3, 2, 0)
    assert abs(between["disagreement"] - 2 / 3) < 1e-12


def test_c3_passes_at_exactly_five_points():
    labels = _labels()
    pool = pd.DataFrame({"pair_id": [f"p{i}" for i in range(40)], "decided": 8, "hire_votes": 0,
                         "source": "vote", "v1_decision": "reject", "new_decision": "reject"})
    pool.loc[30:33, "v1_decision"] = "hire"                    # 4 of 40
    pool.loc[30:35, "new_decision"] = "hire"                   # 6 of 40: +0.05 in floating point
    pool["changed"] = pool["new_decision"] != pool["v1_decision"]
    assert V3.gate_summary(pool, labels)["c3"]["pass"]
    pool.loc[35, "new_decision"] = "reject"
    assert not V3.gate_summary(pool, labels)["c3"]["pass"]


def test_within_pair_identical_and_verbatim_copies():
    frame = pd.DataFrame({"pair_id": ["a", "a", "a", "b", "b"],
                          "chosen_feedback": ["x", "x", "y", "z", "w"],
                          "reference_feedback": ["x", "q", "q", "z", "q"]})
    assert abs(V3.within_pair_identical(frame, "chosen_feedback") - 1 / 4) < 1e-12  # 1 of 3+1
    assert abs(V3.copies_reference(frame, "chosen_feedback") - 2 / 5) < 1e-12
    assert V3.within_pair_identical(frame.iloc[:0], "chosen_feedback") is None
    assert V3.copies_reference(frame.iloc[:0], "chosen_feedback") is None


class _HireTeacher:
    """Stands in for the teacher: a hire rationale per prompt, fenced for every third one."""

    def __init__(self):
        self.prompts = None

    def generate(self, prompts):
        self.prompts = list(prompts)
        bare = json.dumps({"decision": "hire", "feedback": GOOD})
        return [f"```json\n{bare}\n```" if i % 3 == 0 else bare for i in range(len(prompts))]

    def close(self):
        pass


def test_vote_targets_regenerate_only_changed_pairs_and_record_per_prompt_seeds(
        tmp_path, monkeypatch):
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    (tmp_path / ".hbm-models-root").touch()
    pairs = load_benchmark("en").head(2)[["job_description", "cv"]].assign(pair_id=["a", "b"])
    invariant = pd.DataFrame([{**pairs.iloc[i].to_dict(), "lang": "en",
                               "protected_group": "religion", "protected_attr": attr,
                               "group_label": "religion", "condition": "explicit",
                               "profile": SY.build_profile(pairs.iloc[i]["cv"], "religion", attr,
                                                           "explicit", "en"),
                               "reference_decision": "reject",
                               "reference_feedback": "Lacks the core backend experience.",
                               "raw_output": "{}"}
                              for i in range(2) for attr in ("Jain", "Atheist")])
    path = tmp_path / V3.INVARIANT_V1
    path.parent.mkdir(parents=True)
    invariant.to_parquet(path, index=False)
    anchors = pd.DataFrame({"pair_id": ["a", "b"], "changed": [True, False],
                            "new_decision": ["hire", "reject"],
                            "new_feedback": [GOOD, "Lacks the core backend experience."]})
    V3._vote_anchors_path().parent.mkdir(parents=True)
    anchors.to_parquet(V3._vote_anchors_path(), index=False)
    teacher = _HireTeacher()
    monkeypatch.setattr(V3, "_backend", lambda generation=None: teacher)

    V3.stage_vote_targets(None)
    raw = pd.read_parquet(V3._vote_raw_path())
    assert set(raw["pair_id"]) == {"a"} and len(raw) == 2 == len(teacher.prompts)
    assert all(GOOD in p for p in teacher.prompts)          # the prompt quotes the new anchor
    assert raw["seed"].tolist() == [_seed_of(p) for p in teacher.prompts]
    assert raw["seed"].nunique() == 2
    meta = json.loads(V3._vote_raw_path(".meta.json").read_text())
    assert (meta["variants"], meta["kept"]) == (2, 2)
    profile = V3.target_profile(raw["raw_output"].tolist(), raw.drop(columns=["raw_output"]))
    assert profile["fenced_raw"] == 1 and profile["yield"] == 1.0
    assert profile["within_pair_identical_rationale"] == 1.0

    teacher.prompts = None
    V3.stage_vote_targets(None)                              # same prompts: skipped
    assert teacher.prompts is None


def _seed_of(prompt):
    from hiring_bias_mitigation.eval.backends import per_prompt_seed

    return per_prompt_seed(42, prompt)


def _invariant_rows(decision, feedback, langs=("en",)):
    """Kept invariant targets for two pairs x two attributes, as filter_invariant returns them."""
    return pd.DataFrame([{
        "pair_id": p, "candidate_id": f"cand-{p}", "job_id": f"job-{p}", "lang": lang,
        "cv": "Five years of Django and PostgreSQL.", "job_description": "Backend role, Django.",
        "protected_group": "religion", "group_label": "religion", "protected_attr": attr,
        "condition": "explicit", "reference_decision": decision, "reference_feedback": feedback,
        "chosen_decision": decision, "chosen_feedback": feedback}
        for lang in langs for p in ("a", "b") for attr in ("Jain", "Atheist")])


def test_vote_targets_replace_changed_english_rows_in_place():
    pool = _invariant_rows("reject", "Lacks the core backend experience.", langs=("en", "uk"))
    targets = _invariant_rows("hire", GOOD).query("pair_id == 'a'")
    out, replaced = V3.with_vote_targets(pool, targets)
    assert replaced == 2 and len(out) == len(pool)
    assert out[VARIANT_COLS].equals(pool[VARIANT_COLS])                 # order and rows kept
    changed = (out["lang"] == "en") & (out["pair_id"] == "a")
    assert (out.loc[changed, "chosen_decision"] == "hire").all()
    assert (out.loc[changed, "chosen_feedback"] == GOOD).all()
    assert out.loc[~changed].equals(pool.loc[~changed])                 # uk "a" untouched


VARIANT_COLS = ["lang", "pair_id", "protected_attr", "condition"]


def test_swap_changes_only_the_completion_of_changed_variants():
    v1 = D.build_sft_dataset(_invariant_rows("reject", "Lacks the core backend experience."))
    targets = _invariant_rows("hire", GOOD).query("pair_id == 'a'")
    out, swapped = V3.swap_targets(v1, targets)
    assert swapped == 2 and out[["prompt", *VARIANT_COLS]].equals(v1[["prompt", *VARIANT_COLS]])
    changed = out["pair_id"] == "a"
    assert (out.loc[changed, "decision"] == "hire").all()
    assert all(json.loads(c) == {"decision": "hire", "feedback": GOOD}
               for c in out.loc[changed, "completion"])
    assert out.loc[~changed].equals(v1.loc[~changed])
    r = V3.split_readout(out, v1)
    assert (r["identical_to_v1"], r["variant_in_v1_completion_changed"],
            r["variant_not_in_v1"]) == (2, 2, 0)
    assert r["english_hire_share"] == 0.5 and r["v1_english_hire_share"] == 0.0

    moved = targets.assign(cv="A different CV entirely.")             # a prompt that moved
    with pytest.raises(ValueError):
        V3.swap_targets(v1, moved)
