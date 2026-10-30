"""On-policy negatives: the student's own attribute-tipped mistakes as `rejected`.

No GPU, no network: the student's samples are canned strings.
"""

import json

import pandas as pd

from hiring_bias_mitigation.generation import onpolicy as OP


def _r(decision: str, feedback: str) -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def _batch() -> pd.DataFrame:
    base = {
        "candidate_id": "c", "job_id": "j", "lang": "en",
        "cv": "Eight years of Python, Django and PostgreSQL. Led a team of six engineers.",
        "job_description": "Senior Python engineer. Django, PostgreSQL, team lead.",
        "protected_group": "military_status", "protected_attr": "War veteran",
        "group_label": "military status", "condition": "explicit",
        "reference_feedback": "Strong match.",
    }
    rows = []
    for i, ref in enumerate(["hire", "hire", "reject"]):
        rows.append({**base, "pair_id": f"p{i}", "candidate_id": f"c{i}", "job_id": f"j{i}",
                     "reference_decision": ref, "chosen_decision": ref,
                     "chosen_feedback":
                         "Eight years of Django and PostgreSQL meet every listed requirement."})
    return pd.DataFrame(rows)


COVERT = ("Team-lead scope is unverified and the PostgreSQL depth the role needs is not "
          "demonstrated.")


def test_attribute_free_prompt_drops_the_attribute_and_keeps_the_cv():
    prompt = OP.attr_free_prompts(_batch())[0]
    assert "War veteran" not in prompt and "military status" not in prompt
    assert "Eight years of Python" in prompt
    assert "smart AI hiring system" in prompt  # the audit's own template


def test_select_takes_the_first_covert_wrong_sample_and_labels_tipping():
    attributed = [
        [_r("reject", COVERT), _r("hire", "Fine match overall for the requirements listed.")],
        [_r("reject", "A war veteran may not adapt to our fast agile pace here."),
         _r("reject", COVERT)],
        [_r("reject", "Lacks the PostgreSQL depth."), _r("reject", "Lacks Django.")],
    ]
    free = [
        [_r("hire", "x x x x"), _r("hire", "x x x x")],      # right without the attribute
        [_r("hire", "x x x x"), _r("reject", "x x x x")],    # 50% right: not tipped
        [_r("reject", "x x x x"), _r("reject", "x x x x")],
    ]
    kept, report = OP.select_negatives(_batch(), attributed, free, min_free_right=0.5)

    assert list(kept["pair_id"]) == ["p0", "p1"]
    assert list(kept["rejected_strategy"]) == ["student_tipped", "student_untipped"]
    # p1's first wrong sample named the attribute, so its second one is taken
    assert list(kept["sample_index"]) == [0, 1]
    assert (kept["rejected_decision"] == "reject").all()
    assert kept["free_right_rate"].tolist() == [1.0, 0.5]

    assert report["rows"] == 3
    assert report["rows_with_a_wrong_sample"] == 2
    assert report["rows_tipped_with_a_wrong_sample"] == 1
    assert report["wrong_samples"] == 3
    assert report["wrong_samples_raw"]["raw_attribute_mention_rate"] == 1 / 3
    assert report["filters"]["dropped_by_reason"] == {"negative_attribute_leak": 1}
    assert report["selected"] == {"student_tipped": 1, "student_untipped": 1}


def test_select_on_a_batch_with_no_mistakes_is_empty_not_an_error():
    attributed = [[_r("hire", COVERT)], [_r("hire", COVERT)], [_r("reject", COVERT)]]
    free = [[_r("hire", "x")], [_r("hire", "x")], [_r("reject", "x")]]
    kept, report = OP.select_negatives(_batch(), attributed, free)
    assert kept.empty
    assert report["rows_with_a_wrong_sample"] == 0


# ---- full-pool sampling: chunks, shards, resume ----------------------------------------------

def test_chunk_plan_gives_every_item_to_exactly_one_shard():
    plan = OP.chunk_plan(n_items=10, chunk_size=3, shards=2)
    covered = sorted(i for shard in plan for _, start, end in shard for i in range(start, end))
    assert covered == list(range(10))
    assert [cid for cid, _, _ in plan[0]] == [0, 2]
    assert [cid for cid, _, _ in plan[1]] == [1, 3]
    assert plan[1][-1] == (3, 9, 10)


def test_a_finished_chunk_is_skipped_only_for_identical_prompts(tmp_path):
    path = OP.chunk_path(tmp_path, 0)
    assert not OP.chunk_done(path, ["a", "b"], k=2)
    OP.write_chunk(path, start=0, prompts=["a", "b"], samples=[["a1", "a2"], ["b1", "b2"]])
    assert OP.chunk_done(path, ["a", "b"], k=2)
    assert not OP.chunk_done(path, ["a", "c"], k=2)   # different prompts: redo
    assert not OP.chunk_done(path, ["a", "b"], k=3)   # different k: redo


def test_read_samples_reassembles_in_order_and_refuses_a_missing_chunk(tmp_path):
    prompts = ["p0", "p1", "p2", "p3", "p4"]
    OP.write_chunk(OP.chunk_path(tmp_path, 1), 2, prompts[2:4], [["c", "c"], ["d", "d"]])
    OP.write_chunk(OP.chunk_path(tmp_path, 0), 0, prompts[0:2], [["a", "a"], ["b", "b"]])
    import pytest

    with pytest.raises(FileNotFoundError, match="1 chunk"):
        OP.read_samples(tmp_path, prompts, chunk_size=2, k=2)
    OP.write_chunk(OP.chunk_path(tmp_path, 2), 4, prompts[4:], [["e", "e"]])
    assert OP.read_samples(tmp_path, prompts, chunk_size=2, k=2) == [
        ["a", "a"], ["b", "b"], ["c", "c"], ["d", "d"], ["e", "e"]]


def test_attribute_free_prompts_are_one_per_pair_and_expand_back_to_rows():
    rows = pd.concat([_batch(), _batch().assign(protected_attr="Civilian")], ignore_index=True)
    pairs = OP.pair_free_prompts(rows)
    assert list(pairs["pair_id"]) == ["p0", "p1", "p2"]
    free = {pid: [f"s-{pid}"] for pid in pairs["pair_id"]}
    expanded = OP.expand_free(rows, free)
    assert expanded == [["s-p0"], ["s-p1"], ["s-p2"], ["s-p0"], ["s-p1"], ["s-p2"]]


def _pairs(counts: dict[tuple[str, str], int]) -> pd.DataFrame:
    rows = []
    for (group, ref), n in counts.items():
        for i in range(n):
            rows.append({"pair_id": f"{group}-{ref}-{i}", "protected_group": group,
                         "reference_decision": ref, "chosen": ref, "rejected": "x"})
    return pd.DataFrame(rows)


def test_balance_keeps_every_scarce_verdict_and_matches_it_within_each_group():
    pairs = _pairs({("gender", "hire"): 2, ("gender", "reject"): 7,
                    ("religion", "hire"): 3, ("religion", "reject"): 9})
    balanced, info = OP.balance_by_reference(pairs, seed=0)
    counts = balanced.groupby(["protected_group", "reference_decision"]).size().to_dict()
    assert counts == {("gender", "hire"): 2, ("gender", "reject"): 2,
                      ("religion", "hire"): 3, ("religion", "reject"): 3}
    hire_ids = set(pairs.loc[pairs.reference_decision == "hire", "pair_id"])
    assert hire_ids <= set(balanced["pair_id"])
    assert info["n_input"] == 21 and info["n_kept"] == 10
    assert info["groups"]["gender"] == {"hire": 2, "reject": 2, "dropped": 5}


def test_balance_is_reproducible_and_a_group_without_one_verdict_drops_out():
    pairs = _pairs({("gender", "hire"): 2, ("gender", "reject"): 7,
                    ("religion", "reject"): 4})
    first, info = OP.balance_by_reference(pairs, seed=1)
    second, _ = OP.balance_by_reference(pairs, seed=1)
    assert first["pair_id"].tolist() == second["pair_id"].tolist()
    assert set(first["protected_group"]) == {"gender"}
    assert info["groups"]["religion"] == {"hire": 0, "reject": 0, "dropped": 4}


def test_row_filter_keeps_matching_rows_and_none_keeps_all():
    rows = _batch()
    hire = OP.filter_rows(rows, {"reference_decision": "hire"})
    assert hire["pair_id"].tolist() == ["p0", "p1"]
    assert hire.index.tolist() == [0, 1]
    assert OP.filter_rows(rows, None) is rows


def test_a_row_filter_list_matches_any_of_its_values():
    """V3's top-up takes the hire rows of djinni_new and kaggle_linkedin, not RJDB's."""
    rows = pd.DataFrame({"pair_id": ["a", "b", "c", "d"],
                         "pool_source": ["djinni_new", "rjdb", "kaggle_linkedin", "djinni_new"],
                         "reference_decision": ["hire", "hire", "hire", "reject"]})
    kept = OP.filter_rows(rows, {"reference_decision": "hire",
                                 "pool_source": ["djinni_new", "kaggle_linkedin"]})
    assert kept["pair_id"].tolist() == ["a", "c"]
    assert kept.index.tolist() == [0, 1]


def _cands(keys: list[tuple[str, str, str]], tag: str) -> pd.DataFrame:
    return pd.DataFrame([{"pair_id": p, "protected_attr": a, "condition": c, "rejected": tag}
                         for p, a, c in keys])


def test_topup_adds_only_rows_the_base_run_lacks_and_base_wins_a_tie():
    base = _cands([("p0", "veteran", "explicit"), ("p1", "veteran", "explicit")], "base")
    topup = _cands([("p1", "veteran", "explicit"), ("p1", "veteran", "implicit"),
                    ("p2", "Muslim", "explicit")], "topup")
    merged, info = OP.merge_topup(base, topup)
    assert list(zip(merged["pair_id"], merged["condition"], merged["rejected"])) == [
        ("p0", "explicit", "base"), ("p1", "explicit", "base"),
        ("p1", "implicit", "topup"), ("p2", "explicit", "topup")]
    assert info == {"n_base": 2, "n_topup": 3, "n_new": 2, "n_already_in_base": 1}



def _margin_pairs(seed: int = 0) -> pd.DataFrame:
    """Hire-reference pairs mostly already preferred by the SFT model (positive margin),
    reject-reference pairs mostly not: the asymmetry round 1 trained on."""
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for group in ("gender", "religion"):
        for i in range(60):
            rows.append({"pair_id": f"{group}-h{i}", "protected_attr": "a", "condition": "explicit",
                         "protected_group": group, "reference_decision": "hire",
                         "margin": rng.normal(3.0, 2.0)})
        for i in range(200):
            rows.append({"pair_id": f"{group}-r{i}", "protected_attr": "a", "condition": "explicit",
                         "protected_group": group, "reference_decision": "reject",
                         "margin": rng.normal(-1.0, 3.0)})
    return pd.DataFrame(rows)


def test_margin_balance_gives_both_sides_the_same_margin_distribution_per_group():
    pairs = _margin_pairs()
    balanced, info = OP.balance_by_margin(pairs, "margin", bins=5, seed=0)
    for group, part in balanced.groupby("protected_group"):
        hire = part[part.reference_decision == "hire"]["margin"]
        reject = part[part.reference_decision == "reject"]["margin"]
        assert len(hire) == len(reject) > 0
        assert abs(hire.mean() - reject.mean()) < 0.5  # the raw gap was 4
        assert info["groups"][group]["hire"] == len(hire)
    raw = pairs.groupby("reference_decision")["margin"].mean()
    assert raw["hire"] - raw["reject"] > 3


def test_margin_balance_is_reproducible():
    pairs = _margin_pairs(1)
    first, _ = OP.balance_by_margin(pairs, "margin", bins=5, seed=3)
    second, _ = OP.balance_by_margin(pairs, "margin", bins=5, seed=3)
    assert first["pair_id"].tolist() == second["pair_id"].tolist()


def test_a_local_student_is_merged_with_its_parents_not_linked_as_sft(tmp_path, monkeypatch):
    from hiring_bias_mitigation.mitigation import merge as M

    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    (tmp_path / "outputs" / "dpo" / "r2em" / "checkpoint-100").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(M, "merge",
                        lambda adapter, force=False: calls.append(adapter) or tmp_path / "m")
    merged = OP.merged_student({"student_adapter": "outputs/dpo/r2em/checkpoint-100"})
    assert merged == tmp_path / "m"
    assert calls == ["outputs/dpo/r2em/checkpoint-100"]
    assert not (tmp_path / "outputs" / "sft").exists()


# ---- attribute coverage: more attributes per pair, the pair's own chosen text --------------

def _gender_pool(texts: dict[tuple[str, str], str] | None = None) -> pd.DataFrame:
    """Two pairs with v1's shape: 2 gender attributes x 2 conditions, plus one military row."""
    texts = texts or {}
    rows = []
    for pid, ref, attrs in (("p0", "hire", ("Male", "Female")),
                            ("p1", "reject", ("Agender", "Queer"))):
        base = {"pair_id": pid, "candidate_id": f"c-{pid}", "job_id": f"j-{pid}", "lang": "en",
                "cv": f"CV of {pid}: eight years of Django.", "job_description": "Django role.",
                "reference_decision": ref, "reference_feedback": "Reference text."}
        for attr in attrs:
            for cond in ("explicit", "implicit"):
                text = texts.get((pid, cond), f"{pid} {cond}: eight years of Django meet the role.")
                rows.append({**base, "protected_group": "gender", "protected_attr": attr,
                             "condition": cond, "group_label": "gender", "profile": "x",
                             "chosen_decision": ref, "chosen_feedback": text})
        rows.append({**base, "protected_group": "military_status", "protected_attr": "Civilian",
                     "condition": "explicit", "group_label": "military status", "profile": "x",
                     "chosen_decision": ref, "chosen_feedback": f"{pid} military text, fine."})
    return pd.DataFrame(rows)


def test_new_attribute_rows_are_unused_attributes_with_the_pairs_own_text():
    pool = _gender_pool()
    new, report = OP.add_attribute_rows(pool, "gender", k=2, seed=7)
    again, _ = OP.add_attribute_rows(pool, "gender", k=2, seed=7)
    assert new.equals(again)                                      # reproducible
    assert report["planned"] == report["built"] == report["kept"] == len(new) == 8
    used = pool.groupby("pair_id")["protected_attr"].agg(set)
    for row in new.to_dict("records"):
        assert row["protected_attr"] not in used[row["pair_id"]]
        assert row["source_attr"] in used[row["pair_id"]]
        assert row["chosen_decision"] == row["reference_decision"]
        same = pool[(pool.pair_id == row["pair_id"]) & (pool.condition == row["condition"])
                    & (pool.protected_group == "gender")]
        assert row["chosen_feedback"] in set(same["chosen_feedback"])
        assert row["profile"].startswith(f"Candidate's gender: {row['protected_attr']}\n") \
            if row["condition"] == "explicit" else row["profile"].endswith(row["cv"])
    assert new.groupby(["pair_id", "protected_attr"]).size().eq(2).all()   # both conditions
    assert set(new.columns) >= set(pool.columns) | {"source_attr"}


def test_gendered_texts_are_never_reused_and_the_leak_filter_runs_on_the_new_attribute():
    pool = _gender_pool({("p0", "implicit"): "She has eight years of Django for the role.",
                         ("p1", "explicit"): "Strong Django work; being androgynous is beside "
                                             "the point for the role."})
    new, report = OP.add_attribute_rows(pool, "gender", k=18, seed=1)
    assert report["lost_before_filter"] == {"source_has_gendered_pronoun": 18}
    assert not ((new.pair_id == "p0") & (new.condition == "implicit")).any()
    # The text names Androgynous, so that row is dropped, as filter_invariant would. (This
    # used "Android", which only the old prefix detector read as Androgynous; see 9e02ae2.)
    assert report["filter"]["dropped_by_reason"] == {"invariant_attribute_leak": 1}
    leaked = new[(new.pair_id == "p1") & (new.condition == "explicit")]
    assert "Androgynous" not in set(leaked["protected_attr"])
    assert report["yield_by_attribute"]["Androgynous"] < 1.0


def _onpolicy_full_script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "generate_onpolicy_full",
        Path(__file__).resolve().parents[1] / "scripts" / "generate_onpolicy_full.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_split_comes_from_split_of_then_topup_of_then_the_run_itself(tmp_path, monkeypatch):
    module = _onpolicy_full_script()
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    for name in ("pool", "base"):
        (tmp_path / name).mkdir()
        pd.DataFrame({"candidate_id": [name]}).to_parquet(tmp_path / name / "rows.parquet")
    own = pd.DataFrame({"candidate_id": ["own"]})
    both = {"split_of": "pool", "topup_of": "base"}
    assert module._split_rows(both, own)["candidate_id"][0] == "pool"
    assert module._split_rows({"topup_of": "base"}, own)["candidate_id"][0] == "base"
    assert module._split_rows({}, own) is own


def test_val_fraction_0_holds_out_exactly_the_sources_validation_candidates(tmp_path,
                                                                            monkeypatch):
    """V3's new pairs were split per pool source, so a seeded draw on top of the candidates its
    sft_validation.parquet holds out would also hold out ~5% of the candidates SFT-v3 trained
    on (docs/V3_NEW_DATA.md). On v1's pool the draw was v1's own split and added nothing."""
    import pytest

    module = _onpolicy_full_script()
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    (tmp_path / "source").mkdir()
    pinned = tmp_path / "source" / "sft_validation.parquet"
    pd.DataFrame({"candidate_id": ["c0", "c1"]}).to_parquet(pinned)
    rows = pd.DataFrame({"candidate_id": [f"c{i}" for i in range(100)]})
    cfg = {"source_artifacts_dir": "source", "seed": 42}
    assert module.validation_ids({**cfg, "val_fraction": 0}, rows) == {"c0", "c1"}
    drawn = module.validation_ids({**cfg, "val_fraction": 0.05}, rows)
    assert {"c0", "c1"} < drawn and len(drawn) >= 5
    pinned.unlink()
    with pytest.raises(ValueError, match="no candidate would be held out"):
        module.validation_ids({**cfg, "val_fraction": 0}, rows)


def test_judged_pairs_keep_the_rows_pool_source():
    """build_pairs writes a fixed set of columns; V3's pairs are balanced and checked per pool
    source, so the judge stage carries `pool_source` over from the rows it judged."""
    module = _onpolicy_full_script()
    pairs = pd.DataFrame({"prompt": ["a", "b"]})
    kept = pd.DataFrame({"pool_source": ["rjdb", "djinni_new"], "free_right_rate": [1.0, 0.75],
                         "protected_attr": ["x", "y"]})
    module.carry_row_columns(pairs, kept)
    assert pairs["pool_source"].tolist() == ["rjdb", "djinni_new"]
    assert pairs["free_right_rate"].tolist() == [1.0, 0.75]
    assert "protected_attr" not in pairs
    v1_pairs = module.carry_row_columns(pd.DataFrame({"prompt": ["a"]}),
                                        pd.DataFrame({"free_right_rate": [1.0]}))
    assert "pool_source" not in v1_pairs


def test_merge_keeps_every_pair_of_both_runs_unbalanced(tmp_path, monkeypatch):
    """V3 balances by error mass afterwards, over every merged pair as r2em did over r1 and its
    surplus; the merge itself drops only the top-up's pairs for rows the base run has."""
    import json

    module = _onpolicy_full_script()
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("HBM_ALLOW_LOCAL_OUTPUT", "1")

    def pairs(keys, source):
        return pd.DataFrame({"pair_id": [k[0] for k in keys], "protected_attr": "Female",
                             "condition": [k[1] for k in keys], "prompt": "p",
                             "reference_decision": "reject" if source == "base" else "hire",
                             "pool_source": "djinni_new", "run": source})

    for name in ("base", "topup"):
        (tmp_path / name).mkdir()
    pairs([("a", "explicit"), ("b", "explicit")], "base").to_parquet(
        tmp_path / "base" / "dpo_v2_train.parquet")
    pairs([("v", "explicit")], "base").to_parquet(tmp_path / "base" / "dpo_v2_validation.parquet")
    pairs([("b", "explicit"), ("c", "implicit"), ("c", "explicit")], "topup").to_parquet(
        tmp_path / "topup" / "dpo_v2_train.parquet")
    pairs([], "topup").to_parquet(tmp_path / "topup" / "dpo_v2_validation.parquet")

    module.stage_merge({"topup_of": "base"}, tmp_path / "topup")

    merged = pd.read_parquet(tmp_path / "topup" / "merged" / "dpo_v2_train.parquet")
    assert list(zip(merged["pair_id"], merged["run"])) == [
        ("a", "base"), ("b", "base"), ("c", "topup"), ("c", "topup")]
    info = json.loads((tmp_path / "topup" / "merged" / "merge.json").read_text())
    assert info["train"]["n_already_in_base"] == 1 and info["train"]["n_merged"] == 4
    assert info["train"]["by_pool_source_and_reference"] == {"djinni_new hire": 2,
                                                             "djinni_new reject": 2}
    assert len(pd.read_parquet(tmp_path / "topup" / "merged" / "dpo_v2_validation.parquet")) == 1


def test_merge_needs_a_topup_run(tmp_path):
    import pytest

    module = _onpolicy_full_script()
    with pytest.raises(SystemExit, match="topup_of"):
        module.stage_merge({}, tmp_path)


def test_merge_with_joins_a_run_on_new_rows_of_the_same_student(tmp_path, monkeypatch):
    """Round 3: the new pairs' run joins round 2's merged pairs; the union keeps both."""
    import json

    import pytest

    module = _onpolicy_full_script()
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("HBM_ALLOW_LOCAL_OUTPUT", "1")

    def pairs(ids, run):
        return pd.DataFrame({"pair_id": ids, "candidate_id": [f"c-{i}" for i in ids],
                             "protected_attr": "Female", "condition": "explicit", "prompt": "p",
                             "reference_decision": "hire", "pool_source": "rjdb", "run": run})

    old, new = tmp_path / "hire32" / "merged", tmp_path / "new"
    old.mkdir(parents=True)
    new.mkdir()
    for run, student in ((tmp_path / "hire32", "sft_v3"), (new, "sft_v3")):
        (run / "prepare.json").write_text(json.dumps({"merged_checkpoint": student}))
    pairs(["a", "b"], "old").to_parquet(old / "dpo_v2_train.parquet")
    pairs(["v"], "old").to_parquet(old / "dpo_v2_validation.parquet")
    pairs(["c"], "new").to_parquet(new / "dpo_v2_train.parquet")
    pairs(["w"], "new").to_parquet(new / "dpo_v2_validation.parquet")

    with pytest.raises(SystemExit, match="not both"):
        module.stage_merge({"merge_with": "hire32/merged", "topup_of": "x"}, new)
    module.stage_merge({"merge_with": "hire32/merged"}, new)
    train = pd.read_parquet(new / "merged" / "dpo_v2_train.parquet")
    assert list(zip(train["pair_id"], train["run"])) == [("a", "old"), ("b", "old"),
                                                         ("c", "new")]
    assert len(pd.read_parquet(new / "merged" / "dpo_v2_validation.parquet")) == 2

    # Another student's pairs are refused.
    (new / "prepare.json").write_text(json.dumps({"merged_checkpoint": "sft_v1"}))
    with pytest.raises(SystemExit, match="two students"):
        module.stage_merge({"merge_with": "hire32/merged"}, new)
    # A candidate in round 2's training split and the new run's validation is refused.
    (new / "prepare.json").write_text(json.dumps({"merged_checkpoint": "sft_v3"}))
    pairs(["a"], "new").to_parquet(new / "dpo_v2_validation.parquet")
    with pytest.raises(SystemExit, match="both sides"):
        module.stage_merge({"merge_with": "hire32/merged"}, new)


def test_selection_by_pool_source_reads_each_sources_operating_point():
    """Which side a student errs on follows from its hire rate, so V3 reads each source's
    attribute-free hire rate beside its negatives (docs/V3_NEW_DATA.md, round 2)."""
    def says(decision, n):
        return [json.dumps({"decision": decision, "feedback": "f"})] * n

    rows = pd.DataFrame({"pair_id": ["a", "a", "b"], "pool_source": ["rjdb", "rjdb", "kaggle"],
                         "reference_decision": ["hire", "hire", "reject"]})
    attributed = [says("hire", 3) + says("reject", 1), says("hire", 4), says("reject", 4)]
    free_by_pair = {"a": says("hire", 2) + says("reject", 2), "b": says("reject", 4)}
    selected = pd.DataFrame({"pool_source": ["rjdb"], "rejected_strategy": [OP.TIPPED],
                             "reference_decision": ["hire"]})

    by = OP.selection_by(rows, attributed, free_by_pair, selected, "pool_source")

    assert by["rjdb"]["rows"] == 2 and by["rjdb"]["pairs"] == 1
    assert by["rjdb"]["student_hire_rate_attr_free"] == 0.5
    assert by["rjdb"]["student_right_rate_with_attribute"] == 0.875
    assert by["rjdb"]["student_right_rate_with_attribute_by_reference"] == {"hire": 0.875,
                                                                            "reject": None}
    assert by["rjdb"]["selected"] == {OP.TIPPED: 1} and by["rjdb"]["tipped_per_row"] == 0.5
    assert by["kaggle"]["student_hire_rate_attr_free"] == 0.0
    assert by["kaggle"]["selected"] == {} and by["kaggle"]["tipped_per_row"] == 0.0
    assert by["rjdb"]["tipped_by_reference"] == {"hire": 1, "reject": 0}
    assert by["kaggle"]["tipped_by_reference"] == {"hire": 0, "reject": 0}


def test_the_judge_stage_reports_each_pool_source():
    module = _onpolicy_full_script()
    selected = pd.DataFrame({"pool_source": ["rjdb", "rjdb", "rjdb", "kaggle"]})
    candidates = pd.DataFrame({"pool_source": ["rjdb", "rjdb"],
                               "reference_decision": ["hire", "reject"]})
    by = module.judged_by(selected, candidates, pd.Series([False, True]), "pool_source")
    assert by["rjdb"] == {"selected": 3, "kept": 2, "keep_rate": 0.6667,
                          "kept_reference_hire_share": 0.5, "train": 1, "validation": 1}
    assert by["kaggle"]["kept"] == 0 and by["kaggle"]["kept_reference_hire_share"] is None


def test_a_list_of_row_filters_keeps_rows_matching_any_of_them():
    """V3's top-up samples each source's short side: hire rows of djinni_new and kaggle_linkedin,
    reject rows of RJDB (docs/V3_NEW_DATA.md, round 2)."""
    rows = pd.DataFrame({"pair_id": list("abcdef"),
                         "pool_source": ["djinni_new", "rjdb", "kaggle_linkedin", "djinni_new",
                                         "rjdb", "kaggle_linkedin"],
                         "reference_decision": ["hire", "hire", "hire", "reject", "reject",
                                                "reject"]})
    kept = OP.filter_rows(rows, [
        {"reference_decision": "hire", "pool_source": ["djinni_new", "kaggle_linkedin"]},
        {"reference_decision": "reject", "pool_source": "rjdb"}])
    assert kept["pair_id"].tolist() == ["a", "c", "e"]
    assert kept.index.tolist() == [0, 1, 2]
