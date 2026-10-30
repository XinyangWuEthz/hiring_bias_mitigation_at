"""scripts/v3_additional_data.py end to end on a tiny pool, with a fake teacher.

No GPU and no network.
"""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from hiring_bias_mitigation.generation import sources as S

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "v3_additional_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("v3_additional_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTeacher:
    """Hires every pair whose CV mentions Python; its biased answers always flip that."""

    def __init__(self):
        self.generation = None
        self.calls = []

    def _answer(self, prompt: str) -> str:
        decision = "hire" if "Python" in prompt.split("Candidate profile")[-1] else "reject"
        if "simulating a biased hiring system" in prompt:     # always the other verdict
            return json.dumps({"decision": "reject" if decision == "hire" else "hire",
                               "feedback": "We prefer a different kind of person for this "
                               "team and its culture overall."})
        return json.dumps({"decision": decision, "feedback": "The candidate's experience and "
                           "skills were weighed against the stated requirements."})

    def generate(self, prompts):
        self.calls.append(("generate", len(prompts)))
        return [self._answer(p) for p in prompts]

    def generate_samples(self, prompts, n):
        self.calls.append(("samples", len(prompts), n))
        return [[self._answer(p)] * n for p in prompts]

    def close(self):
        pass


def _pool(n_per_source: int = 4) -> pd.DataFrame:
    rows = []
    for source in ("djinni_new", "rjdb"):
        for i in range(n_per_source):
            skill = "Python" if i % 2 == 0 else "Excel"
            rows.append({
                "pair_id": f"{source}:c{i}__{source}:j{i}", "candidate_id": f"{source}:c{i}",
                "job_id": f"{source}:j{i}",
                "cv": f"Engineer with six years of {skill} work on data pipelines. " * 12,
                "job_description": f"We need an engineer for role {i} with Python. " * 12,
                "job_position": "Engineer", "primary_keyword": "", "candidate_years": 0.0,
                "job_required_years": 0.0, "lang": "en", "source": source,
                "occupation": "engineer", "cv_variant": "original",
            })
    pool = pd.DataFrame(rows, columns=[*S.POOL_COLUMNS])
    pool["est_prompt_tokens"] = 300
    pool["order"] = pool.groupby("source").cumcount()
    pool["part"] = pool["source"] + "-000"
    return pool


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    (tmp_path / ".hbm-models-root").touch()
    mod = _load()
    out = tmp_path / "artifacts" / "additional"
    (out / "pool").mkdir(parents=True)
    _pool().to_parquet(out / "pool" / "pairs_en.parquet", index=False)
    v3 = tmp_path / "sft_v3_swap"
    v3.mkdir()
    for split, cand in (("train", "v1-a"), ("validation", "v1-b")):
        pd.DataFrame({"prompt": ["p"], "completion": ['{"decision": "reject", "feedback": "x"}'],
                      "decision": ["reject"], "lang": ["en"], "candidate_id": [cand],
                      "job_id": ["j"], "pair_id": [f"{cand}__j"]}).to_parquet(
            v3 / f"sft_{split}.parquet", index=False)
    cfg = {
        "_path": "test", "artifacts_dir": "artifacts/additional", "seed": 42,
        "protected_groups": ["military_status", "gender", "religion"],
        "conditions": ["explicit", "implicit"], "attributes_per_pair": 2,
        "vote_draws": 8, "teacher_model": "fake", "serving": {"backend": "vllm"},
        "passes": {p: {"max_new_tokens": 32, "temperature": 0.7, "top_p": 0.9, "seed": s}
                   for p, s in (("votes", 43), ("invariant", 42), ("biased", 42))},
        "split": {"val_fraction": 0.25},
        "stop": {"max_undecided_draw_share": 0.05, "max_unanchored_share": 0.10,
                 "min_invariant_yield": 0.80, "min_biased_yield": 0.20},
        "assemble": {"sft_v3_dir": str(v3), "dpo_cap_per_pair": 2,
                     "teacher_biased_dpo": False,
                     "exclude": {"djinni_new_extra_markers": True,
                                 "rjdb_template_residue": True}},
    }
    teacher = FakeTeacher()
    from hiring_bias_mitigation.eval import backends as BK
    monkeypatch.setattr(BK, "build_backend", lambda _cfg: teacher)
    return mod, cfg, teacher, out


def test_generate_report_assemble(setup):
    mod, cfg, teacher, out = setup
    mod.stage_generate(cfg, ["all"], list(mod.PASSES))
    for part in ("djinni_new-000", "rjdb-000"):
        for name in mod.PASSES:
            assert (out / "parts" / part / f"{name}.parquet").exists()
    votes = pd.read_parquet(out / "parts" / "rjdb-000" / "votes.parquet")
    assert len(votes) == 4 * 8 and votes["request_seed"].nunique() == 4
    invariant = pd.read_parquet(out / "parts" / "rjdb-000" / "invariant.parquet")
    assert len(invariant) == 4 * 12                         # v1's 12 variants per pair

    # A rerun on the same prompts generates nothing.
    calls = len(teacher.calls)
    mod.stage_generate(cfg, ["all"], list(mod.PASSES))
    assert len(teacher.calls) == calls

    mod.stage_report(cfg)
    report = json.loads((out / "report.json").read_text())
    assert report["vote"]["all"]["anchored"] == 8
    assert report["vote"]["by"]["rjdb"]["anchor_hire_share"] == 0.5

    mod.stage_assemble(cfg)
    # Default: SFT and the on-policy input only; no teacher-biased DPO is written.
    assert not list(out.glob("*dpo*"))
    onpolicy = pd.read_parquet(out / "raw" / "en_invariant.parquet")
    assert list(onpolicy.columns[:18]) == mod.V1_INVARIANT_COLUMNS
    assert set(onpolicy["pool_source"]) == {"djinni_new", "rjdb"}
    cfg["assemble"]["teacher_biased_dpo"] = True
    mod.stage_assemble(cfg)
    new_train = pd.read_parquet(out / "additional_sft_train.parquet")
    new_val = pd.read_parquet(out / "additional_sft_validation.parquet")
    combined = pd.read_parquet(out / "sft_train.parquet")
    dpo_train = pd.read_parquet(out / "teacher_biased_dpo_train.parquet")
    capped = pd.read_parquet(out / "teacher_biased_dpo_capped_train.parquet")
    balanced = pd.read_parquet(out / "teacher_biased_dpo_capped_balanced_train.parquet")
    assert capped.groupby("pair_id").size().max() <= 2
    assert balanced["decision"].value_counts().nunique() == 1      # as many hire as reject
    assert set(balanced["pair_id"]) <= set(capped["pair_id"])
    assert {"anchor_hire_votes", "anchor_unstable", "anchor_changed", "reference_decision",
            "pool_source"} <= set(capped.columns)
    gen = json.loads((out / "generation_report.json").read_text())
    assert gen["splits"]["teacher_biased_dpo_train"]["teacher_biased_dpo_capped_balanced"][
        "decision_contrast_share"] == 1.0
    assert not set(new_train["candidate_id"]) & set(new_val["candidate_id"])
    assert set(new_train["pool_source"]) <= {"djinni_new", "rjdb"}
    # The publish guard's names: new Djinni rows are Djinni.
    assert set(new_train["source"]) <= {"djinni", "rjdb"}
    assert (new_train["reference_decision"] == new_train["decision"]).all()
    # SFT-v3's swap rows come first, unchanged; the new rows follow.
    assert combined["candidate_id"].iloc[0] == "v1-a" and combined["source"].iloc[0] == "djinni"
    assert combined["pool_source"].iloc[0] == "djinni_v1"
    assert len(combined) == 1 + len(new_train)
    assert "v1-b" in set(pd.read_parquet(out / "sft_validation.parquet")["candidate_id"])
    assert len(dpo_train) and (dpo_train["chosen"] != dpo_train["rejected"]).all()
    # The SFT completion carries the vote's verdict, written as JSON, not a fenced raw answer.
    assert new_train["completion"].str.startswith('{"decision"').all()


def test_unknown_part_is_refused(setup):
    mod, cfg, _, _ = setup
    with pytest.raises(SystemExit):
        mod.stage_generate(cfg, ["nope-000"], list(mod.PASSES))


def test_v1_dpo_reads_the_vote_anchor_and_keeps_sft_v3s_split(setup, tmp_path):
    """The flip check uses the vote's verdict; changed pairs take the regenerated targets."""
    mod, cfg, _, out = setup
    key = {"candidate_id": "v1-b", "job_id": "j", "pair_id": "v1-b__j", "cv": "cv",
           "job_description": "jd", "job_position": "", "primary_keyword": "",
           "candidate_years": 0.0, "job_required_years": 0.0, "lang": "en",
           "protected_group": "gender", "protected_attr": "Male", "condition": "explicit",
           "group_label": "gender", "profile": "p", "reference_feedback": "v1 feedback"}
    ok = json.dumps({"decision": "reject", "feedback": "The record lacks the core skills "
                     "the role asks for in its requirements."})
    hire = json.dumps({"decision": "hire", "feedback": "Strong record across every skill "
                       "the role asks for in its requirements."})
    pd.DataFrame([{**key, "reference_decision": "reject", "raw_output": ok}]).to_parquet(
        tmp_path / "inv.parquet")
    pd.DataFrame([{**key, "reference_decision": "hire", "raw_output": hire,
                   "v1_reference_decision": "reject", "seed": 1}]).to_parquet(
        tmp_path / "regen.parquet")
    # v1's biased answer says reject: no flip against v1's reject, a flip against the vote's hire.
    pd.DataFrame([{**key, "reference_decision": "reject", "raw_output": ok}]).to_parquet(
        tmp_path / "biased.parquet")
    pd.DataFrame({"pair_id": ["v1-b__j"], "new_decision": ["hire"],
                  "new_feedback": ["vote feedback"], "changed": [True], "hire_votes": [7],
                  "decided": [8], "source": ["vote"],
                  "unstable": [True]}).to_parquet(tmp_path / "anchors.parquet")
    cfg["v1_dpo"] = {"vote_anchors": str(tmp_path / "anchors.parquet"),
                     "v1_invariant": str(tmp_path / "inv.parquet"),
                     "vote_targets": str(tmp_path / "regen.parquet"),
                     "v1_biased": str(tmp_path / "biased.parquet")}
    mod.stage_v1_dpo(cfg)
    val = pd.read_parquet(out / "teacher_biased_v1_dpo_validation.parquet")   # v1-b: validation
    assert len(val) == 1
    assert len(pd.read_parquet(out / "teacher_biased_v1_dpo_train.parquet")) == 0
    row = val.iloc[0]
    assert json.loads(row["chosen"])["decision"] == "hire"         # the regenerated target
    assert json.loads(row["rejected"])["decision"] == "reject" and row["rejected_flipped_decision"]
    assert row["source"] == "djinni" and row["pool_source"] == "djinni_v1"
    assert row["reference_decision"] == "hire"
    # 7 of 8 is a stable vote; that it overturned v1 is `anchor_changed`, not `anchor_unstable`.
    assert not bool(row["anchor_unstable"]) and bool(row["anchor_changed"])


def test_a_source_that_crosses_a_stop_condition_gets_no_further_part(setup):
    """The pre-registered stop conditions are read in the job, on each source's first part."""
    mod, cfg, _, out = setup
    pool = pd.read_parquet(out / "pool" / "pairs_en.parquet")
    rjdb = pool["source"] == "rjdb"
    pool.loc[rjdb, "part"] = ["rjdb-000", "rjdb-000", "rjdb-001", "rjdb-001"]
    pool.to_parquet(out / "pool" / "pairs_en.parquet", index=False)
    cfg["stop"]["min_invariant_yield"] = 1.01       # no part can meet it
    mod.stage_generate(cfg, ["rjdb-000", "rjdb-001", "djinni_new-000"], list(mod.PASSES))
    check = json.loads((out / "parts" / "rjdb-000" / "stop_check.json").read_text())
    assert check["reasons"] and check["readout"]["invariant_yield"] < 1.01
    assert not (out / "parts" / "rjdb-001").exists()             # skipped
    assert (out / "parts" / "djinni_new-000" / "biased.parquet").exists()   # other sources run
    meta = json.loads((out / "parts" / "rjdb-000" / "votes.meta.json").read_text())
    assert meta["engine_load_seconds"] is not None and meta["pass_sha256"]


def test_balanced_view_balances_each_pool_source_on_its_own():
    """A hire-heavy source must not supply another source's hire side."""
    mod = _load()
    rows = pd.DataFrame({
        "pool_source": ["rjdb"] * 6 + ["djinni_new"] * 6,
        "decision": ["hire"] * 5 + ["reject"] + ["hire"] + ["reject"] * 5,
        "pair_id": [f"p{i}" for i in range(12)],
    })
    out = mod.balanced(rows, seed=0)
    per = out.groupby(["pool_source", "decision"]).size().to_dict()
    assert per == {("djinni_new", "hire"): 1, ("djinni_new", "reject"): 1,
                   ("rjdb", "hire"): 1, ("rjdb", "reject"): 1}


def test_quality_exclusions_drop_contradicting_djinni_cvs_and_rjdb_residue():
    mod = _load()
    cfg = {"assemble": {"exclude": {"djinni_new_extra_markers": True,
                                    "rjdb_template_residue": True}}}
    pairs = pd.DataFrame({
        "pair_id": ["a", "b", "c", "d"],
        "source": ["djinni_new", "djinni_new", "rjdb", "kaggle_linkedin"],
        "cv": ["She led the data team for five years.", "Led the data team for five years.",
               "Education: Details not provided.", "Company Name City , State"],
    })
    excluded, report = mod.quality_exclusions(cfg, pairs)
    assert report["djinni_new_extra_marker"] == 1 and report["rjdb_template_residue"] == 1
    assert excluded == {"a", "c"}            # Kaggle's anonymisation placeholders are kept
    cfg["assemble"]["exclude"] = {}
    assert mod.quality_exclusions(cfg, pairs)[0] == set()


def test_a_job_without_the_biased_pass_still_reads_the_stop_conditions(setup):
    mod, cfg, _, out = setup
    mod.stage_generate(cfg, ["rjdb-000"], ["votes", "invariant"])
    assert not (out / "parts" / "rjdb-000" / "biased.parquet").exists()
    check = json.loads((out / "parts" / "rjdb-000" / "stop_check.json").read_text())
    assert "invariant_yield" in check["readout"] and "biased_yield" not in check["readout"]


def test_a_later_round_draws_new_parts_into_its_own_directory(setup, tmp_path):
    """Round 3 reads the first round's pool and never redraws a part SFT-v3 trained on."""
    mod, cfg, _, out = setup
    pool = pd.read_parquet(out / "pool" / "pairs_en.parquet")
    pool["part"] = pool["source"] + "-" + (pool["order"] // 2).map("{:03d}".format)
    pool.to_parquet(out / "pool" / "pairs_en.parquet", index=False)
    mod.stage_generate(cfg, ["djinni_new-000", "rjdb-000"], ["votes", "invariant"])
    mod.stage_assemble(cfg)
    first_sft = {s: pd.read_parquet(out / f"sft_{s}.parquet") for s in ("train", "validation")}

    later = {**cfg, "artifacts_dir": "artifacts/new-pairs",
             "pool_dir": "artifacts/additional/pool", "earlier_rounds": ["artifacts/additional"]}
    new = tmp_path / "artifacts" / "new-pairs"
    with pytest.raises(SystemExit, match="earlier round"):
        mod.stage_generate(later, ["djinni_new-000"], ["votes", "invariant"])
    with pytest.raises(SystemExit, match="pool"):
        mod.stage_pool(later)
    mod.stage_generate(later, ["all"], ["votes", "invariant"])
    assert sorted(p.name for p in (new / "parts").iterdir()) == ["djinni_new-001", "rjdb-001"]
    mod.stage_assemble(later)
    raw = pd.read_parquet(new / "raw" / "en_invariant.parquet")
    assert set(raw["pair_id"]) == set(pool.loc[pool["part"].str.endswith("-001"), "pair_id"])
    # The first round's outputs are untouched.
    for split, frame in first_sft.items():
        pd.testing.assert_frame_equal(pd.read_parquet(out / f"sft_{split}.parquet"), frame)
    # The split is the pool's: the same seeded draw over every candidate of the pool.
    assert mod.validation_candidates(later, pool) == mod.validation_candidates(cfg, pool)

    # A candidate that reached an earlier round's SFT data stops the assembly.
    leaked = first_sft["train"].copy()
    leaked.loc[0, "candidate_id"] = raw["candidate_id"].iloc[0]
    leaked.to_parquet(out / "sft_train.parquet", index=False)
    with pytest.raises(ValueError, match="candidate"):
        mod.stage_assemble(later)


def test_r2em_drops_rows_the_vote_overturns_and_its_legacy_pool(setup, tmp_path):
    mod, cfg, _, out = setup
    r2 = tmp_path / "r2em"
    r2.mkdir()

    def row(pair, decision, pool):
        other = "reject" if decision == "hire" else "hire"
        return {"prompt": f"p-{pair}-{pool}",
                "chosen": json.dumps({"decision": decision, "feedback": "f"}),
                "rejected": json.dumps({"decision": other, "feedback": "g"}),
                "decision": decision, "lang": "en", "candidate_id": f"c-{pair}",
                "job_id": "j", "pair_id": pair, "source": pool, "reference_decision": decision}

    pd.DataFrame([row("a", "reject", "round1"),           # the vote hires a: conflicting
                  row("b", "hire", "round1"),             # agrees with the vote
                  row("c", "reject", "legacy")]).to_parquet(r2 / "dpo_round2_train.parquet")
    pd.DataFrame([row("b", "hire", "round1")]).to_parquet(r2 / "dpo_round2_validation.parquet")
    pd.DataFrame({"pair_id": ["a", "b", "c"], "new_decision": ["hire", "hire", "reject"],
                  "hire_votes": [7, 8, 0], "decided": [8, 8, 8],
                  "changed": [True, False, False],
                  "source": ["vote"] * 3}).to_parquet(tmp_path / "votes.parquet")
    cfg["r2em"] = {"dir": str(r2), "name": "dpo_round2",
                   "vote_anchors": str(tmp_path / "votes.parquet"),
                   "drop_conflicting": True, "drop_legacy": True}
    mod.stage_r2em(cfg)
    train = pd.read_parquet(out / "r2em_v3_train.parquet")
    assert train["pair_id"].tolist() == ["b"]
    assert train["r2em_pool"].tolist() == ["round1"] and train["source"].tolist() == ["djinni"]
    report = json.loads((out / "r2em_v3_report.json").read_text())
    assert report["train"]["conflicting_with_vote"] == 1 and report["train"]["legacy"] == 1


def _dpo_row(pair, decision, pool_source, candidate, **extra):
    other = "reject" if decision == "hire" else "hire"
    return {"prompt": f"prompt {pair} {candidate}",
            "chosen": json.dumps({"decision": decision,
                                  "feedback": "Six years of Django match the stated stack."}),
            "rejected": json.dumps({"decision": other,
                                    "feedback": "Too little production work for this role."}),
            "decision": decision, "rejected_decision": other, "lang": "en",
            "protected_group": "military_status", "protected_attr": "War veteran",
            "condition": "explicit", "candidate_id": candidate, "job_id": f"j-{pair}",
            "pair_id": pair, "pool_source": pool_source, "reference_decision": decision,
            **extra}


@pytest.fixture
def dpo_v3(setup, tmp_path):
    """r2em_v3 beside the anchors, and an error-mass build of on-policy pairs elsewhere."""
    mod, cfg, _, out = setup
    out.mkdir(parents=True, exist_ok=True)
    v1 = {"r2em_pool": "round1", "source": "djinni", "student_margin": -1.0, "p_wrong": 0.73,
          "anchor_hire_votes": 8, "anchor_decided": 8, "anchor_unstable": False,
          "anchor_changed": False, "anchor_source": "vote"}
    pd.DataFrame([_dpo_row("v1a", "hire", "djinni_v1", "c-v1a", **v1)]).to_parquet(
        out / "r2em_v3_train.parquet")
    pd.DataFrame([_dpo_row("v1b", "reject", "djinni_v1", "c-v1b", **v1)]).to_parquet(
        out / "r2em_v3_validation.parquet")
    pd.DataFrame({"pair_id": ["n1", "n2", "n3"], "hire_votes": [8, 1, 6], "decided": [8, 8, 8],
                  "reference_decision": ["hire", "reject", "hire"],
                  "anchor_source": "vote", "source": ["djinni_new", "rjdb", "djinni_new"]}
                 ).to_parquet(out / "anchors_en.parquet")
    onp = tmp_path / "onpolicy-error-mass"
    onp.mkdir()
    build = {"source": "round1", "student_margin": 0.5}
    train = [_dpo_row("n1", "hire", "djinni_new", "c-n1", p_wrong=0.62, **build),
             _dpo_row("n2", "reject", "rjdb", "c-n2", p_wrong=0.62, **build)]
    validation = [_dpo_row("n3", "hire", "djinni_new", "c-n3", **build)]
    (onp / "dpo_v3_onpolicy_report.json").write_text(json.dumps({
        "parameters": {"balance_by": "pool_source"},
        "balance": {"by": "pool_source", "left_out": ["kaggle_linkedin"],
                    "groups": {"djinni_new": {"after": {}}, "rjdb": {"after": {}},
                               "kaggle_linkedin": {"left_out": "no hire-side pairs"}}},
        "validation_balance": {"n_kept": 1}}))
    cfg["dpo_v3"] = {"onpolicy_dir": str(onp), "onpolicy_name": "dpo_v3_onpolicy"}

    def write(train_rows=train, validation_rows=validation):
        pd.DataFrame(train_rows).to_parquet(onp / "dpo_v3_onpolicy_train.parquet")
        pd.DataFrame(validation_rows).to_parquet(onp / "dpo_v3_onpolicy_validation.parquet")

    write()
    return mod, cfg, out, write, train, validation


def test_dpo_v3_puts_r2em_v3_first_and_keeps_each_parts_student_apart(dpo_v3):
    mod, cfg, out, *_ = dpo_v3
    mod.stage_dpo_v3(cfg)
    train = pd.read_parquet(out / "dpo_v3_train.parquet")
    assert train["dpo_v3_part"].tolist() == ["r2em_v3", "onpolicy_v3", "onpolicy_v3"]
    assert train["pool_source"].tolist() == ["djinni_v1", "djinni_new", "rjdb"]
    assert train["source"].tolist() == ["djinni", "djinni", "rjdb"]
    assert train["r2em_pool"].tolist() == ["round1"] * 3
    # r2em's margins are SFT v1's and keep that name; SFT-v3's are the on-policy rows'.
    assert train["sft_v1_margin"].tolist()[0] == -1.0 and pd.isna(train["sft_v1_margin"][1])
    assert pd.isna(train["student_margin"][0]) and train["student_margin"][1] == 0.5
    assert train["anchor_hire_votes"].tolist() == [8, 8, 1]
    assert (train["reference_decision"] == train["decision"]).all()
    for source in ("djinni_v1", "djinni_new", "rjdb"):
        assert len(pd.read_parquet(out / "dpo_v3_checks" / source / "pairs.parquet")) == 1
    report = json.loads((out / "dpo_v3_report.json").read_text())
    assert report["onpolicy_build"]["left_out"] == ["kaggle_linkedin"]
    assert report["train"]["rows_by_part"] == {"onpolicy_v3": 2, "r2em_v3": 1}
    assert report["train"]["error_mass_sft_v3"] == {"djinni_new hire": 0.62, "rjdb reject": 0.62}
    assert set(report["train"]["direction"]["by_pool_source"]) == {"djinni_v1", "djinni_new",
                                                                   "rjdb"}
    assert len(pd.read_parquet(out / "dpo_v3_validation.parquet")) == 2


def test_dpo_v3_refuses_an_on_policy_pair_against_its_vote(dpo_v3):
    mod, cfg, _, write, train, validation = dpo_v3
    flipped = {**train[0], "decision": "reject", "rejected_decision": "hire"}
    write([flipped, train[1]], validation)
    with pytest.raises(ValueError, match="against their pair's vote anchor"):
        mod.stage_dpo_v3(cfg)


def test_dpo_v3_builds_an_on_policy_only_set_from_a_later_round(dpo_v3, tmp_path):
    """Round 3: no r2em_v3, the vote anchors from several directories, the files renamed."""
    mod, cfg, out, *_ = dpo_v3
    later = tmp_path / "artifacts" / "new-pairs-2"
    later.mkdir(parents=True)
    anchors = pd.read_parquet(out / "anchors_en.parquet")
    anchors[anchors["pair_id"] == "n2"].to_parquet(later / "anchors_en.parquet")
    anchors[anchors["pair_id"] != "n2"].to_parquet(out / "anchors_en.parquet")
    spec = {**cfg["dpo_v3"], "r2em": False, "name": "dpo_v3_5k",
            "anchors_from": [str(out), str(later)]}
    mod.stage_dpo_v3({**cfg, "artifacts_dir": "artifacts/new-pairs-2", "dpo_v3": spec})
    train = pd.read_parquet(later / "dpo_v3_5k_train.parquet")
    assert train["dpo_v3_part"].tolist() == ["onpolicy_v3", "onpolicy_v3"]
    assert train["anchor_hire_votes"].tolist() == [8, 1]
    assert sorted(p.name for p in (later / "dpo_v3_5k_checks").iterdir()) == ["djinni_new",
                                                                             "rjdb"]
    report = json.loads((later / "dpo_v3_5k_report.json").read_text())
    assert report["train"]["rows_by_part"] == {"onpolicy_v3": 2}
    assert not (later / "dpo_v3_train.parquet").exists()
    # A pair anchored in two of the directories is refused.
    anchors.to_parquet(later / "anchors_en.parquet")
    with pytest.raises(ValueError, match="two of"):
        mod.stage_dpo_v3({**cfg, "artifacts_dir": "artifacts/new-pairs-2", "dpo_v3": spec})


def test_dpo_v3_refuses_a_candidate_on_both_sides_of_the_split(dpo_v3):
    mod, cfg, _, write, train, validation = dpo_v3
    write(train, [*validation, {**train[0], "prompt": "another variant of n1"}])
    with pytest.raises(ValueError, match="both sides of the split"):
        mod.stage_dpo_v3(cfg)
