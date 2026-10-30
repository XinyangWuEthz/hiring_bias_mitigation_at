"""GRPO pilot: the prompt set, the rewards, and the weights handed to vLLM."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hiring_bias_mitigation.generation import grpo_prompts as G
from hiring_bias_mitigation.mitigation import grpo


def _rows(n_hire: int, n_reject: int, tipped_hire: int, tipped_reject: int) -> pd.DataFrame:
    rows = []
    for side, n, tipped in (("hire", n_hire, tipped_hire), ("reject", n_reject, tipped_reject)):
        for i in range(n):
            rows.append({"pair_id": f"{side}-p{i // 3}", "candidate_id": f"{side}-c{i // 3}",
                         "job_id": f"j{i % 5}", "lang": "en", "reference_decision": side,
                         "protected_group": ("gender", "religion", "military_status")[i % 3],
                         "protected_attr": f"a{i % 3}",
                         "condition": ("explicit", "implicit")[i % 2],
                         "prompt": f"prompt {side} {i}", "tipped": i < tipped})
    return pd.DataFrame(rows)


# ---- prompt set ---------------------------------------------------------------------------

def test_train_set_is_exactly_half_each_verdict_with_the_asked_tipped_share():
    rows = _rows(n_hire=60, n_reject=400, tipped_hire=30, tipped_reject=100)
    train, plan = G.balanced_train(rows, 40, tipped_share=0.5, seed=1)
    assert train["reference_decision"].value_counts().to_dict() == {"hire": 20, "reject": 20}
    for side in ("hire", "reject"):
        assert train[train["reference_decision"] == side]["tipped"].sum() == 10
        assert plan[side]["drawn_tipped"] == 10 and plan[side]["drawn_untipped"] == 10
    assert not train.duplicated(G.ROW_KEY).any()


def test_a_side_short_of_tipped_rows_fills_with_untipped_ones():
    rows = _rows(n_hire=60, n_reject=60, tipped_hire=3, tipped_reject=30)
    train, plan = G.balanced_train(rows, 40, tipped_share=0.5)
    assert plan["hire"] == {"available": 60, "available_tipped": 3, "drawn_tipped": 3,
                            "drawn_untipped": 17}
    assert (train["reference_decision"] == "hire").sum() == 20


def test_a_side_too_small_for_its_half_refuses():
    with pytest.raises(ValueError, match="hire-reference rows"):
        G.balanced_train(_rows(5, 100, 2, 10), 40, tipped_share=0.5)
    with pytest.raises(ValueError, match="in half"):
        G.balanced_train(_rows(50, 50, 0, 0), 41, tipped_share=0.5)


def test_the_draw_is_seeded():
    rows = _rows(80, 80, 20, 20)
    a, _ = G.balanced_train(rows, 40, 0.5, seed=7)
    b, _ = G.balanced_train(rows, 40, 0.5, seed=7)
    c, _ = G.balanced_train(rows, 40, 0.5, seed=8)
    assert a.equals(b) and not a.equals(c)


def test_held_out_candidates_never_reach_training():
    rows = _rows(60, 60, 10, 10)
    held = {"hire-c0", "reject-c3"}
    train_pool, held_pool = G.split_by_candidate(rows, held)
    assert set(held_pool["candidate_id"]) == held
    assert not set(train_pool["candidate_id"]) & held
    validation = G.balanced_validation(held_pool, 4)
    assert validation["reference_decision"].value_counts().to_dict() == {"hire": 2, "reject": 2}


def test_tipped_flags_are_the_union_of_both_selections():
    rows = _rows(3, 3, 0, 0)
    k8 = pd.DataFrame([{**rows.iloc[0][G.ROW_KEY].to_dict(), "tipped": True},
                       {**rows.iloc[1][G.ROW_KEY].to_dict(), "tipped": False}])
    k32 = pd.DataFrame([{**rows.iloc[4][G.ROW_KEY].to_dict(), "tipped": True}])
    marked = G.mark_tipped(rows, G.tipped_keys(k8, k32))
    assert marked["tipped"].tolist() == [True, False, False, False, True, False]


def test_prompts_attach_by_row_position_and_refuse_a_mismatch():
    rows = _rows(2, 1, 0, 0).drop(columns="prompt")
    prompts = pd.DataFrame({"kind": ["attr_free", "attributed", "attributed", "attributed"],
                            "key": ["hire-p0", "2", "0", "1"],
                            "prompt": ["free", "row two", "row zero", "row one"]})
    assert G.attach_prompts(rows, prompts)["prompt"].tolist() == ["row zero", "row one",
                                                                  "row two"]
    with pytest.raises(ValueError, match="different runs"):
        G.attach_prompts(rows.iloc[:2], prompts)


def _with_pull(rows: pd.DataFrame, rate_tipped: dict, rate_untipped: dict) -> pd.DataFrame:
    """Every row's student hire rate set by verdict and tipped flag."""
    rate = [(rate_tipped if t else rate_untipped)[r]
            for r, t in zip(rows["reference_decision"], rows["tipped"])]
    counts = {p: (round(h * 8), 8) for p, h in zip(rows["prompt"], rate)}
    return G.attach_sft_rates(rows, counts)


def test_student_samples_give_each_row_its_rate_and_pull():
    rows = _rows(2, 2, 0, 0)
    samples = pd.DataFrame({
        "prompt": [rows["prompt"][0]] * 4 + [rows["prompt"][1]] * 2 + [rows["prompt"][2]] * 2
        + [rows["prompt"][3]] * 2,
        "raw_output": [_answer("hire")] * 3 + [_answer("reject"), _answer("hire"),
                                               "no answer", _answer("hire"), _answer("reject"),
                                               _answer("reject"), _answer("reject")]})
    k8 = G.sample_hire_counts(samples)
    k32 = {rows["prompt"][1]: (16, 32)}
    out = G.attach_sft_rates(rows, k32, k8)
    assert out["sft_hire_rate"].tolist() == [0.75, 0.5, 0.5, 0.0]
    assert out["sft_samples"].tolist() == [4, 32, 2, 2]
    # hire rows err by rejecting, reject rows by hiring; pull = sqrt(p(1-p))
    assert out["pull"].tolist() == pytest.approx([0.75 ** 0.5 * 0.5, 0.5, 0.5, 0.0])
    with pytest.raises(KeyError, match="no stored student samples"):
        G.attach_sft_rates(rows, k32)


def test_signal_balance_gives_both_halves_the_same_pull():
    # hire side: few errors per row (pull 0.33 tipped); reject side: many (pull 0.48 tipped)
    rows = _with_pull(_rows(100, 400, 60, 200), {"hire": 7 / 8, "reject": 3 / 8},
                      {"hire": 1.0, "reject": 0.0})
    train, plan = G.signal_balanced_train(rows, 80, seed=3)
    assert train["reference_decision"].value_counts().to_dict() == {"hire": 40, "reject": 40}
    assert plan["hire"]["drawn_tipped"] == 40  # the weaker side takes every tipped row it can
    assert plan["reject"]["drawn_tipped"] < 40
    pulls = train.groupby("reference_decision")["pull"].sum()
    assert abs(pulls["hire"] - pulls["reject"]) <= 0.5  # one reject row's pull
    assert plan["hire"]["pull"] == pytest.approx(pulls["hire"], abs=0.01)
    assert not train.duplicated(G.ROW_KEY).any()


def _mixed_pull(rows: pd.DataFrame, informative_untipped: dict) -> pd.DataFrame:
    """Tipped rows err 3 of 8 times; the first `informative_untipped[side]` untipped rows of a
    side 1 of 8 times; every other row never."""
    rate = []
    for side, group in rows.groupby("reference_decision", sort=False):
        seen = 0
        for tipped in group["tipped"]:
            wrong = 3 / 8 if tipped else (1 / 8 if seen < informative_untipped[side] else 0.0)
            seen += not tipped
            rate.append(1 - wrong if side == "hire" else wrong)
    counts = {p: (round(r * 8), 8) for p, r in zip(rows["prompt"], rate)}
    return G.attach_sft_rates(rows, counts)


def test_pull_fill_takes_rows_with_signal_before_rows_without():
    rows = _mixed_pull(_rows(200, 200, 20, 20), {"hire": 30, "reject": 30})
    random, _ = G.signal_balanced_train(rows, 100, seed=4)
    pulled, plan = G.signal_balanced_train(rows, 100, seed=4, fill="pull")
    assert pulled["reference_decision"].value_counts().to_dict() == {"hire": 50, "reject": 50}
    for side in ("hire", "reject"):
        assert plan[side]["drawn_tipped"] == 20  # attribute-tipped rows still go first
        assert plan[side]["drawn_informative"] == 50
        assert plan[side]["available_informative"] == 50
    assert (pulled["pull"] > 0).sum() == 100 > (random["pull"] > 0).sum()
    assert pulled["pull"].sum() > random["pull"].sum()
    with pytest.raises(ValueError, match="fill"):
        G.signal_balanced_train(rows, 100, fill="best")


def test_informative_half_is_the_smaller_side_with_a_sampled_error():
    rows = _mixed_pull(_rows(200, 200, 20, 20), {"hire": 30, "reject": 70})
    assert G.informative_half(rows) == 50
    train, plan = G.signal_balanced_train(rows, 2 * G.informative_half(rows), fill="pull")
    assert plan["hire"]["drawn_informative"] == 50  # the short side uses every one
    assert (train["pull"] > 0).all()


def test_signal_balance_refuses_a_side_too_small_and_is_seeded():
    rows = _with_pull(_rows(30, 400, 10, 100), {"hire": 0.5, "reject": 0.5},
                      {"hire": 1.0, "reject": 0.0})
    with pytest.raises(ValueError, match="hire-reference rows"):
        G.signal_balanced_train(rows, 80)
    rows = _with_pull(_rows(100, 100, 50, 50), {"hire": 0.75, "reject": 0.5},
                      {"hire": 1.0, "reject": 0.0})
    a, _ = G.signal_balanced_train(rows, 40, seed=5)
    b, _ = G.signal_balanced_train(rows, 40, seed=5)
    assert a.equals(b)


# ---- rewards ------------------------------------------------------------------------------

def _answer(decision: str, feedback: str = "Strong match for the stack and seniority asked") -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def test_decision_reward_is_the_attribute_free_reference_in_both_directions():
    completions = [_answer("hire"), _answer("reject"), _answer("reject"), _answer("hire"),
                   "I cannot decide this.", _answer("Hire")]
    reference = ["hire", "hire", "reject", "reject", "hire", "HIRE"]
    rewards = grpo.decision_reward(completions, reference, lang=["en"] * 6)
    assert rewards == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_decision_reward_reports_the_sampled_hire_rate_per_reference_side():
    seen = {}
    grpo.decision_reward([_answer("hire"), _answer("reject"), _answer("hire")],
                         ["hire", "hire", "reject"], log_metric=seen.__setitem__)
    assert seen == {"sampled_hire_rate_ref_hire": 0.5, "sampled_hire_rate_ref_reject": 1.0}


@pytest.mark.parametrize("completion, ok", [
    (_answer("reject"), True),
    ("  " + _answer("hire") + "\n", True),
    ("```json\n" + _answer("hire") + "\n```", False),
    ("Decision: " + _answer("hire"), False),
    (json.dumps({"decision": "hire", "feedback": "Good fit.", "score": 3}), False),
    (json.dumps({"decision": "hire"}), False),
    (json.dumps({"decision": "hire", "feedback": ["a", "b"]}), False),
    (_answer("hire", "Too short"), False),
    (_answer("hire", "one two three four"), True),
    (_answer("hire", " ".join(["word"] * 45)), True),
    (_answer("hire", " ".join(["word"] * 46)), False),
    ('{"decision": "hire", "feedback": "unterminated', False),
])
def test_format_reward_wants_exactly_the_json_answer(completion, ok):
    assert grpo.format_reward([completion]) == [grpo.FORMAT_REWARD if ok else 0.0]


def test_rewards_ignore_the_extra_columns_trl_passes():
    kwargs = {"prompts": ["p"], "completion_ids": [[1]], "pair_id": ["x"],
              "protected_group": ["gender"], "trainer_state": None, "log_extra": None}
    assert grpo.decision_reward([_answer("hire")], ["hire"], **kwargs) == [1.0]
    assert grpo.format_reward([_answer("hire")], reference_decision=["hire"], **kwargs) == [0.2]


# ---- weights to vLLM ----------------------------------------------------------------------

@pytest.mark.parametrize("name, multimodal, expected", [
    ("model.layers.3.mlp.up_proj.weight", True, "model.language_model.layers.3.mlp.up_proj.weight"),
    ("model.embed_tokens.weight", True, "model.language_model.embed_tokens.weight"),
    ("model.language_model.norm.weight", True, "model.language_model.norm.weight"),
    ("model.visual.blocks.0.attn.qkv.weight", True, "model.visual.blocks.0.attn.qkv.weight"),
    ("lm_head.weight", True, "lm_head.weight"),
    ("model.layers.3.mlp.up_proj.weight", False, "model.layers.3.mlp.up_proj.weight"),
])
def test_text_model_names_become_checkpoint_names(name, multimodal, expected):
    assert grpo.vllm_param_name(name, multimodal) == expected


def _tiny_peft():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from torch import nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.up_proj = nn.Linear(8, 16, bias=True)
            self.norm = nn.LayerNorm(8)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(8, 4, bias=False)

    torch.manual_seed(0)
    model = Tiny().to(torch.bfloat16)
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "up_proj"],
                             lora_dropout=0.0)
    model = peft.get_peft_model(model, config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.5)
    return model


def test_merged_weights_match_peft_merge_and_leave_the_base_untouched():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    pushed = dict(grpo.merged_named_params(model, multimodal=True))

    reference = copy.deepcopy(model).float().merge_and_unload().state_dict()
    expected = {grpo.vllm_param_name(k, True): v for k, v in reference.items()}
    assert set(pushed) == set(expected)
    for name, value in pushed.items():
        assert value.dtype == torch.bfloat16
        assert torch.allclose(value.float(), expected[name].to(torch.bfloat16).float(),
                              atol=1e-2), name
    assert not any("lora_" in n or "base_layer" in n for n in pushed)
    assert all(n.startswith(("model.language_model.", "lm_head.")) for n in pushed)
    lora = "model.language_model.layers.0.q_proj.weight"
    base = before["base_model.model.model.layers.0.q_proj.base_layer.weight"]
    assert not torch.equal(pushed[lora], base)  # the adapter is in what vLLM gets
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} was written"


def test_a_zero_adapter_pushes_the_base_weights_exactly():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
    pushed = dict(grpo.merged_named_params(model))
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        clean = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        assert torch.equal(pushed[clean], param), clean


# ---- decision monitor ---------------------------------------------------------------------

def test_projected_hire_rate_carries_relative_changes_onto_sfts_benchmark_rates():
    sft = {"hire_rate_ref_hire": 1.0, "hire_rate_ref_reject": 0.21875}
    same = grpo.projected_hire_rate(sft, sft, n_side=32)
    assert same["hire_rate_projected"] == pytest.approx(0.333 * 0.658 + 0.667 * 0.091)
    # smoke job 5065891 at step 20: fewer false hires, three fewer true hires of 32
    later = {"hire_rate_ref_hire": 0.90625, "hire_rate_ref_reject": 0.0625}
    out = grpo.projected_hire_rate(later, sft, n_side=32)
    assert out["projected_hire_rate_ref_hire"] == pytest.approx(0.658 * 29.5 / 32.5)
    assert out["projected_hire_rate_ref_reject"] == pytest.approx(0.091 * 2.5 / 7.5)
    assert 0.20 < out["hire_rate_projected"] < 0.23
    # r1-300's own change on the benchmark lands on its own hire rate
    r1 = {"hire_rate_ref_hire": 0.476 / 0.658, "hire_rate_ref_reject": 0.048 / 0.091}
    base = {"hire_rate_ref_hire": 1.0, "hire_rate_ref_reject": 1.0}
    assert grpo.projected_hire_rate(r1, base, n_side=10_000)["hire_rate_projected"] == \
        pytest.approx(0.191, abs=0.002)


def test_the_monitor_projects_onto_its_own_sft_parents_benchmark_rates():
    pytest.importorskip("transformers")
    rows = pd.DataFrame({"reference_decision": ["hire", "reject"]})
    default = grpo._decision_monitor_class()(rows, None, ".", 25, (0.2, 0.4), None)
    assert default.sft_benchmark == grpo.SFT_BENCHMARK
    # SFT-v3 (reports/reference_metrics.csv): 57.73% where GPT-4o hires, 5.266% where it rejects
    v3 = {"hire_rate_ref_hire": 0.5773, "hire_rate_ref_reject": 0.05266}
    monitor = grpo._decision_monitor_class()(rows, None, ".", 25, (0.15, 0.35), None,
                                             sft_benchmark=v3)
    assert monitor.sft_benchmark["reference_hire_share"] == 0.333
    same = {"hire_rate_ref_hire": 1.0, "hire_rate_ref_reject": 0.2}
    out = grpo.projected_hire_rate(same, same, 128, monitor.sft_benchmark)
    assert out["hire_rate_projected"] == pytest.approx(0.333 * 0.5773 + 0.667 * 0.05266)


def test_merged_into_base_runs_the_pushed_weights_and_restores_the_base_bit_for_bit():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    pushed = dict(grpo.merged_named_params(model))
    with grpo.merged_into_base(model), model.disable_adapter():
        for name, module in grpo.lora_layers(model):
            clean = name.removeprefix("base_model.model.") + ".weight"
            assert torch.equal(module.base_layer.weight, pushed[clean])
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} not restored"


def test_merge_retention_sees_a_delta_too_small_for_bf16():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    assert grpo.merge_retention(model)["kept_fraction"] == pytest.approx(1.0, abs=0.05)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.mul_(1e-5)  # far below half a bf16 unit of the base weights
    tiny = grpo.merge_retention(model)
    assert tiny["kept_fraction"] < 0.2 and tiny["unchanged"] > 0.9
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
    assert grpo.merge_retention(model) == {"kept_fraction": 1.0, "unchanged": 1.0}


# ---- margin monitor -----------------------------------------------------------------------

def test_margin_bins_are_left_closed_like_pandas_cut():
    edges = grpo.MARGIN_EDGES
    margins = [-9.0, edges[0], -1.0, 0.0, 0.5, edges[-1], 12.0]
    expected = pd.cut(margins, [-np.inf, *edges, np.inf], right=False, labels=False).tolist()
    assert grpo.margin_bins(margins).tolist() == expected == [0, 1, 2, 4, 4, 7, 7]


def test_the_projection_weights_each_bins_change_by_the_benchmarks_share():
    edges = (-1.0, 1.0)                       # three bins: confident reject, borderline, hire
    share_hire, share_reject = [0.2, 0.3, 0.5], [0.9, 0.1, 0.0]
    parent = np.array([-3.0, -2.0, 0.5, 0.5, 0.2, -0.5, 4.0, 5.0])
    current = parent.copy()
    current[[2, 3]] = -0.1                    # borderline: hires go from 3 of 4 to 1 of 4
    current[6] = -1.0                         # confident hires: 1 of 2 lost
    sft = {"hire_rate_ref_hire": 0.6, "hire_rate_ref_reject": 0.05, "reference_hire_share": 0.25}
    out = grpo.project_from_margins(parent, current, edges, share_hire, share_reject, sft)
    assert [b["n"] for b in out["bins"]] == [2, 4, 2]
    assert [b["delta_hire_share"] for b in out["bins"]] == [0.0, -0.5, -0.5]
    assert out["delta_tpr"] == pytest.approx(0.3 * -0.5 + 0.5 * -0.5)
    assert out["delta_fpr"] == pytest.approx(0.1 * -0.5)
    assert out["projected_tpr"] == pytest.approx(0.6 - 0.4)
    assert out["hire_rate_projected"] == pytest.approx(0.25 * 0.2 + 0.75 * 0.0)
    # A bin no monitor row falls in moves nothing and says how much weight went unseen.
    out = grpo.project_from_margins(parent[:2], current[:2], edges, share_hire, share_reject, sft)
    assert out["delta_tpr"] == 0.0 and out["unseen_ref_hire"] == pytest.approx(0.8)


def test_monitor_rows_cover_every_level_of_the_parents_confidence():
    rates = [0.0] * 50 + [0.05] * 3 + [0.4] * 10 + [0.6] * 10 + [0.95] * 10 + [1.0] * 40
    rows = pd.DataFrame({"sft_hire_rate": rates, "candidate_id": [f"c{i}" for i in range(123)],
                         "reference_decision": ["hire", "reject"] * 61 + ["hire"]})
    picked, plan = G.monitor_rows(rows, per_bin=8, seed=3)
    assert [v["drawn"] for v in plan.values()] == [8, 3, 0, 8, 8, 0, 8, 8]
    assert [v["available"] for v in plan.values()] == [50, 3, 0, 10, 10, 0, 10, 40]
    assert list(plan)[-1] == "[0.99, 1]" and (picked["sft_hire_rate"] == 1.0).sum() == 8
    again, _ = G.monitor_rows(rows, per_bin=8, seed=3)
    assert picked.equals(again) and not picked.equals(G.monitor_rows(rows, 8, seed=4)[0])


class _State:
    def __init__(self, step: int = 0, zero: bool = True):
        self.global_step, self.is_world_process_zero = step, zero


def _margin_monitor(tmp_path, score, stop=0.08):
    pytest.importorskip("transformers")
    settings = {"edges": [-1.0, 1.0], "share_ref_hire": [0.2, 0.3, 0.5],
                "share_ref_reject": [0.9, 0.1, 0.0], "reference_hire_share": 0.25,
                "tpr_drop_stop": stop}
    return grpo._margin_monitor_class()(["p1", "p2", "p3", "p4"], score, tmp_path, 25, settings,
                                        {"hire_rate_ref_hire": 0.6, "hire_rate_ref_reject": 0.05})


def test_the_margin_monitor_scores_the_parent_once_and_a_resumed_run_reads_it(tmp_path):
    calls = []
    monitor = _margin_monitor(tmp_path, lambda p: calls.append(p) or [-2.0, 0.5, 0.5, 3.0])
    monitor.on_train_begin(None, _State(0), None)
    assert calls == [["p1", "p2", "p3", "p4"]]
    assert monitor.sft_benchmark["reference_hire_share"] == 0.25
    saved = json.loads((tmp_path / "margin_monitor_parent.json").read_text())["margins"]
    resumed = _margin_monitor(tmp_path, lambda p: pytest.fail("the parent is not rescored"))
    resumed.on_train_begin(None, _State(200), None)
    assert resumed.parent.tolist() == saved
    other = tmp_path / "fresh"
    other.mkdir()
    with pytest.raises(RuntimeError, match="before any update"):
        _margin_monitor(other, lambda p: [0.0] * 4).on_train_begin(None, _State(200), None)


def test_the_margin_monitor_stops_when_the_projected_tpr_falls_past_its_limit(tmp_path):
    readings = iter([[-2.0, 0.5, 0.5, 3.0], [-2.0, 0.5, 0.3, 3.0], [-2.0, -0.5, 0.5, 3.0]])
    monitor = _margin_monitor(tmp_path, lambda p: next(readings), stop=0.10)
    monitor.on_train_begin(None, _State(0), None)
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    monitor.on_step_end(None, _State(10), control)          # not a reading step
    monitor.on_step_end(None, _State(25), control)          # unchanged decisions
    assert not monitor.aborted and not control.should_training_stop
    monitor.on_step_end(None, _State(50), control)          # one of two borderline hires lost
    lines = [json.loads(x) for x in (tmp_path / "margin_monitor.jsonl").read_text().splitlines()]
    assert [x["step"] for x in lines] == [25, 50]
    assert lines[1]["delta_tpr"] == pytest.approx(0.3 * -0.5) and lines[1]["stop"]
    assert monitor.aborted and control.should_save and control.should_training_stop
    assert monitor.reason == "aborted: projected TPR fell more than 10 pp"
