"""DAPO: overlong shaping, dynamic sampling and the verdict balance, the replay into TRL, the
stop, and the config against GRPO's."""

from __future__ import annotations

import inspect
import json
import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hiring_bias_mitigation.mitigation import dapo
from hiring_bias_mitigation.mitigation import grpo as G
from hiring_bias_mitigation.mitigation import training_common as C

REPO = Path(__file__).resolve().parents[1]
DAPO_CONFIG = REPO / "configs/mitigation/dapo/qwen3.5-9b_en_only_dapo_v3_all_gh200.yaml"
GRPO_CONFIG = REPO / "configs/mitigation/grpo/qwen3.5-9b_en_only_grpo_v3_all_gh200.yaml"


def _answer(decision: str) -> str:
    return json.dumps({"decision": decision, "feedback": "Solid experience matching the role."})


def _group(prompt: str, side: str, size: int) -> list[dict]:
    return [{"prompt": prompt, "reference_decision": side, "lang": "en", "pair_id": prompt,
             "protected_attr": "a", "condition": "explicit"} for _ in range(size)]


def _generated(rows: list[dict], script: dict[str, list[str]]) -> tuple:
    """A `_generate` output: each prompt's group answers its script, in order (again from the
    start for a prompt that comes back in the batch)."""
    completions, seen = [], defaultdict(int)
    for row in rows:
        answers = script[row["prompt"]]
        completions.append(_answer(answers[seen[row["prompt"]] % len(answers)]))
        seen[row["prompt"]] += 1
    n = len(rows)
    return ([[1, 2]] * n, [[7] * (3 + i % 2) for i in range(n)], None, completions,
            [[-0.1]] * n, {}, None, [])


# ---- overlong shaping ---------------------------------------------------------------------

@pytest.mark.parametrize("length, expected", [(40, 0.0), (80, 0.0), (84, -0.25), (96, -1.0),
                                              (120, -1.0)])
def test_overlong_penalty_is_zero_then_falls_linearly_to_the_penalty(length, expected):
    assert dapo.overlong_penalty(length, 96, 16) == pytest.approx(expected)
    assert dapo.overlong_penalty(length, 96, 0) == 0.0


def test_overlong_reward_reads_completion_lengths_and_ignores_trls_other_columns():
    reward = dapo.make_overlong_reward(96, 16, 0.5)
    assert reward.__name__ == "overlong_reward"
    out = reward(completion_ids=[[1] * 38, [1] * 88, [1] * 96], completions=["a", "b", "c"],
                 reference_decision=["hire"] * 3, trainer_state=None)
    assert out == pytest.approx([0.0, -0.25, -0.5])


# ---- dynamic sampling ---------------------------------------------------------------------

def test_a_group_is_informative_only_when_its_decisions_disagree():
    assert dapo.informative([True, False, True])
    assert not dapo.informative([True] * 4)
    assert not dapo.informative([False] * 4)


def test_quotas_are_half_per_verdict_and_refuse_an_odd_count():
    assert dapo.quotas(8, balance=True) == {"hire": 4, "reject": 4}
    assert dapo.quotas(3, balance=False) == {dapo.ANY: 3}
    with pytest.raises(ValueError, match="evenly"):
        dapo.quotas(3, balance=True)


def test_rows_must_arrive_as_whole_contiguous_groups():
    dapo.check_groups(_group("a", "hire", 3) + _group("b", "reject", 3), 3)
    with pytest.raises(ValueError, match="whole groups"):
        dapo.check_groups(_group("a", "hire", 4), 3)
    with pytest.raises(ValueError, match="one prompt"):
        dapo.check_groups(_group("a", "hire", 2) + _group("b", "hire", 4), 3)


def test_take_rows_keeps_the_chosen_completions_and_concat_joins_rounds():
    rows = _group("a", "hire", 2) + _group("b", "reject", 2)
    generated = _generated(rows, {"a": ["hire", "reject"], "b": ["reject", "hire"]})
    part = dapo.take_rows(generated, [2, 3])
    assert part[3] == generated[3][2:] and part[1] == generated[1][2:]
    assert part[2] is None and part[5] == {} and part[7] == []
    joined = dapo.concat_rollouts([part, dapo.take_rows(generated, [0])])
    assert joined[3] == [generated[3][2], generated[3][3], generated[3][0]]
    with pytest.raises(ValueError, match="rollout_func"):
        dapo.take_rows((*generated[:5], {"x": [1, 2, 3, 4]}, *generated[6:]), [0])
    with pytest.raises(ValueError, match="TRL changed"):
        dapo.take_rows(generated[:7], [0])


def test_generate_fields_are_what_the_installed_trl_returns():
    from trl import GRPOTrainer

    source = inspect.getsource(GRPOTrainer._generate)
    returned = source.rstrip().splitlines()[-1].strip()
    assert returned == "return " + ", ".join(dapo.GENERATE_FIELDS)


def _sampler(script, refills):
    """generate/decide/refill callables for dynamic_sample, with a log of what each saw."""
    calls = {"generate": [], "refill": 0}
    queue = list(refills)

    def generate(batch):
        calls["generate"].append([r["prompt"] for r in batch[::2]])
        return _generated(batch, script)

    def decide(batch, generated):
        return [json.loads(c)["decision"] for c in generated[3]]

    def refill():
        calls["refill"] += 1
        return queue.pop(0)

    return generate, decide, refill, calls


def test_dynamic_sampling_keeps_disagreeing_groups_per_verdict_until_the_quota_is_full():
    script = {"h1": ["hire", "reject"], "h_ok": ["hire", "hire"], "r1": ["hire", "reject"],
              "r_bad": ["hire", "hire"], "h2": ["reject", "hire"], "r2": ["reject", "hire"],
              "h3": ["hire", "reject"], "r3": ["reject", "hire"]}
    first = (_group("h1", "hire", 2) + _group("h_ok", "hire", 2) + _group("r1", "reject", 2)
             + _group("r_bad", "reject", 2))
    refill = (_group("h3", "hire", 2) + _group("h2", "hire", 2) + _group("r2", "reject", 2)
              + _group("r3", "reject", 2))
    generate, decide, refill_fn, calls = _sampler(script, [refill])
    shortfalls = []
    rows, generated, n_filler, stats = dapo.dynamic_sample(
        first, generate, decide, refill_fn, 2, {"hire": 2, "reject": 2}, 8,
        lambda n: shortfalls.append(n) or n)
    assert [r["prompt"] for r in rows[::2]] == ["h1", "r1", "h3", "r2"]
    assert [json.loads(c)["decision"] for c in generated[3]] == \
        ["hire", "reject", "hire", "reject", "hire", "reject", "reject", "hire"]
    assert n_filler == 0 and shortfalls == [2, 0] and calls["refill"] == 1
    assert stats["rounds"] == 2 and stats["probed"] == {"hire": 4, "reject": 4}
    assert stats["informative"] == {"hire": 3, "reject": 3}
    assert stats["kept"] == {"hire": 2, "reject": 2}
    # The first probe is the main sampler's draw, read as GRPO would have read it.
    assert stats["first"] == {"rows": {"hire": 4, "reject": 4},
                              "correct": {"hire": 3, "reject": 1},
                              "hire": {"hire": 3, "reject": 3},
                              "groups": {"hire": 2, "reject": 2},
                              "informative": {"hire": 1, "reject": 1}}


def test_a_rank_with_its_quota_keeps_probing_while_another_rank_is_short():
    script = {p: ["hire", "reject"] for p in ("h1", "r1", "h2", "r2")}
    first = _group("h1", "hire", 2) + _group("r1", "reject", 2)
    generate, decide, refill_fn, calls = _sampler(
        script, [_group("h2", "hire", 2) + _group("r2", "reject", 2)])
    others = iter([3, 0])
    rows, _, n_filler, stats = dapo.dynamic_sample(
        first, generate, decide, refill_fn, 2, {"hire": 1, "reject": 1}, 8,
        lambda n: max(n, next(others)))
    assert [r["prompt"] for r in rows[::2]] == ["h1", "r1"] and n_filler == 0
    assert calls["refill"] == 1 and stats["rounds"] == 2


def test_after_the_last_round_dropped_groups_make_up_the_quota_at_the_end():
    script = {"h1": ["hire", "reject"], "h_ok": ["hire", "hire"], "r_ok": ["reject", "reject"],
              "r_bad": ["hire", "hire"], "h_ok2": ["hire", "hire"], "r_ok2": ["reject", "reject"]}
    first = _group("h1", "hire", 2) + _group("h_ok", "hire", 2) + _group("r_ok", "reject", 2)
    refill = _group("h_ok2", "hire", 2) + _group("r_bad", "reject", 2) + \
        _group("r_ok2", "reject", 2)
    generate, decide, refill_fn, calls = _sampler(script, [refill])
    rows, generated, n_filler, stats = dapo.dynamic_sample(
        first, generate, decide, refill_fn, 2, {dapo.ANY: 3}, 2, lambda n: n)
    assert n_filler == 2 and stats["rounds"] == 2 and calls["refill"] == 1
    assert [r["prompt"] for r in rows[::2]] == ["h1", "h_ok", "r_ok"]
    assert len(generated[3]) == len(rows) == 6


@pytest.mark.parametrize("kept_by_rank, expected", [
    # Both verdicts as full as their quota: everything trains.
    ([{"hire": 4, "reject": 4}] * 2, [{"hire": 4, "reject": 4}] * 2),
    # Hire is scarcer over all ranks (5): reject is cut to 5, from the first ranks on.
    ([{"hire": 4, "reject": 4}, {"hire": 1, "reject": 4}],
     [{"hire": 4, "reject": 4}, {"hire": 1, "reject": 1}]),
    # One verdict dry everywhere: nothing trains.
    ([{"hire": 0, "reject": 4}, {"hire": 0, "reject": 3}],
     [{"hire": 0, "reject": 0}, {"hire": 0, "reject": 0}]),
])
def test_the_verdicts_train_equally_many_groups_over_all_ranks(kept_by_rank, expected):
    got = [dapo.balanced_allowance(kept_by_rank, r) for r in range(len(kept_by_rank))]
    assert got == expected
    assert sum(a["hire"] for a in got) == sum(a["reject"] for a in got)


def test_trained_groups_cut_the_later_kept_groups_of_a_verdict_and_never_train_filler():
    sides = ["hire", "reject", "reject", "hire", "reject", "hire"]
    assert dapo.trained_groups(sides, 4, None) == [True, True, True, True, False, False]
    assert dapo.trained_groups(sides, 4, {"hire": 2, "reject": 1}) == \
        [True, True, False, True, False, False]


def test_the_fill_window_reads_a_mean_and_patience_zero_never_stops():
    window = dapo.FillWindow(patience=3, min_fill=0.5)
    # Means over the last three: -, -, 0.6, 0.6, 0.27.
    assert [window.add(f) for f in (0.0, 1.0, 0.8, 0.0, 0.0)] == [False] * 4 + [True]
    off = dapo.FillWindow(patience=0, min_fill=0.5)
    assert not any(off.add(0.0) for _ in range(5))


def test_refill_order_is_shared_by_the_ranks_and_disjoint_between_them():
    a, b = dapo.RefillOrder(10, seed=5), dapo.RefillOrder(10, seed=5)
    for _ in range(3):  # rounds: the two ranks stay in step and never share a prompt
        rank0, rank1 = a.take(3, rank=0, world=2), b.take(3, rank=1, world=2)
        assert len(rank0) == len(rank1) == 3 and not set(rank0) & set(rank1)
    order = dapo.RefillOrder(10, seed=5)
    first_pass = [i for _ in range(5) for i in order.take(2)]
    assert sorted(first_pass) == list(range(10))
    assert dapo.RefillOrder(10, seed=6).take(10) != dapo.RefillOrder(10, seed=5).take(10)


# ---- the trainer: replay into TRL -----------------------------------------------------------

class _Accelerator:
    device = "cpu"
    is_main_process = True

    def __init__(self, rank: int = 0, world: int = 1):
        self.process_index = rank
        self.num_processes = world

    def gather(self, tensor):
        return tensor if tensor.dim() else tensor.reshape(1)


def _bare_trainer(cls, tmp_path, **settings):
    trainer = object.__new__(cls)
    settings = {**dapo.DYNAMIC_SAMPLING_DEFAULTS, "dynamic_sampling_max_rounds": 2,
                "dynamic_sampling_patience": 2, **settings}
    trainer.__dict__.update(
        dynamic_sampling=settings, stop_reason=None, _replay=None, _refill_order=None,
        _fill_window=dapo.FillWindow(int(settings["dynamic_sampling_patience"]),
                                     float(settings["dynamic_sampling_min_fill"])),
        num_generations=2, accelerator=_Accelerator(), model=SimpleNamespace(training=True),
        state=SimpleNamespace(global_step=7),
        args=SimpleNamespace(seed=42, output_dir=str(tmp_path)),
        _metrics={"train": defaultdict(list)})
    return trainer


@pytest.fixture
def fake_trl(monkeypatch, tmp_path):
    """DAPOTrainer whose parent (TRL's GRPOTrainer) generation and scoring are fakes: scoring
    calls `self._generate` the way TRL does and returns a batch of masks and advantages."""
    import torch
    from trl import GRPOTrainer

    script, scored = {}, []

    def fake_generate(self, prompts):
        return _generated([{"prompt": p} for p in prompts], script)

    def fake_score(self, inputs):
        generated = self._generate([r["prompt"] for r in inputs])
        scored.append(([r["prompt"] for r in inputs], generated[3]))
        n = len(inputs)
        return {"completion_mask": torch.ones(n, 3, dtype=torch.long),
                "advantages": torch.ones(n), "num_items_in_batch": torch.tensor(n * 3)}

    monkeypatch.setattr(GRPOTrainer, "_generate", fake_generate)
    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", fake_score)
    cls = dapo._dapo_trainer_class()
    return SimpleNamespace(cls=cls, script=script, scored=scored, out=tmp_path,
                           make=lambda **s: _bare_trainer(cls, tmp_path, **s))


def test_only_kept_rollouts_are_scored_and_trl_receives_exactly_them(fake_trl):
    trainer = fake_trl.make()
    fake_trl.script.update({"h1": ["hire", "reject"], "h_ok": ["hire", "hire"],
                            "r1": ["reject", "hire"], "r_bad": ["hire", "hire"],
                            "h2": ["reject", "hire"], "r2": ["hire", "reject"]})
    trainer._refill = lambda n_groups: (_group("h2", "hire", 2) + _group("r2", "reject", 2)
                                        + _group("h_ok", "hire", 2) + _group("r_bad", "reject", 2))
    first = (_group("h1", "hire", 2) + _group("h_ok", "hire", 2) + _group("r1", "reject", 2)
             + _group("r_bad", "reject", 2))
    output = trainer._generate_and_score_completions(first)
    prompts, completions = fake_trl.scored[0]
    assert prompts[::2] == ["h1", "r1", "h2", "r2"]
    assert [json.loads(c)["decision"] for c in completions] == \
        ["hire", "reject", "reject", "hire", "reject", "hire", "hire", "reject"]
    assert trainer._replay is None
    assert int(output["completion_mask"].sum()) == 24 and int(output["num_items_in_batch"]) == 24
    metrics = trainer._metrics["train"]
    assert metrics["dapo/rounds"] == [2.0] and metrics["dapo/fill"] == [1.0]
    assert metrics["dapo/informative_rate_ref_hire"] == [0.5]
    assert metrics["dapo/probe_zero_variance"] == [0.5]
    assert metrics["dapo/probe_accuracy"] == [0.5]
    assert metrics["dapo/probe_hire_rate_ref_hire"] == [0.75]
    assert metrics["dapo/probe_hire_rate_ref_reject"] == [0.75]
    record = json.loads((fake_trl.out / "dynamic_sampling.jsonl").read_text().splitlines()[0])
    assert record["step"] == 7 and record["kept_hire"] == 2 and record["trained_reject"] == 2
    assert record["trained_rows"] == ["h1|a|explicit", "r1|a|explicit", "h2|a|explicit",
                                      "r2|a|explicit"]


def test_a_partly_filled_batch_masks_only_its_filler(fake_trl):
    trainer = fake_trl.make(dynamic_sampling_balance=False)
    fake_trl.script.update({"h1": ["hire", "reject"], "r1": ["reject", "hire"],
                            "h2": ["reject", "hire"], "h_ok": ["hire", "hire"]})
    batch = (_group("h1", "hire", 2) + _group("r1", "reject", 2) + _group("h2", "hire", 2)
             + _group("h_ok", "hire", 2))
    trainer._refill = lambda n_groups: _group("h_ok", "hire", 2) * 4
    output = trainer._generate_and_score_completions(batch)
    assert output["completion_mask"][:6].sum() == 18 and output["completion_mask"][6:].sum() == 0
    assert output["advantages"][:6].tolist() == [1.0] * 6 and output["advantages"][6:].sum() == 0
    assert int(output["num_items_in_batch"]) == 18
    assert trainer._metrics["train"]["dapo/fill"] == [0.75]


def test_a_dry_verdict_trains_nothing_and_stops_the_run(fake_trl):
    """The per-verdict quota is only a cap: without the balance cut, the reject groups here
    would train alone while the fill read 0.5."""
    trainer = fake_trl.make()
    fake_trl.script.update({"h_ok": ["hire", "hire"], "r1": ["reject", "hire"],
                            "r2": ["hire", "reject"]})
    batch = (_group("h_ok", "hire", 2) + _group("r1", "reject", 2) + _group("h_ok", "hire", 2)
             + _group("r2", "reject", 2))
    trainer._refill = lambda n_groups: batch
    output = trainer._generate_and_score_completions(batch)
    assert int(output["completion_mask"].sum()) == 0 and int(output["num_items_in_batch"]) == 0
    assert float(output["advantages"].abs().sum()) == 0.0
    assert trainer._metrics["train"]["dapo/kept_ref_reject"] == [2]
    assert trainer.stop_reason is None
    trainer._generate_and_score_completions(batch)
    assert trainer.stop_reason and "2 rollouts" in trainer.stop_reason
    assert trainer._metrics["train"]["dapo/fill"] == [0.0, 0.0]
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    fake_trl.cls.StopWhenDry(trainer).on_step_end(None, None, control)
    assert control.should_save and control.should_training_stop


def test_evaluation_skips_dynamic_sampling(fake_trl):
    trainer = fake_trl.make()
    fake_trl.script.update({"h_ok": ["hire", "hire"]})
    trainer.model.training = False
    output = trainer._generate_and_score_completions(_group("h_ok", "hire", 2))
    assert int(output["completion_mask"].sum()) == 6
    assert "dapo/fill" not in trainer._metrics["train"]


def test_refill_gives_each_rank_its_own_whole_groups_of_dataset_rows(fake_trl):
    from datasets import Dataset

    data = Dataset.from_list([_group(f"p{i}", "hire", 1)[0] for i in range(6)])
    batches = []
    for rank in (0, 1):
        trainer = fake_trl.make()
        trainer.train_dataset, trainer.accelerator = data, _Accelerator(rank, world=2)
        batches.append(trainer._refill(2))
    for batch in batches:
        dapo.check_groups(batch, 2)
        assert len(batch) == 4 and set(batch[0]) == set(data.column_names)
    assert not {r["prompt"] for r in batches[0]} & {r["prompt"] for r in batches[1]}


# ---- training: the hooks in grpo.train, the manifest, and the stop on resume ----------------

def _fake_training(monkeypatch, tmp_path):
    """grpo.train with the model, data, vLLM and W&B replaced, and a trainer class that records
    how it was built."""
    built = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            built.update(kwargs)
            self.model = kwargs["model"]
            self.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
            self.vllm_generation = None
            self.args = kwargs["args"]
            self.num_generations = self.args.num_generations
            self.use_vllm = self.args.use_vllm
            self.vllm_importance_sampling_correction = self.args.vllm_importance_sampling_correction
            self.vllm_importance_sampling_mode = self.args.vllm_importance_sampling_mode
            self.vllm_importance_sampling_clip_max = self.args.vllm_importance_sampling_clip_max
            self.vllm_importance_sampling_clip_min = self.args.vllm_importance_sampling_clip_min
            self.tools = self.environments = None

        def add_callback(self, callback):
            built.setdefault("callbacks", []).append(callback)

        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            raise AssertionError("this wiring test must not evaluate rewards")

        def _generate_and_score_completions(self, inputs):
            raise AssertionError("this wiring test must not generate completions")

        def log(self, logs, start_time=None):
            raise AssertionError("this wiring test must not execute trainer logging")

        def train(self, resume_from_checkpoint=None):
            built["resume"] = resume_from_checkpoint
            built["count_diagnostics_installed"] = self._count_diagnostics_installed
            assert self._calculate_rewards.__func__ is not FakeTrainer._calculate_rewards
            assert self.log.__func__ is not FakeTrainer.log

        def save_model(self, path):
            pass

    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    monkeypatch.setattr(C, "wandb_init", lambda cfg: None)
    monkeypatch.setattr(C, "wandb_resume", lambda cfg, out: None)
    monkeypatch.setattr(C, "resolve_run_dir", lambda cfg: tmp_path)
    monkeypatch.setattr(C, "load_generated_dataset", lambda cfg, kind: {"train": ["p"] * 32})
    monkeypatch.setattr(C, "load_model_and_tokenizer", lambda cfg: ("model", tokenizer))
    monkeypatch.setattr(C, "peft_config", lambda cfg: None)
    monkeypatch.setattr(C, "memory_guard_callbacks", lambda cfg: [])
    monkeypatch.setattr(C, "cost_callbacks", lambda: [])
    monkeypatch.setattr(G, "render_prompts", lambda ds, tok, kw: ds)
    monkeypatch.setattr(G, "install_weight_sync", lambda gen, policy: None)
    monkeypatch.setattr(G, "patch_vllm_engine_kwargs", lambda extra: None)
    monkeypatch.setattr(dapo, "_dapo_trainer_class", lambda: FakeTrainer)
    return FakeTrainer, built


def _cpu_cfg(path: Path, tmp_path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    cfg.update(dtype="float32", use_wandb=False, output_dir=str(tmp_path), data={})
    return cfg


def test_grpo_train_defaults_are_grpos_and_dapo_records_its_objective(monkeypatch, tmp_path):
    import trl

    trainer_cls, built = _fake_training(monkeypatch, tmp_path)
    monkeypatch.setattr(trl, "GRPOTrainer", trainer_cls)
    G.train(_cpu_cfg(GRPO_CONFIG, tmp_path))
    manifest = json.loads((tmp_path / "run_config.json").read_text())
    assert built["reward_funcs"] == G.REWARD_FUNCS and "dynamic_sampling" not in built
    assert manifest["objective"] == "grpo" and manifest["mitigation_family"] == "grpo"
    assert manifest["status"] == "done" and manifest["beta"] == 0.04
    assert built["count_diagnostics_installed"]
    assert manifest["rollout_diagnostics"]["basic_count_weighted_logging"] is True
    assert manifest["rollout_diagnostics"]["detailed"] is False
    assert manifest["sft_checkpoint"] == "outputs/sft/qwen3.5-9b_en_only_sft_v3_additional"

    built.clear()
    check = dapo.check_config  # as on four GPUs; WORLD_SIZE itself would start torch.distributed
    monkeypatch.setattr(dapo, "check_config", lambda cfg: check(cfg, world_size=4))
    dapo.train(_cpu_cfg(DAPO_CONFIG, tmp_path))
    manifest = json.loads((tmp_path / "run_config.json").read_text())
    names = [f.__name__ for f in built["reward_funcs"]]
    assert names == ["decision_reward", "format_reward", "overlong_reward"]
    assert built["dynamic_sampling"]["groups_per_rank"] == 8
    assert manifest["objective"] == "dapo" and manifest["mitigation_family"] == "grpo"
    assert manifest["beta"] == 0.0 and manifest["epsilon_high"] == 0.28
    assert manifest["loss_type"] == "dapo" and manifest["steps_per_generation"] == 16
    assert manifest["sft_checkpoint"] == "outputs/sft/qwen3.5-9b_en_only_sft_v3_additional"
    assert manifest["status"] == "done"
    assert built["count_diagnostics_installed"]
    assert manifest["rollout_diagnostics"]["basic_count_weighted_logging"] is True
    assert manifest["rollout_diagnostics"]["detailed"] is False


def test_grpo_train_adds_the_margin_monitor_on_the_builds_monitor_rows(monkeypatch, tmp_path):
    import pandas as pd
    import trl

    trainer_cls, built = _fake_training(monkeypatch, tmp_path)
    monkeypatch.setattr(trl, "GRPOTrainer", trainer_cls)
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"prompt": ["a", "b", "c"]}).to_parquet(data / "grpo_monitor.parquet")
    cfg = _cpu_cfg(GRPO_CONFIG, tmp_path)
    cfg.update(data={"artifacts_dir": str(data)}, margin_monitor={
        "edges": list(G.MARGIN_EDGES), "share_ref_hire": [0.125] * 8,
        "share_ref_reject": [0.125] * 8, "tpr_drop_stop": 0.08})
    G.train(cfg)
    monitors = [c for c in built["callbacks"] if type(c).__name__ == "MarginMonitorCallback"]
    assert len(monitors) == 1 and monitors[0].prompts == ["a", "b", "c"]
    assert monitors[0].every == 25 and monitors[0].tpr_drop_stop == 0.08
    assert built["count_diagnostics_installed"]


def test_detailed_diagnostics_attach_both_splits_before_render_and_record_sources(
    monkeypatch, tmp_path
):
    import trl
    from datasets import Dataset

    from hiring_bias_mitigation.mitigation import (
        decision_eval,
        grpo_anchor_metadata,
        grpo_diagnostics,
    )

    trainer_cls, built = _fake_training(monkeypatch, tmp_path)
    trainer_cls.__name__ = "GRPOTrainer"
    monkeypatch.setattr(trl, "GRPOTrainer", trainer_cls)
    splits = {
        "train": Dataset.from_dict({"prompt": ["train"] * 32,
                                    "reference_decision": ["hire"] * 32}),
        "validation": Dataset.from_dict({"prompt": ["validation"] * 16,
                                         "reference_decision": ["reject"] * 16}),
    }
    monkeypatch.setattr(C, "load_generated_dataset", lambda cfg, kind: splits)
    events = []

    def attach(dataset, *, split, resolve_path):
        events.append("attach_" + split)
        votes, draws = (5, 8) if split == "train" else (4, 6)
        dataset = dataset.add_column("prompt_id", [f"{split}-{i}" for i in range(len(dataset))])
        dataset = dataset.add_column("winning_votes", [votes] * len(dataset))
        dataset = dataset.add_column("decided_draws", [draws] * len(dataset))
        return dataset, {"split": split, "rows": len(dataset), "source_sha256": split + "-sha"}

    def render(dataset, tokenizer, kwargs):
        events.append("render")
        assert {"prompt_id", "winning_votes", "decided_draws"} <= set(dataset.column_names)
        return dataset

    original_install = grpo_diagnostics.install

    def record_install(trainer, **kwargs):
        built["diagnostics_strata"] = kwargs["strata"]
        return original_install(trainer, **kwargs)

    monkeypatch.setattr(grpo_anchor_metadata, "attach_anchor_metadata", attach)
    monkeypatch.setattr(grpo_diagnostics, "install", record_install)
    monkeypatch.setattr(G, "render_prompts", render)
    monkeypatch.setattr(decision_eval, "eval_rows", lambda rows, count, seed: rows)
    monkeypatch.setattr(G, "_sync_validation_class", lambda: lambda *args, **kwargs: object())
    monkeypatch.setattr(G, "_decision_monitor_class", lambda: lambda *args, **kwargs:
                        SimpleNamespace(aborted=False))
    cfg = _cpu_cfg(GRPO_CONFIG, tmp_path)
    cfg["rollout_diagnostics"] = {"enabled": True}
    G.train(cfg)
    assert events == ["attach_train", "attach_validation", "render"]
    assert built["diagnostics_strata"] == ["hire/5_of_8", "reject/4_of_6"]
    assert splits["validation"]["prompt_id"][0] == "validation-0"
    assert built["count_diagnostics_installed"]
    manifest = json.loads((tmp_path / "run_config.json").read_text())
    metadata = manifest["rollout_diagnostics"]["anchor_metadata"]
    assert metadata["schema"] == "grpo-anchor-metadata-splits-v1"
    assert metadata["splits"] == {
        "train": {"split": "train", "rows": 32, "source_sha256": "train-sha"},
        "validation": {"split": "validation", "rows": 16, "source_sha256": "validation-sha"},
    }


@pytest.mark.parametrize("status, refused", [("done", False), ("training", False),
                                             ("stopped: dynamic sampling filled under", True),
                                             ("aborted: hire rate left the band", True)])
def test_a_run_that_stopped_itself_is_not_resumed_past_the_stop(tmp_path, status, refused):
    (tmp_path / "run_config.json").write_text(json.dumps({"status": status}))
    cfg = {"resume": True, "output_dir": str(tmp_path)}
    if refused:
        with pytest.raises(RuntimeError, match="would train past that stop"):
            dapo.refuse_stopped_resume(cfg)
    else:
        dapo.refuse_stopped_resume(cfg)
    dapo.refuse_stopped_resume({**cfg, "resume": False})


# ---- config -------------------------------------------------------------------------------

def _cfg(**over) -> dict:
    cfg = {"loss_type": "dapo", "epsilon": 0.2, "epsilon_high": 0.28, "max_completion_length": 96,
           "num_generations": 16, "per_device_train_batch_size": 8,
           "gradient_accumulation_steps": 8, "steps_per_generation": 16, "save_steps": 24}
    cfg.update(over)
    return cfg


def test_the_shipped_config_passes_with_eight_groups_per_rank_on_four_gpus():
    cfg = yaml.safe_load(DAPO_CONFIG.read_text())
    settings = dapo.check_config(cfg, world_size=4)
    assert settings["groups_per_rank"] == 8 and settings["dynamic_sampling"]
    assert dapo.quotas(settings["groups_per_rank"], True) == {"hire": 4, "reject": 4}


@pytest.mark.parametrize("over, message", [
    ({"loss_type": "grpo"}, "loss_type: dapo"),
    ({"epsilon_high": 0.1}, "below epsilon"),
    ({"max_completion_length": None}, "max_completion_length"),
    ({"overlong_buffer": 96}, "overlong_buffer"),
    ({"steps_per_generation": 3}, "whole groups"),
    ({"steps_per_generation": 6}, "evenly"),
    ({"generation_batch_size": 510}, "split evenly"),
    ({"steps_per_generation": 12, "num_generations": 8}, "whole number of optimizer steps"),
    ({"save_steps": 25}, "multiple of the 2 optimizer steps"),
    ({"dynamic_sampling_min_fill": 1.5}, "between 0 and 1"),
    ({"dynamic_sampling_patience": -1}, "at least 0"),
])
def test_config_refuses_what_breaks_dapo(over, message):
    cfg = _cfg(**over)
    if over.get("max_completion_length", 0) is None:
        cfg.pop("max_completion_length")
    with pytest.raises(ValueError, match=message):
        dapo.check_config(cfg, world_size=4)


def test_config_warns_when_clip_higher_cannot_bind(caplog, monkeypatch):
    # The project's loggers do not propagate; caplog listens on the root logger.
    monkeypatch.setattr(dapo.log, "propagate", True)
    with caplog.at_level(logging.WARNING):
        dapo.check_config(_cfg(steps_per_generation=8, save_steps=25), world_size=4)
    assert "clip-higher cannot bind" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        dapo.check_config(_cfg(), world_size=4)
    assert "clip-higher" not in caplog.text


def test_dapo_v3_all_differs_from_grpo_v3_all_only_in_the_declared_settings():
    """Same prompts, start, learning rate, prompts per optimizer step and monitor; the
    differences are DAPO's, the verdict balance, and checkpoints on rollout boundaries
    (docs/DAPO_V3.md)."""
    a = yaml.safe_load(GRPO_CONFIG.read_text())
    b = yaml.safe_load(DAPO_CONFIG.read_text())
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ == {
        "stage", "objective", "output_dir", "wandb_run_name", "save_steps",
        "loss_type", "beta", "epsilon", "epsilon_high", "steps_per_generation",
        "mask_truncated_completions", "overlong_buffer", "overlong_penalty",
        *dapo.DYNAMIC_SAMPLING_DEFAULTS}
    assert a["data_config"] == b["data_config"] and a["learning_rate"] == b["learning_rate"]
