"""CPU checks for restored-state LR intervention, isolation and fail-closed resume."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from hiring_bias_mitigation.mitigation import grpo_continuation as G


def contract(tmp_path):
    source = tmp_path / "historical" / "checkpoint-425"
    output = tmp_path / "pilot"
    source.mkdir(parents=True)
    output.mkdir()
    for name in G.REQUIRED_FILES:
        (source / name).write_text("saved-state")
    (source / "trainer_state.json").write_text(json.dumps({"global_step": 425}))
    settings = {
        "source_checkpoint": str(source),
        "source_step": 425,
        "max_steps": 500,
        "world_size": 4,
        "target_learning_rate": 3e-6,
        "source_learning_rate": 3e-5,
        "source_files_sha256": {name: G.sha256(source / name) for name in G.REQUIRED_FILES},
    }
    cfg = {"max_steps": 500, "learning_rate": 3e-6}
    return G.load_contract(settings, cfg, output, lambda path: path), cfg


def args():
    return SimpleNamespace(
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        max_steps=500,
        learning_rate=3e-6,
        world_size=4,
        ignore_data_skip=False,
        num_iterations=1,
        steps_per_generation=8,
        gradient_accumulation_steps=8,
        process_index=0,
        save_steps=5,
    )


def restored_optimizer():
    parameters = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(3))]
    optimizer = torch.optim.AdamW(
        [{"params": [parameters[0]]}, {"params": [parameters[1]]}], lr=3e-5
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, step / 5))
    for _ in range(425):
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    optimizer_state, scheduler_state = (
        copy.deepcopy(optimizer.state_dict()),
        copy.deepcopy(scheduler.state_dict()),
    )
    new_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    resumed = torch.optim.AdamW(
        [{"params": [new_parameters[0]]}, {"params": [new_parameters[1]]}], lr=3e-6
    )
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed, lambda step: min(1.0, step / 5))
    resumed.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    return resumed, resumed_scheduler


def test_restored_checkpoint_overrides_config_lr_and_intervention_preserves_moments(tmp_path):
    c, _ = contract(tmp_path)
    optimizer, scheduler = restored_optimizer()
    assert [group["lr"] for group in optimizer.param_groups] == [3e-5, 3e-5]
    saved = copy.deepcopy(optimizer.state_dict())
    receipt = G.apply_learning_rate(c, optimizer, scheduler, 425, 3e-5)
    assert receipt["before"]["scheduler_last_epoch"] == 425
    assert scheduler.last_epoch == 425 and scheduler._step_count == 426
    for index, state in optimizer.state_dict()["state"].items():
        for name in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(state[name], saved["state"][index][name])
    for expected_step in (426, 427, 428):
        for group in optimizer.param_groups:
            assert group["lr"] == group["initial_lr"] == 3e-6
            for parameter in group["params"]:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
        assert scheduler.last_epoch == expected_step
        assert scheduler.get_last_lr() == [3e-6, 3e-6]


def test_callback_records_actual_first_update_and_rejects_lr_reversion(tmp_path):
    c, _ = contract(tmp_path)
    optimizer, scheduler = restored_optimizer()
    training_args = args()
    callback = G.callback_class()(c, c["source"], copy.copy(training_args))
    state = SimpleNamespace(global_step=425, save_steps=25)
    callback.on_train_begin(training_args, state, None, optimizer=optimizer, lr_scheduler=scheduler)
    callback.on_step_begin(training_args, state, None, optimizer=optimizer, lr_scheduler=scheduler)
    events = [
        json.loads(line)
        for line in (c["output"] / "continuation_rank0.jsonl").read_text().splitlines()
    ]
    assert events[1]["update_step"] == 426
    assert events[1]["actual_learning_rates"] == [3e-6, 3e-6]
    assert events[0]["data_skip_microbatches"] == 3400
    assert events[0]["restored_save_steps"] == 25
    assert state.save_steps == events[0]["actual_save_steps"] == 5
    assert "vLLM internal sampling RNG" in events[0]["rng_limitation"]
    optimizer.param_groups[0]["lr"] = 3e-5
    with pytest.raises(RuntimeError, match="LR changed"):
        callback.on_step_begin(
            training_args, state, None, optimizer=optimizer, lr_scheduler=scheduler
        )


def test_resumed_hf_state_uses_five_step_checkpoint_cadence(tmp_path):
    from transformers import TrainerControl, TrainerState
    from transformers.trainer_callback import DefaultFlowCallback
    from transformers.trainer_utils import IntervalStrategy, SaveStrategy

    c, _ = contract(tmp_path)
    optimizer, scheduler = restored_optimizer()
    training_args = args()
    training_args.save_strategy = SaveStrategy.STEPS
    training_args.logging_strategy = IntervalStrategy.NO
    training_args.eval_strategy = IntervalStrategy.NO
    state = TrainerState(global_step=425, max_steps=500, save_steps=25)
    control = TrainerControl()
    callback = G.callback_class()(c, c["source"], copy.copy(training_args))
    callback.on_train_begin(
        training_args, state, control, optimizer=optimizer, lr_scheduler=scheduler
    )
    state.global_step = 430
    control = DefaultFlowCallback().on_step_end(training_args, state, control)
    assert control.should_save and state.save_steps == 5


@pytest.mark.parametrize("name", ["rng_state_2.pth", "optimizer.pt", "adapter_model.safetensors"])
def test_mutated_or_missing_source_fails_closed(tmp_path, name):
    c, cfg = contract(tmp_path)
    (c["source"] / name).unlink()
    with pytest.raises(ValueError, match="hash mismatch"):
        G.load_contract(c["settings"], cfg, c["output"], lambda path: path)


def test_no_source_write_and_new_run_directory_required(tmp_path):
    c, cfg = contract(tmp_path)
    assert G.choose_resume(c, None) == str(c["source"])
    assert all(
        G.sha256(c["source"] / name) == digest
        for name, digest in c["settings"]["source_files_sha256"].items()
    )
    with pytest.raises(ValueError, match="outside the original"):
        G.load_contract(c["settings"], cfg, c["source"].parent / "new", lambda path: path)
    (c["output"] / "partial.bin").write_text("partial")
    with pytest.raises(ValueError, match="not empty"):
        G.choose_resume(c, None)


def test_resume_only_matching_unfinished_pilot_with_complete_rng(tmp_path):
    c, _ = contract(tmp_path)
    checkpoint = c["output"] / "checkpoint-430"
    checkpoint.mkdir()
    for name in G.REQUIRED_FILES:
        (checkpoint / name).write_text("pilot-state")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 430}))
    (c["output"] / "run_config.json").write_text(
        json.dumps({"continuation": {"identity_sha256": c["identity_sha256"]}})
    )
    assert G.choose_resume(c, str(checkpoint)) == str(checkpoint)
    (checkpoint / "rng_state_3.pth").unlink()
    with pytest.raises(ValueError, match="rng_state_3"):
        G.choose_resume(c, str(checkpoint))


@pytest.mark.parametrize(
    "field,value",
    [
        ("ignore_data_skip", True),
        ("gradient_accumulation_steps", 4),
        ("world_size", 1),
        ("max_steps", 75),
        ("learning_rate", 3e-5),
    ],
)
def test_incorrect_resume_arguments_fail_closed(tmp_path, field, value):
    c, _ = contract(tmp_path)
    current, source = args(), args()
    setattr(current, field, value)
    with pytest.raises(ValueError):
        G.validate_arguments(c, current, source)


def test_missing_adam_state_or_wrong_scheduler_step_rejected(tmp_path):
    c, _ = contract(tmp_path)
    optimizer, scheduler = restored_optimizer()
    scheduler.last_epoch = 424
    with pytest.raises(ValueError, match="global_step mismatch"):
        G.apply_learning_rate(c, optimizer, scheduler, 425, 3e-5)
    scheduler.last_epoch = 425
    optimizer.state.clear()
    with pytest.raises(ValueError, match="state was not restored"):
        G.apply_learning_rate(c, optimizer, scheduler, 425, 3e-5)


def test_nonconstant_scheduler_cannot_be_mistaken_for_lr_only_intervention(tmp_path):
    c, _ = contract(tmp_path)
    optimizer, scheduler = restored_optimizer()
    scheduler.lr_lambdas = [lambda step: 0.9] * len(optimizer.param_groups)
    with pytest.raises(ValueError, match="post-warmup plateau"):
        G.apply_learning_rate(c, optimizer, scheduler, 425, 3e-5)
