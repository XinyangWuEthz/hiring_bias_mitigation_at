"""Opt-in, state-preserving GRPO continuation with a post-restore LR intervention.

Only constant-with-warmup schedules past warmup are supported. HF restores the
optimizer and scheduler before on_train_begin, so changing TrainingArguments alone
would silently keep the checkpoint's old learning rate. No checkpoint is rewritten.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from ..utils.logging import get_logger

log = get_logger(__name__)

REQUIRED_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    *(f"rng_state_{rank}.pth" for rank in range(4)),
)
PRESERVED_ARGUMENTS = (
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "num_generations",
    "num_iterations",
    "steps_per_generation",
    "generation_batch_size",
    "seed",
    "data_seed",
    "ignore_data_skip",
    "shuffle_dataset",
    "optim",
    "beta",
    "loss_type",
    "scale_rewards",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "epsilon",
    "epsilon_high",
    "delta",
    "reward_weights",
    "importance_sampling_level",
    "mask_truncated_completions",
    "cast_lm_head_to_fp32",
    "max_completion_length",
    "disable_dropout",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_epsilon",
    "max_grad_norm",
    "bf16",
    "fp16",
    "lr_scheduler_type",
    "warmup_steps",
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "ddp_find_unused_parameters",
    "use_vllm",
    "vllm_mode",
    "vllm_tensor_parallel_size",
    "vllm_gpu_memory_utilization",
    "vllm_max_model_length",
    "vllm_enable_sleep_mode",
    "vllm_importance_sampling_correction",
)
RNG_LIMITATION = (
    "HF per-rank Python/NumPy/Torch RNG states and prompt data position are restored; "
    "the vLLM internal sampling RNG was not checkpointed and restarts with each engine. "
    "This is not a bit-identical stochastic continuation of the historical run."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    return getattr(value, "value", value)


def load_contract(settings: dict, cfg: dict, output_dir, resolve_path) -> dict:
    """Validate source identities before allocating a model, without touching source files."""
    source = Path(resolve_path(settings["source_checkpoint"])).resolve()
    output = Path(output_dir).resolve()
    if output == source.parent or output.is_relative_to(source.parent):
        raise ValueError("continuation output must be outside the original run directory")
    if int(settings["world_size"]) != 4:
        raise ValueError("continuation requires the original four-rank topology")
    step, stop = int(settings["source_step"]), int(settings["max_steps"])
    lr, old_lr = float(settings["target_learning_rate"]), float(settings["source_learning_rate"])
    if not (math.isfinite(lr) and math.isfinite(old_lr) and 0 < lr < old_lr and stop > step):
        raise ValueError("continuation requires a positive lower LR and a later absolute max_steps")
    if int(cfg.get("max_steps", -1)) != stop or float(cfg["learning_rate"]) != lr:
        raise ValueError("continuation LR/max_steps must match the training configuration")
    expected = settings["source_files_sha256"]
    if not set(REQUIRED_FILES).issubset(expected):
        raise ValueError(
            "continuation source hashes must cover adapter, optimizer, scheduler, "
            "state, args and four RNG files"
        )
    for name, digest in expected.items():
        if Path(name).name != name or len(digest) != 64:
            raise ValueError("invalid continuation source file identity")
        if not (source / name).is_file() or sha256(source / name) != digest:
            raise ValueError(f"continuation source hash mismatch: {name}")
    state = json.loads((source / "trainer_state.json").read_text())
    if int(state["global_step"]) != step:
        raise ValueError("continuation source global_step differs from the frozen step")
    identity = {**settings, "source_checkpoint": str(source), "source_files_sha256": dict(expected)}
    binding = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "settings": identity,
        "identity_sha256": binding,
        "source": source,
        "output": output,
        "source_step": step,
        "max_steps": stop,
        "target_learning_rate": lr,
        "source_learning_rate": old_lr,
        "rng_limitation": RNG_LIMITATION,
    }


def choose_resume(contract: dict, latest_checkpoint: str | None) -> str:
    """Resume this pilot if it exists, otherwise read the frozen external checkpoint."""
    output = contract["output"]
    manifest = output / "run_config.json"
    if latest_checkpoint is None:
        # A half-written run must not silently restart and overwrite its evidence.
        if any(output.iterdir()):
            raise ValueError("continuation output is not empty and has no completed checkpoint")
        return str(contract["source"])
    latest = Path(latest_checkpoint).resolve()
    if latest.parent != output or not manifest.is_file():
        raise ValueError("pilot resume requires its own checkpoint and run manifest")
    existing = json.loads(manifest.read_text()).get("continuation", {})
    if existing.get("identity_sha256") != contract["identity_sha256"]:
        raise ValueError("pilot continuation identity differs from its run manifest")
    state = json.loads((latest / "trainer_state.json").read_text())
    if not contract["source_step"] < int(state["global_step"]) < contract["max_steps"]:
        raise ValueError("pilot resume step is outside the unfinished continuation window")
    for name in REQUIRED_FILES:
        if not (latest / name).is_file():
            raise ValueError(f"pilot resume lacks {name}")
    return str(latest)


def initialize_tracking_after_resume(accelerator, initialize_tracking, broadcast_error) -> None:
    """Write tracking metadata only after every rank has checked the fresh output.

    Call after Trainer construction, when Accelerator has initialized distributed
    synchronization. W&B first sets up its run in on_train_begin, so the run ID is
    still available before that callback while no rank can fail its empty check
    on another rank's new wandb_run_id file.
    """
    accelerator.wait_for_everyone()
    error = None
    if accelerator.is_main_process:
        try:
            initialize_tracking()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = broadcast_error(error)
    if error is not None:
        raise RuntimeError(f"continuation tracking initialization failed: {error}")
    accelerator.wait_for_everyone()


def validate_arguments(contract: dict, args, source_args) -> dict:
    """Only the declared LR, run length and monitoring/checkpoint settings may change."""
    changes = {}
    for name in PRESERVED_ARGUMENTS:
        before, after = _plain(getattr(source_args, name, None)), _plain(getattr(args, name, None))
        if before != after:
            changes[name] = {"source": before, "current": after}
    if changes:
        raise ValueError(f"continuation changed preserved training arguments: {changes}")
    if _plain(args.lr_scheduler_type) != "constant_with_warmup":
        raise ValueError("continuation only supports constant_with_warmup")
    if args.ignore_data_skip or contract["source_step"] <= args.warmup_steps:
        raise ValueError("continuation must preserve data skip and start after warmup")
    if (
        int(args.max_steps) != contract["max_steps"]
        or float(args.learning_rate) != contract["target_learning_rate"]
    ):
        raise ValueError("resolved arguments differ from continuation LR/max_steps")
    if int(args.world_size) != int(contract["settings"]["world_size"]):
        raise ValueError("continuation must use the original four ranks")
    if args.num_iterations != 1 or args.steps_per_generation != args.gradient_accumulation_steps:
        raise ValueError("continuation requires a complete rollout at each optimizer checkpoint")
    return {name: _plain(getattr(args, name, None)) for name in PRESERVED_ARGUMENTS}


def load_source_arguments(contract: dict):
    # The local, user-owned training_args file is hash-verified by load_contract first.
    import torch

    return torch.load(
        contract["source"] / "training_args.bin", map_location="cpu", weights_only=False
    )


def _scheduler(scheduler):
    from torch.optim.lr_scheduler import LambdaLR

    scheduler = getattr(scheduler, "scheduler", scheduler)
    if not isinstance(scheduler, LambdaLR):
        raise ValueError("continuation requires a restored LambdaLR scheduler")
    return scheduler


def apply_learning_rate(
    contract: dict, optimizer, scheduler, step: int, expected_old_lr: float
) -> dict:
    """Change only LR fields after HF restoration; never reset moments or step counters."""
    scheduler = _scheduler(scheduler)
    if scheduler.last_epoch != step:
        raise ValueError("restored scheduler global_step mismatch")
    if not optimizer.state:
        raise ValueError("continuation optimizer state was not restored")
    if any(
        parameter not in optimizer.state
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ):
        raise ValueError("continuation optimizer state is incomplete")
    steps = sorted({float(value["step"]) for value in optimizer.state.values()})
    if steps != [float(step)]:
        raise ValueError(f"restored Adam steps differ from global_step: {steps}")
    for value in optimizer.state.values():
        if "exp_avg" not in value or "exp_avg_sq" not in value:
            raise ValueError("continuation requires restored Adam first and second moments")
    before = {
        "optimizer_lrs": [float(group["lr"]) for group in optimizer.param_groups],
        "optimizer_initial_lrs": [float(group["initial_lr"]) for group in optimizer.param_groups],
        "scheduler_base_lrs": list(scheduler.base_lrs),
        "scheduler_last_lrs": list(scheduler.get_last_lr()),
        "scheduler_last_epoch": scheduler.last_epoch,
        "scheduler_step_count": scheduler._step_count,
    }
    for key in (
        "optimizer_lrs",
        "optimizer_initial_lrs",
        "scheduler_base_lrs",
        "scheduler_last_lrs",
    ):
        if not all(
            math.isclose(value, expected_old_lr, rel_tol=1e-12, abs_tol=0) for value in before[key]
        ):
            raise ValueError(f"restored {key} does not match expected LR {expected_old_lr}")
    if not all(
        fn(point) == 1.0
        for fn in scheduler.lr_lambdas
        for point in (step, step + 1, contract["max_steps"])
    ):
        raise ValueError("continuation scheduler is not on its constant post-warmup plateau")
    lr = contract["target_learning_rate"]
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["initial_lr"] = lr
    scheduler.base_lrs = [lr] * len(optimizer.param_groups)
    scheduler._last_lr = [lr] * len(optimizer.param_groups)
    return {
        "before": before,
        "after_learning_rate": lr,
        "optimizer_state_count": len(optimizer.state),
        "optimizer_steps": steps,
        "global_step": step,
        "warmup_restarted": False,
        "optimizer_moments_preserved": True,
    }


def callback_class():
    from transformers import TrainerCallback

    class ContinuationCallback(TrainerCallback):
        def __init__(self, contract, resume_path, source_args):
            self.contract, self.resume_path, self.source_args = (
                contract,
                Path(resume_path),
                source_args,
            )
            self.start_step = int(
                json.loads((self.resume_path / "trainer_state.json").read_text())["global_step"]
            )

        def _write(self, args, event):
            path = self.contract["output"] / f"continuation_rank{args.process_index}.jsonl"
            with path.open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

        def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            preserved = validate_arguments(self.contract, args, self.source_args)
            if int(state.global_step) != self.start_step:
                raise ValueError("Trainer did not restore the selected continuation checkpoint")
            # TrainerState.load_from_json happens after compute_steps and restores the
            # historical save_steps=25. TrainingArguments(save_steps=5) alone is ignored.
            prior_save_steps = state.save_steps
            if int(args.save_steps) != args.save_steps or args.save_steps <= 0:
                raise ValueError("continuation requires an absolute positive save_steps")
            state.save_steps = int(args.save_steps)
            expected = (
                self.contract["source_learning_rate"]
                if self.resume_path == self.contract["source"]
                else self.contract["target_learning_rate"]
            )
            receipt = apply_learning_rate(
                self.contract, optimizer, lr_scheduler, self.start_step, expected
            )
            self._write(
                args,
                {
                    "event": "restored_and_lr_set",
                    **receipt,
                    "resume_from_checkpoint": str(self.resume_path),
                    "source_identity_sha256": self.contract["identity_sha256"],
                    "source_files_sha256": self.contract["settings"]["source_files_sha256"],
                    "preserved_arguments": preserved,
                    "restored_save_steps": prior_save_steps,
                    "actual_save_steps": state.save_steps,
                    "rng_limitation": RNG_LIMITATION,
                    "data_skip_microbatches": self.start_step * args.gradient_accumulation_steps,
                },
            )
            log.info(
                "continuation rank %d: restored step %d; LR %.8g -> %.8g; Adam moments retained",
                args.process_index,
                state.global_step,
                expected,
                self.contract["target_learning_rate"],
            )

        def _check_lr(self, optimizer, lr_scheduler):
            lr = self.contract["target_learning_rate"]
            scheduler = _scheduler(lr_scheduler)
            values = [group["lr"] for group in optimizer.param_groups] + list(
                scheduler.get_last_lr()
            )
            if not all(math.isclose(value, lr, rel_tol=1e-12, abs_tol=0) for value in values):
                raise RuntimeError(f"continuation LR changed unexpectedly: {values}")
            return [float(group["lr"]) for group in optimizer.param_groups]

        def on_step_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            if state.save_steps != int(args.save_steps):
                raise RuntimeError("continuation save cadence changed unexpectedly")
            lrs = self._check_lr(optimizer, lr_scheduler)
            if state.global_step < self.start_step + 3:
                self._write(
                    args,
                    {
                        "event": "before_optimizer_update",
                        "update_step": state.global_step + 1,
                        "actual_learning_rates": lrs,
                    },
                )

        def on_step_end(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
            lrs = self._check_lr(optimizer, lr_scheduler)
            if state.global_step <= self.start_step + 3 or state.global_step % 5 == 0:
                self._write(
                    args,
                    {
                        "event": "after_optimizer_update",
                        "global_step": state.global_step,
                        "actual_learning_rates": lrs,
                        "scheduler_last_epoch": _scheduler(lr_scheduler).last_epoch,
                    },
                )

    return ContinuationCallback
