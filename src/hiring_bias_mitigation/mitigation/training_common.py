"""Shared plumbing for the SFT and preference-optimisation runs.

Both stages train the same three target models on the same generated dataset and differ only
in the objective, so everything except the trainer lives here: model loading, LoRA setup,
dataset loading, the run manifest, and the memory arithmetic that decides whether a config
can run at all.

Memory on this machine. The DGX Spark GB10 has 128 GB of unified LPDDR5X shared between CPU
and GPU, which is generous for a 12B model and useless if you assume it behaves like 128 GB
of HBM -- bandwidth is ~273 GB/s, so anything compute-bound runs, anything bandwidth-bound
crawls. A full fine-tune needs roughly 8 bytes/param (bf16 weights + grads + fp32 Adam
moments, with the master copy), so:

    4B  full FT ~  32 GB   -- fits, and is the only target we full-fine-tune
    12B full FT ~  96 GB   -- fits on paper, leaves nothing for activations; use LoRA
    12B LoRA    ~  26 GB   -- weights only, plus a few hundred MB of adapter state

The configs encode that: `qwen3.5-4b` full fine-tunes, the two 12B models are LoRA.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..utils.config import require_output_root, resolve_output_path
from ..utils.logging import get_logger

log = get_logger(__name__)

DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]


@dataclass
class TrainArtifacts:
    output_dir: Path
    train_rows: int
    eval_rows: int


def load_generated_dataset(cfg: dict, kind: str):
    """Loads a training split written by scripts/generate_training_data.py.

    `kind` is "sft", "dpo" or "kto", and names the parquet pair to read. The KTO split holds
    the same responses as the DPO one, unpaired into (prompt, completion, label).

    The dataset config names a directory of parquet files rather
    than a Hub repo by default, because the generation run is local and publishing is opt-in.
    """
    import pandas as pd
    from datasets import Dataset

    data_cfg = cfg["data"]
    if data_cfg.get("hub_repo"):
        from datasets import load_dataset

        log.info("loading %s split from the Hub: %s", kind, data_cfg["hub_repo"])
        return load_dataset(data_cfg["hub_repo"], name=kind)

    root = Path(resolve_output_path(data_cfg["artifacts_dir"]))
    splits = {}
    for split in ("train", "validation"):
        path = root / f"{kind}_{split}.parquet"
        if not path.exists():
            if split == "validation":
                continue
            raise FileNotFoundError(
                f"{path} is missing. Run scripts/generate_training_data.py with "
                f"{data_cfg.get('_config_path', 'the matching generation config')} first."
            )
        frame = pd.read_parquet(path)
        frame = apply_data_filters(frame, data_cfg)
        # Evaluation runs every `eval_steps`, so its cost multiplies across the run: on DPO a
        # pass over 976 validation pairs took 35 minutes. A fixed, seeded subsample keeps the
        # curve comparable between evaluations at a fraction of the time.
        eval_limit = cfg.get("eval_limit_rows")
        if split == "validation" and eval_limit and len(frame) > int(eval_limit):
            frame = frame.sample(n=int(eval_limit), random_state=int(cfg.get("seed", 42)))
        splits[split] = Dataset.from_pandas(frame, preserve_index=False)
        log.info("%s/%s: %d rows", kind, split, len(frame))
    return splits


def apply_data_filters(frame, data_cfg: dict):
    """Language / group / condition subsetting, so one generated dataset serves many runs.

    This is how a per-language or per-group mitigation experiment is expressed without
    regenerating data: an experiment that mitigates only military status in Ukrainian trains
    on the Ukrainian military-status slice of the same artifacts.
    """
    languages = data_cfg.get("languages")
    if languages:
        frame = frame[frame["lang"].isin(languages)]
    groups = data_cfg.get("protected_groups")
    if groups:
        frame = frame[frame["protected_group"].isin(groups)]
    conditions = data_cfg.get("conditions")
    if conditions:
        frame = frame[frame["condition"].isin(conditions)]
    limit = data_cfg.get("limit_rows")
    if limit:
        if data_cfg.get("longest_rows"):
            # For a smoke run, the first rows prove nothing: whether a configuration fits in
            # memory is decided by the longest sequences in the set, and those are not at the
            # top of the file. Sorting by length makes the smoke test the worst case it is
            # meant to catch.
            # SFT rows carry a completion; preference rows carry a chosen and a rejected
            # response, and DPO puts both through the model on every step.
            lengths = frame["prompt"].str.len()
            for column in ("completion", "chosen", "rejected"):
                if column in frame.columns:
                    lengths = lengths + frame[column].str.len()
            frame = frame.loc[lengths.sort_values(ascending=False).index]
        frame = frame.head(int(limit))
    return frame.reset_index(drop=True)


def load_model_and_tokenizer(cfg: dict, for_training: bool = True):
    import torch
    from transformers import AutoTokenizer

    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"dtype": dtype, "trust_remote_code": True}
    if cfg.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    model = load_merged(cfg["base_model"], **kwargs)
    if for_training:
        model.config.use_cache = False
    return model, tokenizer


def load_merged(base_model: str, **kwargs):
    """`base_model` as weights: a plain model as it is, an adapter merged into its base.

    A LoRA run saves only the adapter, so `from_pretrained` on that directory has no weights to
    read. DPO continuing from an SFT run has to load the base, apply the adapter and merge it
    down -- the SFT behaviour becomes the starting point, and the DPO stage is then free to fit
    a fresh adapter on top of it.

    The adapter is merged with every adapter it was itself trained on top of, parents first
    (merge.adapter_chain), as the audits merge it. An iterative DPO round continues from a DPO
    checkpoint, whose adapter_config names the plain base: merging that adapter alone would
    start the round from base + DPO, without SFT.
    """
    from transformers import AutoModelForCausalLM

    from .merge import adapter_chain, merge_chain

    _, adapter = _resolve_adapter(base_model)
    if adapter is None:
        return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    chain = adapter_chain(Path(adapter))
    source, _ = _resolve_adapter(str(chain[0]))
    log.info("merging %s into %s", " then ".join(map(str, chain)), source)
    return merge_chain(AutoModelForCausalLM.from_pretrained(source, **kwargs), chain)


def resolve_sft_checkpoint(spec: str) -> str:
    """A local run directory under $HBM_OUTPUT_ROOT, or a Hub adapter repo id, as a local path.

    The published adapters (`Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en`, ...) are the
    final SFT artefacts, and a DPO run that continues from one should name it directly rather
    than depend on a copy someone placed on the drive. A Hub id is fetched into the HF cache
    (adapter files only, never the base weights) and handed to the same local-directory logic
    `_resolve_adapter` already applies, so the adapter is merged into its base at load time.
    """
    import re

    local = Path(resolve_output_path(spec))
    if local.exists():
        return str(local)
    if re.fullmatch(r"[\w.-]+/[\w.-]+", spec):
        import huggingface_hub

        try:
            # The tokenizer files too: dpo.main hands this folder to load_model_and_tokenizer
            # as base_model, which reads the tokenizer from it.
            snapshot = huggingface_hub.snapshot_download(
                spec, allow_patterns=["adapter_config.json", "adapter_model*", "tokenizer*",
                                      "special_tokens_map.json", "chat_template.jinja",
                                      "README.md", "run_config.json"],
            )
        except Exception as exc:  # the hub raises a dozen types; none is recoverable here
            raise FileNotFoundError(
                f"sft_checkpoint {spec!r} is neither a directory under the output root nor a "
                f"Hub adapter that could be fetched ({exc})"
            ) from exc
        if (Path(snapshot) / "adapter_config.json").is_file():
            log.info("sft_checkpoint %s fetched from the Hub into %s", spec, snapshot)
            return snapshot
        raise FileNotFoundError(f"sft_checkpoint {spec!r}: the Hub repo has no adapter_config.json")
    raise FileNotFoundError(
        f"sft_checkpoint {local} not found -- run the matching configs/mitigation/sft/ config "
        "first, name a Hub adapter repo, or clear `sft_checkpoint` to run preference "
        "optimisation directly on the base model."
    )


def _resolve_adapter(base_model: str) -> tuple[str, str | None]:
    """Splits a path into (weights to load, adapter to merge).

    Returns the input unchanged for a plain model. For a PEFT output directory -- one holding
    `adapter_config.json` and no model weights -- returns the base model it was trained from
    plus the adapter itself.
    """
    path = Path(base_model)
    config = path / "adapter_config.json"
    if not config.is_file():
        return base_model, None

    payload = json.loads(config.read_text(encoding="utf-8"))
    parent = payload.get("base_model_name_or_path")
    if not parent:
        raise ValueError(
            f"{config} has no base_model_name_or_path, so the adapter cannot be placed on "
            "its base model. Re-run the training that wrote it."
        )
    return parent, str(path)


def peft_config(cfg: dict):
    """Builds a LoraConfig, or None for a full fine-tune."""
    if not cfg.get("use_lora", True):
        return None
    from peft import LoraConfig

    return LoraConfig(
        r=cfg.get("lora_r", 32),
        lora_alpha=cfg.get("lora_alpha", 64),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", DEFAULT_LORA_TARGETS),
        bias="none",
        task_type="CAUSAL_LM",
    )


def resolve_run_dir(cfg: dict) -> Path:
    require_output_root()
    path = Path(resolve_output_path(cfg["output_dir"]))
    path.mkdir(parents=True, exist_ok=True)
    return path


def last_checkpoint(output_dir: str | Path) -> str | None:
    """The newest checkpoint-N the trainer finished writing, or None.

    transformers writes trainer_state.json after the weights, optimizer and RNG state, so a
    checkpoint cut mid-save by a time limit lacks it and is skipped instead of resumed into.
    """
    import re

    root = Path(output_dir)
    if not root.is_dir():
        return None
    done = [(int(m.group(1)), d) for d in root.iterdir()
            if (m := re.fullmatch(r"checkpoint-(\d+)", d.name))
            and (d / "trainer_state.json").is_file()]
    return str(max(done)[1]) if done else None


def is_main_process() -> bool:
    """Rank 0 under torchrun, or the only process."""
    import os

    return int(os.environ.get("RANK", "0")) == 0


def write_run_manifest(cfg: dict, output_dir: Path, extra: dict | None = None) -> Path:
    """Records what produced a checkpoint, next to the checkpoint.

    `scripts/run_eval_all.py` discovers trained runs by looking for this file, and reads the
    base model and mitigation family out of it -- so an evaluation never needs flags telling
    it what a directory contains.
    """
    manifest = {
        "base_model": cfg["base_model"],
        "stage": cfg.get("stage"),
        "use_lora": cfg.get("use_lora", True),
        "data_config": cfg.get("data_config"),
        "config_path": cfg.get("_config_path"),
        "languages": cfg.get("data", {}).get("languages"),
        "protected_groups": cfg.get("data", {}).get("protected_groups"),
        **(extra or {}),
    }
    path = output_dir / "run_config.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def wandb_init(cfg: dict) -> None:
    """Prepares W&B, and guarantees it cannot take the training run down with it.

    Logging is not the experiment. An unreachable server, a revoked key or a mistyped entity
    must degrade to a local log, never abort a run that is about to spend GPU-hours -- this
    failed all six SFT runs at `on_train_begin` with `entity <name> not found`, after the
    model was loaded and the dataset tokenised.

    The entity is probed here, before training starts, so the fallback is chosen while it is
    still cheap and is visible in the log rather than discovered in a traceback. So is the
    connection itself: a server that cannot be reached (a compute node without a route out)
    means offline logging, kept on disk for `wandb sync`, not no logging.
    """
    import os

    if not cfg.get("use_wandb", True):
        os.environ["WANDB_MODE"] = "disabled"
        return

    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "hiring-bias-mitigation"))
    # Metrics only. Weights stay on the local model drive: the adapters are the study's
    # artefacts and they are not going to a third-party service. Set explicitly rather than
    # left unset, so an inherited WANDB_LOG_MODEL from the shell or .env cannot turn uploads
    # back on without anyone choosing it.
    os.environ["WANDB_LOG_MODEL"] = str(cfg.get("wandb_log_model", "false"))

    if os.environ.get("WANDB_MODE") in ("offline", "disabled"):
        return

    entity = os.environ.get("WANDB_ENTITY")
    try:
        import wandb

        viewer = wandb.Api(timeout=20).viewer
        if not entity:
            return
        available = {viewer.entity, *(viewer.teams or [])}
        if entity not in available:
            log.warning(
                "W&B entity %r is not one of %s -- disabling W&B for this run. Set "
                "WANDB_ENTITY in .env to one of those to log online; training is unaffected.",
                entity, sorted(a for a in available if a),
            )
            _disable_wandb()
    except Exception as exc:
        # Offline alone still resolved the entity at on_train_begin (see _disable_wandb), so
        # the entity goes too; the curves stay on disk for `wandb sync`.
        log.warning("W&B unreachable (%s) -- logging offline for this run", exc)
        os.environ["WANDB_MODE"] = "offline"
        os.environ.pop("WANDB_ENTITY", None)
    finally:
        # wandb.Api() loads wandb's process-wide settings from the environment once, and the
        # trainer's wandb.init copies them: without a teardown, the fallback above and the run
        # id set later by wandb_resume would be ignored.
        teardown = getattr(sys.modules.get("wandb"), "teardown", None)
        if teardown is not None:
            teardown()


def wandb_resume(cfg: dict, output_dir: Path) -> None:
    """One W&B run per output directory for a run that resumes, so a schedule split into
    time-limited segments is one curve rather than one W&B run per segment.

    The id lives in `output_dir/wandb_run_id`. When the file does not exist yet, an id already
    in the environment (an earlier segment's offline run) is adopted, otherwise a new one.
    """
    import os
    import secrets

    if not cfg.get("use_wandb", True) or not cfg.get("resume"):
        return
    path = Path(output_dir) / "wandb_run_id"
    if path.is_file():
        run_id = path.read_text().strip()
    else:
        run_id = os.environ.get("WANDB_RUN_ID") or secrets.token_hex(4)
        path.write_text(run_id + "\n")
    os.environ["WANDB_RUN_ID"] = run_id
    os.environ["WANDB_RESUME"] = "allow"


def _disable_wandb() -> None:
    """Turns W&B off hard, rather than trusting offline mode to stay local.

    `WANDB_MODE=offline` was not enough: the client still resolved the configured entity at
    `on_train_begin` and raised `entity ... not found during upsertBucket`, killing the run
    after the model was loaded and the dataset tokenised. Clearing the entity as well removes
    the thing it was trying to resolve, and "disabled" removes the client from the path
    entirely. Logging is not the experiment, so it gets no chance to end one.
    """
    import os

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ.pop("WANDB_ENTITY", None)


def supported_kwargs(cls, candidates: dict, label: str) -> dict:
    """Drops keyword arguments the installed class does not accept, saying which and why.

    TRL and transformers rename and remove config fields between majors, and a single
    unsupported keyword raises from `__init__` -- which does not fail one run, it fails the
    whole stage, because every config in it hits the same line. Dropping with a warning keeps
    a version bump from costing a day, and the warning is what stops a silently-ignored field
    from changing the experiment without changing the config that describes it.
    """
    import inspect

    accepted = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in accepted.values()):
        return dict(candidates)
    keep = {k: v for k, v in candidates.items() if k in accepted}
    for name in sorted(set(candidates) - set(keep)):
        log.warning(
            "%s does not accept %r in %s %s -- dropping it; check whether it was renamed",
            label, name, cls.__module__.split(".")[0], _installed_version(cls),
        )
    return keep


def _installed_version(cls) -> str:
    import importlib

    try:
        return getattr(importlib.import_module(cls.__module__.split(".")[0]), "__version__", "?")
    except Exception:  # pragma: no cover
        return "?"


def _warmup_kwargs(cfg: dict, n_train: int | None) -> dict:
    """`warmup_ratio` where it exists, an equivalent `warmup_steps` where it does not.

    transformers 5.x removed `warmup_ratio`, and passing it raises a TypeError from
    `TrainingArguments.__init__` -- which took out all sixteen training runs in this study in
    ninety seconds, and then the sixteen audits that depended on their checkpoints. Translating
    keeps one number in the config meaningful across both major versions.
    """
    import inspect
    import math
    import os

    from transformers import TrainingArguments

    ratio = float(cfg.get("warmup_ratio", 0.03))
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in supported:
        return {"warmup_ratio": ratio}
    if "warmup_steps" not in supported:  # pragma: no cover - would be a third signature
        log.warning("this transformers exposes neither warmup_ratio nor warmup_steps")
        return {}

    if "warmup_steps" in cfg:
        return {"warmup_steps": int(cfg["warmup_steps"])}
    if not n_train:
        log.warning(
            "warmup_ratio=%.3f cannot be converted to steps without the training-set size; "
            "falling back to no warmup", ratio,
        )
        return {}

    # Under torchrun each of WORLD_SIZE ranks takes its own batch, so one optimizer step uses
    # per-device batch x accumulation x ranks rows. Counting one rank, a 4-GPU run warmed up
    # for four times the intended steps (job 5107507: 471 of its 3,923, not 118).
    ranks = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    effective_batch = max(
        1,
        int(cfg.get("per_device_train_batch_size", 2))
        * int(cfg.get("gradient_accumulation_steps", 8))
        * ranks,
    )
    steps_per_epoch = math.ceil(n_train / effective_batch)
    total = max(1, steps_per_epoch * int(cfg.get("num_train_epochs", 2)))
    steps = max(1, round(ratio * total))
    log.info("warmup_ratio %.3f -> warmup_steps %d (of %d total, %d rank(s))",
             ratio, steps, total, ranks)
    return {"warmup_steps": steps}


def _mem_available_gb() -> float:
    """System-wide available memory, from the kernel's own estimate.

    On the Spark, GPU allocations come out of the same unified pool as everything else, so
    `torch.cuda.memory_reserved` alone cannot tell whether the *machine* is about to run out —
    and it is the machine running out that kills the desktop and the NVIDIA driver along with
    the run.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1_048_576
    except OSError:
        pass
    return float("inf")


def _memory_guard_class():
    """Built lazily so importing this module does not import transformers."""
    from transformers import TrainerCallback

    class MemoryGuardCallback(TrainerCallback):
        """Stops a run cleanly before the system runs out of memory, and logs the trajectory.

        Why this exists. A run grew its memory for eight hours and then took the whole machine
        down: the kernel OOM-killed system services, the NVIDIA driver failed allocations, and
        the run died without writing a final adapter. A run that stops itself at a floor loses
        a few steps; one killed by the kernel loses the run and whatever else was running.

        On every evaluation — where the peak is — the CUDA cache is released (evaluation
        allocates full-logit buffers the next training step does not reuse) and available
        memory is checked against `floor_gb`. Logging steps check too, without releasing.
        """

        def __init__(self, floor_gb: float = 12.0):
            self.floor_gb = floor_gb
            self.history: list[tuple[int, float]] = []

        def on_evaluate(self, args, state, control, **kwargs):
            self._check(state, control, released=True)
            return control

        def on_log(self, args, state, control, **kwargs):
            self._check(state, control, released=False)
            return control

        def _check(self, state, control, released: bool) -> None:
            if released:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # pragma: no cover - no CUDA in tests
                    pass
            available = _mem_available_gb()
            self.history.append((state.global_step, available))
            if released:
                log.info("memory: %.1f GB available at step %d", available, state.global_step)
            if available < self.floor_gb:
                log.error(
                    "memory guard: %.1f GB available at step %d, below the %.1f GB floor -- "
                    "stopping and saving now rather than letting the kernel kill the machine",
                    available, state.global_step, self.floor_gb,
                )
                control.should_training_stop = True
                control.should_save = True

    return MemoryGuardCallback


def _cost_callback_class():
    """Built lazily so importing this module does not import transformers."""
    from transformers import TrainerCallback

    from ..utils import cost

    class CostCallback(TrainerCallback):
        """Puts what a training segment did into its step's cost record (utils/cost.py).

        The parameters it trains against the model's total are the measured basis of any
        "fewer parameters" claim; the optimizer steps, rows, tokens and samples per second say
        what the step's GPU-hours bought; `checkpoint_times` gives the seconds from the start of
        training to each saved checkpoint, so an audited checkpoint can be paired with what it
        cost to reach. Rank 0 only. A segment resumed from a checkpoint counts its own steps,
        from the first step it takes.
        """

        def __init__(self):
            self.first_step: int | None = None
            self.checkpoints: dict[str, str] = {}

        def on_train_begin(self, args, state, control, model=None, train_dataloader=None,
                           **kwargs):
            if state.is_world_process_zero and model is not None:
                params = list(model.parameters())
                rows, tokens = _dataset_size(train_dataloader)
                cost.note(
                    params_total=sum(p.numel() for p in params),
                    params_trainable=sum(p.numel() for p in params if p.requires_grad),
                    world_size=args.world_size,
                    per_device_batch=args.per_device_train_batch_size,
                    gradient_accumulation=args.gradient_accumulation_steps,
                    effective_batch=args.per_device_train_batch_size
                    * args.gradient_accumulation_steps * args.world_size,
                    precision="bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
                    train_rows=rows,
                    train_tokens_per_epoch=tokens,
                    output_dir=str(args.output_dir),
                )
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                self.checkpoints[str(state.global_step)] = (
                    datetime.now().astimezone().isoformat(timespec="seconds"))
                cost.note(checkpoint_times=dict(self.checkpoints))
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if self.first_step is None:
                self.first_step = state.global_step - 1
            return control

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                final = next((e for e in reversed(state.log_history) if "train_runtime" in e),
                             {})
                taken = 0 if self.first_step is None else state.global_step - self.first_step
                cost.note(
                    global_step=state.global_step,
                    train_steps=taken,
                    train_runtime_s=final.get("train_runtime"),
                    train_samples_per_s=final.get("train_samples_per_second"),
                    train_flos=state.total_flos,
                    torch_peak_gib=_torch_peak_gib(),
                )
            return control

    return CostCallback


def _dataset_size(dataloader) -> tuple[int | None, int | None]:
    """Rows of the training set, and its tokens per epoch over every `*input_ids` column
    (SFT's `input_ids`; DPO's prompt, chosen and rejected ids). None when unknown, e.g. for
    GRPO, whose completions are generated during training."""
    try:
        dataset = dataloader.dataset
        rows = len(dataset)
    except Exception:
        return None, None
    try:
        columns = [c for c in dataset.column_names if c.endswith("input_ids")]
        tokens = sum(len(ids) for c in columns for ids in dataset[c]) if columns else None
    except Exception:
        tokens = None
    return rows, tokens


def _torch_peak_gib() -> float | None:
    """This rank's peak allocated CUDA memory. NVML's device memory, in the record, also
    counts what vLLM reserves up front, so for a colocated vLLM it says little."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:
        return None


def cost_callbacks() -> list:
    """Training facts for the step's cost record; none when no step is being recorded."""
    from ..utils import cost

    return [_cost_callback_class()()] if cost.tracked() else []


def memory_guard_callbacks(cfg: dict) -> list:
    floor = float(cfg.get("memory_floor_gb", 12.0))
    if floor <= 0:
        return []
    log.info("memory guard: stop cleanly below %.1f GB available", floor)
    return [_memory_guard_class()(floor)]


def configure_cuda_allocator() -> None:
    """Expandable segments, set before CUDA initialises.

    Variable-length batches allocate a differently sized logit tensor every step; the default
    allocator keeps each size's blocks, and on unified memory that reserved pool is system RAM.
    Expandable segments grow and shrink one region instead, which is the documented remedy for
    exactly this growth pattern.
    """
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def early_stopping_callbacks(cfg: dict, has_eval: bool) -> list:
    """Stops a run once the validation loss has stopped improving.

    Measured on this data: eval loss bottoms around step 300 of 1054 and rises monotonically
    after, while train loss keeps falling — by step 800 the model is worse on held-out data
    than it was at step 100. The optimum arrives inside the first epoch, so two epochs is not
    a budget, it is 700 steps of memorising the training split.

    `load_best_model_at_end` already rescues the *weights* (the step-300 checkpoint is kept and
    restored), so this does not change which model is saved. What it saves is the GPU time
    spent producing checkpoints that are then discarded.

    Patience counts **evaluations**, not steps: at the default `eval_steps=100` a patience of 3
    means "300 steps without a new best". Set `early_stopping_patience: 0` to disable.
    """
    if not has_eval:
        return []
    patience = int(cfg.get("early_stopping_patience", 3))
    if patience <= 0:
        return []
    from transformers import EarlyStoppingCallback

    threshold = float(cfg.get("early_stopping_threshold", 0.0))
    log.info(
        "early stopping: patience %d evaluation(s) (%d steps at eval_steps=%d), threshold %.4f",
        patience, patience * int(cfg.get("eval_steps", 100)), int(cfg.get("eval_steps", 100)),
        threshold,
    )
    return [EarlyStoppingCallback(
        early_stopping_patience=patience, early_stopping_threshold=threshold
    )]


def training_arguments(cfg: dict, output_dir: Path, has_eval: bool, n_train: int | None = None):
    """Shared TrainingArguments, with the eval/save strategy tied to whether a val split exists.

    `n_train` is only needed to turn a warmup *ratio* into warmup *steps* on transformers
    versions that dropped `warmup_ratio` (it is gone in 5.x). Without it the ratio is applied
    to a conservative default rather than silently ignored -- a warmup that quietly becomes
    zero changes the training run without changing the config that describes it.
    """
    from transformers import TrainingArguments

    strategy = "steps" if has_eval else "no"
    kwargs = _warmup_kwargs(cfg, n_train)
    # DDP with LoRA: unused-parameter search off, non-reentrant checkpointing. Passed only when
    # a config sets them, so single-GPU configs keep transformers' defaults.
    kwargs.update({key: cfg[key] for key in ("ddp_find_unused_parameters",
                                              "gradient_checkpointing_kwargs",
                                              "lr_scheduler_kwargs") if key in cfg})
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 2),
        # A hard cap for probes run against a deadline; -1 (the default) trains the schedule.
        max_steps=int(cfg.get("max_steps", -1)),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        **kwargs,
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=strategy,
        eval_steps=cfg.get("eval_steps", 100),
        save_strategy=strategy,
        # A run segmented by a time limit saves more often than it evaluates; transformers only
        # allows that without load_best_model_at_end, which such a config turns off.
        save_steps=cfg.get("save_steps", cfg.get("eval_steps", 100)),
        save_total_limit=cfg.get("save_total_limit", 2),
        load_best_model_at_end=has_eval and bool(cfg.get("load_best_model_at_end", True)),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=cfg.get("dtype", "bfloat16") == "bfloat16",
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=cfg.get("optim", "adamw_torch"),
        # Both default to off in transformers. Label smoothing is aimed at a measured effect:
        # predictive entropy fell 35% across one run while validation loss rose, i.e. the model
        # was growing confident rather than correct.
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        label_smoothing_factor=float(cfg.get("label_smoothing_factor", 0.0)),
        seed=cfg.get("seed", 42),
        report_to=["wandb"] if cfg.get("use_wandb", True) else [],
        run_name=cfg.get("wandb_run_name"),
        remove_unused_columns=False,
    )
