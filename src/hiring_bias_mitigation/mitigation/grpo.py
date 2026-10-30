"""GRPO from the SFT model: the decision itself as the reward, no teacher-written negatives.

Why this arm. Three rounds of DPO continued from SFT gained at best -0.96 pp unstable sets at
matched hire rate (CI -1.96 to +0.04; docs/ITERATIVE_DPO.md). DPO needs a `rejected` text,
and the student's own attribute-tipped mistakes ran dry and point one way. GRPO needs only a
prompt and a way to score a completion: the policy samples `num_generations` answers per
attributed prompt and each is rewarded for reaching the pair's attribute-free reference
verdict, so a wrong answer in either direction costs the same.

Rewards, plain Python, no model in the loop:

  decision_reward  1.0 when `parse_output(completion).decision` equals `reference_decision`
  format_reward    0.2 when the completion is exactly one JSON object with string `decision`
                   and `feedback` fields and the feedback runs FEEDBACK_WORDS words; the SFT
                   student meets it on 99.9% of its samples, so it guards against drift

Policy and reference. The SFT adapter is merged into Qwen3.5-9B in memory and a fresh LoRA is
fitted on top (training_common.load_model_and_tokenizer, as for DPO). With PEFT, TRL takes the
reference log-probabilities with the adapter disabled, so the KL term (`beta`) pulls toward
SFT. `run_config.json` names the SFT checkpoint, so merge.adapter_chain merges SFT then GRPO.

Rollouts. TRL's colocated vLLM, one engine per GPU, is started from the plain base model and
receives the policy's weights before every generation. Three things are changed from TRL's
defaults, each checked by a test or by `SyncValidationCallback` before a run trains:

  * Merged weights only, computed out of place. TRL merges the adapter into the base layers
    in place and unmerges it after the push (`merge_adapter`, `unmerge_adapter`): in bf16
    that is a rounding round trip on the frozen weights every step, and those weights are
    also the KL reference. `merged_named_params` adds the LoRA delta in fp32 into a copy.
    vLLM's LoRA path, which does not reproduce Qwen3.5 adapters (mitigation/merge.py), is
    never involved. A merged bf16 model still drops the parts of a young adapter's delta
    below half a bf16 unit of its weight (smoke run 5065791: 8-16% kept at step 10), so
    rollouts and the audit see the merged model, and `merge_retention` says how far it lags.
  * Names. The policy is the text model (`model.layers.N...`); vLLM serves the checkpoint's
    Qwen3_5ForConditionalGeneration, whose loader maps `model.language_model.` names, the
    names the checkpoint itself uses. `vllm_param_name` translates.
  * Engine arguments. TRL builds `LLM(...)` itself; `vllm_engine_kwargs` from the config
    (on Daint, `gdn_prefill_backend: triton`) are added to that call.

Prompts arrive rendered through the audit's chat template (`_chat_wrap`, thinking off), so
TRL tokenises them as plain text and applies no template of its own.

Run:
    python -m hiring_bias_mitigation.mitigation.grpo --config configs/mitigation/grpo/<x>.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval.parsing import HIRE, REJECT, parse_output
from ..utils.config import load_config
from ..utils.logging import get_logger
from . import training_common as C

log = get_logger(__name__)

DECISION_REWARD = 1.0
FORMAT_REWARD = 0.2
#: Feedback length the format reward accepts, in words. The prompt asks for at most 30; the
#: SFT student writes 14 to 27 (1st to 99th percentile of 16k samples).
FEEDBACK_WORDS = (4, 45)

#: GRPOConfig fields a config may set; absent keys keep TRL's defaults.
GRPO_CONFIG_PASSTHROUGH = (
    "beta", "num_generations", "temperature", "top_p", "top_k", "max_completion_length",
    "loss_type", "scale_rewards", "epsilon", "epsilon_high", "num_iterations",
    "steps_per_generation", "generation_batch_size", "mask_truncated_completions",
    "disable_dropout", "reward_weights", "log_completions", "num_completions_to_print",
    "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization", "vllm_max_model_length",
    "vllm_enable_sleep_mode", "vllm_importance_sampling_correction", "cast_lm_head_to_fp32",
    "shuffle_dataset",
)


# ---- rewards ------------------------------------------------------------------------------

def decision_reward(completions: list[str], reference_decision: list[str],
                    lang: list[str] | None = None, log_metric=None, **_) -> list[float]:
    """DECISION_REWARD when the parsed decision is the attribute-free reference, else 0.

    Also reports the share of sampled completions that hire, per reference side: the
    operating point that DPO's round 1 lost first.
    """
    langs = lang or ["en"] * len(completions)
    decisions = [parse_output(c, g).decision for c, g in zip(completions, langs)]
    references = [str(r).lower() for r in reference_decision]
    if log_metric is not None:
        for side in (HIRE, REJECT):
            mine = [d for d, r in zip(decisions, references) if r == side]
            if mine:
                log_metric(f"sampled_hire_rate_ref_{side}",
                           sum(d == HIRE for d in mine) / len(mine))
    return [DECISION_REWARD if d == r else 0.0 for d, r in zip(decisions, references)]


def well_formed(completion: str) -> bool:
    """Exactly one JSON object, nothing around it, with string `decision` and `feedback` and a
    feedback of FEEDBACK_WORDS words. Stricter than parse_output, which repairs and extracts."""
    try:
        obj = json.loads(completion.strip())
    except (json.JSONDecodeError, AttributeError):
        return False
    if not isinstance(obj, dict) or set(obj) != {"decision", "feedback"}:
        return False
    if not (isinstance(obj["decision"], str) and isinstance(obj["feedback"], str)):
        return False
    low, high = FEEDBACK_WORDS
    return low <= len(obj["feedback"].split()) <= high


def format_reward(completions: list[str], **_) -> list[float]:
    return [FORMAT_REWARD if well_formed(c) else 0.0 for c in completions]


def free_consistency_reward(completions: list[str], free_hire_rate: list[float],
                            lang: list[str] | None = None, **_) -> list[float]:
    """Agreement with SFT-v3's own attribute-free answer for the pair: its sampled hire share
    when the completion hires, one minus it when the completion rejects, 0 without a decision.

    Why (docs/GRPO_IMPROVE.md, A2). On 3,366 of grpo-v3-all's 9,140 rows the vote anchor and
    SFT-v3's attribute-free majority disagree, so the decision reward there teaches a different
    hiring standard, not invariance to the attribute. This reward pays for answering as the
    model answers without the attribute; `free_hire_rate` comes from
    scripts/build_grpo_free_target.py.
    """
    langs = lang or ["en"] * len(completions)
    out = []
    for c, g, p in zip(completions, langs, free_hire_rate, strict=True):
        d = parse_output(c, g).decision
        out.append(float(p) if d == HIRE else 1.0 - float(p) if d == REJECT else 0.0)
    return out


REWARD_FUNCS = [decision_reward, format_reward]


# ---- data ---------------------------------------------------------------------------------

def render_prompts(dataset, tokenizer, chat_template_kwargs: dict | None = None):
    """`prompt` through the audit's chat template, as `_to_conversational` renders a DPO
    prompt; every other column stays for the reward functions."""
    from ..eval.backends import _chat_wrap

    return dataset.map(lambda row: {"prompt": _chat_wrap(tokenizer, row["prompt"],
                                                         chat_template_kwargs)})


# ---- the policy's weights into vLLM ---------------------------------------------------------

def vllm_param_name(name: str, multimodal: bool) -> str:
    """A text-model parameter name as the checkpoint names it, for a multimodal vLLM model."""
    if multimodal and name.startswith("model.") and not name.startswith(
            ("model.language_model.", "model.visual.")):
        return "model.language_model." + name[len("model."):]
    return name


def lora_layers(model):
    """(name, module) for every layer that carries the active LoRA adapter."""
    adapter = getattr(model, "active_adapter", "default")
    for name, module in model.named_modules():
        if adapter in getattr(module, "lora_A", {}) and hasattr(module, "base_layer"):
            yield name, module


def merged_weight(module, adapter: str = "default"):
    """The layer's base weight with its LoRA delta added in fp32, cast once to the base dtype:
    the closest the base dtype can come to what HF + PEFT computes unmerged."""
    weight = module.base_layer.weight.detach()
    a = module.lora_A[adapter].weight.detach().float()
    b = module.lora_B[adapter].weight.detach().float()
    delta = b @ a
    if getattr(module, "fan_in_fan_out", False):
        delta = delta.T
    return (weight.float() + delta * module.scaling[adapter]).to(weight.dtype)


def merged_named_params(model, multimodal: bool = False):
    """(vLLM name, merged weight) for every parameter the served model holds.

    A LoRA layer yields `merged_weight`; the base weight itself is never written. LoRA's own
    parameters are skipped, they exist only inside the merged weight.
    """
    adapter = getattr(model, "active_adapter", "default")
    layers = dict(lora_layers(model))
    prefix = getattr(model, "prefix", "lora_")
    for name, param in model.named_parameters():
        if prefix in name:
            continue
        clean = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        weight = param.data
        if name.endswith(".base_layer.weight") and name[: -len(".base_layer.weight")] in layers:
            weight = merged_weight(layers[name[: -len(".base_layer.weight")]], adapter)
        yield vllm_param_name(clean, multimodal), weight


def merge_retention(model) -> dict:
    """How much of the LoRA delta survives a merge into the base dtype.

    bf16 has 8 significant bits: a delta element below half a unit in the last place of its
    base weight rounds away. With |W| ~ 1e-2 that unit is ~6e-5, and a GRPO adapter at lr 1e-5
    moves its delta by ~2e-6 per element in 10 steps, so a merged (served, audited) model can
    lag the HF + PEFT policy by most of the update early in a run. `kept_fraction` is the
    projection of what survives onto the delta (1.0: all of it); `unchanged` the share of
    weights the merge leaves bit-identical.
    """
    import torch

    adapter = getattr(model, "active_adapter", "default")
    kept = total = unchanged = count = 0.0
    with torch.no_grad():
        for _, module in lora_layers(model):
            base = module.base_layer.weight.detach().float()
            delta = merged_weight(module, adapter).float() - base
            a = module.lora_A[adapter].weight.detach().float()
            b = module.lora_B[adapter].weight.detach().float()
            exact = b @ a
            if getattr(module, "fan_in_fan_out", False):
                exact = exact.T
            exact = exact * module.scaling[adapter]
            kept += float((delta * exact).sum())
            total += float((exact * exact).sum())
            unchanged += float((delta == 0).sum())
            count += delta.numel()
    return {"kept_fraction": kept / total if total else 1.0,
            "unchanged": unchanged / count if count else 1.0}


class merged_into_base:
    """Context manager: the merged weights written into the base layers, the originals kept on
    the CPU and copied back bit for bit on exit. Inside it, with the adapter disabled, HF runs
    exactly the weights vLLM is sent."""

    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        import torch

        adapter = getattr(self.model, "active_adapter", "default")
        with torch.no_grad():
            for _, module in lora_layers(self.model):
                weight = module.base_layer.weight
                merged = merged_weight(module, adapter)
                self.saved.append((weight, weight.detach().to("cpu", copy=True)))
                weight.copy_(merged)
        return self.model

    def __exit__(self, *exc):
        import torch

        with torch.no_grad():
            for weight, original in self.saved:
                weight.copy_(original.to(weight.device))
        self.saved = []
        return False


def vllm_model(vllm_generation):
    """The model object inside a colocated TRL engine (the same path TRL's own sync uses)."""
    return vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model


def install_weight_sync(vllm_generation, policy) -> None:
    """Routes TRL's colocated sync through `merged_named_params`."""
    served = type(vllm_model(vllm_generation)).__name__
    multimodal = served.endswith("ForConditionalGeneration")
    log.info("vLLM serves %s; pushing merged weights%s", served,
             " under the checkpoint's model.language_model. names" if multimodal else "")
    vllm_generation._iter_named_params = lambda: merged_named_params(policy, multimodal)


def patch_vllm_engine_kwargs(extra: dict) -> None:
    """Adds `extra` to the `LLM(...)` call TRL makes for its colocated engine."""
    if not extra:
        return
    import functools

    from trl.generation import vllm_generation as VG

    original = getattr(VG.LLM, "__wrapped_by_grpo__", VG.LLM)

    @functools.wraps(original)
    def build(*args, **kwargs):
        return original(*args, **{**kwargs, **extra})

    build.__wrapped_by_grpo__ = original
    VG.LLM = build
    log.info("vLLM engine kwargs added to TRL's LLM(...): %s", extra)


# ---- callbacks ----------------------------------------------------------------------------

def _op_control():
    """scripts/operating_point_control.py as a module: its `validate` checks and thresholds
    are the ones the sync is held to."""
    import importlib.util
    import sys

    from ..utils.config import REPO_ROOT

    name = "operating_point_control"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "scripts" / "operating_point_control.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _broadcast(value):
    """Rank 0's value on every rank (identity without torch.distributed)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def _sync_validation_class():
    from transformers import TrainerCallback

    class SyncValidationCallback(TrainerCallback):
        """Does the vLLM engine serve the policy's merged weights? Checked on every rank's own
        engine when a segment starts and every `every` steps after.

        Three margin passes on the same held-out rows: vLLM (`logprob_token_ids` after the
        forced decision prefix, the op-margin-v2 scorer), HF on the merged weights (written into
        the base layers, adapter off, restored after), and HF + PEFT unmerged (eval/margins.py).
        The check is vLLM against HF-merged, with operating_point_control's `validate`
        thresholds: the same weights in two engines. HF-merged against HF + PEFT is recorded
        beside it with `merge_retention`, since a bf16 merge drops the smallest parts of a young
        adapter (see merge_retention) -- the merged model is the one the audit serves. A failed
        check stops the run before it trains on rollouts from the wrong weights.
        """

        def __init__(self, trainer, prompts: list[str], output_dir, every: int,
                     template_kwargs: dict | None, lang: str = "en"):
            self.trainer = trainer
            self.prompts = prompts
            self.output_dir = Path(output_dir)
            self.every = every
            self.template_kwargs = template_kwargs
            self.lang = lang

        def validate(self, step: int) -> dict:
            from vllm import SamplingParams

            from ..eval.backends import _chat_wrap
            from ..eval.margins import PREFIX, WORDS, decision_margins, decision_token_ids

            summary = _op_control().validation_summary
            trainer, tokenizer = self.trainer, self.trainer.processing_class
            policy = trainer.accelerator.unwrap_model(trainer.model)
            trainer.vllm_generation.sync_weights()
            ids = decision_token_ids(tokenizer, WORDS[self.lang])
            texts = [_chat_wrap(tokenizer, p, self.template_kwargs) + PREFIX
                     for p in self.prompts]
            params = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=list(ids))
            outs = trainer.vllm_generation.llm.generate(texts, params, use_tqdm=False)
            lp = [o.outputs[0].logprobs[0] for o in outs]
            vllm_margin = [d[ids[0]].logprob - d[ids[1]].logprob for d in lp]
            decided = sum(o.outputs[0].token_ids[0] in ids for o in outs) / len(outs)

            def hf(model):
                return decision_margins(model, tokenizer, self.prompts, lang=self.lang,
                                        batch_size=4, chat_template_kwargs=self.template_kwargs)

            with merged_into_base(policy), policy.disable_adapter():
                merged_margin = hf(policy)
            peft_margin = hf(policy)
            sync = summary(vllm_margin, merged_margin)
            merge = summary(merged_margin, peft_margin)
            merge.pop("passed")
            return {"step": step, "n": len(texts), "vllm_decision_token_share": decided,
                    "passed": sync["passed"], "vllm_vs_hf_merged": sync,
                    "hf_merged_vs_hf_peft": merge, "merge_retention": merge_retention(policy)}

        def _run(self, state, control, when: str):
            result = self.validate(state.global_step)
            passed = result["passed"]
            if state.is_world_process_zero:
                with (self.output_dir / "sync_validation.jsonl").open("a") as fh:
                    fh.write(json.dumps({**result, "when": when}) + "\n")
            sync, merge = result["vllm_vs_hf_merged"], result["hf_merged_vs_hf_peft"]
            log.info("vLLM sync %s at step %d: vLLM vs HF on the merged weights: agreement %.4f, "
                     "rank correlation %.4f, max |diff| %.3f; merged vs HF + PEFT: agreement "
                     "%.4f, median |diff| %.3f; merge keeps %.2f of the delta",
                     "PASSED" if passed else "FAILED", state.global_step,
                     sync["decision_agreement_hf"], sync["rank_correlation_hf"],
                     sync["abs_diff_hf"]["max"], merge["decision_agreement_hf"],
                     merge["abs_diff_hf"]["median"], result["merge_retention"]["kept_fraction"])
            # Every rank checks its own engine; one bad engine stops all of them.
            import torch
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor([0 if passed else 1], device=self.trainer.accelerator.device)
                dist.all_reduce(flag)
                passed = int(flag.item()) == 0
            if not passed:
                raise RuntimeError(f"vLLM does not serve the policy's merged weights ({when}, "
                                   f"step {state.global_step}): see sync_validation.jsonl")

        def on_train_begin(self, args, state, control, **kwargs):
            self._run(state, control, "segment start")

        def on_step_end(self, args, state, control, **kwargs):
            if self.every and state.global_step % self.every == 0:
                self._run(state, control, "after step")

    return SyncValidationCallback


#: SFT on the 2,700-set benchmark (outputs/raw/Qwen3.5-9B--en--sft--adapter--en_only_cscs):
#: the reference hires 33.3% of prompts; SFT hires 65.8% of those and 9.1% of the others,
#: 28.0% in all. r1-300, which collapsed to 19.1%, hired 47.6% and 4.8%.
SFT_BENCHMARK = {"reference_hire_share": 0.333, "hire_rate_ref_hire": 0.658,
                 "hire_rate_ref_reject": 0.091}
#: A run from another SFT parent gives that parent's two rates as `sft_benchmark` in its config
#: (reports/reference_metrics.csv: tpr_vs_gpt4o_pct and fpr_vs_gpt4o_pct, as fractions).


def projected_hire_rate(metrics: dict, baseline: dict, n_side: int,
                        sft_benchmark: dict = SFT_BENCHMARK) -> dict:
    """The monitor's hire rate carried onto the benchmark's scale, per reference side.

    The monitor rows are held-out pool rows, harder than the benchmark's on the reject side
    (SFT hires 22% of them against 9% on the benchmark) and easier on the hire side, so a
    change in points does not transfer. A relative change does, to first order: each side's
    hire rate since SFT, as a ratio (counts +0.5 against an empty side), multiplies SFT's
    benchmark rate on that side, and the two are weighted by the benchmark's reference share.
    Unchanged rates give SFT's 28.0%; r1-300's ratios would give its 19%.
    """
    projected = 0.0
    out = {}
    for side, weight in (("hire", sft_benchmark["reference_hire_share"]),
                         ("reject", 1 - sft_benchmark["reference_hire_share"])):
        key = f"hire_rate_ref_{side}"
        ratio = (metrics[key] * n_side + 0.5) / (baseline[key] * n_side + 0.5)
        rate = min(1.0, sft_benchmark[key] * ratio)
        out[f"projected_{key}"] = rate
        projected += weight * rate
    out["hire_rate_projected"] = projected
    return out


def _decision_monitor_class():
    from transformers import TrainerCallback

    class DecisionMonitorCallback(TrainerCallback):
        """Greedy decisions on fixed held-out rows every `every` steps (decision_eval's
        generate and metrics, rank 0), logged to W&B and decision_eval.jsonl; stops the run,
        saving first, when the projected hire rate leaves `band`.

        The SFT baseline is the same model with the adapter disabled, scored once per run. The
        same rows also go through rank 0's vLLM engine after a sync (decision_served_eval.jsonl):
        the merged weights, the model the audit will serve, which lags HF + PEFT while the
        adapter is young (merge_retention). The band is checked on HF + PEFT, the leading one.
        """

        def __init__(self, rows, tokenizer, output_dir, every: int, band: tuple[float, float],
                     template_kwargs: dict | None,
                     max_new_tokens: int = 96, lang: str = "en", vllm_generation=None,
                     sft_benchmark: dict | None = None):
            self.rows = rows.reset_index(drop=True)
            self.sft_benchmark = {**SFT_BENCHMARK, **(sft_benchmark or {})}
            self.vllm_generation = vllm_generation
            self.tokenizer = tokenizer
            self.output_dir = Path(output_dir)
            self.every = every
            self.band = band
            self.template_kwargs = template_kwargs
            self.max_new_tokens = max_new_tokens
            self.lang = lang
            self.baseline = None
            self.aborted = False

        def _n_side(self) -> int:
            return int(self.rows["reference_decision"].value_counts().min())

        def _baseline_path(self) -> Path:
            return self.output_dir / "decision_eval_sft.json"

        def _evaluate(self, model) -> tuple[list[str], dict]:
            from .decision_eval import decision_metrics, generate

            outputs = generate(model, self.tokenizer, self.rows["prompt"].tolist(),
                               max_new_tokens=self.max_new_tokens,
                               template_kwargs=self.template_kwargs)
            return outputs, decision_metrics(self.rows, outputs, self.lang)

        def _evaluate_served(self) -> tuple[list[str], dict]:
            from vllm import SamplingParams

            from ..eval.backends import _chat_wrap
            from .decision_eval import decision_metrics

            self.vllm_generation.sync_weights()
            texts = [_chat_wrap(self.tokenizer, p, self.template_kwargs)
                     for p in self.rows["prompt"]]
            params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)
            outs = self.vllm_generation.llm.generate(texts, params, use_tqdm=False)
            outputs = [o.outputs[0].text for o in outs]
            return outputs, decision_metrics(self.rows, outputs, self.lang)

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if not state.is_world_process_zero:
                return
            path = self._baseline_path()
            if path.is_file():
                self.baseline = json.loads(path.read_text())
                return
            with model.disable_adapter():
                _, self.baseline = self._evaluate(model)
            path.write_text(json.dumps(self.baseline, indent=2))
            log.info("decision monitor SFT baseline: hire rate %.3f (ref hire %.3f, ref reject "
                     "%.3f), agree %.3f", self.baseline["hire_rate"],
                     self.baseline["hire_rate_ref_hire"], self.baseline["hire_rate_ref_reject"],
                     self.baseline["agree_reference"])

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if not self.every or state.global_step % self.every:
                return
            stop = False
            if state.is_world_process_zero:
                from .decision_eval import record

                outputs, metrics = self._evaluate(model)
                metrics.update(projected_hire_rate(metrics, self.baseline, self._n_side(),
                                               self.sft_benchmark))
                low, high = self.band
                stop = not low <= metrics["hire_rate_projected"] <= high
                metrics["out_of_band"] = stop
                record(self.output_dir, state.global_step, self.rows, outputs, metrics)
                log.info("decision monitor at step %d: agree %.3f, projected hire rate %.3f "
                         "(band %.2f-%.2f), invalid %.3f", state.global_step,
                         metrics["agree_reference"], metrics["hire_rate_projected"], low, high,
                         metrics["invalid_rate"])
                if self.vllm_generation is not None:
                    served_out, served = self._evaluate_served()
                    served.update(projected_hire_rate(served, self.baseline, self._n_side(),
                                                       self.sft_benchmark))
                    record(self.output_dir, state.global_step, self.rows, served_out, served,
                           prefix="decision_served")
                    log.info("served (merged) at step %d: agree %.3f, projected hire rate %.3f, "
                             "invalid %.3f", state.global_step, served["agree_reference"],
                             served["hire_rate_projected"], served["invalid_rate"])
            if _broadcast(stop):
                log.error("projected hire rate left the band at step %d: saving and stopping",
                          state.global_step)
                self.aborted = True
                control.should_save = True
                control.should_training_stop = True

    return DecisionMonitorCallback


#: Bin edges on the SFT parent's decision margin, logit(hire) - logit(reject): the logits of
#: hire probabilities 1, 10, 30, 50, 70, 90 and 99%. scripts/monitor_calibration.py computes a
#: run's `margin_monitor` shares on these edges.
MARGIN_EDGES = (-4.595, -2.197, -0.847, 0.0, 0.847, 2.197, 4.595)


def margin_bins(margins, edges=MARGIN_EDGES):
    """Bin b holds edges[b-1] <= margin < edges[b]: 0 below the first edge, len(edges) above
    the last (pandas.cut with right=False)."""
    import numpy as np

    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(margins, dtype=float),
                           side="right")


def project_from_margins(parent, current, edges, share_ref_hire, share_ref_reject,
                         sft_benchmark: dict = SFT_BENCHMARK) -> dict:
    """The benchmark's hire rates projected from the monitor rows' decisions (margin > 0).

    Rows are binned by the parent's margin. Each bin's change in hire share since the parent,
    weighted by the share of benchmark prompts the parent puts in that bin where GPT-4o hires
    (`share_ref_hire`), is the change in the hire rate where GPT-4o hires (TPR); with
    `share_ref_reject`, where it rejects (FPR). The bins carry what the decision monitor's ratio
    could not: the benchmark's TPR moves in the parent's 50-99% bins, which its rows hardly
    held. Run on the benchmark's own margins, this estimate is within 1 pp of the TPR change of
    every arm compared so far (scripts/monitor_calibration.py validate). A bin with no monitor
    row adds nothing and its weight is reported as `unseen_*`.
    """
    import numpy as np

    parent, current = np.asarray(parent, dtype=float), np.asarray(current, dtype=float)
    bins = margin_bins(parent, edges)
    out = {"bins": [], "unseen_ref_hire": 0.0, "unseen_ref_reject": 0.0}
    delta_tpr = delta_fpr = 0.0
    for b in range(len(edges) + 1):
        mask = bins == b
        if mask.any():
            delta = float((current[mask] > 0).mean() - (parent[mask] > 0).mean())
            delta_tpr += share_ref_hire[b] * delta
            delta_fpr += share_ref_reject[b] * delta
        else:
            delta = None
            out["unseen_ref_hire"] += share_ref_hire[b]
            out["unseen_ref_reject"] += share_ref_reject[b]
        out["bins"].append({"n": int(mask.sum()), "delta_hire_share": delta})
    tpr = sft_benchmark["hire_rate_ref_hire"] + delta_tpr
    fpr = sft_benchmark["hire_rate_ref_reject"] + delta_fpr
    weight = sft_benchmark["reference_hire_share"]
    out.update(delta_tpr=delta_tpr, delta_fpr=delta_fpr, projected_tpr=tpr, projected_fpr=fpr,
               hire_rate_projected=weight * tpr + (1 - weight) * fpr)
    return out


def vllm_margins(vllm_generation, tokenizer, prompts: list[str],
                 template_kwargs: dict | None = None, lang: str = "en") -> list[float]:
    """Decision margins of the weights the colocated engine serves, the op-margin-v2 scorer:
    `logprob_token_ids` after the forced decision prefix (as SyncValidationCallback reads)."""
    from vllm import SamplingParams

    from ..eval.backends import _chat_wrap
    from ..eval.margins import PREFIX, WORDS, decision_token_ids

    ids = decision_token_ids(tokenizer, WORDS[lang])
    texts = [_chat_wrap(tokenizer, p, template_kwargs) + PREFIX for p in prompts]
    params = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=list(ids))
    outs = vllm_generation.llm.generate(texts, params, use_tqdm=False)
    return [o.outputs[0].logprobs[0][ids[0]].logprob - o.outputs[0].logprobs[0][ids[1]].logprob
            for o in outs]


def _margin_monitor_class():
    from transformers import TrainerCallback

    class MarginMonitorCallback(TrainerCallback):
        """The benchmark's TPR, projected from decision margins (`project_from_margins`) on
        held-out rows drawn across the parent's confidence (grpo_prompts.monitor_rows), every
        `every` steps on rank 0; stops the run, saving first, when the projected TPR has fallen
        more than `tpr_drop_stop` since the parent.

        Why. The decision monitor's held-out rows are ones the parent hires with near certainty,
        and its ratio projection never read GRPO-v3-all or DAPO-v3-all below SFT-v3's TPR by
        more than 8.1 and 2.5 pp; the benchmark measured 10.1 for both, past the standard's
        10 pp guardrail.

        `score(prompts)` returns the margins of the served weights (`vllm_margins` after a sync
        in training). The parent's margins are scored when the run starts, before any update,
        and kept in margin_monitor_parent.json, which a resumed run reads instead.
        """

        def __init__(self, prompts: list[str], score, output_dir, every: int, settings: dict,
                     sft_benchmark: dict | None = None):
            self.prompts = prompts
            self.score = score
            self.output_dir = Path(output_dir)
            self.every = every
            self.edges = tuple(settings.get("edges", MARGIN_EDGES))
            self.share_ref_hire = list(settings["share_ref_hire"])
            self.share_ref_reject = list(settings["share_ref_reject"])
            if not len(self.share_ref_hire) == len(self.share_ref_reject) == len(self.edges) + 1:
                raise ValueError("margin_monitor: one share per bin, len(edges) + 1 bins")
            self.tpr_drop_stop = float(settings["tpr_drop_stop"])
            self.sft_benchmark = {**SFT_BENCHMARK, **(sft_benchmark or {}),
                                  **({"reference_hire_share": settings["reference_hire_share"]}
                                     if "reference_hire_share" in settings else {})}
            self.parent = None
            self.aborted = False
            self.reason = None

        def _parent_path(self) -> Path:
            return self.output_dir / "margin_monitor_parent.json"

        def on_train_begin(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            import numpy as np

            path = self._parent_path()
            if path.is_file():
                self.parent = np.asarray(json.loads(path.read_text())["margins"])
                if len(self.parent) != len(self.prompts):
                    raise ValueError(f"{path} holds {len(self.parent)} margins for "
                                     f"{len(self.prompts)} monitor rows")
                return
            if state.global_step:
                raise RuntimeError(f"resuming at step {state.global_step} without {path}: the "
                                   "parent's margins can only be scored before any update")
            self.parent = np.asarray(self.score(self.prompts), dtype=float)
            path.write_text(json.dumps({"margins": self.parent.tolist()}))
            counts = np.bincount(margin_bins(self.parent, self.edges),
                                 minlength=len(self.edges) + 1)
            log.info("margin monitor: %d rows, per parent-margin bin %s", len(self.parent),
                     counts.tolist())

        def on_step_end(self, args, state, control, **kwargs):
            if not self.every or state.global_step % self.every:
                return
            stop = False
            if state.is_world_process_zero:
                result = project_from_margins(self.parent, self.score(self.prompts), self.edges,
                                              self.share_ref_hire, self.share_ref_reject,
                                              self.sft_benchmark)
                stop = result["delta_tpr"] < -self.tpr_drop_stop
                result.update(step=state.global_step, stop=stop)
                with (self.output_dir / "margin_monitor.jsonl").open("a") as fh:
                    fh.write(json.dumps(result) + "\n")
                log.info("margin monitor at step %d: projected TPR %.3f (%+.1f pp), FPR %.3f "
                         "(%+.1f pp), hire rate %.3f; stop below %+.1f pp TPR",
                         state.global_step, result["projected_tpr"], 100 * result["delta_tpr"],
                         result["projected_fpr"], 100 * result["delta_fpr"],
                         result["hire_rate_projected"], -100 * self.tpr_drop_stop)
            if _broadcast(stop):
                self.aborted = True
                self.reason = (f"aborted: projected TPR fell more than "
                               f"{100 * self.tpr_drop_stop:g} pp")
                log.error("%s at step %d: saving and stopping", self.reason, state.global_step)
                control.should_save = True
                control.should_training_stop = True

    return MarginMonitorCallback


# ---- training -----------------------------------------------------------------------------

def grpo_config_candidates(cfg: dict) -> dict:
    return {key: cfg[key] for key in GRPO_CONFIG_PASSTHROUGH if key in cfg}


def train(cfg: dict, trainer_cls=None, reward_funcs=None, trainer_kwargs: dict | None = None,
          manifest_extra=None) -> C.TrainArtifacts:
    """GRPO by default. A variant of the objective (mitigation/dapo.py) passes its trainer class,
    its rewards, extra keyword arguments for the trainer, and `manifest_extra`, a function of the
    GRPOConfig whose keys are added to run_config.json; the data, the policy, the weight sync,
    the monitors and the manifest stay this function's."""
    from ..utils.seed import set_seed

    set_seed(cfg.get("seed", 42))
    C.wandb_init(cfg)
    output_dir = C.resolve_run_dir(cfg)
    continuation = None
    resume = None
    if cfg.get("continuation"):
        from ..utils.config import resolve_output_path
        from . import grpo_continuation

        continuation = grpo_continuation.load_contract(
            cfg["continuation"], cfg, output_dir, resolve_output_path)
        resume = grpo_continuation.choose_resume(
            continuation, C.last_checkpoint(output_dir) if cfg.get("resume") else None)
    if continuation is None and C.is_main_process():
        C.wandb_resume(cfg, output_dir)

    splits = C.load_generated_dataset(cfg, cfg.get("data_kind", "grpo"))
    diagnostics_settings = cfg.get("rollout_diagnostics") or {}
    if not isinstance(diagnostics_settings, dict) or set(diagnostics_settings) - {"enabled"}:
        raise ValueError("rollout_diagnostics accepts only an enabled boolean")
    detailed_diagnostics = diagnostics_settings.get("enabled", False)
    if not isinstance(detailed_diagnostics, bool):
        raise ValueError("rollout_diagnostics.enabled must be a boolean")
    anchor_metadata_receipt = None
    diagnostics_strata = ()
    if detailed_diagnostics:
        if trainer_cls is not None:
            raise ValueError("detailed rollout diagnostics support the default GRPO trainer only")
        from ..utils.config import resolve_output_path
        from .grpo_anchor_metadata import attach_anchor_metadata

        anchor_metadata_receipt = {"schema": "grpo-anchor-metadata-splits-v1", "splits": {}}
        for split in ("train", "validation"):
            if split in splits:
                splits[split], receipt = attach_anchor_metadata(
                    splits[split], split=split, resolve_path=resolve_output_path)
                anchor_metadata_receipt["splits"][split] = receipt
        diagnostics_strata = sorted({
            f"{row['reference_decision']}/{row['winning_votes']}_of_{row['decided_draws']}"
            for split in anchor_metadata_receipt["splits"] for row in splits[split]
        })
    direct_contract = None
    full_contract = None
    if cfg.get("full_benchmark_monitor"):
        from ..utils.config import resolve_output_path
        from . import full_benchmark_monitor

        if continuation is None:
            raise ValueError("full_benchmark_monitor is only authorized for the continuation pilot")
        if cfg.get("benchmark_dev_monitor") or cfg.get("margin_monitor"):
            raise ValueError("full_benchmark_monitor replaces the dev and projection monitors")
        if int(cfg.get("save_steps", 25)) != 5 or int(cfg.get("max_steps", -1)) != 500:
            raise ValueError("full benchmark pilot requires save_steps=5 and absolute max_steps=500")
        full_contract = full_benchmark_monitor.load_contract(
            cfg["full_benchmark_monitor"], cfg.get("sft_checkpoint"), resolve_output_path)
        frozen_source = full_contract["manifest"]["continuation_source"]
        current_source = continuation["settings"]
        if (Path(resolve_output_path(frozen_source["source_checkpoint"])).resolve()
                != continuation["source"] or any(
                    frozen_source[key] != current_source[key]
                    for key in ("source_step", "target_learning_rate", "source_learning_rate",
                                "max_steps", "world_size", "source_files_sha256"))):
            raise ValueError("full benchmark monitor and continuation source identities differ")
        full_benchmark_monitor.validate_train_isolation(full_contract, splits["train"])
    if cfg.get("benchmark_dev_monitor"):
        from ..utils.config import resolve_output_path
        from .benchmark_dev_monitor import (
            load_contract,
            require_training_ready,
            validate_train_isolation,
        )

        if cfg.get("margin_monitor"):
            raise ValueError("benchmark_dev_monitor replaces margin_monitor; configure only one")
        direct_contract = load_contract(cfg["benchmark_dev_monitor"], cfg.get("sft_checkpoint"),
                                        resolve_output_path)
        require_training_ready(direct_contract)
        if "candidate_id" not in splits["train"].column_names:
            raise ValueError("benchmark_dev_monitor requires training candidate IDs")
        validate_train_isolation(
            direct_contract["rows"], splits["train"]["candidate_id"],
            direct_contract["manifest"]["split"]["candidate_ids"])
    model, tokenizer = C.load_model_and_tokenizer(cfg)
    chat_kwargs = cfg.get("chat_template_kwargs")
    train_ds = render_prompts(splits["train"], tokenizer, chat_kwargs)

    base_args = C.training_arguments(cfg, output_dir, has_eval=False,
                                     n_train=len(train_ds)).to_dict()
    # No TRL evaluation (the decision monitor is the eval), but checkpoints all the same.
    base_args.update(save_strategy="steps", save_steps=int(cfg.get("save_steps", 25)))
    if not (cfg.get("use_vllm") and cfg.get("vllm_mode") == "colocate"):
        raise ValueError("this trainer expects use_vllm: true and vllm_mode: colocate")
    patch_vllm_engine_kwargs(cfg.get("vllm_engine_kwargs") or {})

    from trl import GRPOConfig, GRPOTrainer

    bias_projection = bool((cfg.get("bias_projection") or {}).get("enabled", False))
    if bias_projection:
        if trainer_cls is not None or detailed_diagnostics:
            raise ValueError("bias_projection supports the default GRPO trainer without "
                             "detailed rollout diagnostics")
        from .grpo_bias_projection import trainer_class as bias_projected

        trainer_cls = bias_projected(
            GRPOTrainer, lambda completion, lang: parse_output(completion, lang).decision)
    anchor = cfg.get("operating_point_anchor") or {}
    if anchor and (trainer_cls is not None or detailed_diagnostics):
        raise ValueError("operating_point_anchor supports the default GRPO trainer without "
                         "detailed rollout diagnostics or another trainer variant")
    args = GRPOConfig(**base_args,
                      **C.supported_kwargs(GRPOConfig, grpo_config_candidates(cfg), "GRPOConfig"))
    continuation_callback = None
    if continuation is not None:
        source_args = grpo_continuation.load_source_arguments(continuation)
        grpo_continuation.validate_arguments(continuation, args, source_args)
        continuation_callback = grpo_continuation.callback_class()(continuation, resume, source_args)
    consistency = bool((cfg.get("consistency_reward") or {}).get("enabled", False))
    if consistency:
        if reward_funcs is not None or detailed_diagnostics:
            raise ValueError("consistency_reward supports the default rewards without detailed "
                             "rollout diagnostics")
        if "free_hire_rate" not in train_ds.column_names:
            raise ValueError("consistency_reward needs a free_hire_rate column "
                             "(scripts/build_grpo_free_target.py)")
        reward_funcs = [*REWARD_FUNCS, free_consistency_reward]
    trainer = (trainer_cls or GRPOTrainer)(
        model=model,
        reward_funcs=reward_funcs or REWARD_FUNCS,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=C.peft_config(cfg),
        callbacks=C.memory_guard_callbacks(cfg) + C.cost_callbacks()
        + ([continuation_callback] if continuation_callback is not None else []),
        **(trainer_kwargs or {}),
    )
    if anchor:
        import pandas as pd

        from ..eval.backends import _chat_wrap
        from ..eval.margins import PREFIX, WORDS, decision_token_ids
        from ..utils.config import resolve_output_path
        from .grpo_operating_anchor import callback_class as anchor_callback

        anchor_rows = pd.read_parquet(resolve_output_path(anchor["rows"]))
        trainer.add_callback(anchor_callback()(
            tokenizer=tokenizer,
            prompts=[_chat_wrap(tokenizer, p, chat_kwargs) + PREFIX for p in anchor_rows["prompt"]],
            weight=float(anchor["weight"]), rows_per_step=int(anchor["rows_per_step"]),
            seed=int(cfg.get("seed", 42)), token_ids=decision_token_ids(tokenizer, WORDS["en"]),
            rank=trainer.accelerator.process_index, world=trainer.accelerator.num_processes,
            metrics=trainer._metrics["train"]))
        log.info("operating-point anchor: %d rows from %s, weight %g, %d rows per step and rank",
                 len(anchor_rows), anchor["rows"], float(anchor["weight"]),
                 int(anchor["rows_per_step"]))
    from .grpo_diagnostics import install as install_count_diagnostics

    count_diagnostics_receipt = install_count_diagnostics(
        trainer, parse_decision=lambda completion, lang: parse_output(completion, lang).decision,
        output_dir=output_dir, detailed=detailed_diagnostics, strata=diagnostics_strata)
    if continuation is not None:
        grpo_continuation.initialize_tracking_after_resume(
            trainer.accelerator, lambda: C.wandb_resume(cfg, output_dir), _broadcast)
    policy = trainer.accelerator.unwrap_model(trainer.model)
    install_weight_sync(trainer.vllm_generation, policy)

    from .decision_eval import eval_rows

    monitor = None
    if "validation" in splits:
        rows = splits["validation"].to_pandas()
        seed = int(cfg.get("seed", 42))
        check = eval_rows(rows, int(cfg.get("sync_validate_rows", 256)), seed)
        trainer.add_callback(_sync_validation_class()(
            trainer, check["prompt"].tolist(), output_dir,
            int(cfg.get("sync_validate_steps", 0)), chat_kwargs))
        if direct_contract is None and full_contract is None:
            band = cfg.get("hire_rate_band", [0.20, 0.40])
            monitor = _decision_monitor_class()(
                eval_rows(rows, int(cfg.get("decision_eval_rows", 256)), seed), tokenizer,
                output_dir, int(cfg.get("decision_eval_steps", 25)),
                (float(band[0]), float(band[1])),
                chat_kwargs,
                int(cfg.get("decision_eval_max_new_tokens", 96)),
                vllm_generation=trainer.vllm_generation,
                sft_benchmark=cfg.get("sft_benchmark"))
            trainer.add_callback(monitor)
    else:
        log.warning("no validation split: the vLLM sync is NOT validated; legacy hire-rate "
                    "monitor is unavailable")

    direct_monitor = None
    if direct_contract is not None:
        from .benchmark_dev_monitor import callback_class, served_decisions

        def score_benchmark_dev(rows):
            return served_decisions(trainer.vllm_generation, tokenizer, rows,
                                    direct_contract["manifest"]["scorer"])

        direct_monitor = callback_class()(
            direct_contract, score_benchmark_dev, trainer.vllm_generation.sync_weights, output_dir)
        trainer.add_callback(direct_monitor)
        log.info("direct benchmark-dev monitor on %d rows; legacy projection monitors disabled",
                 len(direct_contract["rows"]))

    full_monitor = None
    if full_contract is not None:
        def score_full_local(rows):
            return full_benchmark_monitor.served_decisions(
                trainer.vllm_generation, tokenizer, rows, full_contract["manifest"]["scorer"])

        # The callback shards rows across all four ranks; score_full_local must not shard.
        full_monitor = full_benchmark_monitor.callback_class()(
            full_contract, score_full_local, trainer.vllm_generation.sync_weights, output_dir)
        trainer.add_callback(full_monitor)
        log.info("direct full benchmark pilot monitor on %d rows; checks every five updates",
                 len(full_contract["rows"]))

    margin = None
    if cfg.get("margin_monitor"):
        import pandas as pd

        from ..utils.config import resolve_output_path

        settings = cfg["margin_monitor"]
        rows_path = (Path(resolve_output_path(cfg["data"]["artifacts_dir"]))
                     / f"{cfg.get('data_kind', 'grpo')}_monitor.parquet")
        monitor_prompts = pd.read_parquet(rows_path, columns=["prompt"])["prompt"].tolist()

        def served_margins(prompts):
            trainer.vllm_generation.sync_weights()
            return vllm_margins(trainer.vllm_generation, tokenizer, prompts, chat_kwargs)

        margin = _margin_monitor_class()(
            monitor_prompts, served_margins, output_dir,
            int(settings.get("steps", cfg.get("decision_eval_steps", 25))), settings,
            cfg.get("sft_benchmark"))
        trainer.add_callback(margin)
        log.info("margin monitor on %d rows of %s", len(monitor_prompts), rows_path)

    manifest = {
        "mitigation_family": "grpo",
        "objective": "grpo",
        "beta": args.beta,
        "num_generations": args.num_generations,
        "rewards": {"decision": DECISION_REWARD, "format": FORMAT_REWARD,
                    "feedback_words": list(FEEDBACK_WORDS),
                    **({"free_consistency": 1.0} if consistency else {})},
        "n_train": len(train_ds),
        "sft_checkpoint": cfg.get("sft_checkpoint"),
    }
    manifest["rollout_diagnostics"] = count_diagnostics_receipt
    manifest["bias_projection"] = bias_projection
    manifest["operating_point_anchor"] = anchor or None
    if anchor_metadata_receipt is not None:
        manifest["rollout_diagnostics"]["anchor_metadata"] = anchor_metadata_receipt
    if continuation is not None:
        manifest["continuation"] = {
            **continuation["settings"], "identity_sha256": continuation["identity_sha256"],
            "resume_from_checkpoint": resume, "rng_limitation": continuation["rng_limitation"],
        }
        manifest["training_arguments"] = args.to_dict()
    if direct_contract is not None:
        manifest["benchmark_dev_monitor"] = direct_contract["identity"]
        manifest["benchmark_dev_monitor_identity_sha256"] = direct_contract["identity_sha256"]
    if full_contract is not None:
        manifest["full_benchmark_monitor"] = full_contract["identity"]
        manifest["full_benchmark_monitor_identity_sha256"] = full_contract["identity_sha256"]
    if manifest_extra is not None:
        manifest.update(manifest_extra(args))
    # Before training as well: a run cut by a time limit never reaches the final write, and
    # its checkpoint-N adapters need `sft_checkpoint` to be merged with their SFT parent.
    if C.is_main_process():
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": "training"})
    if continuation is None:
        resume = C.last_checkpoint(output_dir) if cfg.get("resume") else None
    if resume:
        log.info("resuming from %s", resume)
    log.info("%s %s on %d prompts x %d generations", manifest["objective"].upper(),
             cfg["base_model"], len(train_ds), args.num_generations)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    # A trainer that stops itself (dapo.DAPOTrainer, when its prompts run dry) says why.
    status = (full_monitor.reason if full_monitor and full_monitor.aborted
              else direct_monitor.reason if direct_monitor and direct_monitor.aborted
              else "aborted: hire rate left the band" if monitor and monitor.aborted
              else margin.reason if margin and margin.aborted
              else getattr(trainer, "stop_reason", None) or "done")
    if C.is_main_process():
        tokenizer.save_pretrained(str(output_dir))
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": status})
    return C.TrainArtifacts(output_dir, len(train_ds), 0)


def main() -> None:
    C.configure_cuda_allocator()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("stage", "grpo")
    if cfg.get("sft_checkpoint"):
        cfg["base_model"] = C.resolve_sft_checkpoint(cfg["sft_checkpoint"])
    artifacts = train(cfg)
    log.info("GRPO complete -> %s", artifacts.output_dir)


if __name__ == "__main__":
    main()
