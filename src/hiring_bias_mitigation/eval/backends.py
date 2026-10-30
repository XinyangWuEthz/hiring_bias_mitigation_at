"""Generation backends. One interface, two implementations, chosen per run.

`vllm`
    Default for every audit and for training-data generation. Batched, and it loads a LoRA
    adapter alongside a base model, which is how the SFT and DPO runs are evaluated without
    materialising a merged checkpoint per experiment.

`transformers`
    Required by the embedding-based mitigation, which needs forward hooks on the residual
    stream -- vLLM's scheduler gives no stable place to hang those. Also the fallback on a
    machine without a working vLLM build. Slower; fine for a single condition, painful for a
    full sweep.

Decoding configuration is recorded on every run and printed in the report. Audit-paper
reporting requirement 5: decoding is stochastic, a single sample cannot separate bias from
sampling variance, and the inconsistency rate is partly a measure of that variance. Greedy
decoding (`temperature: 0.0`) is the default here for exactly that reason -- it removes the
variance so a change between two runs is attributable to the mitigation. Set `n_samples > 1`
with a non-zero temperature when you want to measure the variance instead of removing it.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..utils import cost
from ..utils.logging import get_logger

log = get_logger(__name__)

# vLLM defaults to forking its EngineCore worker. On this stack that fails for the Gemma
# architectures with "Cannot re-initialize CUDA in forked subprocess" -- something in their
# startup path touches the CUDA context before the fork, where Qwen's does not. The two Qwen
# runs of a six-run sweep therefore succeeded and all four Gemma-family runs died in seconds.
#
# `spawn` starts the worker from a clean interpreter, which sidesteps it entirely. The cost is
# that the child re-imports the parent's __main__, so every entry point must be a real file
# guarded by `if __name__ == "__main__"` -- true of every script here, but it is why a
# `python - <<EOF` heredoc fails under spawn with a confusing FileNotFoundError on '<stdin>'.
#
# setdefault, and set at import time: this module is always imported before vLLM itself
# (vLLM is imported lazily inside the backends), and an explicit VLLM_WORKER_MULTIPROC_METHOD
# in the environment still wins.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    n_samples: int = 1
    #: vLLM only. Sampled decoding gives every prompt's request the one seed `seed` instead of
    #: `per_prompt_seed(seed, prompt)`. That is the request every sampled run sent before
    #: per-prompt seeds, so this exists to replay such a run and nothing else: under one seed,
    #: prompts share their sampling noise (see `VLLMBackend.generate`). A record without this
    #: key predates per-prompt seeds.
    shared_seed: bool = False

    def as_dict(self) -> dict:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "n_samples": self.n_samples,
            "greedy": self.temperature == 0.0,
            "shared_seed": self.shared_seed,
        }


class Backend(Protocol):
    def generate(self, prompts: list[str]) -> list[str]:
        ...

    def close(self) -> None:
        ...


#: Passed into the chat template. Thinking mode is OFF by default, for two reasons.
#:
#: Practical: a reasoning model spends its whole budget on a visible thinking trace and gets
#: truncated before it ever emits the JSON. Qwen3.5-4B produced a 100% parse-failure rate at
#: max_new_tokens=512 this way -- every response was a "Thinking Process:" preamble cut off
#: mid-sentence. Raising the budget instead would multiply the cost of a ~1M-generation audit
#: several times over to pay for text no measure reads.
#:
#: Scientific: the audit study this work extends evaluated non-reasoning models, so
#: non-thinking mode is what keeps a baseline here comparable to its published baseline. It
#: also keeps the arm honest -- "does an inference-time reasoning trace reduce hiring bias?"
#: is a real question, but it is a *mitigation* to be tested as its own arm (the
#: `zero_shot_cot` and `reasoning` prompt strategies already probe a version of it), not a
#: silent property of the baseline.
DEFAULT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}


def _chat_wrap(tokenizer, prompt: str, template_kwargs: dict | None = None) -> str:
    """Applies the model's chat template.

    Every model here is instruction-tuned, so the screening prompt must arrive as a user turn
    with the model's own template. Sending the bare string instead is a silent quality
    regression that looks like a fairness result: the model answers worse, refusals go up,
    and the inconsistency rate rises for reasons that have nothing to do with the attribute.
    """
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    kwargs = DEFAULT_CHAT_TEMPLATE_KWARGS if template_kwargs is None else template_kwargs
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    except (TypeError, ValueError) as exc:
        # Not every template accepts every switch. Falling back is correct -- a model with no
        # thinking mode needs no flag to disable it -- but say so, because on a model that
        # DOES reason and rejects the kwarg the run would silently fill with truncated
        # thinking traces.
        log.warning(
            "chat template rejected %s (%s); falling back to the plain template. If this "
            "model has a thinking mode, check the parse-failure rate before trusting the run.",
            kwargs, exc,
        )
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def per_prompt_seed(base_seed: int, prompt: str) -> int:
    """The sampling seed of one prompt: a stable hash of the run's base seed and its text.

    The text is the prompt as the caller passed it, before the chat template, which is what
    prompts.parquet stores, so any stored sample's seed can be recomputed. sha256 rather than
    `hash()`, whose value for a str changes with every interpreter (PYTHONHASHSEED). The value
    stays below 2**62: vLLM runs the n children of a request on seed .. seed + n - 1 and keeps
    seeds in an int64, and two prompts' child ranges overlap only when their seeds land within
    n of each other, about n / 2**61 per pair of prompts.
    """
    digest = hashlib.sha256(f"{int(base_seed)}:{prompt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 2


@dataclass
class VLLMBackend:
    model: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lora_path: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 8192
    dtype: str = "auto"
    trust_remote_code: bool = True
    #: Concurrent sequences. Left to vLLM's default (256) unless set.
    #:
    #: The Qwen3.5 line carries Mamba/GDN layers, and each decode sequence needs its own
    #: Mamba cache block carved out of the KV cache. On a large model the KV cache can end up
    #: small enough that 256 blocks do not exist, and startup fails with "max_num_seqs
    #: exceeds available Mamba cache blocks" rather than degrading gracefully. Capping this
    #: makes the run robust to however much memory happens to be free at launch.
    max_num_seqs: int | None = None
    chat_template_kwargs: dict | None = None
    #: Extra `vllm.EngineArgs` fields, passed to `LLM(...)` as given. Exists for platform
    #: switches the repo has no reason to model one by one -- e.g. `gdn_prefill_backend:
    #: triton` on CSCS GH200 nodes, which have no system CUDA toolkit for FlashInfer to
    #: JIT-compile its Qwen3.5 linear-attention prefill kernel with.
    engine_kwargs: dict | None = None
    _llm: object = None
    _tokenizer: object = None
    _lora_request: object = None

    def _load(self):
        if self._llm is not None:
            return
        with cost.phase("load"):
            self._start_engine()

    def _start_engine(self):
        from transformers import AutoTokenizer
        from vllm import LLM

        log.info("loading %s through vLLM (lora=%s)", self.model, self.lora_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        self._llm = LLM(
            model=self.model,
            dtype=self.dtype,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            enable_lora=self.lora_path is not None,
            max_lora_rank=64,
            seed=self.generation.seed,
            **({"max_num_seqs": self.max_num_seqs} if self.max_num_seqs else {}),
            **(self.engine_kwargs or {}),
        )
        if self.lora_path:
            from vllm.lora.request import LoRARequest

            self._lora_request = LoRARequest("mitigation", 1, self.lora_path)

    def _sampling_params(self, prompts: list[str], n: int):
        """The sampling params of one `LLM.generate` call over `prompts`: a single
        SamplingParams when every request is the same, else one per prompt, which vLLM 0.31
        pairs with the prompts one to one.

        Greedy decoding has no random stream and carries no seed. Sampled decoding seeds each
        prompt with `per_prompt_seed(generation.seed, prompt)`, or, under
        `generation.shared_seed`, sends every prompt the one seed `generation.seed` -- the
        request every sampled run sent before per-prompt seeds.
        """
        from vllm import SamplingParams

        g = self.generation

        def params(seed: int | None):
            return SamplingParams(max_tokens=g.max_new_tokens, temperature=g.temperature,
                                  top_p=g.top_p, seed=seed, n=int(n))

        if g.temperature <= 0:
            return params(None)
        if g.shared_seed:
            return params(g.seed)
        return [params(per_prompt_seed(g.seed, p)) for p in prompts]

    def generate(self, prompts: list[str]) -> list[str]:
        """One completion per prompt.

        Greedy decoding, the audit default, sends one SamplingParams without a seed, as it
        always has. Sampled decoding seeds each prompt from its own text, as `generate_samples`
        does: `per_prompt_seed(generation.seed, prompt)`, independent of the prompt's batch,
        chunk or position. Identical prompts in one call still draw identical text; several
        draws of one prompt come from `generate_samples`.

        Until 2026-10-20 sampled decoding here sent every prompt the one seed
        `generation.seed`, and the teacher ran on that. configs/generation/teacher.yaml (T=0.7,
        scripts/generate_training_data.py) drew artifacts/semisynthetic-v1 on the DGX Spark,
        and negatives_v2_small*.yaml drew the batch-01 negatives. Neither config regenerates
        its artifacts byte for byte any more. Under one seed, vLLM 0.31's V2 model runner gives
        prompts of equal token length the same noise (see `generate_samples`), and the classic
        runner, which seeds a generator per request and draws the noise per decode step, gives
        every prompt the same noise at each step. The check of 2026-10-19 found no
        equal-length replay in v1 or batch-01 (batch-01: no duplicate text in 720), but the v1
        anchors look like one noise draw shared by all 3,000 prompts: they hire none of the
        288 pairs that eight independent Daint draws (job 5073453) hire 1-4 times. Job 5103584
        (2026-10-20, scripts/anchor_seed_check.py on branch xinyang/anchor-seed-check)
        confirmed it: on the classic runner (VLLM_USE_V2_MODEL_RUNNER=0) the shared seed 42
        redrew v1's 3,000 anchors on Daint with the same decision for 98.9% of pairs and
        byte-identical text for 40.3% (two independent draws: 0.56%), while per-prompt seeds
        on the same runner matched the eight-draw vote. `generation.shared_seed: true` sends
        the old request again; a replay of v1 also needs the classic runner.
        """
        self._load()
        params = self._sampling_params(prompts, n=1)
        wrapped = [_chat_wrap(self._tokenizer, p, self.chat_template_kwargs) for p in prompts]
        kwargs = {"lora_request": self._lora_request} if self._lora_request else {}
        with cost.phase("generate"):
            outputs = self._llm.generate(wrapped, params, **kwargs)
        _count_vllm(outputs)
        _note_prefix_cache(self._llm)
        return [o.outputs[0].text for o in outputs]

    def generate_samples(self, prompts: list[str], n: int) -> list[list[str]]:
        """`n` samples per prompt from one request each, at the configured temperature.

        Repeating a prompt n times under a fixed per-request seed returns n copies of one
        answer; asking for n samples in one request draws n different ones, reproducibly.
        On-policy negatives need exactly that: several of the student's own answers per prompt.

        Each prompt's request carries its own seed, `per_prompt_seed(generation.seed, prompt)`.
        With one seed for every prompt, vLLM 0.31 runs child sample j of each request on
        seed + j, and its V2 model runner draws the sampling noise from (seed, token position,
        token id). Any two prompts of equal token length therefore got the same noise in
        sample j. In onpolicy-full (check of 2026-10-19), 57% of same-index samples were
        byte-identical between attribute variants of one pair that tokenise to the same length
        (7% of variant pairs) and none between variants of different lengths. An attribute
        prompt is always longer than its pair's attribute-free prompt, so the free-vs-
        attributed comparison a row's tipping is read from was not coupled. The seed depends on
        the base seed and the prompt text only, so a resumed or re-sharded run redraws the same
        samples. The same model on the same prompts under the same base seed still draws the
        same samples; a run meant to add independent draws must set its own `seed`.

        Earlier on-policy runs -- batch-01-onpolicy, onpolicy-full, onpolicy-hire32,
        onpolicy-it2 and onpolicy-gender-pi1 -- sampled under the shared seed (`seed=42`).
        onpolicy-hire32's samples 0-7 replay onpolicy-full's (same model, prompts and seeds:
        84% byte-identical, 99.9% the same decision), and onpolicy-gender-pi1's samples of the
        prompts it shares with onpolicy-it2 replay those (81%, 99.8%). Re-running their
        configs now draws different samples, unless `generation.shared_seed` is set.
        """
        self._load()
        params = self._sampling_params(prompts, n=n)
        wrapped = [_chat_wrap(self._tokenizer, p, self.chat_template_kwargs) for p in prompts]
        kwargs = {"lora_request": self._lora_request} if self._lora_request else {}
        with cost.phase("generate"):
            outputs = self._llm.generate(wrapped, params, **kwargs)
        _count_vllm(outputs)
        _note_prefix_cache(self._llm)
        return [[c.text for c in o.outputs] for o in outputs]

    def close(self) -> None:
        self._llm = None
        self._lora_request = None
        _free_gpu()


@dataclass
class TransformersBackend:
    """HF generate loop, with an optional activation hook for the embedding mitigation."""

    model: str
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    lora_path: str | None = None
    dtype: str = "bfloat16"
    batch_size: int = 8
    trust_remote_code: bool = True
    chat_template_kwargs: dict | None = None
    activation_editor: object = None  # mitigation.embedding.ActivationEditor
    _model: object = None
    _tokenizer: object = None

    def _load(self):
        if self._model is not None:
            return
        with cost.phase("load"):
            self._load_model()

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("loading %s through transformers", self.model)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Decoder-only generation must pad on the left, or the generated continuation starts
        # after a run of pad tokens and the batch produces different text than the same
        # prompts would one at a time.
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            dtype=getattr(torch, self.dtype),
            device_map="auto",
            trust_remote_code=self.trust_remote_code,
        )
        if self.lora_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self.lora_path)
        self._model.eval()
        if self.activation_editor is not None:
            self.activation_editor.attach(self._model)

    def generate(self, prompts: list[str]) -> list[str]:
        self._load()
        with cost.phase("generate"):
            return self._generate(prompts)

    def _generate(self, prompts: list[str]) -> list[str]:
        import torch

        wrapped = [_chat_wrap(self._tokenizer, p, self.chat_template_kwargs) for p in prompts]
        results: list[str] = []
        prompt_tokens = output_tokens = 0
        # vLLM prints its own progress bar; this loop printed nothing at all, so a run of tens
        # of thousands of prompts was hours of silence with no way to tell progress from a
        # hang, and no ETA to plan the queue around.
        started = time.monotonic()
        n_batches = max(1, math.ceil(len(wrapped) / self.batch_size))
        log_every = max(1, n_batches // 50)
        for index, start in enumerate(range(0, len(wrapped), self.batch_size)):
            batch = wrapped[start : start + self.batch_size]
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=8192
            ).to(self._model.device)
            with torch.no_grad():
                out = self._model.generate(
                    **enc,
                    max_new_tokens=self.generation.max_new_tokens,
                    do_sample=self.generation.temperature > 0,
                    temperature=self.generation.temperature or None,
                    top_p=self.generation.top_p,
                    pad_token_id=self._tokenizer.pad_token_id,
                )
            # Slice off the prompt so only the continuation is returned.
            for i in range(len(batch)):
                results.append(
                    self._tokenizer.decode(
                        out[i][enc["input_ids"].shape[1] :], skip_special_tokens=True
                    )
                )
            if cost.tracked():
                prompt_tokens += int(enc["attention_mask"].sum())
                output_tokens += int(
                    (out[:, enc["input_ids"].shape[1] :] != self._tokenizer.pad_token_id).sum()
                )
            if index % log_every == 0 or index == n_batches - 1:
                done = len(results)
                elapsed = time.monotonic() - started
                rate = done / elapsed if elapsed else 0.0
                remaining = (len(wrapped) - done) / rate if rate else float("nan")
                log.info(
                    "transformers: %d/%d prompts (%.1f%%) %.2f prompt/s, eta %.1f min",
                    done, len(wrapped), 100 * done / len(wrapped), rate, remaining / 60,
                )
        cost.add(requests=len(wrapped), completions=len(wrapped),
                 prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        return results

    def close(self) -> None:
        if self.activation_editor is not None:
            self.activation_editor.detach()
        self._model = None
        _free_gpu()


def _count_vllm(outputs) -> None:
    """Requests and tokens of one vLLM call, for the step's cost record (utils/cost.py):
    throughput and cost per thousand decisions are computed from them."""
    if not cost.tracked():
        return
    try:
        cost.add(
            requests=len(outputs),
            completions=sum(len(o.outputs) for o in outputs),
            prompt_tokens=sum(len(o.prompt_token_ids or ()) for o in outputs),
            output_tokens=sum(len(c.token_ids) for o in outputs for c in o.outputs),
        )
    except Exception as e:  # a renamed vLLM field must not stop a generation
        log.warning("cost: could not count the tokens of a vLLM call: %r", e)


def _note_prefix_cache(llm) -> None:
    """vLLM's prefix-cache counters so far, in prompt tokens, for the step's cost record:
    `prefix_cache_queries` looked up, `prefix_cache_hits` found (docs/THROUGHPUT_B.md).

    They count over the engine's life, so they are notes (the last value wins), not counters.
    `LLM.get_metrics` works only with stat logging on (`engine_kwargs.disable_log_stats:
    false`); an engine started without it, which is every audit's, notes nothing.
    """
    if not cost.tracked():
        return
    try:
        metrics = llm.get_metrics()
    except Exception:  # stat logging off, or an engine without metrics
        return
    totals = {"vllm:prefix_cache_queries": 0, "vllm:prefix_cache_hits": 0}
    for metric in metrics:
        if metric.name in totals:
            totals[metric.name] += int(getattr(metric, "value", 0) or 0)
    cost.note(prefix_cache_queries=totals["vllm:prefix_cache_queries"],
              prefix_cache_hits=totals["vllm:prefix_cache_hits"])


def _free_gpu() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def build_backend(cfg: dict, activation_editor=None) -> Backend:
    """Instantiates the backend a run config asks for."""
    kind = cfg.get("backend", "vllm")
    generation = GenerationConfig(**cfg.get("generation", {}))
    common = {
        "model": cfg["model"],
        "generation": generation,
        "lora_path": cfg.get("lora_path"),
        "chat_template_kwargs": cfg.get("chat_template_kwargs"),
    }
    if activation_editor is not None and kind != "transformers":
        raise ValueError(
            "the embedding mitigation edits activations through forward hooks, which needs "
            "backend: transformers. Set it in the run config."
        )
    if kind == "vllm":
        return VLLMBackend(
            **common,
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.90),
            max_model_len=cfg.get("max_model_len", 8192),
            dtype=cfg.get("dtype", "auto"),
            max_num_seqs=cfg.get("max_num_seqs"),
            engine_kwargs=cfg.get("engine_kwargs"),
        )
    if kind == "transformers":
        return TransformersBackend(
            **common,
            dtype=cfg.get("dtype", "bfloat16"),
            batch_size=cfg.get("batch_size", 8),
            activation_editor=activation_editor,
        )
    raise ValueError(f"unknown backend {kind!r}; expected 'vllm' or 'transformers'")


def resolve_hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
