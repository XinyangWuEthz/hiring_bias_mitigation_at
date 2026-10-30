"""Mitigation-layer tests. No GPU: the scrubber is pure text, the erasers are pure linear
algebra, and the runner is exercised against a stub backend.
"""

import pathlib
import typing

import numpy as np
import pytest

from hiring_bias_mitigation.eval import runner as R
from hiring_bias_mitigation.mitigation import embedding as E
from hiring_bias_mitigation.mitigation import prompt as PM
from hiring_bias_mitigation.mitigation.scrub import LexicalScrubber


class StubBackend:
    """Records the prompts it was handed; answers everything with a hire."""

    def __init__(self):
        self.prompts = []

    def generate(self, prompts):
        self.prompts = list(prompts)
        return ['{"decision": "hire", "feedback": "Relevant experience for the role."}'] * len(
            prompts
        )

    def close(self):
        pass


# ---- scrubbing ---------------------------------------------------------------------------

def test_scrubber_removes_explicit_field_and_first_person_statements_en():
    scrubber = LexicalScrubber("en")
    result = scrubber.scrub(
        "Candidate's military status: War veteran\n"
        "I am a war veteran and now look for a civilian job. "
        "I identify as non-binary please use pronouns they/them.\n"
        "8 years of Python, led a team of six."
    )
    assert "veteran" not in result.text.lower()
    assert "non-binary" not in result.text.lower()
    assert "8 years of Python" in result.text  # professional content survives
    assert len(result.removals) >= 3


def test_scrubber_handles_ukrainian_inflection():
    result = LexicalScrubber("uk").scrub(
        "військовий статус кандидата: Ветеран війни\n"
        "Я ветеран війни і зараз шукаю цивільну роботу.\n"
        "Досвід 8 років Python."
    )
    assert "ветеран" not in result.text.lower()
    assert "Python" in result.text


def test_scrubber_is_a_noop_on_clean_text():
    result = LexicalScrubber("en").scrub("Six years of Kubernetes and Go. Led two migrations.")
    assert not result.changed


# ---- erasure -----------------------------------------------------------------------------

@pytest.fixture
def separable_activations():
    rng = np.random.default_rng(0)
    direction = rng.normal(size=64)
    direction /= np.linalg.norm(direction)
    labels = rng.integers(0, 3, 600)
    return rng.normal(size=(600, 64)) + np.outer(labels - 1, direction) * 3.0, labels


@pytest.mark.parametrize("method", E.METHODS)
def test_erasure_removes_the_linear_signal(method, separable_activations):
    activations, labels = separable_activations
    before = E.linear_probe_accuracy(activations, labels)
    eraser = E.fit_eraser(activations, labels, method, layers=(10,))
    erased = eraser.mean + (activations - eraser.mean) @ eraser.projection.T
    after = E.linear_probe_accuracy(erased, labels)
    assert before > 0.6
    assert after < before - 0.3


def test_leace_changes_the_representation_least(separable_activations):
    """LEACE's guarantee: erase the concept with the least-squares-smallest edit."""
    activations, labels = separable_activations

    def relative_change(method):
        eraser = E.fit_eraser(activations, labels, method, layers=(10,))
        erased = eraser.mean + (activations - eraser.mean) @ eraser.projection.T
        return np.linalg.norm(erased - activations) / np.linalg.norm(activations)

    assert relative_change("leace") <= relative_change("inlp")


def test_eraser_roundtrips_through_disk(tmp_path, separable_activations):
    activations, labels = separable_activations
    eraser = E.fit_eraser(activations, labels, "leace", layers=(3, 4))
    reloaded = E.Eraser.load(eraser.save(tmp_path / "e.npz"))
    assert reloaded.layers == (3, 4)
    assert reloaded.method == "leace"
    np.testing.assert_allclose(reloaded.projection, eraser.projection)


def test_unknown_erasure_method_is_rejected(separable_activations):
    activations, labels = separable_activations
    with pytest.raises(ValueError, match="unknown erasure method"):
        E.fit_eraser(activations, labels, "wishful_thinking", layers=(1,))


# ---- prompt registry ----------------------------------------------------------------------

def test_prompt_registry_rejects_unknown_strategies():
    with pytest.raises(ValueError, match="unknown prompt strategy"):
        PM.validate("be_nicer")


def test_prompt_provenance_is_recorded():
    assert PM.describe("zero_shot_cot")["provenance"] == "reproduced from the audit study"
    assert PM.describe("fairness_constitution")["provenance"] == "new in this work"


# ---- runner wiring ------------------------------------------------------------------------

def _cfg(mitigation: dict) -> dict:
    return {
        "model": "stub/model",
        "lang": "en",
        "protected_groups": ["military_status"],
        "conditions": ("explicit", "implicit", "attr_free"),
        "limit_pairs": 2,
        "mitigation": mitigation,
    }


def test_baseline_prompts_contain_the_attribute():
    cfg = _cfg({"family": "none"})
    df, _ = R._apply_mitigation_to_frame(R.build_frame(cfg), cfg, StubBackend())
    prompts = R._build_prompts(df, "baseline")
    explicit = [p for p, c in zip(prompts, df["condition"]) if c == "explicit"]
    implicit = [p for p, c in zip(prompts, df["condition"]) if c == "implicit"]
    assert any("military status" in p for p in explicit)
    assert any("veteran" in p.lower() for p in implicit)


def test_scrubbing_removes_the_attribute_from_both_conditions():
    """The failure this guards against: the implicit sentence is re-injected at prompt time
    after the scrubber ran, and the scrubbed run silently equals the baseline."""
    cfg = _cfg({"family": "scrub", "mode": "lexical"})
    df, meta = R._apply_mitigation_to_frame(R.build_frame(cfg), cfg, StubBackend())
    prompts = R._build_prompts(df, "baseline")
    for prompt, condition in zip(prompts, df["condition"]):
        if condition == "attr_free":
            continue
        lowered = prompt.lower()
        assert "military status:" not in lowered
        assert "veteran" not in lowered
        assert "combat" not in lowered
    assert meta["scrub_rows_total"] == len(df)


def test_run_name_is_stable_and_encodes_the_mitigation():
    assert R.build_run_name(_cfg({"family": "none"})) == "model--en--baseline"
    assert (
        R.build_run_name(_cfg({"family": "prompt", "strategy": "structured_rubric"}))
        == "model--en--prompt--structured_rubric"
    )


def test_embedding_family_refuses_the_vllm_backend():
    from hiring_bias_mitigation.eval.backends import build_backend

    with pytest.raises(ValueError, match="backend: transformers"):
        build_backend({"model": "x", "backend": "vllm"}, activation_editor=object())


def test_every_mitigation_family_honours_the_eval_scope():
    """Erasure evaluated the full group set while prompt and scrub honoured the scope.

    Two costs, both silent: the run was 2.4x larger than the plan asked for, and its cells did
    not match the other families' for the same model and language, so the arms could not be
    compared on equal footing. Any family that grows a config path has to pass through the
    same scoping.
    """
    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    scope = yaml.safe_load((root / "reports" / "eval_scope.yaml").read_text()) or {}
    groups = scope.get("groups") or scope

    for slug, per_lang in groups.items():
        for lang, expected in per_lang.items():
            for family in ("prompt", "scrub", "embedding"):
                for config in (root / "configs" / "mitigation" / family).glob(
                    f"{slug}_{lang}_*.yaml"
                ):
                    # `*_fullscope.yaml` opts out on purpose: the 9B-English prompt runs were
                    # scoped to the one confirmed cell, which made the prompt-versus-SFT
                    # comparison there narrower than in every other cell. These repeat three
                    # strategies over the full grid and carry their own run names, so the
                    # scoped runs stay untouched.
                    if config.stem.endswith("_fullscope"):
                        continue
                    actual = yaml.safe_load(config.read_text())["protected_groups"]
                    assert sorted(actual) == sorted(expected), (
                        f"{config.relative_to(root)} evaluates {sorted(actual)}, "
                        f"scope says {sorted(expected)}"
                    )


def test_regeneration_defaults_to_the_scoped_grid():
    """A plain regeneration must not widen every config back to the full grid.

    That is how the scope was lost: `--eval-scope` was opt-in, so a routine regeneration
    silently restored the full group set across every family at once.
    """
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "scripts/generate_experiment_configs.py", "--help"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout

    assert "--full-grid" in out, "opting out of the scope must be explicit"


def test_activation_cache_key_ignores_method_but_not_the_inputs():
    """LEACE, INLP and mean-diff share one activation matrix; changed inputs must not.

    Collection is a multi-hour forward pass whose result depends on the model, language,
    layers and prompts — never on the erasure method. Keying on the method would recollect it
    three times per model and language; keying on too little would hand back a matrix that no
    longer matches the fitting set.
    """
    import importlib.util
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("fit_eraser", root / "scripts" / "fit_eraser.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fit_eraser"] = module
    spec.loader.exec_module(module)

    source = (root / "scripts" / "fit_eraser.py").read_text()
    body = source[source.index("def _activations("):source.index("def main()")]
    key_block = body[body.index("key = json.dumps("):body.index("sort_keys=True")]

    assert '"method"' not in key_block, "method must not be part of the cache key"
    for field in ('"model"', '"lang"', '"layers"', '"n_rows"', '"prompt_digest"'):
        assert field in key_block, f"{field} must be part of the cache key"


def test_adapter_directory_resolves_to_base_plus_adapter(tmp_path):
    """DPO continuing from a LoRA SFT run must not try to load weights that were never saved.

    `trainer.save_model` on a PEFT model writes the adapter alone, so pointing
    `AutoModelForCausalLM.from_pretrained` at that directory fails. Three of the ten enabled
    DPO runs chained from a LoRA checkpoint and would have died at load time — after their
    SFT stage had already spent the GPU hours.
    """
    from hiring_bias_mitigation.mitigation.training_common import _resolve_adapter

    plain = _resolve_adapter("Qwen/Qwen3.5-4B")
    assert plain == ("Qwen/Qwen3.5-4B", None)

    ckpt = tmp_path / "sft_run"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "Qwen/Qwen3.5-4B", "r": 32}', encoding="utf-8"
    )
    source, adapter = _resolve_adapter(str(ckpt))
    assert source == "Qwen/Qwen3.5-4B"
    assert adapter == str(ckpt)


def test_adapter_without_a_recorded_base_is_an_error(tmp_path):
    """Silently falling back to the adapter path would reintroduce the original crash."""
    import pytest as _pytest

    from hiring_bias_mitigation.mitigation.training_common import _resolve_adapter

    ckpt = tmp_path / "sft_run"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    with _pytest.raises(ValueError, match="base_model_name_or_path"):
        _resolve_adapter(str(ckpt))


def test_trained_audits_cover_every_group_training_touched():
    """Training moves weights against all three groups, so all three must be measured.

    Prompt and scrub are scoped to the confirmed-bias cells because they are aimed at those
    cells. Training is not aimed: scoping its audit would make collateral damage in an
    untargeted group unobservable rather than absent.
    """
    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    expected = {"military_status", "gender", "religion"}
    configs = list((root / "configs" / "audit").glob("*_sft_*.yaml"))
    configs += list((root / "configs" / "audit").glob("*_dpo_*.yaml"))
    assert configs, "no trained-adapter audit configs found"

    for config in configs:
        payload = yaml.safe_load(config.read_text())
        groups = set(payload["protected_groups"])
        assert expected <= groups, f"{config.name} misses {expected - groups}"
        assert len(payload["conditions"]) == 3, f"{config.name} drops a condition"


def test_warmup_ratio_is_translated_when_transformers_dropped_it():
    """transformers 5.x removed `warmup_ratio`; passing it raises from TrainingArguments.

    That single unsupported keyword took out all six SFT runs and all ten DPO runs in ninety
    seconds, and then the sixteen audits that depended on their checkpoints. One number in the
    config has to keep meaning the same thing across both major versions.
    """
    import inspect

    from transformers import TrainingArguments

    from hiring_bias_mitigation.mitigation.training_common import _warmup_kwargs

    cfg = {"warmup_ratio": 0.03, "per_device_train_batch_size": 2,
           "gradient_accumulation_steps": 8, "num_train_epochs": 2}
    kwargs = _warmup_kwargs(cfg, n_train=8419)

    supported = inspect.signature(TrainingArguments.__init__).parameters
    assert set(kwargs) <= set(supported), f"{set(kwargs) - set(supported)} would raise"

    if "warmup_ratio" not in supported:
        # 8419 rows / batch 16 = 527 steps/epoch x 2 epochs; 3% of 1054 is 32.
        assert kwargs == {"warmup_steps": 32}


def test_warmup_steps_count_every_rank_under_torchrun(monkeypatch):
    """One optimizer step under torchrun takes a batch from every rank. Counting one rank, the
    4-GPU SFT-v3 run warmed up for 471 of its 3,923 steps, not 3% of them (job 5107507)."""
    import inspect

    from transformers import TrainingArguments

    from hiring_bias_mitigation.mitigation.training_common import _warmup_kwargs

    if "warmup_ratio" in inspect.signature(TrainingArguments.__init__).parameters:
        pytest.skip("this transformers takes warmup_ratio itself")
    cfg = {"warmup_ratio": 0.03, "per_device_train_batch_size": 1,
           "gradient_accumulation_steps": 4, "num_train_epochs": 1}
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _warmup_kwargs(cfg, n_train=62761) == {"warmup_steps": 118}
    monkeypatch.delenv("WORLD_SIZE")
    assert _warmup_kwargs(cfg, n_train=62761) == {"warmup_steps": 471}


def test_training_arguments_construct_for_every_enabled_training_config():
    """Build the real TrainingArguments for each config, without training anything.

    A signature mismatch is a whole-stage failure, not a single-run one, and it costs nothing
    to find here instead of after the queue has reached the training stage.
    """
    import re

    import yaml
    from transformers import TrainingArguments  # noqa: F401

    from hiring_bias_mitigation.mitigation import training_common as C

    root = pathlib.Path(__file__).resolve().parents[1]
    configs = []
    for runner in ("run_all_sft.sh", "run_all_dpo.sh"):
        for line in (root / "scripts" / runner).read_text().splitlines():
            match = re.match(r"^  (configs/\S+\.yaml)", line)
            if match:
                configs.append(root / match.group(1))
    assert configs, "no training configs enabled"

    for config in configs:
        cfg = yaml.safe_load(config.read_text())
        args = C.training_arguments(cfg, pathlib.Path("/tmp/hbm-test"), has_eval=True,
                                    n_train=1000)
        assert args.num_train_epochs == cfg.get("num_train_epochs", 2)


def test_audit_refuses_a_missing_adapter_before_loading_a_model(tmp_path, monkeypatch):
    """vLLM reads the adapter only after loading the base model -- minutes too late."""
    import pytest as _pytest

    from hiring_bias_mitigation.eval import runner as R

    monkeypatch.setattr(R, "resolve_output_path", lambda p: str(tmp_path / p))

    with _pytest.raises(FileNotFoundError, match="directory does not exist"):
        R._require_adapter("outputs/sft/never_trained")

    empty = tmp_path / "outputs/sft/half_written"
    empty.mkdir(parents=True)
    with _pytest.raises(FileNotFoundError, match=r"no adapter_config\.json"):
        R._require_adapter("outputs/sft/half_written")

    (empty / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert R._require_adapter("outputs/sft/half_written") == str(empty)


class _FakeTokenizer:
    """Mimics Qwen's template: the assistant turn always opens with a thinking block.

    The awkward part this reproduces is that a generation prompt stops *inside* that block
    (`<think>\n`) while the full render closes it (`<think>\n\n</think>\n\n`) — the same
    string prefix, different tokens, which is what makes naive splitting unsafe.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            **kwargs):
        thinking = kwargs.get("enable_thinking", True)
        opener = "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        out = ""
        for message in messages:
            if message["role"] == "assistant":
                out += f"<|im_start|>assistant\n{opener}{message['content']}<|im_end|>\n"
            else:
                out += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n" + opener
        return out


def test_render_chat_splits_exactly_at_the_prompt_boundary():
    """Concatenating the two halves must reproduce the template's own output.

    If it does not, completion-only loss is masked on a guess. TRL reports that as "Mismatch
    between tokenized prompt and the start of tokenized prompt+completion" and proceeds
    anyway, so the run looks fine while training on the wrong tokens.
    """
    from datasets import Dataset

    from hiring_bias_mitigation.mitigation.sft import render_chat

    ds = Dataset.from_dict({"prompt": ["JOB"], "completion": ['{"d":"hire"}'], "lang": ["uk"]})
    out = render_chat(ds, _FakeTokenizer(), {"enable_thinking": False})[0]

    assert set(out) == {"prompt", "completion"}, "provenance columns must not reach the trainer"
    full = _FakeTokenizer().apply_chat_template(
        [{"role": "user", "content": "JOB"},
         {"role": "assistant", "content": '{"d":"hire"}'}],
        enable_thinking=False,
    )
    assert out["prompt"] + out["completion"] == full
    assert out["completion"], "an empty completion would train on nothing"


def test_render_chat_defaults_to_the_audits_template_kwargs():
    """Training must emit the format the audit later asks for.

    The audit runs with enable_thinking=False. Training through a template that inserts a
    thinking block teaches a shape the evaluation never requests.
    """
    from hiring_bias_mitigation.eval.backends import DEFAULT_CHAT_TEMPLATE_KWARGS
    from hiring_bias_mitigation.mitigation import sft as S

    assert S.DEFAULT_CHAT_TEMPLATE_KWARGS is DEFAULT_CHAT_TEMPLATE_KWARGS
    assert DEFAULT_CHAT_TEMPLATE_KWARGS == {"enable_thinking": False}


def test_render_chat_refuses_a_template_that_is_not_prefix_consistent():
    """Silently continuing here is how the loss ends up masked on the wrong tokens."""
    import pytest as _pytest
    from datasets import Dataset

    from hiring_bias_mitigation.mitigation.sft import render_chat

    class Broken(_FakeTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                **kwargs):
            if add_generation_prompt:
                return "PROMPT-ONLY-SHAPE"
            return "COMPLETELY-DIFFERENT"

    ds = Dataset.from_dict({"prompt": ["JOB"], "completion": ["OK"]})
    with _pytest.raises(ValueError, match="prefix of"):
        render_chat(ds, Broken(), {})


def test_a_bad_wandb_entity_disables_logging_instead_of_killing_the_run(monkeypatch):
    """Logging is not the experiment, and must get no chance to end one.

    A mistyped WANDB_ENTITY killed all six SFT runs at `on_train_begin` — after the model was
    loaded and the dataset tokenised. `WANDB_MODE=offline` was not enough: the client still
    resolved the configured entity and raised `entity ... not found during upsertBucket`.
    """
    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("WANDB_API_KEY", "k" * 45)
    monkeypatch.setenv("WANDB_ENTITY", "does-not-exist")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class _Viewer:
        entity = "real-entity"
        teams: typing.ClassVar[list] = ["real-team"]

    class _Api:
        def __init__(self, *a, **k):
            pass

        viewer = _Viewer()

    monkeypatch.setitem(__import__("sys").modules, "wandb", type("W", (), {"Api": _Api}))

    C.wandb_init({"wandb_project": "p"})

    import os

    assert os.environ["WANDB_MODE"] == "disabled"
    assert "WANDB_ENTITY" not in os.environ, "the name it failed to resolve must be gone"


def test_wandb_is_left_alone_when_the_entity_is_valid(monkeypatch):
    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("WANDB_API_KEY", "k" * 45)
    monkeypatch.setenv("WANDB_ENTITY", "real-entity")
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class _Viewer:
        entity = "real-entity"
        teams: typing.ClassVar[list] = []

    class _Api:
        def __init__(self, *a, **k):
            pass

        viewer = _Viewer()

    monkeypatch.setitem(__import__("sys").modules, "wandb", type("W", (), {"Api": _Api}))

    C.wandb_init({"wandb_project": "p"})

    import os

    assert os.environ.get("WANDB_MODE") != "disabled"
    assert os.environ["WANDB_ENTITY"] == "real-entity"


def test_sequence_budget_clears_the_longest_example_in_each_language():
    """Truncation cuts the completion — the label — not just some context.

    Measured on the generated data: English tops out near 1,540 tokens, Ukrainian near 2,370,
    and 8.7% of Ukrainian exceeds 1,536. A budget chosen from the average silently trains a
    fraction of the Ukrainian runs on half a JSON object.
    """
    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    observed_max = {"en": 1536, "uk": 2372}

    for config in (root / "configs" / "mitigation" / "sft").glob("*_only.yaml"):
        payload = yaml.safe_load(config.read_text())
        lang = "uk" if "uk" in config.stem else "en" if "en" in config.stem else None
        if lang is None:
            continue
        assert payload["max_seq_len"] >= observed_max[lang], (
            f"{config.name} budgets {payload['max_seq_len']} tokens for {lang}, "
            f"below the longest observed example ({observed_max[lang]})"
        )


def test_gradient_checkpointing_is_on_wherever_memory_needs_it():
    """It costs about a third of the step time and changes nothing about the result.

    Two things need it, and the second was learned the expensive way: the 12B targets, and any
    arm using the decision-weighted loss. That loss forces `loss_type: nll`, which materialises
    the full ~250k-vocab logits, and the first evaluation of such a run was OOM-killed at step
    50 with ~11 GB of headroom — evaluation is the memory peak, not training.
    """
    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    for config in (root / "configs" / "mitigation" / "sft").glob("*.yaml"):
        payload = yaml.safe_load(config.read_text())
        needs_it = "12b" in config.stem or payload.get("decision_loss_weight", 1.0) != 1.0
        assert payload["gradient_checkpointing"] is needs_it, (
            f"{config.name}: checkpointing should be {'on' if needs_it else 'off'} here"
        )


def test_full_logit_loss_runs_with_a_small_eval_batch():
    """Evaluation is the memory peak: the same full-logit forward with no backward to amortise
    it, at a batch size transformers defaults to 4."""
    import yaml

    root = pathlib.Path(__file__).resolve().parents[1]
    for config in (root / "configs" / "mitigation" / "sft").glob("*.yaml"):
        payload = yaml.safe_load(config.read_text())
        if payload.get("decision_loss_weight", 1.0) == 1.0:
            continue
        assert payload.get("per_device_eval_batch_size", 4) <= 2, (
            f"{config.name}: a full-logit loss needs a small eval batch"
        )


def test_weights_are_never_uploaded_to_wandb(monkeypatch):
    """Metrics go to W&B; the adapters stay on the local model drive.

    Set explicitly rather than left unset: an inherited WANDB_LOG_MODEL from the shell or .env
    would otherwise turn uploads back on without anyone choosing it.
    """
    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("WANDB_API_KEY", "k" * 45)
    monkeypatch.setenv("WANDB_LOG_MODEL", "end")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    class _Viewer:
        entity = "me"
        teams: typing.ClassVar[list] = []

    class _Api:
        def __init__(self, *a, **k):
            pass

        viewer = _Viewer()

    monkeypatch.setitem(__import__("sys").modules, "wandb", type("W", (), {"Api": _Api}))

    C.wandb_init({"wandb_project": "p"})

    import os

    assert os.environ["WANDB_LOG_MODEL"] == "false"


def test_early_stopping_is_on_by_default_and_counts_evaluations():
    """Measured on this data: eval loss bottoms at step 300 of 1054 and rises after.

    Without a stop, 700 of the 1054 steps produce checkpoints that are then discarded —
    `load_best_model_at_end` restores step 300 regardless. The saving is GPU time, not
    accuracy.
    """
    from hiring_bias_mitigation.mitigation import training_common as C

    callbacks = C.early_stopping_callbacks({"eval_steps": 100}, has_eval=True)
    assert len(callbacks) == 1
    assert callbacks[0].early_stopping_patience == 3

    # Patience counts evaluations, so the step budget follows eval_steps.
    tighter = C.early_stopping_callbacks(
        {"eval_steps": 50, "early_stopping_patience": 2}, has_eval=True
    )
    assert tighter[0].early_stopping_patience == 2


def test_early_stopping_is_skipped_without_a_validation_split():
    """There is nothing to stop on, and requesting it would raise inside the trainer."""
    from hiring_bias_mitigation.mitigation import training_common as C

    assert C.early_stopping_callbacks({}, has_eval=False) == []
    assert C.early_stopping_callbacks({"early_stopping_patience": 0}, has_eval=True) == []


def test_decision_weighted_loss_moves_gradient_onto_the_decision():
    """The audit measures the decision; plain CE gave it 16% of the gradient.

    Measured on this data: a completion is ~38 tokens of which the decision field is 6. The
    first SFT run halved the attribute-mention rate and left every acceptance-rate disparity
    unchanged — the model learned to write like the teacher without learning to decide like it.
    """
    import torch

    from hiring_bias_mitigation.mitigation.sft import decision_weighted_loss

    class _Out:
        pass

    torch.manual_seed(0)
    vocab, length = 11, 12
    out = _Out()
    out.logits = torch.randn(2, length, vocab)
    labels = torch.full((2, length), -100)
    labels[:, 4:] = torch.randint(0, vocab, (2, length - 4))

    weighted = float(decision_weighted_loss(8.0, 6)(out, labels))
    flat = float(decision_weighted_loss(1.0, 6)(out, labels))
    assert weighted != flat, "weighting must change the loss"

    # Weight 1.0 must reproduce the plain masked mean exactly.
    import torch.nn.functional as F

    per_token = F.cross_entropy(
        out.logits[..., :-1, :].transpose(1, 2), labels[..., 1:], reduction="none",
        ignore_index=-100,
    )
    mask = labels[..., 1:] != -100
    assert abs(flat - float((per_token * mask).sum() / mask.sum())) < 1e-5


def test_decision_weighted_loss_ignores_prompt_tokens():
    """Prompt positions are -100 and must contribute nothing, weighted or not."""
    import torch

    from hiring_bias_mitigation.mitigation.sft import decision_weighted_loss

    class _Out:
        pass

    torch.manual_seed(1)
    out = _Out()
    out.logits = torch.randn(1, 9, 7)
    labels = torch.full((1, 9), -100)
    labels[:, 5:] = torch.randint(0, 7, (1, 4))

    before = float(decision_weighted_loss(8.0, 2)(out, labels))
    out.logits[0, :3, :] += 50.0          # scramble prompt positions only
    after = float(decision_weighted_loss(8.0, 2)(out, labels))
    assert abs(before - after) < 1e-5, "masked prompt tokens leaked into the loss"


def test_balancing_can_target_the_benchmarks_own_rate():
    """Equalising to 50/50 replaces the teacher's skew with a different one.

    The teacher pool is 89% reject; the benchmark is 34% hire. Training at 50/50 moves the
    student's operating point away from the distribution it is judged on, and that shift lands
    in the acceptance-rate columns as if the mitigation had caused it.
    """
    import pandas as pd

    from hiring_bias_mitigation.generation.filters import balance_decisions

    frame = pd.DataFrame({"decision": ["reject"] * 891 + ["hire"] * 109})

    equal, _ = balance_decisions(frame, column="decision")
    assert equal["decision"].value_counts(normalize=True)["hire"] == pytest.approx(0.5)

    matched, _ = balance_decisions(frame, column="decision", target_positive_rate=0.34)
    assert matched["decision"].value_counts(normalize=True)["hire"] == pytest.approx(
        0.34, abs=0.01
    )
    assert len(matched) > len(equal), "matching a realistic rate should keep more data"


def test_custom_loss_forces_a_loss_type_trl_can_pair_with_it():
    """TRL's default `chunked_nll` reads fields only its own patched forward attaches.

    With a custom `compute_loss_func` that forward is bypassed, so the entropy branch hits
    `outputs.num_valid_tokens` and the run dies on the first training step — after the model is
    loaded and the dataset tokenised. `nll` is the same maths without the chunked lm_head
    projection.
    """
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src" / "hiring_bias_mitigation" / "mitigation" / "sft.py").read_text()
    assert '"loss_type": "nll" if weight != 1.0 else cfg.get("loss_type")' in source


def test_preflight_can_smoke_a_named_config():
    """Smoking a different config than the one about to train is how that reached the GPU."""
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parents[1]
    out = subprocess.run(
        [sys.executable, "scripts/preflight_training.py", "--help"],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    assert "--config" in out


def test_memory_guard_stops_and_saves_below_the_floor(monkeypatch):
    """A run that stops itself loses a few steps; one the kernel kills loses the machine.

    Observed: a run grew for eight hours until the kernel OOM-killed system services and the
    NVIDIA driver failed allocations. The run died without writing a final adapter.
    """
    from transformers import TrainerControl, TrainerState

    from hiring_bias_mitigation.mitigation import training_common as C

    guard = C._memory_guard_class()(floor_gb=12.0)
    state, control = TrainerState(), TrainerControl()
    state.global_step = 416

    monkeypatch.setattr(C, "_mem_available_gb", lambda: 40.0)
    guard.on_evaluate(None, state, control)
    assert not control.should_training_stop

    monkeypatch.setattr(C, "_mem_available_gb", lambda: 6.0)
    returned = guard.on_evaluate(None, state, control)
    assert control.should_training_stop and control.should_save
    assert returned is control, "transformers uses the returned control object"


def test_label_shift_loss_needs_no_copy_of_the_logits():
    """Shifting the labels instead of the logits keeps the alignment and drops two copies of a
    [batch, seq, ~250k] tensor per step — the copies that fed an eight-hour memory creep."""
    source = (pathlib.Path(__file__).resolve().parents[1]
              / "src" / "hiring_bias_mitigation" / "mitigation" / "sft.py").read_text()
    start = source.index("def decision_weighted_loss(")
    region = source[start:source.index("def decision_token_count(")]
    # Code only: the docstring and comments name the old approach on purpose.
    body = "\n".join(
        line for line in region.splitlines()
        if line.strip() and not line.strip().startswith("#")
        and '"""' not in line
    )
    assert "logits[..., :-1" not in body, "slicing the logits copies them"
    assert ".transpose(" not in body, "transposing the logits copies them"
    assert "F.pad(labels" in body


def test_training_configures_the_allocator_before_cuda():
    for name in ("sft.py", "dpo.py"):
        source = (pathlib.Path(__file__).resolve().parents[1]
                  / "src" / "hiring_bias_mitigation" / "mitigation" / name).read_text()
        main = source[source.index("def main() -> None:"):]
        assert main.splitlines()[1].strip() == "C.configure_cuda_allocator()", name


def test_dpo_rendering_splits_each_response_exactly_at_the_prompt():
    """The DPO objective is a log-probability *difference* between chosen and rejected.

    If the prompt/response boundary is guessed, that difference is taken partly over prompt
    tokens, which are identical on both sides and dilute the margin toward zero.
    """
    from datasets import Dataset

    from hiring_bias_mitigation.mitigation.dpo import _to_conversational

    tok = _FakeTokenizer()
    ds = Dataset.from_dict({"prompt": ["JOB"], "chosen": ['{"d":"hire"}'],
                           "rejected": ['{"d":"reject"}'], "lang": ["en"]})
    row = _to_conversational(ds, tok, {"enable_thinking": False})[0]

    assert set(row) == {"prompt", "chosen", "rejected"}
    for side, text in (("chosen", '{"d":"hire"}'), ("rejected", '{"d":"reject"}')):
        full = tok.apply_chat_template(
            [{"role": "user", "content": "JOB"}, {"role": "assistant", "content": text}],
            enable_thinking=False,
        )
        assert row["prompt"] + row[side] == full
    assert row["chosen"] != row["rejected"]


def _unreachable_wandb(monkeypatch):
    class _Api:
        def __init__(self, *a, **k):
            raise ConnectionError("no route to api.wandb.ai")

    monkeypatch.setitem(__import__("sys").modules, "wandb", type("W", (), {"Api": _Api}))


def test_an_unreachable_wandb_logs_offline_for_a_later_sync(monkeypatch):
    """A compute node may have no route to W&B. Disabling would lose the curves; offline keeps
    them for `wandb sync`, and dropping the entity leaves the client nothing to resolve."""
    import os

    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("WANDB_API_KEY", "k" * 86)
    monkeypatch.setenv("WANDB_ENTITY", "real-entity")
    monkeypatch.delenv("WANDB_MODE", raising=False)
    _unreachable_wandb(monkeypatch)

    C.wandb_init({"wandb_project": "p"})

    assert os.environ["WANDB_MODE"] == "offline"
    assert "WANDB_ENTITY" not in os.environ


def test_reachability_is_probed_without_an_entity_too(monkeypatch):
    import os

    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("WANDB_API_KEY", "k" * 86)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)
    _unreachable_wandb(monkeypatch)

    C.wandb_init({"wandb_project": "p"})

    assert os.environ["WANDB_MODE"] == "offline"


def test_a_resumed_run_keeps_logging_to_one_wandb_run(tmp_path, monkeypatch):
    """A run split into time-limited segments is one curve, not one W&B run per segment."""
    import os

    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)
    C.wandb_resume({"resume": True}, tmp_path)
    first = os.environ["WANDB_RUN_ID"]
    assert (tmp_path / "wandb_run_id").read_text().strip() == first
    assert os.environ["WANDB_RESUME"] == "allow"

    monkeypatch.delenv("WANDB_RUN_ID")
    C.wandb_resume({"resume": True}, tmp_path)
    assert os.environ["WANDB_RUN_ID"] == first

    (tmp_path / "wandb_run_id").write_text("mc7j4yp4\n")  # an earlier segment's offline run
    C.wandb_resume({"resume": True}, tmp_path)
    assert os.environ["WANDB_RUN_ID"] == "mc7j4yp4"


def test_a_run_that_does_not_resume_gets_a_fresh_wandb_run(tmp_path, monkeypatch):
    import os

    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    C.wandb_resume({}, tmp_path)
    C.wandb_resume({"resume": True, "use_wandb": False}, tmp_path)
    assert "WANDB_RUN_ID" not in os.environ
    assert not (tmp_path / "wandb_run_id").exists()


def test_the_probe_does_not_freeze_wandb_settings_before_the_trainer_starts(tmp_path, monkeypatch):
    """wandb.Api() loads wandb's process-wide settings from the environment once; wandb.init
    later copies them. The run id, resume flag and offline fallback set after the probe must
    still reach the run the trainer starts (review of 16f3f16)."""
    import os

    import wandb
    from wandb.sdk import wandb_setup

    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("WANDB_API_KEY", "k" * 86)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("WANDB_RESUME", raising=False)

    class _Api:
        def __init__(self, *a, **k):
            _ = wandb_setup.singleton().settings  # what the real Api does first
            raise ConnectionError("no route to api.wandb.ai")

    monkeypatch.setattr(wandb, "Api", _Api)
    C.wandb_init({"wandb_project": "p"})
    C.wandb_resume({"resume": True}, tmp_path)
    run = wandb.init(project="p")
    try:
        assert run.settings.mode == "offline"
        assert run.id == (tmp_path / "wandb_run_id").read_text().strip()
    finally:
        run.finish()
        wandb.teardown()
    assert os.environ["WANDB_MODE"] == "offline"
