"""CPU integration checks for the real diagnostic output contract.

Only CUDA/LLM/distributed effects are faked. The production score_live,
check_request_outputs, atomic, parquet I/O, collect_live, write_arm, fingerprint,
official parser and summarize functions execute unchanged. Synthetic fixtures stay
under pytest temporary directories and are never presented as experiment results.

Override DIAGNOSTIC_SCRIPT to review a newer candidate without changing this test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

WORK = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = WORK / "scripts/grpo_425_diagnostic.py"
PARSER = Path(
    os.environ.get(
        "OFFICIAL_PARSER",
        WORK / "src/hiring_bias_mitigation/eval/parsing.py",
    )
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def harness(tmp_path, monkeypatch):
    script = Path(os.environ.get("DIAGNOSTIC_SCRIPT", DEFAULT_SCRIPT))
    diagnostic = load("live_contract_diagnostic", script)
    parser = load("live_contract_official_parser", PARSER)
    events = []
    options = {"bad_tokens": False, "warmup_inf": False, "flip": None, "invalid_decision": False}
    n = 31050
    rows = pd.DataFrame(
        {
            "prompt_id": [f"row-{i:05d}" for i in range(n)],
            "prompt": [f"prompt-{i}" for i in range(n)],
            "pair_id": [f"pair-{i}" for i in range(n)],
            "candidate_id": [f"candidate-{i // 69}" for i in range(n)],
            "job_id": [f"job-{i // 69}" for i in range(n)],
            "protected_group": "gender",
            "protected_attr": "synthetic",
            "condition": "attr_free",
            "lang": "en",
            "reference_decision": ["hire" if i < 10350 else "reject" for i in range(n)],
        }
    )
    rows.to_parquet(tmp_path / "rows.parquet", index=False)
    tokens = [
        {
            "prompt_id": r.prompt_id,
            "wrapped_prompt": r.prompt,
            "input_ids": [int(r.prompt.rsplit("-", 1)[1]), 42],
        }
        for r in rows.itertuples()
    ]
    (tmp_path / "tokenization").mkdir()
    pd.DataFrame(tokens).to_parquet(tmp_path / "tokenization/inputs.parquet", index=False)
    diagnostic.atomic(
        tmp_path / "tokenization/result.json",
        {
            "binding_sha256": "test-binding",
            "status": "complete",
            "files": [diagnostic.fingerprint(tmp_path / "tokenization/inputs.parquet")],
        },
    )
    pd.DataFrame({"prompt": [f"warmup-{i}" for i in range(256)]}).to_parquet(
        tmp_path / "warmup.parquet", index=False
    )
    for kind in ["cpu", "gpu"]:
        diagnostic.atomic(
            tmp_path / "weights" / kind / "result.json",
            {
                "status": "complete",
                "binding_sha256": "test-binding",
                "files": [],
                "serving_tensor_names": ["packed.weight"],
                "model_directory": f"/fake/{kind}",
            },
        )
    diagnostic.atomic(
        tmp_path / "weights/comparison.json",
        {
            "binding_sha256": "test-binding",
            "status": "complete",
            "all_tensors_equal": True,
        },
    )
    diagnostic.atomic(
        tmp_path / "controls.json",
        [
            {"control_prompt_id": "row-00002"},
            {"control_prompt_id": "row-00003"},
        ],
    )
    manifest = {
        "binding_sha256": "test-binding",
        "rows": diagnostic.fingerprint(tmp_path / "rows.parquet"),
        "warmup": diagnostic.fingerprint(tmp_path / "warmup.parquet"),
        "sft_directory": "/fake/sft",
        "base_model": "fake/base",
        "matched_controls": diagnostic.fingerprint(tmp_path / "controls.json"),
        "mismatch_prompt_ids": ["row-00000", "row-00001"],
        "limitations": ["Synthetic CPU test"],
    }

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    @dataclass
    class Compilation:
        static_forward_context: object
        mode: str = "test"

    class NeverCopy:
        def __deepcopy__(self, memo):
            raise AssertionError("runtime object must not be copied")

    class LLM:
        def __init__(self, **kwargs):
            events.append(("engine", int(os.environ["RANK"]), kwargs))
            config = SimpleNamespace(compilation_config=Compilation({"layer": NeverCopy()}))
            self.llm_engine = SimpleNamespace(vllm_config=config)

        def generate(self, prompts, params, use_tqdm=False):
            events.append(("generate", int(os.environ["RANK"]), params.max_tokens, len(prompts)))
            if params.max_tokens == 1:
                value = float("inf") if options["warmup_inf"] else 0.5
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                logprobs=[
                                    {
                                        1: SimpleNamespace(logprob=value),
                                        2: SimpleNamespace(logprob=0.1),
                                    }
                                ]
                            )
                        ]
                    )
                    for _ in prompts
                ]
            requests = []
            for position, prompt in enumerate(prompts):
                index = int(prompt.rsplit("-", 1)[1])
                hire = index % 2 == 0
                if options["flip"] == index:
                    hire = not hire
                text = json.dumps(
                    {"decision": "hire" if hire else "reject", "reason": "a\u2028b"},
                    ensure_ascii=False,
                )
                if options["invalid_decision"]:
                    text = "unparseable response"
                token_ids = [index, 42]
                if options["bad_tokens"] and position == 0:
                    token_ids[-1] = 43
                requests.append(
                    SimpleNamespace(
                        prompt=prompt,
                        prompt_token_ids=token_ids,
                        outputs=[
                            SimpleNamespace(text=text, token_ids=[7, 8], finish_reason="stop")
                        ],
                    )
                )
            return requests

    tokenizer = SimpleNamespace(pad_token=None, eos_token="eos")
    replacements = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda rank: events.append(("device", rank))),
            device=lambda *args: args,
            distributed=SimpleNamespace(
                init_process_group=lambda *args, **kwargs: events.append(("init", kwargs)),
                barrier=lambda: events.append(("barrier",)),
                destroy_process_group=lambda: events.append(("destroy",)),
            ),
        ),
        "transformers": SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
        ),
        "vllm": SimpleNamespace(LLM=LLM, SamplingParams=Params),
        "hiring_bias_mitigation.eval.backends": SimpleNamespace(
            _chat_wrap=lambda tok, text, kwargs: text
        ),
        "hiring_bias_mitigation.eval.margins": SimpleNamespace(
            PREFIX='{"decision":"',
            WORDS={"en": ["hire", "reject"]},
            decision_token_ids=lambda *args: [1, 2],
        ),
        "hiring_bias_mitigation.eval.parsing": parser,
    }
    for name in ["hiring_bias_mitigation", "hiring_bias_mitigation.eval"]:
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    for name, value in replacements.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(diagnostic, "runtime_info", lambda: {"cpu_integration_test": True})

    def push(*args):
        events.append(("push", int(os.environ["RANK"])))
        return {"test_fake_weight_push": True}

    monkeypatch.setattr(diagnostic, "push_weights", push)
    monkeypatch.setattr(
        diagnostic,
        "served_parameters",
        lambda *args: [
            {"name": "packed.weight", "shape": [2], "dtype": "test", "sha256": "test-equal"},
        ],
    )

    def metrics(raw):
        return {
            "n": len(raw),
            "invalid_n": int((raw.outcome != "decided").sum()),
            "decided_n": int((raw.outcome == "decided").sum()),
        }

    monkeypatch.setattr(diagnostic, "module", lambda name: SimpleNamespace(metrics=metrics))

    def rank_run(rank, kind="cpu", smoke=False):
        for key, value in {"RANK": rank, "LOCAL_RANK": rank, "WORLD_SIZE": 4}.items():
            monkeypatch.setenv(key, str(value))
        if smoke:
            diagnostic.score_live(tmp_path, manifest, kind, smoke=True)
        else:
            diagnostic.score_live(tmp_path, manifest, kind)

    return SimpleNamespace(
        D=diagnostic,
        out=tmp_path,
        manifest=manifest,
        rows=rows,
        tokens=tokens,
        parser=parser,
        rank_run=rank_run,
        events=events,
        options=options,
    )


def test_score_collect_summarize_full_bank_with_real_parquet(harness):
    h = harness
    for kind in ["cpu", "gpu"]:
        h.options["flip"] = 17 if kind == "gpu" else None
        for rank in range(4):
            h.rank_run(rank, kind)
        h.D.collect_live(h.out, h.manifest, kind)
        raw = pd.read_parquet(h.out / f"arms/{kind}_live/raw.parquet")
        assert raw.prompt_id.tolist() == h.rows.prompt_id.tolist()
        assert len(raw) == 31050 and raw.prompt_id.is_unique
        assert raw.outcome.eq("decided").all()
        receipt = h.D.read(h.out / f"arms/{kind}_live/result.json")
        assert receipt["actual_input_ids_ordered_sha256"] == h.D.digest(
            [r["input_ids"] for r in h.tokens]
        )
        counts = [
            h.D.read(h.out / f"arms/{kind}_live/rank-{r}/result.json")["rows"] for r in range(4)
        ]
        assert counts == [7763, 7763, 7762, 7762]
        # Real Arrow read turns variable-length integer lists into numpy arrays.
        requests = pd.read_parquet(h.out / f"arms/{kind}_live/request_outputs.parquet")
        assert isinstance(requests.actual_input_ids.iloc[0], np.ndarray)
    base_records = pd.read_parquet(h.out / "arms/cpu_live/request_outputs.parquet").to_dict(
        "records"
    )
    for arm in ["cpu_historical", "gpu_historical"]:
        h.D.write_arm(h.out, h.manifest, arm, base_records, {})
    cache = pd.read_parquet(h.out / "arms/cpu_historical/raw.parquet").drop(columns="prompt_id")
    cache.to_parquet(h.out / "cache.parquet", index=False)
    failed = pd.read_parquet(h.out / "arms/gpu_live/raw.parquet")[
        ["prompt_id", "decision", "raw_output"]
    ]
    failed.to_parquet(h.out / "failed.parquet", index=False)
    h.manifest["source_trajectory"] = {
        "cached_models": {"step425": {"raw": h.D.fingerprint(h.out / "cache.parquet")}}
    }
    h.manifest["failed_startup_raw"] = h.D.fingerprint(h.out / "failed.parquet")
    h.D.summarize(h.out, h.manifest)
    result = h.D.read(h.out / "result.json")
    assert result["status"] == "complete" and result["all_four_arms_completed"]
    assert result["comparisons"]["cpu_live__to__gpu_live"]["decision_differences"] == 1
    assert result["comparisons"]["cpu_live__to__gpu_live"]["different_prompt_ids"] == ["row-00017"]
    assert result["comparisons"]["failed_startup__to__gpu_live"]["decision_differences"] == 0
    assert not list(h.out.rglob("*.tmp"))
    # All eight rank files contain finite, standards-compliant JSON.
    for path in h.out.glob("arms/*_live/rank-*/*.json"):
        json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))


def test_collect_rejects_missing_rank(harness):
    h = harness
    h.rank_run(0)
    with pytest.raises(ValueError, match="missing a matching completed rank"):
        h.D.collect_live(h.out, h.manifest, "cpu")
    assert not (h.out / "arms/cpu_live/result.json").exists()


def test_collect_rejects_parquet_changed_after_receipt(harness):
    h = harness
    h.rank_run(0)
    path = h.out / "arms/cpu_live/rank-0/request_outputs.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "raw_output"] = '{"decision":"reject"}'
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="fingerprint changed"):
        h.D.collect_live(h.out, h.manifest, "cpu")


def test_live_bad_token_ids_persist_failure_and_prevent_collection(harness):
    h = harness
    h.options["bad_tokens"] = True
    with pytest.raises(ValueError, match="actual vLLM live input"):
        h.rank_run(0)
    receipt = h.D.read(h.out / "arms/cpu_live/rank-0/result.json")
    assert receipt["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete or foreign"):
        h.D.collect_live(h.out, h.manifest, "cpu")


def test_repeated_live_phase_rejected_before_cuda_or_overwrite(harness):
    h = harness
    path = h.out / "arms/cpu_live/rank-0"
    path.mkdir(parents=True)
    (path / "engine.json").write_text("old evidence")
    with pytest.raises(ValueError, match="already has evidence"):
        h.rank_run(0)
    assert h.events == []
    assert (path / "engine.json").read_text() == "old evidence"


def test_nonfinite_warmup_cannot_produce_success_receipt(harness):
    h = harness
    h.options["warmup_inf"] = True
    with pytest.raises(ValueError, match="Out of range float"):
        h.rank_run(0)
    assert not (h.out / "arms/cpu_live/rank-0/result.json").exists()
    assert not any(event[0] == "generate" and event[2] == 512 for event in h.events)


def test_smoke_uses_four_first_rows_two_pushes_full_warmup_and_no_formal_arms(harness):
    h = harness
    h.manifest["retry"] = {"synthetic_cpu_test": True}
    for rank in range(4):
        h.rank_run(rank, smoke=True)
        frame = pd.read_parquet(h.out / f"smoke/live/rank-{rank}/request_outputs.parquet")
        assert frame.prompt_id.tolist() == [f"row-{rank:05d}"]
        assert h.events.count(("push", rank)) == 2
        assert h.events.count(("generate", rank, 1, 256)) == 1
        assert h.events.count(("generate", rank, 512, 1)) == 1
    h.D.collect_smoke(h.out, h.manifest)
    result = h.D.read(h.out / "smoke/result.json")
    assert result["status"] == "complete" and result["rows"] == 4
    assert result["included_in_benchmark_metrics"] is False
    assert not (h.out / "arms").exists()


@pytest.mark.parametrize("failure", ["wrong_prompt_id", "invalid_decision"])
def test_smoke_collection_rejects_wrong_ids_or_invalid_decisions(harness, failure):
    h = harness
    h.manifest["retry"] = {"synthetic_cpu_test": True}
    h.options["invalid_decision"] = failure == "invalid_decision"
    for rank in range(4):
        h.rank_run(rank, smoke=True)
    if failure == "wrong_prompt_id":
        path = h.out / "smoke/live/rank-0/request_outputs.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "prompt_id"] = "row-00009"
        frame.to_parquet(path, index=False)
        # A valid file receipt must not substitute for checking the prescribed row.
        receipt_path = path.parent / "result.json"
        receipt = h.D.read(receipt_path)
        receipt["files"] = [
            h.D.fingerprint(path) if p["path"] == str(path) else p for p in receipt["files"]
        ]
        h.D.atomic(receipt_path, receipt)
    expected = "predeclared first row" if failure == "wrong_prompt_id" else "invalid decision"
    with pytest.raises(ValueError, match=expected):
        h.D.collect_smoke(h.out, h.manifest)
    assert not (h.out / "smoke/result.json").exists()
    assert not (h.out / "arms").exists()
