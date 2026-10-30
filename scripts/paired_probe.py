"""Fresh 16-draw attributed / attribute-free diagnosis on every GRPO-v3-all row.

Prepare on CPU, sample two shards per model on four GPUs, then analyze only a complete run.
Uses the audited merged model chain because this repo's Qwen3.5 vLLM LoRA path is invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "src"))
N = 16
SEED = 20261022
MODELS = {
    "sft_v3": "qwen3.5-9b_en_only_sft_v3_additional",
    "dapo_final": "qwen3.5-9b_en_only_dapo_v3_all",
}
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
DECODING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 96,
    "n": N,
    "base_seed": SEED,
    "enable_thinking": False,
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    tmp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def request_seed(row_id: int, view: str, prompt: str) -> int:
    # Reserve 32 adjacent seeds per request; vLLM n=16 uses seed ... seed+15.
    value = f"{SEED}:{row_id}:{view}:{prompt}".encode()
    return (int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & ((1 << 58) - 1)) * 32


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def prepare(run: Path, root: Path) -> None:
    if (run / "manifest.json").exists():
        raise ValueError(f"Already prepared: {run}. Use sample/analyze to resume.")
    source = root / "artifacts/semisynthetic-v3"
    train_path = source / "grpo-v3-all/grpo_train.parquet"
    rows = pd.read_parquet(train_path).reset_index(drop=True)
    assert len(rows) == 9140, len(rows)
    assert rows.reference_decision.value_counts().to_dict() == {"hire": 4570, "reject": 4570}
    assert not rows.duplicated(KEY).any()
    rows.insert(0, "row_id", range(len(rows)))
    inputs = [train_path, source / "grpo-v3-all/report.json"]
    free_parts, pool_parts, anchor_parts = [], [], []
    for pool_name, anchor_name, cfg_name in (
        ("onpolicy-new", "new-pairs", "v3_new_pairs.yaml"),
        ("onpolicy-new2", "new-pairs-2", "v3_new_pairs_2.yaml"),
    ):
        import yaml

        cfg = REPO / "configs/generation" / cfg_name
        total_votes = int(yaml.safe_load(cfg.read_text())["vote_draws"])
        pp = source / pool_name / "prompts.parquet"
        rp = source / pool_name / "rows.parquet"
        ap = source / anchor_name / "anchors_en.parquet"
        inputs.extend([pp, rp, ap, cfg])
        prompts = pd.read_parquet(pp)
        pool = pd.read_parquet(rp)
        attributed = prompts.loc[prompts.kind == "attributed", ["key", "prompt"]].copy()
        attributed["key"] = attributed["key"].astype(int)
        assert len(attributed) == len(pool) and not attributed.key.duplicated().any()
        assert set(attributed.key) == set(range(len(pool)))
        pool["pool_prompt"] = (
            attributed.set_index("key").prompt.reindex(range(len(pool))).to_numpy()
        )
        pool_parts.append(
            pool[[*KEY, "pool_prompt", "reference_decision"]].rename(
                columns={"reference_decision": "pool_reference"}
            )
        )
        free = prompts.loc[prompts.kind == "attr_free", ["key", "prompt"]].rename(
            columns={"key": "pair_id", "prompt": "free_prompt"}
        )
        free["pool_source"] = pool_name
        free_parts.append(free)
        anchors = pd.read_parquet(ap)
        assert not anchors.pair_id.duplicated().any()
        anchors["anchor_total_votes"] = total_votes
        anchors["anchor_reject_votes"] = anchors.decided - anchors.hire_votes
        anchors["anchor_winner_votes"] = anchors[["hire_votes", "anchor_reject_votes"]].max(axis=1)
        anchor_parts.append(
            anchors[
                [
                    "pair_id",
                    "candidate_id",
                    "job_id",
                    "reference_decision",
                    "hire_votes",
                    "decided",
                    "anchor_total_votes",
                    "anchor_reject_votes",
                    "anchor_winner_votes",
                ]
            ].rename(
                columns={
                    "candidate_id": "anchor_candidate_id",
                    "job_id": "anchor_job_id",
                    "reference_decision": "anchor_reference",
                    "hire_votes": "anchor_hire_votes",
                    "decided": "anchor_valid_votes",
                }
            )
        )
    rows = rows.merge(pd.concat(pool_parts), on=KEY, how="left", validate="one_to_one")
    assert (rows.prompt == rows.pool_prompt).all(), "Attributed prompts differ from source pool"
    assert (rows.reference_decision == rows.pool_reference).all()
    rows = rows.drop(columns=["pool_prompt", "pool_reference"])
    rows = rows.merge(pd.concat(free_parts), on="pair_id", how="left", validate="many_to_one")
    rows = rows.merge(pd.concat(anchor_parts), on="pair_id", how="left", validate="many_to_one")
    assert rows.free_prompt.notna().all() and rows.anchor_reference.notna().all()
    assert (rows.prompt != rows.free_prompt).all()
    assert (rows.candidate_id == rows.anchor_candidate_id).all()
    assert (rows.job_id == rows.anchor_job_id).all()
    assert (rows.reference_decision == rows.anchor_reference).all()
    assert (rows.anchor_valid_votes <= rows.anchor_total_votes).all()
    assert (rows.anchor_hire_votes >= 0).all() and (rows.anchor_reject_votes >= 0).all()
    assert (rows.anchor_winner_votes > rows.anchor_valid_votes / 2).all()
    assert (
        (rows.anchor_hire_votes > rows.anchor_reject_votes) == (rows.reference_decision == "hire")
    ).all()
    rows = rows.sort_values("row_id").reset_index(drop=True)
    all_seeds = [
        request_seed(int(r.row_id), view, prompt)
        for r in rows.itertuples()
        for view, prompt in (("attributed", r.prompt), ("attr_free", r.free_prompt))
    ]
    assert len(all_seeds) == len(set(all_seeds))
    models = {}
    for label, name in MODELS.items():
        model = root / "outputs/merged" / name
        merge_path = model / "merged_from.json"
        info = json.loads(merge_path.read_text())
        expected = [root / "outputs/sft" / MODELS["sft_v3"]]
        if label == "dapo_final":
            expected += [root / "outputs/dapo" / name]
        assert info["adapters"] == [str(p) for p in expected]
        mtimes = info["adapter_mtime"]
        if not isinstance(mtimes, list):
            mtimes = [mtimes]
        adapters = []
        for adapter, stamp in zip(expected, mtimes, strict=True):
            weight = adapter / "adapter_model.safetensors"
            assert weight.stat().st_mtime == stamp, f"Stale merge: {weight}"
            adapters.append({"path": str(adapter), "sha256": sha(weight)})
        if label == "dapo_final":
            final = expected[-1]
            assert sha(final / "checkpoint-85/adapter_model.safetensors") == adapters[-1]["sha256"]
            assert (
                json.loads((final / "checkpoint-85/trainer_state.json").read_text())["global_step"]
                == 85
            )
        model_files = [*model.glob("*.safetensors"), *model.glob("*.json"), *model.glob("*.jinja")]
        model_files = sorted(set(model_files))
        models[label] = {
            "path": str(model),
            "adapters": adapters,
            "merge": info,
            "files": {
                str(p): {
                    "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns,
                    "sha256": sha(p),
                }
                for p in model_files
            },
        }
    run.mkdir(parents=True, exist_ok=True)
    write_parquet(run / "rows.parquet", rows)
    manifest = {
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "unique_pairs": rows.pair_id.nunique(),
        "samples_per_view": N,
        "total_expected_completions": 2 * len(rows) * 2 * N,
        "decoding": DECODING,
        "models": models,
        "seed_scheme": "sha256(base:row_id:view:prompt), 32-spaced; same across models",
        "inputs": {str(p): sha(p) for p in inputs},
        "rows_sha256": sha(run / "rows.parquet"),
        "commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "code": {
            str(p): sha(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("paired_probe_analysis.py"),
                REPO / "src/hiring_bias_mitigation/eval/backends.py",
                REPO / "src/hiring_bias_mitigation/eval/parsing.py",
            ]
        },
        "classification": "Attributed 16/16 opposite-anchor decisions; free 16/16 correct -> A/B, free 16/16 wrong -> C+D and reconsider RL; mixed/invalid separate; majority sensitivity also reported",  # noqa: E501
        "free_sampling": "One fresh 16-draw request per original row, including repeated pair prompts; independent row/view seeds",  # noqa: E501
        "limitations": [
            "Agreement with anchor is not ground truth",
            "16 observations do not establish zero true success probability",
            "Rows share 1767 pairs; pair counts are reported",
            "Uses audited BF16 merged weights, merge retention retained in metadata",
        ],
    }
    write_json(run / "manifest.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ["rows", "unique_pairs", "total_expected_completions", "commit"]
            }
        ),
        flush=True,
    )


def check_chunk(frame: pd.DataFrame, ids: list[int], label: str) -> None:
    assert len(frame) == len(ids) * 2 * N
    assert set(frame.model) == {label}
    expected = {(i, v, s) for i in ids for v in ("attributed", "attr_free") for s in range(N)}
    actual = list(frame[["row_id", "view", "sample_index"]].itertuples(index=False, name=None))
    assert len(actual) == len(set(actual)) and set(actual) == expected


def sample(run: Path, label: str, shard: int, shards: int) -> None:
    from hiring_bias_mitigation.eval.backends import GenerationConfig, VLLMBackend, _chat_wrap
    from hiring_bias_mitigation.eval.parsing import parse_output

    manifest = json.loads((run / "manifest.json").read_text())
    assert sha(run / "rows.parquet") == manifest["rows_sha256"]
    for path, expected in manifest["code"].items():
        assert sha(Path(path)) == expected, f"Code changed after preparation: {path}"
    model_info = manifest["models"][label]
    for path, info in model_info["files"].items():
        st = Path(path).stat()
        assert st.st_size == info["size"] and st.st_mtime_ns == info["mtime_ns"], path
    rows = pd.read_parquet(run / "rows.parquet")
    rows = rows.loc[rows.row_id % shards == shard].reset_index(drop=True)
    output = run / "samples" / label / f"shard-{shard}"
    output.mkdir(parents=True, exist_ok=True)
    # An early small chunk makes startup and parser health reviewable without losing work.
    bounds = [(0, min(8, len(rows)))] + [
        (s, min(s + 128, len(rows))) for s in range(8, len(rows), 128)
    ]
    todo = []
    for number, (start, stop) in enumerate(bounds):
        path = output / f"chunk-{number:04d}.parquet"
        ids = rows.iloc[start:stop].row_id.tolist()
        if path.exists():
            check_chunk(pd.read_parquet(path), ids, label)
        else:
            todo.append((number, start, stop, path))
    if not todo:
        print(f"COMPLETE {label} shard={shard}: all chunks already present", flush=True)
        return
    backend = VLLMBackend(
        model=model_info["path"],
        generation=GenerationConfig(temperature=1.0, top_p=1.0, max_new_tokens=96, seed=SEED),
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_model_len=4096,
        max_num_seqs=128,
        dtype="bfloat16",
        chat_template_kwargs={"enable_thinking": False},
        engine_kwargs={"gdn_prefill_backend": "triton", "language_model_only": True},
    )
    started = time.time()
    backend._load()
    import torch
    import transformers
    import vllm
    from vllm import SamplingParams

    v2 = backend._llm.llm_engine.vllm_config.use_v2_model_runner
    assert v2 is False, f"Expected classic per-request RNG runner, got use_v2_model_runner={v2}"
    runtime = {
        "model": label,
        "shard": shard,
        "shards": shards,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "use_v2_model_runner": v2,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "decoding": DECODING,
        "model_path": model_info["path"],
    }
    write_json(output / "runtime.json", runtime)
    try:
        for number, start, stop, path in todo:
            requests = [
                (int(r.row_id), view, prompt)
                for r in rows.iloc[start:stop].itertuples()
                for view, prompt in (("attributed", r.prompt), ("attr_free", r.free_prompt))
            ]
            wrapped = [
                _chat_wrap(backend._tokenizer, p, {"enable_thinking": False})
                for _, _, p in requests
            ]
            longest = max(
                len(backend._tokenizer.encode(p, add_special_tokens=False)) for p in wrapped
            )
            assert longest + 96 <= 4096, longest
            seeds = [request_seed(i, v, p) for i, v, p in requests]
            params = [
                SamplingParams(n=N, temperature=1.0, top_p=1.0, top_k=-1, max_tokens=96, seed=seed)
                for seed in seeds
            ]
            results = backend._llm.generate(wrapped, params, use_tqdm=False)
            assert len(results) == len(requests)
            records = []
            for (row_id, view, _), request, seed in zip(requests, results, seeds, strict=True):
                assert len(request.outputs) == N
                for completion in request.outputs:
                    parsed = parse_output(completion.text, "en")
                    records.append(
                        {
                            "model": label,
                            "row_id": row_id,
                            "view": view,
                            "sample_index": completion.index,
                            "seed": seed,
                            "sample_seed": seed + completion.index,
                            "decision": parsed.decision,
                            "outcome": parsed.outcome,
                            "finish_reason": completion.finish_reason,
                            "raw_output": completion.text,
                            "output_tokens": len(completion.token_ids),
                        }
                    )
            frame = pd.DataFrame(records)
            check_chunk(frame, rows.iloc[start:stop].row_id.tolist(), label)
            write_parquet(path, frame)
            print(
                f"PROGRESS {label} shard={shard} chunk={number} rows={stop}/{len(rows)} completions={stop * 2 * N} "  # noqa: E501
                f"elapsed_s={time.time() - started:.1f} outcomes={frame.outcome.value_counts().to_dict()}",  # noqa: E501
                flush=True,
            )
        write_json(
            output / "complete.json",
            {
                **runtime,
                "rows": len(rows),
                "completions": len(rows) * 2 * N,
                "elapsed_s": time.time() - started,
            },
        )
    finally:
        backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "sample", "analyze"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("/ritom/scratch/cscs/zhang/xw"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run, args.root)
    elif args.command == "sample":
        assert args.model and 0 <= args.shard < args.shards
        sample(args.run, args.model, args.shard, args.shards)
    else:
        from paired_probe_analysis import analyze

        analyze(args.run)


if __name__ == "__main__":
    main()
