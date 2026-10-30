"""The training anchors' procedure, rerun on the benchmark: does the 122B teacher's
attribute-free verdict agree with the benchmark's GPT-4o reference, and how noisy is one sample?

Why. Every training target carries the teacher's attribute-free verdict: SFT's invariant
responses, DPO's chosen side and GRPO's reward. That verdict is one sample of
synth.REFERENCE_PROMPT at the settings of configs/generation/teacher.yaml (temperature 0.7,
top_p 0.9, seed 42, thinking off). The anchors reject 85.8% of the training pool, the
benchmark's GPT-4o reference 66.7% of its 450 pairs, but the two sets of CVs differ, so the gap
cannot say whether the anchors are stricter. This reruns the procedure on the benchmark's own
450 attribute-free pairs, three ways from one model load:

  single   one sample at the v1 settings, the way each anchor was made (a fresh draw: the v1
           run was on another machine and vLLM version, so it is not the same sample)
  sampled  k more samples per pair: the vote split and the majority verdict
  greedy   temperature 0: the teacher's modal verdict under the same prompt

REFERENCE_PROMPT tells the teacher to "be decisive: a genuinely weak match is a reject"; the
students' screening prompt does not. The same teacher on that prompt, greedy, is the audit in
configs/audit/qwen3.5-122b_en_teacher_gh200.yaml; the two together separate the prompt's push
from the teacher's own standard.

Outputs are benchmark outputs: outputs/teacher_anchor/ and reports/teacher_anchor/, never
artifacts/ and never training data.

    python scripts/teacher_anchor_check.py --stage generate    # GPU, tensor parallel 4
    python scripts/teacher_anchor_check.py --stage summarize   # CPU, writes reports/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hiring_bias_mitigation.data import benchmark as B  # noqa: E402
from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

TEACHER = "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"
#: The anchors' decoding, from configs/generation/teacher.yaml.
V1_GENERATION = {"max_new_tokens": 320, "temperature": 0.7, "top_p": 0.9, "seed": 42}
GREEDY = {"max_new_tokens": 320, "temperature": 0.0, "top_p": 1.0, "seed": 42}
#: The teacher's serving settings on a GH200 node, as for its judge runs.
SERVING = {"backend": "vllm", "tensor_parallel_size": 4, "gpu_memory_utilization": 0.80,
           "max_model_len": 8192, "max_num_seqs": 128, "dtype": "auto",
           "engine_kwargs": {"gdn_prefill_backend": "triton"}}
DECIDED = ("hire", "reject")
KEEP = ["pair_id", "candidate_id", "job_id", "job_position", "reference_decision"]


def run_name(model: str, lang: str) -> str:
    return f"{model.split('/')[-1]}--{lang}"


def to_rows(pairs: pd.DataFrame, single: list[str], sampled: list[list[str]],
            greedy: list[str]) -> pd.DataFrame:
    """One row per generation: (pair, mode, sample index) with its raw text."""
    rows = []
    for i, pair in enumerate(pairs[KEEP].to_dict("records")):
        rows.append({**pair, "mode": "single", "sample_index": 0, "raw_output": single[i]})
        rows += [{**pair, "mode": "sampled", "sample_index": j, "raw_output": text}
                 for j, text in enumerate(sampled[i])]
        rows.append({**pair, "mode": "greedy", "sample_index": 0, "raw_output": greedy[i]})
    return pd.DataFrame(rows)


def parse(frame: pd.DataFrame, lang: str) -> pd.DataFrame:
    parsed = [parse_output(text, lang) for text in frame["raw_output"]]
    return frame.assign(decision=[p.decision for p in parsed],
                        outcome=[p.outcome for p in parsed],
                        feedback=[p.feedback for p in parsed])


def agreement(decision: pd.Series, reference: pd.Series) -> dict:
    """Reject share and agreement with the reference over decided rows, by reference side."""
    decided = decision.isin(DECIDED)
    d, r = decision[decided], reference[decided]
    out = {"n": int(decision.size), "n_decided": int(decided.sum()),
           "reject_share": float((d == "reject").mean()) if len(d) else None,
           "agreement": float((d == r).mean()) if len(d) else None}
    for side in DECIDED:
        on = r == side
        out[f"agreement_on_reference_{side}"] = float((d[on] == side).mean()) if on.any() else None
    out["confusion"] = {f"reference {side}": {k: int(v) for k, v in
                                              d[r == side].value_counts().items()}
                        for side in DECIDED}
    return out


def votes(frame: pd.DataFrame) -> pd.DataFrame:
    """Per pair: hire votes among the sampled draws, decided draws, and the majority."""
    sampled = frame[frame["mode"] == "sampled"]
    g = sampled.groupby("pair_id")
    out = pd.DataFrame({"hire": g["decision"].apply(lambda s: int((s == "hire").sum())),
                        "decided": g["decision"].apply(lambda s: int(s.isin(DECIDED).sum())),
                        "reference": g["reference_decision"].first()})
    out["majority"] = "tie"
    out.loc[2 * out["hire"] > out["decided"], "majority"] = "hire"
    out.loc[2 * out["hire"] < out["decided"], "majority"] = "reject"
    return out


def summarize(frame: pd.DataFrame) -> dict:
    """Every number the report states, from the parsed generations."""
    reference = frame.groupby("pair_id")["reference_decision"].first()
    by_mode = {mode: frame[frame["mode"] == mode].set_index("pair_id")["decision"]
               for mode in ("single", "greedy")}
    sampled = frame[frame["mode"] == "sampled"]
    v = votes(frame)
    k = int(sampled.groupby("pair_id").size().max()) if len(sampled) else 0
    full = v[v["decided"] == k]
    single = by_mode["single"].reindex(reference.index)
    return {
        "pairs": int(reference.size),
        "samples_per_pair": k,
        "reference_reject_share": float((reference == "reject").mean()),
        "single": agreement(single, reference),
        "greedy": agreement(by_mode["greedy"].reindex(reference.index), reference),
        "sampled_per_draw": agreement(sampled["decision"].reset_index(drop=True),
                                      sampled["reference_decision"].reset_index(drop=True)),
        "majority": {**agreement(v["majority"], v["reference"]),
                     "ties": int((v["majority"] == "tie").sum())},
        "vote_split": {f"{h} of {k} hire": int(n) for h, n in
                       full["hire"].value_counts().sort_index().items()},
        "unanimous_share": float(full["hire"].isin([0, k]).mean()) if len(full) else None,
        "single_matches_majority": float(
            (single.reindex(v.index) == v["majority"])[v["majority"] != "tie"].mean()),
        "undecided_share_by_mode": {m: float((~frame[frame["mode"] == m]["decision"]
                                              .isin(DECIDED)).mean())
                                    for m in ("single", "sampled", "greedy")},
    }


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def render(name: str, s: dict, meta: dict) -> str:
    def row(label: str, a: dict) -> str:
        cells = [a["n_decided"], a["reject_share"], a["agreement"],
                 a["agreement_on_reference_hire"], a["agreement_on_reference_reject"]]
        return "| " + " | ".join([label] + [_fmt(c) for c in cells]) + " |"

    k = s["samples_per_pair"]
    return "\n".join([
        f"# Training-anchor procedure on the benchmark: {name}",
        "",
        "Generated by `scripts/teacher_anchor_check.py`; regenerate it, do not edit it. "
        f"Prompt: `synth.REFERENCE_PROMPT` (sha256 {meta['prompt_sha256'][:12]}), decoding "
        f"{meta['decoding']}, thinking off, on the {s['pairs']} attribute-free benchmark pairs. "
        "Reference: the benchmark's GPT-4o verdict, which rejects "
        f"{s['reference_reject_share']:.3f} of the pairs.",
        "",
        "| verdict | decided | reject share | agreement with GPT-4o | on GPT-4o hire | "
        "on GPT-4o reject |",
        "|---|---|---|---|---|---|",
        row("single sample (as each anchor)", s["single"]),
        row(f"each of {k} sampled draws", s["sampled_per_draw"]),
        row(f"majority of {k} (ties left out: {s['majority']['ties']})", s["majority"]),
        row("greedy", s["greedy"]),
        "",
        f"Hire votes among {k} draws, pairs: {s['vote_split']}. Unanimous: "
        f"{_fmt(s['unanimous_share'])}. The single sample matches the majority on "
        f"{_fmt(s['single_matches_majority'])} of pairs without a tie.",
        "",
        f"Undecided (refused or unparsable) share by mode: {s['undecided_share_by_mode']}.",
        "",
    ])


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True,
                              text=True, check=True).stdout

    try:
        status = git("status", "--porcelain", "--untracked-files=no", "--", "scripts", "src")
        return {"commit": git("rev-parse", "HEAD").strip(),
                "uncommitted": [line[3:] for line in status.splitlines() if line.strip()]}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted": None}


def stage_generate(args) -> None:
    from hiring_bias_mitigation.eval import backends as BK

    pairs = B.load_benchmark(args.lang)
    prompts = SY.reference_prompts(pairs, args.lang)
    backend = BK.build_backend({"model": args.model, "generation": V1_GENERATION, **SERVING})
    try:
        single = backend.generate(prompts)  # the anchors' procedure: one draw per pair
        sampled = backend.generate_samples(prompts, args.k)
        backend.generation = BK.GenerationConfig(**GREEDY)
        greedy = backend.generate(prompts)
    finally:
        backend.close()
    frame = parse(to_rows(pairs, single, sampled, greedy), args.lang)
    out = Path(resolve_output_path("outputs/teacher_anchor"))
    out.mkdir(parents=True, exist_ok=True)
    name = run_name(args.model, args.lang)
    frame.to_parquet(out / f"{name}.parquet", index=False)
    meta = {"model": args.model, "lang": args.lang, "pairs": len(pairs), "k": args.k,
            "decoding": V1_GENERATION, "greedy": GREEDY, "serving": SERVING,
            "prompt_sha256": _sha256_text(SY.REFERENCE_PROMPT[args.lang]),
            "code": code_state()}
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / name}.parquet: {len(frame)} generations")


def stage_summarize(args) -> None:
    name = run_name(args.model, args.lang)
    raw = Path(resolve_output_path("outputs/teacher_anchor"))
    frame = pd.read_parquet(raw / f"{name}.parquet")
    meta = json.loads((raw / f"{name}.meta.json").read_text())
    summary = {**summarize(frame), "generation_meta": meta, "summary_code": code_state()}
    out = REPO_ROOT / "reports" / "teacher_anchor"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / f"{name}.md").write_text(render(name, summary, meta), encoding="utf-8")
    print(render(name, summary, meta))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("generate", "summarize"), required=True)
    parser.add_argument("--model", default=TEACHER)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--k", type=int, default=8, help="sampled draws per pair")
    args = parser.parse_args()
    {"generate": stage_generate, "summarize": stage_summarize}[args.stage](args)


if __name__ == "__main__":
    main()
