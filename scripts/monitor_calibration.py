"""The margin monitor's calibration (mitigation/grpo.py, `project_from_margins`), from files on
disk. CPU, seconds.

  calibrate  The share of benchmark prompts in each bin of the parent's decision margin, where
             GPT-4o hires and where it rejects: the `margin_monitor` block of a training config.
             It reads the parent's op-margin-v2 margins
             (outputs/operating_point/v2/<run>--adapter.parquet) and GPT-4o's verdicts
             (`reference_decision` in outputs/raw/<run>.parquet), attributed prompts only.
  validate   The estimate run on the benchmark itself. Each arm's change in hire share per bin
             of the parent's margin is taken over all attributed prompts, whatever GPT-4o said,
             weighted by those shares, and set against the arm's actual TPR and FPR change. If
             the bins carry the change here, a monitor whose held-out rows cover them can see
             it.
  score      (GPU) One merged model's margins on a build's monitor rows (grpo_monitor.parquet),
             with the scorer the training monitor uses (grpo.vllm_margins), to
             outputs/monitor_calibration/<label>.parquet. scripts/cscs/monitor_calibration.sbatch
             runs one model per GPU.
  transfer   Whether held-out rows move as benchmark prompts do. For each scored arm, the
             monitor's projection from the parent's and the arm's margins on the held-out rows,
             set against the arm's TPR and FPR change on the benchmark (by margins).

The calibration uses one fact about the benchmark: how the parent's margins are distributed,
the same kind of fact as the parent's benchmark TPR and FPR already in `sft_benchmark`. No
trained arm's benchmark result enters it; `validate` and `transfer` read arms' results only to
check the monitor, after their audits.

    python scripts/monitor_calibration.py calibrate \
        --parent Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional
    python scripts/monitor_calibration.py validate \
        --parent Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional \
        --arm DAPO-v3-all=Qwen3.5-9B--en--grpo--adapter--en_only_dapo_v3_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.mitigation.grpo import (  # noqa: E402
    MARGIN_EDGES,
    margin_bins,
    project_from_margins,
)
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

KEY = ["pair_id", "protected_group", "protected_attr", "condition"]


def load_parent(run: str) -> pd.DataFrame:
    """The parent's attributed benchmark prompts: key, GPT-4o's verdict, the parent's margin."""
    raw = pd.read_parquet(Path(resolve_output_path("outputs/raw")) / f"{run}.parquet",
                          columns=[*KEY, "reference_decision"])
    raw = raw[raw["condition"] != "attr_free"]
    frame = raw.merge(load_margins(run), on=KEY, how="inner")
    if len(frame) != len(raw):
        raise ValueError(f"{run}: {len(raw)} attributed prompts, {len(frame)} with a margin")
    return frame.assign(gpt4o_hire=frame["reference_decision"].str.lower() == "hire")


def load_margins(run: str) -> pd.DataFrame:
    path = Path(resolve_output_path("outputs/operating_point/v2")) / f"{run}--adapter.parquet"
    return pd.read_parquet(path, columns=[*KEY, "margin"])


def bench_shares(margins, gpt4o_hire, edges=MARGIN_EDGES) -> dict:
    """Per bin of the parent's margin, its share of the prompts where GPT-4o hires and of
    those where it rejects; the parent's TPR and FPR by margins; GPT-4o's hire share."""
    margins, gpt4o_hire = np.asarray(margins, dtype=float), np.asarray(gpt4o_hire, dtype=bool)
    bins = margin_bins(margins, edges)
    n_bins = len(edges) + 1
    hire = np.bincount(bins[gpt4o_hire], minlength=n_bins) / gpt4o_hire.sum()
    reject = np.bincount(bins[~gpt4o_hire], minlength=n_bins) / (~gpt4o_hire).sum()
    return {"edges": [float(e) for e in edges],
            "share_ref_hire": [round(float(s), 4) for s in hire],
            "share_ref_reject": [round(float(s), 4) for s in reject],
            "reference_hire_share": round(float(gpt4o_hire.mean()), 4),
            "parent_tpr": float((margins[gpt4o_hire] > 0).mean()),
            "parent_fpr": float((margins[~gpt4o_hire] > 0).mean())}


def validate(parent: pd.DataFrame, arm_margins: pd.DataFrame, shares: dict) -> dict:
    """Estimated (label-free, binned) against actual TPR and FPR change of one arm, in pp."""
    arm = parent[KEY].merge(arm_margins, on=KEY, how="left")
    if arm["margin"].isna().any():
        raise ValueError(f"{int(arm['margin'].isna().sum())} prompts without the arm's margin")
    hire = parent["gpt4o_hire"].to_numpy()
    p, a = parent["margin"].to_numpy(), arm["margin"].to_numpy()
    estimate = project_from_margins(p, a, shares["edges"], shares["share_ref_hire"],
                                    shares["share_ref_reject"],
                                    {"hire_rate_ref_hire": shares["parent_tpr"],
                                     "hire_rate_ref_reject": shares["parent_fpr"],
                                     "reference_hire_share": shares["reference_hire_share"]})
    return {"tpr_actual_pp": 100 * float((a[hire] > 0).mean() - (p[hire] > 0).mean()),
            "tpr_estimate_pp": 100 * estimate["delta_tpr"],
            "fpr_actual_pp": 100 * float((a[~hire] > 0).mean() - (p[~hire] > 0).mean()),
            "fpr_estimate_pp": 100 * estimate["delta_fpr"],
            "bin_change_pp": [None if b["delta_hire_share"] is None
                              else round(100 * b["delta_hire_share"], 1)
                              for b in estimate["bins"]]}


def scores_dir() -> Path:
    return Path(resolve_output_path("outputs/monitor_calibration"))


def prompt_digest(prompts: list[str]) -> str:
    import hashlib

    h = hashlib.sha1()
    for p in prompts:
        h.update(p.encode())
        h.update(b"\x1e")
    return h.hexdigest()


def score(label: str, model: str, rows: Path, gpu_memory_utilization: float,
          output_dir: Path | None = None) -> None:
    from types import SimpleNamespace

    from vllm import LLM

    from hiring_bias_mitigation.mitigation.grpo import vllm_margins

    out = output_dir or scores_dir()
    target = out / f"{label}.parquet"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite calibration scores: {target}")
    prompts = pd.read_parquet(rows, columns=["prompt"])["prompt"].tolist()
    llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=8192,
              trust_remote_code=True, seed=42, max_num_seqs=64, gdn_prefill_backend="triton",
              language_model_only=True)
    margins = vllm_margins(SimpleNamespace(llm=llm), llm.get_tokenizer(), prompts,
                           {"enable_thinking": False})
    if len(margins) != len(prompts) or not np.isfinite(margins).all():
        raise ValueError("Calibration scoring returned missing or nonfinite margins")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"margin": margins, "row_index": range(len(prompts))}).assign(model=model, rows=str(rows),
                                             prompts_sha1=prompt_digest(prompts)).to_parquet(
        target, index=False)
    print(f"{label}: {len(margins)} margins, hire share {np.mean(np.asarray(margins) > 0):.3f}")


def load_scored(label: str, output_dir: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet((output_dir or scores_dir()) / f"{label}.parquet")


def transfer(parent_scores: pd.DataFrame, arm_scores: pd.DataFrame, parent: pd.DataFrame,
             arm_margins: pd.DataFrame, shares: dict) -> dict:
    """The monitor's held-out projection against the benchmark's change, in pp."""
    if parent_scores["prompts_sha1"].iloc[0] != arm_scores["prompts_sha1"].iloc[0]:
        raise ValueError("the parent and the arm were scored on different monitor rows")
    projected = project_from_margins(parent_scores["margin"], arm_scores["margin"],
                                     shares["edges"], shares["share_ref_hire"],
                                     shares["share_ref_reject"],
                                     {"hire_rate_ref_hire": shares["parent_tpr"],
                                      "hire_rate_ref_reject": shares["parent_fpr"],
                                      "reference_hire_share": shares["reference_hire_share"]})
    bench = validate(parent, arm_margins, shares)
    return {"tpr_benchmark_pp": bench["tpr_actual_pp"],
            "tpr_monitor_pp": 100 * projected["delta_tpr"],
            "fpr_benchmark_pp": bench["fpr_actual_pp"],
            "fpr_monitor_pp": 100 * projected["delta_fpr"],
            "rows_per_bin": [b["n"] for b in projected["bins"]]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("calibrate", "validate", "transfer"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--parent", required=True, help="the parent's run name")
        if name == "validate":
            cmd.add_argument("--arm", action="append", required=True, help="LABEL=run name")
        if name == "transfer":
            cmd.add_argument("--scores-dir", type=Path)
            cmd.add_argument("--parent-label", required=True, help="the parent's `score` label")
            cmd.add_argument("--arm", action="append", required=True,
                             help="LABEL=run name, LABEL being the arm's `score` label")
    cmd = sub.add_parser("score")
    cmd.add_argument("--label", required=True)
    cmd.add_argument("--model", required=True, help="a merged model directory")
    cmd.add_argument("--rows", required=True, help="a build's grpo_monitor.parquet")
    cmd.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    cmd.add_argument("--scores-dir", type=Path)
    args = parser.parse_args()

    if args.command == "score":
        score(args.label, args.model, Path(resolve_output_path(args.rows)),
              args.gpu_memory_utilization, args.scores_dir)
        return
    parent = load_parent(args.parent)
    shares = bench_shares(parent["margin"], parent["gpt4o_hire"])
    if args.command == "calibrate":
        print(f"# {args.parent}: {len(parent)} attributed benchmark prompts; by margins TPR "
              f"{shares['parent_tpr']:.4f}, FPR {shares['parent_fpr']:.4f}")
        print("margin_monitor:")
        for key in ("edges", "share_ref_hire", "share_ref_reject", "reference_hire_share"):
            print(f"  {key}: {shares[key]}")
        return
    if args.command == "transfer":
        parent_scores = load_scored(args.parent_label, args.scores_dir)
        print("arm | TPR change: benchmark, monitor | FPR change: benchmark, monitor (pp) | "
              "monitor rows per parent-margin bin")
        for spec in args.arm:
            label, run = spec.split("=", 1)
            r = transfer(parent_scores, load_scored(label, args.scores_dir), parent, load_margins(run), shares)
            print(f"{label} | {r['tpr_benchmark_pp']:+.2f}, {r['tpr_monitor_pp']:+.2f} | "
                  f"{r['fpr_benchmark_pp']:+.2f}, {r['fpr_monitor_pp']:+.2f} | "
                  f"{r['rows_per_bin']}")
        return
    print("arm | TPR change: actual, estimate | FPR change: actual, estimate (pp) | "
          "hire-share change per parent-margin bin (pp)")
    for spec in args.arm:
        label, run = spec.split("=", 1)
        r = validate(parent, load_margins(run), shares)
        print(f"{label} | {r['tpr_actual_pp']:+.2f}, {r['tpr_estimate_pp']:+.2f} | "
              f"{r['fpr_actual_pp']:+.2f}, {r['fpr_estimate_pp']:+.2f} | {r['bin_change_pp']}")


if __name__ == "__main__":
    main()
