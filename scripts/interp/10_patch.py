"""Activation patching for one (model spec, dataset, contrast): residual, components, heads.

    uv run python scripts/interp/10_patch.py --spec base --dataset pool \\
        --contrast mil_combat_civ_impl --device cuda:1

Pairs are drawn from the manifest: aligned pairs whose *base-model* effect is at least
--min-effect logits, a seeded sample of --n-pairs. Selection never looks at the spec being
patched, so base and SFT on the same dataset and contrast run on the same pairs.

Per batch of pairs, one forward of the source prompts B captures every activation; then the
target prompts A are rerun once per patch:

    residual    the residual entering each of the 32 layers, and the final pre-norm residual,
                over the whole suffix and over the last tokens only (`late`)
    components  each token mixer (attention or GDN) and each MLP output, same two classes
    atp         attribution-patching estimates for all 128 attention heads, 768 GDN heads,
                32 mixers and 32 MLPs (one backward pass)

and finally the top heads by AtP, plus random controls, are re-measured by real patching.
Outputs go to outputs/interp/patching/<spec>/<dataset>/<contrast>/, one part per batch, so an
interrupted run resumes; summaries go to reports/interp/patching/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from hiring_bias_mitigation.interp import arch  # noqa: E402
from hiring_bias_mitigation.interp import margins as M  # noqa: E402
from hiring_bias_mitigation.interp import patching as PT  # noqa: E402
from hiring_bias_mitigation.interp import spec as SP  # noqa: E402
from hiring_bias_mitigation.interp.io import data_config, interp_dir, report_dir, write_meta  # noqa: E402
from hiring_bias_mitigation.interp.segment import Segmentation  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("interp.patch")

RESID_CLASSES = ("suffix", "late")
COMPONENT_CLASSES = ("suffix", "late")


def select_pairs(
    manifest: pd.DataFrame, contrast: str, n: int, min_effect: float, seed: int
) -> pd.DataFrame:
    rows = manifest[
        (manifest["contrast_id"] == contrast)
        & manifest["aligned"]
        & (manifest["dm_base"].abs() >= min_effect)
    ]
    if rows.empty:
        raise SystemExit(f"no aligned pairs for {contrast} with |dm_base| >= {min_effect}")
    return rows.sample(n=min(n, len(rows)), random_state=seed).reset_index(drop=True)


def load_segments(dataset: str, lang: str, uids: set[str]) -> dict[str, Segmentation]:
    segs = pd.read_parquet(interp_dir("segments") / f"{dataset}-{lang}.parquet")
    segs = segs[segs["prompt_uid"].isin(uids)]
    return {r["prompt_uid"]: Segmentation.from_record(r) for r in segs.to_dict("records")}


def load_completed(out_dir: Path) -> dict[str, pd.DataFrame]:
    """pairs/real/atp of complete batches only, one row set per pair.

    A batch that failed part-way leaves a pairs- part with no real-/atp- parts; its pairs are
    re-run under a new name on resume. Reading every part would count them twice.
    """
    parts: dict[str, list] = {"pairs": [], "real": [], "atp": []}
    seen: set[str] = set()
    for f in sorted(out_dir.glob("pairs-*.parquet")):
        tag = f.name[len("pairs-"):]
        if not ((out_dir / f"real-{tag}").exists() and (out_dir / f"atp-{tag}").exists()):
            continue
        pairs = pd.read_parquet(f)
        if seen & set(pairs["pair_uid"]):
            raise AssertionError(f"{f.name}: pairs already present in another complete batch")
        seen |= set(pairs["pair_uid"])
        parts["pairs"].append(pairs)
        parts["real"].append(pd.read_parquet(out_dir / f"real-{tag}"))
        parts["atp"].append(pd.read_parquet(out_dir / f"atp-{tag}"))
    return {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}


def completed_pairs(out_dir: Path) -> set[str]:
    """pair_uids whose batch finished: a pairs-, real- and atp- part with the same name."""
    done = set()
    for f in out_dir.glob("pairs-*.parquet"):
        tag = f.name[len("pairs-") :]
        if (out_dir / f"real-{tag}").exists() and (out_dir / f"atp-{tag}").exists():
            done |= set(pd.read_parquet(f, columns=["pair_uid"])["pair_uid"])
    return done


def run_batches(
    model,
    tokenizer,
    ro,
    pairs,
    segs,
    out_dir: Path,
    token_budget: int,
    atp_budget: int,
    n_layers: int,
) -> None:
    """Batches sized by tokens, not rows: the pool's longest prompts are 3x the median.

    Resumes by pair: batches of an interrupted run (whatever their size was) are kept, and
    only pairs without a complete pairs/real/atp triple are run again.
    """
    resid = PT.residual_points(n_layers)
    points = resid + PT.component_points(n_layers) + [("head_in", i) for i in range(n_layers)]
    configs = [
        {"kind": k, "layer": layer, "cls": c} for k, layer in resid for c in RESID_CLASSES
    ] + [
        {"kind": k, "layer": layer, "cls": c}
        for k, layer in PT.component_points(n_layers)
        for c in COMPONENT_CLASSES
    ]
    done = completed_pairs(out_dir)
    todo = pairs[~pairs["pair_uid"].isin(done)]
    lengths = [
        max(segs[a].n_tokens, segs[b].n_tokens) for a, b in zip(todo["uid_a"], todo["uid_b"])
    ]
    order = np.argsort(lengths, kind="stable")
    todo = todo.iloc[order].reset_index(drop=True)
    lengths = [lengths[i] for i in order]
    log.info(
        "%d of %d pairs already done; %d to run",
        len(done & set(pairs["pair_uid"])),
        len(pairs),
        len(todo),
    )
    for group in PT.token_batches(lengths, token_budget):
        rows = todo.iloc[group]
        tag = f"t{len(done) + group[0]:05d}-{len(done) + group[-1]:05d}.parquet"
        batch = PT.make_batch(
            rows,
            [segs[u] for u in rows["uid_a"]],
            [segs[u] for u in rows["uid_b"]],
            tokenizer.pad_token_id,
        )
        started = time.time()
        source = PT.capture_source(model, batch, points)
        m_a = PT.margins_of(model, ro, batch.ids_a, batch.mask_a)
        m_b = PT.margins_of(model, ro, batch.ids_b, batch.mask_b)
        rows[["pair_uid", "candidate_id"]].assign(m_a=m_a, m_b=m_b).to_parquet(
            out_dir / f"pairs-{tag}", index=False
        )
        PT.sweep(model, ro, batch, source, configs).to_parquet(out_dir / f"real-{tag}", index=False)
        for key in resid:  # AtP does not need the residual, and its backward needs the room
            del source[key]
        torch.cuda.empty_cache()
        keys = list(source)
        atp = []
        for sub in PT.token_batches([lengths[i] for i in group], atp_budget):
            atp.append(
                PT.attribution(
                    model,
                    ro,
                    PT.subset(batch, sub),
                    PT.subset_source(source, batch, sub, keys),
                    n_layers,
                    classes=COMPONENT_CLASSES,
                )
            )
            torch.cuda.empty_cache()
        pd.concat(atp, ignore_index=True).to_parquet(out_dir / f"atp-{tag}", index=False)
        del source
        torch.cuda.empty_cache()
        log.info(
            "%d pairs (longest %d tokens): %d patches in %.0fs",
            len(rows),
            max(lengths[i] for i in group),
            len(configs),
            time.time() - started,
        )


def verify_heads(
    model,
    tokenizer,
    ro,
    pairs,
    segs,
    out_dir: Path,
    token_budget: int,
    n_layers: int,
    top_k: int,
    n_random: int,
    seed: int,
) -> None:
    """Real patching of the heads AtP ranks highest, and of random heads as a control."""
    path = out_dir / "verify.parquet"
    if path.exists():
        return
    done = load_completed(out_dir)
    p, atp = done["pairs"], done["atp"]
    heads = PT.summarise(
        atp[atp["kind"] == "head_in"], p, by=["layer", "head", "cls"], value="atp_delta", n_boot=1
    )
    chosen = set()
    for cls in COMPONENT_CLASSES:
        top = heads[heads["cls"] == cls].reindex(
            heads[heads["cls"] == cls]["share_removed"].abs().sort_values(ascending=False).index
        )
        chosen |= {(int(r.layer), int(r.head)) for r in top.head(top_k).itertuples()}
    rng = np.random.default_rng(seed)
    every = [
        (layer, h) for layer in range(n_layers) for h in range(arch.head_shape(model, layer)[0])
    ]
    rest = [x for x in every if x not in chosen]
    controls = {rest[i] for i in rng.choice(len(rest), size=n_random, replace=False)}
    configs = [
        {"kind": "head_in", "layer": layer, "head": h, "cls": c, "control": False}
        for layer, h in sorted(chosen)
        for c in COMPONENT_CLASSES
    ] + [
        {"kind": "head_in", "layer": layer, "head": h, "cls": c, "control": True}
        for layer, h in sorted(controls)
        for c in COMPONENT_CLASSES
    ]
    lengths = [
        max(segs[a].n_tokens, segs[b].n_tokens) for a, b in zip(pairs["uid_a"], pairs["uid_b"])
    ]
    order = np.argsort(lengths, kind="stable")
    pairs = pairs.iloc[order].reset_index(drop=True)
    out = []
    for group in PT.token_batches([lengths[i] for i in order], token_budget):
        rows = pairs.iloc[group]
        batch = PT.make_batch(
            rows,
            [segs[u] for u in rows["uid_a"]],
            [segs[u] for u in rows["uid_b"]],
            tokenizer.pad_token_id,
        )
        source = PT.capture_source(model, batch, [("head_in", i) for i in range(n_layers)])
        for cfg in configs:
            m = PT.margins_of(
                model,
                ro,
                batch.ids_a,
                batch.mask_a,
                [
                    (
                        "head_in",
                        cfg["layer"],
                        batch.classes[cfg["cls"]],
                        source[("head_in", cfg["layer"])],
                        cfg["head"],
                    )
                ],
            )
            out.append(batch.rows[["pair_uid"]].assign(**cfg, m_patched=m))
        del source
        torch.cuda.empty_cache()
    pd.concat(out, ignore_index=True).to_parquet(path, index=False)
    log.info("verified %d heads (+%d random controls)", len(chosen), len(controls))


def summarise_run(out_dir: Path, name: str) -> None:
    done = load_completed(out_dir)
    p, real, atp = done["pairs"], done["real"], done["atp"]
    real["layer"] = real["layer"].fillna(-1).astype(int)
    rep = report_dir("patching", name)
    p.to_csv(rep / "pairs.csv", index=False)
    PT.summarise(real, p, by=["kind", "layer", "cls"]).to_csv(rep / "real.csv", index=False)
    PT.summarise(atp, p, by=["kind", "layer", "head", "cls"], value="atp_delta").to_csv(
        rep / "atp.csv", index=False
    )
    if (out_dir / "verify.parquet").exists():
        v = pd.read_parquet(out_dir / "verify.parquet")
        PT.summarise(v, p, by=["layer", "head", "cls", "control"]).to_csv(
            rep / "verify.csv", index=False
        )
    log.info("summaries in %s", rep)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", required=True)
    parser.add_argument("--dataset", choices=("grid", "pool"), required=True)
    parser.add_argument("--contrast", required=True)
    parser.add_argument("--n-pairs", type=int, default=64)
    parser.add_argument("--min-effect", type=float, default=0.5)
    parser.add_argument(
        "--token-budget", type=int, default=4800, help="rows x longest prompt per patching batch"
    )
    parser.add_argument(
        "--atp-budget", type=int, default=2400, help="rows x longest prompt per AtP backward pass"
    )
    parser.add_argument("--top-k", type=int, default=24, help="heads per class to verify")
    parser.add_argument("--n-random", type=int, default=8, help="random control heads")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter", action="append", default=[], metavar="NAME=PATH")
    args = parser.parse_args()

    cfg = data_config()
    # The pair selection is part of a run's identity: a resumed run must patch the same pairs.
    run = f"{args.contrast}-n{args.n_pairs}-s{args.seed}"
    name = f"{args.spec}-{args.dataset}-{run}"
    manifest = pd.read_parquet(
        interp_dir("manifest") / f"contrasts-{args.dataset}-{cfg['lang']}.parquet"
    )
    pairs = select_pairs(manifest, args.contrast, args.n_pairs, args.min_effect, args.seed)
    out_dir = interp_dir("patching", args.spec, args.dataset, run)
    selected = out_dir / "selected_pairs.parquet"
    if selected.exists():
        if set(pd.read_parquet(selected)["pair_uid"]) != set(pairs["pair_uid"]):
            raise SystemExit(f"{out_dir} holds a different pair selection; use a new --seed")
    else:
        pairs.to_parquet(selected, index=False)
    segs = load_segments(args.dataset, cfg["lang"], set(pairs["uid_a"]) | set(pairs["uid_b"]))
    log.info("%s: %d pairs", name, len(pairs))

    overrides = dict(item.split("=", 1) for item in args.adapter)
    specs = SP.load_spec_set(overrides=overrides)
    model, tokenizer = SP.load_models(specs, [args.spec], device=args.device)
    for param in model.parameters():
        param.requires_grad_(False)
    ro = M.readout(model, tokenizer, cfg["lang"])
    n_layers = len(arch.decoder_layers(model))

    with SP.use(model, specs.specs[args.spec]):
        run_batches(
            model, tokenizer, ro, pairs, segs, out_dir, args.token_budget, args.atp_budget, n_layers
        )
        verify_heads(
            model,
            tokenizer,
            ro,
            pairs,
            segs,
            out_dir,
            args.token_budget,
            n_layers,
            args.top_k,
            args.n_random,
            args.seed,
        )
    write_meta(
        out_dir / "run",
        spec=args.spec,
        dataset=args.dataset,
        contrast=args.contrast,
        pairs=len(pairs),
        min_effect=args.min_effect,
        seed=args.seed,
    )
    summarise_run(out_dir, name)


if __name__ == "__main__":
    main()
