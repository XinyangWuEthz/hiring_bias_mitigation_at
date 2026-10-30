"""v3 additional data: the SFT-v3 pipeline on English pairs beyond V1's (docs/V3_NEW_DATA.md).

    CFG=configs/generation/v3_additional_data.yaml
    python scripts/v3_additional_data.py --config $CFG --stage pool        # CPU
    python scripts/v3_additional_data.py --config $CFG --stage generate --parts rjdb-000 rjdb-001
    python scripts/v3_additional_data.py --config $CFG --stage generate --parts all
    python scripts/v3_additional_data.py --config $CFG --stage report      # CPU
    python scripts/v3_additional_data.py --config $CFG --stage v1_dpo      # CPU
    python scripts/v3_additional_data.py --config $CFG --stage assemble    # CPU
    python scripts/v3_additional_data.py --config $CFG --stage dpo_v3      # CPU, after round 2

SFT-v3 (docs/SFT_V3_ANCHORS.md) is v1's construction with two changes: the 8-draw vote's
majority as the anchor, and the whole-word `metrics.mentions_attribute` (9e02ae2) in every
filter. Its data sets cover V1's 3,000 English Djinni pairs (`artifacts/semisynthetic-v3/
sft_v3_swap`, `sft_v3_rebalanced`). This script runs the same construction on pairs V1 never
used, so the new rows extend those sets rather than replace them. V1's pairs are not drawn
again: they keep their v3 anchors and targets, and enter here only so new pairs can be checked
against them. Per new pair, in v1's order:

1. reference -- `synth.REFERENCE_PROMPT` drawn 8 times; `generation.vote.vote_anchors`. A new
   pair has no v1 anchor to fall back on, so a tie, no decided draw, or no usable majority
   feedback leaves it unanchored and out of every later pass;
2. variants -- `synth.build_variant_frame`, v1's 3 groups x 2 attributes x 2 conditions, from a
   generator seeded by the part's name, so a part's variants do not depend on other parts;
3. invariant -- `synth.INVARIANT_PROMPT` with the vote anchor quoted, `filters.filter_invariant`;
4. biased -- `synth.BIASED_PROMPT`, `filters.filter_biased`.

Every sampled pass seeds each prompt from its own text (`per_prompt_seed`): base 43 for the
vote, as job 5073453, and 42 for the variant passes, as the vote targets (job 5103710).

`pool` builds the new pairs from three sources and cuts them into parts named by source:
`djinni_new` (candidates V1 never used, one job each, through `pool.match_pairs`), `rjdb`
(train split) and `kaggle_linkedin`. A new pair whose (JD, CV) text repeats a V1 pair's or an
earlier new pair's is dropped, because two copies of one prompt would draw identical samples;
so is a prompt that cannot fit `max_model_len`. `generate` loads the teacher once and runs
every pass for each part it is given; each pass of each part is its own file, written
atomically and skipped when its prompts are already done, so a job cut by the time limit loses
at most the pass in flight. `v1_dpo` builds V1's pairs' DPO rows on the vote anchors from v1's
raw passes (CPU). `assemble` builds v1's SFT and DPO sets from the new kept rows, split by
candidate (a seeded 5% of each source's candidates in validation), and writes:
`additional_{sft,dpo}_*` (new rows alone); `sft_*` (SFT-v3's swap set, rows, order and split
exactly as built, with the new rows after it); `dpo_*` and `kto_*` (V1's pairs and the new
pairs); `dpo_capped_*`, at most `dpo_cap_per_pair` rows per pool pair at the natural decision
mix; and `dpo_capped_balanced_*`, the same with hire and reject rows equal. The biased teacher
never answers hire, so balancing makes half the rows a hire-over-reject decision contrast and
none the reverse (`contrast`); which view to train on is the user's choice.

A later round draws further parts of the same pool into its own directory, so the first
round's outputs, SFT-v3's training data among them, stay as built (round 3,
`configs/generation/v3_new_pairs.yaml`). `pool_dir` names the pool it reads, so its parts and
its validation split are the first round's, and `pool` refuses to run. `earlier_rounds` names
the directories whose parts it must not draw again: `generate` refuses such a part, `all`
skips them, and `assemble` refuses a candidate that is in an earlier round's `sft_*` files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data import external as X  # noqa: E402
from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.generation import dataset as D  # noqa: E402
from hiring_bias_mitigation.generation import filters as F  # noqa: E402
from hiring_bias_mitigation.generation import pool as PL  # noqa: E402
from hiring_bias_mitigation.generation import sources as S  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402
from hiring_bias_mitigation.generation import vote as V  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    load_config,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("v3_additional_data")
LANG = "en"
SOURCE_ORDER = ("djinni_new", "rjdb", "kaggle_linkedin")
#: Characters per token, an overestimate for Qwen3.5 on this text (the external eval frames'
#: token calibration found chars/3.5 never undercounts). Used only to drop prompts that could
#: not fit.
CHARS_PER_TOKEN = 3.5
PASSES = ("votes", "invariant", "biased")


def _code() -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout

    return {"commit": git("rev-parse", "--short", "HEAD").strip(),
            "uncommitted": [line for line in git("status", "--porcelain").splitlines() if line]}


def _out(cfg: dict) -> Path:
    return Path(resolve_output_path(cfg["artifacts_dir"]))


def _pool_path(cfg: dict) -> Path:
    """This directory's pool, or `pool_dir`'s for a later round of the same pool."""
    base = (Path(resolve_output_path(cfg["pool_dir"])) if cfg.get("pool_dir")
            else _out(cfg) / "pool")
    return base / "pairs_en.parquet"


def _earlier_rounds(cfg: dict) -> list[Path]:
    return [Path(resolve_output_path(p)) for p in cfg.get("earlier_rounds") or []]


def drawn_earlier(cfg: dict, parts) -> list[str]:
    """The parts an earlier round has voted on. Their pairs are in that round's SFT data, so a
    later round that drew them again would hand the student pairs it trained on."""
    return [p for p in parts if any((d / "parts" / p / "votes.parquet").exists()
                                    for d in _earlier_rounds(cfg))]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), "utf-8")
    tmp.replace(path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _part_seed(part: str, base: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}:{part}".encode()).digest()[:4], "big")


def part_order(parts) -> list[str]:
    return sorted(set(parts), key=lambda p: (SOURCE_ORDER.index(p.rsplit("-", 1)[0]), p))


# ---- pool -----------------------------------------------------------------------------------

def v1_pairs(cfg: dict) -> pd.DataFrame:
    """V1's English pairs: excluded from generation, used for exclusion and dedup only."""
    v1 = pd.read_parquet(resolve_output_path(cfg["pool"]["v1_reference"]))
    return v1.drop(columns=["raw_output"], errors="ignore")


def djinni_new_pairs(cfg: dict, v1: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Djinni candidates V1 did not use, each with one job, through v1's matcher."""
    spec = cfg["pool"]["djinni_new"]
    pool_cfg = PL.PoolConfig(lang=LANG, n_pairs=int(spec["n_pairs"]), jobs_per_candidate=1,
                             seed=int(spec["seed"]))
    candidates, jobs = PL.load_djinni(LANG)
    candidates = PL.filter_candidates(candidates, pool_cfg)
    before = len(candidates)
    candidates = candidates[~candidates["id"].astype(str).isin(set(v1["candidate_id"]))]
    pairs = PL.match_pairs(candidates.reset_index(drop=True), PL.filter_jobs(jobs, pool_cfg),
                           pool_cfg)
    report = {"candidates_after_filters": before, "candidates_not_in_v1": len(candidates),
              "pairs": len(pairs)}
    return pairs.assign(source="djinni_new", occupation=pairs["primary_keyword"],
                        cv_variant="original"), report


def external_pairs(cfg: dict) -> tuple[pd.DataFrame, dict]:
    spec = cfg["pool"]["external"]
    pool_cfg = PL.PoolConfig(lang=LANG, seed=int(spec.get("seed", 42)),
                             jobs_per_candidate=int(spec.get("jobs_per_candidate", 2)),
                             sources=tuple(spec["pairs_per_source"]),
                             pairs_per_source={k: int(v) for k, v in
                                               spec["pairs_per_source"].items()},
                             rjdb_include_weakened=bool(spec.get("rjdb_include_weakened", False)))
    return PL.build_pool_with_report(pool_cfg)


def _text_key(frame: pd.DataFrame) -> pd.Series:
    return frame["job_description"].str.strip() + "\x1e" + frame["cv"].str.strip()


def stage_pool(cfg: dict) -> None:
    require_output_root()
    if cfg.get("pool_dir"):
        raise SystemExit(f"pool: this round reads the pool in {cfg['pool_dir']}; rebuilding it "
                         "here could cut different parts")
    out = _out(cfg) / "pool"
    v1 = v1_pairs(cfg)
    new, new_report = djinni_new_pairs(cfg, v1)
    ext, funnel = external_pairs(cfg)
    pool = pd.concat([f[S.POOL_COLUMNS] for f in (new, ext) if len(f)], ignore_index=True)
    report = {"built": datetime.now(timezone.utc).isoformat(), "code": _code(),
              "config": cfg["_path"], "djinni_new": new_report, "external_funnel": funnel,
              "by_source_built": pool["source"].value_counts().to_dict()}

    # One copy of each (JD, CV) text, and none that repeats a V1 pair's.
    key = _text_key(pool)
    dup = key.duplicated() | key.isin(set(_text_key(v1)))
    report["text_duplicates_dropped"] = pool.loc[dup, "source"].value_counts().to_dict()
    pool = pool[~dup].reset_index(drop=True)
    if pool["pair_id"].duplicated().any() or pool["pair_id"].isin(set(v1["pair_id"])).any():
        raise ValueError("pool: a pair_id repeats within the pool or repeats a V1 pair")
    if pool["candidate_id"].isin(set(v1["candidate_id"])).any():
        raise ValueError("pool: a V1 candidate reached the new pool")

    tokens = pd.Series([len(p) / CHARS_PER_TOKEN for p in SY.reference_prompts(pool, LANG)])
    # The invariant prompt is the longest pass: profile + the quoted anchor + instructions.
    over = tokens + int(cfg["pool"]["invariant_overhead_tokens"]) > int(
        cfg["pool"]["max_prompt_tokens"])
    report["over_length_dropped"] = pool.loc[over, "source"].value_counts().to_dict()
    pool = pool[~over].assign(est_prompt_tokens=tokens[~over].round().astype(int))
    pool = pool.reset_index(drop=True)

    assert_no_leakage(pool, "v3_additional_data.pool")
    # The external builders already drop held-out text; this also covers the Djinni rows.
    leaks = X.find_external_leaks(pool)
    report["external_holdout_dropped"] = (pool.loc[leaks["row"].unique(), "source"]
                                          .value_counts().to_dict() if len(leaks) else {})
    pool = pool.drop(index=leaks["row"].unique()).reset_index(drop=True)
    X.assert_no_external_leakage(pool, "v3_additional_data.pool")

    size = int(cfg["generate"]["part_size"])
    pool["order"] = pool.groupby("source").cumcount()
    pool["part"] = [f"{s}-{o // size:03d}" for s, o in zip(pool["source"], pool["order"])]
    _write_parquet(pool, out / "pairs_en.parquet")
    report["by_source"] = pool["source"].value_counts().to_dict()
    report["parts"] = {p: int((pool["part"] == p).sum()) for p in part_order(pool["part"])}
    report["est_prompt_tokens"] = {s: g["est_prompt_tokens"].describe().round(0).to_dict()
                                   for s, g in pool.groupby("source")}
    report["cv_chars_median"] = {s: int(g["cv"].str.len().median())
                                 for s, g in pool.groupby("source")}
    report["jd_chars_median"] = {s: int(g["job_description"].str.len().median())
                                 for s, g in pool.groupby("source")}
    _write_json(out / "pool_report.json", report)
    log.info("pool: %d pairs in %d parts: %s", len(pool), pool["part"].nunique(),
             report["by_source"])


# ---- generate -------------------------------------------------------------------------------

def _paths(cfg: dict, part: str, name: str) -> tuple[Path, Path]:
    base = _out(cfg) / "parts" / part
    return base / f"{name}.parquet", base / f"{name}.meta.json"


def _pass_sha256(cfg: dict, name: str) -> str:
    """What else, besides the prompts, decides a pass's generations."""
    spec = {"teacher": cfg["teacher_model"], "generation": cfg["passes"][name],
            "n": int(cfg["vote_draws"]) if name == "votes" else 1}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _done(cfg: dict, part: str, name: str, prompts: list[str]) -> bool:
    path, meta = _paths(cfg, part, name)
    if not (path.exists() and meta.exists()):
        return False
    try:
        recorded = json.loads(meta.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return (recorded.get("prompts_sha256") == V.prompts_sha256(prompts)
            and recorded.get("pass_sha256") == _pass_sha256(cfg, name))


def part_anchors(cfg: dict, part: str, pairs: pd.DataFrame) -> pd.DataFrame:
    draws = pd.read_parquet(_paths(cfg, part, "votes")[0])
    anchors = V.vote_anchors(draws, pairs["pair_id"], LANG)
    return pairs.merge(anchors, on="pair_id", validate="one_to_one")


def part_variants(cfg: dict, part: str, anchored: pd.DataFrame) -> pd.DataFrame:
    synth_cfg = SY.SynthConfig(lang=LANG, protected_groups=tuple(cfg["protected_groups"]),
                               conditions=tuple(cfg["conditions"]),
                               attributes_per_pair=int(cfg["attributes_per_pair"]),
                               seed=_part_seed(part, int(cfg["seed"])))
    return SY.build_variant_frame(anchored, synth_cfg)


def _vllm_version() -> str | None:
    try:
        import vllm

        return vllm.__version__
    except Exception:  # the fake teacher of the tests, or no vLLM on this node
        return None


def _generate_pass(cfg, backend, part, name, frame, prompts, generation, sampled_n=None,
                   load_seconds=None):
    from hiring_bias_mitigation.eval import backends as BK

    start = time.time()
    if sampled_n:
        draws = backend.generate_samples(prompts, sampled_n)
        out = V.draw_frame(frame["pair_id"].tolist(), draws, LANG)
    else:
        out = frame.assign(raw_output=backend.generate(prompts))
    seeds = [BK.per_prompt_seed(int(generation["seed"]), p) for p in prompts]
    out["request_seed"] = np.repeat(seeds, sampled_n) if sampled_n else seeds
    path, meta = _paths(cfg, part, name)
    _write_parquet(out, path)
    _write_json(meta, {
        "part": part, "pass": name, "prompts": len(prompts), "rows": len(out),
        "prompts_sha256": V.prompts_sha256(prompts), "pass_sha256": _pass_sha256(cfg, name),
        "teacher": cfg["teacher_model"], "serving": cfg["serving"], "generation": generation,
        "seeding": "per_prompt_seed(generation.seed, prompt)",
        "generation_seconds": round(time.time() - start, 1),
        "engine_load_seconds": load_seconds, "vllm": _vllm_version(), "code": _code(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "written": datetime.now(timezone.utc).isoformat()})
    log.info("%s %s: %d prompts in %.0f s", part, name, len(prompts), time.time() - start)


def stop_reasons(cfg: dict, part: str, pairs: pd.DataFrame) -> tuple[list[str], dict]:
    """docs/V3_NEW_DATA.md's stop conditions, read from one part's own files.

    A pass that was not run (a job with `--passes votes invariant`) has no readout, and its
    condition is not checked.
    """
    limits = cfg["stop"]
    draws = pd.read_parquet(_paths(cfg, part, "votes")[0])
    anchors = part_anchors(cfg, part, pairs)
    readout = {"undecided_draw_share": float((~draws["decision"].isin(V.DECIDED)).mean()),
               "unanchored_share": float((~anchors["anchored"]).mean())}
    for name, filt in (("invariant", F.filter_invariant), ("biased", F.filter_biased)):
        path = _paths(cfg, part, name)[0]
        if path.exists():
            raw = pd.read_parquet(path)
            _, rep = filt(raw["raw_output"].tolist(), raw.drop(columns=["raw_output"]), LANG)
            readout[f"{name}_yield"] = rep["yield"]
    checks = [("undecided_draw_share", ">", limits["max_undecided_draw_share"]),
              ("unanchored_share", ">", limits["max_unanchored_share"]),
              ("invariant_yield", "<", limits["min_invariant_yield"]),
              ("biased_yield", "<", limits["min_biased_yield"])]
    reasons = [f"{k} {readout[k]:.4f} {op} {limit}" for k, op, limit in checks
               if k in readout and (readout[k] > limit if op == ">" else readout[k] < limit)]
    return reasons, readout


def stage_generate(cfg: dict, parts: list[str], passes: list[str]) -> None:
    from hiring_bias_mitigation.eval import backends as BK

    require_output_root()
    pool = pd.read_parquet(_pool_path(cfg))
    known = part_order(pool["part"])
    earlier = set(drawn_earlier(cfg, known))
    wanted = [p for p in known if p not in earlier] if parts == ["all"] else parts
    unknown = sorted(set(wanted) - set(known))
    if unknown:
        raise SystemExit(f"unknown part(s) {unknown}; the pool has {known[:5]}...")
    repeated = sorted(set(wanted) & earlier)
    if repeated:
        raise SystemExit(f"part(s) {repeated} were drawn by an earlier round "
                         f"({cfg['earlier_rounds']}); this round draws new parts only")

    engine: dict[str, object] = {}

    def backend_for(name: str):
        # One engine for the job; each pass sets its own decoding before it runs. The engine is
        # loaded before the first pass's timer starts, so its seconds are the pass's alone.
        generation = dict(cfg["passes"][name])
        load_seconds = None
        if "llm" not in engine:
            start = time.time()
            engine["llm"] = BK.build_backend({**cfg["serving"], "model": cfg["teacher_model"],
                                              "generation": generation})
            if hasattr(engine["llm"], "_load"):
                engine["llm"]._load()
            load_seconds = round(time.time() - start, 1)
        engine["llm"].generation = BK.GenerationConfig(**generation)
        return engine["llm"], generation, load_seconds

    checked: dict[str, list[str]] = {}

    try:
        # Explicit parts run in the order given, so a job can put each source's first part
        # first and have it read against the stop conditions while the rest runs.
        for part in (wanted if parts == ["all"] else list(dict.fromkeys(wanted))):
            source = part.rsplit("-", 1)[0]
            if checked.get(source):
                log.warning("%s: skipped, %s stopped on its first part: %s", part, source,
                            checked[source])
                continue
            pairs = pool[pool["part"] == part].reset_index(drop=True)
            if "votes" in passes:
                prompts = SY.reference_prompts(pairs, LANG)
                if _done(cfg, part, "votes", prompts):
                    log.info("%s votes: already drawn for these prompts, skipped", part)
                else:
                    backend, generation, load = backend_for("votes")
                    _generate_pass(cfg, backend, part, "votes", pairs, prompts, generation,
                                   sampled_n=int(cfg["vote_draws"]), load_seconds=load)
            if not _paths(cfg, part, "votes")[0].exists():
                log.warning("%s: no votes yet; the variant passes are skipped", part)
                continue
            anchored = part_anchors(cfg, part, pairs)
            variants = part_variants(cfg, part, anchored[anchored["anchored"]]
                                     .reset_index(drop=True))
            for name, builder in (("invariant", SY.invariant_prompts),
                                  ("biased", SY.biased_prompts)):
                if name not in passes:
                    continue
                prompts = builder(variants, LANG)
                if _done(cfg, part, name, prompts):
                    log.info("%s %s: already generated for these prompts, skipped", part, name)
                    continue
                backend, generation, load = backend_for(name)
                _generate_pass(cfg, backend, part, name, variants, prompts, generation,
                               load_seconds=load)
            if source not in checked and "invariant" in passes:
                # The first part of each source in this job is read against the stop conditions;
                # a source that crosses one gets no further part (docs/V3_NEW_DATA.md).
                reasons, readout = stop_reasons(cfg, part, pairs)
                checked[source] = reasons
                _write_json(_out(cfg) / "parts" / part / "stop_check.json",
                            {"part": part, "readout": readout, "reasons": reasons,
                             "limits": cfg["stop"]})
                if reasons:
                    log.warning("%s: STOP for %s: %s", part, source, reasons)
                else:
                    log.info("%s: stop conditions clear: %s", part, readout)
    finally:
        if "llm" in engine:
            engine["llm"].close()


# ---- report and assemble --------------------------------------------------------------------

def _voted_parts(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    pool = pd.read_parquet(_pool_path(cfg))
    return pool, [p for p in part_order(pool["part"]) if _paths(cfg, p, "votes")[0].exists()]


def _filtered(cfg: dict, done: list[str]) -> tuple[dict, dict]:
    """Kept rows and filter reports of every variant pass on disk, by pass."""
    kept, reports = {"invariant": [], "biased": []}, {}
    for part in done:
        for name, filt in (("invariant", F.filter_invariant), ("biased", F.filter_biased)):
            path = _paths(cfg, part, name)[0]
            if not path.exists():
                continue
            raw = pd.read_parquet(path)
            rows, reports[f"{part}/{name}"] = filt(raw["raw_output"].tolist(),
                                                   raw.drop(columns=["raw_output"]), LANG)
            kept[name].append(rows)
    return kept, reports


#: IT job titles (the data-quality review's classifier, 2026-10-20), for the persona readout.
IT_TITLE = re.compile(
    r"(?i)\b(?:software|developer|programmer|devops|sre|data (?:scientist|engineer|analyst"
    r"|architect|warehouse)|database|dba|sql|java|python|\.net|c#|c\+\+|php|ruby|javascript"
    r"|react|angular|node(?:\.?js)?|salesforce|sap|oracle|sharepoint|web|ui|ux|front[- ]?end"
    r"|back[- ]?end|full[- ]?stack|it|information technology|network|system(?:s)? (?:admin\w*"
    r"|engineer|analyst)|sysadmin|help ?desk|service desk|desktop support|technical support"
    r"|cyber\w*|information security|security (?:analyst|engineer|architect)|qa"
    r"|quality assurance|tester|test (?:engineer|analyst|automation|lead|manager)|cloud|aws"
    r"|azure|linux|unix|mainframe|etl|bi|business intelligence|tableau|hadoop|spark|big data"
    r"|machine learning|ml|ai|computer|application (?:developer|support|analyst|engineer"
    r"|architect)|solutions? architect|scrum master|technical (?:lead|architect"
    r"|project manager)|it project|ios|android|mobile (?:app\w*|developer|engineer)|embedded"
    r"|firmware|peoplesoft|workday|servicenow|dynamics|informatica|cognos|abap|golang|scala"
    r"|kotlin|swift|perl|rpa|blockchain|game developer)\b")
#: Djinni's own non-technical role keywords (`primary_keyword`).
DJINNI_NON_IT = {
    "HR", "Marketing", "Sales", "Design", "Project Manager", "Product Manager", "Support",
    "Copywriting", "Lead Generation", "Business Analyst", "Artist", "Finances", "Legal", "SEO",
    "Customer Success", "Other", "Sales Manager", "Account Manager", "Marketing Manager",
    "Recruiter", "Office Manager", "Lead Gen", "Technical Writing", "Translator", "Media Buyer"}
TECH_WORDS = re.compile(r"(?i)\btechnical\b|\btech stack\b|\bprogramming\b|\bcoding\b")


def persona_readout(anchors: pd.DataFrame) -> dict:
    """`synth.REFERENCE_PROMPT` calls the teacher a technical recruiter on every pair.

    Per pool source, IT against non-IT roles: the anchor's hire share and the share of anchor
    rationales that cite technical skills. If the non-IT rows diverge, the persona may be
    steering anchors on the external sources (data-quality review, 2026-10-20).
    """
    a = anchors[anchors["anchored"]].copy()
    if a.empty:
        return {}
    titled = a["job_position"].fillna("").map(lambda s: bool(IT_TITLE.search(str(s))))
    djinni = a["source"].str.startswith("djinni")
    kaggle_it = a["occupation"].eq("INFORMATION-TECHNOLOGY")
    a["role"] = np.where(np.where(djinni, ~a["primary_keyword"].isin(DJINNI_NON_IT),
                                  titled | kaggle_it), "it", "non_it")
    a["hire"] = a["reference_decision"].eq("hire")
    a["tech"] = a["reference_feedback"].fillna("").str.contains(TECH_WORDS)
    out: dict[str, dict] = {}
    for (source, role), g in a.groupby(["source", "role"], sort=True):
        out.setdefault(source, {})[role] = {"pairs": len(g),
                                            "anchor_hire_share": round(float(g["hire"].mean()), 4),
                                            "tech_word_share": round(float(g["tech"].mean()), 4)}
    return out


def stage_report(cfg: dict) -> None:
    pool, done = _voted_parts(cfg)
    if not done:
        raise SystemExit("report: no part has votes yet")
    anchors = pd.concat([part_anchors(cfg, p, pool[pool["part"] == p].reset_index(drop=True))
                         for p in done], ignore_index=True)
    draws = pd.concat([pd.read_parquet(_paths(cfg, p, "votes")[0]) for p in done],
                      ignore_index=True)
    _, reports = _filtered(cfg, done)
    by_source: dict[str, dict] = {}
    for key, rep in reports.items():
        part, name = key.split("/")
        agg = by_source.setdefault(part.rsplit("-", 1)[0], {}).setdefault(
            name, {"n_input": 0, "n_kept": 0, "dropped_by_reason": {}})
        agg["n_input"] += rep["n_input"]
        agg["n_kept"] += rep["n_kept"]
        for reason, n in rep["dropped_by_reason"].items():
            agg["dropped_by_reason"][reason] = agg["dropped_by_reason"].get(reason, 0) + n
    for passes in by_source.values():
        for agg in passes.values():
            agg["yield"] = round(agg["n_kept"] / agg["n_input"], 4) if agg["n_input"] else None
    seconds = {}
    for p in done:
        for name in PASSES:
            meta = _paths(cfg, p, name)[1]
            if meta.exists():
                m = json.loads(meta.read_text())
                seconds.setdefault(name, []).append(
                    {"part": p, "prompts": m["prompts"], "seconds": m["generation_seconds"]})
    # What the biased teacher answers, by anchor: v1's never said hire (0 of 36,000), which
    # makes every DPO decision contrast hire over reject (see `contrast`).
    from hiring_bias_mitigation.eval.parsing import parse_output

    biased_decisions: dict[str, dict] = {}
    for p in done:
        path = _paths(cfg, p, "biased")[0]
        if path.exists():
            raw = pd.read_parquet(path, columns=["reference_decision", "raw_output"])
            got = [parse_output(text, LANG).decision for text in raw["raw_output"]]
            counts = biased_decisions.setdefault(p.rsplit("-", 1)[0], {})
            for anchor, decision in zip(raw["reference_decision"], got):
                key = f"anchor {anchor} -> {decision}"
                counts[key] = counts.get(key, 0) + 1
    stop_checks = {}
    for p in done:
        path = _out(cfg) / "parts" / p / "stop_check.json"
        if path.exists():
            stop_checks[p] = json.loads(path.read_text())
    _write_parquet(anchors, _out(cfg) / "anchors_en.parquet")
    _write_json(_out(cfg) / "report.json", {
        "built": datetime.now(timezone.utc).isoformat(), "code": _code(), "parts_voted": done,
        "vote": V.vote_report(anchors, draws, by=anchors["source"]),
        "filters_by_source": by_source, "biased_decisions_by_source": biased_decisions,
        "persona_by_source": persona_readout(anchors), "stop_checks": stop_checks,
        "generation_seconds": seconds})
    log.info("report: %d pairs voted over %d parts", len(anchors), len(done))


def validation_candidates(cfg: dict, frame: pd.DataFrame) -> set[str]:
    """A seeded `val_fraction` of each source's candidates (none of them is a V1 candidate)."""
    rng = np.random.default_rng(int(cfg["seed"]))
    val: set[str] = set()
    for source in SOURCE_ORDER:
        ids = np.array(sorted(frame.loc[frame["source"] == source, "candidate_id"].unique()))
        if len(ids):
            n = max(1, int(len(ids) * float(cfg["split"]["val_fraction"])))
            val |= set(rng.permutation(ids)[:n].tolist())
    return val


def _decision(completions: pd.Series) -> pd.Series:
    return completions.map(lambda c: json.loads(c)["decision"])


def anchor_columns(anchors: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: its vote, in the same terms for V1's pairs and for new ones.

    `anchor_unstable` is the vote alone -- no decided draw, or a majority of at most 5 of the
    decided draws -- so it means the same on every row. On V1's pairs `vote.UNSTABLE` also
    counted a verdict the vote overturned; that is `anchor_changed` here, False for new pairs.
    """
    hire, decided = anchors["hire_votes"].astype(int), anchors["decided"].astype(int)
    share = np.maximum(hire, decided - hire) / decided.clip(lower=1)
    return pd.DataFrame({
        "pair_id": anchors["pair_id"], "anchor_hire_votes": hire, "anchor_decided": decided,
        "anchor_unstable": (decided == 0) | (share <= V.UNSTABLE_MAX_SHARE),
        "anchor_changed": (anchors["changed"].astype(bool) if "changed" in anchors
                           else False),
        "anchor_source": anchors["anchor_source"] if "anchor_source" in anchors
        else anchors["source"],
    })


def with_anchor(rows: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """SFT or DPO rows with their pair's vote, and `reference_decision`, the anchor's verdict.

    `reference_decision` is the column `mitigation/decision_eval.py` and
    `scripts/eval_dpo_checkpoints.py` read, as every v2 DPO set carries it; a DPO row's
    `decision` is its chosen verdict, which is the anchor's.
    """
    out = rows.merge(anchor_columns(anchors), on="pair_id", how="left", validate="many_to_one")
    if out["anchor_hire_votes"].isna().any():
        raise ValueError("a row has no anchor")
    if "decision" not in out:
        out["decision"] = _decision(out["chosen"])
    return out.assign(reference_decision=out["decision"])


def provenance(rows: pd.DataFrame, pool_source) -> pd.DataFrame:
    """`source` as the publish guard reads it, and `pool_source`, the part of the pool.

    `sources.is_publishable` knows `djinni`, `rjdb` and `kaggle_linkedin`. V1's pairs and the
    new Djinni pairs are both Djinni, so they are `source="djinni"`; `pool_source` keeps
    `djinni_v1` and `djinni_new` apart for readouts.
    """
    pool_source = pd.Series(pool_source, index=rows.index) if not isinstance(
        pool_source, pd.Series) else pool_source
    return rows.assign(pool_source=pool_source,
                       source=pool_source.where(~pool_source.str.startswith("djinni"), "djinni"))


def contrast(dpo: pd.DataFrame) -> dict:
    """Which decision contrasts a DPO set teaches.

    v1's biased teacher never answers hire (0 of 36,000 English generations), so every
    `rejected` says reject: a hire row contrasts hire with reject, and a reject row contrasts
    two rejects that differ only in their rationale. The share of hire rows is therefore the
    share of rows whose gradient moves the decision, all of it toward hire.
    """
    rejected = _decision(dpo["rejected"]) if len(dpo) else pd.Series(dtype=str)
    pairs = pd.Series(list(zip(dpo["decision"], rejected)), dtype=object)
    return {"rows": len(dpo), "pairs": int(dpo["pair_id"].nunique()),
            "chosen_rejected": {f"{c}>{r}": int(n) for (c, r), n in pairs.value_counts().items()},
            "decision_contrast_share": float((dpo["decision"].to_numpy()
                                              != rejected.to_numpy()).mean()) if len(dpo) else None,
            "by_pool_source": dpo["pool_source"].value_counts().to_dict(),
            "by_pool_source_and_contrast": {
                f"{s} {c}>{r}": int(n) for (s, c, r), n in
                pd.Series(list(zip(dpo["pool_source"], dpo["decision"], rejected)), dtype=object)
                .value_counts().items()} if len(dpo) else {}}


def capped(dpo: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """At most `cap` rows per pool pair, rows of different protected groups first.

    A pair's 12 variants share one CV and one JD, so its DPO rows are near-duplicates of each
    other; the cap spreads a set of a given size over more pairs.
    """
    rng = np.random.default_rng(seed)
    shuffled = dpo.iloc[rng.permutation(len(dpo))].copy()
    shuffled["_group_rank"] = shuffled.groupby(["pair_id", "protected_group"]).cumcount()
    shuffled = shuffled.sort_values(["pair_id", "_group_rank"], kind="stable")
    out = shuffled.groupby("pair_id").head(cap).drop(columns="_group_rank")
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def balanced(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    """As many hire as reject rows within each pool source, the larger side drawn down.

    Balancing the pooled rows would let a source with a high hire share (RJDB's matched CVs
    were written to fit their JD) supply the hire side, and leave the Djinni rows -- the
    benchmark's own kind -- reject-heavy: an operating-point shift by source.
    """
    rng = np.random.default_rng(seed)
    kept = []
    for _, source_rows in rows.groupby("pool_source", sort=True):
        sides = source_rows.groupby("decision")
        n = int(sides.size().min()) if source_rows["decision"].nunique() == 2 else 0
        kept.extend(g.iloc[rng.permutation(len(g))[:n]] for _, g in sides if n)
    if not kept:
        return rows.iloc[:0]
    out = pd.concat(kept, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


#: RJDB's GPT-4 template left in a CV ("Education: Details not provided", "[insert company name
#: here]"), which the teacher reads as a fact about the candidate, e.g. a missing degree.
RJDB_TEMPLATE_RESIDUE = re.compile(
    r"(?i)\b(?:not provided|not specified|if not applicable|please disregard)\b|\[insert")


def quality_exclusions(cfg: dict, pairs: pd.DataFrame) -> tuple[set[str], dict]:
    """Pairs the teacher has answered but training should not use, decided on their CV.

    * `djinni_new` CVs pass V1's marker list only (`pool._MARKERS`), which misses names,
      third-person pronouns and maternity leave; those contradict an injected gender. Here they
      meet the external sources' list (`sources.has_external_extra_marker`). V1's own rows, in
      SFT-v3's sets, keep V1's filter (about the same 3.3% would fail it there too).
    * RJDB CVs with template residue (`RJDB_TEMPLATE_RESIDUE`).

    Dropping at assembly keeps every generated part valid; the cost is the GPU time spent on
    about 3-4% of these sources' pairs.
    """
    rules = cfg["assemble"].get("exclude", {})
    djinni = (pairs["source"] == "djinni_new") & bool(rules.get("djinni_new_extra_markers"))
    marker = djinni & pairs["cv"].map(S.has_external_extra_marker)
    rjdb = (pairs["source"] == "rjdb") & bool(rules.get("rjdb_template_residue"))
    residue = rjdb & pairs["cv"].str.contains(RJDB_TEMPLATE_RESIDUE)
    report = {"rules": rules, "djinni_new_extra_marker": int(marker.sum()),
              "rjdb_template_residue": int(residue.sum()),
              "pairs_checked_by_source": pairs["source"].value_counts().to_dict()}
    return set(pairs.loc[marker | residue, "pair_id"]), report


def direction_readout(dpo: pd.DataFrame, seed: int, max_rows: int = 6000) -> dict:
    """`direction_check`'s surface checks (D1-D3, D5, D6) overall and per pool source.

    D1 fails by construction here: v1's biased rejected keeps the reject verdict on reject
    rows. D3 is the wording shortcut; v1's English set separates at 1.000.
    """
    from hiring_bias_mitigation.generation import direction_check as DC

    def one(rows):
        return DC.surface_report(DC.subsample(rows, max_rows, seed), seed=seed)["overall"]

    return {"overall": one(dpo),
            "by_pool_source": {s: one(g.reset_index(drop=True))
                               for s, g in dpo.groupby("pool_source", sort=True)}}


def stage_v1_dpo(cfg: dict) -> None:
    """V1's pairs' teacher-biased DPO rows on the vote anchors, from v1's raw passes; CPU only.

    Not for training: v1's biased-teacher negatives separate from chosen by wording (D3 =
    1.000), which is why r2em replaced them, and the user ruled them out on 2026-10-20. The
    files are named `teacher_biased_*` and kept only as a comparison.

    `synth.BIASED_PROMPT` does not quote the anchor, so v1's biased generations stand for every
    V1 pair, including the 192 whose verdict the vote changed; only `filter_biased`'s flip check
    reads the anchor, and it is re-run against the vote's. The chosen side is SFT-v3's: v1's
    invariant targets on unchanged pairs and job 5103710's regenerated targets on changed ones.
    Both sides go through the current filters, with the whole-word attribute matcher. The split
    is SFT-v3's: a candidate in `sft_v3_swap`'s validation is in validation here.
    """
    require_output_root()
    spec = cfg["v1_dpo"]
    anchors = pd.read_parquet(resolve_output_path(spec["vote_anchors"]))
    changed = set(anchors.loc[anchors["changed"], "pair_id"])
    keep = set(anchors["pair_id"])
    inv = pd.read_parquet(resolve_output_path(spec["v1_invariant"]))
    inv = inv[inv["pair_id"].isin(keep) & ~inv["pair_id"].isin(changed)]
    regen = pd.read_parquet(resolve_output_path(spec["vote_targets"]))
    regen = regen[regen["pair_id"].isin(changed)].drop(columns=["v1_reference_decision", "seed"])
    chosen_raw = pd.concat([inv, regen[inv.columns]], ignore_index=True)
    invariant, rep_inv = F.filter_invariant(chosen_raw["raw_output"].tolist(),
                                            chosen_raw.drop(columns=["raw_output"]), LANG)
    by_pair = anchors.set_index("pair_id")
    biased_raw = pd.read_parquet(resolve_output_path(spec["v1_biased"]))
    biased_raw = biased_raw[biased_raw["pair_id"].isin(keep)].copy()
    biased_raw["reference_decision"] = biased_raw["pair_id"].map(by_pair["new_decision"])
    biased_raw["reference_feedback"] = biased_raw["pair_id"].map(by_pair["new_feedback"])
    biased, rep_b = F.filter_biased(biased_raw["raw_output"].tolist(),
                                    biased_raw.drop(columns=["raw_output"]), LANG)
    dpo = with_anchor(D.build_dpo_dataset(invariant, biased), anchors)
    dpo = provenance(dpo, "djinni_v1")
    assert_no_leakage(dpo, "v3_additional_data.v1_dpo")
    base = Path(resolve_output_path(cfg["assemble"]["sft_v3_dir"]))
    val = set(pd.read_parquet(base / "sft_validation.parquet", columns=["candidate_id"])
              ["candidate_id"])
    is_val = dpo["candidate_id"].isin(val)
    out = _out(cfg)
    _write_parquet(dpo[~is_val].reset_index(drop=True),
                   out / "teacher_biased_v1_dpo_train.parquet")
    _write_parquet(dpo[is_val].reset_index(drop=True),
                   out / "teacher_biased_v1_dpo_validation.parquet")
    _write_json(out / "teacher_biased_v1_dpo_report.json", {
        "built": datetime.now(timezone.utc).isoformat(), "code": _code(), "inputs": spec,
        "pairs": len(keep), "changed_pairs": len(changed), "invariant": rep_inv,
        "biased": rep_b, "rows": len(dpo), "train": int((~is_val).sum()),
        "validation": int(is_val.sum()),
        "by_decision": dpo["decision"].value_counts().to_dict(),
        "rejected_flipped_share": float(dpo["rejected_flipped_decision"].mean()),
        "rejected_mentions_attribute_share": float(dpo["rejected_mentions_attribute"].mean()),
        "rows_on_unstable_anchors": int(dpo["anchor_unstable"].sum()),
        "rows_on_changed_anchors": int(dpo["anchor_changed"].sum()),
        "contrast": contrast(dpo)})
    log.info("v1_dpo: %d rows (%s)", len(dpo), dpo["decision"].value_counts().to_dict())


V1_INVARIANT_COLUMNS = [
    "pair_id", "candidate_id", "job_id", "cv", "job_description", "job_position",
    "primary_keyword", "candidate_years", "job_required_years", "lang", "reference_decision",
    "reference_feedback", "protected_group", "protected_attr", "condition", "group_label",
    "profile", "raw_output",
]


def stage_r2em(cfg: dict) -> None:
    """r2em's DPO pairs, made consistent with the vote anchors; CPU only.

    V3's DPO set is r2em plus new on-policy pairs drawn the same way (user, 2026-10-20). r2em
    was built on V1's pairs with v1's anchors, so a row whose chosen verdict the vote overturns
    would push DPO against the anchor SFT-v3 trains on (49 train and 20 validation rows, all
    chosen reject where the vote hires); its `legacy` rows are v1's biased-teacher negatives
    (42 and 3). Both are dropped by default (`r2em:` in the config). r2em's own pool label,
    which it wrote into `source`, becomes `r2em_pool`; `source` and `pool_source` then follow
    `provenance`. Rows keep r2em's split.
    """
    require_output_root()
    spec = cfg["r2em"]
    votes = pd.read_parquet(resolve_output_path(spec["vote_anchors"]))
    by_pair = votes.set_index("pair_id")["new_decision"]
    out, report = _out(cfg), {"inputs": spec}
    for split in ("train", "validation"):
        rows = pd.read_parquet(Path(resolve_output_path(spec["dir"]))
                               / f"{spec['name']}_{split}.parquet")
        rows = rows.rename(columns={"source": "r2em_pool"})
        if not rows["pair_id"].isin(by_pair.index).all():
            raise ValueError("r2em: a pair is outside the vote")
        conflict = rows["pair_id"].map(by_pair) != rows["decision"]
        legacy = rows["r2em_pool"] == "legacy"
        drop = ((conflict & bool(spec["drop_conflicting"]))
                | (legacy & bool(spec["drop_legacy"])))
        kept = provenance(with_anchor(rows[~drop].reset_index(drop=True), votes), "djinni_v1")
        assert_no_leakage(kept, f"v3_additional_data.r2em.{split}")
        _write_parquet(kept, out / f"r2em_v3_{split}.parquet")
        report[split] = {"rows": len(rows), "conflicting_with_vote": int(conflict.sum()),
                         "legacy": int(legacy.sum()), "dropped": int(drop.sum()),
                         "kept": len(kept),
                         "kept_on_unstable_anchors": int(kept["anchor_unstable"].sum()),
                         "contrast": contrast(kept)}
    _write_json(out / "r2em_v3_report.json", {
        "built": datetime.now(timezone.utc).isoformat(), "code": _code(), **report})
    log.info("r2em: %s", {s: (report[s]["rows"], report[s]["kept"])
                          for s in ("train", "validation")})


#: The anchor columns `with_anchor` adds; a frame that already has them is re-anchored.
ANCHOR_COLUMNS = ("anchor_hire_votes", "anchor_decided", "anchor_unstable", "anchor_changed",
                  "anchor_source")
#: V3's DPO set in order: V1's pairs (r2em on the vote anchors), then the new pairs' on-policy
#: part. The two were scored by different students: SFT v1 for r2em, SFT-v3 for the new part.
DPO_V3_PARTS = ("r2em_v3", "onpolicy_v3")


def stage_dpo_v3(cfg: dict) -> None:
    """V3's DPO set: r2em_v3, then the on-policy pairs from SFT-v3; CPU only.

    The on-policy pairs are SFT-v3's own covert, attribute-tipped mistakes on the new pairs
    (configs/generation/negatives_v3_onpolicy_*), merged with the k=32 top-up and balanced by
    error mass within each pool source (scripts/build_balanced_dpo.py --balance-by
    pool_source). r2em_v3 is V1's pairs (`--stage r2em`). docs/V3_NEW_DATA.md, round 2.

    Each row keeps `pool_source` and says which part it came from (`dpo_v3_part`). The
    error-mass build's pool label, which it wrote into `source`, becomes `r2em_pool`, as in
    r2em_v3. `student_margin` and `p_wrong` are SFT-v3's; r2em_v3's, from SFT v1, move to
    `sft_v1_margin` and `sft_v1_p_wrong`.

    Refused before anything is written: a row whose rejected keeps the chosen decision, an
    on-policy row whose chosen verdict is not its pair's vote anchor, a repeated prompt, a
    candidate on both sides of the split, and a benchmark id. Writes
    dpo_v3_{train,validation}.parquet, dpo_v3_report.json (contrasts per part and pool source,
    the error-mass build's balance, the GPU-free direction readout per pool source) and, for
    the D4 job, each pool source's training pairs as dpo_v3_checks/<pool source>/pairs.parquet.

    Round 3's on-policy set (docs/V3_NEW_DATA.md) is built the same way from a later round's
    config: `r2em: false` leaves r2em_v3 out, `anchors_from` names every directory whose
    anchors_en.parquet holds a pair's vote anchor, and `name` replaces the `dpo_v3` prefix of
    every file it writes.
    """
    require_output_root()
    spec = cfg["dpo_v3"]
    out = _out(cfg)
    name = str(spec.get("name", "dpo_v3"))
    parts = DPO_V3_PARTS if spec.get("r2em", True) else DPO_V3_PARTS[1:]
    onpolicy = Path(resolve_output_path(spec["onpolicy_dir"]))
    anchor_dirs = [Path(resolve_output_path(d)) for d in spec.get("anchors_from") or []] or [out]
    anchors = pd.concat([pd.read_parquet(d / "anchors_en.parquet") for d in anchor_dirs],
                        ignore_index=True)
    if anchors["pair_id"].duplicated().any():
        raise ValueError("dpo_v3: a pair has a vote anchor in two of "
                         f"{[str(d) for d in anchor_dirs]}")
    vote = anchors.set_index("pair_id")["reference_decision"]
    build = json.loads((onpolicy / f"{spec['onpolicy_name']}_report.json").read_text())
    report: dict = {"built": datetime.now(timezone.utc).isoformat(), "code": _code(),
                    "inputs": spec, "onpolicy_build": {
                        "parameters": build.get("parameters"),
                        "left_out": build["balance"].get("left_out", []),
                        "after_by_pool_source": {
                            s: g.get("after") for s, g in build["balance"].get("groups",
                                                                            {}).items()},
                        "validation_balance": build.get("validation_balance")}}
    frames = {}
    for split in ("train", "validation"):
        new = pd.read_parquet(onpolicy / f"{spec['onpolicy_name']}_{split}.parquet").rename(
            columns={"source": "r2em_pool"})
        anchored = new["pair_id"].map(vote)
        if anchored.isna().any():
            raise ValueError(f"dpo_v3 {split}: {int(anchored.isna().sum())} on-policy rows "
                             "have no vote anchor")
        against = anchored != new["decision"]
        if against.any():
            raise ValueError(f"dpo_v3 {split}: {int(against.sum())} on-policy rows choose "
                             "against their pair's vote anchor")
        new = new.drop(columns=[c for c in ANCHOR_COLUMNS if c in new.columns])
        new = provenance(with_anchor(new, anchors), new["pool_source"])
        pieces = [new.assign(dpo_v3_part=DPO_V3_PARTS[1])]
        if DPO_V3_PARTS[0] in parts:
            r2em = pd.read_parquet(out / f"r2em_v3_{split}.parquet").rename(
                columns={"student_margin": "sft_v1_margin", "p_wrong": "sft_v1_p_wrong"})
            pieces.insert(0, r2em.assign(dpo_v3_part=DPO_V3_PARTS[0]))
        both = pd.concat(pieces, ignore_index=True)
        same = both["decision"] == both["rejected_decision"]
        if same.any():
            raise ValueError(f"dpo_v3 {split}: {int(same.sum())} rows keep the chosen decision")
        repeated = both["prompt"].duplicated()
        if repeated.any():
            raise ValueError(f"dpo_v3 {split}: {int(repeated.sum())} prompts repeat")
        assert_no_leakage(both, f"v3_additional_data.dpo_v3.{split}")
        frames[split] = both
    shared = set(frames["train"]["candidate_id"]) & set(frames["validation"]["candidate_id"])
    if shared:
        raise ValueError(f"dpo_v3: {len(shared)} candidates are on both sides of the split")
    if set(frames["train"]["prompt"]) & set(frames["validation"]["prompt"]):
        raise ValueError("dpo_v3: a prompt is in both splits")

    seed = int(cfg["seed"])
    for split, frame in frames.items():
        _write_parquet(frame, out / f"{name}_{split}.parquet")
        report[split] = {
            "rows_by_part": frame["dpo_v3_part"].value_counts().to_dict(),
            "contrast": contrast(frame),
            "unstable_anchors_by_part": frame.groupby("dpo_v3_part")["anchor_unstable"]
            .sum().astype(int).to_dict(),
        }
    train = frames["train"]
    onpolicy_train = train[train["dpo_v3_part"] == DPO_V3_PARTS[1]]
    report["train"]["error_mass_sft_v3"] = {
        f"{s} {d}": round(float(g["p_wrong"].sum()), 2)
        for (s, d), g in onpolicy_train.groupby(["pool_source", "decision"])}
    report["train"]["direction"] = direction_readout(train, seed)
    for source, rows in train.groupby("pool_source", sort=True):
        _write_parquet(rows.reset_index(drop=True),
                       out / f"{name}_checks" / str(source) / "pairs.parquet")
    _write_json(out / f"{name}_report.json", report)
    log.info("%s: %s", name, {s: len(f) for s, f in frames.items()})


def stage_assemble(cfg: dict, allow_partial: bool = False) -> None:
    """SFT for training, the on-policy DPO build's input, and optionally teacher-biased DPO.

    DPO for training is not assembled here: it is r2em (`--stage r2em`) plus pairs drawn by
    `scripts/generate_onpolicy_full.py` with this directory as `source_artifacts_dir`, once the
    SFT model DPO starts from exists. This stage writes what that build reads:
    `raw/en_invariant.parquet` (v1's columns, new pairs only) and `sft_validation.parquet`,
    whose candidates it keeps in validation.
    """
    require_output_root()
    pool, done = _voted_parts(cfg)
    # A later round's pairs must be new to the student: none of its candidates may be in an
    # earlier round's SFT data, whichever part they came from.
    drawn = set(pool.loc[pool["part"].isin(done), "candidate_id"])
    for earlier in _earlier_rounds(cfg):
        for split in ("train", "validation"):
            path = earlier / f"sft_{split}.parquet"
            if path.exists():
                seen = drawn & set(pd.read_parquet(path, columns=["candidate_id"])["candidate_id"])
                if seen:
                    raise ValueError(f"assemble: {len(seen)} candidate(s) of this round are in "
                                     f"{path}, e.g. {sorted(seen)[:3]}")
    missing = [f"{p}/invariant" for p in done if not _paths(cfg, p, "invariant")[0].exists()]
    if missing and not allow_partial:
        raise SystemExit(f"assemble: voted parts without an invariant pass {missing}. Finish "
                         "them, or pass --allow-partial to assemble what is there.")
    out = _out(cfg)
    kept, reports = _filtered(cfg, done)
    if not kept["invariant"]:
        raise SystemExit("assemble: no invariant pass yet")
    invariant = pd.concat(kept["invariant"], ignore_index=True)
    biased = pd.concat(kept["biased"], ignore_index=True) if kept["biased"] else pd.DataFrame()
    anchors = pd.concat([part_anchors(cfg, p, pool[pool["part"] == p].reset_index(drop=True))
                         for p in done], ignore_index=True)
    excluded, exclusion_report = quality_exclusions(cfg, pool[pool["part"].isin(done)])
    invariant = invariant[~invariant["pair_id"].isin(excluded)]
    biased = biased[~biased["pair_id"].isin(excluded)] if len(biased) else biased

    # Each pool source is balanced on its own, as v1 balanced its one source.
    built = D.build_sft_dataset(invariant)
    balance, sources_balanced = {}, []
    for source, rows in built.groupby("source", sort=True):
        rows, balance[source] = F.balance_decisions(rows.reset_index(drop=True),
                                                    seed=int(cfg["seed"]), column="decision")
        sources_balanced.append(rows)
    sft = pd.concat(sources_balanced, ignore_index=True)
    sft = provenance(with_anchor(sft, anchors), sft["source"])
    assert_no_leakage(sft, "v3_additional_data.assemble.sft")
    if not sft["pair_id"].isin(set(pool["pair_id"])).all():
        raise ValueError("assemble: sft has a row outside the pool")
    val = validation_candidates(cfg, pool)
    base = Path(resolve_output_path(cfg["assemble"]["sft_v3_dir"]))
    splits: dict[str, dict] = {}
    is_val = sft["candidate_id"].isin(val)
    new = {"train": sft[~is_val].reset_index(drop=True),
           "validation": sft[is_val].reset_index(drop=True)}
    for split, rows in new.items():
        _write_parquet(rows, out / f"additional_sft_{split}.parquet")
        # SFT-v3's swap set, unchanged, with the new rows after it.
        v3 = pd.read_parquet(base / f"sft_{split}.parquet")
        if set(v3["candidate_id"]) & set(rows["candidate_id"]):
            raise ValueError(f"assemble: a new candidate is already in SFT-v3 {split}")
        v3 = provenance(v3.assign(reference_decision=v3["decision"]), "djinni_v1")
        combined = pd.concat([v3, rows], ignore_index=True)
        _write_parquet(combined, out / f"sft_{split}.parquet")
        splits[f"sft_{split}"] = {
            "rows": len(combined), "sft_v3_rows": len(v3), "new_rows": len(rows),
            "by_pool_source_and_decision": {
                f"{s} {d}": int(n) for (s, d), n in
                combined.groupby(["pool_source", "decision"]).size().items()}}

    # The on-policy build's input: every kept pair's raw invariant rows, in v1's schema.
    raw = pd.concat([pd.read_parquet(_paths(cfg, p, "invariant")[0]) for p in done
                     if _paths(cfg, p, "invariant")[0].exists()], ignore_index=True)
    raw = raw[~raw["pair_id"].isin(excluded)].reset_index(drop=True)
    raw = provenance(raw[[*V1_INVARIANT_COLUMNS, "source"]], raw["source"])
    _write_parquet(raw, out / "raw" / "en_invariant.parquet")
    splits["onpolicy_input"] = {"rows": len(raw), "pairs": int(raw["pair_id"].nunique()),
                                "by_pool_source": raw["pool_source"].value_counts().to_dict()}

    # Teacher-biased DPO: opt-in, named for what it is, and never a training view by default.
    if cfg["assemble"].get("teacher_biased_dpo"):
        dpo = D.build_dpo_dataset(invariant, biased)
        dpo = provenance(with_anchor(dpo, anchors), dpo["source"]) if len(dpo) else dpo
        cap = int(cfg["assemble"]["dpo_cap_per_pair"])
        is_val = dpo["candidate_id"].isin(val)
        for split, rows in (("train", dpo[~is_val]), ("validation", dpo[is_val])):
            v1_path = out / f"teacher_biased_v1_dpo_{split}.parquet"
            frames = [pd.read_parquet(v1_path)] if v1_path.exists() else []
            both = pd.concat([*frames, rows], ignore_index=True)
            views = {"teacher_biased_dpo": both,
                     "teacher_biased_dpo_capped": capped(both, cap, int(cfg["seed"]))}
            views["teacher_biased_dpo_capped_balanced"] = balanced(
                views["teacher_biased_dpo_capped"], int(cfg["seed"]))
            for name, frame in views.items():
                _write_parquet(frame, out / f"{name}_{split}.parquet")
            splits[f"teacher_biased_dpo_{split}"] = {
                name: contrast(frame) for name, frame in views.items()}
            if split == "train":
                splits["teacher_biased_direction_checks_train"] = {
                    name: direction_readout(frame, int(cfg["seed"]))
                    for name, frame in views.items() if len(frame)}
    r2em = {s: (out / f"r2em_v3_{s}.parquet").exists() for s in ("train", "validation")}
    _write_json(out / "generation_report.json", {
        "built": datetime.now(timezone.utc).isoformat(), "code": _code(),
        "teacher_model": cfg["teacher_model"], "parts": done, "sft_v3_dir": str(base),
        "balance_new_sft_rows_by_source": balance, "quality_exclusions": exclusion_report,
        "splits": splits, "filters": reports, "validation_candidates": len(val),
        "r2em_v3_built": r2em,
        "parts_without_biased_pass": [p for p in done
                                      if not _paths(cfg, p, "biased")[0].exists()]})
    log.info("assemble: %s", {k: v for k, v in splits.items() if not k.startswith("teacher")})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True,
                        choices=["pool", "generate", "report", "v1_dpo", "r2em", "assemble",
                                 "dpo_v3"])
    parser.add_argument("--parts", nargs="+", default=["all"])
    parser.add_argument("--passes", nargs="+", default=list(PASSES), choices=PASSES)
    parser.add_argument("--allow-partial", action="store_true",
                        help="assemble even when a voted part lacks its invariant pass")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_path"] = args.config
    {"pool": lambda: stage_pool(cfg),
     "generate": lambda: stage_generate(cfg, args.parts, args.passes),
     "report": lambda: stage_report(cfg),
     "v1_dpo": lambda: stage_v1_dpo(cfg),
     "r2em": lambda: stage_r2em(cfg),
     "assemble": lambda: stage_assemble(cfg, args.allow_partial),
     "dpo_v3": lambda: stage_dpo_v3(cfg)}[args.stage]()


if __name__ == "__main__":
    main()
