#!/usr/bin/env python
"""Per-stage GPU and wall-clock cost of the project's Slurm jobs, for the final analysis.

    python scripts/cost_report.py                     # sacct + step records + job logs
    python scripts/cost_report.py --since 2026-10-13
    python scripts/cost_report.py --no-sacct          # rebuild from reports/cost/jobs.csv

Three sources, the most precise first:

1. Step records ($HBM_COST_DIR, from utils/cost.py; `cost <stage> <label> <cmd>` in
   scripts/cscs/env.sh). One per step: the GPUs its processes actually ran on, wall time,
   NVML energy, generated tokens, trained parameters.
2. Slurm accounting for every job whose name starts with --prefix: what the job held (a
   whole node, four GH200, whatever its steps used) and the node's energy.
3. For a job with no records -- it ran before they existed -- its log's step markers
   (`=== <time> <text>`, env.sh::step), timed from marker to marker. A log does not say how
   many GPUs a step used, so such a step is charged the job's whole allocation and marked
   `gpu_basis=allocated`.

GPU-hours on three bases, never mixed in one column:

    measured       a recorded step: wall x the GPUs its processes ran on. Methods are compared
                   on this.
    reconstructed  a step rebuilt from a log: wall x the job's allocated GPUs, an upper bound
                   on use (CSCS bills whole nodes, so it is what the step held).
    allocated      a job: elapsed x allocated GPUs, the budget spent. For a job with records,
                   allocated minus measured is the `idle` row: setup, CPU-only work, the GPUs
                   a one-GPU step left free.

A step's stage comes from its record, else from its job's name (`JOB_STAGE`, default
`datagen`), with setup and housekeeping steps (`HOUSEKEEPING`) split off as `other`. Its arm
comes from the record, else from the first matching rule in reports/cost/labels.csv, which is
written by hand: `pattern` is a regex searched in "job name | job id | config | label |
command", and a non-empty `arm`, `stage` or `recipe` (yes/no) cell sets that field. Data
generation shared between arms keeps an empty arm until a rule assigns it.

Writes reports/cost/: steps.csv (one row per step), jobs.csv (one per Slurm job), by_stage.csv,
by_arm.csv, checkpoints.csv (GPU-hours to each saved checkpoint of a recorded training run),
throughput.csv (generation steps: requests, tokens, GPU-seconds per thousand requests),
params.csv (trained against base parameters, per adapter under outputs/) and COST.md. All of
them are regenerated; only labels.csv is edited.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime
from math import prod
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hiring_bias_mitigation.utils import cost  # noqa: E402

STAGE_ORDER = ("datagen", "train", "eval", "patching", "other")
#: Jobs with no records: the stage their steps are charged to. Every other job of the project
#: generated or checked training data (DEFAULT_STAGE).
JOB_STAGE = {
    "xw-sft": "train", "xw-dpo": "train", "xw-grpo": "train", "xw-dapo": "train",
    "xw-audit": "eval", "xw-opc": "eval", "xw-eval-arm": "eval", "xw-dpo-eval": "eval",
    "xw-grpo-signal": "eval", "xw-merge-retention": "eval", "xw-monitor-cal": "eval",
    "xw-env": "other", "xw-download": "other", "xw-seeds-tests": "other",
    "xw-seeds-tests2": "other",
}
DEFAULT_STAGE = "datagen"
#: Log steps that are setup or housekeeping whatever the job: W&B uploads, environment and
#: download checklists ("0/4 network ..."), test runs.
HOUSEKEEPING = re.compile(r"W&B|wandb|^\d+/\d+ |unit tests|^tests$|ruff|imports resolve|"
                          r"^(red|green):")
MARKER = re.compile(r"^=== (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
CONFIG_IN_TEXT = re.compile(r"configs/\S+?\.yaml")
SACCT_FIELDS = ("JobID", "JobName", "Partition", "State", "Start", "End", "ElapsedRaw",
                "NNodes", "AllocTRES", "ConsumedEnergyRaw", "WorkDir")
#: sacct's energy field holds garbage near 2**64 when the node's counter failed.
MAX_SANE_ENERGY_J = 1e12
STALE_SECONDS = 3 * cost.HEARTBEAT_SECONDS
#: An arm's `recipe_gpu_h`: what producing it once takes -- its data and its training. A step
#: counts if it ended normally or was cut by the time limit or a signal (training and
#: generation resume from what it saved). Failed, cancelled, killed and still-running steps do
#: not, unless a labels.csv rule says `recipe=yes` (a cancelled job whose finished passes a
#: later job reused) or `recipe=no`. Checks of the data (direction checks, D4, a redraw
#: control) are generation-stage work but not part of the recipe.
RECIPE_STAGES = ("datagen", "train")
RECIPE_STATUS = ("ok", "timeout", "terminated")
QA_LABEL = re.compile(r"direction[ -]check|\bD4\b|d4-check|redraw control|^check:", re.I)
LIVE_STATES = {"RUNNING", "PENDING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING"}

STEP_COLUMNS = (
    "source", "job_id", "job_name", "start", "end", "wall_s", "stage", "label", "arm",
    "config", "status", "n_gpus", "gpu_basis", "gpu_h", "gpu_model", "gpu_energy_kwh",
    "util_mean_pct", "mem_peak_gib", "requests", "completions", "prompt_tokens",
    "output_tokens", "load_s", "generate_s", "params_trainable", "params_total",
    "params_modified", "train_steps", "global_step", "train_runtime_s", "train_flos",
    "train_rows", "train_tokens_per_epoch", "effective_batch", "world_size", "precision",
    "torch_peak_gib", "checkpoint_times", "run_name", "n_prompts", "served_model",
    "prefix_cache_queries", "prefix_cache_hits", "recipe",
    "host", "commit", "dirty", "software", "gpu_driver", "command", "record",
)
JOB_COLUMNS = (
    "job_id", "job_name", "partition", "state", "start", "end", "elapsed_h", "nodes", "gpus",
    "node_h", "gpu_h_alloc", "node_energy_kwh", "workdir", "source", "gpu_h_steps",
    "gpu_h_idle",
)
CHECKPOINT_COLUMNS = (
    "output_dir", "arm", "checkpoint_step", "saved_at", "job_id", "n_gpus",
    "gpu_h_segment_to_checkpoint", "gpu_h_earlier_segments", "gpu_h_to_checkpoint",
)


# --------------------------------------------------------------------------- sources


def _naive_local(value: str | None) -> datetime | None:
    """sacct's and the logs' local wall-clock times, and the records' ISO times converted to
    the same local zone, so all three compare directly."""
    if not value or value in {"Unknown", "None"}:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is not None:
        moment = moment.astimezone().replace(tzinfo=None)
    return moment


def _iso(moment: datetime | None) -> str:
    return moment.isoformat(sep=" ", timespec="seconds") if moment else ""


def parse_sacct_row(row: dict) -> dict:
    tres = dict(item.split("=", 1) for item in row["AllocTRES"].split(",") if "=" in item)
    gpus = int(tres.get("gres/gpu") or 0)
    nodes = int(row["NNodes"] or 1)
    elapsed = int(row["ElapsedRaw"] or 0)
    energy = row.get("ConsumedEnergyRaw") or ""
    energy_j = float(energy) if energy.isdigit() and float(energy) < MAX_SANE_ENERGY_J else None
    return {
        "job_id": row["JobID"],
        "job_name": row["JobName"],
        "partition": row["Partition"],
        "state": (row["State"].split() or ["?"])[0],
        "start": _iso(_naive_local(row["Start"])),
        "end": _iso(_naive_local(row["End"])),
        "elapsed_h": round(elapsed / 3600, 4),
        "nodes": nodes,
        "gpus": gpus,
        "node_h": round(elapsed * nodes / 3600, 4),
        "gpu_h_alloc": round(elapsed * gpus / 3600, 4),
        "node_energy_kwh": round(energy_j / 3.6e6, 3) if energy_j else None,
        "workdir": row.get("WorkDir", ""),
    }


def fetch_sacct(since: str, prefix: str) -> list[dict]:
    env = {**os.environ, "SLURM_TIME_FORMAT": "standard"}
    command = ["sacct", "-X", "--parsable2", "--noheader", "-S", since, "-E", "now",
               "--format=" + ",".join(SACCT_FIELDS)]
    if os.environ.get("USER"):
        command += ["-u", os.environ["USER"]]
    out = subprocess.run(command, capture_output=True, text=True, env=env, timeout=300,
                         check=True).stdout
    jobs = []
    for line in out.splitlines():
        row = dict(zip(SACCT_FIELDS, line.split("|"), strict=False))
        if row.get("JobName", "").startswith(prefix):
            jobs.append(parse_sacct_row(row))
    return jobs


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _number(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def cached_jobs(path: Path) -> list[dict]:
    jobs = read_csv(path)
    for job in jobs:
        for key in ("elapsed_h", "node_h", "gpu_h_alloc", "node_energy_kwh"):
            job[key] = _number(job.get(key))
        for key in ("nodes", "gpus"):
            job[key] = int(_number(job.get(key)) or 0)
    return jobs


def load_records(folder: Path) -> list[dict]:
    records = []
    for path in sorted(folder.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and data.get("schema"):
            data["_file"] = path.name
            records.append(data)
    return records


def load_labels(path: Path) -> list[tuple[re.Pattern, str, str, str]]:
    """(pattern, arm, stage, recipe) per rule; an empty cell leaves that field alone."""
    rules = []
    for row in read_csv(path):
        pattern = (row.get("pattern") or "").strip()
        if pattern and not pattern.startswith("#"):
            rules.append((re.compile(pattern), (row.get("arm") or "").strip(),
                          (row.get("stage") or "").strip(),
                          (row.get("recipe") or "").strip().lower()))
    return rules


# --------------------------------------------------------------------------- steps


def _still_running(record: dict, job: dict | None, now: datetime) -> bool:
    if job is not None:
        return job["state"] in LIVE_STATES
    last = _naive_local(record.get("end"))
    return last is not None and (now - last).total_seconds() < STALE_SECONDS


def record_step(record: dict, jobs: dict[str, dict], now: datetime) -> dict:
    gpu = record.get("gpu") or {}
    notes = record.get("notes") or {}
    counters = notes.get("counters") or {}
    facts = notes.get("facts") or {}
    phases = notes.get("phases") or {}
    slurm = record.get("slurm") or {}
    code = record.get("code") or {}
    job = jobs.get(str(slurm.get("job_id", "")))
    status = record.get("status")
    if status == "running" and not _still_running(record, job, now):
        status = "killed"  # its last heartbeat is when it was last seen alive
    wall = float(record.get("wall_s") or 0.0)
    count, gpu_h = gpu.get("count"), record.get("gpu_h")
    # Only NVML's process list measures use; a count of visible or declared devices is what
    # the step could use, and stays apart from it.
    basis = "used" if gpu.get("basis") == "nvml-processes" else (gpu.get("basis") or "used")
    if count is None:
        count = job["gpus"] if job else None
        gpu_h = round(wall * count / 3600, 4) if count is not None else None
        basis = "allocated" if job else "unknown"
    energy = gpu.get("energy_j")
    software = record.get("software") or {}
    driver = " / CUDA ".join(v for v in (gpu.get("driver"), gpu.get("cuda_driver")) if v)
    return {
        "source": "record",
        "job_id": str(slurm.get("job_id", "")),
        "job_name": slurm.get("job_name", ""),
        "start": _iso(_naive_local(record.get("start"))),
        "end": _iso(_naive_local(record.get("end"))),
        "wall_s": round(wall, 1),
        "stage": record.get("stage", ""),
        "label": record.get("label", ""),
        "arm": record.get("arm") or "",
        "config": record.get("config") or "",
        "status": status,
        "n_gpus": count,
        "gpu_basis": basis,
        "gpu_h": gpu_h,
        "gpu_model": gpu.get("model", ""),
        "gpu_energy_kwh": round(energy / 3.6e6, 4) if energy else None,
        "util_mean_pct": gpu.get("util_mean_pct"),
        "mem_peak_gib": gpu.get("mem_peak_gib"),
        "requests": counters.get("requests"),
        "completions": counters.get("completions"),
        "prompt_tokens": counters.get("prompt_tokens"),
        "output_tokens": counters.get("output_tokens"),
        "load_s": phases.get("load"),
        "generate_s": phases.get("generate"),
        "params_trainable": facts.get("params_trainable"),
        "params_total": facts.get("params_total"),
        "params_modified": facts.get("params_modified"),
        "train_steps": facts.get("train_steps"),
        "global_step": facts.get("global_step"),
        "train_runtime_s": facts.get("train_runtime_s"),
        "train_flos": facts.get("train_flos"),
        "train_rows": facts.get("train_rows"),
        "train_tokens_per_epoch": facts.get("train_tokens_per_epoch"),
        "effective_batch": facts.get("effective_batch"),
        "world_size": facts.get("world_size"),
        "precision": facts.get("precision"),
        "torch_peak_gib": facts.get("torch_peak_gib"),
        "checkpoint_times": json.dumps(facts["checkpoint_times"], sort_keys=True)
        if facts.get("checkpoint_times") else None,
        "run_name": facts.get("run_name") or facts.get("output_dir") or "",
        "n_prompts": facts.get("n_prompts"),
        "served_model": facts.get("served_model"),
        "prefix_cache_queries": facts.get("prefix_cache_queries"),
        "prefix_cache_hits": facts.get("prefix_cache_hits"),
        "host": record.get("host", ""),
        "commit": code.get("commit", ""),
        "dirty": code.get("dirty"),
        "software": ", ".join(f"{k} {v}" for k, v in software.items()),
        "gpu_driver": driver,
        "command": record.get("command") or "",
        "record": record.get("_file", ""),
        "_output_dir": facts.get("output_dir") or "",
    }


def _blank_step() -> dict:
    return dict.fromkeys(STEP_COLUMNS)


def log_steps(job: dict, logs_dir: Path, now: datetime) -> list[dict]:
    """A job's steps from its log's `step` markers, each charged the job's allocation."""
    path = logs_dir / f"{job['job_name']}-{job['job_id']}.out"
    start = _naive_local(job["start"].replace(" ", "T")) if job["start"] else None
    end = _naive_local(job["end"].replace(" ", "T")) if job["end"] else None
    end = end or now
    markers: list[tuple[datetime, str]] = []
    header_config = commit = ""
    if path.exists():
        with path.open(encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f):
                if n < 5 and line.startswith("host "):
                    if m := re.search(r"\bconfig (\S+)", line):
                        header_config = m.group(1)
                    if m := re.search(r"\bcommit (\S+)", line):
                        commit = m.group(1)
                if m := MARKER.match(line.rstrip("\n")):
                    markers.append((datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S"),
                                    m.group(2).strip()))
    job_stage = JOB_STAGE.get(job["job_name"], DEFAULT_STAGE)
    spans: list[tuple[datetime | None, datetime, str, str]] = []
    if not markers:
        spans.append((start, end, "(whole job; no step markers)" if path.exists()
                      else "(whole job; no log)", job_stage))
    else:
        if start and markers[0][0] > start:
            spans.append((start, markers[0][0], "setup", "other"))
        for i, (moment, text) in enumerate(markers):
            stop = markers[i + 1][0] if i + 1 < len(markers) else end
            stage = "other" if HOUSEKEEPING.search(text) else job_stage
            spans.append((moment, stop, text, stage))
    steps = []
    for begin, stop, text, stage in spans:
        wall = max(0.0, (stop - begin).total_seconds()) if begin else job["elapsed_h"] * 3600
        config = CONFIG_IN_TEXT.search(text)
        step = _blank_step()
        step.update(
            source="log" if path.exists() else "sacct",
            job_id=job["job_id"], job_name=job["job_name"],
            start=_iso(begin), end=_iso(stop), wall_s=round(wall, 1),
            stage=stage, label=text[:200], _text=text, arm="",
            config=config.group(0) if config else header_config,
            status="ok" if job["state"] == "COMPLETED" else job["state"].lower(),
            n_gpus=job["gpus"], gpu_basis="allocated",
            gpu_h=round(wall * job["gpus"] / 3600, 4), commit=commit,
            record=path.name if path.exists() else "",
        )
        steps.append(step)
    return steps


def _first_arm(text: str, rules: list[tuple]) -> str:
    return next((arm for pattern, arm, *_ in rules if arm and pattern.search(text)), "")


def _text(step: dict) -> str:
    """A log step's whole marker; the stored `label` is cut to 200 characters."""
    return step.get("_text") or step.get("label") or ""


def apply_labels(steps: list[dict], rules: list[tuple]) -> None:
    """Fills each step's empty arm, and sets its stage and recipe override, from the first
    matching rule.

    The key is "job name | job id | config | label | command", with a log step's whole
    marker as the label. A step that names several configs (one margin pass over four arms,
    a merge check of several adapters) belongs to all their arms, joined with "+"."""
    for step in steps:
        if not step["arm"]:
            named = CONFIG_IN_TEXT.findall(_text(step))
            arms = list(dict.fromkeys(a for c in named if (a := _first_arm(c, rules))))
            if len(arms) > 1:
                step["arm"] = "+".join(arms)
        key = " | ".join(str(step.get(k) or "") for k in ("job_name", "job_id", "config"))
        key = f"{key} | {_text(step)} | {step.get('command') or ''}"
        for pattern, arm, stage, recipe in rules:
            if pattern.search(key):
                if arm and not step["arm"]:
                    step["arm"] = arm
                if stage:
                    step["stage"] = stage
                if recipe in {"yes", "no"}:
                    step["_recipe"] = recipe == "yes"
                break


def mark_recipe(steps: list[dict]) -> None:
    """Whether each step is part of its arm's recipe (see RECIPE_STATUS)."""
    for step in steps:
        if "_recipe" in step:
            step["recipe"] = step["_recipe"]
        else:
            step["recipe"] = (step["stage"] in RECIPE_STAGES
                              and step["status"] in RECIPE_STATUS
                              and not QA_LABEL.search(_text(step)))


# --------------------------------------------------------------------------- tables


def _sum(rows: list[dict], key: str) -> float:
    return sum(float(r[key]) for r in rows if r.get(key) not in (None, ""))


def _sum_or_none(rows: list[dict], key: str, digits: int = 3) -> float | None:
    """None, shown as blank, when no row has the value: not measured is not zero."""
    values = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return round(sum(values), digits) if values else None


def _stages(steps: list[dict]) -> list[str]:
    seen = {s["stage"] for s in steps}
    return [s for s in STAGE_ORDER if s in seen] + sorted(seen - set(STAGE_ORDER))


def job_table(jobs: list[dict], steps: list[dict], recorded: set[str]) -> list[dict]:
    """One row per job. Idle time is known only for a job with step records: a job rebuilt
    from its log has its steps charged the whole allocation, idle time included."""
    rows = []
    for job in jobs:
        mine = [s for s in steps if s["job_id"] == job["job_id"]]
        used = round(_sum(mine, "gpu_h"), 4)
        alloc = job["gpu_h_alloc"] or 0.0
        has_records = job["job_id"] in recorded
        rows.append({
            **job,
            "source": "records" if has_records else (mine[0]["source"] if mine else "none"),
            "gpu_h_steps": used,
            "gpu_h_idle": round(max(0.0, alloc - used), 4) if has_records else None,
        })
    return rows


def stage_table(steps: list[dict], jobs: list[dict]) -> list[dict]:
    rows = []
    for stage in _stages(steps):
        mine = [s for s in steps if s["stage"] == stage]
        rows.append({
            "stage": stage,
            "steps": len(mine),
            "wall_h": round(_sum(mine, "wall_s") / 3600, 3),
            "gpu_h": round(_sum(mine, "gpu_h"), 3),
            "gpu_h_measured": _sum_or_none([s for s in mine if s["gpu_basis"] == "used"],
                                           "gpu_h"),
            "gpu_h_reconstructed": _sum_or_none(
                [s for s in mine if s["gpu_basis"] == "allocated"], "gpu_h"),
            "gpu_h_other_basis": _sum_or_none(
                [s for s in mine if s["gpu_basis"] not in {"used", "allocated"}], "gpu_h"),
            "gpu_h_not_ok": round(_sum([s for s in mine if s["status"] not in
                                        {"ok", "running"}], "gpu_h"), 3),
            "gpu_energy_kwh": _sum_or_none(mine, "gpu_energy_kwh"),
        })
    recorded = [j for j in jobs if j.get("gpu_h_idle") is not None]
    rows.append({"stage": f"idle ({len(recorded)} recorded jobs)", "steps": "", "wall_h": "",
                 "gpu_h": _sum_or_none(recorded, "gpu_h_idle")})
    with_energy = [j for j in jobs if j.get("node_energy_kwh") not in (None, "")]
    rows.append({"stage": "total allocated", "steps": len(jobs), "wall_h": None,
                 "gpu_h": round(_sum(jobs, "gpu_h_alloc"), 3),
                 "node_h": round(_sum(jobs, "node_h"), 3),
                 "node_energy_kwh": round(_sum(with_energy, "node_energy_kwh"), 1),
                 "node_h_with_energy": round(_sum(with_energy, "node_h"), 3)})
    return rows


def arm_table(steps: list[dict]) -> list[dict]:
    """Per arm, GPU-hours by stage. A step shared with other arms ("a+b" in steps.csv) counts
    in full for each of them, so a row is what reproducing that arm from scratch costs, and
    `shared_gpu_h` says how much of it other arms also used. Summing a column over arms
    counts the shared steps more than once; by_stage.csv has the project's totals."""
    stages = _stages(steps)
    arms = sorted({a for s in steps for a in s["arm"].split("+") if a})
    rows = []
    for arm in [*arms, ""]:
        mine = [s for s in steps if s["arm"] == arm
                or (arm and arm in s["arm"].split("+"))]
        row = {"arm": arm or "(unassigned)"}
        for stage in stages:
            row[f"{stage}_gpu_h"] = round(_sum([s for s in mine if s["stage"] == stage],
                                               "gpu_h"), 3)
        row["shared_gpu_h"] = round(_sum([s for s in mine if "+" in s["arm"]], "gpu_h"), 3)
        row["total_gpu_h"] = round(_sum(mine, "gpu_h"), 3)
        made = [s for s in mine if s["stage"] in RECIPE_STAGES]
        # An arm with no data or training step here (the published SFT, a reference audit)
        # has no recipe in this report: not covered, not zero.
        row["recipe_gpu_h"] = (round(_sum([s for s in made if s["recipe"]], "gpu_h"), 3)
                               if made and arm else None)
        row["excluded_gpu_h"] = (round(_sum([s for s in made if not s["recipe"]], "gpu_h"), 3)
                                 if made and arm else None)
        row["wall_h"] = round(_sum(mine, "wall_s") / 3600, 3)
        row["steps"] = len(mine)
        rows.append(row)
    return rows


def checkpoint_table(steps: list[dict]) -> list[dict]:
    """GPU-hours up to each saved checkpoint of a recorded training run: the segment's time
    from its start to the save, times its GPUs, plus every earlier recorded segment of the same
    output directory. A run started before the records existed has no row here."""
    segments: dict[str, list[dict]] = {}
    for s in steps:
        if s["source"] == "record" and s["stage"] == "train" and s.get("_output_dir"):
            segments.setdefault(s["_output_dir"], []).append(s)
    rows = []
    for output_dir, mine in segments.items():
        mine.sort(key=lambda s: s["start"])
        earlier = 0.0
        for s in mine:
            start = _naive_local(s["start"])
            for step, saved in sorted(json.loads(s["checkpoint_times"] or "{}").items(),
                                      key=lambda kv: int(kv[0])):
                at = _naive_local(saved)
                if start is None or at is None or s["n_gpus"] in (None, ""):
                    continue
                this = (at - start).total_seconds() * float(s["n_gpus"]) / 3600
                rows.append({
                    "output_dir": output_dir, "arm": s["arm"], "checkpoint_step": int(step),
                    "saved_at": _iso(at), "job_id": s["job_id"], "n_gpus": s["n_gpus"],
                    "gpu_h_segment_to_checkpoint": round(this, 4),
                    "gpu_h_earlier_segments": round(earlier, 4),
                    "gpu_h_to_checkpoint": round(earlier + this, 4),
                })
            earlier += float(s["gpu_h"] or 0)
    return rows


def throughput_table(steps: list[dict]) -> list[dict]:
    rows = []
    for s in steps:
        requests = _number(s.get("requests"))
        if not requests:
            continue
        wall, gen = float(s["wall_s"] or 0), _number(s.get("generate_s"))
        gpus = _number(s.get("n_gpus")) or 0
        out_tokens = _number(s.get("output_tokens"))
        # vLLM's counters, in prompt tokens; only an engine with stat logging on keeps them.
        queries, hits = _number(s.get("prefix_cache_queries")), _number(s.get("prefix_cache_hits"))
        rows.append({
            "job_id": s["job_id"], "stage": s["stage"], "label": s["label"], "arm": s["arm"],
            "config": s["config"], "run_name": s["run_name"], "gpu_model": s["gpu_model"],
            "n_gpus": s["n_gpus"], "requests": int(requests), "completions": s["completions"],
            "prompt_tokens": s["prompt_tokens"], "output_tokens": s["output_tokens"],
            "wall_s": wall, "load_s": s["load_s"], "generate_s": gen,
            "requests_per_s": round(requests / gen, 2) if gen else None,
            "output_tokens_per_s": round(out_tokens / gen, 1) if gen and out_tokens else None,
            "gpu_s_per_1k_requests": round(gpus * wall / requests * 1000, 1),
            "gpu_s_per_1k_requests_generate_only":
                round(gpus * gen / requests * 1000, 1) if gen else None,
            "prefix_cache_queries": s.get("prefix_cache_queries"),
            "prefix_cache_hits": s.get("prefix_cache_hits"),
            "prefix_cache_hit_rate":
                round(hits / queries, 4) if queries and hits is not None else None,
        })
    return rows


# --------------------------------------------------------------------------- parameters


def safetensors_params(path: Path) -> tuple[int, int, int]:
    """(parameters, of which vision, of which multi-token-prediction head) from a safetensors
    header, without loading tensors. Qwen3.5 checkpoints carry an `mtp.*` head that LoRA does
    not train and vLLM does not run here (no speculative decoding)."""
    with path.open("rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(size))
    total = vision = mtp = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        n = prod(meta["shape"])
        total += n
        if "visual" in key or "vision" in key:
            vision += n
        elif key.startswith("mtp.") or ".mtp." in key:
            mtp += n
    return total, vision, mtp


def base_params(base: str | None, hf_home: Path, depth: int = 0) -> tuple:
    """(total, vision, mtp) parameters of the model an adapter was trained on, following
    adapter chains (a DPO adapter on an SFT adapter) down to the full checkpoint."""
    if not base or depth > 4:
        return None, None, None
    folder = Path(base)
    if not folder.is_dir():
        org_name = base.replace("/", "--")
        snapshots = sorted((hf_home / "hub" / f"models--{org_name}" / "snapshots").glob("*"),
                           key=lambda p: p.stat().st_mtime)
        if not snapshots:
            return None, None, None
        folder = snapshots[-1]
    adapter_config = folder / "adapter_config.json"
    if adapter_config.exists():
        parent = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
        return base_params(parent, hf_home, depth + 1)
    shards = [p for p in folder.glob("*.safetensors") if not p.name.startswith("adapter_")]
    if not shards:
        return None, None, None
    total = vision = mtp = 0
    for shard in shards:
        t, v, m = safetensors_params(shard)
        total, vision, mtp = total + t, vision + v, mtp + m
    return total, vision, mtp


def params_table(outputs_root: Path, hf_home: Path) -> list[dict]:
    rows = []
    for family_dir in sorted((outputs_root / "outputs").glob("*")):
        for run in sorted(family_dir.glob("*")):
            adapter = run / "adapter_model.safetensors"
            config_path = run / "adapter_config.json"
            if not (adapter.exists() and config_path.exists()):
                continue
            config = json.loads(config_path.read_text())
            trained, _, _ = safetensors_params(adapter)
            base = config.get("base_model_name_or_path")
            total, vision, mtp = base_params(base, hf_home)
            text = total - vision - mtp if total is not None else None
            rows.append({
                "run": run.name, "family": family_dir.name, "base_model": base,
                "lora_r": config.get("r"), "lora_alpha": config.get("lora_alpha"),
                "target_modules": "+".join(sorted(config.get("target_modules") or [])),
                "params_trainable": trained, "params_base": total, "params_base_vision": vision,
                "params_base_mtp": mtp, "params_base_text": text,
                "trainable_pct_of_text": round(100 * trained / text, 4) if text else None,
            })
    return rows


# --------------------------------------------------------------------------- output


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(columns or dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in fields})


def _md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "(none)\n"
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join("" if row.get(c) is None else str(row.get(c))
                                       for c in columns) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(steps, jobs, stages, arms, checkpoints, throughput, params, since,
                    prefix) -> str:
    measured = sum(1 for s in steps if s["gpu_basis"] == "used")
    rebuilt = sorted({s["job_id"] for s in steps if s["source"] in {"log", "sacct"}})
    no_log = sorted({s["job_id"] for s in steps if s["source"] == "sacct"})
    live = sorted({j["job_id"] for j in jobs if j["state"] in LIVE_STATES})
    unknown = [s for s in steps if s["gpu_basis"] == "unknown"]
    killed = [s for s in steps if s["status"] == "killed"]
    no_energy = [j for j in jobs if j.get("node_energy_kwh") in (None, "")]
    unassigned = round(_sum([s for s in steps if not s["arm"] and s["stage"] != "other"],
                            "gpu_h"), 3)
    shared_once = round(_sum([s for s in steps if "+" in s["arm"]], "gpu_h"), 3)
    stage_cols = ["stage", "steps", "wall_h", "gpu_h", "gpu_h_measured", "gpu_h_reconstructed",
                  "gpu_h_not_ok", "gpu_energy_kwh"]
    arm_cols = ["arm", *[c for c in (arms[0].keys() if arms else []) if c.endswith("_gpu_h")],
                "wall_h", "steps"]
    total = next((r for r in stages if r["stage"] == "total allocated"), {})
    energy_note = (f"{total.get('node_energy_kwh', 0)} kWh of node energy by Slurm"
                   + (f", a lower bound: {len(no_energy)} jobs report none"
                      if no_energy else ""))
    parts = [
        "# GPU and time cost\n",
        f"Generated by `scripts/cost_report.py` on {datetime.now():%Y-%m-%d %H:%M}, from Slurm "
        f"jobs named `{prefix}*` since {since}: {len(jobs)} jobs; {measured} steps measured by "
        f"step records, {len(rebuilt)} jobs reconstructed from their logs and sacct. Every "
        "number here is in `reports/cost/*.csv` with its job ID; do not edit this file, rerun "
        "the script.\n",
        f"Allocated: **{total.get('gpu_h', 0)} GPU-h** ({total.get('node_h', 0)} node-h summed "
        f"over the jobs, not elapsed time; {energy_note}).\n",
        "## Definitions\n",
        "- `gpu_h_measured`: wall time x the GPUs a step's processes ran on, from a step "
        "record (NVML's process list). The number to compare methods on.\n"
        "- `gpu_h_reconstructed`: wall time x the job's allocated GPUs, for a step rebuilt from "
        "a log. CSCS bills whole nodes, so it is what the step held, and an upper bound on what "
        "it used. Quote it as \"allocated GPU-h (reconstructed from Slurm and job logs)\", "
        "never in one column with measured use.\n"
        "- `idle`: allocated minus used, for jobs with step records only. A reconstructed "
        "job's idle time is inside its steps' `gpu_h_reconstructed`.\n"
        "- `gpu_h_not_ok`: steps that failed, hit the time limit, were cancelled or killed. A "
        "training or generation job resumes from its checkpoints and cached chunks, so this "
        "bounds the waste from above.\n"
        "- `gpu_energy_kwh`: NVML energy of the GPUs a recorded step used; blank where nothing "
        "was measured. `node_energy_kwh` (jobs.csv) is Slurm's whole-node figure.\n"
        "- `recipe_gpu_h` (by arm): what producing the arm once takes -- its data generation and "
        "training, shared steps in full; failed, cancelled and killed attempts and checks of "
        "the data left out, segments cut by the time limit kept (`recipe` in steps.csv says "
        "which steps count). Blank: the arm's data and training are not in these jobs.\n",
        "## By stage\n",
        _md_table(stages, stage_cols),
        "## By arm\n",
        "Rows overlap and must not be added up: a step shared by several arms (`a+b` in "
        "steps.csv) counts in full in each, and `shared_gpu_h` is the part of a row that other "
        "arms also count (included in `total_gpu_h`, not added to it). Only `train_gpu_h` adds "
        f"up across arms. Shared steps counted once: {shared_once} GPU-h. The project's total "
        "is the `total allocated` row above. Arms come from step records (`HBM_COST_ARM`, "
        "`--arm`) and from the rules in `reports/cost/labels.csv`; unassigned "
        f"non-housekeeping GPU-hours: {unassigned}.\n",
        _md_table(arms, arm_cols),
        "## Cost of each saved checkpoint\n",
        "From recorded training runs only: the GPU-hours from the start of the run's first "
        "recorded segment to the save of each checkpoint. Pair an audited checkpoint with this, "
        "not with its arm's total.\n",
        _md_table(checkpoints, ["output_dir", "checkpoint_step", "job_id", "n_gpus",
                                "gpu_h_to_checkpoint"]),
        "## Throughput of generation steps\n",
        _md_table(throughput, ["job_id", "stage", "label", "config", "gpu_model", "n_gpus",
                               "requests", "output_tokens", "generate_s",
                               "requests_per_s", "gpu_s_per_1k_requests"]),
        "## Trained parameters\n",
        "`params_base_text` leaves out the vision tower and the multi-token-prediction head, "
        "which the adapters do not train and the audits do not run.\n",
        _md_table(params, ["run", "family", "lora_r", "params_trainable", "params_base_text",
                           "trainable_pct_of_text"]),
        "## Gaps\n",
        f"- Jobs still running or pending, counted only up to now and kept out of "
        f"`gpu_h_not_ok` and the recipes: {', '.join(live) or 'none'}.\n"
        f"- Jobs with no log and no records, charged whole to their job-name stage: "
        f"{', '.join(no_log) or 'none'}.\n"
        "- Jobs without node energy in sacct: "
        f"{', '.join(j['job_id'] for j in no_energy) or 'none'}.\n"
        f"- Recorded steps with an unknown GPU count: {len(unknown)}; last seen running in a "
        f"job that has ended (`killed`): {len(killed)}.\n"
        "- A reconstructed job is one span per logged step; a split inside a step (e.g. per "
        "generation pass) comes from the job's own report, not from here.\n"
        "- Work outside these Slurm jobs is not here: the published SFT adapter "
        "(`Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en`) and the v1 data, trained and "
        "generated elsewhere, and anything run without `cost` on a login node or another "
        "machine.\n",
        "## Stating a measured claim\n",
        "- **Speed: the same work done differently.** Two steps doing identical work (the same "
        "prompts and decoding, the same training data and schedule) that differ only in a "
        "setting, the software or the hardware -- prefix caching on or off, batch order, one "
        "engine or another. Faster by `1 - t_new / t_ref`, on `generate_s` for steady-state "
        "throughput (it leaves out `load_s`) or on `gpu_h_measured` end to end; say which. "
        "Run each condition at least twice and give the median and range.\n"
        "- **Methods: cost to reach an effect.** Methods do different work per step (a GRPO "
        "step generates rollouts, DPO computes a reference pass), so equal steps are not equal "
        "work. Compare them from a common starting checkpoint, on the same evaluation, as the "
        "GPU-hours each needed to reach the checkpoint that was audited "
        "(`checkpoints.csv`, or `recipe_gpu_h` for a whole run), next to the effect.\n"
        "- **Parameters**: fewer by `1 - p_new / p_ref`, from `params_trainable` or "
        "`params_modified`, against the same base (`params_base_text`).\n"
        "- Quote both rows' job IDs and the basis (measured or reconstructed).\n",
    ]
    return "\n".join(parts)


def output_root() -> Path:
    """$HBM_OUTPUT_ROOT; on a login node, where only the jobs set it, the model drive the
    checkout sits in (its .hbm-models-root marker), as every worktree under $XW does."""
    if os.environ.get("HBM_OUTPUT_ROOT", "").strip():
        return Path(os.environ["HBM_OUTPUT_ROOT"].strip())
    return REPO.parent if (REPO.parent / ".hbm-models-root").exists() else REPO


def main(argv: list[str] | None = None) -> int:
    root = output_root()
    records = Path(os.environ.get("HBM_COST_DIR", "").strip() or root / "cost" / "records")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--since", default="2026-10-01", help="sacct start date")
    parser.add_argument("--prefix", default="xw-", help="job-name prefix of the project's jobs")
    parser.add_argument("--records", type=Path, default=records)
    parser.add_argument("--logs", type=Path, default=root / "logs")
    parser.add_argument("--outputs-root", type=Path, default=root,
                        help="where outputs/<family>/<run>/ adapters live")
    parser.add_argument("--hf-home", type=Path,
                        default=Path(os.environ.get("HF_HOME") or root / "hf-cache"))
    parser.add_argument("--labels", type=Path, default=REPO / "reports/cost/labels.csv")
    parser.add_argument("--out", type=Path, default=REPO / "reports/cost")
    parser.add_argument("--no-sacct", action="store_true",
                        help="reuse the jobs in <out>/jobs.csv instead of calling sacct")
    args = parser.parse_args(argv)

    now = datetime.now()
    if args.no_sacct:
        jobs = cached_jobs(args.out / "jobs.csv")
    else:
        try:
            jobs = fetch_sacct(args.since, args.prefix)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"sacct unavailable ({e}); reusing {args.out / 'jobs.csv'}", file=sys.stderr)
            jobs = cached_jobs(args.out / "jobs.csv")
    by_id = {job["job_id"]: job for job in jobs}

    records = load_records(args.records)
    steps = [record_step(r, by_id, now) for r in records]
    recorded = {s["job_id"] for s in steps if s["job_id"]}
    for job in jobs:
        if job["job_id"] not in recorded:
            steps.extend(log_steps(job, args.logs, now))
    apply_labels(steps, load_labels(args.labels))
    mark_recipe(steps)
    steps.sort(key=lambda s: (s["start"] or "", s["job_id"]))

    job_rows = job_table(jobs, steps, recorded)
    stages = stage_table(steps, job_rows)
    arms = arm_table(steps)
    checkpoints = checkpoint_table(steps)
    throughput = throughput_table(steps)
    params = params_table(args.outputs_root, args.hf_home)

    write_csv(args.out / "steps.csv", steps, STEP_COLUMNS)
    write_csv(args.out / "jobs.csv", job_rows, JOB_COLUMNS)
    write_csv(args.out / "by_stage.csv", stages)
    write_csv(args.out / "by_arm.csv", arms)
    write_csv(args.out / "checkpoints.csv", checkpoints, CHECKPOINT_COLUMNS)
    write_csv(args.out / "throughput.csv", throughput)
    write_csv(args.out / "params.csv", params)
    (args.out / "COST.md").write_text(
        render_markdown(steps, job_rows, stages, arms, checkpoints, throughput, params,
                        args.since, args.prefix), encoding="utf-8")
    print(f"{len(jobs)} jobs, {len(records)} records, {len(steps)} steps -> {args.out}")
    for row in stages:
        print(f"  {row['stage']:<26} {row['gpu_h']!s:>9} GPU-h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
