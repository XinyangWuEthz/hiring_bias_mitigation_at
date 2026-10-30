"""DAPO from the SFT model: GRPO's rewards, policy and rollouts (mitigation/grpo.py) with DAPO's
changes (Yu et al. 2025, arXiv 2503.14476; the verl recipe), plus a strict balance of the two
reference verdicts in what trains.

DAPO changes four things in GRPO and drops the KL term. Where each stands here:

  token-level loss     TRL's `loss_type: dapo`, TRL 1.x's default: every GRPO run in this
                       repository already used it (GRPO-v3-all's training_args.bin). DAPO configs
                       set it explicitly, and `check_config` refuses another.
  clip-higher          `epsilon` 0.2 and `epsilon_high` 0.28. The clip binds only on an update made
                       off the rollout's policy. With one optimizer step per rollout
                       (steps_per_generation == gradient_accumulation_steps, num_iterations 1) the
                       ratio is 1, and clip_ratio/* stayed 0 through GRPO-v3 and GRPO-v3-all. The
                       DAPO configs take two optimizer steps per rollout; the paper takes 16.
  overlong shaping     `overlong_reward`, the paper's soft punishment (Eq. 13): 0 up to
                       max_completion_length - overlong_buffer tokens, then linear down to
                       -overlong_penalty at max_completion_length. Not its overlong filtering
                       (`mask_truncated_completions`), which its best run did without. Answers
                       here are ~38 tokens of JSON against 96 (GRPO-v3-all: longest untruncated
                       67, one of ~146k completions cut at 96), so it is a guard, not a signal.
  dynamic sampling     `DAPOTrainer`, below.
  no KL                `beta: 0`. SFT-v3 is the start, no longer a pull; the verdict balance and
                       the decision monitor's band are what hold the operating point.

Dynamic sampling. A group whose `num_generations` decisions are all right or all wrong has zero
advantage and teaches nothing; with TRL's token-level loss its tokens still count in the
denominator. GRPO-v3-all had such groups for 50% of its prompts in its first eighth and 96% in
its last (frac_reward_zero_std, job 5116822). DAPO drops them and samples more prompts until
the batch is full of groups that disagree:

  1. Probe. TRL's own `_generate` rolls out this rank's generation batch on its colocated vLLM.
     A group is informative when 0 < correct < num_generations, correct by the parsed decision
     against the reference: the paper's Eq. 11, verl's `filter_groups.metric: acc`. Format and
     length do not count; a group that differs only in format is dropped.
  2. Keep informative groups up to this rank's quota: with `dynamic_sampling_balance`, half per
     reference verdict.
  3. While any rank is short, every rank probes a refill batch, at most
     `dynamic_sampling_max_rounds` rounds in all. Refills are training prompts in the order of a
     seeded permutation the ranks share, each rank its own slice. They are drawn beside the main
     sampler (verl instead consumes its epoch iterator), so the epoch counts the main sampler's
     prompts only and a prompt can train more than once in it; dynamic_sampling.jsonl lists the
     prompts each rollout trained.
  4. Balance. The per-verdict quota is only a cap: a verdict that runs dry would leave the other
     to train alone, and without KL the verdict mix is what holds the hire rate. So with
     `dynamic_sampling_balance` each verdict trains as many groups as the scarcer one has kept,
     counted over all ranks (`balanced_allowance`).
  5. Score only what trains. The kept rollouts are replayed into TRL's
     `_generate_and_score_completions` through `_generate`, so the HF log-probabilities (old
     policy, vLLM importance sampling) are computed for kept rows only, and the rewards,
     advantages and losses are TRL's own. A rank short after the last round makes up its quota
     with dropped groups; those, and the groups step 4 cuts, keep their place with the loss mask
     zeroed.
  6. Stop when dry. `dapo/fill` is the share of the quota that trains. When its mean over the
     last `dynamic_sampling_patience` rollouts falls under `dynamic_sampling_min_fill`, the run
     saves and stops (patience 0: never); the prompts no longer split the policy on both
     verdicts. verl raises after `max_num_gen_batches` 10. A stopped run refuses `resume: true`.

What TRL logs for the batch (reward, frac_reward_zero_std, sampled_hire_rate_ref_*) describes the
trained rows, after filtering and balance, and is not comparable with a GRPO run's.
`dapo/probe_*` are the same readings on each rollout's first probe, the main sampler's draw:
those compare with GRPO. A resumed run restarts the refill order and the fill window.

Run:
    python -m hiring_bias_mitigation.mitigation.dapo --config configs/mitigation/dapo/<x>.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, deque
from pathlib import Path

from ..eval.parsing import HIRE, REJECT, parse_output
from ..generation.grpo_prompts import ROW_KEY
from ..utils.config import load_config, resolve_output_path
from ..utils.logging import get_logger
from . import grpo as G
from . import training_common as C

log = get_logger(__name__)

#: Config keys of this module (not GRPOConfig fields) and their defaults.
DYNAMIC_SAMPLING_DEFAULTS = {
    "dynamic_sampling": True,
    "dynamic_sampling_balance": True,
    "dynamic_sampling_max_rounds": 8,
    "dynamic_sampling_min_fill": 0.5,
    "dynamic_sampling_patience": 25,
}
OVERLONG_DEFAULTS = {"overlong_buffer": 16, "overlong_penalty": 1.0}
#: The quota key when the verdicts are not balanced.
ANY = "any"
SIDES = (HIRE, REJECT)
#: TRL 1.15's `GRPOTrainer._generate` returns these, each a list with one entry per completion
#: except `extra_fields` (a dict; only a rollout_func fills it) and `tool_images` (empty without
#: tools). A field may be None (no tools, no images).
GENERATE_FIELDS = ("prompt_ids", "completion_ids", "tool_mask", "completions", "logprobs",
                   "extra_fields", "images", "tool_images")


# ---- overlong shaping ---------------------------------------------------------------------

def overlong_penalty(length: int, max_length: int, buffer: int, penalty: float = 1.0) -> float:
    """DAPO's soft overlong punishment (Eq. 13): 0 up to `max_length - buffer` tokens, then
    linear down to -penalty at `max_length`, where TRL truncates."""
    start = max_length - buffer
    if buffer <= 0 or length <= start:
        return 0.0
    return -penalty * min(1.0, (length - start) / buffer)


def make_overlong_reward(max_length: int, buffer: int, penalty: float):
    def overlong_reward(completion_ids: list[list[int]], **_) -> list[float]:
        return [overlong_penalty(len(ids), max_length, buffer, penalty) for ids in completion_ids]

    return overlong_reward


# ---- dynamic sampling, without TRL ----------------------------------------------------------

def side_of(row: dict) -> str:
    return str(row["reference_decision"]).lower()


def row_key(row: dict) -> str:
    """A training row's identity in dynamic_sampling.jsonl (grpo_prompts.ROW_KEY)."""
    return "|".join(str(row.get(k)) for k in ROW_KEY)


def check_groups(rows: list[dict], group_size: int) -> None:
    """Whole groups, contiguous: what TRL's RepeatSampler hands each rank when its share of a
    generation batch is a multiple of `num_generations` (check_config)."""
    if len(rows) % group_size:
        raise ValueError(f"{len(rows)} rows are not whole groups of {group_size}")
    for start in range(0, len(rows), group_size):
        if len({rows[i]["prompt"] for i in range(start, start + group_size)}) != 1:
            raise ValueError(f"rows {start}-{start + group_size - 1} are not one prompt's group")


def quotas(n_groups: int, balance: bool) -> dict[str, int]:
    if not balance:
        return {ANY: n_groups}
    if n_groups % 2:
        raise ValueError(f"{n_groups} groups per rank cannot be split evenly between verdicts")
    return {HIRE: n_groups // 2, REJECT: n_groups // 2}


def informative(correct: list[bool]) -> bool:
    """Eq. 11: some completions of the group are right and some are wrong."""
    return 0 < sum(correct) < len(correct)


def take_rows(generated: tuple, index: list[int]) -> tuple:
    """`_generate`'s output restricted to the completions at `index`, in that order."""
    if len(generated) != len(GENERATE_FIELDS):
        raise ValueError(f"_generate returned {len(generated)} fields, expected "
                         f"{GENERATE_FIELDS}: TRL changed; check dapo.GENERATE_FIELDS")
    n = len(generated[GENERATE_FIELDS.index("completion_ids")])
    out = []
    for name, value in zip(GENERATE_FIELDS, generated):
        if name == "extra_fields":
            if value:
                raise ValueError("extra_fields from a rollout_func cannot be filtered by row")
            out.append({})
        elif value is None or (isinstance(value, list) and not value and n):
            out.append(value)
        elif len(value) != n:
            raise ValueError(f"_generate's {name} has {len(value)} entries for {n} completions")
        else:
            out.append([value[i] for i in index])
    return tuple(out)


def concat_rollouts(parts: list[tuple]) -> tuple:
    """`take_rows` outputs one after another, as one `_generate` output."""
    out = []
    for k, name in enumerate(GENERATE_FIELDS):
        values = [part[k] for part in parts]
        if name == "extra_fields":
            out.append({})
        elif all(v is None for v in values):
            out.append(None)
        elif any(v is None for v in values):
            raise ValueError(f"_generate's {name} is None in some rounds only")
        else:
            out.append([x for v in values for x in v])
    return tuple(out)


def _counts() -> dict[str, int]:
    return {HIRE: 0, REJECT: 0}


def dynamic_sample(rows: list[dict], generate, decide, refill, group_size: int,
                   quota: dict[str, int], max_rounds: int, largest_shortfall):
    """DAPO's dynamic sampling for one rank (steps 1-3 of the module docstring).

    generate(rows) -> a `_generate` output for `rows`; decide(rows, generated) -> the parsed
    decision of each row; refill() -> this rank's next batch of whole groups;
    largest_shortfall(n) -> the largest `n` over all ranks, a collective every rank calls once
    per round (identity on one process).

    Returns (rows, generated, n_filler, stats): the kept groups and then `n_filler` dropped ones
    that make up the quota, and their rollouts. `stats` counts per verdict the groups probed,
    informative and kept, and on the first probe (the main sampler's draw) the rows, the right
    decisions, the hires, the groups and the informative groups.
    """
    need = dict(quota)
    n_groups = sum(quota.values())
    kept_rows, kept_parts, spare = [], [], []
    stats = {"rounds": 0, "probed": _counts(), "informative": _counts(), "kept": _counts(),
             "first": {k: _counts() for k in ("rows", "correct", "hire", "groups",
                                              "informative")}}
    batch = rows
    for round_ in range(1, max_rounds + 1):
        check_groups(batch, group_size)
        generated = generate(batch)
        decisions = decide(batch, generated)
        for start in range(0, len(batch), group_size):
            index = list(range(start, start + group_size))
            side = side_of(batch[start])
            key = side if side in need else ANY
            correct = [decisions[i] == side for i in index]
            useful = informative(correct)
            if side in SIDES:
                stats["probed"][side] += 1
                stats["informative"][side] += useful
                if round_ == 1:
                    first = stats["first"]
                    first["rows"][side] += group_size
                    first["correct"][side] += sum(correct)
                    first["hire"][side] += sum(decisions[i] == HIRE for i in index)
                    first["groups"][side] += 1
                    first["informative"][side] += useful
            if useful and need.get(key, 0) > 0:
                need[key] -= 1
                if side in SIDES:
                    stats["kept"][side] += 1
                kept_rows.extend(batch[i] for i in index)
                kept_parts.append(take_rows(generated, index))
            elif len(spare) < n_groups:
                spare.append(([batch[i] for i in index], take_rows(generated, index)))
        stats["rounds"] = round_
        if largest_shortfall(sum(need.values())) == 0:
            break
        if round_ < max_rounds:
            batch = refill()
    n_filler = sum(need.values())
    for group_rows, part in spare[:n_filler]:
        kept_rows.extend(group_rows)
        kept_parts.append(part)
    return kept_rows, concat_rollouts(kept_parts), n_filler, stats


def balanced_allowance(kept_by_rank: list[dict[str, int]], rank: int) -> dict[str, int]:
    """How many of its kept groups per verdict `rank` trains, so that the two verdicts train
    equally many over all ranks: each verdict's total is cut to the smaller one, taken from the
    ranks in order. Every rank computes the same split from the same gathered counts."""
    target = min(sum(kept.get(side, 0) for kept in kept_by_rank) for side in SIDES)
    out = {}
    for side in SIDES:
        left = target
        for r, kept in enumerate(kept_by_rank):
            take = min(kept.get(side, 0), left)
            left -= take
            if r == rank:
                out[side] = take
    return out


def trained_groups(sides: list[str], n_kept: int, allowance: dict[str, int] | None) -> list[bool]:
    """Per group of `dynamic_sample`'s rows: does it train? The first `n_kept` are kept groups
    and the rest filler; with an allowance, a verdict's kept groups beyond it do not train, the
    later-found ones first."""
    used = defaultdict(int)
    out = []
    for g, side in enumerate(sides):
        trains = g < n_kept and (allowance is None or used[side] < allowance.get(side, 0))
        used[side] += trains
        out.append(trains)
    return out


class RefillOrder:
    """Training-prompt indices for refill rounds: one seeded permutation after another, the same
    on every rank, so `take(k, rank, world)` gives each rank its own k prompts of a round."""

    def __init__(self, n: int, seed: int):
        import numpy as np

        self.n = n
        self.rng = np.random.default_rng(seed)
        self.queue: list[int] = []

    def take(self, k: int, rank: int = 0, world: int = 1) -> list[int]:
        chosen = []
        while len(chosen) < k * world:
            if not self.queue:
                # A round that runs into the next permutation takes its prompts not yet chosen
                # in this round first, so no prompt appears twice in it while n allows.
                fresh = self.rng.permutation(self.n).tolist()
                taken = set(chosen)
                self.queue = [i for i in fresh if i not in taken] + \
                    [i for i in fresh if i in taken]
            chosen.append(self.queue.pop(0))
        return chosen[rank * k:(rank + 1) * k]


class FillWindow:
    """The last `patience` fills; `dry` when their mean is under `min_fill` (patience 0: never)."""

    def __init__(self, patience: int, min_fill: float):
        self.patience = patience
        self.min_fill = min_fill
        self.fills = deque(maxlen=max(1, patience))

    def add(self, fill: float) -> bool:
        self.fills.append(fill)
        return (self.patience > 0 and len(self.fills) == self.patience
                and sum(self.fills) / self.patience < self.min_fill)


# ---- config -------------------------------------------------------------------------------

def check_config(cfg: dict, world_size: int | None = None) -> dict:
    """DAPO's settings with their defaults, checked before any model loads.

    Refuses what would make the run something other than DAPO or break dynamic sampling, and
    warns when clip-higher cannot bind.
    """
    world = int(world_size or os.environ.get("WORLD_SIZE", "1"))
    settings = {**DYNAMIC_SAMPLING_DEFAULTS, **OVERLONG_DEFAULTS,
                **{k: cfg[k] for k in (*DYNAMIC_SAMPLING_DEFAULTS, *OVERLONG_DEFAULTS) if k in cfg}}
    if cfg.get("loss_type") != "dapo":
        raise ValueError("DAPO's token-level loss: set loss_type: dapo")
    epsilon = float(cfg.get("epsilon", 0.2))
    epsilon_high = float(cfg.get("epsilon_high", epsilon))
    if epsilon_high < epsilon:
        raise ValueError(f"epsilon_high {epsilon_high} is below epsilon {epsilon}")
    if "max_completion_length" not in cfg:
        raise ValueError("set max_completion_length: the overlong reward is measured against it")
    if not 0 <= int(settings["overlong_buffer"]) < int(cfg["max_completion_length"]):
        raise ValueError("overlong_buffer must be at least 0 and below max_completion_length")
    accumulation = int(cfg.get("gradient_accumulation_steps", 8))
    per_device = int(cfg.get("per_device_train_batch_size", 2))
    if cfg.get("generation_batch_size") is not None:
        rows_per_rank, rest = divmod(int(cfg["generation_batch_size"]), world)
        if rest:
            raise ValueError(f"generation_batch_size {cfg['generation_batch_size']} is not split "
                             f"evenly over {world} ranks")
        steps_per_generation = rows_per_rank // per_device
    else:
        steps_per_generation = int(cfg.get("steps_per_generation", accumulation))
        rows_per_rank = per_device * steps_per_generation
    group_size = int(cfg.get("num_generations", 8))
    if settings["dynamic_sampling"]:
        if rows_per_rank % group_size:
            raise ValueError(
                f"each rank generates {rows_per_rank} completions per generation batch, not whole "
                f"groups of {group_size}: dynamic sampling filters whole groups on each rank")
        quotas(rows_per_rank // group_size, bool(settings["dynamic_sampling_balance"]))
        if int(settings["dynamic_sampling_max_rounds"]) < 1:
            raise ValueError("dynamic_sampling_max_rounds must be at least 1")
        if not 0 <= float(settings["dynamic_sampling_min_fill"]) <= 1:
            raise ValueError("dynamic_sampling_min_fill must be between 0 and 1")
        if int(settings["dynamic_sampling_patience"]) < 0:
            raise ValueError("dynamic_sampling_patience must be at least 0 (0: no fill stop)")
    if steps_per_generation % accumulation:
        raise ValueError(f"steps_per_generation {steps_per_generation} is not a whole number of "
                         f"optimizer steps ({accumulation} micro-steps each)")
    # A checkpoint inside a rollout resumes onto a fresh rollout of the interrupted prompts:
    # they train once more, and every later rollout shifts by a step.
    rollout_steps = steps_per_generation // accumulation * int(cfg.get("num_iterations", 1))
    if int(cfg.get("save_steps", 25)) % rollout_steps:
        raise ValueError(f"save_steps {cfg.get('save_steps', 25)} is not a multiple of the "
                         f"{rollout_steps} optimizer steps a rollout trains")
    if epsilon_high > epsilon and rollout_steps == 1:
        log.warning("clip-higher cannot bind: every update is on the rollout's own policy "
                    "(steps_per_generation %d, accumulation %d, num_iterations 1)",
                    steps_per_generation, accumulation)
    settings["groups_per_rank"] = rows_per_rank // group_size
    return settings


def refuse_stopped_resume(cfg: dict) -> None:
    """A run that stopped itself (dry prompts, or the monitor's band) is not resumed past that
    stop by a resubmission with `resume: true`."""
    if not cfg.get("resume"):
        return
    manifest = Path(resolve_output_path(cfg["output_dir"])) / "run_config.json"
    if not manifest.is_file():
        return
    status = str(json.loads(manifest.read_text()).get("status", ""))
    if status.startswith(("stopped:", "aborted:")):
        raise RuntimeError(f"{manifest.parent} ended with '{status}'; resume: true would train "
                           "past that stop. Audit it as it is, or move it aside to start over.")


def manifest_extra(settings: dict):
    def extra(args) -> dict:
        return {
            "objective": "dapo",
            "loss_type": args.loss_type,
            "epsilon": args.epsilon,
            "epsilon_high": args.epsilon_high,
            "steps_per_generation": args.steps_per_generation,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_iterations": args.num_iterations,
            "mask_truncated_completions": args.mask_truncated_completions,
            "dynamic_sampling": {k: settings[k] for k in (*DYNAMIC_SAMPLING_DEFAULTS,
                                                          "groups_per_rank")},
            "rewards": {"decision": G.DECISION_REWARD, "format": G.FORMAT_REWARD,
                        "feedback_words": list(G.FEEDBACK_WORDS),
                        "overlong": {"max_completion_length": args.max_completion_length,
                                     "buffer": settings["overlong_buffer"],
                                     "penalty": settings["overlong_penalty"]}},
        }

    return extra


# ---- the trainer --------------------------------------------------------------------------

def _completion_text(completion) -> str:
    """A completion as text: plain for a text prompt, the assistant message's content for a
    conversational one."""
    return completion if isinstance(completion, str) else completion[-1]["content"]


def _dapo_trainer_class():
    import torch
    from accelerate.utils import gather_object
    from transformers import TrainerCallback
    from trl import GRPOTrainer

    class DAPOTrainer(GRPOTrainer):
        """GRPOTrainer with DAPO's dynamic sampling (module docstring, steps 1-6)."""

        class StopWhenDry(TrainerCallback):
            """Saves and stops once the trainer has set `stop_reason`."""

            def __init__(self, trainer):
                self.trainer = trainer

            def on_step_end(self, args, state, control, **kwargs):
                if self.trainer.stop_reason:
                    control.should_save = True
                    control.should_training_stop = True

        def __init__(self, *args, dynamic_sampling: dict | None = None, **kwargs):
            self.dynamic_sampling = dynamic_sampling
            self.stop_reason = None
            self._replay = None
            self._refill_order = None
            self._fill_window = FillWindow(
                int(dynamic_sampling["dynamic_sampling_patience"]),
                float(dynamic_sampling["dynamic_sampling_min_fill"])) if dynamic_sampling else None
            super().__init__(*args, **kwargs)
            if dynamic_sampling:
                self.add_callback(self.StopWhenDry(self))

        def _generate(self, prompts):
            if self._replay is not None:
                replay, self._replay = self._replay, None
                return replay
            return super()._generate(prompts)

        def _gather_counts(self, values: list[float]) -> list[list[float]]:
            """Each rank's `values`, in rank order (a collective)."""
            local = torch.tensor(values, device=self.accelerator.device, dtype=torch.float32)
            return self.accelerator.gather(local).view(-1, len(values)).tolist()

        def _largest_shortfall(self, n: int) -> int:
            return int(max(row[0] for row in self._gather_counts([n])))

        def _refill(self, n_groups: int) -> list[dict]:
            if self._refill_order is None:
                self._refill_order = RefillOrder(len(self.train_dataset), self.args.seed + 1)
            index = self._refill_order.take(n_groups, self.accelerator.process_index,
                                            self.accelerator.num_processes)
            rows = [dict(self.train_dataset[i]) for i in index]
            return [dict(row) for row in rows for _ in range(self.num_generations)]

        def _generate_and_score_completions(self, inputs):
            if not (self.dynamic_sampling and self.model.training):
                return super()._generate_and_score_completions(inputs)
            balance = bool(self.dynamic_sampling["dynamic_sampling_balance"])
            group_size = self.num_generations
            n_groups = len(inputs) // group_size
            parent_generate = super()._generate

            def decide(rows, generated):
                completions = generated[GENERATE_FIELDS.index("completions")]
                return [parse_output(_completion_text(c), r.get("lang", "en")).decision
                        for c, r in zip(completions, rows)]

            rows, generated, n_filler, stats = dynamic_sample(
                list(inputs), lambda batch: parent_generate([r["prompt"] for r in batch]),
                decide, lambda: self._refill(n_groups), group_size, quotas(n_groups, balance),
                int(self.dynamic_sampling["dynamic_sampling_max_rounds"]),
                self._largest_shortfall)
            sides = [side_of(rows[g * group_size]) for g in range(n_groups)]
            allowance = None
            if balance:
                kept_by_rank = [{HIRE: h, REJECT: r} for h, r in self._gather_counts(
                    [stats["kept"][HIRE], stats["kept"][REJECT]])]
                allowance = balanced_allowance(kept_by_rank, self.accelerator.process_index)
            trains = trained_groups(sides, n_groups - n_filler, allowance)
            self._replay = generated
            try:
                output = super()._generate_and_score_completions(rows)
            finally:
                self._replay = None
            for g, train in enumerate(trains):
                if not train:
                    output["completion_mask"][g * group_size:(g + 1) * group_size] = 0
                    output["advantages"][g * group_size:(g + 1) * group_size] = 0
            # Every rank, masked or not: num_items_in_batch is a sum over ranks.
            mask = output["completion_mask"]
            if "tool_mask" in output:
                mask = mask * output["tool_mask"]
            output["num_items_in_batch"] = self.accelerator.gather(mask.sum()).sum()
            self._record(stats, sides, trains, rows, n_groups)
            return output

        def _record(self, stats: dict, sides: list[str], trains: list[bool], rows: list[dict],
                    n_groups: int) -> None:
            group_size = self.num_generations
            trained = {side: sum(t and s == side for s, t in zip(sides, trains)) for side in SIDES}
            first = stats["first"]
            names = [f"{key}_{side}" for key in ("probed", "informative", "kept") for side in SIDES]
            names += [f"trained_{side}" for side in SIDES] + ["needed"]
            names += [f"first_{key}_{side}" for key in first for side in SIDES]
            values = [stats[key][side] for key in ("probed", "informative", "kept")
                      for side in SIDES]
            values += [trained[side] for side in SIDES] + [n_groups]
            values += [first[key][side] for key in first for side in SIDES]
            total = dict(zip(names, (sum(col) for col in zip(*self._gather_counts(values)))))
            trained_keys = gather_object([row_key(rows[g * group_size])
                                          for g, t in enumerate(trains) if t])

            def rate(num: str, den: str) -> float:
                return total[num] / total[den] if total[den] else float("nan")

            fill = (total[f"trained_{HIRE}"] + total[f"trained_{REJECT}"]) / total["needed"]
            metrics = self._metrics["train"]
            metrics["dapo/rounds"].append(float(stats["rounds"]))
            metrics["dapo/fill"].append(fill)
            for side in SIDES:
                metrics[f"dapo/kept_ref_{side}"].append(total[f"kept_{side}"])
                metrics[f"dapo/informative_rate_ref_{side}"].append(
                    rate(f"informative_{side}", f"probed_{side}"))
                metrics[f"dapo/probe_hire_rate_ref_{side}"].append(
                    rate(f"first_hire_{side}", f"first_rows_{side}"))
            groups = total[f"first_groups_{HIRE}"] + total[f"first_groups_{REJECT}"]
            metrics["dapo/probe_zero_variance"].append(
                1 - (total[f"first_informative_{HIRE}"] + total[f"first_informative_{REJECT}"])
                / groups if groups else float("nan"))
            first_rows = total[f"first_rows_{HIRE}"] + total[f"first_rows_{REJECT}"]
            metrics["dapo/probe_accuracy"].append(
                (total[f"first_correct_{HIRE}"] + total[f"first_correct_{REJECT}"]) / first_rows
                if first_rows else float("nan"))
            if self.accelerator.is_main_process:
                record = {"step": self.state.global_step, "rounds": stats["rounds"],
                          "fill": fill, **total, "trained_rows": trained_keys}
                with (Path(self.args.output_dir) / "dynamic_sampling.jsonl").open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
            if self._fill_window.add(fill) and not self.stop_reason:
                window = self._fill_window
                self.stop_reason = (f"stopped: dynamic sampling filled under {window.min_fill} "
                                    f"of its quota on average over {window.patience} rollouts")
                log.error("%s (step %d); saving and stopping", self.stop_reason,
                          self.state.global_step)

    return DAPOTrainer


# ---- training -----------------------------------------------------------------------------

def train(cfg: dict) -> C.TrainArtifacts:
    settings = check_config(cfg)
    refuse_stopped_resume(cfg)
    rewards = [G.decision_reward, G.format_reward,
               make_overlong_reward(int(cfg["max_completion_length"]),
                                    int(settings["overlong_buffer"]),
                                    float(settings["overlong_penalty"]))]
    return G.train(cfg, trainer_cls=_dapo_trainer_class(), reward_funcs=rewards,
                   trainer_kwargs={"dynamic_sampling": settings
                                   if settings["dynamic_sampling"] else None},
                   manifest_extra=manifest_extra(settings))


def main() -> None:
    C.configure_cuda_allocator()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("stage", "dapo")
    if cfg.get("sft_checkpoint"):
        cfg["base_model"] = C.resolve_sft_checkpoint(cfg["sft_checkpoint"])
    artifacts = train(cfg)
    log.info("DAPO complete -> %s", artifacts.output_dir)


if __name__ == "__main__":
    main()
