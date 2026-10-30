"""Count-based rollout diagnostics. These hooks never modify a training tensor.

Basic sampled hire rates are collected from actual reward-call rows. For DAPO these are
the rows after sampling selection, including filler/balance rows, not its unscored probes.
Detailed diagnostics support the unmodified TRL GRPOTrainer
only. They inspect its returned, post-importance-correction rollout before shuffling.
An effective group means at least one nonzero advantage, not a proven useful gradient.
Importance masking here affects the policy term; the KL term remains in the loss.
"""

from __future__ import annotations

import json
import math
import types
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "grpo-rollout-counts-v1"
SIDES = ("hire", "reject")
RATE_KEYS = tuple(f"sampled_hire_rate_ref_{side}" for side in SIDES)
COUNT_KEYS = (
    "groups", "all_correct_groups", "mixed_decision_groups", "all_wrong_groups",
    "reward_zero_std_groups", "effective_adv_groups", "eligible_effective_adv_groups",
    "retained_effective_adv_groups", "fully_lost_effective_adv_groups",
    "empty_only_effective_adv_groups", "fully_zero_weighted_eligible_adv_groups",
    "fully_threshold_masked_eligible_adv_groups", "format_only_effective_groups",
    "retained_format_only_effective_groups", "completions", "nonempty_completions",
    "empty_completions", "completion_tokens", "policy_is_weight_sum",
    "missing_sampling_logprob_tokens", "scored_logprob_tokens", "abs_logprob_difference_sum",
    "policy_zero_weight_tokens", "policy_retained_tokens", "retained_effective_adv_tokens",
    "policy_zero_weight_completions", "policy_partly_zero_weight_completions",
    "effective_adv_completions", "retained_effective_adv_completions",
    "is_threshold_masked_completions", "is_threshold_masked_tokens",
    "is_zero_without_threshold_completions", "is_zero_without_threshold_tokens",
)


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def sampled_counts(records):
    counts = Counter()
    for row in records:
        side = row["reference_decision"]
        if side not in SIDES:
            raise ValueError(f"unknown reference decision: {side}")
        counts[f"sampled_count_ref_{side}"] += 1
        counts[f"sampled_hire_count_ref_{side}"] += row["decision"] == "hire"
    return counts


def token_diagnostics(completion_mask, old_logps, sampling_logps, is_ratio, advantage,
                      *, threshold_masked=False, zero_without_threshold=False):
    """Inspect actual policy weights; NaN sampler logps are counted, never hidden."""
    length = len(completion_mask)
    if len(old_logps) != length or len(sampling_logps) != length:
        raise ValueError("log-probability and completion-mask lengths differ")
    if len(is_ratio) == 1:
        is_ratio = is_ratio * length
    if len(is_ratio) != length:
        raise ValueError("importance weights do not align with completion tokens")
    counts = Counter(completions=1)
    for mask, old, sample, weight in zip(
        completion_mask, old_logps, sampling_logps, is_ratio, strict=True
    ):
        if not mask:
            continue
        if not math.isfinite(old) or not math.isfinite(weight) or weight < 0:
            raise ValueError("nonfinite policy log-probability or importance weight")
        counts["completion_tokens"] += 1
        counts["policy_is_weight_sum"] += weight
        if math.isnan(sample):
            counts["missing_sampling_logprob_tokens"] += 1
        elif math.isfinite(sample):
            counts["scored_logprob_tokens"] += 1
            counts["abs_logprob_difference_sum"] += abs(old - sample)
        else:
            raise ValueError("infinite sampling log-probability")
        if weight == 0:
            counts["policy_zero_weight_tokens"] += 1
        else:
            counts["policy_retained_tokens"] += 1
            if advantage != 0:
                counts["retained_effective_adv_tokens"] += 1
    has_tokens = counts["completion_tokens"] > 0
    retained = counts["policy_retained_tokens"] > 0
    counts["empty_completions"] = int(not has_tokens)
    counts["nonempty_completions"] = int(has_tokens)
    counts["policy_zero_weight_completions"] = int(has_tokens and not retained)
    counts["policy_partly_zero_weight_completions"] = int(
        retained and counts["policy_zero_weight_tokens"] > 0
    )
    counts["effective_adv_completions"] = int(advantage != 0)
    counts["retained_effective_adv_completions"] = int(advantage != 0 and retained)
    counts["is_threshold_masked_completions"] = int(has_tokens and threshold_masked)
    counts["is_threshold_masked_tokens"] = counts["completion_tokens"] * threshold_masked
    counts["is_zero_without_threshold_completions"] = int(has_tokens and zero_without_threshold)
    counts["is_zero_without_threshold_tokens"] = (
        counts["completion_tokens"] * zero_without_threshold
    )
    return counts


def group_counts(records, group_size=16):
    """Group globally ordered completions, after gathering across rank boundaries."""
    if len(records) % group_size:
        raise ValueError("global rollout is not a whole number of reward groups")
    result = defaultdict(Counter)
    for start in range(0, len(records), group_size):
        group = records[start:start + group_size]
        identity = ("prompt_id", "reference_decision", "winning_votes", "decided_draws")
        if any(tuple(row[k] for k in identity) != tuple(group[0][k] for k in identity)
               for row in group):
            raise ValueError("a global reward group contains different prompt metadata")
        row = group[0]
        wins, draws = row["winning_votes"], row["decided_draws"]
        if not (isinstance(wins, int) and isinstance(draws, int) and 0 < wins <= draws):
            raise ValueError("invalid anchor vote counts")
        side = row["reference_decision"]
        if side not in SIDES:
            raise ValueError("invalid reference side")
        key = f"{side}/{wins}_of_{draws}"
        counts = Counter(groups=1)
        correct = sum(item["decision"] == side for item in group)
        counts["all_correct_groups"] = int(correct == group_size)
        counts["mixed_decision_groups"] = int(0 < correct < group_size)
        counts["all_wrong_groups"] = int(correct == 0)
        zeros = {item["reward_zero_std"] for item in group}
        if len(zeros) != 1:
            raise ValueError("inconsistent actual reward standard-deviation flags")
        counts["reward_zero_std_groups"] = int(zeros.pop())
        effective = any(item["advantage"] != 0 for item in group)
        eligible = [item for item in group if item["advantage"] != 0
                    and item["token_counts"]["completion_tokens"]]
        retained = any(item["token_counts"]["retained_effective_adv_completions"]
                       for item in group)
        counts["effective_adv_groups"] = int(effective)
        counts["eligible_effective_adv_groups"] = int(bool(eligible))
        counts["retained_effective_adv_groups"] = int(retained)
        counts["fully_lost_effective_adv_groups"] = int(effective and not retained)
        counts["empty_only_effective_adv_groups"] = int(effective and not eligible)
        counts["fully_zero_weighted_eligible_adv_groups"] = int(bool(eligible) and not retained)
        counts["fully_threshold_masked_eligible_adv_groups"] = int(
            bool(eligible) and all(item["is_threshold_masked"] for item in eligible)
        )
        decision_constant = len({item["decision_reward"] for item in group}) == 1
        format_varies = len({item["format_reward"] for item in group}) > 1
        counts["format_only_effective_groups"] = int(
            effective and decision_constant and format_varies
        )
        counts["retained_format_only_effective_groups"] = int(
            retained and decision_constant and format_varies
        )
        for item in group:
            counts.update(item["token_counts"])
        result[key].update(counts)
    return result


def count_rates(counts):
    return {
        "reward_zero_std_group_fraction": safe_ratio(
            counts["reward_zero_std_groups"], counts["groups"]),
        "effective_adv_group_fraction": safe_ratio(
            counts["effective_adv_groups"], counts["groups"]),
        "retained_effective_adv_group_fraction": safe_ratio(
            counts["retained_effective_adv_groups"], counts["groups"]),
        "effective_adv_group_retention": safe_ratio(
            counts["retained_effective_adv_groups"], counts["effective_adv_groups"]),
        "policy_zero_weight_completion_fraction": safe_ratio(
            counts["policy_zero_weight_completions"],
            counts["nonempty_completions"]),
        "policy_zero_weight_token_fraction": safe_ratio(
            counts["policy_zero_weight_tokens"], counts["completion_tokens"]),
        "is_threshold_masked_completion_fraction": safe_ratio(
            counts["is_threshold_masked_completions"], counts["nonempty_completions"]),
        "is_threshold_masked_token_fraction": safe_ratio(
            counts["is_threshold_masked_tokens"], counts["completion_tokens"]),
        "abs_logprob_difference_mean": safe_ratio(
            counts["abs_logprob_difference_sum"], counts["scored_logprob_tokens"]),
        "policy_is_weight_token_mean": safe_ratio(
            counts["policy_is_weight_sum"], counts["completion_tokens"]),
    }


class CountWindow:
    def __init__(self, strata=()):
        self.basic = Counter()
        self.strata = defaultdict(Counter)
        for key in strata:
            self.strata[key] = Counter()
        self.reward_calls = 0
        self.rollouts = 0

    def add_samples(self, records):
        self.basic.update(sampled_counts(records))
        self.reward_calls += 1

    def add_rollout(self, records, group_size):
        for key, counts in group_counts(records, group_size).items():
            self.strata[key].update(counts)
        self.rollouts += 1

    def snapshot(self):
        basic = dict(self.basic)
        for side in SIDES:
            basic.setdefault(f"sampled_count_ref_{side}", 0)
            basic.setdefault(f"sampled_hire_count_ref_{side}", 0)
            basic[f"sampled_hire_rate_ref_{side}"] = safe_ratio(
                basic[f"sampled_hire_count_ref_{side}"], basic[f"sampled_count_ref_{side}"])
        total = Counter()
        strata = {}
        for key, counts in sorted(self.strata.items()):
            counts = Counter({key: counts[key] for key in COUNT_KEYS})
            total.update(counts)
            strata[key] = {"counts": dict(counts), "rates": count_rates(counts)}
        total = Counter({key: total[key] for key in COUNT_KEYS})
        return {"basic": basic, "reward_calls": self.reward_calls, "rollouts": self.rollouts,
                "strata": strata, "total": {"counts": dict(total), "rates": count_rates(total)}}


def _tolist(value):
    return value.detach().cpu().tolist()


def _reward_snapshot(trainer, rewards, group_size):
    # Match the actual TRL float32 aggregation and nanstd, not a Python reconstruction.
    import torch
    from trl.trainer.utils import nanstd

    names = trainer.reward_func_names
    if sorted(names) != ["decision_reward", "format_reward"]:
        raise ValueError("detailed diagnostics require the two original GRPO rewards")
    with torch.no_grad():
        totals = (rewards * trainer.reward_weights.to(rewards.device).unsqueeze(0)).nansum(1)
        if not torch.isfinite(rewards).all() or totals.numel() % group_size:
            raise ValueError("detailed reward snapshot is nonfinite or not whole groups")
        # TRL treats an evaluation group of one as std=0, not nanstd's undefined value.
        std = (nanstd(totals.view(-1, group_size), dim=1) if group_size > 1
               else torch.zeros_like(totals))
        zero_std = torch.isclose(std, torch.zeros_like(std)).repeat_interleave(group_size)
    values = _tolist(rewards)
    return [{"decision_reward": row[names.index("decision_reward")],
             "format_reward": row[names.index("format_reward")], "reward_zero_std": zero}
            for row, zero in zip(values, _tolist(zero_std), strict=True)]


def _gather_payload(payload):
    from accelerate.utils import gather_object

    gathered = gather_object([payload])
    errors = [part["error"] for part in gathered if part.get("error")]
    if errors:
        raise ValueError("rollout diagnostics failed on a rank: " + "; ".join(errors))
    return [row for part in gathered for row in part["rows"]]


def _sampling_snapshot(trainer, result):
    """Recompute threshold flags with TRL's tensor operations, without a forward pass."""
    import torch

    if "tool_mask" in result:
        raise ValueError("detailed diagnostics do not support tool masks")
    with torch.no_grad():
        delta = ((result["old_per_token_logps"] - result["sampling_per_token_logps"])
                 * result["completion_mask"])
        delta = torch.nan_to_num(delta, nan=0.0)
        raw_ratio = torch.exp(delta.sum(dim=-1, keepdim=True))
        threshold_mask = raw_ratio > trainer.vllm_importance_sampling_clip_max
        expected = raw_ratio.masked_fill(threshold_mask, value=0.0)
        actual = result["importance_sampling_ratio"]
        if not torch.equal(expected, actual):
            raise ValueError("actual vLLM IS weights differ from the supported sequence-mask rule")
        unexplained_zero = (actual == 0) & ~threshold_mask
    return [{"is_threshold_masked": flag[0], "is_zero_without_threshold": other[0]}
            for flag, other in zip(_tolist(threshold_mask), _tolist(unexplained_zero), strict=True)]


def install(trainer, *, parse_decision, output_dir, detailed=False, strata=(),
            gather=_gather_payload, reward_snapshot=_reward_snapshot,
            sampling_snapshot=_sampling_snapshot):
    """Install observation-only hooks before train(); return the logging receipt.

    The always-on basic counter observes each actual reward call, after DAPO sampling
    selection; it does not observe unscored probes. Detailed groups are observed once per rollout.
    Raw counts accumulate until Trainer.log and are then divided once for that window.
    """
    if getattr(trainer, "_count_diagnostics_installed", False):
        raise ValueError("count diagnostics already installed")
    if detailed:
        if type(trainer).__name__ != "GRPOTrainer":
            raise ValueError("detailed rollout diagnostics support TRL GRPOTrainer only")
        if trainer.num_generations != 16 or trainer.args.scale_rewards != "group":
            raise ValueError("detailed diagnostics require groups of 16 with group scaling")
        if trainer.args.multi_objective_aggregation != "sum_then_normalize":
            raise ValueError("detailed diagnostics require sum_then_normalize rewards")
        if trainer.args.loss_type != "dapo":
            raise ValueError("detailed diagnostics require the original dapo loss type")
        if not trainer.use_vllm or not trainer.vllm_importance_sampling_correction:
            raise ValueError("detailed diagnostics require the actual vLLM correction")
        if (trainer.vllm_importance_sampling_mode != "sequence_mask"
                or trainer.vllm_importance_sampling_clip_max != 3.0
                or trainer.vllm_importance_sampling_clip_min is not None):
            raise ValueError("detailed diagnostics require sequence_mask with max=3 and no minimum")
        if (trainer.args.mask_truncated_completions
                or trainer.args.off_policy_mask_threshold is not None
                or trainer.args.top_entropy_quantile != 1.0
                or trainer.tools or trainer.environments):
            raise ValueError("detailed diagnostics do not support other policy masks or tools")
    windows = {"train": CountWindow(strata), "eval": CountWindow(strata)}
    pending = {}
    original_rewards = trainer._calculate_rewards
    original_generate = trainer._generate_and_score_completions
    original_log = trainer.log
    path = Path(output_dir) / "rollout_diagnostics.jsonl"

    def mode():
        return "train" if trainer.model.training else "eval"

    def calculate(self, inputs, prompts, completions, completion_ids_list):
        result = original_rewards(inputs, prompts, completions, completion_ids_list)
        payload = {"rows": [], "error": None}
        try:
            if len(inputs) != len(completions):
                raise ValueError("reward rows and completions differ")
            records = [{"reference_decision": str(row["reference_decision"]).lower(),
                        "decision": parse_decision(completion, row.get("lang", "en"))}
                       for row, completion in zip(inputs, completions, strict=True)]
            if detailed:
                k = self.num_generations if mode() == "train" else self.num_generations_eval
                snapshot = reward_snapshot(self, result, k)
                world = self.accelerator.num_processes
                if len(snapshot) != world * len(inputs):
                    raise ValueError("unexpected unequal TRL reward shards")
                offset = self.accelerator.process_index * len(inputs)
                for i, (record, row) in enumerate(zip(records, inputs, strict=True)):
                    record.update({key: row[key] for key in (
                        "prompt_id", "winning_votes", "decided_draws")})
                    record.update(snapshot[offset + i])
                pending[mode()] = records
            payload["rows"] = records
        except Exception as exc:  # noqa: BLE001 -- all ranks must reach the error gather.
            payload["error"] = f"{type(exc).__name__}: {exc}"
        global_records = gather(payload)
        windows[mode()].add_samples(global_records)
        return result

    def generate(self, *args, **kwargs):
        result = original_generate(*args, **kwargs)
        if not detailed:
            return result
        payload = {"rows": [], "error": None}
        try:
            records = pending.pop(mode())
            arrays = {key: _tolist(result[key]) for key in (
                "completion_mask", "old_per_token_logps", "sampling_per_token_logps",
                "importance_sampling_ratio", "advantages")}
            if any(len(value) != len(records) for value in arrays.values()):
                raise ValueError("returned rollout and reward-call rows differ")
            correction = sampling_snapshot(self, result)
            if len(correction) != len(records):
                raise ValueError("importance-correction snapshot length differs")
            for i, record in enumerate(records):
                advantage = arrays["advantages"][i]
                if not isinstance(advantage, (int, float)) or not math.isfinite(advantage):
                    raise ValueError("expected one finite advantage per completion")
                record["advantage"] = advantage
                record.update(correction[i])
                record["token_counts"] = token_diagnostics(
                    arrays["completion_mask"][i], arrays["old_per_token_logps"][i],
                    arrays["sampling_per_token_logps"][i],
                    arrays["importance_sampling_ratio"][i], advantage,
                    threshold_masked=record["is_threshold_masked"],
                    zero_without_threshold=record["is_zero_without_threshold"])
            payload["rows"] = records
        except Exception as exc:  # noqa: BLE001 -- all ranks must reach the error gather.
            payload["error"] = f"{type(exc).__name__}: {exc}"
        global_records = gather(payload)
        k = self.num_generations if mode() == "train" else self.num_generations_eval
        windows[mode()].add_rollout(global_records, k)
        return result

    def log(self, logs, start_time=None):
        current_mode = mode()
        snapshot = windows[current_mode].snapshot()
        if not snapshot["reward_calls"]:
            return original_log(logs, start_time)
        prefix = "eval_" if current_mode == "eval" else ""
        # TRL.log otherwise overwrites our counts with its mean of rollout means.
        for key in RATE_KEYS:
            self._metrics[current_mode].pop(key, None)
        for key, value in snapshot["basic"].items():
            if value is not None:
                logs[prefix + key] = value
        if detailed:
            for label, stats in {"all": snapshot["total"], **snapshot["strata"]}.items():
                for key, value in {**stats["counts"], **stats["rates"]}.items():
                    if value is not None:
                        logs[f"{prefix}rollout/{label}/{key}"] = value
        if self.accelerator.is_main_process and snapshot["reward_calls"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            event = {"schema": SCHEMA, "global_step": self.state.global_step,
                     "mode": current_mode, "detailed": detailed,
                     "policy_mask_excludes_kl": True, **snapshot}
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        original_log(logs, start_time)
        windows[current_mode] = CountWindow(strata)

    trainer._calculate_rewards = types.MethodType(calculate, trainer)
    trainer._generate_and_score_completions = types.MethodType(generate, trainer)
    trainer.log = types.MethodType(log, trainer)
    trainer._count_diagnostics_installed = True
    return {"schema": SCHEMA, "basic_count_weighted_logging": True,
            "detailed": detailed, "path": str(path), "policy_mask_excludes_kl": True,
            "basic_scope": "actual reward-call rows; after DAPO selection, not unscored probes",
            "aggregation": "sum raw counts across ranks and log window, then divide"}
