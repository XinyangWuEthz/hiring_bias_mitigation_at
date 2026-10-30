"""GRPO with the global decision-bias component projected out of the advantages.

Why (docs/GRPO_IMPROVE.md, section 2). In GRPO-v3-all the benchmark decisions moved between
saved checkpoints as one threshold: in 14 of 20 windows at least 95% of the changed decisions
went the same way, and full TPR swung by up to 24 pp in 25 steps. From step ~150 a step had two
or fewer groups with any advantage, so the step's net push toward hire or toward reject was
close to a coin flip.

What. Write s_i = +1 when completion i decides hire, -1 when it rejects, 0 when it has no
decision. A bias b added to the hire-minus-reject decision logit of every prompt has policy
gradient sum_i A_i * dlog pi(d_i)/db = sum_i A_i (1[hire_i] - p_g) = (1/2) sum_i A_i s_i, since
each group's advantages sum to zero. The projection subtracts alpha * (s_i - mean_g s) from
every advantage, with one alpha for the whole generation batch, chosen so that
sum_i A'_i s_i = 0:

    alpha = sum_i A_i s_i / sum_i (s_i - mean_g s) s_i

Each group's advantages still sum to zero, a group whose decisions all agree is untouched, and
groups are reweighted against each other rather than dropped: with only hire-anchored groups
informative, a group that hires more than the batch's average still pushes toward hire, one
that hires less pushes the other way. The first-order push on a shared bias is zero; the push
on the ordering of prompts stays.

The batch is the generation batch of one optimizer step (num_iterations 1), gathered over all
ranks, so alpha is identical on every rank.
"""

from __future__ import annotations

import torch

SIGN = {"hire": 1.0, "reject": -1.0}


def project_bias(advantages: torch.Tensor, signs: torch.Tensor, group_size: int,
                 eps: float = 1e-8) -> tuple[torch.Tensor, dict]:
    """Advantages with the shared decision-bias push removed, and the step's statistics.

    `advantages` and `signs` are 1-D over the global batch in group order (`group_size`
    consecutive completions per prompt).
    """
    if advantages.shape != signs.shape or advantages.dim() != 1:
        raise ValueError("advantages and signs must be equal-length 1-D tensors")
    if advantages.numel() % group_size:
        raise ValueError("batch is not a whole number of groups")
    a, s = advantages.double(), signs.double()
    centered = s - s.view(-1, group_size).mean(dim=1).repeat_interleave(group_size)
    push = float((a * s).sum())
    spread = float((centered * s).sum())
    alpha = push / spread if spread > eps else 0.0
    adjusted = a - alpha * centered if spread > eps else a
    return adjusted.to(advantages.dtype), {
        "push_before": push, "push_after": float((adjusted * s).sum()), "alpha": alpha,
        "decision_spread": spread,
        "mixed_groups": int((centered.view(-1, group_size).abs().sum(dim=1) > 0).sum()),
    }


def trainer_class(base, parse_decision):
    """`base` (TRL's GRPOTrainer) with `project_bias` applied to each training batch."""

    class BiasProjectedGRPOTrainer(base):
        def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
            self._bias_signs = [SIGN.get(parse_decision(c, row.get("lang", "en")), 0.0)
                                for c, row in zip(completions, inputs, strict=True)]
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        def _generate_and_score_completions(self, inputs):
            output = super()._generate_and_score_completions(inputs)
            if not self.model.training:
                return output
            local = output["advantages"]
            signs = torch.tensor(self._bias_signs, dtype=local.dtype, device=local.device)
            if signs.shape != local.shape:
                raise ValueError("decision signs and advantages differ in length")
            everyone = self.accelerator.gather(local)
            all_signs = self.accelerator.gather(signs)
            adjusted, stats = project_bias(everyone, all_signs, self.num_generations)
            n = local.numel()
            start = self.accelerator.process_index * n
            output["advantages"] = adjusted[start:start + n].to(local.dtype)
            for key, value in stats.items():
                self._metrics["train"][f"bias_projection/{key}"].append(float(value))
            return output

    return BiasProjectedGRPOTrainer
