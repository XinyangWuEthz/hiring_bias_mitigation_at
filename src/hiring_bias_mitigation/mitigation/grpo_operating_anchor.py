"""GRPO with an operating-point anchor: a penalty on the policy's mean hire probability moving
away from SFT-v3's on a fixed set of held-out prompts SFT-v3 is unsure of.

Why (docs/GRPO_IMPROVE.md, round 1 and R2-a). In every GRPO-improve arm the benchmark's hire
threshold wandered by several pp of TPR per 25 steps, and not through the first-order
decision-bias gradient: projecting that out (A1) doubled the swing. That push looks like a
restoring force on the training rows, which the policy is sure of by step ~125, so nothing
holds the threshold where the benchmark's borderline prompts sit. This adds one there. Across
the seven arms of the monitor calibration (job 5123830), the mean margin shift on these rows
correlates 0.93 with the arm's benchmark TPR change.

What. Once per optimizer step, after gradient clipping and before the optimizer, every rank
draws `rows_per_step` anchor prompts with its own seeded RNG and forwards them one at a time
(no padding: Qwen3.5's linear-attention layers are not checked against left padding here) with
the op-margin-v2 decision prefix. With p = sigmoid(logit_hire - logit_reject) and p_SFT the
same prompt's value with the LoRA disabled (computed once, cached), the penalty is

    weight * S^2,   S = mean over all ranks' rows of (p - p_SFT)

Its gradient, 2 * weight * S * mean(dp/dtheta), is computed with torch.autograd.grad, summed
over ranks with one all_reduce, and added to the parameters' .grad. Only the batch mean is
held: individual prompts can move, the operating point cannot drift.

Why not in compute_loss. The first version added the penalty to each micro-step's loss, with
the forward through the unwrapped module next to DDP's. Job 5141612 then hung at step 2 on a
DDP bucket all-reduce that one rank never reached. Here DDP's hooks never see the anchor
graph, and every rank runs the same two collectives each step.
"""

from __future__ import annotations

import numpy as np
import torch


def anchor_penalty(p_policy: torch.Tensor, p_reference: torch.Tensor, weight: float):
    """weight * (mean(p_policy) - mean(p_reference))^2, and the shift itself."""
    shift = p_policy.mean() - p_reference.mean()
    return weight * shift.pow(2), shift


def _all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return tensor


def callback_class():
    from transformers import TrainerCallback

    class OperatingAnchorCallback(TrainerCallback):
        """See the module docstring. `prompts` are the anchor rows rendered through the chat
        template with the decision prefix appended; `token_ids` the first tokens of hire and
        reject after it; `metrics` the trainer's `_metrics["train"]`."""

        def __init__(self, *, tokenizer, prompts: list[str], weight: float, rows_per_step: int,
                     seed: int, token_ids: tuple[int, int], rank: int, world: int, metrics):
            self.ids = [tokenizer(p, add_special_tokens=False, return_tensors="pt")["input_ids"]
                        for p in prompts]
            self.weight, self.k = float(weight), int(rows_per_step)
            self.hire_id, self.reject_id = token_ids
            self.rng = np.random.default_rng(seed + 1000 * rank)
            self.world = world
            self.metrics = metrics
            self.ref: dict[int, float] = {}

        def _prob(self, model, i: int, device) -> torch.Tensor:
            out = model(input_ids=self.ids[i].to(device), logits_to_keep=1)
            logits = out.logits[0, -1].float()
            return torch.sigmoid(logits[self.hire_id] - logits[self.reject_id])

        def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
            if self.weight <= 0:
                return
            params = [p for p in model.parameters() if p.requires_grad]
            device = params[0].device
            picks = [int(i) for i in self.rng.choice(len(self.ids), size=self.k, replace=False)]
            missing = [i for i in picks if i not in self.ref]
            if missing:
                with torch.no_grad(), model.disable_adapter():
                    for i in missing:
                        self.ref[i] = float(self._prob(model, i, device))
            p = torch.stack([self._prob(model, i, device) for i in picks])
            ref = torch.tensor([self.ref[i] for i in picks], device=device)
            # S over every rank's rows: one all_reduce of (sum of p - p_SFT, count).
            local = torch.stack([(p.detach() - ref).sum(),
                                 torch.tensor(float(self.k), device=device)])
            total = _all_reduce_sum(local.clone())
            shift = total[0] / total[1]
            # d(weight * S^2) = 2 weight S dS, dS = sum over all rows of dp / n_rows.
            scale = 2.0 * self.weight * shift / total[1]
            grads = torch.autograd.grad(p.sum(), params, allow_unused=True)
            flat = torch.cat([(g if g is not None else torch.zeros_like(q)).reshape(-1).float()
                              for g, q in zip(grads, params, strict=True)])
            flat = _all_reduce_sum(flat) * scale
            offset = 0
            for q in params:
                n = q.numel()
                g = flat[offset:offset + n].view_as(q).to(q.dtype)
                q.grad = g if q.grad is None else q.grad + g
                offset += n
            self.metrics["anchor/shift"].append(float(shift))
            self.metrics["anchor/penalty"].append(float(self.weight * shift ** 2))
            self.metrics["anchor/grad_norm"].append(float(flat.norm()))

    return OperatingAnchorCallback
