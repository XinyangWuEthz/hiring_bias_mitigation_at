"""The decision margin, read in fp32 from the residual stream rather than from bf16 logits.

The existing margin scripts take `logits[:, -1].float()` -- a cast *after* the bf16 matmul. At
the logit magnitudes this model produces for the two decision words (16-32), bf16's spacing is
0.125, so every margin is quantised to an eighth of a logit: the operating-point thresholds in
`reports/operating_point/` are visibly multiples of it. That was harmless for a hire/reject
threshold. It is not harmless here, where SFT's whole effect on the per-set margin spread is
~0.26 logits and a single head's patching effect is a fraction of that.

So the margin is recomputed from the input to the final RMSNorm: the norm applied in fp32
(Qwen3.5's `(1 + w)` scaling), then a dot product with the fp32 readout direction
`W_U[hire] - W_U[reject]`. Neither the final norm nor `lm_head` is adapted by SFT or DPO, so
this direction is the *same vector* for every model spec -- which is also what makes direct
logit attribution comparable across them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..eval.decision import decision_token_ids
from . import arch
from .hooks import Hooks


@dataclass
class Readout:
    hire_id: int
    reject_id: int
    direction: torch.Tensor  # (d,) fp32, W_U[hire] - W_U[reject]
    norm_weight: torch.Tensor  # (d,) fp32
    eps: float

    def normed(self, prenorm: torch.Tensor) -> torch.Tensor:
        x = prenorm.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * (1.0 + self.norm_weight)

    def margin(self, prenorm: torch.Tensor) -> torch.Tensor:
        return self.normed(prenorm) @ self.direction


def readout(model, tokenizer, words: str | tuple[str, str] = "en") -> Readout:
    hire_id, reject_id = decision_token_ids(tokenizer, words)
    w_u = model.get_output_embeddings().weight
    norm = arch.final_norm(model)
    return Readout(
        hire_id, reject_id,
        direction=(w_u[hire_id].float() - w_u[reject_id].float()).detach(),
        norm_weight=norm.weight.detach().float(),
        eps=float(norm.eps),
    )


def encode(tokenizer, texts: list[str], device) -> dict:
    """Left-padded batch with positions counted from the first real token.

    Without explicit `position_ids` Qwen3.5 numbers positions from the first *pad*, so the same
    prompt would get different absolute positions in different batches. RoPE only sees
    relative positions inside attention, but making them identical removes the question.
    """
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc = {k: v.to(device) for k, v in enc.items()}
    enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    return enc


@torch.no_grad()
def score_batch(model, tokenizer, ro: Readout, texts: list[str], resid_layers: bool = False
                ) -> dict[str, np.ndarray]:
    """Margins for one batch; optionally the last-position residual after every layer."""
    device = next(model.parameters()).device
    enc = encode(tokenizer, texts, device)
    # Left padding puts every row's decision position at index -1.
    last = torch.tensor([-1])
    n_layers = len(arch.decoder_layers(model))
    with Hooks(model) as h:
        h.capture("final_prenorm", positions=last)
        if resid_layers:
            h.capture("embed", positions=last)
            for layer in range(n_layers):
                h.capture("resid_post", layer, positions=last)
        out = model(**enc, use_cache=False, logits_to_keep=1)
        h.assert_fired(1)
        prenorm = h.cache["final_prenorm"][:, 0]
        result = {
            "margin_fp32": ro.margin(prenorm).cpu().numpy(),
        }
        logits = out.logits[:, -1]
        result["margin_bf16"] = (logits[:, ro.hire_id].float()
                                 - logits[:, ro.reject_id].float()).cpu().numpy()
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        result["logprob_hire"] = logprobs[:, ro.hire_id].cpu().numpy()
        result["logprob_reject"] = logprobs[:, ro.reject_id].cpu().numpy()
        result["top1"] = logits.argmax(-1).cpu().numpy()
        if resid_layers:
            stack = [h.cache["embed"][:, 0]] + [h.cache[f"resid_post.L{i}"][:, 0]
                                                for i in range(n_layers)]
            result["resid"] = torch.stack(stack, dim=1).to(torch.float16).cpu().numpy()
    return result
