"""The operating-point anchor of mitigation/grpo_operating_anchor.py, on CPU with a toy model."""
from __future__ import annotations

import contextlib
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation.grpo_operating_anchor import (  # noqa: E402
    anchor_penalty,
    callback_class,
)

HIRE, REJECT, VOCAB = 1, 2, 4


class Toy(torch.nn.Module):
    """Logits from a fixed per-token table, plus a trainable 'adapter' bias on hire."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.zeros(10, VOCAB))
        self.table[:, HIRE] = torch.linspace(-1, 1, 10)
        self.adapter = torch.nn.Parameter(torch.zeros(()))
        self.enabled = True

    def forward(self, input_ids, logits_to_keep=None):
        logits = self.table[input_ids].clone()
        if self.enabled:
            logits[..., HIRE] = logits[..., HIRE] + self.adapter
        return SimpleNamespace(logits=logits)

    @contextlib.contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def tok(text, add_special_tokens=False, return_tensors="pt"):
    return {"input_ids": torch.tensor([[int(c) for c in text]])}


def make(weight=10.0, k=3):
    metrics = defaultdict(list)
    cb = callback_class()(tokenizer=tok, prompts=["01", "05", "09"], weight=weight,
                          rows_per_step=k, seed=0, token_ids=(HIRE, REJECT), rank=0, world=1,
                          metrics=metrics)
    return cb, metrics


def test_penalty_is_the_squared_mean_shift():
    pen, shift = anchor_penalty(torch.tensor([0.6, 0.4]), torch.tensor([0.5, 0.3]), 10.0)
    assert torch.isclose(shift, torch.tensor(0.1)) and torch.isclose(pen, torch.tensor(0.1))


def test_no_gradient_at_the_reference():
    model, (cb, metrics) = Toy(), make()
    cb.on_pre_optimizer_step(None, None, None, model=model)
    assert float(model.adapter.grad) == 0.0 and metrics["anchor/shift"] == [0.0]


def test_gradient_matches_autograd_of_the_penalty_and_adds_to_existing_grad():
    model, (cb, metrics) = Toy(), make(weight=10.0, k=3)
    model.adapter.data.fill_(1.0)  # the policy hires more than SFT on every prompt
    model.adapter.grad = torch.tensor(0.5)
    cb.on_pre_optimizer_step(None, None, None, model=model)
    # the same penalty by plain autograd
    ref_model = Toy()
    ref_model.adapter.data.fill_(1.0)
    ids = [torch.tensor([[0, 1]]), torch.tensor([[0, 5]]), torch.tensor([[0, 9]])]

    def prob(m, x):
        lg = m(x).logits[0, -1]
        return torch.sigmoid(lg[HIRE] - lg[REJECT])

    with ref_model.disable_adapter():
        p_ref = torch.stack([prob(ref_model, x) for x in ids]).detach()
    pen, _ = anchor_penalty(torch.stack([prob(ref_model, x) for x in ids]), p_ref, 10.0)
    pen.backward()
    assert torch.isclose(model.adapter.grad, 0.5 + ref_model.adapter.grad, atol=1e-6)
    assert model.adapter.grad > 0.5  # descent lowers the hire bias
    assert metrics["anchor/shift"][-1] > 0


def test_reference_is_cached_and_does_not_move_with_the_policy():
    model, (cb, _) = Toy(), make()
    cb.on_pre_optimizer_step(None, None, None, model=model)
    before = dict(cb.ref)
    model.adapter.data.fill_(-2.0)
    model.adapter.grad = None
    cb.on_pre_optimizer_step(None, None, None, model=model)
    assert cb.ref == before
