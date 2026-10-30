"""The bias projection of mitigation/grpo_bias_projection.py, on CPU."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation.grpo_bias_projection import project_bias, trainer_class  # noqa: E402

G = 16


def group_advantages(hires: int, anchor: str) -> tuple[torch.Tensor, torch.Tensor]:
    """TRL's group-scaled advantages for one group with `hires` of 16 deciding hire."""
    s = torch.tensor([1.0] * hires + [-1.0] * (G - hires))
    r = (s == (1.0 if anchor == "hire" else -1.0)).float()
    a = (r - r.mean()) / (r.std(unbiased=False) + 1e-4) if r.std() > 0 else torch.zeros(G)
    return a, s


def batch(*groups):
    parts = [group_advantages(h, anchor) for h, anchor in groups]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def test_one_sided_batch_has_its_push_removed_but_keeps_its_ordering():
    a, s = batch((15, "hire"), (8, "hire"), (16, "reject"))
    assert (a * s).sum() > 1
    out, stats = project_bias(a, s, G)
    assert abs(float((out * s).sum())) < 1e-4
    assert stats["mixed_groups"] == 2
    # each group still sums to zero
    assert torch.allclose(out.view(-1, G).sum(dim=1), torch.zeros(3), atol=1e-4)
    # the two hire-anchored groups now push in opposite directions
    pushes = (out * s).view(-1, G).sum(dim=1)
    assert pushes[0] * pushes[1] < 0
    # a group whose decisions agree is untouched
    assert torch.equal(out[2 * G:], a[2 * G:])


def test_balanced_batch_is_left_alone():
    a, s = batch((12, "hire"), (4, "reject"))
    assert abs(float((a * s).sum())) < 1e-3
    out, stats = project_bias(a, s, G)
    assert torch.allclose(out, a, atol=1e-3)
    assert abs(stats["alpha"]) < 1e-3


def test_no_decision_variance_is_a_no_op():
    a, s = batch((16, "hire"), (0, "hire"))
    out, stats = project_bias(a, s, G)
    assert torch.equal(out, a) and stats["alpha"] == 0.0


def test_shapes_are_checked():
    with pytest.raises(ValueError):
        project_bias(torch.zeros(15), torch.zeros(15), G)
    with pytest.raises(ValueError):
        project_bias(torch.zeros(16), torch.zeros(32), G)


class FakeAccelerator:
    """Two ranks' gather, played on one process: the trainer gathers advantages, then signs."""

    def __init__(self, rank, other_advantages, other_signs):
        self.process_index = rank
        self.others = [other_advantages, other_signs]

    def gather(self, local):
        other = self.others.pop(0)
        return torch.cat([local, other] if self.process_index == 0 else [other, local])


class FakeBase:
    def __init__(self, advantages):
        self._adv = advantages
        self._metrics = {"train": defaultdict(list)}
        self.model = SimpleNamespace(training=True)
        self.num_generations = G

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        return None

    def _generate_and_score_completions(self, inputs):
        completions = [row["text"] for row in inputs]
        self._calculate_rewards(inputs, None, completions, None)
        return {"advantages": self._adv.clone()}


def test_trainer_applies_one_alpha_over_ranks():
    a, s = batch((15, "hire"), (8, "hire"))
    cls = trainer_class(FakeBase, lambda text, lang: text)
    results = []
    for rank in (0, 1):
        mine = slice(rank * G, (rank + 1) * G)
        other = slice((1 - rank) * G, (2 - rank) * G)
        trainer = cls(a[mine])
        trainer.accelerator = FakeAccelerator(rank, a[other], s[other])
        inputs = [{"text": "hire" if v > 0 else "reject"} for v in s[mine]]
        results.append(trainer._generate_and_score_completions(inputs)["advantages"])
    joined = torch.cat(results)
    expected, _ = project_bias(a, s, G)
    assert torch.allclose(joined, expected, atol=1e-5)
