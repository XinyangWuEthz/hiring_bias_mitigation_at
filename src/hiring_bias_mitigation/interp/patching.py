"""Activation patching between attribute variants, batched over contrast pairs.

One contrast pair is two prompts that differ only in the attribute span: the *target* A (the
run that is patched) and the *source* B (where patched values come from). A patch replaces an
activation of A, at a set of A's positions, by B's activation at the matching positions. Every
position class used here lies after the span, where the two prompts are identical text and
align from the end (`segment.align`).

The effect of a patch on one pair is how much of the attribute effect it removes:

    removed = s * (m_A - m_patched),   s = sign(m_A - m_B),   normalised by |m_A - m_B|

so 1 means "patching this alone turns A's decision margin into B's", 0 "no effect". With `s`
taken per pair, effects of opposite direction do not cancel: this measures the attribute
*sensitivity* a component carries (the team's primary metric; see the results notebook, §4).
The directional variant is the same sum restricted to pairs whose effect points the group's
way. Aggregation is a ratio of sums over pairs (`summarise`), never a mean of per-pair ratios.

Batches are left-padded, and rows carry different numbers of positions per class, so patches
are boolean masks over (row, position) with the source laid onto the target's coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from . import arch
from .hooks import Hooks
from .margins import Readout
from .segment import Segmentation

#: Position classes patched over, all after the span. `suffix` is everything from the span
#: separator to the decision token; `late` is the chat-template close plus the forced prefix.
CLASSES = ("suffix", "late", "final")


@dataclass
class PairBatch:
    ids_a: torch.Tensor  # (B, Ta) left-padded
    ids_b: torch.Tensor  # (B, Tb)
    mask_a: torch.Tensor
    mask_b: torch.Tensor
    #: for every target position, the index of the aligned source position (-1 outside the suffix)
    src_index: torch.Tensor  # (B, Ta)
    classes: dict[str, torch.Tensor]  # name -> (B, Ta) bool
    rows: pd.DataFrame


def _left_pad(seqs: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    offsets = []
    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = torch.tensor(s)
        mask[i, width - len(s) :] = 1
        offsets.append(width - len(s))
    return ids, mask, offsets


def make_batch(
    rows: pd.DataFrame, seg_a: list[Segmentation], seg_b: list[Segmentation], pad_id: int
) -> PairBatch:
    ids_a, mask_a, off_a = _left_pad([s.input_ids for s in seg_a], pad_id)
    ids_b, mask_b, off_b = _left_pad([s.input_ids for s in seg_b], pad_id)
    n, ta = ids_a.shape
    src_index = torch.full((n, ta), -1, dtype=torch.long)
    classes = {c: torch.zeros((n, ta), dtype=torch.bool) for c in CLASSES}
    for i, (a, b) in enumerate(zip(seg_a, seg_b)):
        start = a.suffix_start()
        shift = a.n_tokens - b.n_tokens
        pos = np.arange(start, a.n_tokens)
        if a.input_ids[start:] != b.input_ids[start - shift :]:
            raise AssertionError(f"pair {rows.iloc[i]['pair_uid']}: suffixes are not identical")
        src_index[i, off_a[i] + pos] = torch.from_numpy(off_b[i] + pos - shift)
        classes["suffix"][i, off_a[i] + pos] = True
        classes["late"][i, off_a[i] + a.bounds["chat_close"][0] : off_a[i] + a.n_tokens] = True
        classes["final"][i, -1] = True
    return PairBatch(ids_a, ids_b, mask_a, mask_b, src_index, classes, rows.reset_index(drop=True))


def subset(batch: PairBatch, idx: list[int]) -> PairBatch:
    """Rows `idx` of a batch, with padding columns no remaining row uses trimmed away."""
    keep_a = batch.mask_a[idx].any(0)
    keep_b = batch.mask_b[idx].any(0)
    cut_a, cut_b = int((~keep_a).cumprod(0).sum()), int((~keep_b).cumprod(0).sum())
    src = batch.src_index[idx][:, cut_a:]
    src = torch.where(src >= 0, src - cut_b, src)
    return PairBatch(
        batch.ids_a[idx][:, cut_a:],
        batch.ids_b[idx][:, cut_b:],
        batch.mask_a[idx][:, cut_a:],
        batch.mask_b[idx][:, cut_b:],
        src,
        {k: v[idx][:, cut_a:] for k, v in batch.classes.items()},
        batch.rows.iloc[idx].reset_index(drop=True),
    )


def subset_source(source: dict, batch: PairBatch, idx: list[int], keys) -> dict:
    """The matching rows of a captured source, trimmed like `subset` trims the target."""
    cut_a = int((~batch.mask_a[idx].any(0)).cumprod(0).sum())
    return {k: source[k][idx][:, cut_a:] for k in keys}


def token_batches(lengths: list[int], budget: int) -> list[list[int]]:
    """Consecutive index groups whose padded size (rows x longest) stays within `budget`."""
    groups, current, longest = [], [], 0
    for i, n in enumerate(lengths):
        if current and (len(current) + 1) * max(longest, n) > budget:
            groups.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, n)
    if current:
        groups.append(current)
    return groups


def _inputs(ids, mask, device) -> dict:
    ids, mask = ids.to(device), mask.to(device)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "position_ids": (mask.cumsum(-1) - 1).clamp(min=0),
    }


#: The hook points patched: name -> (kind, layer). Residual points are the input to each
#: decoder layer plus the final pre-norm residual (33 depths); components are the token mixer
#: (attention or GDN) and the MLP of every layer, and `head_in` for per-head work.
def residual_points(n_layers: int) -> list[tuple[str, int | None]]:
    return [("resid_pre", layer) for layer in range(n_layers)] + [("final_prenorm", None)]


def component_points(n_layers: int) -> list[tuple[str, int]]:
    return [(kind, layer) for layer in range(n_layers) for kind in ("mixer", "mlp")]


@torch.no_grad()
def capture_source(
    model, batch: PairBatch, points: list[tuple[str, int | None]]
) -> dict[tuple[str, int | None], torch.Tensor]:
    """B's activations at every point, laid out on A's positions (zeros outside the suffix)."""
    device = next(model.parameters()).device
    with Hooks(model) as h:
        for kind, layer in points:
            h.capture(kind, layer, positions=None)
        model(**_inputs(batch.ids_b, batch.mask_b, device), use_cache=False, logits_to_keep=1)
        h.assert_fired(1)
        raw = h.cache
        idx = batch.src_index.to(device)
        rows = torch.arange(idx.shape[0], device=device).unsqueeze(1).expand_as(idx)
        out = {}
        for kind, layer in points:
            x = raw.pop(Hooks.key(kind, layer))  # freed as soon as it is re-laid out
            aligned = x[rows, idx.clamp(min=0)]
            aligned[idx < 0] = 0
            out[(kind, layer)] = aligned
    return out


@torch.no_grad()
def margins_of(model, ro: Readout, ids, mask, patches=()) -> np.ndarray:
    """fp32 decision margins of a batch, with optional patches [(kind, layer, mask, src, head)]."""
    device = next(model.parameters()).device
    with Hooks(model) as h:
        for kind, layer, where, source, head in patches:
            h.patch_where(kind, layer, where, source, head=head)
        h.capture("final_prenorm", positions=torch.tensor([-1]))
        model(**_inputs(ids, mask, device), use_cache=False, logits_to_keep=1)
        h.assert_fired(1)
        return ro.margin(h.cache["final_prenorm"][:, 0]).cpu().numpy()


def sweep(model, ro: Readout, batch: PairBatch, source: dict, configs: list[dict]) -> pd.DataFrame:
    """Runs A once per config {kind, layer, cls[, head]}; one row per (pair, config)."""
    out = []
    for cfg in configs:
        where = batch.classes[cfg["cls"]]
        src = source[(cfg["kind"], cfg["layer"])]
        m = margins_of(
            model,
            ro,
            batch.ids_a,
            batch.mask_a,
            [(cfg["kind"], cfg["layer"], where, src, cfg.get("head"))],
        )
        out.append(batch.rows[["pair_uid"]].assign(**cfg, m_patched=m))
    return pd.concat(out, ignore_index=True)


def attribution(
    model, ro: Readout, batch: PairBatch, source: dict, n_layers: int, classes=("suffix", "late")
) -> pd.DataFrame:
    """Attribution patching (AtP): first-order estimates for every head, mixer and MLP at once.

    One forward and backward pass of A gives the gradient of its margin with respect to each
    activation; the estimated margin change from patching component c over a position class is
    (a_B - a_A) . dm/da, summed over those positions. Heads are slices of the output-projection
    input (`head_in`), so 896 head estimates cost no more than one. A first-order estimate can
    be badly wrong where the model is nonlinear (attention saturation), so every head reported
    is re-measured by real patching (`sweep`) and the two are compared.
    """
    device = next(model.parameters()).device
    store: dict = {}

    def keep(name):
        def hook(_m, args, kwargs=None):
            x = args[0]
            x.retain_grad()
            store[name] = x

        return hook

    def keep_out(name):
        def hook(_m, _a, output):
            x = output[0] if isinstance(output, tuple) else output
            x.retain_grad()
            store[name] = x

        return hook

    layers = arch.decoder_layers(model)
    handles = [
        arch.embed_tokens(model).register_forward_hook(lambda _m, _a, out: out.requires_grad_(True))
    ]
    for i, block in enumerate(layers):
        handles.append(arch.out_proj(block).register_forward_pre_hook(keep(("head_in", i))))
        handles.append(arch.mixer(block).register_forward_hook(keep_out(("mixer", i))))
        handles.append(block.mlp.register_forward_hook(keep_out(("mlp", i))))
    final = {}
    handles.append(
        arch.final_norm(model).register_forward_pre_hook(
            lambda _m, args: final.__setitem__("x", args[0])
        )
    )
    try:
        with torch.enable_grad():
            model(**_inputs(batch.ids_a, batch.mask_a, device), use_cache=False, logits_to_keep=1)
            m_a = ro.margin(final["x"][:, -1])
            m_a.sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    rows = []
    for (kind, layer), act in store.items():
        delta = (source[(kind, layer)].to(act.dtype) - act.detach()).float()
        grad = act.grad.float()
        for cls in classes:
            where = batch.classes[cls].to(device).unsqueeze(-1).float()
            contrib = delta * grad * where  # (B, T, d)
            if kind == "head_in":
                n_heads, dim = arch.head_shape(model, layer)
                per_head = contrib.sum(1).view(contrib.shape[0], n_heads, dim).sum(-1)
                for head in range(n_heads):
                    rows.append(
                        pd.DataFrame(
                            {
                                "pair_uid": batch.rows["pair_uid"],
                                "kind": kind,
                                "layer": layer,
                                "head": head,
                                "cls": cls,
                                "atp_delta": per_head[:, head].cpu().numpy(),
                            }
                        )
                    )
            else:
                rows.append(
                    pd.DataFrame(
                        {
                            "pair_uid": batch.rows["pair_uid"],
                            "kind": kind,
                            "layer": layer,
                            "head": -1,
                            "cls": cls,
                            "atp_delta": contrib.sum((1, 2)).cpu().numpy(),
                        }
                    )
                )
    store.clear()  # drop the retained activations and their gradients
    return pd.concat(rows, ignore_index=True)


def summarise(
    effects: pd.DataFrame,
    pairs: pd.DataFrame,
    by: list[str],
    value: str = "m_patched",
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Share of the attribute effect each patch removes, with a candidate-clustered bootstrap CI.

    `pairs` carries pair_uid, candidate_id, m_a, m_b. For real patches `value` is the patched
    margin; for AtP estimates pass value="atp_delta" (an estimated m_patched - m_a).
    """
    p = pairs.set_index("pair_uid")
    e = effects.join(p[["candidate_id", "m_a", "m_b"]], on="pair_uid")
    s = np.sign(e["m_a"] - e["m_b"])
    change = e[value] if value == "atp_delta" else e[value] - e["m_a"]
    e = e.assign(removed=-s * change, size=(e["m_a"] - e["m_b"]).abs())
    rng = np.random.default_rng(seed)
    out = []
    for key, g in e.groupby(by, sort=False):
        per = g.groupby("candidate_id")[["removed", "size"]].sum()
        r, z = per["removed"].to_numpy(), per["size"].to_numpy()
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        boots = r[idx].sum(1) / z[idx].sum(1)
        out.append(
            {
                **dict(zip(by, key if isinstance(key, tuple) else (key,))),
                "share_removed": r.sum() / z.sum(),
                "ci_low": np.quantile(boots, 0.025),
                "ci_high": np.quantile(boots, 0.975),
                "removed_logits_per_pair": g["removed"].mean(),
                "pairs": len(g),
            }
        )
    return pd.DataFrame(out)
