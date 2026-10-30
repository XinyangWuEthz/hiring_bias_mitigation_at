"""Capture and patch activations at named hook points, and prove every hook actually ran.

A patching experiment that silently patches nothing returns the clean run's logits, which
reads as "this component has no effect" -- the most plausible-looking wrong answer there is.
The kernels make it more than hypothetical: transformers swaps Qwen3.5's GDN implementation
for a hub kernel when one is installed, and a replaced forward bypasses hooks on the modules
it subsumes. So every hook counts its calls, and `Hooks.assert_fired` refuses a run in which
any registered hook fired a number of times other than the number of forward passes.

Positions are given per batch row, because left-padded rows place the same prompt position
at different indices: a (B, P) LongTensor, or a 1-D tensor shared by every row, or `None` for
the whole sequence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import torch

from . import arch

Positions = torch.Tensor | None


def _rows(x: torch.Tensor, positions: Positions) -> tuple[torch.Tensor, torch.Tensor] | None:
    if positions is None:
        return None
    pos = positions.to(x.device).long()
    if pos.ndim == 1:
        pos = pos.unsqueeze(0).expand(x.shape[0], -1)
    rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(pos)
    return rows, pos


def gather(x: torch.Tensor, positions: Positions) -> torch.Tensor:
    """x[b, positions[b]] -> (B, P, ...)."""
    idx = _rows(x, positions)
    return x if idx is None else x[idx[0], idx[1]]


def scatter(x: torch.Tensor, positions: Positions, values: torch.Tensor,
            channels: slice | None = None) -> torch.Tensor:
    """A copy of x with x[b, positions[b], channels] = values."""
    out = x.clone()
    idx = _rows(x, positions)
    values = values.to(device=x.device, dtype=x.dtype)
    if idx is None:
        if channels is None:
            out[...] = values
        else:
            out[..., channels] = values
    elif channels is None:
        out[idx[0], idx[1]] = values
    else:
        out[idx[0], idx[1], channels] = values
    return out


class Hooks:
    """A context manager owning a set of hooks on one model.

        with Hooks(model) as h:
            h.capture("resid_post", 12, positions=last)
            h.patch("head_in", 19, positions=last, values=v, head=3)
            model(**batch)
            h.assert_fired(1)
            h.cache["resid_post.L12"]
    """

    def __init__(self, model):
        self.model = model
        self.layers = arch.decoder_layers(model)
        self.cache: dict[str, torch.Tensor] = {}
        self.calls: Counter = Counter()
        self._handles: list = []
        self._keys: list[str] = []

    # -- registration ---------------------------------------------------------------------

    def _module_and_mode(self, kind: str, layer: int | None):
        """(module, "pre" | "post") at which a kind of activation is visible."""
        if kind == "final_prenorm":
            return arch.final_norm(self.model), "pre"
        if kind == "embed":
            return arch.embed_tokens(self.model), "post"
        block = self.layers[layer]
        return {
            "resid_pre": (block, "pre"),
            "resid_mid": (block.post_attention_layernorm, "pre"),
            "resid_post": (block, "post"),
            "mixer": (arch.mixer(block), "post"),
            "mlp": (block.mlp, "post"),
            "head_in": (arch.out_proj(block), "pre"),
        }[kind]

    def _register(self, kind: str, layer: int | None, key: str,
                  edit: Callable[[torch.Tensor], torch.Tensor | None]) -> str:
        module, mode = self._module_and_mode(kind, layer)
        if key in self._keys:
            key = f"{key}~{len(self._keys)}"
        self._keys.append(key)

        if mode == "pre":
            def pre_hook(_module, args, kwargs):
                self.calls[key] += 1
                if args:
                    new = edit(args[0])
                    return None if new is None else ((new, *args[1:]), kwargs)
                name = "hidden_states" if "hidden_states" in kwargs else next(iter(kwargs))
                new = edit(kwargs[name])
                return None if new is None else (args, {**kwargs, name: new})

            self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        else:
            def post_hook(_module, _args, output):
                self.calls[key] += 1
                # Full attention returns (output, weights); GDN, MLPs and layers a tensor.
                tensor = output[0] if isinstance(output, tuple) else output
                new = edit(tensor)
                if new is None:
                    return None
                return (new, *output[1:]) if isinstance(output, tuple) else new

            self._handles.append(module.register_forward_hook(post_hook))
        return key

    @staticmethod
    def key(kind: str, layer: int | None = None, head: int | None = None, tag: str = "") -> str:
        name = kind if layer is None else f"{kind}.L{layer}"
        name += "" if head is None else f".H{head}"
        return name + (f"#{tag}" if tag else "")

    def capture(self, kind: str, layer: int | None = None, positions: Positions = None,
                head: int | None = None, detach: bool = True, tag: str = "") -> str:
        """Stores the activation (optionally one head's channels) at `positions`."""
        key = self.key(kind, layer, head, tag or "capture")
        channels = None if head is None else arch.head_slice(self.model, layer, head)

        def edit(x):
            value = gather(x, positions)
            if channels is not None:
                value = value[..., channels]
            self.cache[key.split("#")[0]] = value.detach().clone() if detach else value
            return None

        return self._register(kind, layer, key, edit)

    def patch(self, kind: str, layer: int | None, positions: Positions, values: torch.Tensor,
              head: int | None = None, tag: str = "") -> str:
        """Replaces the activation at `positions` (one head's channels, if given) by `values`."""
        key = self.key(kind, layer, head, tag or "patch")
        channels = None if head is None else arch.head_slice(self.model, layer, head)
        return self._register(kind, layer, key, lambda x: scatter(x, positions, values, channels))

    def patch_where(self, kind: str, layer: int | None, mask: torch.Tensor, source: torch.Tensor,
                    head: int | None = None, tag: str = "") -> str:
        """Replaces the activation wherever `mask` (B, T) is true by `source` (B, T, d).

        The batched form of `patch`: rows of a padded batch patch different numbers of
        positions, which a (B, P) index cannot express. `source` is already laid out on the
        target's positions (see `interp/patching.align_source`).
        """
        key = self.key(kind, layer, head, tag or "patch_where")
        channels = None if head is None else arch.head_slice(self.model, layer, head)

        def edit(x):
            m = mask.to(x.device).unsqueeze(-1)
            src = source.to(device=x.device, dtype=x.dtype)
            if channels is None:
                return torch.where(m, src, x)
            out = x.clone()
            out[..., channels] = torch.where(m, src[..., channels], x[..., channels])
            return out

        return self._register(kind, layer, key, edit)

    def add(self, kind: str, layer: int | None, positions: Positions, delta: torch.Tensor,
            tag: str = "") -> str:
        """Adds `delta` at `positions` -- steering, rather than replacement."""
        key = self.key(kind, layer, None, tag or "add")

        def edit(x):
            return scatter(x, positions, gather(x, positions) + delta.to(x.dtype))

        return self._register(kind, layer, key, edit)

    # -- bookkeeping ----------------------------------------------------------------------

    def assert_fired(self, n_forward: int) -> None:
        wrong = {k: self.calls[k] for k in self._keys if self.calls[k] != n_forward}
        if wrong:
            raise AssertionError(
                f"hooks fired the wrong number of times for {n_forward} forward pass(es): "
                f"{wrong}. A kernel may have replaced the module "
                "forward; check scripts/interp/00_env_check.py output."
            )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._keys.clear()

    def __enter__(self) -> Hooks:
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
