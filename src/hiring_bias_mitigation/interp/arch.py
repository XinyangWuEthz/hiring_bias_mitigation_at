"""Qwen3.5 hook points and per-head slices. The only file that names the model's modules.

Qwen3.5-9B is a hybrid: 24 Gated-DeltaNet layers (`layer.linear_attn`, a linear-attention
recurrence) and 8 full softmax-attention layers (`layer.self_attn`, at layers 3, 7, ..., 31).
Both kinds write to the residual stream through one output projection, and both are
multi-head before it, so a "head" is the same thing in both: a slice of the input to that
projection.

    full attention   o_proj input  = 16 heads x 256, *after* the sigmoid output gate
    GDN              out_proj input = 32 value heads x 128, after the gated RMSNorm

Two layouts matter for anything finer than a head output:

* `q_proj` emits query and gate interleaved per head -- head h is `[h*512, h*512+256)` for the
  query and the next 256 for the gate (`Qwen3_5Attention.forward` views it as (16, 512) and
  chunks the last dim). The gate is computed from the destination position alone.
* Grouped heads: query head h reads key/value head `h // 4`; GDN value head j reads key head
  `j // 2`. Patching a key/value head moves every query head that reads it.

Under PEFT, `o_proj` and friends become `lora.Linear` wrappers. Hooks attach to the wrapper,
never to its `base_layer`, or the adapter's contribution is silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..mitigation.embedding import _decoder_layers

FULL = "full_attention"
LINEAR = "linear_attention"

#: Kinds of activation a hook can read or write.
#:   resid_pre    input to layer L               (forward pre-hook on the layer)
#:   resid_mid    after the mixer, before the MLP (pre-hook on post_attention_layernorm)
#:   resid_post   output of layer L              (forward hook on the layer)
#:   mixer        token-mixer output, attention or GDN, as added to the residual
#:   mlp          MLP output, as added to the residual
#:   head_in      input to the mixer's output projection, (..., n_heads * head_dim)
#:   final_prenorm  input to the final RMSNorm (the residual the readout sees)
KINDS = ("resid_pre", "resid_mid", "resid_post", "mixer", "mlp", "head_in", "final_prenorm")


def text_config(model):
    config = model.config
    return getattr(config, "text_config", config)


def layer_types(model) -> list[str]:
    return list(text_config(model).layer_types)


def decoder_layers(model):
    return _decoder_layers(model)


def mixer(layer):
    return layer.self_attn if layer.block_type == FULL else layer.linear_attn


def out_proj(layer):
    m = mixer(layer)
    return m.o_proj if layer.block_type == FULL else m.out_proj


def final_norm(model):
    """The final RMSNorm, through any PEFT wrapping."""
    for path in ("model.norm", "model.model.norm", "base_model.model.model.norm"):
        node = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "weight") and not hasattr(node, "layers"):
            return node
    raise AttributeError(f"no final norm found on {type(model).__name__}")


def embed_tokens(model):
    return model.get_input_embeddings()


def head_shape(model, layer_idx: int) -> tuple[int, int]:
    """(n_heads, head_dim) of the output-projection input at one layer."""
    cfg = text_config(model)
    if layer_types(model)[layer_idx] == FULL:
        return cfg.num_attention_heads, cfg.head_dim
    return cfg.linear_num_value_heads, cfg.linear_value_head_dim


def head_slice(model, layer_idx: int, head: int) -> slice:
    _, dim = head_shape(model, layer_idx)
    return slice(head * dim, (head + 1) * dim)


def query_slice(head: int, head_dim: int) -> slice:
    return slice(head * 2 * head_dim, head * 2 * head_dim + head_dim)


def gate_slice(head: int, head_dim: int) -> slice:
    return slice(head * 2 * head_dim + head_dim, (head + 1) * 2 * head_dim)


def kv_head_of(model, head: int) -> int:
    cfg = text_config(model)
    return head // (cfg.num_attention_heads // cfg.num_key_value_heads)


def gdn_key_head_of(model, value_head: int) -> int:
    cfg = text_config(model)
    return value_head // (cfg.linear_num_value_heads // cfg.linear_num_key_heads)


@dataclass(frozen=True)
class Component:
    """One node of a circuit: a head, a whole mixer, an MLP, or a residual position."""

    kind: str
    layer: int
    head: int | None = None

    @property
    def name(self) -> str:
        return f"{self.kind}.L{self.layer}" + ("" if self.head is None else f".H{self.head}")


def components(model, heads: bool = True) -> list[Component]:
    """Every mixer (or every head of it) and every MLP, in layer order."""
    out = []
    for layer in range(len(layer_types(model))):
        if heads:
            n_heads, _ = head_shape(model, layer)
            out.extend(Component("head_in", layer, h) for h in range(n_heads))
        else:
            out.append(Component("mixer", layer))
        out.append(Component("mlp", layer))
    return out
