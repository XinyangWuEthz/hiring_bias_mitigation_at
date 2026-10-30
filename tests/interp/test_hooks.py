"""Hook and readout identities on a tiny random Qwen3.5, in fp32 on CPU.

Every patching result rests on these: a capture/patch round trip changes nothing, a patch
before the span changes nothing, patching every mixer and MLP output along the end-aligned
suffix reproduces the source run exactly, head slices sum to the projection, and the fp32
readout equals the logit difference. Each is an identity, so each is tested exactly rather
than "approximately plausible".
"""

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402

from hiring_bias_mitigation.interp import arch  # noqa: E402
from hiring_bias_mitigation.interp import segment as S  # noqa: E402
from hiring_bias_mitigation.interp.hooks import Hooks, gather, scatter  # noqa: E402
from hiring_bias_mitigation.interp.margins import Readout, encode  # noqa: E402

VOCAB = 97


def _device() -> str:
    """The tiny model runs where the real one does.

    transformers binds Qwen3.5's GDN to the `fla` Triton kernels at import time when `fla` is
    installed, and those kernels only run on CUDA tensors. So with `fla` present the tests run
    on the GPU -- checking the identities on the kernels the real runs use -- and without CUDA
    they fall back to the torch reference path, or skip if `fla` would be selected anyway.
    """
    if torch.cuda.is_available():
        return "cuda"
    try:
        import fla.ops  # noqa: F401
    except ImportError:
        return "cpu"
    pytest.skip("fla is installed but CUDA is not: the GDN kernels cannot run on CPU")


DEV = _device()


@pytest.fixture(scope="module")
def model():
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], pad_token_id=0)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    m = Qwen3_5ForCausalLM(cfg).eval().to(DEV)
    # Norm weights start at zero ((1 + w) scaling); randomise them so the readout test
    # would catch a missing `1 +`.
    with torch.no_grad():
        for name, p in m.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.normal_(0, 0.1)
    return m


def _segmentation(span_len: int, n_suffix: int = 14) -> S.Segmentation:
    """A layout-only segmentation: 5 prefix tokens, a span, then a fixed suffix."""
    sizes = {"chat_open": 1, "header": 1, "job": 1, "profile_open": 2, "span": span_len,
             "span_sep": 1, "cv": 5, "profile_close": 1, "body": 2, "format": 1,
             "chat_close": 2, "forced_prefix": 2}
    assert sum(sizes[s] for s in S.SUFFIX_SEGMENTS) == n_suffix
    bounds, cursor = {}, 0
    for name in S.SEGMENTS:
        bounds[name] = (cursor, cursor + sizes[name])
        cursor += sizes[name]
    return S.Segmentation(list(range(cursor)), bounds)


def _pair():
    """Two token sequences sharing prefix and suffix, with spans of different length."""
    g = torch.Generator().manual_seed(1)
    prefix = torch.randint(1, VOCAB, (5,), generator=g)
    suffix = torch.randint(1, VOCAB, (14,), generator=g)
    a = torch.cat([prefix, torch.randint(1, VOCAB, (3,), generator=g), suffix])
    b = torch.cat([prefix, torch.randint(1, VOCAB, (6,), generator=g), suffix])
    return a.unsqueeze(0), b.unsqueeze(0), _segmentation(3), _segmentation(6)


@torch.no_grad()
def test_capture_then_patch_back_is_bit_identical(model):
    x = torch.randint(1, VOCAB, (2, 12))
    with Hooks(model) as h:
        h.capture("resid_post", 1)
        clean = model(input_ids=x.to(DEV)).logits
        h.assert_fired(1)
        values = h.cache["resid_post.L1"]
    with Hooks(model) as h:
        h.patch("resid_post", 1, None, values)
        patched = model(input_ids=x.to(DEV)).logits
        h.assert_fired(1)
    assert torch.equal(clean, patched)


@torch.no_grad()
def test_patching_the_prefix_changes_nothing(model):
    a, b, sa, sb = _pair()
    pa, pb = S.align(sa, sb, "prefix")
    clean = model(input_ids=a.to(DEV)).logits[:, -1]
    with Hooks(model) as h:
        h.capture("resid_post", 2, positions=torch.tensor(pb))
        model(input_ids=b.to(DEV))
        source = h.cache["resid_post.L2"]
    with Hooks(model) as h:
        h.patch("resid_post", 2, torch.tensor(pa), source)
        patched = model(input_ids=a.to(DEV)).logits[:, -1]
    torch.testing.assert_close(patched, clean, rtol=0, atol=1e-6)


@torch.no_grad()
def test_end_aligned_suffix_patch_reproduces_the_source(model):
    """Every mixer and MLP output at every suffix position from B, into A, gives B's logits.

    The residual at a suffix position is its embedding plus the sum of what every mixer and
    MLP wrote there; the tokens are the same, so replacing every write reproduces B's
    residual -- *if* the position map lines the suffixes up. A map off by one fails this.
    """
    a, b, sa, sb = _pair()
    classes = [c for c in S.POSITION_CLASSES if c not in ("prefix", "span")]
    pa = np.concatenate([S.align(sa, sb, c)[0] for c in classes])
    pb = np.concatenate([S.align(sa, sb, c)[1] for c in classes])
    n_layers = len(arch.decoder_layers(model))
    with Hooks(model) as h:
        for layer in range(n_layers):
            h.capture("mixer", layer, positions=torch.tensor(pb))
            h.capture("mlp", layer, positions=torch.tensor(pb))
        target = model(input_ids=b.to(DEV)).logits[:, -1]
        source = dict(h.cache)
    with Hooks(model) as h:
        for layer in range(n_layers):
            h.patch("mixer", layer, torch.tensor(pa), source[f"mixer.L{layer}"])
            h.patch("mlp", layer, torch.tensor(pa), source[f"mlp.L{layer}"])
        patched = model(input_ids=a.to(DEV)).logits[:, -1]
        h.assert_fired(1)
    unpatched = model(input_ids=a.to(DEV)).logits[:, -1]
    assert not torch.allclose(unpatched, target, atol=1e-4), "the two spans should differ"
    torch.testing.assert_close(patched, target, rtol=0, atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("layer", [0, 3])  # one GDN layer, one full-attention layer
def test_head_slices_sum_to_the_output_projection(model, layer):
    block = arch.decoder_layers(model)[layer]
    proj = arch.out_proj(block)
    n_heads, dim = arch.head_shape(model, layer)
    z = torch.randn(3, 5, n_heads * dim, device=DEV)
    slices = [arch.head_slice(model, layer, h) for h in range(n_heads)]
    total = sum(z[..., sl] @ proj.weight[:, sl].T for sl in slices)
    torch.testing.assert_close(total, proj(z), rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_head_patch_touches_only_that_head(model):
    x = torch.randint(1, VOCAB, (1, 10))
    with Hooks(model) as h:
        h.capture("head_in", 3, positions=None)
        model(input_ids=x.to(DEV))
        z = h.cache["head_in.L3"]
    replacement = torch.zeros(1, 1, 16)
    with Hooks(model) as h:
        h.patch("head_in", 3, torch.tensor([9]), replacement, head=1)
        h.capture("head_in", 3, positions=None, tag="after")
        model(input_ids=x.to(DEV))
    # The capture registered after the patch sees the patched input.
    after = h.cache["head_in.L3"]
    changed = (after != z).any(dim=1).squeeze(0)
    assert changed[16:32].any() and not changed[:16].any() and not changed[32:].any()


@torch.no_grad()
def test_fp32_readout_equals_the_logit_difference(model):
    x = torch.randint(1, VOCAB, (3, 9))
    norm = arch.final_norm(model)
    w_u = model.get_output_embeddings().weight
    ro = Readout(5, 7, (w_u[5] - w_u[7]).float(), norm.weight.float(), norm.eps)
    with Hooks(model) as h:
        h.capture("final_prenorm", positions=torch.tensor([-1]))
        logits = model(input_ids=x.to(DEV)).logits[:, -1]
        prenorm = h.cache["final_prenorm"][:, 0]
    torch.testing.assert_close(ro.margin(prenorm), logits[:, 5] - logits[:, 7], rtol=1e-5,
                               atol=1e-5)


class _IdTokenizer:
    """Turns space-separated integers into ids, padding on the left like the real setup."""

    pad_token_id = 0
    padding_side = "left"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        seqs = [[int(t) for t in s.split()] for s in texts]
        width = max(map(len, seqs))
        ids = [[0] * (width - len(s)) + s for s in seqs]
        mask = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


@torch.no_grad()
def test_left_padded_batch_matches_unbatched(model):
    tok = _IdTokenizer()
    texts = ["3 14 15 92 65 35 89 79 32 38", "46 26 43", "83 27 95 2 88 41"]
    alone = [model(**encode(tok, [t], DEV)).logits[0, -1] for t in texts]
    batched = model(**encode(tok, texts, DEV)).logits[:, -1]
    for i, ref in enumerate(alone):
        torch.testing.assert_close(batched[i], ref, rtol=1e-4, atol=1e-4)


def test_assert_fired_refuses_a_hook_that_never_ran(model):
    with Hooks(model) as h:
        h.capture("mlp", 0)
        with pytest.raises(AssertionError, match="wrong number of times"):
            h.assert_fired(1)


def test_gather_scatter_per_row_positions():
    x = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    pos = torch.tensor([[1, 4], [0, 2]])
    got = gather(x, pos)
    assert torch.equal(got[0, 1], x[0, 4]) and torch.equal(got[1, 0], x[1, 0])
    out = scatter(x, pos, torch.zeros(2, 2, 3))
    assert out[0, 1].sum() == 0 and out[0, 4].sum() == 0 and out[0, 0].sum() != 0
    assert x[0, 1].sum() != 0, "scatter must not modify its input"
