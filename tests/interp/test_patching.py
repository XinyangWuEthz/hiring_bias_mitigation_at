"""The batched patching engine on a tiny random Qwen3.5: exact identities, then AtP sanity."""

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("transformers")

from hiring_bias_mitigation.interp import arch  # noqa: E402
from hiring_bias_mitigation.interp import patching as PT  # noqa: E402
from hiring_bias_mitigation.interp.margins import Readout  # noqa: E402

from .test_hooks import VOCAB, _segmentation  # noqa: E402


def _batch():
    """Two pairs of different lengths, so left padding and end alignment are both exercised."""
    g = torch.Generator().manual_seed(3)
    segs_a, segs_b = [], []
    for span_a, span_b in ((3, 6), (5, 2)):
        prefix = torch.randint(1, VOCAB, (5,), generator=g).tolist()
        suffix = torch.randint(1, VOCAB, (14,), generator=g).tolist()
        a = _segmentation(span_a)
        b = _segmentation(span_b)
        a.input_ids = prefix + torch.randint(1, VOCAB, (span_a,), generator=g).tolist() + suffix
        b.input_ids = prefix + torch.randint(1, VOCAB, (span_b,), generator=g).tolist() + suffix
        segs_a.append(a)
        segs_b.append(b)
    rows = pd.DataFrame({"pair_uid": ["p0", "p1"], "candidate_id": ["c0", "c1"]})
    return PT.make_batch(rows, segs_a, segs_b, pad_id=0)


def _readout(model):
    norm = arch.final_norm(model)
    w_u = model.get_output_embeddings().weight
    return Readout(5, 7, (w_u[5] - w_u[7]).float().detach(), norm.weight.float().detach(), norm.eps)


def test_patching_the_final_residual_gives_the_source_margin(model):
    batch, ro = _batch(), _readout(model)
    n = len(arch.decoder_layers(model))
    source = PT.capture_source(model, batch, PT.residual_points(n))
    m_b = PT.margins_of(model, ro, batch.ids_b, batch.mask_b)
    m_a = PT.margins_of(model, ro, batch.ids_a, batch.mask_a)
    out = PT.sweep(model, ro, batch, source,
                   [{"kind": "final_prenorm", "layer": None, "cls": "final"},
                    {"kind": "resid_pre", "layer": 0, "cls": "suffix"}])
    final = out[out.kind == "final_prenorm"]["m_patched"].to_numpy()
    first = out[out.kind == "resid_pre"]["m_patched"].to_numpy()
    np.testing.assert_allclose(final, m_b, atol=1e-5)
    # The suffix tokens are identical, so their embeddings are: patching them changes nothing.
    np.testing.assert_allclose(first, m_a, atol=1e-5)
    assert not np.allclose(m_a, m_b, atol=1e-3)


def test_every_component_over_the_suffix_reproduces_the_source(model):
    batch, ro = _batch(), _readout(model)
    n = len(arch.decoder_layers(model))
    source = PT.capture_source(model, batch, PT.component_points(n))
    patches = [(k, layer, batch.classes["suffix"], source[(k, layer)], None)
               for k, layer in PT.component_points(n)]
    m = PT.margins_of(model, ro, batch.ids_a, batch.mask_a, patches)
    np.testing.assert_allclose(m, PT.margins_of(model, ro, batch.ids_b, batch.mask_b), atol=1e-5)


def test_attribution_matches_a_real_patch_in_sign_and_scale(model):
    batch, ro = _batch(), _readout(model)
    for p in model.parameters():
        p.requires_grad_(False)
    n = len(arch.decoder_layers(model))
    points = PT.component_points(n) + [("head_in", i) for i in range(n)]
    source = PT.capture_source(model, batch, points)
    atp = PT.attribution(model, ro, batch, source, n, classes=("final",))
    m_a = PT.margins_of(model, ro, batch.ids_a, batch.mask_a)
    # The last MLP at the final position acts on the margin almost linearly.
    real = PT.sweep(model, ro, batch, source, [{"kind": "mlp", "layer": n - 1, "cls": "final"}])
    est = atp[(atp.kind == "mlp") & (atp.layer == n - 1)]["atp_delta"].to_numpy()
    np.testing.assert_allclose(est, real["m_patched"].to_numpy() - m_a, rtol=0.2, atol=1e-3)
    heads = atp[atp.kind == "head_in"]
    assert heads.groupby("layer")["head"].nunique().tolist() == [4, 4, 4, 4]


def test_summarise_is_a_ratio_of_sums_oriented_per_pair():
    pairs = pd.DataFrame({"pair_uid": ["a", "b"], "candidate_id": ["c1", "c2"],
                          "m_a": [2.0, -1.0], "m_b": [0.0, 1.0]})
    effects = pd.DataFrame({"pair_uid": ["a", "b"], "kind": "mlp", "m_patched": [1.0, 0.0]})
    out = PT.summarise(effects, pairs, by=["kind"])
    # pair a removes 1 of 2, pair b removes 1 of 2 in the opposite direction: 2 / 4.
    assert out["share_removed"].iloc[0] == pytest.approx(0.5)


def test_a_trimmed_sub_batch_patches_like_the_full_batch(model):
    batch, ro = _batch(), _readout(model)
    n = len(arch.decoder_layers(model))
    points = PT.component_points(n)
    source = PT.capture_source(model, batch, points)
    full = PT.sweep(model, ro, batch, source, [{"kind": "mixer", "layer": 3, "cls": "suffix"}])
    for i in range(2):
        sub = PT.subset(batch, [i])
        sub_source = PT.subset_source(source, batch, [i], points)
        alone = PT.sweep(model, ro, sub, sub_source, [{"kind": "mixer", "layer": 3, "cls": "suffix"}])
        assert alone["m_patched"].iloc[0] == pytest.approx(full["m_patched"].iloc[i], abs=1e-4)
        # a source captured on the sub-batch alone agrees with the trimmed one
        fresh = PT.capture_source(model, sub, points)
        assert torch.allclose(fresh[("mixer", 3)], sub_source[("mixer", 3)], atol=1e-4)


def test_token_batches_respect_the_budget():
    groups = PT.token_batches([100, 100, 300, 300, 900, 2000], budget=900)
    assert groups == [[0, 1, 2], [3], [4], [5]]
    assert sorted(i for g in groups for i in g) == list(range(6))
