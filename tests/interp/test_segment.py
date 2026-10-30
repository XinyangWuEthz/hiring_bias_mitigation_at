"""Segmentation, set validation, the position map, the decision tokens and the manifest.

Most tests use a character-level stub tokenizer, so they run anywhere; one group runs on the
real Qwen3.5 tokenizer when it is in the local cache, because the failure that matters --
BPE merging across a segment boundary -- only exists in a real vocabulary.
"""

import numpy as np
import pandas as pd
import pytest

from hiring_bias_mitigation.data import benchmark as B
from hiring_bias_mitigation.data import injection as I
from hiring_bias_mitigation.data import prompts as P
from hiring_bias_mitigation.eval.decision import decision_token_ids
from hiring_bias_mitigation.interp import manifest as MF
from hiring_bias_mitigation.interp import render
from hiring_bias_mitigation.interp import segment as S


class CharTokenizer:
    """One token per character; optional merges to imitate BPE across boundaries."""

    chat_template = None  # _chat_wrap then passes the prompt through unchanged

    def __init__(self, merges: tuple[str, ...] = ()):
        self.merges = merges

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, i = [], [], 0
        while i < len(text):
            width = next((len(m) for m in self.merges if text.startswith(m, i)), 1)
            ids.append(hash(text[i:i + width]) % 50_000)
            offsets.append((i, i + width))
            i += width
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


@pytest.fixture(scope="module")
def grid():
    pairs = B.load_benchmark("en").head(3)
    return I.build_eval_set(pairs, ["military_status", "religion"], "en").assign(dataset="grid")


def _row(grid, condition, attr):
    sel = grid[(grid["condition"] == condition) & (grid["protected_attr"] == attr)]
    return sel.iloc[0].to_dict()


@pytest.mark.parametrize("condition,attr", [("explicit", "Civilian"),
                                            ("implicit", "Participant in combat actions"),
                                            ("attr_free", I.ATTR_FREE)])
def test_pieces_reproduce_build_prompt(grid, condition, attr):
    row = _row(grid, condition, attr)
    assert "".join(t for _, t in S.prompt_pieces(row)) == P.build_prompt(row)


def test_char_segmentation_covers_every_token(grid):
    seg = S.segment(CharTokenizer(), _row(grid, "explicit", "War veteran"))
    assert not seg.violations
    assert seg.bounds["chat_open"][0] == 0 and seg.bounds["forced_prefix"][1] == seg.n_tokens
    ends = [seg.bounds[s][1] for s in S.SEGMENTS[:-1]]
    starts = [seg.bounds[s][0] for s in S.SEGMENTS[1:]]
    assert ends == starts, "segments must tile the sequence"
    assert seg.span_len() == len("Candidate's military status: War veteran")
    label, value = seg.span_parts["span_label"], seg.span_parts["span_value"]
    assert value[1] - value[0] == len(" War veteran") and label[1] == value[0]


def test_attr_free_has_an_empty_span(grid):
    seg = S.segment(CharTokenizer(), _row(grid, "attr_free", I.ATTR_FREE))
    assert seg.span_len() == 0 and seg.bounds["span_sep"][0] == seg.bounds["span_sep"][1]


def test_a_merge_across_the_span_is_a_violation(grid):
    # "n\n" glues the last character of "War veteran" to the separator: the token is not
    # whitespace, so the suffix would carry attribute text.
    seg = S.segment(CharTokenizer(merges=("n\n",)), _row(grid, "explicit", "War veteran"))
    assert any("crosses the end of span" in v for v in seg.violations)


def test_a_whitespace_merge_is_recorded_not_fatal(grid):
    row = {**_row(grid, "explicit", "War veteran"), "cv": "\nfirst line of a CV"}
    seg = S.segment(CharTokenizer(merges=("\n\n",)), row)
    assert not seg.violations and any("span_sep" in m for m in seg.merges)


def test_set_validation_and_end_alignment(grid):
    tok = CharTokenizer()
    rows = grid[(grid["pair_id"] == grid["pair_id"].iloc[0]) & (grid["condition"] == "implicit")
                & (grid["protected_group"] == "military_status")]
    segs = [S.segment(tok, r) for r in rows.to_dict("records")]
    assert S.validate_set(segs) == []
    a, b = segs[0], segs[-1]
    assert a.span_len() != b.span_len()
    for cls in ("span_sep", "post_span3", "cv", "instruction", "chat_close", "final"):
        pa, pb = S.align(a, b, cls)
        assert [a.input_ids[i] for i in pa] == [b.input_ids[i] for i in pb]
    with pytest.raises(ValueError):
        S.align(a, b, "span")


def test_set_validation_catches_a_broken_variant(grid):
    tok = CharTokenizer()
    rows = grid[(grid["pair_id"] == grid["pair_id"].iloc[0]) & (grid["condition"] == "explicit")
                & (grid["protected_group"] == "religion")].to_dict("records")
    rows[1] = {**rows[1], "cv": rows[1]["cv"] + " (edited)"}
    assert S.validate_set([S.segment(tok, r) for r in rows])


def test_segmentation_round_trips_through_a_record(grid):
    seg = S.segment(CharTokenizer(), _row(grid, "explicit", "Atheist"))
    back = S.Segmentation.from_record(seg.to_record())
    assert back.bounds == seg.bounds and back.input_ids == seg.input_ids
    assert back.span_parts == seg.span_parts


# ---- decision tokens -------------------------------------------------------------------

class WordTokenizer:
    def __init__(self, table):
        self.table = table

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.table[text]}


def test_decision_token_ids():
    tok = WordTokenizer({'{"decision": "': [1, 2], '{"decision": "hire': [1, 2, 7],
                         '{"decision": "reject': [1, 2, 9, 4]})
    assert decision_token_ids(tok, "en") == (7, 9)


def test_decision_token_ids_refuses_a_prefix_that_retokenises():
    tok = WordTokenizer({'{"decision": "': [1, 2], '{"decision": "hire': [1, 3],
                         '{"decision": "reject': [1, 2, 9]})
    with pytest.raises(ValueError, match="tokenises differently"):
        decision_token_ids(tok, "en")


def test_decision_token_ids_refuses_a_shared_first_token():
    tok = WordTokenizer({'{"decision": "': [1], '{"decision": "hire': [1, 5, 6],
                         '{"decision": "reject': [1, 5, 8]})
    with pytest.raises(ValueError, match="share a first token"):
        decision_token_ids(tok, "en")


# ---- manifest --------------------------------------------------------------------------

def test_manifest_pairs_contrasts_and_selects_on_base_only(grid):
    tok = CharTokenizer()
    frame = render.render(grid, tok)
    segs = S.segment_frame(frame, tok)
    rng = np.random.default_rng(0)
    margins = {s: pd.DataFrame({"prompt_uid": frame["prompt_uid"],
                                "margin_fp32": rng.normal(size=len(frame))})
               for s in ("base", "sft")}
    contrasts = MF.expand_contrasts({"lang": "en", "contrasts": [
        {"id": "mil", "group": "military_status", "a": "Participant in combat actions",
         "b": "Civilian", "conditions": ["implicit"]},
        {"id": "rel", "group": "religion", "a": "Atheist", "b": "*", "conditions": ["explicit"]},
        {"id": "pres", "group": "military_status", "a": "Civilian", "b": "attr_free",
         "conditions": ["explicit"]},
    ]})
    assert len(contrasts) == 1 + 8 + 1
    manifest = MF.build_manifest(segs, contrasts, margins, threshold=0.0)
    assert manifest.groupby("contrast_id").size().eq(3).all(), "one row per pair per contrast"
    assert manifest["aligned"].all()
    row = manifest[manifest["contrast_id"] == "mil_impl"].iloc[0]
    m = margins["base"].set_index("prompt_uid")["margin_fp32"]
    assert row["dm_base"] == pytest.approx(m[row["uid_a"]] - m[row["uid_b"]])
    assert {"dm_sft", "sel_effect", "sel_flip", "sel_random"} <= set(manifest.columns)
    again = MF.build_manifest(segs, contrasts, margins, threshold=0.0)
    pd.testing.assert_frame_equal(manifest, again)


# ---- the real tokenizer ------------------------------------------------------------------

def _real_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", local_files_only=True)
    except Exception:
        return None


@pytest.mark.skipif(_real_tokenizer() is None, reason="Qwen3.5 tokenizer not cached")
def test_real_tokenizer_layout_holds_on_benchmark_rows(grid):
    tok = _real_tokenizer()
    frame = render.render(grid, tok, {"enable_thinking": False})
    segs = S.segment_frame(frame, tok, {"enable_thinking": False})
    assert (segs["violations"] == "").all()
    assert segs["set_ok"].all()
    row = segs[segs["condition"] == "explicit"].iloc[0]
    seg = S.Segmentation.from_record(row.to_dict())
    span_text = tok.decode(seg.input_ids[slice(*seg.bounds["span"])])
    assert span_text.startswith("Candidate's ")
    assert tok.decode(seg.input_ids[slice(*seg.bounds["forced_prefix"])]) == '{"decision": "'
