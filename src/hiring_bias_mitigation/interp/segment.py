"""Token segments of a screening prompt, and the position map between two of its variants.

Activation patching compares a model on two inputs position by position. Here the two inputs
are attribute variants of one counterfactual set, and they are *not* the same length: "Civilian"
and "Participant in combat actions" tokenise to different counts. What makes patching possible
anyway is the prompt's layout (`data/prompts.build_prompt`):

    [chat_open][header][job][profile_open] [span] [span_sep][cv][profile_close][body][format]
    [chat_close][forced_prefix]
     ------------------ prefix -----------  attr  ---------------- suffix -------------------

Everything before the span is identical text, so its tokens are identical and -- the model
being causal -- so are its activations: patching there is a null control, not an experiment.
Everything after the span is identical text too, so its tokens line up *from the end*: suffix
position p in variant A is position p - (len_A - len_B) in variant B. The span itself is
mapped only between variants of equal token length; otherwise it is patched as a block or
not at all.

All of that holds only if tokenisation respects the segment boundaries -- a BPE merge across
`value\\n` would put an attribute-dependent token in the "identical" suffix and every
suffix-aligned patch would silently compare different tokens. So segmentation asserts the
layout instead of assuming it, and a set whose variants break it is reported, not patched.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from ..data import prompts as P
from ..eval.backends import _chat_wrap
from ..eval.decision import PREFIX

SEGMENTS = ("chat_open", "header", "job", "profile_open", "span", "span_sep", "cv",
            "profile_close", "body", "format", "chat_close", "forced_prefix")
PREFIX_SEGMENTS = SEGMENTS[:4]
SUFFIX_SEGMENTS = SEGMENTS[5:]

#: Tokens of the CV inside the GDN causal-conv window (kernel 4) of the span's last token: a
#: patch at the span leaks into these three positions through the conv before any mixing.
CONV_WINDOW = 3

#: Position classes the sweeps patch over. `post_span3` and `final` are carved out of their
#: segments; every other class is a union of whole segments.
POSITION_CLASSES = ("prefix", "span", "span_sep", "post_span3", "cv", "instruction",
                    "chat_close", "forced_prefix", "final")
_CLASS_SEGMENTS = {
    "prefix": PREFIX_SEGMENTS,
    "span": ("span",),
    "span_sep": ("span_sep",),
    "instruction": ("profile_close", "body", "format"),
    "chat_close": ("chat_close",),
}


def prompt_pieces(row: dict) -> list[tuple[str, str]]:
    """The baseline user prompt as named pieces whose concatenation *is* `build_prompt(row)`.

    Rebuilt from the same constants `build_prompt` uses and checked against it, so a change to
    the prompt template fails here loudly instead of shifting every boundary.
    """
    lang = row["lang"]
    cv = row["cv"]
    profile = P.build_profile(row)
    if profile == cv:
        span = ""
    else:
        if not profile.endswith("\n" + cv):
            raise ValueError("the profile block no longer ends with the bare CV")
        span = profile[: -len(cv) - 1]
    fmt = P.format_instructions(lang, "baseline")
    pieces = [
        ("header", f"{P._HEADER[lang]}\n\n"),
        ("job", f"{P._JOB_LABEL[lang]}: ```\n{row['job_description']}\n```\n\n"),
        ("profile_open", f"{P._PROFILE_LABEL[lang]}: ```\n"),
        ("span", span),
        ("span_sep", "\n" if span else ""),
        ("cv", cv),
        ("profile_close", "\n```\n\n"),
        ("body", P.STRATEGY_BODIES["baseline"][lang]),
        ("format", f" {P._FORMAT_LEAD[lang]} ```\n{fmt}\n```"),
    ]
    joined = "".join(text for _, text in pieces)
    if joined != P.build_prompt(row, "baseline"):
        raise AssertionError("prompt_pieces no longer reproduces build_prompt; update it")
    return pieces


@dataclass
class Segmentation:
    """Token ids of one model input and the [start, end) token range of every segment."""

    input_ids: list[int]
    bounds: dict[str, tuple[int, int]]
    #: explicit condition only: the label ("Candidate's religion: ") and the value ("Atheist")
    span_parts: dict[str, tuple[int, int]] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    #: whitespace-only tokens straddling a boundary (see `segment`); not violations
    merges: list[str] = field(default_factory=list)

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    def span_len(self) -> int:
        start, end = self.bounds["span"]
        return end - start

    def positions(self, cls: str) -> np.ndarray:
        """Token positions of one position class."""
        if cls in _CLASS_SEGMENTS:
            idx = [np.arange(*self.bounds[s]) for s in _CLASS_SEGMENTS[cls]]
            return np.concatenate(idx) if idx else np.arange(0)
        cv_start, cv_end = self.bounds["cv"]
        if cls == "post_span3":
            return np.arange(cv_start, min(cv_start + CONV_WINDOW, cv_end))
        if cls == "cv":
            return np.arange(min(cv_start + CONV_WINDOW, cv_end), cv_end)
        fp_start, fp_end = self.bounds["forced_prefix"]
        if cls == "forced_prefix":
            return np.arange(fp_start, fp_end - 1)
        if cls == "final":
            return np.array([fp_end - 1])
        raise KeyError(f"unknown position class {cls!r}; expected one of {POSITION_CLASSES}")

    def suffix_start(self) -> int:
        """First token aligned from the end: the span separator, or the CV when there is none."""
        return self.bounds["span_sep"][0]

    def to_record(self) -> dict:
        rec = {"input_ids": np.asarray(self.input_ids, dtype=np.int32), "n_tokens": self.n_tokens,
               "violations": "; ".join(self.violations), "merges": "; ".join(self.merges)}
        for name in SEGMENTS:
            rec[f"{name}_start"], rec[f"{name}_end"] = self.bounds[name]
        for name in ("span_label", "span_value"):
            rec[f"{name}_start"], rec[f"{name}_end"] = self.span_parts.get(name, (-1, -1))
        return rec

    @classmethod
    def from_record(cls, rec: dict) -> Segmentation:
        bounds = {name: (int(rec[f"{name}_start"]), int(rec[f"{name}_end"])) for name in SEGMENTS}
        parts = {name: (int(rec[f"{name}_start"]), int(rec[f"{name}_end"]))
                 for name in ("span_label", "span_value") if int(rec[f"{name}_start"]) >= 0}
        violations = [v for v in str(rec.get("violations") or "").split("; ") if v]
        merges = [v for v in str(rec.get("merges") or "").split("; ") if v]
        return cls(list(map(int, rec["input_ids"])), bounds, parts, violations, merges)


def char_bounds(tokenizer, row: dict, chat_template_kwargs: dict | None = None
                ) -> tuple[str, list[tuple[str, int, int]]]:
    """The full model input and the [start, end) character range of every segment."""
    pieces = prompt_pieces(row)
    prompt = "".join(text for _, text in pieces)
    chat = _chat_wrap(tokenizer, prompt, chat_template_kwargs)
    start = chat.find(prompt)
    if start < 0 or chat.find(prompt, start + 1) >= 0:
        raise AssertionError("the user prompt does not occur exactly once in the chat render")
    full = chat + PREFIX

    spans = [("chat_open", 0, start)]
    cursor = start
    for name, text in pieces:
        spans.append((name, cursor, cursor + len(text)))
        cursor += len(text)
    spans.append(("chat_close", cursor, len(chat)))
    spans.append(("forced_prefix", len(chat), len(full)))
    return full, spans


def segment(tokenizer, row: dict, chat_template_kwargs: dict | None = None) -> Segmentation:
    """Tokenises one row and assigns every token to the segment its first character is in.

    A token that starts in one segment and ends in another is a boundary violation. It is
    recorded rather than raised, so a whole dataset can be segmented and its violations
    counted; `validate_set` and the segment script decide what is fatal.
    """
    full, spans = char_bounds(tokenizer, row, chat_template_kwargs)
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    starts = [s for _, s, _ in spans]
    violations: list[str] = []
    merges: list[str] = []

    owner = []
    for i, (cs, ce) in enumerate(offsets):
        # An empty segment shares its start with the next one, and bisect_right lands on the
        # last segment starting at or before the token -- always the non-empty one.
        k = bisect.bisect_right(starts, cs) - 1
        name, _, seg_end = spans[k]
        if ce > seg_end:
            note = f"token {i} {full[cs:ce]!r} crosses the end of {name}"
            # A CV that starts with a newline merges with the separator before it into one
            # "\n\n" token. That token is the same in every variant carrying a span, so the
            # suffix stays aligned; it only breaks prefix alignment against attr_free, which
            # `presence_alignable` checks. Any other crossing is a real violation.
            (merges if not full[cs:ce].strip() else violations).append(note)
        owner.append(k)

    bounds = {}
    for k, (name, cs, ce) in enumerate(spans):
        members = [i for i, o in enumerate(owner) if o == k]
        if members:
            bounds[name] = (members[0], members[-1] + 1)
        else:
            # Empty: placed at the first token whose text starts at or after the segment.
            at = next((i for i, (s, _) in enumerate(offsets) if s >= cs), len(ids))
            bounds[name] = (at, at)
            if cs != ce:
                violations.append(f"segment {name} has text but no token of its own")

    seg = Segmentation(ids, bounds, {}, violations, merges)
    if bounds["forced_prefix"][1] != len(ids):
        seg.violations.append("the last token is not the end of the forced prefix")
    if row["condition"] == "explicit" and seg.span_len():
        seg.span_parts = _explicit_parts(full, spans, offsets, bounds["span"])
    return seg


def _explicit_parts(full, spans, offsets, span_bounds) -> dict[str, tuple[int, int]]:
    """Splits an explicit span at the label/value colon ("Candidate's religion: | Atheist")."""
    _, cs, ce = next(s for s in spans if s[0] == "span")
    colon = full.rfind(": ", cs, ce)
    if colon < 0:
        return {}
    start, end = span_bounds
    # The value's first token carries the space after the colon (" Atheist").
    split = next((i for i in range(start, end) if offsets[i][0] > colon), end)
    return {"span_label": (start, split), "span_value": (split, end)}


def validate_set(segs: list[Segmentation]) -> list[str]:
    """Checks the layout assumption across the attribute variants of one counterfactual set.

    Pre-span tokens must be identical from the start and post-span tokens identical from the
    end, and every variant must carry a span. The attribute-free variant is checked separately
    (`presence_alignable`), because its layout differs by construction. Returns violations,
    empty when the set's attribute contrasts can be patched.
    """
    problems = [v for s in segs for v in s.violations]
    if any(not s.span_len() for s in segs):
        problems.append("a variant in the set has no attribute span")
        return problems
    ref = segs[0]
    pre = ref.input_ids[: ref.bounds["span"][0]]
    tail = ref.input_ids[ref.suffix_start():]
    for i, s in enumerate(segs):
        if s.input_ids[: s.bounds["span"][0]] != pre:
            problems.append(f"variant {i}: prefix tokens differ")
        if s.input_ids[s.suffix_start():] != tail:
            problems.append(f"variant {i}: tokens from the span separator onward differ")
    return problems


def presence_alignment(attr: Segmentation, free: Segmentation) -> dict[str, bool]:
    """How an attribute variant lines up with the attribute-free prompt of the same pair.

    `suffix`: tokens from the CV onward are identical, so suffix-aligned patching (every
    position class after the span) is valid. `prefix`: the prefixes are identical too. Most
    benchmark CVs start with a newline, which the attribute-free prompt merges into its last
    prefix token -- the prefixes then differ by that one token, which is part of what the
    presence contrast changes rather than an alignment error.
    """
    clean = not attr.violations and not free.violations
    return {
        "suffix": clean and attr.input_ids[attr.bounds["cv"][0]:]
        == free.input_ids[free.bounds["cv"][0]:],
        "prefix": clean and attr.input_ids[: attr.bounds["span"][0]]
        == free.input_ids[: free.bounds["span"][0]],
    }


def align(a: Segmentation, b: Segmentation, cls: str) -> tuple[np.ndarray, np.ndarray]:
    """Matching positions of one class in two variants of the same set: (positions in a, in b).

    Prefix classes map one-to-one from the start, suffix classes from the end. The span maps
    only between equal-length spans, and the separator only when both variants have one.
    """
    pa, pb = a.positions(cls), b.positions(cls)
    if cls == "span" and a.span_len() != b.span_len():
        raise ValueError("spans of different token length have no position-wise map")
    if len(pa) != len(pb):
        raise ValueError(f"class {cls!r} has {len(pa)} positions in one variant, {len(pb)} in "
                         "the other; the variants do not share the layout")
    if cls != "prefix" and cls != "span" and len(pa):
        shift = a.n_tokens - b.n_tokens
        if not np.array_equal(pa - shift, pb):
            raise AssertionError(f"class {cls!r} is not aligned from the end")
    return pa, pb


def segment_frame(frame, tokenizer, chat_template_kwargs: dict | None = None):
    """Segments every rendered row and runs the set-level checks; one output row per input.

    Adds `span_len`, `set_ok` (the row's attribute set passed `validate_set`) and, for
    attribute rows, `presence_suffix_ok` / `presence_prefix_ok` against the pair's
    attribute-free prompt. `prompt_uid` is recomputed from the segmented text and must equal
    the rendered one, so segments and margins provably describe the same model input.
    """
    import pandas as pd

    from ..data.injection import ATTR_FREE
    from .render import prompt_uid

    records, segs = [], []
    for row in frame.to_dict("records"):
        full, _ = char_bounds(tokenizer, row, chat_template_kwargs)
        if "prompt_uid" in row and prompt_uid(full) != row["prompt_uid"]:
            raise AssertionError(f"segmented text differs from the rendered prompt for {row}")
        seg = segment(tokenizer, row, chat_template_kwargs)
        segs.append(seg)
        records.append({**{k: row[k] for k in ("dataset", "pair_id", "candidate_id", "job_id",
                                                "protected_group", "protected_attr",
                                                "condition") if k in row},
                        "prompt_uid": prompt_uid(full), "span_len": seg.span_len(),
                        **seg.to_record()})
    out = pd.DataFrame(records)
    attr = out["protected_attr"] != ATTR_FREE
    out["set_ok"] = True
    out["set_problems"] = ""
    for _, idx in out[attr].groupby(["pair_id", "protected_group", "condition"]).groups.items():
        problems = validate_set([segs[i] for i in idx])
        out.loc[idx, "set_ok"] = not problems
        out.loc[idx, "set_problems"] = "; ".join(problems[:3])
    free = {pid: segs[i] for i, pid in zip(out.index[~attr], out.loc[~attr, "pair_id"])}
    suffix_ok, prefix_ok = [], []
    for i, row in out.iterrows():
        if not attr[i] or row["pair_id"] not in free:
            suffix_ok.append(True)
            prefix_ok.append(True)
            continue
        flags = presence_alignment(segs[i], free[row["pair_id"]])
        suffix_ok.append(flags["suffix"])
        prefix_ok.append(flags["prefix"])
    out["presence_suffix_ok"] = suffix_ok
    out["presence_prefix_ok"] = prefix_ok
    return out
