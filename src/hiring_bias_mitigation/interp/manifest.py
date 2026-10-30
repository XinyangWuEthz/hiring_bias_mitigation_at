"""Contrast pairs: which two variants are compared, fixed before any patching is run.

A contrast is (group, condition, attribute A, attribute B), declared in
`configs/interp/contrasts.yaml`; a contrast pair is one job-CV pair under that contrast. The
contrasts come from the baseline audit (which attributes moved acceptance), never from the
patching results, and selection flags are computed from **base-model margins only**: picking
pairs where SFT happens to have a small effect and then reporting that SFT has a small effect
is the selection loop the mitigation lessons warn about. Every spec is measured on the same
rows.

Flags, all stored, none applied here -- each experiment declares which it uses:

    sel_effect   |dm_base| >= threshold and sign matching the contrast's mean effect
    sel_flip     the base decision differs between A and B
    sel_random   a seeded uniform sample, the control for any selection
    len_match    equal span token length (position-wise span patching allowed)
    aligned      the set passed `segment.validate_set` (or `presence_alignment` for attr_free)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEY = ["pair_id", "protected_group", "protected_attr", "condition"]


def expand_contrasts(spec: dict) -> list[dict]:
    """contrasts.yaml entries -> one dict per (id, group, condition, A, B).

    `b: "*"` expands to every other attribute of the group, and `b: attr_free` makes a
    presence contrast (attribute vs no attribute).
    """
    from ..data.protected_groups import load_group

    out = []
    for c in spec["contrasts"]:
        group, lang = c["group"], spec.get("lang", "en")
        attributes = list(load_group(group, lang).attributes)
        for condition in c["conditions"]:
            bs = [x for x in attributes if x != c["a"]] if c["b"] == "*" else [c["b"]]
            for b in bs:
                if c["a"] not in attributes or (b not in attributes and b != "attr_free"):
                    raise KeyError(f"contrast {c['id']}: {c['a']!r} / {b!r} not in {group}")
                suffix = "" if c["b"] != "*" else "__" + b.lower().replace(" ", "_")
                out.append({"contrast_id": f"{c['id']}_{condition[:4]}{suffix}",
                            "family": c["id"], "protected_group": group, "condition": condition,
                            "attr_a": c["a"], "attr_b": b, "presence": b == "attr_free"})
    return out


def _side(frame: pd.DataFrame, group: str, condition: str, attr: str) -> pd.DataFrame:
    if attr == "attr_free":
        rows = frame[frame["condition"] == "attr_free"]
    else:
        rows = frame[(frame["protected_group"] == group) & (frame["condition"] == condition)
                     & (frame["protected_attr"] == attr)]
    return rows.set_index("pair_id")


def build_manifest(frame: pd.DataFrame, contrasts: list[dict], margins: dict[str, pd.DataFrame],
                   threshold: float = 0.5, random_share: float = 0.3, seed: int = 42
                   ) -> pd.DataFrame:
    """One row per (contrast, pair).

    `frame` carries the grid rows with `prompt_uid`, `candidate_id`, `span_len`, `set_ok` and,
    for attribute rows, `presence_suffix_ok` / `presence_prefix_ok`. `margins` maps a spec
    name to a frame of (prompt_uid, margin_fp32).
    """
    rng = np.random.default_rng(seed)
    parts = []
    for c in contrasts:
        a = _side(frame, c["protected_group"], c["condition"], c["attr_a"])
        b = _side(frame, c["protected_group"], c["condition"], c["attr_b"])
        joined = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        if joined.empty:
            continue
        rows = pd.DataFrame({
            "contrast_id": c["contrast_id"], "family": c["family"],
            "protected_group": c["protected_group"], "condition": c["condition"],
            "attr_a": c["attr_a"], "attr_b": c["attr_b"], "presence": c["presence"],
            "pair_id": joined.index, "candidate_id": joined["candidate_id_a"].to_numpy(),
            "dataset": joined["dataset_a"].to_numpy(),
            "uid_a": joined["prompt_uid_a"].to_numpy(), "uid_b": joined["prompt_uid_b"].to_numpy(),
            "span_len_a": joined["span_len_a"].to_numpy(),
            "span_len_b": joined["span_len_b"].to_numpy(),
        })
        rows["len_match"] = rows["span_len_a"] == rows["span_len_b"]
        if c["presence"]:
            rows["aligned"] = joined["presence_suffix_ok_a"].to_numpy()
            rows["prefix_aligned"] = joined["presence_prefix_ok_a"].to_numpy()
        else:
            rows["aligned"] = (joined["set_ok_a"] & joined["set_ok_b"]).to_numpy()
            rows["prefix_aligned"] = rows["aligned"]
        rows["sel_random"] = rng.random(len(rows)) < random_share
        parts.append(rows.reset_index(drop=True))
    if not parts:
        raise ValueError("no contrast matched any rows; check contrasts.yaml against the grid")
    manifest = pd.concat(parts, ignore_index=True)

    for spec, m in margins.items():
        lookup = m.set_index("prompt_uid")["margin_fp32"]
        manifest[f"m_a_{spec}"] = manifest["uid_a"].map(lookup)
        manifest[f"m_b_{spec}"] = manifest["uid_b"].map(lookup)
        manifest[f"dm_{spec}"] = manifest[f"m_a_{spec}"] - manifest[f"m_b_{spec}"]

    if "base" in margins:
        sign = np.sign(manifest.groupby("contrast_id")["dm_base"].transform("mean"))
        manifest["sel_effect"] = ((manifest["dm_base"].abs() >= threshold)
                                  & (np.sign(manifest["dm_base"]) == sign))
        manifest["sel_flip"] = (manifest["m_a_base"] > 0) != (manifest["m_b_base"] > 0)
    manifest["pair_uid"] = manifest["contrast_id"] + "::" + manifest["pair_id"]
    return manifest

