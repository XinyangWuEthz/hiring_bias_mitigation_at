"""Reading the decision as a number instead of a generated word.

Every model here answers `{"decision": "<word>", ...}`. Forcing the prefix `{"decision": "`
and comparing the next-token scores of the two decision words gives the model's decision as
a continuous margin in a single forward pass, with no generation. Three tools read it this
way -- the adapter diagnostic, the operating-point control and the interpretability pipeline
-- and they must agree on the prefix and on which token stands for which word, so both live
here rather than in each script.
"""

from __future__ import annotations

PREFIX = '{"decision": "'

#: The words the screening prompt asks for, per language (`data/prompts.FORMAT_INSTRUCTIONS`).
WORDS: dict[str, tuple[str, str]] = {"en": ("hire", "reject"), "uk": ("найняти", "відхилити")}


def decision_token_ids(tokenizer, words: str | tuple[str, str] = "en") -> tuple[int, int]:
    """The first token of each decision word as it follows the forced prefix: (hire, reject).

    `words` is a language code or an explicit (hire_word, reject_word) pair -- the latter for
    a model that answers in other words than its prompt asks for (see
    `scripts/operating_point_control.decision_words`).

    The words are tokenised *after* the prefix, not alone: BPE merges across the quote, so
    the standalone tokenisation of "hire" is not the token the model emits there. Both
    failure modes are refused rather than worked around -- a prefix that tokenises differently
    once a word follows it, and two words sharing a first token -- because either would make
    the margin compare the wrong logits while every downstream number still looked plausible.
    """
    pair = WORDS[words] if isinstance(words, str) else tuple(words)
    base = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    ids = []
    for word in pair:
        full = tokenizer(PREFIX + word, add_special_tokens=False)["input_ids"]
        if full[: len(base)] != base:
            raise ValueError(f"the prefix tokenises differently before {word!r}")
        ids.append(full[len(base)])
    if ids[0] == ids[1]:
        raise ValueError(f"{pair[0]!r} and {pair[1]!r} share a first token")
    return ids[0], ids[1]
