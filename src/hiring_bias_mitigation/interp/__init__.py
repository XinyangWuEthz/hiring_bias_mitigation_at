"""Mechanistic analysis: which components carry the attribute into the hiring decision.

The audit and the mitigation arms measure *that* a decision depends on a protected attribute,
and that SFT/DPO reduce the dependence. This package asks *how*: activation patching between
counterfactual prompts that differ only in the injected attribute, on Qwen3.5-9B, to recover a
head-level circuit in the base model and to show what each fine-tune changed in it. The plan,
hypotheses and pass criteria are in `reports/interp/PLAN.md`.

    io.py        artifact paths under outputs/interp/, metadata sidecars
    render.py    the evaluation grid and the pool-derived discovery grid, chat-wrapped
    segment.py   token segments of a prompt and the position map between two variants
    arch.py      Qwen3.5 hook points and per-head slices (the only file naming modules)
    hooks.py     capture and patch activations at those hook points
    spec.py      base / SFT / DPO as one PEFT model with switchable adapters
    extract.py   a LoRA adapter recovered from a merged release (SFT-v3 ships merged only)
    margins.py   the decision margin read in fp32 from the pre-norm residual
    manifest.py  contrast pairs chosen in advance, with selection flags and splits

The data pipeline runs as `scripts/interp/00_env_check.py` ... `04_manifest.py`.
"""
