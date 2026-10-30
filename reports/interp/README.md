# Activation patching: Qwen3.5-9B, English, base vs SFT vs DPO

The spec `sft` is **SFT-v3**, the parent of every DPO and GRPO arm. Its adapter is recovered
from the merged release, and the first round's SFT v1 is the spec `sft_v1`. Setup, gates and
results are in [`docs/INTERP_SFT_V3.md`](../../docs/INTERP_SFT_V3.md). The pilot table further
down is from the first round (SFT v1).

The plan, the hypotheses and their pass/fail criteria are in [`PLAN.md`](PLAN.md). This page
is the run guide for the data pipeline (plan sections D0–D4), which is implemented. The
patching experiments (H1–H11) build on its artifacts.

| Where | What |
|---|---|
| `src/hiring_bias_mitigation/interp/` | Library: render, segment, arch, hooks, spec, margins, manifest |
| `src/hiring_bias_mitigation/eval/decision.py` | Shared decision prefix and token ids (also used by `diagnose_adapter.py` and `operating_point_control.py`) |
| `scripts/interp/` | Numbered CLIs, each idempotent |
| `configs/interp/` | `data.yaml` (grid and pool), `models.yaml` (specs), `contrasts.yaml` |
| `tests/interp/` | No-GPU tests: hook identities on a tiny random Qwen3.5, segmentation, manifest |
| `$HBM_OUTPUT_ROOT/outputs/interp/` | Heavy artifacts (margins, segments, manifests), each with a `.meta.json` |
| `reports/interp/` | Small summaries written by the scripts |

## Run order

```bash
export HBM_OUTPUT_ROOT=/home/samoilenko/hbm-local   # holds the .hbm-models-root marker
# GDN kernels. --no-deps keeps the torch/transformers/vLLM pins; fla-core holds the kernels.
uv pip install --no-deps flash-linear-attention fla-core

uv run python scripts/interp/00_env_check.py         # refuses unmounted paths, records kernels
uv run python scripts/interp/01_fetch_data.py        # model, SFT-v1 adapter, SFT-v3 release, pool, audit runs
uv run python scripts/interp/01b_extract_adapter.py --stage extract            # SFT-v3 adapter from the release
uv run python scripts/interp/01b_extract_adapter.py --stage check --device cuda:1   # against the release
uv run python scripts/interp/03_segments.py --dataset grid    # CPU, ~1 min
uv run python scripts/interp/03_segments.py --dataset pool

# one spec per GPU; --resid also stores the last-position residual of every layer
uv run python scripts/interp/02_margins.py --spec base --dataset grid --device cuda:1 --resid
uv run python scripts/interp/02_margins.py --spec sft  --dataset grid --device cuda:2 --resid
uv run python scripts/interp/02_margins.py --spec base --dataset pool --device cuda:1 --resid
uv run python scripts/interp/02_margins.py --spec dpo  --dataset grid --adapter dpo=/path/to/dpo

uv run python scripts/interp/04_manifest.py --dataset pool    # discovery pairs (base only)
uv run python scripts/interp/04_manifest.py --dataset grid    # confirmation, SFT/DPO
```

Pass `--limit 64` to `02_margins.py` for a pilot. It writes to its own `...-limit64` files.

## The datasets

| Dataset | Rows | Pairs | Used for |
|---|---:|---:|---|
| `grid` | 31,050 | 450 benchmark job–CV pairs × 34 attributes × {explicit, implicit} + attr_free | Confirming circuits; every SFT/DPO comparison |
| `pool` | 20,700 | 300 training-pool pairs, one per candidate, benchmark-disjoint, same grid | Discovering circuits, base model only. SFT trained on the pool, so it is never used to compare fine-tunes |

## Segmentation facts (measured)

- 0 hard boundary violations in either dataset. All 2,700 grid and 1,800 pool attribute sets
  are identical before the span and, aligned from the end, after it.
- Most CVs start and end with a newline. BPE merges these with the neighbouring separator into
  a single `"\n\n"` token. Between attribute variants that token is identical, so alignment
  holds. Against the attribute-free prompt, the prefix differs by that one token in 58% of
  grid pairs (47% of pool pairs). Presence contrasts are therefore patched on suffix positions
  only (`aligned`), and `prefix_aligned` marks the pairs where the prefix matches as well.
- Prompt length: median 719 tokens, p95 1,190, maximum 1,851 (grid). The pool maximum is 2,369.

## Pilot (512 grid rows per spec, RTX 6000 Ada, fla kernels, batch 16)

| | base | SFT |
|---|---:|---:|
| Decision agreement with the audit's greedy decisions, same rows | 99.2% | 99.0% |
| Hire rate, HF vs audit | 12.7% vs 13.5% | 22.3% vs 23.2% |
| Mean \|bf16 margin − fp32 margin\| (logits) | 0.040 | 0.044 |
| LoRA modules attached | — | 128 (32 × 3 MLP + 8 × 4 attention) |
| Throughput, steady state | 0.16 s/row | 0.16 s/row |

A full grid pass is about 1.4 GPU-h per spec, and the pool about 0.9. The bf16 readout error
(0.04 logits) is about a sixth of SFT's whole effect on the per-set margin spread (0.26),
which is why every margin here is read in fp32.

## Inputs still owed by the DPO team

The DPO-v3 release (`Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-dpo-v3-4880-en`) is merged too.
Its own update could be recovered the same way, as the SVD of DPO-v3 merged minus SFT-v3 merged.
That would fit the spec `[sft, dpo]`. `01b_extract_adapter.py` takes its base from
`models.yaml`, so it needs a `--base` option first. Not done yet.

- The adapter directory.
- Whether it was trained on top of the merged SFT weights (spec `[sft, dpo]`, the default in
  `models.yaml`) or continued the SFT adapter itself (spec `[dpo]`).
- The 2,362 + 295 preference pairs, for H10.
