# Activation patching on SFT-v3

The first round of the activation-patching analysis (`reports/interp/`) compared the base model
with SFT v1 (`Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en`). Every later arm (DPO-v3,
GRPO, DAPO) starts from SFT-v3 (`docs/EVAL_STANDARD.md`), so this round reruns the SFT side of
the analysis on SFT-v3. Base-model artifacts do not depend on the SFT adapter and are reused.

## Setup

**Code.** `main` at `33b1dd0`, with uncommitted changes:
`configs/interp/models.yaml`, `scripts/interp/01_fetch_data.py`, `02_margins.py`, the new
`01b_extract_adapter.py`, `01c_merge_gap.py`, `interp/extract.py` and
`tests/interp/test_extract.py`. Every artifact's `.meta.json` records `git_sha` 33b1dd0; the
diff is not recorded there.

Written on 2026-10-27, before any SFT-v3 margin was computed.

- **Question.** Does the analysis read the same on SFT-v3 as on v1? That covers the
  audit agreement, how SFT changes per-set margins, and where patching locates the
  attribute's effect on the three patched contrasts. This round is a rerun, not a new
  hypothesis.
- **Model.** `Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-v3-en` at revision `b6e5f2b`: the
  final adapter of job 5107673 (step 3,923, LoRA r 32 / α 64), released **merged** only. The
  pipeline switches adapters over one base model (`interp/spec.py`), so the adapter is recovered
  as the rank-32 SVD of `merged − base` on each LoRA target module
  (`scripts/interp/01b_extract_adapter.py`, `interp/extract.py`). It is the spec `sft`; v1 is now
  the spec `sft_v1`, with its artifacts renamed, not recomputed.
- **Control.** The base spec, with its existing grid and pool margins, manifests and patching runs.
  SFT-v1 (`sft_v1`) is the comparison for "does it read the same". All comparisons use
  the same rows: the full 31,050-prompt grid, and for patching, the same 64 pairs per contrast,
  selected on base margins only (`10_patch.py`).
- **Expected observation.** The recovered adapter reproduces the merged release (gate S1). The HF
  hire rate on the grid lies near the audit's 22.75%, and within-group instability near its
  1.85% (the release's `evaluation.json`). Whether the patching picture matches v1's is the open
  question. Neither outcome counts against anything.
- **Stop conditions.**
  - S1: the check of the recovered adapter against the merged release, on 256 seeded grid rows,
    must have at least 99% decision agreement and a mean |margin difference| of at most
    0.1 logits.
  - S2: the recovered adapter must have 128 LoRA modules, as v1 has.
  - S3: the full-grid HF hire rate must be within 2 pp of the audit's 22.75%. For scale, v1 was
    0.44 pp off and base 0.68 pp.
  - If any gate fails, nothing downstream runs.
- **Splits.** The grid is the benchmark and is only read here, not tuned on: no setting of
  this round is chosen from grid numbers. The pool is used for discovery, on the base model
  only. SFT-v3 trained on the v1 swap rows as well, so the pool is not held out from it either.

## Result

### Recovering the adapter

From `reports/interp/adapter_extract-sft-v3-svd32.json`, log
`outputs/interp/logs/extract-sft-v3.log`:

- The first extraction refused. 48 GDN tensors (`linear_attn.A_log` and
  `linear_attn.norm.weight` of the 24 GDN layers) differ from the base. Each one equals the
  base's fp32 value cast to bf16, bit for bit, so the release stores them at lower precision
  and they were not trained. The gate now accepts exactly this cast and lists the tensors under
  `stored_at_lower_precision`. Any other difference still refuses. No Qwen3.5 module is kept in
  fp32 at load time, so a bf16 load gives these tensors the same values in the pipeline as in
  the release.
- 128 LoRA modules were recovered (S2 passes). No tensor changed outside the targets.
- Rank 32 fits. Per module, the 33rd singular value is a median 2.7% of the 32nd (maximum
  24.5%). The rank-32 approximation keeps a median 99.78% of `‖merged − base‖²` (minimum 98.9%).
- Adapter fingerprint `43a92ece01c40b00`, at
  `$HBM_OUTPUT_ROOT/outputs/interp/adapters/sft-v3-svd32/`.

### Gate S1: missed on the letter, run continued on the user's decision

From `reports/interp/adapter_check-sft-v3-svd32.json` and
`reports/interp/adapter_merge_gap-sft-v3-svd32.json` (`scripts/interp/01c_merge_gap.py`). Both
use the same 256 seeded grid rows; per-row margins are in
`outputs/interp/adapters/sft-v3-svd32/check.parquet`.

| Comparison | mean \|Δ margin\| | max | decisions agree |
|---|---:|---:|---:|
| Recovered adapter, unmerged, vs the release (S1) | 0.105 | 0.66 | 100% |
| The same adapter, unmerged vs merged in bf16 | 0.101 | 0.50 | 100% |
| Recovered adapter re-merged in bf16 vs the release | 0.093 | 0.45 | 100% |
| SFT-v1's published adapter, unmerged vs merged in bf16 | 0.070 | 0.35 | 99.2% |
| The release vs base, for scale | 8.36 | 18.2 | 87.1% |
| The release vs SFT-v1, for scale | 6.57 | 15.3 | 92.2% |

- **Fact.** S1 required a mean of at most 0.1 logits and got 0.105, with every decision in
  agreement. Merging the same adapter into bf16 weights moves margins by 0.101. For SFT-v1,
  which the first round ran unmerged in the same way, merging moves them by 0.070.
- **Interpretation.** The S1 threshold sits at the floor that bf16 merge rounding sets, so it
  cannot separate extraction loss from rounding. No extraction error is visible on top of the
  rounding, but two independent bf16 roundings cannot be separated cleanly. Not confirmed.
- **Decision.** The user chose to continue with the recovered adapter run unmerged, as v1 was
  (2026-10-27). S2 and S3 stay in force.

### Margins (grid 31,050 and pool 20,700 prompts)

Run on 2026-10-27, on RTX 6000 Ada with batch 16, fp32 readout, and residuals stored.
Logs: `outputs/interp/logs/margins-sft-{grid,pool}.log`. Summaries:
`reports/interp/margins_summary-sft-{grid,pool}-en.json`.

**Gate S3 passes.** The HF hire rate on the grid is 22.61%, against the audit's 22.75% (0.15 pp
off). The within-group unstable share is 1.70%, against the audit's 1.85%. Row-by-row agreement
is not available (see Caveats).

On the same 31,050 grid prompts. All rows except two come from §10 of
`reports/interp/pipeline_results.ipynb`. The per-cell unstable ranges come from its §5, for v1
from the committed first-round version of the notebook. The correlation ranges come from its §6.
Per-pair values are in `outputs/interp/manifest/contrasts-grid-en.parquet` (`dm_base`,
`dm_sft_v1`, `dm_sft`):

| | Base | SFT v1 | SFT-v3 |
|---|---:|---:|---:|
| Hire rate, HF % | 15.02 | 27.61 | 22.61 |
| Margin SD over the grid, logits | 4.01 | 6.03 | 10.39 |
| Within-group unstable sets % | 8.26 | 3.52 | 1.70 |
| Unstable sets %, range over the 6 group × condition cells | 6.2–13.3 | 2.7–4.2 | 0.9–2.7 |
| Mean \|Δm\|, Combat vs Civilian, implicit | 1.04 | 0.41 | 0.49 |
| Mean \|Δm\|, Atheist vs Christian, implicit | 1.06 | 0.33 | 0.41 |
| Share of pairs with Δm > 0: Combat vs Civilian, implicit | 40% | 45% | 73% |
| Share of pairs with Δm > 0: Atheist vs Christian, implicit | 72% | 58% | 28% |
| Correlation of per-pair Δm with base, 19 contrasts | — | 0.14–0.45 | 0.02–0.31 |
| Probe accuracy at layer 28: military explicit / religion implicit (chance 0.20 / 0.11) | 1.00 / 0.85 | 0.90 / 0.45 | 0.45 / 0.22 |

- **Fact.** SFT-v3 has the fewest unstable sets of the three. In logits, its per-pair attribute
  effect is about as large as v1's, and slightly larger on most contrasts. Its margins are 2.6×
  as wide as the base model's, so relative to its own scale the effect is the smallest.
- **Fact.** On the two implicit contrasts, the signed effect left by SFT-v3 points the opposite
  way from the base model's. v1 left it near zero.
- **Fact.** The probe shows the attribute leaving the decision token from the middle layers under
  both SFT models. Under v1 it comes back after full-attention layer 27; under v3 it mostly does
  not.
- **Interpretation, not confirmed.** SFT-v3 may push implicit attribute sentences slightly in the
  direction opposite to the base model's bias, rather than to zero.
- **Unknown.** Whether that residual is above the numerical noise floor (D6, not measured), and
  which components carry it. The patching runs address the second.

The base-model probe accuracies of this run equal the first round's exactly (maximum difference
0.0), so the analysis path is deterministic and only the SFT side changed.

### Patching

Three grid runs, `--spec sft`, 64 pairs each, seed 42, on 2026-10-28/29 (RTX 6000 Ada). Logs:
`outputs/interp/logs/patch-sft-grid-{mil_expl,mil_impl,rel_impl}.log`. The first attempt at
`mil_combat_civ_impl` ran out of GPU memory at model load: another user's job grew on cuda:1
after the free-memory check. Its log is kept as `patch-sft-grid-mil_impl.oom-2026-10-28.log`.
The rerun on cuda:3 completed with no error. Per-run CSVs (`pairs`, `real`, `atp`, `verify`):
`reports/interp/patching/sft-grid-<contrast>-n64-s42/`, beside `sft_v1-grid-*` and `base-grid-*`.
Every SFT-v3 run patches the same 64 `pair_uid`s as base and v1; the notebook asserts this.

All numbers come from §6 of `reports/interp/patching_results.ipynb`: the summary cell and the
`tab:sft` cell.

| 64 identical pairs | Combat vs Civilian, impl | Combat vs Civilian, expl | Atheist vs Christian, impl |
|---|---:|---:|---:|
| Mean \|Δm\|, base → v1 → v3 (logits) | 1.33 → 0.34 → 0.45 | 1.50 → 0.46 → 0.34 | 1.32 → 0.41 → 0.52 |
| Change vs base, v1 / v3 | −74% / −66% | −69% / −77% | −69% / −61% |
| Share at the decision tokens by layer 18, base / v1 / v3 | 0.74 / 0.69 / 0.49 | 0.65 / 0.61 / 0.56 | 0.63 / 0.60 / 0.55 |
| Share at the decision tokens by layer 20, base / v1 / v3 | 0.95 / 0.97 / 0.87 | 0.92 / 0.89 / 0.87 | 0.93 / 0.88 / 0.88 |
| L15.H11, real patching, logits per pair, base / v1 / v3 | 0.66 / 0.09 / 0.13 | 0.51 / 0.16 / 0.10 | 0.35 / 0.12 / 0.13 |
| Component profile r vs base, v1 / v3 | 0.84 / 0.78 | 0.92 / 0.73 | 0.83 / 0.75 |
| Head AtP profile r vs base, v1 / v3 | 0.76 / 0.72 | 0.91 / 0.73 | 0.76 / 0.65 |
| Base top-20 heads still in the top 20, v1 / v3 | 14 / 10 | 16 / 11 | 14 / 13 |
| Kept, attention / GDN / MLP, v1 | 22 / 34 / 18% | 28 / 33 / 29% | 41 / 54 / 55% |
| Kept, attention / GDN / MLP, v3 | 30 / 55 / 32% | 28 / 25 / 26% | 65 / 66 / 81% |

- **Fact.** On the same pairs, SFT-v3 removes 61–77% of the per-pair attribute effect; v1
  removed 69–74%. The remainder still reaches the decision tokens at layers 15–20. L15.H11 is the
  largest verified head under SFT-v3 in all three contrasts.
- **Fact.** The remainder resembles the base model's component profile less under SFT-v3 than
  under v1, on all three contrasts and in both profiles (components and heads).
- **Fact.** "GDN mixers keep the most" holds for v1 in every contrast. For SFT-v3 it holds on
  implicit combat only.
- **Fact.** AtP agrees with real patching under SFT-v3 (Spearman 0.95 / 0.96 / 0.98 on the
  verified heads), and random control heads stay small (|share| at most 0.07).
- **Correction.** The first-round text gave v1's component-profile correlation as r = 0.83–0.84.
  It is 0.83–0.92, because explicit combat is 0.92. The notebook prose and the LaTeX report now
  say so.
- **Interpretation, not confirmed.** SFT-v3 attenuates the same pathway about as much as v1 does,
  and reshapes what remains more. Together with the margins result, on the implicit contrasts
  the remainder mostly points opposite to base. The patching metric orients each pair by its own
  sign, so it shows where the remainder lives, not that it reversed.
- **Unknowns.**
  - Run-to-run variation. Each model was patched once on one pair selection, so the v1 vs v3
    differences have no noise estimate.
  - Which components carry the reversed effect. That needs pairs signed by the base direction.
  - The kept shares sum components that act in series, so they describe a profile, not a
    decomposition.
- **Raw cases.** Patching has no generated responses. The per-pair fp32 margins of both variants
  are in each run's `pairs.csv` (`m_a`, `m_b`).

## Reading

_Filled in after the user's preliminary judgment._

## Caveats

- The SFT-v3 audit's responses are not public, so the E0 check here compares aggregate rates
  with the release's `evaluation.json`, not decisions row by row. With
  `outputs/raw/Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional.parquet` copied from
  Daint, `02_margins.py` runs the row-by-row check instead.
- The audit ran on GH200 with vLLM. The margins here run on HF + PEFT on RTX 6000 Ada.

## Reproducing

```bash
export HBM_OUTPUT_ROOT=/home/samoilenko/hbm-local
uv run python scripts/interp/01_fetch_data.py
uv run python scripts/interp/01b_extract_adapter.py --stage extract
uv run python scripts/interp/01b_extract_adapter.py --stage check --device cuda:1
uv run python scripts/interp/01c_merge_gap.py cuda:1
uv run python scripts/interp/02_margins.py --spec sft --dataset grid --device cuda:1 --resid --batch-size 16
uv run python scripts/interp/02_margins.py --spec sft --dataset pool --device cuda:2 --resid --batch-size 16
uv run python scripts/interp/04_manifest.py --dataset pool
uv run python scripts/interp/04_manifest.py --dataset grid
for c in mil_combat_civ_expl mil_combat_civ_impl rel_atheist_christian_impl; do
  uv run python scripts/interp/10_patch.py --spec sft --dataset grid --contrast $c --device cuda:1
done
```
