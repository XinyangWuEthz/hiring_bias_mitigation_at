# Plan: activation patching and head-level circuit for Qwen3.5-9B (English), base vs SFT vs DPO

## Context

The mitigation chapter has a behavioural result:
- SFT on invariant targets cuts unstable counterfactual sets on 9B-en from 17.3% to 7.2%.
- Moving the operating point explains none of that gain.
- The spread of the decision margin across attribute variants halves (0.52 → 0.26 logits).
- The other team's DPO is built on the SFT student's own mistakes, with the hire rate matched to SFT. It reaches 2.8% unstable vs SFT's 3.7%, which is not significant.

The new chapter needs mechanistic evidence for two things:
1. **The bias circuit:** the head-level circuit by which an injected attribute moves the hire/reject decision in the base model.
2. **What debiasing changed:** how SFT and DPO change that circuit (removing the attribute, routing it away, suppressing its readout, or compensating for it). It should also test the other team's worry that DPO learned a "rationale-strength" shortcut rather than invariance.

Scope:
- English only.
- 2× RTX 6000 Ada, 48 GB each.
- Everything regenerated locally; the Transcend drive is ignored.
- Priority is the **data for patching and its pipeline**.
- Code goes inside the package, in clearly separate `interp` folders.

### Facts verified in this session

- **Architecture:** Qwen3.5-9B has 32 layers.
  - The 24 Gated-DeltaNet (GDN) layers each have 32 value heads × 128.
  - Full attention sits at layers 3, 7, …, 31: 16 query heads, 4 key/value heads, head_dim 256.
  - An elementwise sigmoid **gate** is applied before `o_proj`. It comes out of `q_proj`, interleaved per head as `[q(256) | gate(256)]`; see `modeling_qwen3_5.py:787-818`.
  - RMSNorm multiplies by `(1 + w)`.
- **Module names:**
  - Full attention: `layer.self_attn.{q,k,v,o}_proj`.
  - GDN: `layer.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,conv1d,norm,out_proj}`.
  - MLP: `layer.mlp.{gate,up,down}_proj`.
- **SFT LoRA** (Hub `Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en`) touches only the **MLPs of all 32 layers + q/k/v/o of the 8 full-attention layers**. GDN token-mixers are frozen. Any change in how information moves between positions must therefore come from those 8 layers, or arrive from upstream through the MLPs.
- **Implicit sentences carry more than the attribute** (`data/injection_templates/implicit_en.csv`):
  - "participating in combat actions, *but looking for a civilian job*" vs "civilian, *who never joined armed forces*".
  - Atheist "*would work on any day*" vs the other religions' "*would need day offs*".
  - The two confirmed English biases (combat −9.6%, Atheist +8.7%) may therefore be **availability**, not identity. This must be tested (H6).
- **Logit precision:** the existing margin scripts read **bf16 logits**. At logits of 16–32 the bf16 step is 0.125, so the SFT spread is about 2 quantisation steps. Margins must be read in fp32 from the pre-norm residual: hook the input to `model.model.norm`, apply the norm in fp32, and dot with `W_U[hire] − W_U[reject]`. `norm` and `lm_head` are not adapted, so the readout direction is identical for base, SFT and DPO.
- **Missing kernels:** `flash-linear-attention` and `causal-conv1d` are not installed, so HF uses the slow torch path for GDN.
- **Paths:** `.env`'s `HBM_OUTPUT_ROOT` and `HF_HOME` point at an unmounted drive.

---

## Folder layout

```
src/hiring_bias_mitigation/interp/      library code (only arch.py hard-codes module names)
src/hiring_bias_mitigation/eval/decision.py   NEW shared home of PREFIX / WORDS / decision_token_ids
scripts/interp/                         numbered thin CLIs, idempotent (skip finished outputs)
configs/interp/                         models.yaml, contrasts.yaml, controls_en.yaml, pilot.yaml, stages.yaml
reports/interp/                         small tracked summaries, circuit JSONs, FINDINGS_INTERP.md, figures/
tests/interp/                           no-GPU unit tests (+ `gpu`-marked ones)
$HBM_OUTPUT_ROOT/outputs/interp/        heavy artifacts (margins, caches, manifests)
```

**Library modules** (`interp/`):

| Module | Responsibility | Reuses |
|---|---|---|
| `spec.py` | `ModelSpec` = base + ordered adapters (`base`, `sft`, `dpo: [sft, dpo]` with `reference: sft`). Everything loads in one PEFT process and switches with `set_adapter([...])` / `disable_adapter()`. Also fingerprints adapters and asserts the expected LoRA module count (128 for SFT). | `diagnose_adapter.py` loading, `runner._require_adapter` |
| `arch.py` | Hook-point resolver; per-head slices: query, gate, `kv_of = h//4`; GDN `key_of = j//2` | `mitigation/embedding._decoder_layers` |
| `render.py` | Grid and pool prompts, chat wrap, forced prefix | `runner.build_frame`, `runner._build_prompts`, `backends._chat_wrap`, `generation/dataset._audit_prompt` / `_with_implicit` |
| `segment.py` | Token segments, all the assertions, `PositionMap` (prefix aligned from the start, suffix aligned from the end) | `data/prompts.build_prompt` pieces |
| `hooks.py` | Capture, patch and ablate on (component, positions); asserts each hook fires exactly once; hooks the LoRA wrapper, not `base_layer` | `embedding.ActivationEditor` pattern |
| `attention.py` | Eager attention function registered through `AttentionInterface`: records patterns, replaces pattern or values per query head. sdpa returns no weights. | — |
| `tailcache.py` | Runs the prefix once, then clones the KV, conv and recurrent cache for each patched tail run; about 40× faster sweeps at the final positions | — |
| `margins.py` | Batched fp32 margin, with side outputs: final-position residual for all layers, per-component DLA | — |
| `manifest.py` | Contrast pairs, controls, selection flags, discovery/confirmation split | — |
| `patching.py`, `attribution.py`, `dla.py`, `circuit.py` | Residual, head and path patching; AtP\* (QK fix on full attention, GradDrop); GDN effective attention via the gradient with respect to v; DLA; faithfulness, completeness and minimality | — |
| `probes.py` | Probes, mean-difference steering, LEACE erasure | `embedding.fit_eraser`, `linear_probe_accuracy`, `ActivationEditor` |
| `adapter_mask.py` | Per-module LoRA on/off by saving and restoring `module.scaling[adapter]` | — |
| `dpo_reward.py` | Per-token log π/π_ref, the decision×feedback splice | `mitigation/sft.render_chat` |
| `stats.py` | Ratio-of-means bootstrap clustered by candidate, TOST | `analysis/stability.compare` pattern, `eval/stats` (BH-FDR, paired permutation) |
| `io.py` | Paths through `resolve_output_path("outputs/interp/...")`; a `.meta.json` sidecar per artifact (git sha, spec fingerprint, dtype, GDN kernel path, attention implementation, tokenizer hash) | `utils/config` |

**Consolidation:** move `decision_token_ids` and `PREFIX` / `WORDS` into `eval/decision.py`. They are currently duplicated in `scripts/diagnose_adapter.py` and `scripts/operating_point_control.py`; both scripts will import them from the new module.

---

## Shared definitions

- **Contrast pair:** variants A and B of the same set (same `pair_id` and condition). The attribute effect is Δm = m(A) − m(B), using the fp32 margin.
- **Patch metric:** R = Σ(m_patched − m_B) / Σ(m_A − m_B). This is a ratio of means across pairs, not a mean of ratios.
- **Statistics:** 95% CI from a bootstrap over candidates; BH-FDR across components.
- **Splits:** the circuit is **discovered** on the synthetic pool (disjoint from the benchmark) using base only. It is **confirmed** on the 150 benchmark candidates. SFT and DPO are compared on the benchmark only, because the pool is SFT training data.
- **Ablation:** primary is the **set-mean**: the component's mean over all variants of the set at the same aligned position, which removes only what varies with the attribute. Secondary is resampling from the other variant. Zero ablation is never used.
- **Pair selection:** fixed once, from base margins only. The same pairs are reused for SFT and DPO.

---

## Hypotheses: experiment and pass/fail

| ID | Hypothesis | Experiment | Pass / fail criterion |
|---|---|---|---|
| **E0** | The instrumentation is exact | See Verification | Every identity holds; HF agrees with the audit ≥98% (base); the HF-recomputed unstable share is within 1 pp of the audit's 17.3% and 7.2% |
| **H1** Bottleneck | The attribute's effect reaches the decision through a narrow layer band, then sits in the last ~20 tokens | Residual patching over 33 layers × position classes (`span`, `post_span3`, `cv`, `body+format`, `chat_close`, `forced_prefix`, `final`); B→A, both directions | Pass: a layer L\* exists where the last-K positions give R≥0.8 at layers ≥L\* and R≤0.2 at layers ≤L\*−4. Otherwise (H1b) the information is spread over CV positions, and the circuit must include CV positions. |
| **H2** Router | Full-attention layers, not GDN recurrent state, carry the information from the span to the decision | Patch all full-attention head outputs vs all GDN head outputs at suffix positions, per layer | Full attention dominates if R_attn≥0.6 and R_GDN≤0.3, GDN if the reverse, otherwise mixed. Report whichever holds. |
| **H3** Sparse circuit | A small set C (≤40 heads + ≤8 MLPs, position-specific) explains the effect | AtP\* ranks all 128 attention heads, 768 GDN heads and 32 MLPs. Then: activation-patching check of the top 64 + 64 random components; path patching sender→receiver (receivers = top-10 writers, at their q/k/v/gate or residual inputs); QK / OV / gate decomposition on length-matched pairs; GDN analogue (q/k, v, a/b decay and beta, z) | **Faithfulness** F(C)≥0.8, CI lower bound ≥0.7, on the confirmation split. **Completeness**: ablating C leaves R≤0.2, while random sets of the same type mix leave ≥0.8. **Minimality**: every member contributes ≥0.02. **AtP check**: Spearman ≥0.6 against real patching, false negatives <5%. |
| **H4** Readout | A few late "writer" components put the attribute effect directly onto hire−reject | DLA vs total effect; "writer" means direct effect ≥50% of total, "mover" means small direct effect but large total | DLA sums to the margin within 1e-3; the top-5 DLA writers include the top-3 patched late components |
| **H5** Representation | The attribute can be read linearly from the span early, and from the final positions from layer L\* on; that direction is causal | Probes fitted on the pool cache; mean-difference steering at L\* (dose-response); LEACE erasure | Final-position probe accuracy first exceeds chance+20 pp within ±2 layers of L\*; steering slope CI excludes 0; erasure cuts \|Δm\| by ≥50% with attr_free hire rate moving ≤2 pp |
| **H6** Confound | The implicit combat and Atheist effects come from the availability clause, not the identity words | (a) Per-token attribution within the span: identity tokens vs clause tokens. (b) Hand-written minimal pairs: identity changed with the clause fixed, and clause changed with the identity fixed | Clause-driven if the clause-only effect ≥2× the identity-only effect and the clause carries ≥60% of span attribution. If so, the chapter describes an "availability circuit". |
| **H7** Shared pathway | Military, religion, gender, and explicit vs implicit share the late movers and writers | Jaccard overlap of the top-k components; C_mil evaluated on religion pairs | Shared if Jaccard ≥0.5 and F(C_mil on religion) ≥0.6 |
| **H8** How SFT works | (a) **removed upstream**, (b) **readout suppressed** (still represented, read orthogonally), or (c) **compensated** (new opposite-sign components) | (1) Per-component effects SFT vs base on the same pairs. (2) Final-position probes, base vs SFT. (3) Cross-model patching: base component outputs into SFT; does Δm come back? (4) **Adapter masking**: LoRA only on {attention}, {MLP ≤15}, {MLP >15}, single layers, leave-one-out | (a): probe drops ≥10 pp and mover effects shrink. (b): probe unchanged (±3 pp) and the attribute direction's cosine with the readout falls. (c): opposite-sign components (q<0.05) whose ablation brings back ≥50% of base Δm. Masking names the module group responsible for ≥50% of the gain. Δm also reported divided by each model's margin SD. |
| **H9** DPO and the decision circuit | DPO on top of SFT does not change the circuit at the decision position | H3/H8 effects, DPO vs SFT; TOST with δ = 2× noise floor | Equivalent for ≥90% of circuit components and for per-set Δm spread |
| **H10** DPO shortcut | The other team's worry: DPO's implicit reward sits in the feedback tokens and tracks rationale strength, not invariance. The decision token comes **before** the feedback in the JSON, so a shortcut would show up there. | Per-token r_t = log π_DPO − log π_SFT on chosen and rejected, split into decision tokens, feedback tokens and scaffold. 2×2 splice {decision chosen/rejected} × {feedback chosen/rejected}. Reward of a fixed completion across the attribute variants of its prompt. | Shortcut supported if the feedback main effect ≥3× the decision main effect, the reward correlates with rationale-strength features (including the team's TF-IDF classifier score), and attribute sensitivity is no lower than SFT's. Refuted if the decision tokens take ≥30% and attribute sensitivity is lower than SFT's. |
| **H11** GDN effective attention | GDN heads can be read like attention heads | Gradient of u·o_t with respect to v_s gives exact source weights for the final position | Diagnostic: the share of each GDN mover's weight on span vs CV positions; agrees with span patching within 10% |

---

## Data pipeline (the priority; built and validated before any circuit work)

All of it goes under `$HBM_OUTPUT_ROOT/outputs/interp/`, with a `.meta.json` sidecar per artifact. **No teacher-model generation is needed.**

**D0. Environment and fetch** (`scripts/interp/00_env_check.py`, `01_fetch_data.py`)
- Set `HBM_OUTPUT_ROOT=/home/samoilenko/hbm-local` and `HF_HOME=$HBM_OUTPUT_ROOT/hf-cache`, and create the `.hbm-models-root` marker there.
- Install `flash-linear-attention`. `causal-conv1d` is optional.
- Download:
  - `Qwen/Qwen3.5-9B` and the SFT adapter.
  - The responses dataset runs `Qwen3.5-9B--en--baseline` and `--sft--adapter--en_only`, to `outputs/raw/`.
  - The synthetic data `data/raw/en_invariant.parquet` and `sft_train`, to `artifacts/semisynthetic-v1/`.
- The env check refuses if `HF_HOME` or the output root is on an unmounted path, and records which GDN kernel path is in use.

**D1. Model specs** (`configs/interp/models.yaml`)
- `base`, `sft`, and `dpo` with `stack_on: sft`.
- The DPO path comes from `$HBM_DPO_ADAPTER` / `--adapter dpo=PATH` and is cross-checked against `adapter_config.base_model_name_or_path`.
- A stacked-vs-merged equivalence check runs when the DPO adapter arrives.

**D2. Full-grid fp32 margins** (`02_margins.py`): `margins/{spec}/grid-en.parquet`
- **Rows:** 31,050 per spec (450 pairs × 3 single groups × explicit/implicit + attr_free).
- **Columns:** KEY, candidate_id, job_id, group_id, prompt_uid (sha1 of the text), n_tokens, `margin_fp32`, `margin_bf16` (to quantify the quantisation issue), logprob_hire/reject, top-1 token, decision_hf.
- **Side outputs:**
  - `final_resid/{spec}/` with all 33 layers at the final position, fp16, about 8.4 GB per spec.
  - `dla/{spec}/grid-en.parquet` with per-component DLA, about 120 MB.
- **Pool subset:** about 600 complete pool sets (about 15k prompts) for base and SFT.
- **Check:** the unstable share recomputed from these margins matches the audit (17.3% / 7.2%), and decisions agree with the audit's `decision` column.

**D3. Token segmentation** (`03_segments.py`): `segments/{dataset}-en.parquet`
- One row per prompt: `input_ids` plus token boundaries for `chat_open, header, job, profile_open, span[label|value or identity|clause], post_span3, cv, profile_close, body, format_json, chat_close, forced_prefix, final`.
- Built from the `build_prompt` pieces plus `offset_mapping`.
- **Assertions:**
  - Pre-span ids are identical across a set's variants.
  - Post-span ids are identical when aligned from the end.
  - No token crosses a segment boundary; a merge at `value\n` is a hard error.
  - The span decodes back to its text.
  - The last token completes the forced prefix.
  - Retokenising gives the same ids.
  - attr_free has span length 0.
- Violations are written to `segments/violations.parquet`; the run stops above 0.5%.

**D4. Contrast manifest** (`04_manifest.py`, `configs/interp/contrasts.yaml`): `manifest/contrasts-en.parquet`
- **Contrasts, fixed in advance:**
  - `mil_impl_combat_civ`, `mil_impl_combat_vet`, `mil_expl_combat_civ`
  - `rel_impl_atheist_christian` (plus Atheist vs each other religion), `rel_expl_atheist_christian`
  - `gen_{expl,impl}_male_female`
  - `presence_*`: attribute vs attr_free, suffix-aligned only
- **Columns:** pair_uid, contrast_id, dataset, split (discovery = pool, confirmation = benchmark), both prompt_uids, candidate_id, span_len_A/B, `len_match`, m_A, m_B and Δm per spec, z = Δm/σ.
- **Flags:**
  - `sel_effect`: \|Δm\| ≥ max(5σ_num, 0.5) and its sign matches the group-level bias.
  - `sel_flip`: the margin changes sign between A and B.
  - `sel_random`: a 30% uniform sample, used to check that selection does not bias the results.

**D5. Hand-written controls** (`configs/interp/controls_en.yaml`, rendered on benchmark and pool CVs): `manifest/controls-en.parquet`. This is **the only new text data**.
- (a) **Null swaps:** paraphrases of each implicit sentence with the same meaning and the same token length. They give σ_null.
- (b) **H6 minimal pairs:** the identity changed with the clause fixed, and the clause changed with the identity fixed, length-matched.
- (c) **Length-matched subset tag:** used to check for RoPE and position-shift artifacts.
- About 0.5 GPU-h to score their margins.

**D6. Noise floor** (`05_noise_floor.py`): `noise/{spec}.parquet` and `reports/interp/noise_floor.json`
- σ_num: the same prompt across batch sizes, padding, sdpa vs eager, and fla vs torch. Target ≤0.05 logits.
- σ_null: from the null swaps (D5a).
- Self-patch: patching clean into clean must change nothing (exactly 0).

**D7. Activation caches** (safetensors shards + index parquet)
- **Pool probe cache:** about 10k prompts × 33 layers × 7 position summaries (span mean and last token, post_span3, final-4), fp16, about 19 GB.
- **Head-output cache** for manifest prompts: inputs to `o_proj` / `out_proj` at the last 4 positions, about 4 GB per spec.
- **Attention patterns:** per-segment aggregates for every manifest prompt; full rows for 200 prompts.
- **Set-mean ablation vectors:** last-K positions only.
- Total disk is about 120 GB or less (500 GB free).

**D8. DPO inputs, from the other team**
- **Needed:** the DPO adapter, whether it is stacked on SFT (assumed) or trained on top of merged SFT weights, and the 2,362 + 295 pairs (prompt, chosen, rejected, plus KEY if available). These go to `dpo/pairs.parquet`.
- **Generated here:** `dpo/token_logps-{spec}.parquet`, one row per pair, side, token and token class, and the 2×2 splice set.
- **Before use:** read 50 DPO generations to confirm the output contract (still answers hire/reject in valid JSON).

---

## Execution order (one spec per GPU wherever possible)

| Stage | Work | Estimated GPU-h |
|---|---|---|
| S0 | D0 env, kernels, downloads | 0 (about 1 h wall) |
| S1 | Pilot: throughput at batch 1/4/8, peak fwd+bwd memory at 2.3k tokens, fla vs torch agreement; **E0**; D6 | 1–2 |
| S2 | D2 margins, base ‖ SFT, grid and pool | 4–8 |
| S3 | D3 segments, D4 manifest, D5 controls | about 0.5 |
| S4 | H1 residual sweep (with the tail cache) | 5–8 |
| S5–S7 | H2, then AtP\* → head verification → path patching → QK / OV / gate decomposition (H3, H11) | 9–12 |
| S8 | Circuit evaluation on the confirmation split, H4 DLA, H7 | 3 |
| S9 | H5 probes and steering, H6 confound | 2–3 |
| S10 | H8 on SFT: repeat S4–S6 and S8, cross-model patching, about 20 adapter masks | 12–15 |
| S11 | H9 / H10 once the DPO adapter and pairs arrive (can run in parallel with S10) | 8–10 |

The total is about 45–65 GPU-h, roughly 1.5–2 days on 2 GPUs. The S1 pilot measures real throughput and rescales this table.

**Deliverables:**
- `reports/interp/circuit_{contrast}_{spec}.json` (nodes, edges, R and CI per element)
- Per-hypothesis CSVs
- `reports/interp/FINDINGS_INTERP.md`, written by hand in the style of `MITIGATION_FINDINGS.md`
- Figures under `reports/interp/figures/`

---

## Verification

**Unit tests (no GPU)**, in `tests/interp/`, on a tiny random `Qwen3_5TextConfig` in fp32: layers `[linear ×3, full]`, hidden 64, 4 query / 2 key-value heads, GDN 2 key / 4 value heads.
1. Capturing and patching back the same values gives bit-identical logits.
2. Σ_h W_O[:, h] z_h equals `o_proj(z)` / `out_proj(z)`, including with random LoRA using effective weights.
3. DLA sums to the pre-norm residual, and the fp32 readout equals the logit difference.
4. Patching at prefix positions has no effect.
5. With **unequal** span lengths, patching every mixer and MLP output at every suffix position reproduces the source logits (checks the end-aligned position map).
6. The tail-cache forward (including cloned GDN conv and recurrent state) equals the full forward, both plain and patched.
7. Reassembling the gate and query slices matches the module; the custom attention function matches sdpa.
8. An adapter mask of 0 everywhere equals `disable_adapter`; stacked adapters equal merged-then-applied.
9. AtP matches the true effect of a small perturbation (<5% relative error).
10. Batched left-padded margins equal unbatched ones.
11. Segmentation assertions on a stub tokenizer, including a deliberate boundary merge that must fail; the manifest is deterministic.
12. `decision_token_ids` rejects a prefix that tokenises inconsistently.

**Tokenizer test:** runs only when the real tokenizer is cached; segmentation on 200 real grid prompts.

**Existing suite:** `make local-test` must stay green after the `eval/decision.py` consolidation.

**E0 on the GPU** (`06_e0_sanity.py` writes `reports/interp/E0_sanity.json`). Every item below must pass before S4:
- HF/audit agreement ≥98% for base; SFT agreement measured.
- 128 LoRA modules attached.
- σ_num ≤0.05.
- Self-patch exactly 0.
- The full-suffix patch identity holds.
- The DLA sum equals the margin.
- fla equals torch.
- Every hook fired exactly once.
- The bf16 vs fp32 margin difference is reported.
- The unstable shares are reproduced.
