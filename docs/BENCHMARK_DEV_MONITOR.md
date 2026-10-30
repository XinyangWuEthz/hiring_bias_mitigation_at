# Direct benchmark-dev TPR monitoring

2026-10-23. The user requested inference-only GRPO-v3-all checkpoint trajectories, one submitted
job at a time, and replacement of the semisynthetic projection monitor with direct measurement
on a benchmark dev split. No training is authorized by this protocol.

The candidate split and parent decisions remain frozen. Job 5127818 completed the trajectory
and the proposed dev stopping policy **failed the retrospective check**: full first fails its
10 pp guardrail at retained step 150, while dev first triggers its 8 pp alarm at 400. The
callback is implemented, but this fixed dev policy must not be enabled as the sole full-safety
monitor. The manifest remains `training_authorized=false`; no training is authorized.

The small Git receipt is `configs/analysis/benchmark_dev_monitor_v1.json`. It pins the split,
manifest, parent model and cached checks. Large row files remain under
`$HBM_OUTPUT_ROOT/outputs/benchmark_dev_monitor_v1/` and are not checked into Git.

## Frozen candidate split

`freeze_benchmark_dev.py split` reads released benchmark metadata and GPT-4o binary labels.
It saves IDs before the `materialize` stage reads any SFT-v3 responses. No model margin,
response or outcome is used to choose the partition.

Each of 30 job positions has five candidates. Within each role, choose three for dev and two
for the historical remainder. Choose the 3-of-5 combination minimizing
`abs(5 * selected_hire_pairs - 3 * role_hire_pairs)`. Break ties by SHA256 of
`benchmark-dev-v1|20261023|role|sorted candidate IDs`, with IDs separated by `|`.
All three pairs for a candidate and all 69 prompt variants of each pair remain together.

| Partition | Candidates | Pairs | Hire pairs | Positive candidates | All prompts | Positive prompts |
|---|---:|---:|---:|---:|---:|---:|
| dev | 90 | 270 | 93 | 50 | 18,630 | 6,417 |
| historical remainder | 60 | 180 | 57 | 35 | 12,420 | 3,933 |

The banks are candidate-disjoint and share 123 job IDs. They are not job-disjoint. The entire
benchmark already contributed to reports used during development. The remainder is therefore
not an independent, untouched test. New checkpoint results on these same candidates remain
retrospective exploratory evidence. Every benchmark candidate and job remains excluded from
training data and gradients.

Frozen split SHA256:
`b5fe8fc48ee05ab4e0ef0d2f0101c9bdbddce7e5b2f07ddcbffc993e61035d31`.
The benchmark source CSV hash and exact candidate and pair IDs are in `split.json`.

## Direct measurement

Keep the original binary baseline prompt and released GPT-4o labels. The stricter blind-review
annotation protocol and semisynthetic Qwen anchors do not replace them. Follow the generated
benchmark protocol: greedy decoding, temperature 0, top-p 1, seed 42, maximum 512 new tokens,
thinking off, and the official parser.

This measures the same kind of generated decision as the original TPR guardrail in
`analysis/reference_metrics.py::hire_by_reference`. It includes attribute-free prompts.
Forced-prefix one-token margins are a separate diagnostic and must not be labeled generated TPR.

The callback needs only the 6,417 dev rows with reference label `hire`: 93 pairs, 50 positive
candidates, and 69 variants per pair. It measures exactly the positive-label dev TPR. The full
dev registry retains all 90 candidates, including the 40 with no positive reference pair.
FPR is undefined on the positive-only callback rows and is reported as null. Final audit
reports compute FPR and the other guardrails on the complete benchmark grid.

SFT-v3 hires 3,600 of the 6,417 positive dev prompts, giving a fixed parent TPR of
56.100981767180926%. At optimizer step t:

`delta_tpr_pp = 100 * (current_hire_count - 3600) / 6417`

Do not subtract the full-benchmark SFT-v3 TPR of 57.729468599033815% from a dev rate. The
historical remainder has its own parent TPR, 2,375 / 3,933 = 60.38647342995169%.
Log losses, gains, net change, completeness and the frozen identity alongside each reading.
These are exact measurements of the finite dev bank under the specified decoder, with no
margin bins or projection weights.

## Candidate stopping policy and final evaluation

The prospective candidate policy checks every 25 optimizer steps and stops when
`delta_tpr_pp <= -8.0`. On this bank that is a TPR at or below 48.100981767180926%, or at most
3,086 hires. Save the triggering checkpoint for diagnosis. The selection rule retains the
latest earlier saved checkpoint with a complete reading above -8 pp; if none exists, retain
SFT-v3. A later recovery does not erase the first stop that would have occurred.

Missing or duplicate row IDs, changed frozen identities, scoring failure and unusable
positive-row decisions cause an integrity stop. They cannot be dropped from the denominator
to make TPR look safer. This operational rule does not rewrite historical evaluation metrics,
which retain their existing usable-output denominator and separate coverage guardrail.

The original final guardrail in `docs/EVAL_STANDARD.md` remains unchanged: full-benchmark
generated delta TPR must be at least -10 pp. Exactly -10 pp passes. The 8 pp candidate training
stop is an operational buffer. It does not establish a 10 pp full-benchmark guarantee: dev
composition and the 25-step check spacing can differ from the full benchmark and its trajectory.

Candidate-cluster intervals, if reported, describe sensitivity to candidate composition. A
single-check 95% interval is not a whole-run repeated-check guarantee. The old projection
bootstrap bound is not carried over as an uncertainty bound on direct dev measurement.
`policy.training_authorized=false` remains frozen in the manifest; the prepared training
entry refuses it. This work submits inference only.

## Cached endpoint check after the split freeze

`cached_split_metrics.json` recomputes the existing SFT-v3, checkpoint-525 and final-571 raw
responses with the current official parser. All 31,050 rows per arm are present and usable;
there are no decision or outcome differences from the cached parsing. No new inference was
needed. No split ID was changed after reading these results.

| Model | Full TPR, % | Full delta, pp | Dev TPR, % | Dev delta, pp | Remainder delta, pp |
|---|---:|---:|---:|---:|---:|
| SFT-v3 | 57.72947 | 0 | 56.10098 | 0 | 0 |
| checkpoint-525 | 48.43478 | -9.29469 | 49.35328 | -6.74770 | -13.45029 |
| final-571 | 47.62319 | -10.10628 | 48.20009 | -7.90089 | -13.70455 |

Neither endpoint alone triggers the frozen dev 8 pp line. Final-571 nevertheless fails the
full 10 pp guardrail. At the time of this cached endpoint check the earlier trajectory was pending.
The completed replay below now establishes a dev stop at 400, which is too late to prevent
the first observed full failure at 150. Direct measurement
removes the semisynthetic projection assumption; it does not remove dev/remainder differences.
Do not choose a different split to make this observed discrepancy disappear.

Checkpoint-525 still passes the original full 10 pp guardrail. Its recorded matched-instability
change, -1.02 pp with 95% interval [-1.99, -0.12], remains a valid description of that audit.
It was selected after the final adapter audit, so the conclusion remains exploratory under
`EVAL_STANDARD` rule 3. Full delta below -8 pp does not mean its dev delta is below -8 pp.
The completed earlier trajectory below rules out selecting 525 under the new dev stopping policy.
TPR-only inference cannot independently establish a fairness gain for any newly selected model.

Plot full, dev and historical-remainder generated TPR in optimizer-step order. Show both the
8 pp candidate line and the unchanged 10 pp final line. Report the first retained checkpoint
crossing and its predecessor. Missing intermediate checkpoints bracket the crossing; they do
not identify an exact stopping step. Any policy revision informed by this trajectory must be
reported as development, not independent validation.

## Completed checkpoint replay, 2026-10-23

See `docs/GRPO_CHECKPOINT_TRAJECTORY.md` and the experiment's `trajectory.json` and
`threshold_review.json` for all 20 checkpoints. First retained crossings are full 8 pp = 150,
full 10 pp failure = 150, dev 8 pp = 400, and dev 10 pp failure = 550. At 150 the full/dev
deltas are -10.21256/-7.80739 pp. Dev 8 pp therefore missed the first observed full failure.

Checkpoint 525 cannot be selected after the dev rule's first stop at 400, despite its later
recovery. Its original full-10 TPR pass and historical exploratory result remain recorded.
Keep 8 pp as the early-stop target and 10 pp as the final guardrail; relaxing dev to 10 pp
would stop still later and does not solve the mismatch. The current dev bank is not a
validated sole safety gate. Full 8 pp also first triggers at 150, after the 10 pp guardrail
has already failed, so the 25-step schedule provides no guaranteed 2 pp warning buffer.

Steps 25/50/75 were not retained. All first-crossing statements are limited to the retained
checkpoints; the full earlier history and behavior between samples remain unknown. No
split IDs, labels, published historical results or training authorization were changed.

## Reproducing the frozen bank

The first stage reads only candidate, pair, role and label metadata. The second verifies the
unchanged split and source, rebuilds all 31,050 original prompts, then joins the historical
SFT-v3 raw decisions. It checks complete unique keys, current parser agreement, identical
reference labels and byte-identical reconstructed prompts. Row IDs are SHA256 of the compact
UTF-8 JSON array `[pair_id, protected_group, protected_attr, condition]`.

Run both stages in a new output directory; existing frozen files are never overwritten:

```bash
uv run python scripts/freeze_benchmark_dev.py split \
  --repo "$PWD" --out "$HBM_OUTPUT_ROOT/outputs/benchmark_dev_monitor_v1_new"
uv run python scripts/freeze_benchmark_dev.py materialize \
  --repo "$PWD" --out "$HBM_OUTPUT_ROOT/outputs/benchmark_dev_monitor_v1_new" \
  --output-root "$HBM_OUTPUT_ROOT" \
  --model-identity "$HBM_OUTPUT_ROOT/outputs/monitor_v2_freeze/20261023-intake-v1/source/model_identity.json"
uv run pytest tests/test_freeze_benchmark_dev.py tests/test_benchmark_dev_monitor.py
```

`manifest.json` uses schema `benchmark-dev-monitor-v1`. It pins the source CSV, SFT raw and
metadata, parent model identity, prompt/parser code and each data artifact. Parent decisions
are JSONL `{prompt_id, decision}` in exactly the rows-file order. Its candidate list contains
all 90 dev candidates, while the positive-only rows contain 50. Generated files are stored
outside Git; the receipt fixes the already materialized split and manifest hashes.

The earlier Djinni intake and binary-label workflow remains recorded in
`docs/MONITOR_V2_FREEZE.md` for future independent label research. Its unresolved annotation,
source-transfer and sampling questions are not prerequisites for this direct-dev inference
experiment, and this change does not resolve them. The conclusion that RJDB reject anchors
are questionable remains recorded for blind review.
