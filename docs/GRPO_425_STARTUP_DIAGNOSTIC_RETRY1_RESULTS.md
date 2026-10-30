# Checkpoint 425 diagnostic retry: completed inference and remaining replay gap

Job `5141205` completed all four pre-registered cells, each with 31,050 decisions
and zero invalid outputs. Slurm reports `COMPLETED`, exit `0:0`, elapsed
`00:50:39`, on node `nid005335`. It ran from 2026-10-25 11:53:38 to 12:44:17 UTC.
The repaired GPU startup and inference pipeline worked. No training, optimizer
update, or new allocation was performed while collecting these results.

The two exported weight constructions are equal. Within either inference context,
CPU and GPU constructions produce identical binary decisions. Changing the
engine/request context produces 53 flips, including 39 of the original 52 in the
same direction. However, historical replay still differs on two rows and failed
startup replay still differs on 25 rows. The complete inference experiment therefore
does **not** establish an exact reproduction of the original startup or its unique
numerical cause. The strict startup gate remains unmet and unchanged.

## Frozen identity and repair

- Execution commit: `307a75428d1fa67bb4e6175ada91dc82e33c79d3`.
- Manifest binding: `d371aaa3d8c10af86e98f9ef72540dbdc23920f009ab14d42d6ec387238f7994`.
- Manifest file SHA256: `88cf112a60a6075f64c3f7bdc2f125fee036014051a8c481c70b48a5c092f59a`.
- Frozen plan: [GRPO_425_STARTUP_DIAGNOSTIC_RETRY1.md](GRPO_425_STARTUP_DIAGNOSTIC_RETRY1.md).
- Previous diagnostic job `5140642` and its partial results remain preserved in
  [GRPO_425_STARTUP_DIAGNOSTIC_RESULTS.md](GRPO_425_STARTUP_DIAGNOSTIC_RESULTS.md).

The prior diagnostic failed because added configuration logging deep-copied a
vLLM runtime parameter. The repair records configuration without deep copying,
checks the actual prefix-cache-reset return value, and runs an isolated four-rank
GPU smoke test before constructing fresh formal engines. All four smoke ranks
completed two weight pushes, the original 256-request warmup, served-parameter
hashing, and one fixed request per rank. Smoke requests are excluded from metrics.

All 13 phases exited zero. The smoke and two live phases emitted ignored vLLM
finalizer exceptions during process exit. The script had already saved and sealed
each rank result, synchronized, and destroyed all process groups; the finalizer
then tried to destroy its TP group again. The rank receipts, subsequent collection,
and summary all passed. This is a cleanup defect, not a failed inference phase.
It should be corrected in later maintenance, but does not require rerunning these
measurements. No frozen execution code was changed during this job.

## What H, L, CPU, and GPU mean

CPU and GPU describe weight construction, not the device used for inference. All
four cells perform inference on GH200 GPUs. CPU uses the historical CPU merge;
GPU exports the live-style merged parameters. Both apply the common SFT once,
followed by checkpoint 425.

H is the historical engine/request context: one TP1 engine, 4,000-row request
chunks, at most 256 sequences, 16,384 batched tokens, 8,192 maximum model length,
0.8 GPU memory utilization, engine seed 42, and no explicit request seed. L uses
four TP1 engines and the fixed `canonical_rows[rank::4]` mapping, at most 64
sequences, 4,096 batched tokens and model length, 0.3 GPU memory utilization,
engine seed equal to rank, request seed 42, two weight pushes/cache resets, and
the original warmup. Both use greedy decoding with a 512-token maximum.

The comparison changes this entire context bundle. It does not isolate batching,
cache state, scheduling, a kernel, or a random-seed setting individually.

## Input and weight checks

- All 31,050 reconstructed token sequences match across tokenizer paths. All
  124,200 actual formal requests match the frozen input token IDs, including row
  order. Request-output text equals the corresponding saved raw text.
- All 775 CPU/GPU exported tensors are bitwise equal, including non-LoRA tensors.
  The CPU serialized weight files also match historical fingerprints.
- Each live rank has 331 served parameters after vLLM packing. CPU-L and GPU-L
  hashes match for every parameter on all four ranks.
- SFT merge retention is 99.999577%; checkpoint 425 retention is 96.688718%,
  each checked over 128 layers and above the frozen 80% requirement. The equal
  exported tensors give the same final weights for both constructions.

The old failed startup did not save equivalent actual-token/served-parameter
receipts. These successful checks cannot establish equality of every old runtime
state retrospectively.

## Full-benchmark metrics

The unchanged reference has 10,350 hire rows and 20,700 reject rows. Every listed
source and arm has 100% decision coverage. Agreement is agreement with the frozen
reference labels, not an independent assessment of label quality.

| Source / arm | TP | FP | TPR % | FPR % | Agreement % | TPR change vs SFT, pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT-v3 cache | 5,975 | 1,090 | 57.729469 | 5.265700 | 82.399356 | 0 |
| Historical 425 cache | 6,204 | 1,158 | 59.942029 | 5.594203 | 82.917874 | +2.212560 |
| Failed lower-LR retry startup | 6,192 | 1,158 | 59.826087 | 5.594203 | 82.879227 | +2.096618 |
| CPU-H | 6,203 | 1,157 | 59.932367 | 5.589372 | 82.917874 | +2.202899 |
| GPU-H | 6,203 | 1,157 | 59.932367 | 5.589372 | 82.917874 | +2.202899 |
| CPU-L | 6,195 | 1,156 | 59.855072 | 5.584541 | 82.895330 | +2.125604 |
| GPU-L | 6,195 | 1,156 | 59.855072 | 5.584541 | 82.895330 | +2.125604 |

H to L changes TPR by -0.077295 pp and FPR by -0.004831 pp. Small aggregate
differences do not satisfy the exact-decision startup gate. The full -10 pp final
guardrail is not breached at checkpoint 425, but this does not authorize a
training restart, replace the gate, or undo the historical dev -8 pp first stop
at step 400. Published historical benchmark results remain unchanged.

## Six pre-registered contrasts

Arrows follow the first source to the second. Hire/reject columns refer to the
frozen reference label. H→R and R→H inside cells refer to model decisions.

| Contrast | Decision flips | Hire H→R | Hire R→H | Reject H→R | Reject R→H | Text differences | In original 52 | Outside original 52 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical cache → CPU-H | 2 | 1 | 0 | 1 | 0 | 3,635 | 1 | 1 |
| Failed startup → GPU-L | 25 | 10 | 13 | 2 | 0 | 3,854 | 12 | 13 |
| CPU-H → GPU-H | 0 | 0 | 0 | 0 | 0 | 841 | 0 | 0 |
| CPU-L → GPU-L | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CPU-H → CPU-L | 53 | 24 | 16 | 7 | 6 | 4,999 | 39 | 14 |
| GPU-H → GPU-L | 53 | 24 | 16 | 7 | 6 | 5,052 | 39 | 14 |

The two H arms agree in every binary decision but differ in generated text on
841 rows. Thus even nominally identical weights and settings do not establish
complete generation determinism in these runs. CPU-L and GPU-L agree in both
decisions and generated text for this execution.

## Original 52 rows, new differences, and same-pair controls

On an original mismatch row, the historical and failed-startup decisions are
opposites. Each new binary decision therefore matches exactly one of them.
The table applies separately to both constructions within the stated context.

| Context | Reference label | Rows | Match historical | Match failed startup |
| --- | --- | ---: | ---: | ---: |
| CPU-H and GPU-H | Hire | 40 | 39 | 1 |
| CPU-H and GPU-H | Reject | 12 | 12 | 0 |
| CPU-H and GPU-H | Total | 52 | 51 | 1 |
| CPU-L and GPU-L | Hire | 40 | 11 | 29 |
| CPU-L and GPU-L | Reject | 12 | 1 | 11 |
| CPU-L and GPU-L | Total | 52 | 12 | 40 |

Switching H to L reproduces 39 original flips in their original direction,
28 hire-reference and 11 reject-reference rows. The other 13 original rows do
not flip: 12 retain the historical decision in both contexts, and one retains
the failed-startup decision in both. Matching either source says nothing by
itself about which hiring label is correct.

The 53 H-to-L flips span 13 pairs/candidates. Their 39 original rows span 12
pairs. The 14 new rows comprise 12 hire and two reject rows across nine pairs;
eight of those pairs were already represented in the original 52 and one is a
new pair. These pair sets overlap and their counts must not be added.

The frozen 52 same-pair agreeing controls comprise 40 hire and 12 reject rows
across 14 pairs. Neither H arm changes a control. Both L arms change the same
one hire-reference control from hire to reject; the other 51 controls retain
their common historical/failed-startup decision. This control is included in
the 14 new H-to-L flips, not an additional observation.

The row-level reports retain all original mismatches, frozen matched controls,
and all new differences. No target row, control, label, split, prompt, or cache
was replaced to improve reproduction.

## Additional cross-job replay check

The old failed job's two completed H arms each had TP 6,202 and FP 1,158. Each
differs from its corresponding new H arm on four decisions. This is an additional
descriptive comparison across jobs, not another cell in the pre-registered
factorial and not a substitute for either live measurement. It is further
evidence that nominal historical settings alone have not secured exact replay.

## Interpretation and next step

The repaired diagnostic now executes successfully. In this run, the exported
weight construction is not the observed source of binary differences. Equal
weights under the two engine/request contexts produce 53 different decisions.
That narrows the investigation to execution context and replay state, but the
residual two-row historical gap and 25-row startup gap prevent a claim that the
original 52 differences have been fully explained.

The next engineering target is a single reproducible evaluation contract for
offline audit and monitoring: identical request ordering/sharding, batching,
warmup/cache handling, numerical settings, and recorded environment. The residual
rows should then be inspected under that contract with one context factor changed
at a time. No further job is submitted by this report. Keep the original targets
and strict gate; do not replace them with these new outputs or introduce a mismatch
tolerance to pass. Training remains disabled.

This inference-only replay excludes HF validation forwards and optimizer-resident
training memory. Historical CUDA/TF32 environment values were not recorded.
The separate smoke process leaves no live engine or request state in the formal
engines, but can warm persistent compiler caches. Those are explicit replay limits.
The results do not identify a particular kernel, establish optimization noise or
a fixed point, validate GRPO training callbacks, or demonstrate fairness gains.

The earlier logging changes passed 227 CPU tests plus 15 subtests. The retry
engineering checks passed 44 CPU tests, and this job subsequently passed the
real GPU smoke and formal inference checks. It generated no new training
effective-reward-group or log-prob/importance-mask statistics; historical scalar
logs cannot reconstruct those stratified statistics.

## Evidence and verification

Remote output:
`/ritom/scratch/cscs/zhang/xw/outputs/grpo_425_diagnostic/startup52_retry1_20261025/`.
The local task workspace contains the verified copy at
`outputs/grpo_425_diagnostic/retry1/`.

The terminal-only evidence package contains 186 files, with no missing expected
files. Archive SHA256:
`07fde061a6f4ce29f1688b7f7e46c3f20905e08910026f599a6f91bdc9bf5430`.
All inventory hashes and 152 downloaded files referenced by experiment receipts
were independently checked. Large model weights and duplicate generation chunks
remain remote; their receipts are preserved.

An independent review re-parsed 279,450 rows with the hash-matched official
parser across the four new cells, three source caches, and two old H arms.
It reproduced all six contrasts and full-denominator metrics. A separate check
compared all 124,200 actual request token sequences to the frozen input sequences.
Derived files include `arm_metrics.csv`, `contrasts.csv`, `mismatch52_status.csv`,
`matched_controls_summary.csv`, `report.html`, `independent_review.json`, and
`completed_evidence_verification.json`.
The original `mismatch52_and_controls.csv` retains raw text and decisions for
the 52 targets and 52 controls. `report_validation.json` records report checks.
HTML structure, links, and data are checked; browser visual inspection was not
performed because the earlier file-URL preview was blocked by the app's policy.

This document records completed results separately from the frozen plan. Only
the fork branch `XinyangWuEthz/hiring_bias_mitigation_at:xinyang/dapo` receives
the results commit. No published historical table or unrelated document changes.
