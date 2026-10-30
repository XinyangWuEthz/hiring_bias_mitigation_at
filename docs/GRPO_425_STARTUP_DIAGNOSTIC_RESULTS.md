# Checkpoint 425 diagnostic: failure and partial results

Job `5140642` failed after 37m39s with exit code `1:0`. The complete four-cell
experiment did not finish. CPU-H and GPU-H each completed all 31,050 rows;
CPU-L failed before its first experiment weight push, warmup, or benchmark
request, and GPU-L did not start. No training or additional Slurm job was run.

The frozen execution commit was `4b5061a6cc85e45392d2c1ae29ab28735957c2ba`.
The manifest binding is
`02f0ee627287d26e01e3a98129dbc5a5fa74c582d8ccd004cb8dd1046f071753`;
the manifest file SHA256 is
`17801b2f69082e58971faefcd7b6e2648113b76ffa9f86ab72194d86bd20f396`.
The original protocol in `GRPO_425_STARTUP_DIAGNOSTIC.md` is unchanged.

## Why the job failed

The diagnostic's `engine_info` used `dataclasses.asdict` to record the effective
vLLM configuration. After the live engines initialized, `CompilationConfig` held
runtime layer objects in `static_forward_context`. `asdict` recursively deep-copied
a `BasevLLMParameter`. PyTorch's parameter copy passes two positional arguments,
but vLLM's subclass constructor accepts only one, raising:

```
TypeError: BasevLLMParameter.__new__() takes 2 positional arguments but 3 were given
```

All four live ranks reported this error. This is a bug in the added configuration
logging, not evidence of a bad model update or a failed benchmark prediction.
The exception occurred at the frozen script's line 491, called at line 1059;
the first experiment weight push at line 1061 was not reached. The engines had
loaded the base model and performed their internal initialization. Their diagnostic
rank directories are empty, so no live benchmark outputs exist.

A CPU reproduction with the installed PyTorch 2.13 and vLLM 0.31 confirmed the
same exception using one parameter element and a toy module, with no model load
or GPU. The post-failure repair replaces recursive copying with a JSON-safe read
of declared configuration fields. Runtime tensors/modules become explicit type
markers; `static_forward_context` records its count and omission reason. Settings,
weights, requests, and thresholds are unchanged. The repair is not a successful
GPU replay and does not authorize another allocation.

## What the completed checks establish

- The two reconstructed tokenization paths matched for all 31,050 rows. Maximum
  input length was 1,847 tokens. Both completed inference arms also checked their
  actual request token IDs against that frozen sequence.
- CPU and live-style GPU construction produced the same values for all 775 tensors,
  including non-LoRA tensors. The CPU serialized weight files matched the original
  historical merged-model fingerprints.
- SFT merge retention was 99.9996%; checkpoint 425 retention was 96.6887%, above
  the frozen 80% requirement.
- These checks cover reconstructed inputs and exported weights. The old failed
  startup did not save equivalent live served-weight/token receipts, and this
  job failed before producing them. Do not turn these checks into a claim that
  every old runtime state was identical.

## Completed-arm metrics

The unchanged benchmark has 10,350 hire-reference rows and 20,700 reject-reference
rows. Each listed source/arm has 31,050 valid decisions and zero invalid outputs.
The SFT-v3 baseline is 57.729469% TPR and 5.265700% FPR.

| Source / arm | True positives | False positives | TPR % | FPR % | TPR change vs SFT, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical 425 cache | 6,204 | 1,158 | 59.942029 | 5.594203 | +2.212560 |
| Failed retry1 startup | 6,192 | 1,158 | 59.826087 | 5.594203 | +2.096618 |
| CPU-H | 6,202 | 1,158 | 59.922705 | 5.594203 | +2.193237 |
| GPU-H | 6,202 | 1,158 | 59.922705 | 5.594203 | +2.193237 |
| CPU-L | unavailable | unavailable | unavailable | unavailable | unavailable |
| GPU-L | unavailable | unavailable | unavailable | unavailable | unavailable |

CPU-H and GPU-H have identical binary decisions, but their generated texts differ
on 1,109 rows. Each differs from the historical cache on the same four decisions.
Thus the exact historical replay check did not pass, despite equal exported
weights and matching reconstructed inputs. Its TPR difference is only -0.019324 pp,
but aggregate proximity does not satisfy the frozen exact-decision gate.

## The six pre-registered contrasts

| Contrast | Decision differences | Status |
| --- | ---: | --- |
| Historical cache to CPU-H | 4 | available; exact replay fails |
| Failed startup to GPU-L | unavailable | GPU-L not run |
| CPU-H to GPU-H | 0 | available |
| CPU-L to GPU-L | unavailable | live arms missing |
| CPU-H to CPU-L | unavailable | live arm missing |
| GPU-H to GPU-L | unavailable | live arm missing |

Additional descriptive comparisons of saved decisions follow. The arrows are
from the first source to the second; columns are stratified by the unchanged
reference label. These do not replace the four unavailable planned contrasts.

| Comparison | Hire H to R | Hire R to H | Reject H to R | Reject R to H | Total | Text differences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Historical cache to failed startup | 26 | 14 | 6 | 6 | 52 | 5,065 |
| Historical cache to CPU-H | 2 | 0 | 1 | 1 | 4 | 3,628 |
| Historical cache to GPU-H | 2 | 0 | 1 | 1 | 4 | 3,602 |
| Failed startup to CPU-H | 14 | 24 | 6 | 6 | 50 | 4,949 |
| Failed startup to GPU-H | 14 | 24 | 6 | 6 | 50 | 4,984 |
| CPU-H to GPU-H | 0 | 0 | 0 | 0 | 0 | 1,109 |

## The original 52 differences and matched controls

CPU-H and GPU-H classify these rows identically:

| Reference label | Original differences | Match historical decision | Match failed-startup decision |
| --- | ---: | ---: | ---: |
| Hire | 40 | 38 | 2 |
| Reject | 12 | 11 | 1 |
| Total | 52 | 49 | 3 |

The four new historical-replay differences consist of three original mismatch
rows plus one new row outside the original 52. That new row is one of the 52
frozen same-pair agreeing controls: its decision changes from hire to reject.
The other 51 controls retain their common historical/failed-startup decision.
The four rows span three pairs/candidates. All original mismatch rows, controls,
raw responses, and newly changed rows are retained in the partial-result CSVs.
Matching a previous model decision is not a claim that its hiring label is correct.

## Interpretation and next boundary

No CPU/GPU exported-weight difference was observed in this execution. The two
historical-context arms agree in their decisions, but neither exactly reproduces
the old cache. The remaining original 52 disagreements cannot be attributed to
the live engine/request bundle because neither live arm produced measurements.
The specific numerical or kernel cause remains unresolved. There is no evidence
here for a change to GRPO loss/reward, an optimizer update, or a fairness claim.

Keep the exact startup gate, the TPR rules, data, labels, and historical benchmark
tables unchanged. Do not replace the reference with the newer outputs or silently
accept four mismatches. Any further inference allocation needs new user
authorization and a new frozen execution manifest; the failed run is preserved.
No training is authorized. The completed-run renderer still rejects incomplete
inputs; the failure report uses a separate `failed_partial` analysis schema.

## Evidence and validation

The remote evidence is under
`/ritom/scratch/cscs/zhang/xw/outputs/grpo_425_diagnostic/startup52_20261025/`.
A verified download under the task workspace's `outputs/grpo_425_diagnostic/`
contains the original manifest, execution/submission/phase receipts, tokenization
receipts, tensor comparison, merge receipts, both completed raw and request-output
parquets, frozen comparison targets, and Slurm logs. Large model weight shards
were not downloaded. Duplicate chunk JSONs remain remotely; their request-output
parquets and fingerprints are retained locally.

All 51 inventory files were hash-checked, along with the original monitor manifest
and bound source targets. The hash-matched official parser rechecked all 155,250
rows across the two completed arms, historical cache, failed startup, and SFT;
all saved decisions matched parsing. `partial_analysis.json` records provenance,
and `partial_arm_metrics.csv`, `partial_contrasts.csv`, `partial_mismatch52.csv`,
and `partial_matched_controls.csv` contain the derived comparisons. The local
`failure_report.html` explicitly marks two of four arms completed.

The earlier logging change passed 227 CPU tests and 15 subtests. This is not GPU
training-callback validation. This inference job produces no new training reward
group or importance-mask statistics; old scalar logs cannot recover them.

The engine-snapshot repair passed all 26 inference-diagnostic CPU tests in the
installed environment, including an actual `CompilationConfig` containing a
`BasevLLMParameter`. The regression verifies unchanged parameter bytes, storage
address, object identity, and CUDA initialization state. No GPU rerun was made.
