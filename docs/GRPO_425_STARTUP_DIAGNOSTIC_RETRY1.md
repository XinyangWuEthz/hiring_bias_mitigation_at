# Checkpoint 425 diagnostic: repaired inference retry

The user asked on 2026-10-25 to resolve the repeated failures autonomously.
This authorizes the next inference-only retry described here. It does not
authorize training. Submit one normal-partition allocation with four GH200 GPUs,
a two-hour limit and Requeue=0. Do not run another diagnostic job concurrently.
Keep changes on `XinyangWuEthz/hiring_bias_mitigation_at:xinyang/dapo`.

## Why the earlier runs stopped

- Lower-LR job 5130709 failed in distributed startup before updates. Its startup
  coordination fix is separate from this inference experiment.
- Lower-LR retry 5131944 stopped because its exact startup gate found 52 decision
  disagreements at checkpoint 425. This was a failed reproduction check; no
  optimizer update occurred. The gate and both comparison targets remain frozen.
- Diagnostic job 5140642 completed CPU-H and GPU-H, then failed while recording
  live engine configuration. `dataclasses.asdict` deep-copied a vLLM runtime
  parameter whose constructor is incompatible with that copy operation.
  Commit 60a3050 replaced that traversal with a read-only configuration snapshot.
  Its 26 CPU tests do not establish GPU execution success.

The engineering shortfall was insufficient coverage of the complete live
execution path. The retry adds an isolated GPU startup check before the expensive
formal inference cells, as well as CPU file-chain and installed-API checks.

## Frozen scientific question and inputs

Retain the original protocol in `docs/GRPO_425_STARTUP_DIAGNOSTIC.md`, all 31,050
rows, labels, original 52 disagreements, same-pair controls, checkpoint 425 and
SFT parent, original 256-row warmup, seeds, engine arguments, request layouts,
official parser, and six contrasts. The startup gate remains exact and is not
replaced with the new outputs. No Trainer, optimizer, backward or training rollout
is instantiated. The full replay still omits resident training state and HF
validation forwards, as documented for the original diagnostic.

Use a new directory:
`outputs/grpo_425_diagnostic/startup52_retry1_20261025/`.
Bind the original failed manifest, progress and execution receipts, with parent
job 5140642 and parent execution commit 4b5061a. Preserve every old output and
receipt. Freeze this plan, repaired runtime code and retry sbatch before preparing
the new manifest. The current runtime HEAD must stay fixed until the job ends.

Reconstruct tokenization and both weight sets and compare all tensors. Run all
four formal cells again in the new allocation. Do not splice the old historical
cells into the new factorial comparison: the old same-weight historical replay
already differed by four decisions, so a cross-job comparison could introduce
another explanation. The old two completed cells remain descriptive repeatability
evidence. New CPU-H/GPU-H share one GPU; new CPU-L/GPU-L share the same four-rank
mapping. The common SFT parent is merged once per constructed weight set.

## Isolated GPU startup check

After tokenization, construction and weight comparison, run a separate four-rank
`torchrun` process group with the exact live engine arguments and CPU-constructed
weights. Exercise configuration recording, the first complete weight push, the
original 256-row forced-decision warmup, second complete push/cache reset,
served-parameter hashing, then greedy generation for exactly the first frozen
canonical row of each rank's `rows[rank::4]` shard. These four request identities
are fixed before observing the outputs. Require correct input token IDs, complete
rank receipts and parseable binary decisions, not a particular hire/reject answer.

Save this evidence only under `smoke/`. It is not a fifth benchmark arm and is
excluded from all benchmark metrics. All four smoke processes must exit before
the runner starts the formal cells in their original order, with new engines.
Each formal live engine still performs exactly its original two weight pushes,
256-row warmup and full strided request bank. No smoke request is inserted into
a formal engine. Persistent compiler caches may have been warmed by the smoke;
process isolation does not establish identical low-level hardware/cache state.

Write a current stage receipt per rank so a failure can be localized to engine
construction, first push, warmup, second push, parameter receipt or generation.
Seal this receipt before fingerprinting the completed rank result. Require the
actual boolean acknowledgement from prefix-cache reset; a failed reset stops the
phase instead of recording an unverified success.

## Completion and interpretation

Only four complete formal arms with 31,050 rows each, zero invalid decisions,
verified actual input IDs, weight and source checks count as complete. Record the
same six contrasts by hire/reject and original-52/control membership. If CPU-H
still differs from the historical cache or GPU-L from the failed startup, report
the remaining mismatch explicitly. The factorial can localize a whole engine /
request-context difference; it cannot by itself identify a specific kernel or
prove the cause of all 52 historical disagreements.

The runner never submits or requeues. A failed phase stops the allocation and
preserves evidence for diagnosis; it does not automatically launch another job.
No success or failure in this inference test authorizes training or relaxes the
strict startup gate. Report the actual GPU result separately from CPU test passes.

After completion, write a separate results document, preserve the published
historical benchmark tables, and push only the fork branch. Do not edit
`docs/FRONTIER_RL_SURVEY.md`.

## CPU validation before submission

The actual installed environment passed 44 CPU tests, including the complete
31,050-row synthetic file chain, four-rank smoke isolation, 427 frozen names loaded
twice through installed Qwen/vLLM name-routing code, actual sampling/logprob
containers, and cache-reset failure handling. The loader stand-ins are one-element
CPU parameters; these tests do not validate full-size CUDA packing or generation.
The separate GPU smoke is still required.
