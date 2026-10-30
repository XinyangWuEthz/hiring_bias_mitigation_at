# Checkpoint 425 startup disagreement: inference-only diagnostic

## Authorization and question

The user authorized one inference-only Slurm job on 2026-10-25 to investigate the 52
decision disagreements at the start of lower-LR retry1, job 5131944. No optimizer
update ran in that failed job. This experiment does not authorize training,
resubmission, a different checkpoint, or replacement of the frozen startup target.
Code and this plan are committed before preparing the bound manifest and submitting.

The existing comparison verified the same saved prompt text, row identities,
reference labels, source adapters, and official parsing. It did not establish
identity of rendered input token IDs or of the live merged weight bytes.

Historical checkpoint 425 produced 6,204 hires on 10,350 positive rows; retry1
produced 6,192. Both produced 1,158 hires on 20,700 negative rows. The 52 flips
comprise 40 positive rows and 12 negative rows across 14 pairs. Small aggregate
differences do not satisfy the frozen exact-decision startup gate.

## Fixed inputs

- All 31,050 rows of the retry1 frozen full benchmark, in its original order.
- The original SFT-v3 parent and original GRPO checkpoint 425 adapter. Merge the
  SFT parent once. No checkpoint average or factor average.
- Original historical 425 outputs and failed retry1 startup outputs are read-only
  comparison targets. Preserve their file hashes in the manifest.
- Greedy generation, temperature 0, top-p 1, maximum 512 new tokens, thinking off,
  official parser, unchanged labels and candidate splits.
- Inspect every row. The original 52 disagreements and agreeing rows from their
  pairs are reporting subsets, not a replacement for the full replay.

Source hashes, installed package versions, base-model files, input identities,
execution code, engine settings, and the preparation commit belong in the new
experiment manifest. Outputs must be new files under
`outputs/grpo_425_diagnostic/startup52_20261025/`.

## Controlled comparisons

First compare the two production tokenization paths for every row, including
rendered text, special-token behavior, and input IDs. Preserve any differences;
do not silently rewrite either path to make the check pass.

Construct two weight sets using the inspected production paths: the historical
CPU FP32 merge followed by bf16 storage, and the live serving merge path. Compare
the final tensors by name, shape, dtype, SHA256, differing element count, and
maximum numerical difference, including tensors without LoRA. The live path must
use the same SFT-parent construction and adapter dtype as the failed retry.
Matching merge-retention summaries alone is not byte identity.

Run the following four full-bank cells inside one four-GPU allocation:

| Cell | Weight construction | Engine and request layout |
| --- | --- | --- |
| CPU-H | Historical CPU merge | Historical settings; contiguous chunks of 4,000 |
| GPU-H | Live serving merge | Historical settings; contiguous chunks of 4,000 |
| CPU-L | Historical CPU merge | Live settings; four strided rank shards |
| GPU-L | Live serving merge | Live settings; four strided rank shards |

Historical settings use TP1, maximum 256 sequences, 16,384 batched tokens, model
length 8,192, and memory utilization 0.8. Live settings use one TP1 engine per
rank, maximum 64 sequences, 4,096 batched tokens, model length 4,096, and memory
utilization 0.3, with the inspected external-launcher and weight-push path.
Record effective settings, hardware, numerical flags, and weight-loading receipts.

The historical cells run sequentially on the same GPU; the live cells run
sequentially using the same four-rank GPU mapping. This keeps the weight comparison
from also changing the assigned GPU. The job has a two-hour limit on normal and
Requeue=0. Earlier full audits took about 14–16 minutes per model including merge;
55–90 minutes for this diagnostic is a planning estimate, not a deadline guarantee.
No additional Slurm job or automatic retry is permitted.

## Evidence and interpretation

For each cell report full coverage, invalid decisions, and TPR/FPR with explicit
denominators. Report paired decision and raw-text disagreements for the six
comparisons: historical cache to CPU-H, failed startup to GPU-L, and the four
weight-construction and engine-layout contrasts below.
For the original 52 rows, retain all raw texts and parsed decisions and classify
which flips reproduce. Also list newly introduced disagreements and same-pair
agreeing controls. Count directions separately for reference hire/reject.

Compare CPU-H with GPU-H and CPU-L with GPU-L to measure the effect of weight
construction under held engine settings. Compare CPU-H with CPU-L and GPU-H
with GPU-L to measure the bundled engine/request-layout effect at held weights.
The latter does not separately identify batch size, sharding, memory allocation,
or a particular kernel. Do not claim that a numerical cause was proved solely
because aggregate TPR is similar.

If CPU-H does not reproduce the historical cache, report the remaining replay
disagreement before interpreting the four cells. If GPU-L does not reproduce the
failed startup, the diagnostic has not fully recreated that serving environment.
No Trainer, optimizer, backward pass, or training rollout is instantiated; the
absence of resident training state may therefore remain an explicit difference.
The live cells repeat the original vLLM-only 256-row forced-decision warmup and
weight reload/cache reset, but do not repeat its HF validation forwards. Preserve
the production sampling-parameter distinction: historical greedy requests omit
the per-request seed; the live monitor supplies seed 42. Neither changes the
temperature-zero decoding rule. Record the process's float32 matmul setting and
the vLLM environment override; the old job did not save a numerical-mode receipt.
An exact match in a new inference run does not by itself validate the original
training startup callback or authorize resuming it.

Persist phase receipts and raw outputs as each phase finishes. A failed or timed-out
job yields partial diagnostic evidence, not a completed causal finding. Preserve
that evidence and report the failure without resubmitting. Keep all old frozen
manifests, startup targets, training-authorization flags, and published benchmark
tables unchanged. Commit results separately and push only the user's fork branch
`XinyangWuEthz/hiring_bias_mitigation_at:xinyang/dapo`.
