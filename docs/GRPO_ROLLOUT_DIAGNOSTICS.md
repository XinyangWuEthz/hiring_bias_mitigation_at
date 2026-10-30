# GRPO rollout diagnostics

The old `sampled_hire_rate_ref_hire` and `sampled_hire_rate_ref_reject` averaged
rank-local proportions. Ranks can have different numbers of rows with a given
reference label, so that average is not the pooled sampled hire rate. This is a
logging defect; it does not change the reward returned for a completion or the
independently measured benchmark TPR/FPR.

For example, local counts of 1/1 and 0/9 must pool to 1/10, or 10%. Averaging
the two local proportions would incorrectly report 50%.

The replacement accumulates hire counts and reference-label counts across ranks
and across the logging window, then divides once. A stratum with no observations
has no defined rate. Preserve the counts so the rate can be recomputed. Do not
average batch proportions or interpret these training samples as benchmark TPR.

## What counts as an effective reward group

One group is one prompt's 16 sampled completions from a single rollout. Group
boundaries follow TRL's global generation order, including groups split across
ranks. A repeated prompt in a later rollout is a different group. Validation is
logged separately and uses its actual `num_generations_eval`; a single-completion
validation group has zero reward standard deviation, as in TRL.

Report separately:

- All decisions correct, a mixture of correct and incorrect decisions, or all
  decisions incorrect relative to the unchanged anchor.
- Groups with total-reward variation and groups with nonzero actual advantage.
- Groups whose only reward variation is format reward. These are not evidence
  of a decision-correction signal.
- Groups with nonzero advantage that retain at least one unmasked policy-loss
  token after the actual importance-sampling correction.
- Groups whose policy signal is entirely removed by that correction.

These counts distinguish anchor-correctness variation, total-reward variation,
and the signal left after masking. A nonzero advantage is an available policy
signal, not proof of a nonzero gradient or a useful future update. Even when every
advantage is zero or the policy term is masked, KL and optimizer state can still
affect training.

## Log-prob correction and denominators

The original GRPO-v3-all run used TRL 1.15 `sequence_mask`, maximum ratio 3, and
no minimum ratio. Before clipping, a completion's importance weight is

`q = exp(sum_valid_tokens(old_HF_logp - sampling_vLLM_logp))`.

For `q > 3`, its policy term receives weight zero. Otherwise it receives `q`,
not a binary keep flag and not a weight clamped to 3. The KL term is added
separately and remains active. The diagnostic observes the actual tensors used
by TRL; it must not make another model forward pass or reconstruct a different
loss. Report threshold-masked completion counts over nonempty completions and
masked valid token counts over valid completion tokens, together with total,
nonempty, empty, and missing-log-prob counts. Zero importance weights caused by
numerical underflow are counted separately from threshold masking.
The existing post-mask ratio mean cannot recover either masking fraction.

## Anchor strata and provenance

Detailed strata are reference hire/reject crossed with winning anchor votes and
decided draws, for example `hire/5_of_8`. Do not assume eight decided draws: the
frozen dataset contains some six- and seven-decision anchors. This metadata is
joined from the two frozen source pools and vote-anchor files before prompt
rendering. It does not change rewards or weight examples by vote confidence.

The join accepts only the exact frozen ordered GRPO-v3-all split, checks source
file fingerprints, verifies row identities, prompt text, labels, and strict vote
majorities, and rejects missing or conflicting records. Metadata is added in
memory. Original parquet bytes and original columns remain unchanged. The source
receipt contains separate training and validation joins, with hashes before and
after each join and counts for every stratum.
This verifies provenance, not label truth. The prior finding that RJDB reject
anchors are questionable remains unresolved.

The read-only check on 2026-10-25 verified all 9,140 training rows and 256
validation rows and found the following training strata. These are dataset
counts, not sampled effective-group counts:

| Winning votes / decided draws | Hire rows | Reject rows |
| --- | ---: | ---: |
| 4/6 | 0 | 4 |
| 5/7 | 0 | 9 |
| 5/8 | 956 | 705 |
| 6/8 | 974 | 1,009 |
| 7/7 | 0 | 7 |
| 7/8 | 882 | 596 |
| 8/8 | 1,758 | 2,240 |
| Total | 4,570 | 4,570 |

## Deployment boundary

The pooled hire-rate fix applies to future GRPO runs by default. Detailed rollout
diagnostics require explicit configuration and the frozen metadata contract. Old
logs are not rewritten, and their missing masking fractions cannot be recovered
from existing scalar summaries. The current change authorizes no training and
does not modify frozen prior experiment configurations or authorization flags.

For a separately authorized future run, enable the detailed counters with:

```yaml
rollout_diagnostics:
  enabled: true
```

The supported detailed path is the original TRL GRPOTrainer with groups of 16,
`loss_type=dapo`, group scaling, sum-then-normalize rewards, and
`sequence_mask` with maximum 3 and no minimum. Other policy masks, tools, and
DAPO's custom sampling trainer are explicitly rejected in detailed mode rather
than assigned misleading signal-retention statistics. Basic count-weighted hire
logging still covers actual reward-call rows for the custom DAPO trainer; it
does not count its unscored candidate probes.

Read `<run output>/rollout_diagnostics.jsonl` for each log window's raw counts,
strata, and rates. Null rates mean no denominator; backend scalar logs omit those
rates. `run_config.json` includes the diagnostics receipt and anchor provenance.
Useful detailed keys are `effective_adv_groups`, `eligible_effective_adv_groups`,
`retained_effective_adv_groups`, `format_only_effective_groups`,
`is_threshold_masked_completion_fraction`, and `is_threshold_masked_token_fraction`.
`fully_lost_effective_adv_groups` includes empty completions; use
`fully_threshold_masked_eligible_adv_groups` to isolate complete loss through the
importance threshold. An empty final runtime-summary window does not overwrite
the previous sampled rates.

CPU tests check aggregation, distributed group ordering, denominators, masking,
source conflicts, and unchanged returned training tensors. These are not a GPU
training-callback validation. The separate checkpoint 425 job is inference only;
it will not produce new training reward-group statistics.
