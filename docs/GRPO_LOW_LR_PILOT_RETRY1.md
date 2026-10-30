# GRPO lower-LR pilot: authorized retry 1

On 2026-10-23 the user explicitly confirmed, translated from Chinese: "425→500, LR `3e-6`, stop training if full TPR drops by 8 pp." This authorizes one replacement job after attempt5130709 failed before any optimizer update or benchmark inference.

The experimental protocol in `GRPO_LOW_LR_PILOT.md` is unchanged. The correction in commit `d2cd1ae55d9c02cd2933d167acd2351cb6693902` synchronizes the ranks before writing W&B metadata. It preserves the strict fresh-directory check and does not alter learning, inference, data, labels or stopping rules. The corrected code passed 102 CPU tests, including four real multiprocessing regression tests. The GPU startup gate is still required.

- Start from the original checkpoint425, preserving Adam/scheduler/HF RNG/data position. Continue for at most 75 updates to absolute step500, at constant LR3e-6.
- Keep the original prompt, 9,140 rows, SFT-v3 reference, LoRA, reward and sampling settings.
- Before updates, require all 31,050 serving-model decisions to match the frozen checkpoint425 cache.
- Every five steps, directly score all 10,350 full-benchmark positive rows. The first TPR decline of at least 8 pp relative to SFT-v3 saves and irreversibly stops training. Preserve the preceding safe checkpoint and trigger checkpoint.
- Score the full benchmark at450/475/500 and any trigger; independently audit the terminal merged model. The original full10pp final guardrail, including equality, remains unchanged.
- Keep the original historical comparisons and all limitations: this is an exploratory continuation, not a restarted original-LR control or a fresh independent benchmark test.

Retry paths are separate from the failed attempt:

| Purpose | Path under `HBM_OUTPUT_ROOT` |
| --- | --- |
| Training | `outputs/grpo/qwen3.5-9b_en_only_grpo_v3_all_lr3e6_from425_75step_retry1` |
| Experiment | `outputs/grpo_low_lr_pilot/grpo_v3_all_lr3e6_from425_retry1_20261023` |
| New monitor contract | Experiment directory plus `monitor_bank/manifest.json` |

The new monitor manifest freezes the corrected source and unchanged inputs. A new outer run manifest pins the final config, commit and this authorization. The previous run, manifests and failure evidence remain intact.

Submit exactly one normal-partition job on four GH200 GPUs with a two-hour limit and Slurm automatic requeue disabled. Do not submit another retry automatically if it fails or times out. Expected runtime remains about80–110minutes after allocation; scheduling time is separate.
