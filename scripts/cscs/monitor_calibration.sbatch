#!/bin/bash
# The margin monitor's transfer check (scripts/monitor_calibration.py): each merged model's
# margins on a build's monitor rows, one model per GPU and four at a time, then `transfer` (CPU),
# which sets each arm's monitor projection against its TPR change on the benchmark.
#
#   MODELS="SFT-v3=outputs/merged/qwen3.5-9b_en_only_sft_v3_additional \
#           DAPO-v3-all=outputs/merged/qwen3.5-9b_en_only_dapo_v3_all" \
#   PARENT_RUN=Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional PARENT_LABEL=SFT-v3 \
#   ARMS="DAPO-v3-all=Qwen3.5-9B--en--grpo--adapter--en_only_dapo_v3_all" \
#     scripts/cscs/submit.sh scripts/cscs/monitor_calibration.sbatch
#
# MODELS paths are under $XW. ROWS defaults to grpo-v3-all's monitor rows. From a worktree,
# export REPO=<worktree> before submit.sh.
#
#SBATCH --job-name=xw-monitor-cal
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
ROWS=${ROWS:-artifacts/semisynthetic-v3/grpo-v3-all/grpo_monitor.parquet}
SCORES_DIR=${SCORES_DIR:-$XW/outputs/monitor_calibration}
read -r -a models <<< "${MODELS:?set MODELS}"
read -r -a arms <<< "${ARMS:?set ARMS}"
: "${PARENT_RUN:?set PARENT_RUN}" "${PARENT_LABEL:?set PARENT_LABEL}"
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"

step "margins on $ROWS: ${#models[@]} models, one per GPU, four at a time"
for ((i = 0; i < ${#models[@]}; i += 4)); do
  pids=()
  for ((g = 0; g < 4 && i + g < ${#models[@]}; g++)); do
    entry=${models[i + g]}
    label=${entry%%=*}
    CUDA_VISIBLE_DEVICES=$g cost eval monitor-calibration \
      $PY scripts/monitor_calibration.py score --label "$label" --model "$XW/${entry#*=}" \
      --rows "$ROWS" --gpu-memory-utilization 0.6 \
      --scores-dir "$SCORES_DIR" \
      > "$XW/logs/xw-monitor-cal-${SLURM_JOB_ID:-local}-$label.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "FATAL: a scoring process failed -- see $XW/logs/xw-monitor-cal-${SLURM_JOB_ID:-local}-*.log"; exit 1; }
  done
done
grep -h "margins, hire share" "$XW"/logs/xw-monitor-cal-"${SLURM_JOB_ID:-local}"-*.log || true

step "transfer: monitor projection against the benchmark"
arm_args=()
for a in "${arms[@]}"; do arm_args+=(--arm "$a"); done
$PY scripts/monitor_calibration.py transfer --parent "$PARENT_RUN" \
  --parent-label "$PARENT_LABEL" --scores-dir "$SCORES_DIR" "${arm_args[@]}"
echo "STAGE monitor-calibration OK"
