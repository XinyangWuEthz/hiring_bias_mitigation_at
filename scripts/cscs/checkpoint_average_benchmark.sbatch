#!/bin/bash
# One inference-only job; only the effective-delta mean model needs new generations.
# SFT and all seven source checkpoints reuse the frozen previous trajectory raw outputs.
# Expected 20-40 minutes including merge/generation/CPU analysis; hard limit 60 minutes.
# Four-GH200 node allocation, one TP1 engine on GPU0. No training or nested submission.
#SBATCH --job-name=xw-grpo-mean7
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out
set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at-dapo}
source "$REPO/scripts/cscs/env.sh"
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} commit $(git rev-parse HEAD)"
AVERAGE_DIR=${AVERAGE_DIR:-$XW/outputs/checkpoint_average/grpo_v3_all_delta_avg_425_571_20261023}
cost eval grpo-v3-all-effective-delta-mean7 uv run --no-project \
  --python "$XW/venv/bin/python" python scripts/checkpoint_average_benchmark.py run \
  --output-dir "$AVERAGE_DIR"
