#!/bin/bash
# One job evaluates up to 18 missing checkpoints with four independent TP1 workers. Never submits jobs.
# Resubmission resumes the same frozen outputs after a timeout; failures require review.
# REPO=<checkout> scripts/cscs/submit.sh scripts/cscs/checkpoint_benchmark_trajectory.sbatch
#SBATCH --job-name=xw-grpo-trajectory
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
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
cd "$REPO"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} commit $(git rev-parse HEAD)"
git diff --quiet -- scripts/checkpoint_benchmark_trajectory.py scripts/cscs/checkpoint_benchmark_trajectory.sbatch ${TRAJECTORY_CONFIG:-configs/analysis/grpo_v3_all_checkpoint_trajectory.json} src/hiring_bias_mitigation || { echo 'Uncommitted inference code'; exit 2; }
TRAJECTORY_DIR=${TRAJECTORY_DIR:-$XW/outputs/checkpoint_benchmark/grpo_v3_all_20261023}
# SCORE_STEP=<n> scores that one checkpoint on GPU 0 (e.g. after another one failed a gate).
if [ -n "${SCORE_STEP:-}" ]; then
  cost eval ${TRAJECTORY_LABEL:-grpo-v3-all-checkpoint-trajectory} env CUDA_VISIBLE_DEVICES=0 \
    uv run --no-project --python "$XW/venv/bin/python" python \
    scripts/checkpoint_benchmark_trajectory.py score-next --output-dir "$TRAJECTORY_DIR" \
    --step "$SCORE_STEP"
  "$XW/venv/bin/python" scripts/checkpoint_benchmark_trajectory.py summarize \
    --output-dir "$TRAJECTORY_DIR"
else
  cost eval ${TRAJECTORY_LABEL:-grpo-v3-all-checkpoint-trajectory} uv run --no-project \
    --python "$XW/venv/bin/python" python scripts/checkpoint_benchmark_trajectory.py run \
    --output-dir "$TRAJECTORY_DIR"
fi
