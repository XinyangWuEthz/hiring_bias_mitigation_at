#!/bin/bash
# Authorized repaired retry. Isolated GPU smoke, then four full-bank cells; no training.
# Expected 60–100 minutes on one four-GH200 node; preserve partial evidence on failure.
#SBATCH --job-name=xw-grpo425-diag-r1
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --no-requeue
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
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export HBM_COST_ARM=grpo425-startup-factorial-retry1
OUTPUT_DIR=outputs/grpo_425_diagnostic/startup52_retry1_20261025
cd "$REPO"
cost eval grpo425-startup-factorial-retry1 uv run --no-project --python "$PY" python \
  scripts/grpo_425_diagnostic.py run --output-dir "$OUTPUT_DIR"
