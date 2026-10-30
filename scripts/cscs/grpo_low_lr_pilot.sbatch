#!/bin/bash
# One authorized bounded training pilot, followed by one terminal inference audit.
# Do not resubmit automatically after a stop, failure, or timeout.
#SBATCH --job-name=xw-grpo-lr3e6
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
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
export HBM_COST_ARM=grpo-v3-all-lr3e6-from425
CONFIG=configs/mitigation/grpo/qwen3.5-9b_en_only_grpo_v3_all_lr3e6_from425_gh200.yaml
cd "$REPO"
echo "host $(hostname) job ${SLURM_JOB_ID:-none} commit $(git rev-parse HEAD) config $CONFIG"
uv run --no-project --python "$PY" python scripts/grpo_low_lr_pilot_results.py preflight --config "$CONFIG"
cost train grpo-v3-all-lr3e6-from425 uv run --no-project --python "$PY" python \
  -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m hiring_bias_mitigation.mitigation.grpo --config "$CONFIG"
export CUDA_VISIBLE_DEVICES=0
cost eval grpo-v3-all-lr3e6-terminal-audit uv run --no-project --python "$PY" python \
  scripts/grpo_low_lr_pilot_results.py finalize --config "$CONFIG"
