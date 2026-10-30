#!/bin/bash
# DAPO (mitigation/dapo.py) on one GH200 node, four GPUs under torchrun (DDP), each GPU with its
# own colocated vLLM engine. With `resume: true` the run continues from its newest finished
# checkpoint, so a schedule longer than the time limit is the same job submitted again, one at a
# time:
#
#   CONFIG=configs/mitigation/dapo/<x>.yaml scripts/cscs/submit.sh -p normal -t 06:00:00 \
#     scripts/cscs/train_dapo.sbatch
#
# From a worktree, export REPO=<worktree> before submit.sh: env.sh keeps a preset REPO, this
# script sources env.sh from it, and $REPO/src goes first on PYTHONPATH, ahead of the venv's
# editable install of the main checkout.
#
#SBATCH --job-name=xw-dapo
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
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
mkdir -p "$WANDB_DIR"
CONFIG=${CONFIG:-configs/mitigation/dapo/qwen3.5-9b_en_only_dapo_v3_all_gh200.yaml}
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | config $CONFIG | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
$PY -c "import hiring_bias_mitigation as m; print('package', m.__file__)"

step "W&B: upload offline runs not synced yet"
wandb_upload_offline

step "DAPO on 4 GPUs (DDP, colocated vLLM); resumes from the newest finished checkpoint"
cost train dapo $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m hiring_bias_mitigation.mitigation.dapo --config "$CONFIG"
echo "STAGE dapo OK"
