#!/bin/bash
# DPO on one GH200 node, four GPUs under torchrun (DDP). With `resume: true` in the config the
# run continues from its newest finished checkpoint, so a schedule longer than debug's 30
# minutes is the same job submitted again, one submission at a time:
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/train_dpo.sbatch
#   CONFIG=configs/mitigation/dpo/<x>.yaml scripts/cscs/submit.sh -p debug -t 00:30:00 \
#     scripts/cscs/train_dpo.sbatch
#
#SBATCH --job-name=xw-dpo
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
mkdir -p "$WANDB_DIR"
CONFIG=${CONFIG:-configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1_gh200.yaml}
# `-m` imports the package, not a script that puts src/ first: without this, a run from a
# worktree (REPO=<worktree>) trains on the venv's editable install, the main checkout's code,
# while its log names the worktree's commit. train_sft.sbatch does the same.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | config $CONFIG | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
$PY -c "import hiring_bias_mitigation as m; print('package', m.__file__)"

# Earlier segments' offline runs go up first, so the resumed online run continues one curve.
step "W&B: upload offline runs not synced yet"
wandb_upload_offline

step "DPO on 4 GPUs (DDP); resumes from the newest finished checkpoint if there is one"
cost train dpo $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m hiring_bias_mitigation.mitigation.dpo --config "$CONFIG"
echo "STAGE dpo OK"
