#!/bin/bash
# Round-2 DPO data balanced by error mass (scripts/build_balanced_dpo.py), one GPU stage per job.
# The `candidates` and `build` stages take seconds on CPU and run on the login node.
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/build_balanced_dpo.sbatch judge
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/build_balanced_dpo.sbatch score
#
# judge  the 122B teacher on four GPUs, on the legacy (semisynthetic-v1) pairs' rationales
# score  the SFT student's decision margin on every candidate prompt, one GPU
#
# ARTIFACTS, NAME (the file prefix, `--name`) and STUDENT come from the environment. V3's
# on-policy pairs (docs/V3_NEW_DATA.md, round 2):
#   HBM_COST_ARM=dpo_v3 REPO=$PWD ARTIFACTS=artifacts/semisynthetic-v3/onpolicy-error-mass \
#   NAME=dpo_v3_onpolicy STUDENT=outputs/sft/qwen3.5-9b_en_only_sft_v3_additional \
#     scripts/cscs/submit.sh -p normal -t 01:00:00 scripts/cscs/build_balanced_dpo.sbatch score
#
#SBATCH --job-name=xw-bdpo
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
STAGE=${1:?usage: build_balanced_dpo.sbatch judge|score}
ARTIFACTS=${ARTIFACTS:-artifacts/semisynthetic-v2/round2-error-mass}
NAME=${NAME:-dpo_round2}
STUDENT=${STUDENT:-outputs/sft/qwen3.5-9b-hiring-debias-sft-en}
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

case "$STAGE" in
  judge)
    step "judge: paraphrased attribute references, teacher on four GPUs"
    cost datagen judge $PY scripts/build_balanced_dpo.py judge --artifacts-dir "$ARTIFACTS" \
      --name "$NAME" \
      --tensor-parallel-size 4 --gpu-memory-utilization 0.80 --max-num-seqs 128 \
      --gdn-prefill-backend triton
    ;;
  score)
    step "score: SFT decision margins ($STUDENT)"
    CUDA_VISIBLE_DEVICES=0 cost datagen score \
      $PY scripts/build_balanced_dpo.py score --artifacts-dir "$ARTIFACTS" --name "$NAME" \
      --student "$STUDENT" --batch-size 4
    ;;
  *)
    echo "FATAL: unknown stage $STAGE"; exit 2 ;;
esac
echo "STAGE $STAGE OK"
