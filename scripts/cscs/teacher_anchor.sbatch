#!/bin/bash
# The training anchors' procedure rerun on the benchmark's 450 attribute-free pairs: the 122B
# teacher on all four GPUs (tensor parallel 4) with synth.REFERENCE_PROMPT, one draw, eight
# more and a greedy pass per pair, then the CPU summary into reports/teacher_anchor/
# (scripts/teacher_anchor_check.py). The teacher takes about 5 minutes to load.
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/teacher_anchor.sbatch
#
#SBATCH --job-name=xw-teacher-anchor
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "generate: anchor prompt, one draw + 8 sampled + greedy per pair, tensor parallel 4"
$PY scripts/teacher_anchor_check.py --stage generate
step "summarize"
$PY scripts/teacher_anchor_check.py --stage summarize
echo "STAGE teacher-anchor OK"
