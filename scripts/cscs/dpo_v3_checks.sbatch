#!/bin/bash
# V3's DPO set, checked per pool source (docs/V3_NEW_DATA.md, round 2), one source per GPU.
# `scripts/v3_additional_data.py --stage dpo_v3` writes each pool source's training pairs to
# dpo_v3_checks/<pool source>/pairs.parquet. This runs scripts/check_preference_direction.py on
# each: D1-D3 and D5-D6 on every pair, D4 under SFT-v3 and the base (HF + PEFT) on a 600-pair
# subsample, with direction_report.{json,md} written beside the pairs. r2em_v3's pairs
# (djinni_v1) are the control, checked under the same student.
#
#   HBM_COST_ARM=dpo_v3 REPO=$PWD scripts/cscs/submit.sh -p normal -t 01:00:00 \
#     scripts/cscs/dpo_v3_checks.sbatch
#
# Round 3's on-policy set (docs/V3_NEW_DATA.md) names its own directory:
#   HBM_COST_ARM=dpo_v3_5k REPO=$PWD CHECKS=artifacts/semisynthetic-v3/new-pairs-2/dpo_v3_5k_checks \
#     scripts/cscs/submit.sh -p normal -t 01:00:00 scripts/cscs/dpo_v3_checks.sbatch
#
#SBATCH --job-name=xw-dpo-v3-checks
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
CHECKS=${CHECKS:-artifacts/semisynthetic-v3/additional/dpo_v3_checks}
STUDENT=${STUDENT:-outputs/sft/qwen3.5-9b_en_only_sft_v3_additional}
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | student $STUDENT | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "config checks $CHECKS"

shopt -s nullglob
dirs=("$XW/$CHECKS"/*/)
shopt -u nullglob
if [ ${#dirs[@]} -lt 1 ] || [ ${#dirs[@]} -gt 4 ]; then
  echo "FATAL: expected 1 to 4 pool sources under $XW/$CHECKS, found ${#dirs[@]}"; exit 2
fi

step "direction checks, one pool source per GPU"
pids=(); names=(); fail=0; g=0
for dir in "${dirs[@]}"; do
  name=$(basename "$dir")
  CUDA_VISIBLE_DEVICES=$g cost datagen "direction-check $name" \
    $PY scripts/check_preference_direction.py --pairs "${dir}pairs.parquet" \
    --student "$STUDENT" --also-base --batch-size 4 --student-sample 600 \
    > "$XW/logs/xw-dpo-v3-checks-${SLURM_JOB_ID:-local}-$name.log" 2>&1 &
  pids+=($!); names+=("$name"); g=$((g + 1))
done
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "FAILED: ${names[$i]}"; fail=1; }; done
for name in "${names[@]}"; do
  echo "--- $name"; tail -n 3 "$XW/logs/xw-dpo-v3-checks-${SLURM_JOB_ID:-local}-$name.log"
done
[ "$fail" -eq 0 ] || { echo "FATAL: a direction check failed -- see the logs above"; exit 1; }
echo "STAGE dpo_v3_checks OK"
