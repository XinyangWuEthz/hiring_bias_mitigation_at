#!/bin/bash
#SBATCH --job-name=xw-paired-probe
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
export REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at-dapo}
source "$REPO/scripts/cscs/env.sh"
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
export PATH="$XW/venv/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_USE_V2_MODEL_RUNNER=0
export OMP_NUM_THREADS=4
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HBM_COST_ARM=paired_probe_dapo_sft_v3
: "${RUN_DIR:?RUN_DIR must name the prepared experiment directory}"
cd "$REPO"
echo "job=$SLURM_JOB_ID commit=$(git rev-parse HEAD) run=$RUN_DIR"
pids=()
cleanup() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup TERM INT
for gpu in 0 1 2 3; do
  if ((gpu < 2)); then model=sft_v3; else model=dapo_final; fi
  shard=$((gpu % 2))
  CUDA_VISIBLE_DEVICES=$gpu cost eval "paired-probe-$model-$shard" \
    "$PY" scripts/paired_probe.py sample --run "$RUN_DIR" --model "$model" --shard "$shard" \
    > "$RUN_DIR/$model-shard-$shard.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if ((failed)); then echo "FAILED: a sampler failed; saved chunks remain resumable"; exit 1; fi
"$PY" scripts/paired_probe.py analyze --run "$RUN_DIR"
echo "STAGE paired-probe OK"
