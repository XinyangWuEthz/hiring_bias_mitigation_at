#!/bin/bash
# Benchmark audits on one GH200 node, one config per GPU (up to four), each through vLLM with
# adapters served as merged weights (merge.py merges a DPO checkpoint's SFT parent first).
# Generations are cached in chunks under outputs/raw, so a run cut by the time limit continues
# where it stopped when the same job is submitted again.
#
#   CONFIGS="configs/audit/a.yaml configs/audit/b.yaml" \
#     scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/run_audits.sbatch
#
# A model too large for one GPU takes several: GPUS_PER_CONFIG=4 gives one config all four,
# for a config with tensor_parallel_size: 4 (the 122B teacher).
#
#SBATCH --job-name=xw-audit
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
read -r -a configs <<< "${CONFIGS:?set CONFIGS to up to four audit configs}"
GPC=${GPUS_PER_CONFIG:-1}
[ $(( ${#configs[@]} * GPC )) -le 4 ] \
  || { echo "FATAL: ${#configs[@]} configs x $GPC GPUs each do not fit on four GPUs"; exit 2; }
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "audits: ${configs[*]} ($GPC GPU per config)"
pids=(); fail=0
for i in "${!configs[@]}"; do
  name=$(basename "${configs[$i]}" .yaml)
  devices=$(seq $(( i * GPC )) $(( i * GPC + GPC - 1 )) | paste -sd, -)
  CUDA_VISIBLE_DEVICES=$devices cost eval audit \
    $PY scripts/run_audit.py --config "${configs[$i]}" --no-similarity \
    > "$XW/logs/xw-audit-${SLURM_JOB_ID:-local}-$name.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for c in "${configs[@]}"; do
  name=$(basename "$c" .yaml); echo "--- $name"
  { grep -E "Traceback|Error|written|unstable|parse" "$XW/logs/xw-audit-${SLURM_JOB_ID:-local}-$name.log" || true; } | tail -n 5
done
[ "$fail" -eq 0 ] || { echo "FATAL: an audit failed or was cut -- see the per-config logs"; exit 1; }
echo "STAGE audit OK"
