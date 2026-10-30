#!/bin/bash
# Evaluates one trained arm under docs/EVAL_STANDARD.md, every stage in one job on one GH200
# node, so nothing waits on a dependency chain:
#
#   1. merge the arm's adapter chain, and measure what the merge kept if no measurement is
#      recorded yet (scripts/merge_adapter.py --ensure-retention)
#   2. at once: the benchmark audit (GPU 0) and the arm's op-margin-v2 margin pass (GPU 1);
#      VALIDATE=1 adds the scorer check against HuggingFace (GPU 2, needed once per vLLM
#      version); AGAINST=base adds the base model's margin pass (GPU 3). Before they start,
#      each vLLM GPU waits up to GPU_WAIT seconds (600) for its memory share to be free
#   3. the matched comparison: compare-arms against the reference arm, or compare for base
#   4. the arm's reference row, with the standard's verdict printed
#
#   AUDIT=configs/audit/<arm>_gh200.yaml \
#   AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
#     scripts/cscs/submit.sh scripts/cscs/evaluate_arm.sbatch
#
# AGAINST=base compares an adapter with its own base (SFT against the base model). For any other
# AGAINST, that arm's own margin pass must exist already. Audits and margin passes resume from
# their cached chunks, so a job cut by the time limit continues when it is submitted again.
# The comparison only conforms when the code is committed; the verdict says so otherwise.
# REPO=<worktree> at submission runs the code of another checkout.
#
#SBATCH --job-name=xw-eval-arm
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
AUDIT=${AUDIT:?set AUDIT to the audit config of the arm}
AGAINST=${AGAINST:?set AGAINST to the audit config of the reference arm, or to base}
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
[ -z "$(git status --porcelain -- src scripts 2>/dev/null)" ] \
  || echo "WARNING: uncommitted changes under src/ or scripts/; the comparison will not conform"
L="$XW/logs/xw-eval-arm-${SLURM_JOB_ID:-local}"

step "1. merge $AUDIT and measure the merge"
cost eval merge \
  $PY scripts/merge_adapter.py --config "$AUDIT" --ensure-retention > "$L-merge.log" 2>&1 \
  || { tail -n 20 "$L-merge.log"; echo "FATAL: merge failed"; exit 1; }
grep -E "merge kept|wrote|already merged" "$L-merge.log" | tail -n 4 || true

# vLLM refuses to start on a GPU whose free memory is under the share it is set to take. On
# 2026-10-22 GPUs 0, 2 and 3 of nid005821 held 16 to 19 GiB that no process owned when stage 2
# began, and the audit (0.8 of 95 GiB) failed at once (job 5119285); within minutes the memory
# was free again. So before stage 2, wait up to GPU_WAIT seconds (default 600) until each vLLM
# GPU has its share free. The test is vLLM's own: free >= share x total, by mem_get_info.
wait_for_gpus() {  # <gpu>:<config key>:<the key's default in the code> ...
  $PY - "$AUDIT" "${GPU_WAIT:-600}" "$@" <<'EOF'
import sys, time
import torch, yaml
config, wait, *checks = sys.argv[1:]
cfg = yaml.safe_load(open(config)) or {}
start = time.monotonic()
for check in checks:
    gpu, key, default = check.split(":")
    share = float(cfg.get(key, default))
    while True:
        free, total = (x / 2**30 for x in torch.cuda.mem_get_info(int(gpu)))
        if free >= share * total:
            break
        note = f"GPU {gpu}: {free:.2f} of {total:.2f} GiB free, under {key} {share} ({share * total:.2f} GiB)"
        if time.monotonic() - start >= float(wait):
            sys.exit(f"FATAL: {note} after {time.monotonic() - start:.0f} s")
        print(f"{note}; waiting", flush=True)
        time.sleep(30)
EOF
}

step "2. audit (GPU 0) and margin pass (GPU 1)"
gpus=(0:gpu_memory_utilization:0.90 1:op_control_gpu_memory_utilization:0.55)
[ "$AGAINST" = base ] && gpus+=(3:op_control_gpu_memory_utilization:0.55)
wait_for_gpus "${gpus[@]}" || exit 1
pids=()
CUDA_VISIBLE_DEVICES=0 cost eval audit $PY scripts/run_audit.py --config "$AUDIT" --no-similarity \
  > "$L-audit.log" 2>&1 & pids+=($!)
CUDA_VISIBLE_DEVICES=1 cost eval margins \
  $PY scripts/operating_point_control.py margins --config "$AUDIT" \
  --which adapter > "$L-margins.log" 2>&1 & pids+=($!)
if [ "${VALIDATE:-0}" = 1 ]; then
  CUDA_VISIBLE_DEVICES=2 cost eval validate \
    $PY scripts/operating_point_control.py validate --config "$AUDIT" \
    --which adapter > "$L-validate.log" 2>&1 & pids+=($!)
fi
if [ "$AGAINST" = base ]; then
  CUDA_VISIBLE_DEVICES=3 cost eval margins-base \
    $PY scripts/operating_point_control.py margins --config "$AUDIT" \
    --which base > "$L-base.log" 2>&1 & pids+=($!)
fi
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for part in audit margins validate base; do
  [ -f "$L-$part.log" ] || continue
  echo "--- $part"
  { grep -E "Traceback|Error|wrote|passed|failed|skipping" "$L-$part.log" || true; } | tail -n 4
done
[ "$fail" -eq 0 ] || { echo "FATAL: a stage-2 process failed -- see $L-*.log"; exit 1; }

step "3. matched comparison"
if [ "$AGAINST" = base ]; then
  cost eval compare $PY scripts/operating_point_control.py compare --config "$AUDIT"
else
  cost eval compare $PY scripts/operating_point_control.py compare-arms --config "$AUDIT" \
    --against "$AGAINST"
fi

step "4. reference row and verdict"
$PY scripts/reference_metrics.py --config "$AUDIT" --show
echo "STAGE evaluate-arm OK"
