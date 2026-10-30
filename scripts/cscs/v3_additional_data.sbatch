#!/bin/bash
# v3 additional data (scripts/v3_additional_data.py, pre-registered in docs/V3_NEW_DATA.md): the 122B teacher on
# all four GPUs (tensor parallel 4) runs the vote, invariant and biased passes for the parts it is
# given, loading the model once (about 5 minutes). Every pass of every part is its own file, so a
# job cut by its time limit is resubmitted with the same parts and skips what is done. Submit one
# job at a time on the shared account, with REPO pointing at the checkout whose code should run:
#
#   REPO=$XW/hiring_bias_mitigation_at-v3additional scripts/cscs/submit.sh -t 03:00:00 \
#       scripts/cscs/v3_additional_data.sbatch djinni_new-000 rjdb-000 kaggle_linkedin-000
#   REPO=... scripts/cscs/submit.sh -t 12:00:00 scripts/cscs/v3_additional_data.sbatch rjdb-001 rjdb-002 ...
#
# A later round names its own config, which reads the first round's pool into its own directory
# (round 3, configs/generation/v3_new_pairs.yaml):
#   CFG=configs/generation/v3_new_pairs.yaml PASSES="votes invariant" REPO=... \
#       scripts/cscs/submit.sh -t 02:00:00 scripts/cscs/v3_additional_data.sbatch djinni_new-003 rjdb-002
#
# Per part of 1,500 Djinni or RJDB pairs: about 9 min of votes and 13 min per variant pass at the
# rates of jobs 5073453 and 5103710; Kaggle x LinkedIn prompts are about 2.5x longer. The first part
# of each source is read against the stop conditions (configs stop:), and a source that crosses one
# gets no further part. After the parts, the CPU `report` stage reads every part drawn so far.
#
#SBATCH --job-name=xw-v3-additional
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
[ $# -gt 0 ] || { echo "usage: v3_additional_data.sbatch <part> [<part> ...] | all"; exit 2; }
export HF_HUB_OFFLINE=1
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | repo $REPO"
git status --porcelain | sed 's/^/uncommitted: /'
CFG=${CFG:-configs/generation/v3_additional_data.yaml}
echo "config $CFG"

# PASSES="votes invariant" skips the biased pass, whose teacher-written negatives r2em does not
# use; DPO's rejected side then comes from on-policy sampling (docs/V3_NEW_DATA.md).
PASSES=${PASSES:-votes invariant biased}
step "generate: passes $PASSES for $*"
cost datagen teacher $PY scripts/v3_additional_data.py --config "$CFG" --stage generate \
  --parts "$@" --passes $PASSES
step "report: every part drawn so far"
$PY scripts/v3_additional_data.py --config "$CFG" --stage report
echo "STAGE v3-additional $* OK"
