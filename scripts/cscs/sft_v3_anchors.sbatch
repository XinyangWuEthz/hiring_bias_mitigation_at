#!/bin/bash
# SFT-v3 stages 0 and 1 (scripts/sft_v3_anchors.py, pre-registered in docs/SFT_V3_ANCHORS.md):
# the 122B teacher on all four GPUs (tensor parallel 4) writes new training anchors for the pool,
# then the CPU step that reads them. The teacher takes about 5 minutes to load.
#
#   scripts/cscs/submit.sh -p debug -t 00:20:00 scripts/cscs/sft_v3_anchors.sbatch select
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/sft_v3_anchors.sbatch generate
#   scripts/cscs/submit.sh -p debug -t 00:15:00 scripts/cscs/sft_v3_anchors.sbatch control
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/sft_v3_anchors.sbatch votes
#   scripts/cscs/submit.sh -p debug -t 00:20:00 scripts/cscs/sft_v3_anchors.sbatch vote_targets
#
# `select` copies the blind review's labels, draws both candidate prompts on its 391 pool pairs
# and scores them (stage 0a). `generate` draws the chosen prompt on all 2,992 pairs, regenerates
# the changed pairs' invariant targets and runs the gate (stages 0b and 1). A `generate` that
# runs out of time resumes from its saved draws. `control` draws the unchanged v1 prompt on the
# same 391 pairs and compares it with both candidates (the page's addendum). `votes` draws the
# unchanged v1 prompt at seed 43 on all 2,992 pool pairs, in parts, and reads the earlier
# training sets against it (the second addendum); a run that times out resumes from its parts.
# `vote_targets` takes the vote's majority as the anchor (the third addendum), regenerates v1's
# invariant targets for the pairs whose verdict it changes, one seed per prompt, and reads them
# against v1's targets for the same variants.
#
#SBATCH --job-name=xw-sft-v3-anchors
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
REVIEW=$XW/hiring_bias_mitigation_at/reports/claude_anchor_review/graded_cases.csv

case "${1:-}" in
  select)
    step "labels: the blind review's decision and fit for its 391 pool pairs"
    $PY scripts/sft_v3_anchors.py --stage labels --review "$REVIEW"
    step "stage 0a: both candidate prompts, 8 draws each, on the 391 reviewed pairs"
    $PY scripts/sft_v3_anchors.py --stage select
    step "score: C1, C2 and the chosen prompt"
    $PY scripts/sft_v3_anchors.py --stage score ;;
  generate)
    step "stage 0b: the chosen prompt on all pool pairs, then the changed pairs' targets"
    $PY scripts/sft_v3_anchors.py --stage generate
    step "stage 1: the gate"
    $PY scripts/sft_v3_anchors.py --stage gate ;;
  control)
    step "redraw control: the unchanged v1 prompt, 8 draws, on the 391 reviewed pairs"
    $PY scripts/sft_v3_anchors.py --stage control
    step "compare: the control against both candidates"
    $PY scripts/sft_v3_anchors.py --stage compare ;;
  votes)
    step "pool-wide vote: the unchanged v1 prompt, 8 draws at seed 43, on all pool pairs"
    $PY scripts/sft_v3_anchors.py --stage votes
    step "enrichment: platform check, unstable anchors in earlier training sets, stability"
    $PY scripts/sft_v3_anchors.py --stage enrichment ;;
  vote_targets)
    step "vote anchors: v1's invariant procedure on the pairs whose verdict the vote changes"
    $PY scripts/sft_v3_anchors.py --stage vote_targets
    step "readout: the regenerated targets against v1's, and the flips in earlier training sets"
    $PY scripts/sft_v3_anchors.py --stage vote_readout ;;
  *) echo "usage: sft_v3_anchors.sbatch select|generate|control|votes|vote_targets"; exit 2 ;;
esac
echo "STAGE sft-v3-anchors $1 OK"
