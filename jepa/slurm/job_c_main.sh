#!/bin/bash
# Job C -- the main run.
#
# Safe to requeue: --resume is a no-op when no checkpoint exists, and picks up epoch,
# global step, optimizer, LR schedule and the W&B run id when one does. If SLURM kills
# this at the wall clock, resubmit the identical script.
#
#   sbatch slurm/job_c_main.sh
#   sbatch --time=12:00:00 slurm/job_c_main.sh     # shorter allocation
#
#SBATCH --job-name=jepa_v2_main
#SBATCH --partition=capella
#SBATCH --account=p_atari26
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} | start $(date)"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source "$VENV_PATH/bin/activate"
nvidia-smi

# Derive the training budget from the allocation SLURM actually granted, rather than
# hardcoding it. Overriding the wall clock (`sbatch --time=12:00:00 ...`) would
# otherwise leave a hardcoded --max-hours overrunning it, and the job would be killed
# mid-epoch instead of stopping cleanly.
#
# RESERVE covers the final checkpoint write and the W&B flush.
RESERVE_H=0.4
BUDGET_H=15.6
if END=$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null | sed -n 's/.*EndTime=\([^ ]*\).*/\1/p') \
   && [ -n "$END" ]; then
    SECS=$(( $(date -d "$END" +%s) - $(date +%s) ))
    BUDGET_H=$(python3 -c "print(max(0.25, $SECS/3600.0 - $RESERVE_H))")
    echo "SLURM allocation ends $END -> training budget ${BUDGET_H} h"
else
    echo "Could not read the SLURM end time; falling back to ${BUDGET_H} h"
fi
# target < max: sizing aims slightly under the hard stop so the run decays its LR
# fully rather than being cut off mid-cosine.
TARGET_H=$(python3 -c "print(round($BUDGET_H * 0.98, 2))")

# --target-hours sizes the run from a measured throughput probe; --max-hours is the
# hard stop, so the final checkpoint and W&B flush complete rather than being killed
# mid-write.
python jepa_train.py \
    --inverse-weight auto \
    --run-name "v2-main" \
    --target-hours "$TARGET_H" \
    --max-hours "$BUDGET_H" \
    --batch-size 512 \
    --num-workers 12 \
    --amp-dtype bf16 \
    --eval-heldout \
    --max-val-batches 40 \
    --fig-every 10 \
    --seed 0 \
    --resume

echo "Done $(date)"
