#!/bin/bash
# Job B -- lambda sweep, 4 arms in parallel as a SLURM array.
#
# SimNorm shrank the JEPA term by ~an order of magnitude while the inverse
# cross-entropy is unchanged (measured: 0.026 vs 2.2 nats at init), so v1's 0.1 is far
# off and the right value is genuinely unknown. Each arm writes its result into
# sweep_result.json; job_c reads the winner via --inverse-weight auto.
#
#   sbatch slurm/job_b_sweep.sh
#
#SBATCH --job-name=jepa_v2_sweep
#SBATCH --partition=capella
#SBATCH --account=p_atari26
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

LAMBDAS=(0.003 0.01 0.03 0.1)
LAMBDA=${LAMBDAS[$SLURM_ARRAY_TASK_ID]}

echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME} | lambda=$LAMBDA | $(date)"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source "$VENV_PATH/bin/activate"
nvidia-smi

# --sweep-tag makes this run append to sweep_result.json on completion. Arms whose
# env embeddings collapsed (mean pairwise cosine > 0.9) are disqualified there,
# because a low loss with inert conditioning is not a win.
#
# --max-hours, not an epoch count: the arms only need to run long enough to rank
# lambda, and they must all finish before job_c is submitted.
python jepa_train.py \
    --inverse-weight "$LAMBDA" \
    --sweep-tag "lam$LAMBDA" \
    --run-name "v2-sweep-lam$LAMBDA" \
    --target-hours 1.4 \
    --max-hours 1.5 \
    --batch-size 512 \
    --num-workers 12 \
    --amp-dtype bf16 \
    --eval-heldout \
    --max-val-batches 40 \
    --seed 0

echo "Done $(date)"
