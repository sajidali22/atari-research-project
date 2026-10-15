#!/bin/bash
# Job A -- verify the data and build the .npy_cache directories. CPU ONLY.
#
# Deliberately requests no GPU: this is ~161 GB of decompression and IO, and running
# it inside a GPU job would spend an H100 allocation on zlib.
#
#   sbatch slurm/job_a_cache.sh
#
#SBATCH --job-name=jepa_v2_cache
#SBATCH --partition=capella
#SBATCH --account=p_atari26
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$(hostname)} | start $(date)"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

source "$VENV_PATH/bin/activate"
python -c "import torch, numpy; print('torch', torch.__version__, '| numpy', numpy.__version__)"

# Build the action registry first: it needs ale-py, and a failure here is much
# cheaper to discover now than inside the 16 h training job.
python games.py

# --check refuses to write anything if disk is short or a shard is malformed.
python preflight.py --check --build --splits train val test

echo "Done $(date)"
