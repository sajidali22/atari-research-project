#!/bin/bash
#SBATCH --job-name=atari_fsq_vae
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=capella          # Change to your GPU partition (e.g., capella, alpha)
#SBATCH --nodes=1
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=32G                  # Memory
#SBATCH --cpus-per-task=4          # CPU cores

# ============================================================
# CONFIGURATION - MODIFY THESE AS NEEDED
# ============================================================

# Path to your conda environment or virtual env
# Option 1: Conda
# source ~/.bashrc
# conda activate llm-env

# Option 2: Virtual environment
# source ~/venv/llm/bin/activate
# source /home/sasa880g/DIR/horse/attari/atari-research-project/.venv/bin/activate

export WANDB_API_KEY="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"
# export WANDB_SILENT="true"

# Hugging Face token (if not using huggingface-cli login)
# export HF_TOKEN="hf_your_token_here"

# ============================================================
# SETUP
# ============================================================

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs outputs

# Print GPU info
nvidia-smi

# Print Python environment info
# which python
# python --version

# ============================================================
# RUN TRAINING
# ============================================================

echo ""
echo "Starting training..."
echo ""


uv run train.py

# ============================================================
# COMPLETION
# ============================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
