"""Experiment selector + hyperparameters for the autoencoder baselines.

Edit MODEL_TYPE and the matching block below, then run from the repo root:

    python -m autoencoders.train

Defaults reproduce the checkpoints used in the report (FSQ-VAE, levels [8,5,5,3],
125 epochs). The original config.py was lost from the working tree; this one is
rebuilt from the copy on the Hpc branch, with the commented-out values restored.
"""

import os

from paths import AE_CHECKPOINTS, EXPERT_DATA

# ==========================================
# 1. Directories
# ==========================================
# Expert frame stacks, (N, 84, 84, 4) uint8 under the 'frames' key.
# Produced by data_collection/collect_expert_data.py.
TRAIN_DIR = os.path.join(EXPERT_DATA, "train")
TEST_DIR = os.path.join(EXPERT_DATA, "test")

SAVE_DIR = os.path.join(AE_CHECKPOINTS, "new_run")

# ==========================================
# 2. Model selection
# ==========================================
# Options: 'standard_vae', 'ema_vqvae', 'residual_vqvae', 'fsq_vae'
MODEL_TYPE = "fsq_vae"

# ==========================================
# 3. Architecture-specific hyperparameters
# ==========================================
# FSQ-VAE
FSQ_LEVELS = [8, 5, 5, 3]

# EMA VQ-VAE / Residual VQ-VAE
NUM_EMBEDDINGS = 512
EMBEDDING_DIM = 64
COMMITMENT_COST = 0.25
DECAY = 0.99

# Standard VAE
LATENT_DIM = 256
BETA = 1.0  # weight of the KL term

# ==========================================
# 4. Training
# ==========================================
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EPOCHS = 125  # train.py holds the LR constant for 100 epochs, then decays
