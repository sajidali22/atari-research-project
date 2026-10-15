"""Paths and defaults for the environment-conditioned JEPA (v2).

v1's config.py carried a block of VAE-project settings (MODEL_TYPE, FSQ_LEVELS,
NUM_EMBEDDINGS, COMMITMENT_COST, BETA, ...) that no JEPA file ever read. They are
dropped here; the autoencoder settings now live in ../autoencoders/config.py.
"""

import os

# All paths are anchored to this file's directory, so they resolve from any working
# directory. Data and checkpoints live in the git-ignored ../artifacts/ tree (same
# layout as ../paths.py). The transition datasets are produced by
# ../data_collection/collect_jepa_transitions.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ARTIFACTS = os.path.join(_HERE, "..", "artifacts")
DATA_ROOT = os.path.normpath(os.path.join(_ARTIFACTS, "data", "jepa"))

# The full 2.75M-transition set: 11 games x 5 shards x 50k. Stored CHW.
TRAIN_DIR = os.path.join(DATA_ROOT, "custom_datasets/train")

# The smaller splits, stored HWC. TRAIN_DIR_SMALL is the 550k 11-game split;
# VAL_DIR is the 110k split over those same 11 games.
TRAIN_DIR_SMALL = os.path.join(DATA_ROOT, "custom_datasets_small/train")
VAL_DIR = os.path.join(DATA_ROOT, "custom_datasets_small/val")

# The 4 genuinely held-out games (Alien, Asteroids, Atlantis, IceHockey), 50k each.
# v1's TEST_DIR pointed at custom_datasets/test, which does not exist, so these
# games were unreachable from config and went unused.
TEST_DIR = os.path.join(DATA_ROOT, "custom_datasets_small/test")

CHECKPOINT_DIR = os.path.normpath(os.path.join(_ARTIFACTS, "checkpoints", "jepa"))

# ==========================================
# Environment conditioning
# ==========================================
# Probability of replacing a sample's true game id with UNKNOWN_ID before the
# env-embedding lookup. Trains the UNKNOWN slot as a generic-Atari prior and stops
# the encoder from fragmenting into 11 disjoint per-game subspaces.
ENV_DROPOUT_P = 0.25

# SimNorm (TD-MPC2 Eq. 5): partition each 256-d token into 256/V simplices.
SIMNORM_V = 8

# Weight on the inverse-dynamics loss. SimNorm shrinks the JEPA term by roughly an
# order of magnitude (simplex entries average 1/V = 0.125) while the cross-entropy
# term is unchanged, so v1's 0.1 is far too large: measured at init, JEPA loss is
# ~0.026 against an inverse loss of ~2.2 nats. Sweep {0.003, 0.01, 0.03, 0.1}.
INVERSE_LOSS_WEIGHT = 0.01
