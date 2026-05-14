import os

# ==========================================
# 1. Directory Settings
# ==========================================
TRAIN_DIR = "atari_train_dataset"
TEST_DIR = "atari_test_dataset"
SAVE_DIR = "saved_models"

# Ensure the save folder exists so we don't crash at the end of an epoch
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Model Architecture Selection
# ==========================================
# Options: 'simple', 'residual', or 'vit'
MODEL_TYPE = 'vit'

# ==========================================
# 3. VQ-VAE Hyperparameters
# ==========================================
NUM_EMBEDDINGS = 512  # Size of the dictionary
EMBEDDING_DIM = 64    # Size of each puzzle piece

# ==========================================
# 4. Training Hyperparameters
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10