import os

# ==========================================
# 1. Directory Settings
# ==========================================
TRAIN_DIR = "atari_train_dataset"
TEST_DIR = "atari_test_dataset"
SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Model Architecture Selection
# ==========================================
# CHOOSE ONE: 'vae', 'vqvae_simple', 'vqvae_residual', 'vit_vqvae'
MODEL_TYPE = 'vqvae_residual' 

# ==========================================
# 3. Architecture Specific Hyperparameters
# ==========================================
# For VQ-VAEs (Simple, Residual, ViT)
NUM_EMBEDDINGS = 512  
EMBEDDING_DIM = 64    

# For Standard VAE only
LATENT_DIM = 256
BETA = 1.0 # Weight of the KL-Divergence penalty

# ==========================================
# 4. Universal Training Hyperparameters
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 10