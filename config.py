import os

# ==========================================
# 1. Directory Settings
# ==========================================
TRAIN_DIR = "expert_dataset/train"
TEST_DIR = "expert_dataset/test"
SAVE_DIR = "saved_models" 

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Model Architecture Selection
# ==========================================
# Options: 'standard_vae', 'ema_vqvae', 'residual_vqvae
MODEL_TYPE = 'residual_vqvae' 

# ==========================================
# 3. Architecture Specific Hyperparameters
# ==========================================
# For EMA VQ-VAE
NUM_EMBEDDINGS = 512  
EMBEDDING_DIM = 64    
COMMITMENT_COST: 0.25 
DECAY: 0.99 

# For Standard VAE only
LATENT_DIM = 256
BETA = 1.0 # Weight of the KL-Divergence penalty

# ==========================================
# 4. Training Parameters
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
EPOCHS = 15