import os

# ==========================================
# 1. Directory Settings
# ==========================================
TRAIN_DIR = "atari-DQN/custom_datasets/train"
TEST_DIR = "atari-DQN/custom_datasets/test"
VAL_DIR = "atari-DQN/custom_datasets/val"

SAVE_DIR = "saved_models" 

os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. Model Architecture Selection
# ==========================================
# Options: 'standard_vae', 'ema_vqvae', 'residual_vqvae', 'fsq_vae
MODEL_TYPE = 'fsq_vae' 

# ==========================================
# 3. Architecture Specific Hyperparameters
# ==========================================
# For FSQ-VAE only
FSQ_LEVELS = [8, 5, 5, 3]


# For EMA VQ-VAE
# NUM_EMBEDDINGS = 512  
# EMBEDDING_DIM = 64
# COMMITMENT_COST = 0.25
# DECAY = 0.99

# For Standard VAE only
# LATENT_DIM = 256
# BETA = 1.0 # Weight of the KL-Divergence penalty

# ==========================================
# 4. Training Parameters
# ==========================================
# BATCH_SIZE = 128
# LEARNING_RATE = 3e-4
# EPOCHS = 125



# ##For JEPA
# batch_size = 256
# learning_rate = 1e-4
# weight_decay = 1e-5
# tau_ema = 0.996
# latent_dim = 256
# max_grad_norm = 1.0
# epochs = 15
