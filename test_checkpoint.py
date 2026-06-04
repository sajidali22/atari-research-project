import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# Import your trained architecture
from models.residual_vqvae import AtariResidualVQVAE

def test_model_locally():
    # ---------------------------------------------------------
    # 1. SETUP PATHS (Update these to match your PC's folders!)
    # ---------------------------------------------------------
    MODEL_PATH = "saved_models/fsq_vae_epoch_125.pth"
    
    # Point this directly to ONE of your compressed .npz test files
    TEST_DATA_PATH = "expert_dataset/test/IceHockeyNoFrameskip-v4_expert_50000_frames.npz" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Running inference on: {device}")

    # ---------------------------------------------------------
    # 2. LOAD THE MODEL
    # ---------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
        
    print("🧠 Loading Residual VQ-VAE...")
    model = AtariResidualVQVAE(num_embeddings=512, embedding_dim=64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Freeze the model for testing
    print("✅ Model loaded!")

    # ---------------------------------------------------------
    # 3. LOAD & PREPARE DATA (.npz Extraction)
    # ---------------------------------------------------------
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"❌ Test data not found at {TEST_DATA_PATH}")
        
    print("📦 Unpacking .npz file and loading sample frames...")
    
    # Extract the 'frames' array directly from the .npz archive
    raw_data = np.load(TEST_DATA_PATH)['frames'] 
    
    # Pick 6 random sequences to look at
    random_indices = random.sample(range(len(raw_data)), 6)
    sample_data = raw_data[random_indices]
    
    # Transpose from (Batch, H, W, Channels) to (Batch, Channels, H, W)
    sample_data = np.transpose(sample_data, (0, 3, 1, 2))
    
    # Convert to Tensor and normalize to [0.0, 1.0]
    batch = torch.tensor(sample_data, dtype=torch.float32).to(device) / 255.0

    # ---------------------------------------------------------
    # 4. RUN INFERENCE
    # ---------------------------------------------------------
    print("✨ Generating reconstructions...")
    with torch.no_grad():
        reconstructed, _ = model(batch)

    # ---------------------------------------------------------
    # 5. VISUALIZE RESULTS
    # ---------------------------------------------------------
    # Move back to CPU and Numpy for Matplotlib
    original_images = batch.cpu().numpy()
    recon_images = reconstructed.cpu().numpy()
    
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    fig.suptitle("Top: Original Atari Frames | Bottom: Residual VQ-VAE Reconstructions", fontsize=16)
    
    for i in range(6):
        # We grab index [3] to look at the most recent frame in the 4-frame stack
        orig_frame = original_images[i][3] 
        recon_frame = recon_images[i][3]
        
        # Plot Original
        axes[0, i].imshow(orig_frame, cmap='gray')
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis('off')
        
        # Plot Reconstruction
        axes[1, i].imshow(recon_frame, cmap='gray')
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    plt.show() # Pops open the UI window on your monitor!

if __name__ == "__main__":
    test_model_locally()