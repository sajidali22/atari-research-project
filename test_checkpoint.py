import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# Import our project modules
import config
from dataset import AtariDataset
from vqvae import AtariVQVAE

def test_checkpoint(checkpoint_name="atari_vqvae_epoch_3.pth"):
    """
    Tests the visual fidelity and MSE of a specific model checkpoint.
    """
    print(f"🧪 Testing Checkpoint: {checkpoint_name}")
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Initialize Model and Load Weights
    # We use the same parameters as used in the training script
    model = AtariVQVAE(in_channels=4, num_embeddings=512, embedding_dim=64).to(device)
    
    checkpoint_path = os.path.join("saved_models", checkpoint_name)
    if not os.path.exists(checkpoint_path):
        print(f"❌ File not found: {checkpoint_path}. Check your 'saved_models' folder.")
        return

    # Load the state dictionary
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    # CRITICAL: Set to evaluation mode
    model.eval()
    print("✅ Model weights loaded and set to EVAL mode.")

    # 3. Prepare Test Data
    # We pull from the TEST_DIR to see how well it generalizes to unseen games
    # test_dataset = AtariDataset(config.TEST_DIR)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    while True:
        train_dataset = AtariDataset(config.TRAIN_DIR) # Using training data for the test to ensure we have data to visualize
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        # 4. Run Inference
        # We use torch.no_grad() to save memory and skip gradient math
        with torch.no_grad():
            # Grab a random sample from an unseen game
            original_frames = next(iter(train_loader)).to(device)
            
            # Pass through the VQ-VAE
            reconstructed_frames, vq_loss = model(original_frames)
            
            # Calculate Reconstruction Error (MSE)
            # Formula: MSE = (1/n) * sum((orig - recon)^2)
            mse_error = torch.mean((original_frames - reconstructed_frames) ** 2)

        print(f"📊 Quantitative Results:")
        print(f"   - Reconstruction MSE: {mse_error.item():.6f}")
        print(f"   - VQ Dictionary Loss: {vq_loss.item():.6f}")

        # 5. Visualization Logic
        # Convert tensors back to numpy arrays for plotting
        # Shape: (1, 4, 84, 84) -> (4, 84, 84)
        orig_np = original_frames.squeeze(0).cpu().numpy()
        recon_np = reconstructed_frames.squeeze(0).cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        plt.subplots_adjust(top=0.85)
        fig.suptitle(f"Checkpoint Test: {checkpoint_name}\n(Top: Original | Bottom: VQ-VAE Reconstruction)", fontsize=16)

        for i in range(4):
            # Plot Original
            axes[0, i].imshow(orig_np[i], cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title(f"Original F{i+1}")

            # Plot Reconstructed
            axes[1, i].imshow(recon_np[i], cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title(f"Recon F{i+1}")

        print("\n🖼️ Displaying reconstruction plot...")
        plt.show()

if __name__ == "__main__":
    # Ensure this matches the filename in your 'saved_models' folder
    test_checkpoint("atari_vqvae_epoch_4.pth")