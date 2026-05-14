import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# Import our project modules
import config
from dataset import AtariDataset
from models.vqvae_simple import AtariVQVAE

def diagnose_with_images(checkpoint_path):
    print(f"🔬 Running Full Diagnostic on: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = AtariVQVAE(num_embeddings=512, embedding_dim=64).to(device)
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 2. Load Data
    # We use a larger batch to get a better 'Universal' view of codebook usage
    # test_dataset = AtariDataset(config.TEST_DIR)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
    # batch = next(iter(test_loader)).to(device)
    
    train_dataset = AtariDataset(config.TRAIN_DIR) # Using training data for the test to ensure we have data to visualize
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    batch = next(iter(train_loader)).to(device)

    # 3. Process Batch
    with torch.no_grad():
        # Get reconstructions for the images
        reconstructions, _ = model(batch)
        
        # Get codebook indices for the diagnostic chart
        encoded = model.encoder(batch)
        n, c, h, w = encoded.shape
        flat_encoded = encoded.permute(0, 2, 3, 1).contiguous().view(-1, 64)
        
        # Manually calculate distances to find winning indices
        distances = (torch.sum(flat_encoded**2, dim=1, keepdim=True) 
                    + torch.sum(model.quantizer.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_encoded, model.quantizer.embeddings.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1)
        
        # Metrics
        unique_indices = torch.unique(encoding_indices)
        used_count = unique_indices.numel()

    # 4. Visualization
    fig = plt.figure(figsize=(15, 10))
    # Layout: Top half for the bar chart, bottom half for images
    ax_bar = plt.subplot2grid((3, 4), (0, 0), colspan=4)
    
    # Plot Codebook Usage
    counts = torch.bincount(encoding_indices, minlength=512).cpu().numpy()
    ax_bar.bar(range(512), counts, color='teal', alpha=0.8)
    ax_bar.set_title(f"Codebook Utilization: {used_count}/512 Codes Used ({(used_count/512)*100:.1f}%)", fontsize=14)
    ax_bar.set_xlabel("Code Index (The 'Visual Vocabulary')")
    ax_bar.set_ylabel("Frequency of Use")

    # Pick the first image in the batch to display
    orig_img = batch[0].cpu().numpy()
    recon_img = reconstructions[0].cpu().numpy()

    for i in range(4):
        # Original Row
        ax_orig = plt.subplot2grid((3, 4), (1, i))
        ax_orig.imshow(orig_img[i], cmap='gray')
        ax_orig.set_title(f"Original Frame {i+1}")
        ax_orig.axis('off')

        # Reconstructed Row
        ax_recon = plt.subplot2grid((3, 4), (2, i))
        ax_recon.imshow(recon_img[i], cmap='gray')
        ax_recon.set_title(f"Reconstructed {i+1}")
        ax_recon.axis('off')

    plt.tight_layout()
    print(f"✅ Diagnostic complete. Active codes: {used_count}")
    plt.show()

if __name__ == "__main__":
    # Point this to your epoch 3 or 4 checkpoint
    checkpoint = "saved_models/atari_vqvae_epoch_3.pth"
    diagnose_with_images(checkpoint)