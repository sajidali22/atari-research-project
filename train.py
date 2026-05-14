import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
from tqdm import tqdm

import config
from dataset import AtariDataset

# ---------------------------------------------------------
# IMPORT ASSUMPTIONS:
# Make sure your files are named like this so imports work, 
# or change the file names below to match your setup!
# ---------------------------------------------------------
from models.vae import AtariVAE, vae_loss_function
from models.vqvae_simple import AtariVQVAE as SimpleVQVAE
from models.vqvae_residual import AtariVQVAE as ResidualVQVAE
from models.vit_vqae import AdvancedViTVQVAE

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Starting Training on device: {device}")
    
    # 1. Initialize Weights & Biases
    wandb.init(
        project="atari-universal-feature-extractor",
        name=f"run-{config.MODEL_TYPE}",
        config={
            "model_type": config.MODEL_TYPE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE
        }
    )

    # 2. Dynamic Model Loading
    print(f"🧠 Initializing Model Type: {config.MODEL_TYPE.upper()}")
    if config.MODEL_TYPE == 'vae':
        model = AtariVAE(latent_dim=config.LATENT_DIM).to(device)
        
    elif config.MODEL_TYPE == 'vqvae_simple':
        model = SimpleVQVAE(num_embeddings=config.NUM_EMBEDDINGS, embedding_dim=config.EMBEDDING_DIM).to(device)
        
    elif config.MODEL_TYPE == 'vqvae_residual':
        model = ResidualVQVAE(num_embeddings=config.NUM_EMBEDDINGS, embedding_dim=config.EMBEDDING_DIM).to(device)
        
    elif config.MODEL_TYPE == 'vit_vqvae':
        model = AdvancedViTVQVAE(num_embeddings=config.NUM_EMBEDDINGS, embed_dim=256).to(device)
        
    else:
        raise ValueError("Invalid MODEL_TYPE in config.py!")

    # 3. Dataset and Optimizer
    train_dataset = AtariDataset(config.TRAIN_DIR)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4. The Smart Training Loop
    print(f"🔥 Starting Training Loop for {config.EPOCHS} Epochs...")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_epoch_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{config.EPOCHS}]")
        
        for batch_idx, batch in enumerate(loop):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # --- CONDITIONAL MATH BASED ON MODEL TYPE ---
            if config.MODEL_TYPE == 'vae':
                # VAEs return Means and Log Variances
                reconstructed, mu, logvar = model(batch)
                loss, recon_loss, extra_loss = vae_loss_function(
                    reconstructed, batch, mu, logvar, beta=config.BETA
                )
                loss_name = "KL Divergence"
                
            else:
                # All VQ-VAEs return the Dictionary Commitment Loss
                reconstructed, vq_loss = model(batch)
                recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)
                loss = recon_loss + vq_loss
                extra_loss = vq_loss
                loss_name = "VQ Codebook Loss"
            # --------------------------------------------
            
            loss.backward()
            optimizer.step()
            
            # Tracking and logging
            total_epoch_loss += loss.item()
            loop.set_postfix({"Total Loss": f"{loss.item():.4f}"})
            
            if batch_idx % 50 == 0:
                wandb.log({
                    "Batch": batch_idx + (epoch - 1) * len(train_loader),
                    "Epoch": epoch,
                    "Total Loss": loss.item(),
                    "Reconstruction Loss": recon_loss.item() if config.MODEL_TYPE != 'vae' else recon_loss.item() / config.BATCH_SIZE,
                    loss_name: extra_loss.item()
                })
        
        wandb.log({"Average Epoch Loss": total_epoch_loss / len(train_loader)})
        
        # 5. Save Model
        save_path = os.path.join(config.SAVE_DIR, f"{config.MODEL_TYPE}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
    
    wandb.finish()
    print("Training fully completed!")

if __name__ == "__main__":
    train()