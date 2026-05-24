import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
from tqdm import tqdm

import config
from dataset import AtariDataset

# 🚨 Import our fresh, clean models!
from models.standard_vae import StandardVAE, standard_vae_loss
from models.ema_vqvae import EmaVqVae
from models.residual_vqvae import AtariResidualVQVAE

def train():
    # Will safely fall back to CPU as you requested
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
            "batch_size": config.BATCH_SIZE,
            "num_embeddings": config.NUM_EMBEDDINGS if config.MODEL_TYPE != 'standard_vae' else None,
            "embedding_dim": config.EMBEDDING_DIM if config.MODEL_TYPE != 'standard_vae' else None,
            "commitment_cost": config.COMMITMENT_COST if config.MODEL_TYPE != 'standard_vae' else None,
            "decay": config.DECAY if config.MODEL_TYPE != 'standard_vae' else None,
            
        }
    )

    # 2. Dynamic Model Loading
    print(f"🧠 Initializing Model Type: {config.MODEL_TYPE.upper()}")
    if config.MODEL_TYPE == 'standard_vae':
        model = StandardVAE(latent_dim=config.LATENT_DIM).to(device)
        
    elif config.MODEL_TYPE == 'ema_vqvae':
        model = EmaVqVae(num_embeddings=config.NUM_EMBEDDINGS, embedding_dim=config.EMBEDDING_DIM).to(device)
    
    elif config.MODEL_TYPE == 'residual_vqvae':
        model = AtariResidualVQVAE(
            num_embeddings=config.NUM_EMBEDDINGS, 
            embedding_dim=config.EMBEDDING_DIM,
            commitment_cost=config.COMMITMENT_COST,
            decay=config.DECAY
        ).to(device)
        
    else:
        raise ValueError(f"❌ Invalid MODEL_TYPE '{config.MODEL_TYPE}' in config.py!")

    # 3. Dataset and Optimizer
    train_dataset = AtariDataset(config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=2 # Keeps CPU feeding data quickly
    )
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 4. Training Loop
    print(f"🚀 Starting Training Loop for {config.EPOCHS} Epochs...")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_epoch_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{config.EPOCHS}]")
        
        for batch_idx, batch in enumerate(loop):
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # --- Model Specific Loss Logic ---
            if config.MODEL_TYPE == 'standard_vae':
                reconstructed, mu, logvar = model(batch)
                loss, recon_loss, extra_loss = standard_vae_loss(
                    reconstructed, batch, mu, logvar, beta=config.BETA
                )
                loss_name = "KL Divergence"
                
            elif config.MODEL_TYPE in ['ema_vqvae', 'residual_vqvae']:
                reconstructed, vq_loss = model(batch)
                recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)
                loss = recon_loss + vq_loss
                extra_loss = vq_loss
                loss_name = "VQ Codebook Loss"
            # ---------------------------------
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Tracking and live terminal logging
            total_epoch_loss += loss.item()
            loop.set_postfix({"Total Loss": f"{loss.item():.4f}"})
            
            # Weights & Biases Logging (Every 50 batches)
            if batch_idx % 50 == 0:
                wandb.log({
                    "Batch": batch_idx + (epoch - 1) * len(train_loader),
                    "Epoch": epoch,
                    "Total Loss": loss.item(),
                    "Reconstruction Loss": recon_loss.item() if config.MODEL_TYPE != 'standard_vae' else recon_loss.item() / config.BATCH_SIZE,
                    loss_name: extra_loss.item()
                })
        
        # Log average loss at the end of the epoch
        wandb.log({"Average Epoch Loss": total_epoch_loss / len(train_loader)})
        
        # 5. Save Checkpoint
        save_path = os.path.join(config.SAVE_DIR, f"{config.MODEL_TYPE}_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
    
    wandb.finish()
    print("✅ Training fully completed!")

if __name__ == "__main__":
    train()