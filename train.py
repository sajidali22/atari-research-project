import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import tqdm 
import wandb # 1. Import the logging library

# Import our configurations and data pipeline
import config
from dataset import AtariDataset
from vqvae import AtariVQVAE

def train_vqvae():
    print("🚀 Initializing Universal VQ-VAE Feature Extractor Training...")

    # 2. Initialize Weights & Biases
    # This creates a new project and run on your dashboard
    wandb.init(
        project="atari-universal-feature-extractor", 
        name="vqvae-baseline-test", 
        config={
            "learning_rate": 1e-4,
            "epochs": 10, # Leave at 1 for the CPU test
            "batch_size": 64,
            "embedding_dim": 64,
            "num_embeddings": 512
        }
    )

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Load Dataset
    print("📂 Loading dataset...")
    train_dataset = AtariDataset(config.TRAIN_DIR)
    
    # Notice we now use wandb.config to access our parameters!
    train_loader = DataLoader(
        train_dataset, 
        batch_size=wandb.config.batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Initialize the VQ-VAE Model using tracked config parameters
    model = AtariVQVAE(
        in_channels=4, 
        num_embeddings=wandb.config.num_embeddings, 
        embedding_dim=wandb.config.embedding_dim
    ).to(device)
    
    # Optimizer and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    mse_loss_fn = nn.MSELoss() 
    
    os.makedirs("saved_models", exist_ok=True)

    print("\n🔥 Starting VQ-VAE Training Loop...")
    model.train() 

    for epoch in range(1, wandb.config.epochs + 1):
        total_train_loss = 0
        
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch [{epoch}/{wandb.config.epochs}]", unit="batch")
        
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            # Step A: Zero gradients
            optimizer.zero_grad()
            
            # Step B: Forward pass
            reconstruction, vq_loss = model(batch)
            
            # Step C: Calculate Total Loss
            recon_loss = mse_loss_fn(reconstruction, batch)
            total_loss = recon_loss + vq_loss
            
            # Step D: Backward pass
            total_loss.backward()
            optimizer.step()
            
            total_train_loss += total_loss.item()
            
            # 3. Log metrics to the wandb dashboard!
            # We do this every 10 batches to keep the dashboard fast and clean
            if batch_idx % 10 == 0:
                pbar.set_postfix({"Total Loss": f"{total_loss.item():.4f}"})
                
                wandb.log({
                    "Total Loss": total_loss.item(),
                    "Reconstruction Loss": recon_loss.item(),
                    "VQ Codebook Loss": vq_loss.item(),
                    "Batch": batch_idx + (epoch - 1) * len(train_loader)
                })

            # Quick CPU test break
            # if batch_idx == 10:
            #     break

        # Log the average loss for the whole epoch
        avg_loss = total_train_loss / (batch_idx + 1)
        wandb.log({"Epoch": epoch, "Average Epoch Loss": avg_loss})
        
        print(f"\n✅ Epoch {epoch} Test Complete!\n")
        torch.save(model.state_dict(), f"saved_models/atari_vqvae_epoch_{epoch}.pth")

    # 4. Finish the run cleanly
    wandb.finish()
    print("🎉 VQ-VAE Training Script is ready and fully tracked!")

if __name__ == "__main__":
    train_vqvae()