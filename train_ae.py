import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
from tqdm import tqdm

import config
from dataset import AtariDataset
import math


# 🚨 Import our fresh, clean models!
from models.standard_vae import StandardVAE, standard_vae_loss
from models.ema_vqvae import EmaVqVae
from models.residual_vqvae import AtariResidualVQVAE, weighted_sprite_mse_loss
# from models.fsq_vae import AtariFSQVAE
from models.fsq_vae_decoder import AtariFSQVAE, balanced_sprite_mse_loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def train():
    # Will safely fall back to CPU as you requested
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Starting Training on device: {device}")
    
    # 1. Initialize Weights & Biases
    wandb.init(
        project="atari-universal-feature-extractor",
        name=f"run-{config.MODEL_TYPE}_decoder_HPC",
        config={
            "model_type": config.MODEL_TYPE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "num_embeddings": config.NUM_EMBEDDINGS if config.MODEL_TYPE != 'standard_vae' else None,
            "embedding_dim": config.EMBEDDING_DIM if config.MODEL_TYPE != 'standard_vae' else None
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
        
    elif config.MODEL_TYPE == 'fsq_vae':
        model = AtariFSQVAE(fsq_levels=config.FSQ_LEVELS).to(device)
        
    else:
        raise ValueError(f"❌ Invalid MODEL_TYPE '{config.MODEL_TYPE}' in config.py!")
    
    model = torch.compile(model)

    # 3. Dataset and Optimizer
    train_dataset = AtariDataset(config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4, # Keeps CPU feeding data quickly
        pin_memory=True,
        prefetch_factor=2
    )
    
    test_dataset = AtariDataset(config.TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False)
    # optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    # 3. Wrap the Optimizer in the Scheduler
    # 2. Calculate steps based on your 125 total epochs
    steps_per_epoch = len(train_loader)
    constant_epochs = 100
    decay_epochs = config.EPOCHS - constant_epochs # Should be 25

    # Phase 1: Keep LR completely flat at 3e-4 for the first 100 epochs
    scheduler1 = torch.optim.lr_scheduler.ConstantLR(
        optimizer, 
        factor=1.0, 
        total_iters=constant_epochs * steps_per_epoch
    )

    # Phase 2: Cosine decay down to 1e-5 for the remaining 25 epochs
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=decay_epochs * steps_per_epoch, 
        eta_min=1e-5
    )

    # 3. Chain them together using SequentialLR
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[scheduler1, scheduler2], 
        milestones=[constant_epochs * steps_per_epoch]
    )

    scaler = torch.amp.GradScaler('cuda')
    
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
                
            elif config.MODEL_TYPE in ['ema_vqvae', 'residual_vqvae','fsq_vae']:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    reconstructed, vq_loss = model(batch)
                    # recon_loss = torch.nn.functional.mse_loss(reconstructed, batch)
                    recon_loss = balanced_sprite_mse_loss(
                            reconstructed, 
                            batch,
                            threshold=0.05,       # Adjust to 0.1 if your images are scaled [-1, 1]
                            moving_weight=0.85    # 85% of gradient focus on moving pixels
                        )
                    loss = recon_loss + vq_loss
                    extra_loss = vq_loss
                    loss_name = "VQ Codebook Loss"
                    # ---------------------------------
            
            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            # loss.backward()
            # optimizer.step()
            
            # Tracking and live terminal logging
            total_epoch_loss += loss.item()
            loop.set_postfix({"Total Loss": f"{loss.item():.4f}"})
            
            # Weights & Biases Logging (Every 50 batches)
            if batch_idx % 50 == 0:
                current_lr = scheduler.get_last_lr()[0]
                wandb.log({
                    "Learning Rate": current_lr,
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
        
        model.eval()  # Turn off dropout/batchnorm for evaluation
        total_test_loss = 0.0
        
        with torch.no_grad():
            for test_images in test_loader:
                test_images = test_images.to(device)
                
                # Forward pass
                reconstructed, _ = model(test_images)
                
                # Standard MSE Loss
                test_loss = torch.nn.functional.mse_loss(reconstructed, test_images)
                total_test_loss += test_loss.item()
                
        # Calculate average test loss for the epoch
        avg_test_loss = total_test_loss / len(test_loader)

        # Print the results so you can monitor them in the terminal
        print(f"Epoch {epoch} Summary: Average Test Loss: {avg_test_loss:.4f}")

        # (Optional) Log to Weights & Biases if you are using it
        wandb.log({"Test Loss": avg_test_loss})
    
    wandb.finish()
    print("✅ Training fully completed!")

if __name__ == "__main__":
    train()