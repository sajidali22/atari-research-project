import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import wandb

# Import your custom modules
from dataset_JEPA import AtariTransitionDataset
from models.JEPA import PaperAccurateJEPA 
import config

def train_jepa():
    # 1. Initialize Weights & Biases
    wandb.init(
        project="atari-vjepa-world-model",
        name="run-01-transformer-jepa",
        config={
            "architecture": "Paper-Accurate V-JEPA",
            "dataset": "Atari Expert Transitions",
            "epochs": 15,
            "batch_size": 256,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "tau_ema": 0.996,
            "latent_dim": 256,
            "max_grad_norm": 1.0
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Launching Paper-Accurate V-JEPA Training on: {device}")

    # 2. Data Pipeline
    train_dataset = AtariTransitionDataset(config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=wandb.config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True       
    )

    # 3. Model Setup
    model = PaperAccurateJEPA(num_actions=18, embed_dim=wandb.config.latent_dim, tau=wandb.config.tau_ema).to(device)
    
    # Track the model topology and gradients in W&B
    wandb.watch(model, log="all", log_freq=100)
    
    # 4. Optimizer Setup
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params, 
        lr=wandb.config.learning_rate, 
        weight_decay=wandb.config.weight_decay
    )

    save_dir = "jepa_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    global_step = 0

    print("🔥 Commencing Latent Dynamics Training Loop...")
    
    for epoch in range(wandb.config.epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs}")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # --- Forward Pass ---
            z_pred, z_target, z_context = model(batch, return_latents=True)
            
            # --- 1. Masked JEPA Physics Loss ---
            # --- 1. Masked JEPA Physics Loss ---
            raw_mse = torch.nn.functional.mse_loss(z_pred, z_target, reduction='none')
            mse_per_item = raw_mse.mean(dim=[1, 2])
            jepa_loss = (mse_per_item * batch["mask"]).sum() / batch["mask"].sum().clamp(min=1.0)

            # --- 2. Masked Inverse Dynamics Loss ---
            raw_inv_loss = model.compute_inverse_loss(z_context, z_target, batch["a_t"])
            inv_loss = (raw_inv_loss * batch["mask"]).sum() / batch["mask"].sum().clamp(min=1.0)

            # --- 3. Combined Objective ---
            loss = jepa_loss + (0.1 * inv_loss)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=wandb.config.max_grad_norm)
            
            optimizer.step()
            model.update_target_network()
            
            # --- W&B Step Logging (Upgraded) ---
            wandb.log({
                "train/total_loss": loss.item(),
                "train/jepa_physics_loss": jepa_loss.item(),
                "train/inverse_action_loss": inv_loss.item(), # Monitor this strictly!
                "train/global_step": global_step,
                "train/epoch": epoch
            })
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Epoch Summary
        avg_loss = total_loss / len(train_loader)
        print(f"📈 Epoch {epoch+1} Completed | Average L2 Loss: {avg_loss:.4f}")
        
        # W&B Epoch Logging
        wandb.log({"train/epoch_avg_loss": avg_loss})
        
        # Save Checkpoint
        ckpt_path = os.path.join(save_dir, f"vjepa_atari_ep{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        
        # Optional: Save the checkpoint artifact to W&B cloud
        # wandb.save(ckpt_path)

    # Close W&B session
    wandb.finish()
    print("🎉 Training Complete!")

if __name__ == "__main__":
    train_jepa()