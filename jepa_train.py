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
    # 1. Initialize Weights & Biases for Production Run
    wandb.init(
        project="atari-vjepa-world-model",
        name="run-03-production-100ep",
        config={
            "architecture": "Paper-Accurate V-JEPA",
            "dataset": "Atari Expert Transitions (2.75M)",
            "epochs": 100,
            "steps_per_epoch": 1000,    # Capped epoch length to stop disk thrashing
            "batch_size": 512,          # Doubled thanks to AMP
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
            "tau_ema": 0.996,
            "latent_dim": 256,
            "max_grad_norm": 1.0,
            "mixed_precision": True
        }
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Launching Production V-JEPA Training on: {device}")

    # 2. High-Performance Data Pipeline
    train_dataset = AtariTransitionDataset(config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=wandb.config.batch_size, 
        shuffle=True, 
        num_workers=4,               
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True      
    )

    # Validation Pipeline
    val_dataset = AtariTransitionDataset(config.VAL_DIR)
    val_loader = DataLoader(
        val_dataset,
        batch_size=wandb.config.batch_size,
        shuffle=False,               
        num_workers=2,               
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    # 3. Model Setup
    model = PaperAccurateJEPA(
        num_actions=18, 
        embed_dim=wandb.config.latent_dim, 
        tau=wandb.config.tau_ema
    ).to(device)
    
    wandb.watch(model, log="all", log_freq=500)
    
    # 4. Optimizer, Scheduler, and AMP Setup
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params, 
        lr=wandb.config.learning_rate, 
        weight_decay=wandb.config.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=wandb.config.epochs, 
        eta_min=1e-5
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=wandb.config.mixed_precision)

    save_dir = "production_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    global_step = 0
    best_val_loss = float('inf')

    print("🔥 Commencing Latent Dynamics Production Loop...")
    
    for epoch in range(wandb.config.epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{wandb.config.epochs}")
        
        for step, batch in enumerate(pbar):
            # --- SUB-EPOCH FIX: End early to validate and save frequently ---
            if step >= wandb.config.steps_per_epoch:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast("cuda", enabled=wandb.config.mixed_precision):
                z_pred, z_target, z_context = model(batch, return_latents=True)
                
                # 1. Masked JEPA Physics Loss
                raw_mse = torch.nn.functional.mse_loss(z_pred, z_target, reduction='none')
                mse_per_item = raw_mse.mean(dim=[1, 2])
                jepa_loss = (mse_per_item * batch["mask"]).sum() / batch["mask"].sum().clamp(min=1.0)

                # 2. Masked Inverse Dynamics Loss
                raw_inv_loss = model.compute_inverse_loss(z_context, z_target, batch["a_t"])
                inv_loss = (raw_inv_loss * batch["mask"]).sum() / batch["mask"].sum().clamp(min=1.0)

                # 3. Combined Objective
                loss = jepa_loss + (0.1 * inv_loss)
            
            # --- Scaled Backpropagation ---
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=wandb.config.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            model.update_target_network()
            
            if global_step % 50 == 0:
                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/jepa_physics_loss": jepa_loss.item(),
                    "train/inverse_action_loss": inv_loss.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/epoch": epoch
                })
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
            
        # --- Validation Phase ---
        model.eval()
        val_total_loss = 0.0
        val_jepa_loss_total = 0.0
        val_inv_loss_total = 0.0
        
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = {k: v.to(device) for k, v in val_batch.items()}
                
                with torch.amp.autocast("cuda", enabled=wandb.config.mixed_precision):
                    z_pred, z_target, z_context = model(val_batch, return_latents=True)
                    
                    raw_mse = torch.nn.functional.mse_loss(z_pred, z_target, reduction='none')
                    mse_per_item = raw_mse.mean(dim=[1, 2])
                    val_jepa_loss = (mse_per_item * val_batch["mask"]).sum() / val_batch["mask"].sum().clamp(min=1.0)

                    raw_inv_loss = model.compute_inverse_loss(z_context, z_target, val_batch["a_t"])
                    val_inv_loss = (raw_inv_loss * val_batch["mask"]).sum() / val_batch["mask"].sum().clamp(min=1.0)

                    v_loss = val_jepa_loss + (0.1 * val_inv_loss)
                
                val_total_loss += v_loss.item()
                val_jepa_loss_total += val_jepa_loss.item()
                val_inv_loss_total += val_inv_loss.item()

        val_avg_loss = val_total_loss / len(val_loader)
        val_avg_jepa = val_jepa_loss_total / len(val_loader)
        val_avg_inv = val_inv_loss_total / len(val_loader)
            
        avg_loss = total_loss / wandb.config.steps_per_epoch
        print(f"📈 Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        wandb.log({
            "train/epoch_avg_loss": avg_loss,
            "val/epoch_avg_loss": val_avg_loss,
            "val/jepa_physics_loss": val_avg_jepa,
            "val/inverse_action_loss": val_avg_inv,
        })
        
        # --- Smart Checkpointing Strategy ---
        latest_ckpt = os.path.join(save_dir, "vjepa_latest.pt")
        ckpt_payload = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
        }
        torch.save(ckpt_payload, latest_ckpt)
        
        if (epoch + 1) % 10 == 0:
            milestone_ckpt = os.path.join(save_dir, f"vjepa_atari_ep{epoch+1}.pt")
            torch.save(ckpt_payload, milestone_ckpt)
            
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            best_ckpt = os.path.join(save_dir, "vjepa_best.pt")
            torch.save(ckpt_payload, best_ckpt)
            print(f"⭐ New Best Validation Loss: {best_val_loss:.4f}! Saved vjepa_best.pt")

    wandb.finish()
    print("🎉 100-Epoch Production Training Complete!")

if __name__ == "__main__":
    train_jepa()