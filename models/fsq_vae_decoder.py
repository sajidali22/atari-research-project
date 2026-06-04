import torch
import torch.nn as nn
import torch.nn.functional as F

class FSQBottleneck(nn.Module):
    """
    Finite Scalar Quantization
    Corrected version using sigmoid to safely support even grid levels (e.g., 8).
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.float32))
        
    def forward(self, z):
        # 1. Map to strictly [0, 1]
        z_sig = torch.sigmoid(z)
        
        # 2. Scale to the exact max integer [0, L-1]
        scale = self.levels.view(1, -1, 1, 1) - 1.0
        z_scaled = z_sig * scale
        
        # 3. Quantize cleanly
        z_rounded = torch.round(z_scaled)
        
        # 4. Map back to [-1, 1] for the decoder
        z_q = (z_rounded / scale) * 2.0 - 1.0
        z_orig = (z_sig * scale) / scale * 2.0 - 1.0
        
        # Straight-Through Estimator (STE)
        z_q = z_orig + (z_q - z_orig).detach()
        
        return z_q, torch.tensor(0.0, device=z.device)


class AtariFSQVAE(nn.Module):
    """
    State-of-the-Art Vision Autoencoder for RL World Models.
    Features a high-capacity Resize-Conv decoder to perfectly capture tiny sprites.
    """
    def __init__(self, in_channels=4, hidden_dim=128, fsq_levels=[8, 5, 5, 3]):
        super().__init__()
        
        # -----------------------------------------------------
        # 1. ENCODER
        # -----------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # -----------------------------------------------------
        # 2. FSQ BOTTLENECK
        # -----------------------------------------------------
        self.pre_quant_proj = nn.Conv2d(hidden_dim, len(fsq_levels), kernel_size=1)
        self.fsq = FSQBottleneck(fsq_levels)
        self.post_quant_proj = nn.Conv2d(len(fsq_levels), hidden_dim, kernel_size=1)
        
        # -----------------------------------------------------
        # 3. HIGH-CAPACITY RESIZE-CONV DECODER
        # -----------------------------------------------------
        # Stage 1: Latent Resolution processing
        self.dec_stage1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Stage 2: Mid-Resolution processing
        self.dec_stage2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Stage 3: Full-Resolution final painting
        self.dec_stage3 = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Capture input shape to dynamically prevent any stretching bugs
        B, C, H, W = x.shape
        
        # Encode & Quantize
        z_e = self.encoder(x)
        z_e = self.pre_quant_proj(z_e)
        z_q, vq_loss = self.fsq(z_e)
        
        # Decode Stage 1
        x_d = self.post_quant_proj(z_q)
        x_d = self.dec_stage1(x_d)
        
        # Decode Stage 2 (Dynamic Upsample)
        x_d = F.interpolate(x_d, size=(H // 2, W // 2), mode='nearest')
        x_d = self.dec_stage2(x_d)
        
        # Decode Stage 3 (Dynamic Upsample)
        x_d = F.interpolate(x_d, size=(H, W), mode='nearest')
        reconstructed = self.dec_stage3(x_d)
        
        return reconstructed, vq_loss


# -----------------------------------------------------
# 4. BALANCED TEMPORAL LOSS
# -----------------------------------------------------
def balanced_sprite_mse_loss(reconstructed, target, threshold=0.05, moving_weight=0.85):
    """
    Isolates moving pixels and forces the optimizer to dedicate a fixed 
    percentage of its update to them, bypassing the gradient-washing trap.
    """
    # 1. Temporal difference across stacked frames
    diff = torch.abs(target[:, 1:, :, :] - target[:, :-1, :, :])
    movement_mask, _ = torch.max(diff, dim=1, keepdim=True)
    
    # 2. Mutually exclusive masks
    moving_mask = (movement_mask > threshold).float().expand_as(target)
    static_mask = 1.0 - moving_mask
    
    raw_mse = F.mse_loss(reconstructed, target, reduction='none')
    
    # SAFETY CATCH: If the screen is entirely static or entirely moving (e.g., loading screens)
    if moving_mask.sum() == 0 or static_mask.sum() == 0:
        return raw_mse.mean()
    
    # 3. Isolated means (calculates error without being diluted by the background)
    moving_loss = (raw_mse * moving_mask).sum() / moving_mask.sum()
    static_loss = (raw_mse * static_mask).sum() / static_mask.sum()
    
    # 4. Blend using the fixed ratio (85% focus on the ball/paddles)
    total_loss = (moving_weight * moving_loss) + ((1.0 - moving_weight) * static_loss)
    
    return total_loss