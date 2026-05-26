import torch
import torch.nn as nn

class FSQBottleneck(nn.Module):
    """
    Finite Scalar Quantization
    Transforms continuous vectors into discrete grid coordinates using instantaneous rounding.
    """
    def __init__(self, levels: list[int]):
        super().__init__()
        # The levels define the grid dimensions (e.g., [8, 5, 5, 3] = 600 combinations)
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.float32))
        
    def forward(self, z):
        # 1. Bound the continuous representation tightly between [-1, 1]
        z = torch.tanh(z)
        
        # 2. Scale to the bounds of our specific integer grid
        half_l = (self.levels.view(1, -1, 1, 1) - 1) / 2
        z_scaled = z * half_l
        
        # 3. Quantize (O(1) instant rounding, completely eliminating the codebook!)
        z_rounded = torch.round(z_scaled)
        
        # 4. Scale back to [-1, 1] for the decoder to process smoothly
        z_q = z_rounded / half_l
        
        # 5. Straight-Through Estimator (STE) to allow backpropagation through the rounding step
        z_q = z + (z_q - z).detach()
        
        # We return 0.0 for the vq_loss so you don't have to rewrite your train.py training loop
        return z_q, torch.tensor(0.0, device=z.device)


class AtariFSQVAE(nn.Module):
    """
    State-of-the-Art Vision Autoencoder for RL World Models.
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
        # 2. SOTA BOTTLENECK (FSQ)
        # -----------------------------------------------------
        # Crush the 128 feature channels down to match our number of FSQ levels (4 channels)
        self.pre_quant_proj = nn.Conv2d(hidden_dim, len(fsq_levels), kernel_size=1)
        
        self.fsq = FSQBottleneck(fsq_levels)
        
        # Expand the 4 integer channels back to 128 for the decoder
        self.post_quant_proj = nn.Conv2d(len(fsq_levels), hidden_dim, kernel_size=1)
        
        # -----------------------------------------------------
        # 3. DECODER
        # -----------------------------------------------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim // 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encode to continuous space
        z_e = self.encoder(x)
        z_e = self.pre_quant_proj(z_e)
        
        # Quantize instantly using FSQ grid math
        z_q, vq_loss = self.fsq(z_e)
        
        # Decode back to image pixels
        z_q_expanded = self.post_quant_proj(z_q)
        reconstructed = self.decoder(z_q_expanded)
        
        return reconstructed, vq_loss