import torch
import torch.nn as nn
import torch.nn.functional as F

# Import our previously perfected Codebook!
from models.ema_vqvae import EmaVectorQuantizer

class ResidualBlock(nn.Module):
    """
    A DeepMind-style Residual Block.
    The skip connection allows gradients to flow completely unhindered,
    preventing the network from forgetting fine-grained details.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, num_residual_hiddens, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(num_residual_hiddens, num_hiddens, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualStack(nn.Module):
    """Stacks multiple residual blocks together for deeper feature extraction."""
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([
            ResidualBlock(in_channels, num_hiddens, num_residual_hiddens)
            for _ in range(self.num_residual_layers)
        ])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)

class AtariResidualVQVAE(nn.Module):
    def __init__(self, in_channels=4, num_embeddings=512, embedding_dim=64, num_hiddens=128, num_residual_hiddens=32, num_residual_layers=2, commitment_cost=0.25, decay=0.99):
        super(AtariResidualVQVAE, self).__init__()
        
        # ==========================================
        # 1. ENCODER (84x84 -> 42x42 -> 21x21 -> ResStack)
        # ==========================================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(num_hiddens, num_hiddens, kernel_size=3, stride=1, padding=1),
            ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens),
            nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1) # Map to embedding dimension
        )

        # ==========================================
        # 2. VECTOR QUANTIZER (EMA + Dead Code Revival)
        # ==========================================
        self.quantizer = EmaVectorQuantizer(num_embeddings, embedding_dim)

        # ==========================================
        # 3. DECODER (ResStack -> 21x21 -> 42x42 -> 84x84)
        # ==========================================
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, stride=1, padding=1),
            ResidualStack(num_hiddens, num_hiddens, num_residual_layers, num_residual_hiddens),
            nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(num_hiddens // 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Squashes pixels perfectly back to [0.0, 1.0]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss = self.quantizer(encoded)
        reconstruction = self.decoder(quantized)
        return reconstruction, vq_loss
    

def weighted_sprite_mse_loss(reconstructed, target, multiplier=10.0, threshold=0.01):
    """
    Calculates MSE Loss but heavily penalizes mistakes on moving sprites.
    
    Args:
        reconstructed: The output of the VQ-VAE [B, 4, 84, 84]
        target: The original Atari frames [B, 4, 84, 84]
        multiplier: How much more important moving pixels are (Default: 10x)
        threshold: Minimum pixel color change to be considered "movement"
    """
    # 1. Calculate pixel differences between consecutive frames (Frames 1-2, 2-3, 3-4)
    # diff shape: [B, 3, 84, 84]
    diff = torch.abs(target[:, 1:, :, :] - target[:, :-1, :, :])
    
    # 2. If a pixel changed in ANY of the frame transitions, mark it!
    # movement_mask shape: [B, 1, 84, 84]
    movement_mask, _ = torch.max(diff, dim=1, keepdim=True)
    
    # 3. Create a binary mask (1.0 if moving, 0.0 if static)
    binary_mask = (movement_mask > threshold).float()
    
    # 4. Expand the 1-channel mask to match our 4-channel target
    # expanded_mask shape: [B, 4, 84, 84]
    expanded_mask = binary_mask.expand_as(target)
    
    # 5. Create the Weight Matrix. 
    # Backgrounds default to 1.0. Moving sprites get (1.0 + 9.0 = 10.0)
    weight_matrix = torch.ones_like(target) + (expanded_mask * (multiplier - 1.0))
    
    # 6. Calculate raw, un-averaged MSE
    raw_mse = F.mse_loss(reconstructed, target, reduction='none')
    
    # 7. Apply our weights and average the result
    weighted_mse = (raw_mse * weight_matrix).mean()
    
    return weighted_mse