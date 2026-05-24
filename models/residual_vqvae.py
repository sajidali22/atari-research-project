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