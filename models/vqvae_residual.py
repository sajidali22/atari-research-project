import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# CNN Residual Block
# ==========================================
class ResidualBlock(nn.Module):
    """
    A standard ResNet block. It passes the data through convolutions,
    and then adds the original input back to the result (skip connection).
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            # Padding=1 keeps the image dimensions exactly the same
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return x + self.block(x)

# ==========================================
# The Vector Quantizer (EMA + Dead Code Revival)
# ==========================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.clone(self.embeddings.weight.data))

    def forward(self, inputs):
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = self.embeddings(encoding_indices).view(inputs.shape)

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            # --- DEAD CODE REVIVAL ---
            usage_threshold = 1.0 
            dead_codes = self._ema_cluster_size < usage_threshold
            
            if torch.any(dead_codes):
                num_dead = dead_codes.sum().item()
                random_indices = torch.randperm(flat_inputs.size(0))[:num_dead]
                
                self.embeddings.weight.data[dead_codes] = flat_inputs[random_indices]
                self._ema_w[dead_codes] = flat_inputs[random_indices]
                self._ema_cluster_size[dead_codes] = usage_threshold 
            # -------------------------

            n_total = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n_total + self.num_embeddings * self.epsilon) * n_total)
            
            dw = torch.matmul(encodings.t(), flat_inputs)
            self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw
            self.embeddings.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        return quantized, vq_loss

class AtariVQVAE(nn.Module):
    def __init__(self, in_channels=4, num_embeddings=512, embedding_dim=64):
        super(AtariVQVAE, self).__init__()
        
        # ADVANCED ENCODER
        self.encoder = nn.Sequential(
            # 1. Spatial Reduction (Same as before)
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),         
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),         
            
            # 2. Advanced Feature Processing (Residual Blocks)
            ResidualBlock(64),
            ResidualBlock(64),
            
            # 3. Compress to embedding dimension
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=1, stride=1)
        )

        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # ADVANCED DECODER
        self.decoder = nn.Sequential(
            # 1. Expand from embedding dimension
            nn.Conv2d(embedding_dim, 64, kernel_size=1, stride=1),
            
            # 2. Advanced Feature Processing (Residual Blocks)
            ResidualBlock(64),
            ResidualBlock(64),
            nn.ReLU(),
            
            # 3. Spatial Expansion (Same as before)
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=8, stride=4), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quantized, vq_loss = self.quantizer(encoded)
        reconstruction = self.decoder(quantized)
        return reconstruction, vq_loss