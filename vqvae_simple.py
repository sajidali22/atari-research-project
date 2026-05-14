import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Original Spatial Logic + EMA + Dead Code Revival.
    This forces the model to use all 512 codes for Universal Feature Extraction.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Create the dictionary (codebook)
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

        # Buffers for EMA (Exponential Moving Average) to track code usage
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.clone(self.embeddings.weight.data))

    def forward(self, inputs):
        # 1. Original Flattening Logic
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # 2. Calculate distances to find the closest dictionary codes
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # Create one-hot encodings to count how often each code is used
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # 3. Original Un-flattening Logic (The way that worked for Ms. Pac-Man!)
        quantized = self.embeddings(encoding_indices).view(inputs.shape)

        # 4. EMA Dictionary Updates & Dead Code Revival (Only runs during training)
        if self.training:
            # Update our running count of code usage
            self._ema_cluster_size = self._ema_cluster_size * self.decay + \
                                     (1 - self.decay) * torch.sum(encodings, 0)
            
            # --- DEAD CODE REVIVAL LOGIC ---
            usage_threshold = 1.0 # If a code is used less than this, it is "dead"
            dead_codes = self._ema_cluster_size < usage_threshold
            
            if torch.any(dead_codes):
                num_dead = dead_codes.sum().item()
                
                # Pick random active features from the current game batch
                random_indices = torch.randperm(flat_inputs.size(0))[:num_dead]
                
                # Teleport the dead codes to these new active locations
                self.embeddings.weight.data[dead_codes] = flat_inputs[random_indices]
                self._ema_w[dead_codes] = flat_inputs[random_indices]
                
                # Reset their usage count so they stay alive
                self._ema_cluster_size[dead_codes] = usage_threshold 
            # -------------------------------

            # Standard EMA dictionary weight update
            n_total = torch.sum(self._ema_cluster_size)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self.epsilon)
                / (n_total + self.num_embeddings * self.epsilon) * n_total)
            
            dw = torch.matmul(encodings.t(), flat_inputs)
            self._ema_w = self._ema_w * self.decay + (1 - self.decay) * dw
            
            # Apply the updated weights to the actual dictionary
            self.embeddings.weight.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

        # 5. Losses
        # Because EMA updates the dictionary, we only need the commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        vq_loss = self.commitment_cost * e_latent_loss

        # 6. Straight-through estimator (Copies gradients from decoder to encoder)
        quantized = inputs + (quantized - inputs).detach()

        return quantized, vq_loss

class AtariVQVAE(nn.Module):
    def __init__(self, in_channels=4, num_embeddings=512, embedding_dim=64):
        super(AtariVQVAE, self).__init__()
        
        # ==========================================
        # 1. ENCODER (Math: 84x84 -> 20x20 -> 9x9 -> 7x7)
        # ==========================================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),         
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),         
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, kernel_size=1, stride=1)
        )

        # ==========================================
        # 2. VECTOR QUANTIZER (The Codebook)
        # ==========================================
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)

        # ==========================================
        # 3. DECODER (Math: 7x7 -> 9x9 -> 20x20 -> 84x84)
        # ==========================================
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, kernel_size=1, stride=1),
            nn.ReLU(),
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