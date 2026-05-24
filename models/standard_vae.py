import torch
import torch.nn as nn
import torch.nn.functional as F

class StandardVAE(nn.Module):
    def __init__(self, in_channels=4, latent_dim=256):
        """
        Variational Autoencoder designed for continuous latent extraction 
        of 84x84 Atari frames.
        """
        super(StandardVAE, self).__init__()
        self.latent_dim = latent_dim
        self.flatten_size = 64 * 7 * 7 # 3136

        # ==========================================
        # 1. ENCODER: Compress to Latent Space
        # ==========================================
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), # Output: (32, 20, 20)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),          # Output: (64, 9, 9)
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),          # Output: (64, 7, 7)
            nn.ReLU(True)
        )
        
        # Output the Mean and Log Variance
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # ==========================================
        # 2. DECODER: Decompress back to Image
        # ==========================================
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1), 
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), 
            nn.ReLU(True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=8, stride=4), 
            nn.Sigmoid() # Forces final pixels to be strictly between [0.0, 1.0]
        )

    def reparameterize(self, mu, logvar):
        """ The Reparameterization Trick for backpropagation. """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        encoded = self.encoder(x)
        encoded_flat = encoded.view(encoded.size(0), -1) 
        
        # Get latent variables
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded_flat = self.fc_decode(z)
        decoded_reshaped = decoded_flat.view(-1, 64, 7, 7) 
        reconstruction = self.decoder(decoded_reshaped)
        
        return reconstruction, mu, logvar

def standard_vae_loss(reconstruction, original, mu, logvar, beta=1.0):
    """ Loss calculation combining Reconstruction and KL Divergence. """
    recon_loss = F.mse_loss(reconstruction, original, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + (beta * kl_divergence), recon_loss, kl_divergence