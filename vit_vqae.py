import torch
import torch.nn as nn
from vqvae import VectorQuantizer # Reusing our perfected quantizer!

class ViTEncoder(nn.Module):
    """
    Slices the image into patches and processes them using Self-Attention.
    """
    def __init__(self, in_channels=4, patch_size=12, embed_dim=256, depth=4, heads=8):
        super(ViTEncoder, self).__init__()
        self.patch_size = patch_size
        
        # For an 84x84 image with 12x12 patches, we get a 7x7 grid (49 patches total)
        self.grid_size = 84 // patch_size
        self.num_patches = self.grid_size ** 2 
        
        # 1. Linear projection (Flattening each patch into a vector)
        patch_dim = in_channels * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        
        # 2. Position Embeddings (Tells the Transformer where each patch is located)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 3. Core Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        
        # Step A: "Patchify" the image
        # Shape change: [Batch, Channels, 84, 84] -> [Batch, 49, Channels * P * P]
        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(b, c, -1, p, p).permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(b, -1, c * p * p)
        
        # Step B: Embed and Add Positions
        x = self.patch_embed(x) + self.pos_embed
        
        # Step C: Apply Global Self-Attention
        x = self.transformer(x)
        
        # Step D: Reshape into a 2D grid for our Vector Quantizer
        # Shape change: [Batch, 49, EmbedDim] -> [Batch, EmbedDim, 7, 7]
        x = x.permute(0, 2, 1).contiguous().view(b, -1, self.grid_size, self.grid_size)
        return x

class ViTDecoder(nn.Module):
    """
    Takes the quantized latent codes and rebuilds the Atari frame.
    """
    def __init__(self, out_channels=4, patch_size=12, embed_dim=256, depth=4, heads=8):
        super(ViTDecoder, self).__init__()
        self.patch_size = patch_size
        self.grid_size = 84 // patch_size
        self.num_patches = self.grid_size ** 2
        
        # 1. Position Embeddings for reconstruction
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 2. Core Transformer Decoder (Using Self-Attention to rebuild)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        
        # 3. Projection back to raw pixels
        patch_dim = out_channels * patch_size * patch_size
        self.pixel_projection = nn.Linear(embed_dim, patch_dim)

    def forward(self, x):
        b, embed_dim, h, w = x.shape
        p = self.patch_size
        c = 4 # output channels
        
        # Step A: Flatten the 7x7 grid back into a sequence
        # Shape: [Batch, EmbedDim, 7, 7] -> [Batch, 49, EmbedDim]
        x = x.view(b, embed_dim, -1).permute(0, 2, 1).contiguous()
        
        # Step B: Add positions and process through Transformer
        x = x + self.pos_embed
        x = self.transformer(x)
        
        # Step C: Project vectors back into raw pixel values
        x = self.pixel_projection(x)
        
        # Step D: "Un-Patchify" back into an 84x84 image grid
        x = x.view(b, self.grid_size, self.grid_size, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(b, c, 84, 84)
        
        return torch.sigmoid(x) # Scale pixel values to [0, 1]

class AdvancedViTVQVAE(nn.Module):
    """
    The Ultimate Universal Feature Extractor for GPU.
    """
    def __init__(self, in_channels=4, num_embeddings=512, embed_dim=256):
        super(AdvancedViTVQVAE, self).__init__()
        
        # 1. Transformer Encoder
        self.encoder = ViTEncoder(in_channels=in_channels, embed_dim=embed_dim)
        
        # 2. We must project the Transformer's 256 dimensions down to 64 
        # so our perfectly tuned Quantizer can handle it.
        self.pre_quant_conv = nn.Conv2d(embed_dim, 64, kernel_size=1)
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=64)
        self.post_quant_conv = nn.Conv2d(64, embed_dim, kernel_size=1)
        
        # 3. Transformer Decoder
        self.decoder = ViTDecoder(out_channels=in_channels, embed_dim=embed_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        
        # Compress and Quantize
        encoded_compressed = self.pre_quant_conv(encoded)
        quantized, vq_loss = self.quantizer(encoded_compressed)
        quantized_expanded = self.post_quant_conv(quantized)
        
        # Reconstruct
        reconstruction = self.decoder(quantized_expanded)
        return reconstruction, vq_loss

if __name__ == "__main__":
    # --- Quick GPU Test ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Advanced ViT-VQ-VAE on: {device}")
    
    # Simulating a batch of 8 stacked Atari frames (84x84)
    dummy_batch = torch.rand(8, 4, 84, 84).to(device)
    
    model = AdvancedViTVQVAE().to(device)
    recon, vq_loss = model(dummy_batch)
    
    print(f"Input Data: {dummy_batch.shape}")
    print(f"Output Reconstruction: {recon.shape}")
    print("✅ Model successfully built and processed the data!")