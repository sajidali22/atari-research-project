import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# Import your custom architectures directly from your repository files
# Assuming your definitions are saved in models/residual_vqvae.py and models/fsq_vae.py
from models.residual_vqvae import AtariResidualVQVAE
from models.fsq_vae import AtariFSQVAE, NatureCNN # Assumed NatureCNN definition included here

class UniversalFeatureExtractor(nn.Module):
    """
    Acts as the single point of entry for all feature extraction methods.
    Isolates the encoding pathway and slices away the unnecessary decoders.
    """
    def __init__(self, arch_type, checkpoint_path, feature_dim, freeze_weights=True):
        super().__init__()
        self.arch_type = arch_type
        self.feature_dim = feature_dim

        # 1. Initialize structural backbone
        if arch_type == "NatureCNN":
            self.backbone = NatureCNN(in_channels=4, feature_dim=feature_dim)
        elif arch_type == "FSQ-VAE":
            self.backbone = AtariFSQVAE(in_channels=4, hidden_dim=128, fsq_levels=[8, 5, 5, 3])
        elif arch_type == "Residual-VQVAE":
            self.backbone = AtariResidualVQVAE(in_channels=4, num_embeddings=512, embedding_dim=64)
        else:
            raise ValueError(f"Unknown architecture target string: {arch_type}")

        # 2. Dynamic checkpoint loading for frozen runs
        if checkpoint_path and arch_type != "NatureCNN":
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            
            # Load with strict=False because we only care about the encoder layers
            self.backbone.load_state_dict(clean_dict, strict=False)
            print(f"📦 Successfully loaded frozen {arch_type} weights from {checkpoint_path}")

        # 3. Enforce training isolation
        for param in self.parameters():
            param.requires_grad = not freeze_weights

    def forward(self, x):
        # Scale incoming raw byte frames [0, 255] to floats [0.0, 1.0]
        x_norm = x.float() / 255.0
        
        if self.arch_type == "NatureCNN":
            return self.backbone(x)
            
        elif self.arch_type == "FSQ-VAE":
            # Extract features using your FSQ forward logic up to the bottleneck
            z_e = self.backbone.encoder(x_norm)
            z_e = self.backbone.pre_quant_proj(z_e)
            z_q, _ = self.backbone.fsq(z_e)
            return torch.flatten(z_q, start_dim=1)
            
        elif self.arch_type == "Residual-VQVAE":
            # Extract features using your VQVAE forward logic up to the bottleneck
            encoded = self.backbone.encoder(x_norm)
            quantized, _ = self.backbone.quantizer(encoded)
            return torch.flatten(quantized, start_dim=1)


class PPOAgent(nn.Module):
    """Unified Actor-Critic policy layer running on top of the factory."""
    def __init__(self, config, num_actions):
        super().__init__()
        self.feature_extractor = UniversalFeatureExtractor(
            arch_type=config['extractor']['arch_type'],
            checkpoint_path=config['extractor']['checkpoint_path'],
            feature_dim=config['extractor']['feature_dim'],
            freeze_weights=config['extractor']['freeze_weights']
        )
        
        feature_dim = config['extractor']['feature_dim']
        
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def get_value(self, x):
        return self.critic(self.feature_extractor(x))

    def get_action_and_value(self, x, action=None):
        features = self.feature_extractor(x)
        logits = self.actor(features)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(features)