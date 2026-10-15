import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# Import your custom architectures from the same folder
from models.residual_vqvae import AtariResidualVQVAE
from models.fsq_vae_decoder import AtariFSQVAE

class NatureCNN(nn.Module):
    """
    The canonical 2015 DeepMind Nature network backbone.
    Used as the baseline control group.
    """
    def __init__(self, in_channels=4, feature_dim=512):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # NatureCNN scales byte pixels [0, 255] down to [0.0, 1.0] internally
        return self.main(x.float() / 255.0)


class UniversalFeatureExtractor(nn.Module):
    """
    Acts as the single point of entry for all feature extraction methods.
    Isolates the encoding pathway and discards the decoders.
    """
    def __init__(self, arch_type, checkpoint_path, feature_dim, freeze_weights=True):
        super().__init__()
        self.arch_type = arch_type
        self.feature_dim = feature_dim

        if arch_type == "NatureCNN":
            self.backbone = NatureCNN(in_channels=4, feature_dim=feature_dim)
        elif arch_type == "FSQ-VAE":
            self.backbone = AtariFSQVAE(in_channels=4, hidden_dim=128, fsq_levels=[8, 5, 5, 3])
        elif arch_type == "Residual-VQVAE":
            self.backbone = AtariResidualVQVAE(in_channels=4, num_embeddings=512, embedding_dim=64)
        else:
            raise ValueError(f"Unknown architecture target: {arch_type}")

        # Load weights for pre-trained models
        if checkpoint_path and arch_type != "NatureCNN":
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(clean_dict, strict=False)
            print(f"📦 Successfully loaded frozen {arch_type} weights from {checkpoint_path}")

        # Enforce weight freezing for frozen representation testing
        for param in self.parameters():
            param.requires_grad = not freeze_weights

    def forward(self, x):
        # 1. Fix the dimension mismatch!
        # Gymnasium gives us [Batch, Height, Width, Channels] -> [8, 84, 84, 4]
        # PyTorch needs [Batch, Channels, Height, Width] -> [8, 4, 84, 84]
        if x.shape[-1] == 4:
            x = x.permute(0, 3, 1, 2)
            
        # 2. Scale incoming raw byte frames [0, 255] to floats [0.0, 1.0]
        x_norm = x.float() / 255.0
        
        if self.arch_type == "NatureCNN":
            return self.backbone(x_norm) # Note: Make sure NatureCNN gets x_norm too
            
        elif self.arch_type == "FSQ-VAE":
            z_e = self.backbone.encoder(x_norm)
            z_e = self.backbone.pre_quant_proj(z_e)
            z_q, _ = self.backbone.fsq(z_e)
            return torch.flatten(z_q, start_dim=1)
            
        elif self.arch_type == "Residual-VQVAE":
            encoded = self.backbone.encoder(x_norm)
            quantized, _ = self.backbone.quantizer(encoded)
            return torch.flatten(quantized, start_dim=1)


class PPOAgent(nn.Module):
    """Unified Actor-Critic policy head running on top of the feature extractor."""
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