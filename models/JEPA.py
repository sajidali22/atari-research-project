import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class PatchEmbedding(nn.Module):
    """
    Step 1: The Spatial Tokenizer.
    Origin: Vision Transformer (ViT) architecture (Dosovitskiy et al., 2020).
    Why: Transformers cannot process 2D images directly. We must slice the screen 
    into a grid and project each slice into a 1D continuous token.
    """
    def __init__(self, in_channels=4, patch_size=14, embed_dim=256):
        super().__init__()
        # Atari frames are 84x84. Using a 14x14 patch yields exactly a 6x6 grid.
        # 6 * 6 = 36 total spatial tokens per frame stack.
        self.num_patches = (84 // patch_size) ** 2  
        
        # A Convolution with stride == kernel_size ensures patches never overlap.
        # This acts as a mathematically perfect cookie-cutter.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable Positional Embeddings.
        # Why: Self-attention is permutation invariant (it doesn't know order).
        # We add these parameters so the network knows Token 1 is the "top left" of the screen.
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        # Input 'x' comes in from the dataloader as: [Batch, 4, 84, 84]
        x = self.proj(x) # Shape becomes: [Batch, embed_dim, 6, 6]
        
        # We must flatten the spatial grid to create a linear sequence for the Transformer.
        # .flatten(2) collapses the 6x6 grid into 36. Shape: [Batch, embed_dim, 36]
        # .transpose(1, 2) swaps the dimensions to standard Transformer format: [Batch, Seq_Len, Embed_Dim]
        x = x.flatten(2).transpose(1, 2) # Final Shape: [Batch, 36, 256]
        
        # Add positional spatial awareness via broadcasting
        return x + self.pos_embed


class TransformerEncoder(nn.Module):
    """
    Step 2: The Latent Context & Target Backbone.
    Origin: I-JEPA (Assran et al., 2023).
    Why: Uses Multi-Head Self-Attention to build relationships between spatial patches.
    """
    def __init__(self, in_channels=4, embed_dim=256, depth=4, heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim)
        
        # Why norm_first=True? Pre-LayerNorm is the modern standard for Deep Transformers 
        # (unlike the original 2017 paper). It prevents gradient vanishing in deep networks.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu", 
            batch_first=True,
            norm_first=True 
        )
        # Stack the identical attention layers
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        
        # Final normalization to stabilize the latent vectors before the loss calculation
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Input shape: [Batch, 4, 84, 84]
        x = self.patch_embed(x) # Tokenize: [Batch, 36, 256]
        x = self.blocks(x)      # Apply Self-Attention
        return self.norm(x)     # Output: [Batch, 36, 256]


class TransformerPredictor(nn.Module):
    """
    Step 3: The Action-Conditioned Dynamics Engine.
    Origin: V-JEPA (Bardes et al., 2024).
    Why: Uses self-attention to predict the next state by routing the action 
    prompt strictly to the spatial patches it actually affects.
    """
    def __init__(self, num_actions=18, embed_dim=256, depth=4, heads=8):
        super().__init__()
        # Maps the discrete joystick integer (0-17) into the same 256-d continuous space as the image tokens
        self.action_embed = nn.Embedding(num_actions, embed_dim)
        
        # The predictor is usually shallower than the encoder to force the encoder 
        # to do the heavy lifting of representation learning.
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim * 4, 
            activation="gelu", 
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(predictor_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, context_tokens, action):
        # context_tokens shape: [Batch, 36, 256]
        
        # Embed the action. unsqueeze(1) adds a sequence dimension so it acts as a "token"
        # a_token shape: [Batch, 1, 256]
        a_token = self.action_embed(action).unsqueeze(1)
        
        # Prepend the action token to the spatial sequence. 
        # Sequence length becomes 37 (1 action + 36 image patches).
        # Shape: [Batch, 37, 256]
        x = torch.cat([a_token, context_tokens], dim=1)
        
        # Process through predictor attention layers
        x = self.blocks(x)
        x = self.norm(x)
        
        # CRITICAL SLICE: Drop the first token (the action token) from the output.
        # Why: We only want to predict the 36 spatial image patches to match the Target Encoder's output.
        pred_spatial_tokens = x[:, 1:, :] 
        return pred_spatial_tokens # Final Shape: [Batch, 36, 256]


class PaperAccurateJEPA(nn.Module):
    """
    Step 4: The Joint-Embedding Architecture Orchestrator.
    Why: Manages the asymmetric gradient flows required to prevent representation collapse 
    (where the model cheats by outputting zeros for everything).
    """
    def __init__(self, num_actions=18, embed_dim=256, tau=0.996):
        super().__init__()
        self.tau = tau # Exponential Moving Average coefficient
        
        # --- TRAINABLE NETWORKS (Gradients Flow Here) ---
        # Depth 6 for Context, Depth 3 for Predictor is a standard asymmetric ratio in JEPA papers.
        self.context_encoder = TransformerEncoder(embed_dim=embed_dim, depth=6)
        self.predictor = TransformerPredictor(num_actions=num_actions, embed_dim=embed_dim, depth=3)
        
        # --- TARGET NETWORK (No Gradients Flow Here) ---
        # We create an exact architectural copy of the Context Encoder.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        
        # Explicitly sever the PyTorch autograd computation graph for the target network.
        # This forces the network to learn predictive representations instead of collapsing.
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.inverse_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, num_actions)
        )


    @torch.no_grad() # Decorator ensures no memory is wasted tracking gradients during this step
    def update_target_network(self):
        """
        Executes the Target Network momentum update.
        Formula: phi <- (tau * phi) + (1 - tau) * theta
        Why: Slowly tracks the context encoder's weights to provide a stable, evolving learning target.
        """
        for param_context, param_target in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            # Using in-place operations (.mul_ and .add_) prevents memory fragmentation on the GPU
            param_target.data.mul_(self.tau).add_(param_context.data, alpha=1.0 - self.tau)

    # def forward(self, batch):
    #     # Extract components from the Dataloader batch dictionary
    #     s_t = batch["s_t"]          # [Batch, 4, 84, 84]
    #     a_t = batch["a_t"]          # [Batch]
    #     s_next = batch["s_next"]    # [Batch, 4, 84, 84]
    #     mask = batch["mask"]        # [Batch]
        
    #     # 1. Encode the current frame stack into latent patches
    #     z_t = self.context_encoder(s_t) # [Batch, 36, 256]
        
    #     # 2. Encode the target frame stack. 
    #     # Wrapped in torch.no_grad() as a secondary safety measure against graph leaks.
    #     with torch.no_grad():
    #         z_next_target = self.target_encoder(s_next) # [Batch, 36, 256]
            
    #     # 3. Emulate the world dynamics to predict the future state
    #     z_next_pred = self.predictor(z_t, a_t) # [Batch, 36, 256]
        
    #     # 4. Dense L2 Distance Loss Calculation
    #     # reduction="none" leaves the tensor at shape [Batch, 36, 256].
    #     # .mean(dim=[1, 2]) averages the MSE across all 36 spatial patches and all 256 feature dimensions.
    #     # This collapses the shape down to [Batch], providing exactly one scalar loss value per transition.
    #     loss_per_batch = F.mse_loss(z_next_pred, z_next_target, reduction="none").mean(dim=[1, 2])
        
    #     # 5. Terminal State Masking
    #     # We multiply by the batch mask (0.0 if episode ended, 1.0 if alive).
    #     # Why: It is impossible for the model to predict the start of a brand new game based on 
    #     # the end of a previous game. This mask neutralizes the loss on terminal steps so the 
    #     # gradients aren't corrupted by unpredictable transitions.
    #     masked_loss = (loss_per_batch * mask).sum() / (mask.sum() + 1e-8)
        
    #     return masked_loss
    
    def forward(self, batch, return_latents=False):
        # Extract components from the Dataloader batch dictionary
        s_t = batch["s_t"]          # [Batch, 4, 84, 84]
        a_t = batch["a_t"]          # [Batch]
        s_next = batch["s_next"]    # [Batch, 4, 84, 84]
        mask = batch.get("mask", None) # Safe fallback if mask isn't provided
        
        # 1. Encode the current frame stack into latent patches
        z_t = self.context_encoder(s_t) # [Batch, 36, 256]
        
        # 2. Encode the target frame stack. 
        with torch.no_grad():
            z_next_target = self.target_encoder(s_next) # [Batch, 36, 256]
            
        # 3. Emulate the world dynamics to predict the future state
        z_next_pred = self.predictor(z_t, a_t) # [Batch, 36, 256]
        
        # --- NEW API SUPPORT FOR TRAIN.PY ---
        if return_latents:
            # Returns the raw tensors so train.py can apply masking and inverse loss
            return z_next_pred, z_next_target, z_t
        
        # --- FALLBACK (For dummy testing at the bottom of the script) ---
        loss_per_batch = F.mse_loss(z_next_pred, z_next_target, reduction="none").mean(dim=[1, 2])
        if mask is not None:
            masked_loss = (loss_per_batch * mask).sum() / (mask.sum() + 1e-8)
            return masked_loss
        return loss_per_batch.mean()
    
    def compute_inverse_loss(self, z_current, z_next, true_actions):
        """
        Forces the latent space to encode enough information to deduce 
        which action caused the transition between two states.
        """
        # z_current and z_next shape: [Batch, Sequence_Length, Embed_Dim]
        # We pool across the spatial/sequence dimension to get a global state vector
        z_c_pooled = z_current.mean(dim=1) # Shape: [Batch, Embed_Dim]
        z_n_pooled = z_next.mean(dim=1)    # Shape: [Batch, Embed_Dim]
        
        # Concatenate the before and after states
        combined_states = torch.cat([z_c_pooled, z_n_pooled], dim=-1) # Shape: [Batch, Embed_Dim * 2]
        
        # Predict the action logits
        action_logits = self.inverse_head(combined_states)
        
        # Return standard Cross Entropy against the actual joystick inputs
        return torch.nn.functional.cross_entropy(action_logits, true_actions, reduction='none')

if __name__ == "__main__":
    # Standard dummy test to verify architecture compiles and runs before throwing it at the GPU
    print("Initializing Paper-Accurate V-JEPA Architecture...")
    jepa = PaperAccurateJEPA(num_actions=18, embed_dim=256, tau=0.995)
    
    dummy_batch = {
        "s_t": torch.randn(16, 4, 84, 84),
        "a_t": torch.randint(0, 18, (16,)),
        "s_next": torch.randn(16, 4, 84, 84),
        "mask": torch.ones(16)
    }
    
    loss = jepa(dummy_batch)
    print(f"Forward Pass Successful. Computed L2 Loss: {loss.item():.4f}")