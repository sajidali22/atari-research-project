# Comprehensive Technical Documentation: Universal Feature Extractor for Atari Games

**Project Author / Researcher**: Oleh Shkalikov & Sajid Ali  
**Academic Context**: Research Project in Reinforcement Learning & Self-Supervised Representation Learning (TU Dresden)  
**Document Target**: Complete technical architecture, methodology, implementation, empirical analysis, and theoretical findings for AI Agent & System Integration.

---

## Executive Summary & Research Context

Atari 2600 environments serve as the benchmark standard for visual Reinforcement Learning (RL). In canonical setups (e.g., NatureCNN with DQN or PPO), a convolutional feature extractor is trained from scratch end-to-end for every individual game environment. Despite every Atari game operating on identical observation dimensions—four stacked grayscale frames of size $84 \times 84$ ($\mathbf{X} \in \mathbb{R}^{B \times 4 \times 84 \times 84}$)—this practice forfeits cross-game visual priors, leading to sample inefficiency, prolonged training times, and lack of visual transfer.

This research project explores the design, pre-training, and downstream RL transfer of a **Universal Feature Extractor** capable of serving as a shared visual perception module across the Atari domain. We systematically investigated four distinct self-supervised representation paradigms:
1. **Continuous Variational Autoencoders (Standard VAE)**
2. **Discrete Vector Quantized VAEs with EMA & Dead Code Revival (Residual VQ-VAE)**
3. **Finite Scalar Quantization VAEs with Dynamic Resize-Convolutions (FSQ-VAE)**
4. **Action-Conditioned & Environment-Conditioned Joint Embedding Predictive Architecture (V-JEPA / V2)**

Each representation paradigm was evaluated on **downstream PPO policy learning** across three experimental arms:
- **Control**: NatureCNN trained end-to-end from scratch per game.
- **Independent (Frozen)**: Representation pre-trained on multi-game expert trajectories and locked; PPO trains policy/value heads only.
- **Joint (Unfrozen Fine-Tuning)**: Pre-trained representation used as a warm-start initialization and fine-tuned end-to-end by PPO loss.

---

## 1. Dataset & Collection Engine

To train a universal feature extractor without reward bias, representation pre-training was conducted on high-density expert trajectories harvested across 15 distinct Atari environments.

### 1.1 Game Split Strategy
The 15 games were partitioned to rigorously evaluate in-distribution representation capacity versus zero-shot out-of-distribution (OOD) generalization:

- **Training Set (11 Games - Representation Pre-Training)**:
  - `BeamRiderNoFrameskip-v4`
  - `BreakoutNoFrameskip-v4`
  - `DemonAttackNoFrameskip-v4`
  - `EnduroNoFrameskip-v4`
  - `MsPacmanNoFrameskip-v4`
  - `PongNoFrameskip-v4`
  - `QbertNoFrameskip-v4`
  - `RiverraidNoFrameskip-v4`
  - `RoadRunnerNoFrameskip-v4`
  - `SeaquestNoFrameskip-v4`
  - `SpaceInvadersNoFrameskip-v4`

- **Held-Out Test Set (4 Games - Zero-Shot Generalization Benchmark)**:
  - `AlienNoFrameskip-v4`
  - `AsteroidsNoFrameskip-v4`
  - `AtlantisNoFrameskip-v4`
  - `IceHockeyNoFrameskip-v4`

### 1.2 Data Collection Pipeline Specifications
- **Emulator Configuration**: Gymnasium (`ale_py`), `frameskip=1`, `repeat_action_probability=0.0`.
- **Frame Stacking**: 4-frame history via `VecFrameStack(env, n_stack=4)`.
- **Observation Normalization**: Raw uint8 bytes $[0, 255]$ normalized to `float32` $[0.0, 1.0]$.
- **Data Volume**: 250,000 transition steps per game across 11 games = **2.75 Million transition steps** ($\sim 11.0\text{M}$ raw frames).
- **Storage Format**: Chunked compressed NumPy arrays (`.npz`) containing:
  - `obs`: $\mathbb{R}^{4 \times 84 \times 84}$ (uint8)
  - `actions`: Integer scalar $a_t \in \{0, \dots, 17\}$
  - `next_obs`: $\mathbb{R}^{4 \times 84 \times 84}$ (uint8)
  - `terminals`: Boolean mask indicating episode termination
  - `mask`: Game-over continuation mask ($1.0 - \text{terminal}$)

---

## 2. Model Architectures & Methodologies

### 2.1 Standard Continuous VAE (`AtariStandardVAE`)
- **Encoder**: 
  - $\text{Conv2d}(4 \to 32, k=8, s=4)$ $\to$ ReLU $\to (32, 20, 20)$
  - $\text{Conv2d}(32 \to 64, k=4, s=2)$ $\to$ ReLU $\to (64, 9, 9)$
  - $\text{Conv2d}(64 \to 64, k=3, s=1)$ $\to$ ReLU $\to (64, 7, 7)$
  - Flattened size: $64 \times 7 \times 7 = 3,136$
  - Linear heads for $\mu \in \mathbb{R}^{256}$ and $\log \sigma^2 \in \mathbb{R}^{256}$
- **Bottleneck**: Gaussian sampling via Reparameterization Trick: $z = \mu + \epsilon \cdot \sigma, \quad \epsilon \sim \mathcal{N}(0, I)$.
- **Decoder**: Linear $256 \to 3,136$, transposed convolutions reverting back to $(4, 84, 84)$ with Sigmoid output.
- **Objective**: $\mathcal{L}_{\text{VAE}} = \text{MSE}(\hat{x}, x) + \beta \cdot D_{\text{KL}}(q(z|x) \parallel \mathcal{N}(0, I))$.
- **Failure Modes & Findings**: Suffer from "Posterior Collapse" and severe spatial blurring. Crushing $84 \times 84$ down to $7 \times 7$ loses sub-3-pixel sprites (Pong ball, laser bullets).

---

### 2.2 Discrete Vector Quantized VAE (`AtariResidualVQVAE`)
- **Encoder**: Residual convolutional blocks maintaining spatial grid resolution at $(64, 21, 21)$.
- **Quantizer**: Exponential Moving Average (EMA) Codebook with $K=512$ embedding vectors of dimension $d=64$, commitment cost $\beta=0.25$, decay $\gamma=0.99$.
- **Straight-Through Estimator (STE)**: $z_q = z_e + \text{detach}(z_q - z_e)$.
- **Failure Modes & Findings**:
  - **Zero-Sum Codebook Collapse / Starvation**: Fixed codebook dictionary size ($512$) creates a zero-sum game when applying weighted loss on moving sprites (10x multiplier). The model dedicates dictionary slots to moving objects (paddles, submarines) while robbing static UI elements (Pong/Breakout scoreboards) or creating severe checkerboard grid patterns on high-contrast backgrounds (Ice Hockey rink).

---

### 2.3 Finite Scalar Quantization VAE (`AtariFSQVAE`)
- **Bottleneck Mechanics**: Eliminates dictionary lookup tables, distance calculations, EMA, and commitment losses ($\mathcal{L}_{\text{VQ}} \equiv 0.0$).
- **Quantization Math**:
  1. Sigmoid bounding: $z_{\text{sig}} = \sigma(z) \in [0, 1]$
  2. Grid scaling: $z_{\text{scaled}} = z_{\text{sig}} \cdot (L - 1)$ for grid levels $L = [8, 5, 5, 3]$
  3. Instant rounding: $z_{\text{rounded}} = \text{round}(z_{\text{scaled}})$
  4. Rescaling to $[-1, 1]$: $z_q = \left(\frac{z_{\text{rounded}}}{L - 1}\right) \cdot 2.0 - 1.0$
  5. Straight-Through Estimator: $z_q = z_{\text{orig}} + \text{detach}(z_q - z_{\text{orig}})$
  - Total implicit codebook capacity: $8 \times 5 \times 5 \times 3 = \mathbf{600}$ discrete states.
- **Dynamic Resize-Conv Decoder**: Replaces `ConvTranspose2d` with nearest-neighbor interpolation stages (`F.interpolate`) followed by standard $3 \times 3$ convolutions to eliminate checkerboard artifacts.
- **Balanced Temporal / Sprite MSE Loss**:
  $$\mathcal{L}_{\text{balanced}} = w_{\text{moving}} \cdot \text{MSE}_{\text{moving}} + (1 - w_{\text{moving}}) \cdot \text{MSE}_{\text{static}}$$
  where $w_{\text{moving}} = 0.85$, isolating moving pixels via frame differences $|x_t - x_{t-1}| > 0.05$.

```python
class FSQBottleneck(nn.Module):
    def __init__(self, levels: list[int]):
        super().__init__()
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.float32))
        
    def forward(self, z):
        z_sig = torch.sigmoid(z)
        scale = self.levels.view(1, -1, 1, 1) - 1.0
        z_scaled = z_sig * scale
        z_rounded = torch.round(z_scaled)
        
        z_q = (z_rounded / scale) * 2.0 - 1.0
        z_orig = (z_sig * scale) / scale * 2.0 - 1.0
        return z_orig + (z_q - z_orig).detach(), torch.tensor(0.0, device=z.device)
```

---

### 2.4 Action- & Environment-Conditioned V-JEPA (`PaperAccurateJEPA` V2)
- **Philosophy**: Non-generative, self-supervised joint embedding prediction in latent space without pixel reconstruction.
- **Spatial Tokenization**: Slices $4 \times 84 \times 84$ frame stacks into $14 \times 14$ non-overlapping patches ($6 \times 6 = 36$ spatial tokens, embed dim $d=256$).
- **Environment Task Embedding (`env_embed`)**: TD-MPC2 / DreamerV2 style task conditioning. Maps `game_id` $\in \{0, \dots, 10\}$ to 256-d token. **Environment Dropout (10-15%)** replaces `game_id` with `UNKNOWN_ID` during pre-training to force the model to learn a generic Atari visual/dynamics prior and support zero-shot transfer.
- **Context Encoder $E_\theta$**: 4-layer Vision Transformer (ViT, 8 heads, GELU, Pre-LayerNorm, SimNorm) taking `[env_token, spatial_tokens]` ($37$ tokens total).
- **Target Encoder $E_\phi$**: EMA copy of $E_\theta$ ($\tau = 0.996$), stop-gradient enforced.
- **Action-Conditioned Predictor**: Predicts target representations $\hat{z}_{\text{target}}$ conditioned on context state $z_{\text{context}}$, action embedding $a_t \in \{0, \dots, 17\}$ (mapped to canonical ALE-18), and env token.
- **Loss Formulation**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{physics}} + \lambda_{\text{inv}} \cdot \mathcal{L}_{\text{inv}}$$
  - **Physics Loss**: $\mathcal{L}_{\text{physics}} = \text{MSE}(\hat{z}_{\text{target}}, z_{\text{target}}) \times \text{mask}$.
  - **Inverse Dynamics Regularizer**: Auxiliary head predicting $a_t$ from $(z_{\text{context}}, z_{\text{target}})$ via Information Gain ($\text{H}(a) - \mathcal{L}_{\text{inv}}$), defeating "Action Collapse" (where the predictor relies purely on visual momentum).

---

## 3. Downstream Reinforcement Learning Setup (PPO)

- **RL Framework**: Proximal Policy Optimization (PPO) via `Stable-Baselines3` / `CleanRL` pipeline.
- **Environment Wrappers**: `make_atari_env`, `VecFrameStack(n_stack=4)`, `VecTransposeImage` (ensuring CHW tensor format `[B, 4, 84, 84]`).
- **Canonical Atari PPO Hyperparameters**:
  - Total Timesteps: $10,000,000$ ($10\text{M}$)
  - Parallel Environments: $8$
  - Rollout Steps ($n\_steps$): $128$ (Buffer size = $1,024$)
  - Minibatch Size: $256$
  - Epochs per Update ($n\_epochs$): $4$
  - Learning Rate: $2.5 \times 10^{-4}$ with Linear Schedule Decay ($2.5\text{e-}4 \to 0$)
  - Discount ($\gamma$): $0.99$, GAE Parameter ($\lambda$): $0.95$
  - Clip Coefficient: $0.1$, Entropy Coefficient: $0.01$, Value Coef: $0.5$
  - Gradient Clipping: $0.5$

### 3.1 Feature Extractor Routing & Pooling Architecture
- **Global Average Pooling vs Flattening**:
  - Initial implementations used spatial flattening ($36 \times 256 = 9,216$ dimensions). When fed into PPO's MLP head, this created $4.7\text{M}$ parameters in the first layer, causing the Critic network to **instantly memorize the batch** (explained variance locking at $1.0$, value loss dropping to $0$).
  - **Fix**: Replaced flattening with **Global Average Pooling** across spatial tokens (`z.mean(dim=1)` $\to 256$ dimensions). This eliminated overfitting, reduced policy parameters, and forced true generalization.

---

## 4. Key Empirical Findings & Diagnostic Results

### 4.1 Representation Gap / Transfer Penalty
- End-to-end NatureCNN baselines achieve higher peak scores on individual games because their convolutional filters rewire specifically for that single environment.
- Frozen Universal Extractors offer **massive throughput advantages** (up to 3x higher FPS during rollouts due to zero backpropagation through convolutions) and rapid early learning, but exhibit a score ceiling gap due to static latent capacity constraints.

### 4.2 The "Missing Bullet" Phenomenon in Pixel/Latent MSE
- Reconstruction-based VAEs and latent JEPA MSE loss functions naturally prioritize large visual patches (backgrounds, maze walls, scoreboards) which constitute $>98\%$ of pixel variance.
- Tiny, lethal 2x2 moving projectiles (bullets in *Space Invaders*, lasers in *BeamRider*) contribute negligibly to MSE, causing frozen encoders to filter them out as high-frequency noise.
- **Joint Training (Unfreezing)** allows PPO reward gradients to pass back into the ViT/CNN attention heads, restoring sensitivity to lethal micro-sprites.

### 4.3 Action-Conditioning & Inverse Dynamics
- Standard JEPAs suffer from "Action Collapse" because stacked frames share $75\%$ identical pixels, allowing the predictor to minimize loss by assuming visual momentum alone.
- Incorporating an Inverse Dynamics Loss / Information Gain metric forces the latent space to maintain strict cause-and-effect accountability to joystick inputs.

### 4.4 Zero-Shot Out-of-Distribution Generalization
- Environment Dropout (`UNKNOWN_ID`) during pre-training prevents the ViT context encoder from fragmenting into disjoint per-game subspaces.
- On unseen test games (`Alien`, `Asteroids`, `Atlantis`, `IceHockey`), the environment-conditioned JEPA maintained stable multi-step rollout predictions ($>0.95$ latent similarity over 5 steps) and enabled zero-shot PPO transfer.

---

## 5. Summary Matrix of Tested Paradigms

| Feature / Metric | Control: NatureCNN | Standard VAE | Residual VQ-VAE | SOTA FSQ-VAE | Action V-JEPA (V2) |
|---|---|---|---|---|---|
| **Representation Type** | Continuous (scratch) | Continuous Latent | Discrete Codebook | Discrete Integer Grid | Self-Supervised Latent |
| **Bottleneck Capacity** | $512$ floats | $256$ floats | $512$ vectors ($d=64$) | $600$ grid states | $36 \times 256$ tokens |
| **Auxiliary Losses** | None | KL Divergence | VQ Commitment Loss | None ($\mathcal{L}_{\text{VQ}}=0$) | Inverse Action Loss |
| **Reconstruction Artifacts** | N/A | Heavy Blur | Codebook Starvation | Minor (Fixed by Resize-Conv) | N/A (Decoder-Free) |
| **Rollout Throughput** | Baseline | Fast (Frozen) | Medium (Frozen) | Fast (Frozen) | Ultra-Fast (GAP 256d) |
| **Zero-Shot Transfer** | None (0%) | Poor | Poor | Moderate | **High (Env Dropout)** |

---

## 6. Code Interface Guidelines for Downstream Agents

```python
# Canonical interface for initializing the universal feature extractor in PPO
from factory import JepaExtractor, V2JEPAExtractor

# 1. Frozen Extractor Initialization (256-d Global Average Pooled)
extractor = JepaExtractor(
    observation_space=env.observation_space,
    checkpoint="production_checkpoints/vjepa_v2_best.pt",
    game="SpaceInvaders",
    env_mode="true",   # or "unknown" for zero-shot OOD games
    freeze=True
)

# 2. Forward pass takes standard Gym CHW tensor [B, 4, 84, 84] in [0, 255] or [0.0, 1.0]
features = extractor(obs_tensor) # Returns [B, 256] clean state representations
```

---
*End of Technical Documentation.*
