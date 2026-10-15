"""
Environment-conditioned action-conditional V-JEPA world model.

v2 changes over JEPA_v1.py, all following TD-MPC2 (Hansen et al., 2024):

  1. ENVIRONMENT EMBEDDING. TD-MPC2 trains one agent over 80 heterogeneous tasks
     by conditioning every component on a learnable task embedding e:
     h(s,e), d(z,a,e), R(z,a,e), Q(z,a,e), p(z,e). We do the Atari analogue --
     one learnable embedding per game, injected as an extra token into the
     encoder AND the predictor, and concatenated into the inverse head.

  2. UNKNOWN SLOT. TD-MPC2's stated limitation is that it "requires training on
     all target tasks simultaneously rather than enabling zero-shot generalization
     to completely unseen tasks." We reserve env slot games.UNKNOWN_ID and train it
     by randomly replacing the true game id during training (classifier-free-guidance
     style). At test time an unseen game is run with UNKNOWN -- zero-shot -- and can
     then be few-shot adapted by learning only its own embedding row.

  3. SIMNORM. TD-MPC2 Eq. 5 projects the latent onto L fixed-dimensional simplices
     via softmax. This bounds the representation and removes the degenerate constant
     vector as a reachable fixed point of the stop-grad + EMA system, which is the
     top-ranked liability in THEORETICAL_VALIDATION_DEBATE.md Sec 2.1.

  4. CANONICAL ACTIONS + MASKING. Actions arrive already remapped to the ALE-18
     index space by dataset_JEPA.py, and the inverse-dynamics head masks logits to
     the acting environment's valid actions -- TD-MPC2's action masking for
     heterogeneous action spaces.

Structural note: env_embed lives on PaperAccurateJEPA, NOT inside TransformerEncoder.
copy.deepcopy(self.context_encoder) would otherwise create a second, frozen,
EMA-lagged copy of the table. The encoders receive the embedding VECTOR, never the id.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import games


def to_float(x):
    """Normalise uint8 frames to [0,1] floats; pass floats through untouched.

    dataset_JEPA emits uint8 so the worker->main IPC and host->device copy carry 56 KB
    per sample instead of 226 KB. This is the ONE place that conversion happens, so no
    consumer can silently diverge -- every path that feeds pixels to an encoder goes
    through here, whether via forward() or by calling context_encoder directly.
    """
    if x.dtype == torch.uint8:
        return x.float().div_(255.0)
    return x


def strip_compile_prefix(state_dict):
    """Drop the '_orig_mod.' prefix torch.compile adds to every parameter name.

    jepa_train saves the uncompiled module, so this is normally a no-op -- it exists so
    a checkpoint written by some other compiled path still loads. Same fix as
    PPO/factory.py:53-57.
    """
    if not any(k.startswith("_orig_mod.") for k in state_dict):
        return state_dict
    return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}


class SimNorm(nn.Module):
    """Simplicial normalisation (TD-MPC2 Eq. 5).

    Partitions the final dimension into L groups of size V and softmaxes each group,
    so the latent lives on a product of L simplices: every entry is in (0,1) and each
    group sums to 1.

    Why it matters here: the JEPA objective is minimised by any constant latent, and
    with a stop-gradient EMA target that constant is an actual fixed point of the
    system -- nothing in v1 prevented it except the inverse-dynamics term. A vector on
    a simplex cannot shrink toward zero, so collapse stops being reachable by scale.

    Cost: the mean entry is 1/V = 0.125, so the JEPA MSE term shrinks by roughly an
    order of magnitude while the inverse cross-entropy (in nats) does not move at all.
    The v1 loss weight of 0.1 is therefore far too large post-SimNorm -- see
    config.INVERSE_LOSS_WEIGHT.
    """

    def __init__(self, V=8):
        super().__init__()
        self.V = V

    def forward(self, x):
        shape = x.shape
        if shape[-1] % self.V != 0:
            raise ValueError(f"SimNorm: last dim {shape[-1]} not divisible by V={self.V}")
        x = x.view(*shape[:-1], shape[-1] // self.V, self.V)
        x = F.softmax(x, dim=-1)
        return x.view(*shape)

    def extra_repr(self):
        return f"V={self.V}"


class PatchEmbedding(nn.Module):
    """Spatial tokenizer (ViT, Dosovitskiy et al., 2020).

    84x84 with a 14x14 non-overlapping patch gives exactly a 6x6 = 36 token grid.
    The 4 stacked frames are fused at patchify time, so the sequence axis is purely
    spatial -- there is no temporal token axis.
    """

    def __init__(self, in_channels=4, patch_size=14, embed_dim=256):
        super().__init__()
        self.num_patches = (84 // patch_size) ** 2  # 36
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Self-attention is permutation invariant; these tell the network which
        # token is the top-left of the screen.
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x)                  # [B, D, 6, 6]
        x = x.flatten(2).transpose(1, 2)  # [B, 36, D]
        return x + self.pos_embed


class TransformerEncoder(nn.Module):
    """Latent context / target backbone (I-JEPA, Assran et al., 2023).

    Takes the environment embedding as a VECTOR (not an id) and prepends it as a
    token, so the same visual patch can be interpreted differently depending on which
    game is being played. The env token is sliced off the output, leaving 36 spatial
    tokens for the loss.
    """

    def __init__(self, in_channels=4, embed_dim=256, depth=6, heads=8, use_simnorm=True, simnorm_V=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=in_channels, embed_dim=embed_dim)

        # Pre-LayerNorm (norm_first=True) is the modern standard for deep transformers;
        # it prevents gradient vanishing relative to the original 2017 post-norm design.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.simnorm = SimNorm(simnorm_V) if use_simnorm else nn.Identity()

    def forward(self, x, e):
        # x: [B, 4, 84, 84] uint8 or float   e: [B, D] environment embedding vector
        # Normalisation lives here rather than in the dataset or in forward(), because
        # every pixel path -- forward(), and the eval scripts that call the encoders
        # directly -- necessarily passes through an encoder. Idempotent on floats.
        x = to_float(x)
        x = self.patch_embed(x)                       # [B, 36, D]
        x = torch.cat([e.unsqueeze(1), x], dim=1)     # [B, 37, D]
        x = self.blocks(x)
        x = self.norm(x)
        return self.simnorm(x[:, 1:, :])              # drop env token -> [B, 36, D]


class TransformerPredictor(nn.Module):
    """Action-conditioned dynamics (V-JEPA, Bardes et al., 2024).

    Sequence is [env, action, 36 spatial patches]. Attention routes the action prompt
    to the patches it actually affects, and the env token tells it which game's physics
    to apply -- the same joystick command means different things in Pong and Seaquest.
    """

    def __init__(self, num_actions=18, embed_dim=256, depth=3, heads=8, use_simnorm=True, simnorm_V=8):
        super().__init__()
        # Actions are canonical ALE-18 indices (see games.py), so this table is
        # coherent across games -- unlike v1, where index 3 meant LEFT in Breakout
        # and RIGHT in Seaquest.
        self.action_embed = nn.Embedding(num_actions, embed_dim)

        # Shallower than the encoder, to force the encoder to do the representational work.
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim * 4,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(predictor_layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        # Predictor output must land in the same space as the target encoder output.
        self.simnorm = SimNorm(simnorm_V) if use_simnorm else nn.Identity()

    def forward(self, context_tokens, action, e):
        a_token = self.action_embed(action).unsqueeze(1)               # [B, 1, D]
        x = torch.cat([e.unsqueeze(1), a_token, context_tokens], 1)    # [B, 38, D]
        x = self.blocks(x)
        x = self.norm(x)
        return self.simnorm(x[:, 2:, :])                               # -> [B, 36, D]


class PaperAccurateJEPA(nn.Module):
    """Joint-embedding architecture with TD-MPC2-style environment conditioning.

    Manages the asymmetric gradient flow (trainable context encoder + predictor vs.
    EMA target encoder) that keeps the joint-embedding objective from collapsing.
    """

    def __init__(
        self,
        num_actions=games.NUM_CANONICAL_ACTIONS,
        num_env_slots=games.NUM_ENV_SLOTS,
        embed_dim=256,
        tau=0.996,
        use_simnorm=True,
        simnorm_V=8,
        enc_depth=6,
        pred_depth=3,
        heads=8,
        env_dropout_p=0.25,
    ):
        super().__init__()
        self.tau = tau
        self.embed_dim = embed_dim
        self.num_actions = num_actions
        self.num_env_slots = num_env_slots
        self.env_dropout_p = env_dropout_p
        self.unknown_id = games.UNKNOWN_ID

        # --- ENVIRONMENT EMBEDDING (TD-MPC2 task embedding) ---
        # 16 slots: 11 training games, UNKNOWN, 4 pre-reserved held-out games.
        # Reserving the held-out rows up front means few-shot adaptation needs no
        # checkpoint surgery -- the state_dict shape is identical before and after.
        # max_norm=1 matches TD-MPC2's ||e||_2 <= 1 constraint, which keeps the
        # embedding geometry interpretable (needed for the t-SNE in the diagnostics).
        self.env_embed = nn.Embedding(num_env_slots, embed_dim, max_norm=1.0)
        # nn.Embedding's default N(0,1) init gives a norm of ~sqrt(256)=16 per row, and
        # max_norm only renormalises rows as they are LOOKED UP -- so an untouched row
        # (UNKNOWN early on, and the held-out rows throughout) would sit at norm 16
        # while trained rows sit at 1, and its first lookup would be a 16x discontinuity.
        # Start every slot on the unit sphere instead: random directions, equal scale.
        with torch.no_grad():
            self.env_embed.weight.normal_(0, 1)
            self.env_embed.weight.div_(self.env_embed.weight.norm(dim=1, keepdim=True))

        # --- TRAINABLE NETWORKS ---
        self.context_encoder = TransformerEncoder(
            embed_dim=embed_dim, depth=enc_depth, heads=heads,
            use_simnorm=use_simnorm, simnorm_V=simnorm_V,
        )
        self.predictor = TransformerPredictor(
            num_actions=num_actions, embed_dim=embed_dim, depth=pred_depth, heads=heads,
            use_simnorm=use_simnorm, simnorm_V=simnorm_V,
        )

        # --- TARGET NETWORK (no gradients) ---
        # Deepcopied AFTER env_embed is defined on the parent, so the copy contains
        # no embedding table of its own.
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Inverse dynamics: [z_pooled, z_next_pooled, env] -> action logits.
        self.inverse_head = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.GELU(),
            nn.Linear(256, num_actions),
        )

        # Per-env valid-action mask for TD-MPC2-style action masking.
        # persistent=False: regenerated from games.py at construction, so a registry
        # change can never silently conflict with an old checkpoint.
        self.register_buffer(
            "valid_action_mask",
            torch.from_numpy(games.valid_action_mask()),
            persistent=False,
        )
        self.register_buffer(
            "action_entropy_floor",
            torch.from_numpy(games.action_entropy_floor()),
            persistent=False,
        )

    # ------------------------------------------------------------------ env ids

    def resolve_env_ids(self, g_t, mode="auto"):
        """Map true game ids to the ids actually fed to the network.

        mode="auto"  apply env dropout while training, pass through at eval
        mode="true"  always the real game id
        mode="unk"   always UNKNOWN -- the zero-shot / unconditioned setting
        mode=<int>   force a specific env slot (used by few-shot adaptation)
        """
        if mode == "unk":
            return torch.full_like(g_t, self.unknown_id)
        if isinstance(mode, int):
            return torch.full_like(g_t, mode)
        if mode == "true":
            return g_t
        if mode != "auto":
            raise ValueError(f"unknown env mode {mode!r}")

        if not self.training or self.env_dropout_p <= 0.0:
            return g_t
        drop = torch.rand_like(g_t, dtype=torch.float32) < self.env_dropout_p
        return torch.where(drop, torch.full_like(g_t, self.unknown_id), g_t)

    @torch.no_grad()
    def update_target_network(self):
        """phi <- tau * phi + (1 - tau) * theta over the encoder only.

        env_embed is deliberately NOT in this loop: it is shared, and both the context
        and target encoders consume the same vector on any given step.
        """
        for p_ctx, p_tgt in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            p_tgt.data.mul_(self.tau).add_(p_ctx.data, alpha=1.0 - self.tau)

    # ------------------------------------------------------------------ forward

    def forward(self, batch, return_latents=False, env_ids=None):
        """batch: s_t, a_t (canonical), s_next, mask, g_t.

        env_ids overrides the conditioning ids; when None the true g_t is used.
        The training loop passes resolve_env_ids(...) so that dropout is visible to
        logging, and so the same ids reach the encoder, target encoder and predictor.
        """
        s_t, a_t, s_next = batch["s_t"], batch["a_t"], batch["s_next"]
        mask = batch.get("mask", None)
        if env_ids is None:
            env_ids = batch["g_t"]

        # One lookup, reused everywhere. nn.Embedding with max_norm renormalises the
        # looked-up rows in place, so repeated lookups in a single step are wasteful
        # and (with autograd) needlessly fragile.
        e = self.env_embed(env_ids)  # [B, D]

        z_t = self.context_encoder(s_t, e)
        with torch.no_grad():
            # Same env vector as the context branch: the target must be encoded under
            # the same conditioning, or the prediction task changes meaning under dropout.
            z_next_target = self.target_encoder(s_next, e.detach())
        z_next_pred = self.predictor(z_t, a_t, e)

        if return_latents:
            return z_next_pred, z_next_target, z_t, e

        loss_per_batch = F.mse_loss(z_next_pred, z_next_target, reduction="none").mean(dim=[1, 2])
        if mask is not None:
            return (loss_per_batch * mask).sum() / mask.sum().clamp(min=1.0)
        return loss_per_batch.mean()

    def inverse_logits(self, z_current, z_next, e, env_ids):
        """Action logits, masked to the acting environment's valid actions.

        Masking uses the EFFECTIVE env id, not the true one, so a sample whose env was
        dropped to UNKNOWN faces the full 18-way problem -- the mask is part of the
        conditioning, and must not leak identity the encoder was denied (TD-MPC2's
        action masking for heterogeneous action spaces).
        """
        z_c_pooled = z_current.mean(dim=1)  # [B, D]
        z_n_pooled = z_next.mean(dim=1)     # [B, D]
        combined = torch.cat([z_c_pooled, z_n_pooled, e], dim=-1)  # [B, 3D]
        logits = self.inverse_head(combined)

        valid = self.valid_action_mask[env_ids]  # [B, 18] bool
        # finfo.min rather than -inf: -inf survives an fp16 autocast as NaN under some
        # softmax paths, and every row has at least one valid entry so this is safe.
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def compute_inverse_loss(self, z_current, z_next, true_actions, e, env_ids):
        """Inverse dynamics: which action carried z_current to z_next?

        Returns per-sample cross-entropy, shape [B].

        Known asymmetry, inherited from v1 and unchanged here: z_next comes from the
        no-grad target branch, so this term shapes the context encoder only -- it
        exerts no gradient on the predictor or on action_embed (finding G1).
        """
        logits = self.inverse_logits(z_current, z_next, e, env_ids)
        return F.cross_entropy(logits, true_actions, reduction="none")

    def inverse_correct(self, z_current, z_next, true_actions, e, env_ids):
        """Per-sample top-1 correctness of the inverse head, shape [B] float.

        Accuracy is far easier to read than a cross-entropy in nats, but on its own it
        is misleading across games: Breakout has 4 valid actions and Seaquest 18, so
        chance differs per sample. Pair it with chance_action_prob below.
        """
        logits = self.inverse_logits(z_current, z_next, e, env_ids)
        return (logits.argmax(dim=-1) == true_actions).float()

    def chance_action_prob(self, env_ids):
        """1/|A_g| per sample -- the masked-chance accuracy for inverse_correct."""
        return 1.0 / self.valid_action_mask[env_ids].sum(dim=-1).float()

    def chance_inverse_loss(self, env_ids):
        """log|A_g| per sample -- the masked-chance floor the inverse head must beat.

        Averaged over a batch this is the H(a|game) baseline from finding D3. Reporting
        (chance - L_inv) as information gain prevents the inverse head from taking
        credit for what is really just game recognition.
        """
        return self.action_entropy_floor[env_ids]


if __name__ == "__main__":
    # Verifications 2 and 4 from the plan.
    torch.manual_seed(0)
    print("Initializing environment-conditioned V-JEPA...\n")

    B, V = 16, 8
    model = PaperAccurateJEPA(embed_dim=256, tau=0.995, use_simnorm=True, simnorm_V=V)

    batch = {
        "s_t": torch.randint(0, 256, (B, 4, 84, 84), dtype=torch.uint8),
        "a_t": torch.randint(0, 18, (B,)),
        "s_next": torch.randint(0, 256, (B, 4, 84, 84), dtype=torch.uint8),
        "mask": torch.ones(B),
        "g_t": torch.randint(0, games.NUM_TRAIN_GAMES, (B,)),
    }
    # Keep every sampled action inside its game's valid set, as the dataset guarantees.
    vm = model.valid_action_mask
    batch["a_t"] = torch.tensor([int(torch.nonzero(vm[g]).flatten()[0]) for g in batch["g_t"]])

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.2f}M total, {n_train/1e6:.2f}M trainable")

    # --- structural: exactly one env embedding table, none inside the EMA copy ---
    emb = [n for n, m in model.named_modules() if isinstance(m, nn.Embedding)]
    assert emb == ["env_embed", "predictor.action_embed"], emb
    assert not any(isinstance(m, nn.Embedding) for m in model.target_encoder.modules()), \
        "target_encoder must not carry its own embedding table"
    print(f"  [ok] embedding tables: {emb}; target_encoder carries none")

    # --- forward shapes ---
    model.train()
    env_ids = model.resolve_env_ids(batch["g_t"], mode="true")
    z_pred, z_tgt, z_ctx, e = model(batch, return_latents=True, env_ids=env_ids)
    assert z_pred.shape == (B, 36, 256), z_pred.shape
    assert z_tgt.shape == (B, 36, 256) and z_ctx.shape == (B, 36, 256)
    assert e.shape == (B, 256)
    print(f"  [ok] forward shapes: z_pred/z_target/z_context {tuple(z_pred.shape)}, e {tuple(e.shape)}")

    # --- SimNorm: every group of V is a probability simplex ---
    for name, z in [("z_context", z_ctx), ("z_target", z_tgt), ("z_pred", z_pred)]:
        g = z.view(B, 36, 256 // V, V)
        assert torch.allclose(g.sum(-1), torch.ones_like(g.sum(-1)), atol=1e-5), f"{name} not on simplex"
        assert (z >= 0).all()
    print(f"  [ok] SimNorm: all three latents lie on {256//V} simplices of dim {V}, entries >= 0")
    print(f"       mean entry {z_ctx.mean():.4f} (= 1/V = {1/V:.4f}), max {z_ctx.max():.4f}")

    # --- env embedding norm constraint ---
    norms = model.env_embed.weight.norm(dim=1)
    assert norms.max() <= 1.0 + 1e-5, norms.max()
    print(f"  [ok] ||e||_2 <= 1 for all {model.num_env_slots} env slots (max {norms.max():.4f})")

    # --- env dropout actually fires, and only toward UNKNOWN ---
    ids = model.resolve_env_ids(batch["g_t"].repeat(64), mode="auto")
    frac = (ids == games.UNKNOWN_ID).float().mean().item()
    assert 0.15 < frac < 0.35, f"dropout rate {frac} far from {model.env_dropout_p}"
    assert model.resolve_env_ids(batch["g_t"], mode="unk").eq(games.UNKNOWN_ID).all()
    model.eval()
    assert model.resolve_env_ids(batch["g_t"], mode="auto").equal(batch["g_t"]), "no dropout at eval"
    model.train()
    print(f"  [ok] env dropout fires at {frac:.3f} in train mode, disabled at eval, 'unk' forces UNKNOWN")

    # --- inverse loss + masking ---
    inv = model.compute_inverse_loss(z_ctx, z_tgt, batch["a_t"], e, env_ids)
    chance = model.chance_inverse_loss(env_ids)
    assert inv.shape == (B,) and torch.isfinite(inv).all()
    print(f"  [ok] inverse loss {tuple(inv.shape)} finite, mean {inv.mean():.3f} vs chance {chance.mean():.3f} nats")

    unk_chance = model.chance_inverse_loss(torch.full((B,), games.UNKNOWN_ID))
    assert torch.allclose(unk_chance, torch.full((B,), float(torch.tensor(18.0).log())))
    print(f"  [ok] UNKNOWN chance floor is log(18) = {unk_chance[0]:.3f} nats")

    # --- Verification 4: gradient isolation ---
    loss = F.mse_loss(z_pred, z_tgt) + 0.01 * inv.mean()
    loss.backward()

    assert all(p.grad is None for p in model.target_encoder.parameters()), \
        "target_encoder received gradient -- the EMA branch has leaked"
    print("  [ok] target_encoder: no parameter received a gradient")

    grad_rows = torch.nonzero(model.env_embed.weight.grad.abs().sum(1)).flatten().tolist()
    used = sorted(set(env_ids.tolist()))
    assert set(grad_rows).issubset(set(used)), f"gradient on unused env rows: {set(grad_rows) - set(used)}"
    heldout = [r for r in grad_rows if r > games.UNKNOWN_ID]
    assert not heldout, f"held-out rows must stay untouched during pretraining: {heldout}"
    print(f"  [ok] env_embed gradient confined to rows present in batch: {grad_rows}")
    print(f"       (held-out rows {games.UNKNOWN_ID+1}..{games.NUM_ENV_SLOTS-1} untouched)")

    # --- fallback path (no return_latents) still works ---
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        scalar = model(batch)
    print(f"  [ok] scalar forward path: masked JEPA loss = {scalar.item():.6f}")

    print("\nVerifications 2 and 4 PASSED")
