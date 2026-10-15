"""SB3 feature extractors around the encoders in ../models/ and ../jepa/."""

import os
import sys

import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JEPA_DIR = os.path.join(REPO_ROOT, "jepa")

# keep the head the same for every arch. if we don't set this SB3 uses [] for its
# own CNN and [64,64] for a custom extractor, which isn't a fair comparison.
NET_ARCH = [512]


# ---------------------------------------------------------------------------
# encode fns: model -> flat features, one per bottleneck type.
# obs are already scaled to [0,1] by SB3 so don't divide by 255 again here!
# ---------------------------------------------------------------------------

def _encode_vae(model, x):
    # mu only, sampling here would add noise to the rollouts
    return model.fc_mu(model.encoder(x).flatten(1))


def _encode_vq(model, x):
    """encoder -> VQ (EmaVqVae, AtariResidualVQVAE)"""
    z, _ = model.quantizer(model.encoder(x))
    return z.flatten(1)


def _encode_fsq(model, x):
    """encoder -> proj -> FSQ (AtariFSQVAE)"""
    z, _ = model.fsq(model.pre_quant_proj(model.encoder(x)))
    return z.flatten(1)


def _encode_vit_vq(model, x):
    """ViT encoder -> 1x1 conv -> VQ (AdvancedViTVQVAE)"""
    z, _ = model.quantizer(model.pre_quant_conv(model.encoder(x)))
    return z.flatten(1)


def _builder(module_path, class_name, **kwargs):
    # lazy import, we only want to pay for the arch we actually use
    def build():
        sys.path.insert(0, REPO_ROOT)
        mod = __import__(module_path, fromlist=[class_name])
        return getattr(mod, class_name)(in_channels=4, **kwargs)
    return build


# arch -> (builder, encode fn)
ARCHS = {
    "vae":       (_builder("models.standard_vae", "StandardVAE"), _encode_vae),
    "ema_vqvae": (_builder("models.ema_vqvae", "EmaVqVae",
                           num_embeddings=512, embedding_dim=64), _encode_vq),
    "vqvae":     (_builder("models.residual_vqvae", "AtariResidualVQVAE",
                           num_embeddings=512, embedding_dim=64), _encode_vq),
    "fsq":       (_builder("models.fsq_vae_decoder", "AtariFSQVAE",
                           hidden_dim=128, fsq_levels=[8, 5, 5, 3]), _encode_fsq),
    "vit_vqvae": (_builder("models.vit_vqae", "AdvancedViTVQVAE",
                           num_embeddings=512, embed_dim=256), _encode_vit_vq),
}

# nature = SB3's built in NatureCNN, jepa is seperate because it also needs to know
# which game it's looking at.
ARCH_NAMES = ["nature", "jepa", *sorted(ARCHS)]


# ---------------------------------------------------------------------------
# extractors
# ---------------------------------------------------------------------------

def _load_weights(model, checkpoint):
    sd = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}   # torch.compile prefix
    report = model.load_state_dict(sd, strict=False)
    if report.missing_keys or report.unexpected_keys:
        print(f"  load: {len(report.missing_keys)} missing, "
              f"{len(report.unexpected_keys)} unexpected keys")


class _FreezableExtractor(BaseFeaturesExtractor):
    # requires_grad=False on its own isn't enough. SB3 calls set_training_mode(True)
    # every update, so dropout + the EMA codebook would still change. pin eval().

    def _set_frozen(self, frozen):
        self.frozen = frozen
        if frozen:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def train(self, mode=True):
        return super().train(False if self.frozen else mode)


class PretrainedExtractor(_FreezableExtractor):
    """any pretrained encoder from ../../models/, we throw the decoder away."""

    def __init__(self, observation_space, arch, checkpoint=None, freeze=True):
        build, encode = ARCHS[arch]
        backbone = build()
        if checkpoint:
            _load_weights(backbone, checkpoint)

        # eval() first - EmaVqVae updates its codebook in train mode and a 1 sample
        # dummy pass would mess it up
        backbone.eval()
        with torch.no_grad():
            dim = encode(backbone, torch.zeros(1, *observation_space.shape)).shape[1]
        super().__init__(observation_space, dim)

        self.backbone = backbone
        self.encode = encode
        self._set_frozen(freeze)
        print(f"{arch} extractor | features_dim {dim} | "
              f"{'frozen' if freeze else 'joint'}")

    def forward(self, obs):
        return self.encode(self.backbone, obs)


class JepaExtractor(_FreezableExtractor):
    """V-JEPA context encoder from ../jepa (game conditioned).

    env_mode='unknown' picks the UNKNOWN slot instead of the games own embedding,
    need that for games the world model never trained on.
    """

    def __init__(self, observation_space, checkpoint, game, env_mode="true", freeze=True):
        sys.path.insert(0, JEPA_DIR)
        from JEPA import PaperAccurateJEPA, strip_compile_prefix
        import games as gamelib

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt["model_config"]
        super().__init__(observation_space, 36 * cfg["embed_dim"])  # 6x6 = 36 tokens

        model = PaperAccurateJEPA(**cfg)
        model.load_state_dict(strip_compile_prefix(ckpt["model_state_dict"]))
        self.encoder = model.context_encoder
        self.env_embed = model.env_embed

        slot = gamelib.UNKNOWN_ID if env_mode == "unknown" else gamelib.GAME_TO_ID[game]
        self.register_buffer("env_slot", torch.tensor(slot))   # buffer so .to(device) works

        self._set_frozen(freeze)
        print(f"jepa extractor | env slot {slot} | features_dim {self.features_dim} | "
              f"{'frozen' if freeze else 'joint'}")

    def forward(self, obs):
        e = self.env_embed(self.env_slot.expand(obs.shape[0]))
        return self.encoder(obs, e).flatten(1)


def build_policy_kwargs(arch, checkpoint=None, freeze=True, game=None,
                        env_mode="true", net_arch=NET_ARCH):
    """policy_kwargs for PPO, works for any arch in ARCH_NAMES."""
    kwargs = dict(net_arch=list(net_arch))

    if arch == "nature":
        return kwargs   # CnnPolicy is already NatureCNN, nothing pretrained to load

    if arch == "jepa":
        kwargs.update(
            features_extractor_class=JepaExtractor,
            features_extractor_kwargs=dict(
                checkpoint=os.path.abspath(checkpoint), game=game,
                env_mode=env_mode, freeze=freeze),
        )
    else:
        kwargs.update(
            features_extractor_class=PretrainedExtractor,
            features_extractor_kwargs=dict(
                arch=arch,
                checkpoint=os.path.abspath(checkpoint) if checkpoint else None,
                freeze=freeze),
        )
    return kwargs
