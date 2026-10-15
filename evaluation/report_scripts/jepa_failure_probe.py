"""Why does the V-JEPA encoder fail as a PPO feature? Two measurements.

1. Linear action probe. For each seen game, fit a linear classifier from a frozen
   feature of s_t to the expert's action a_t (expert validation transitions,
   first 2000 for training, next 1000 for testing, i.e. a temporal split).
   If the policy-relevant information is linearly readable, a PPO head on top of
   the frozen feature can in principle find it.
   Features: downsampled pixels, a randomly initialised NatureCNN trunk, the
   pre-trained FSQ-VAE grid, and the V-JEPA context tokens (flattened and pooled).

2. State-distribution shift. The encoders were pre-trained on expert play, but a
   PPO agent starts from a random policy. For each seen game, 2000 random-policy
   transitions are generated with the same wrappers as the expert data, and the
   V-JEPA one-step latent loss and the FSQ-VAE reconstruction MSE are compared
   with the expert validation transitions.

Run from the repository root (CPU is fine, ~15 min):
    python evaluation/report_scripts/jepa_failure_probe.py
Writes evaluation/outputs/results/jepa_failure_probe.json and evaluation/outputs/tables/probe_rows.tex.
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
JDIR = os.path.join(ROOT, "jepa")
sys.path.insert(0, ROOT)
sys.path.insert(0, JDIR)

from JEPA import PaperAccurateJEPA, strip_compile_prefix  # noqa: E402
import games as gamelib  # noqa: E402
from models.fsq_vae_decoder import AtariFSQVAE  # noqa: E402

torch.set_grad_enabled(False)
torch.manual_seed(0)
VAL = os.path.join(ROOT, "artifacts", "data", "jepa", "custom_datasets_small", "val")
N_TRAIN, N_TEST, N_RANDOM = 2000, 1000, 2000
BS = 250


# --------------------------------------------------------------------------- encoders
ck = torch.load(os.path.join(JDIR, "checkpoints", "main", "vjepa_v2_best.pt"),
                map_location="cpu", weights_only=False)
jepa = PaperAccurateJEPA(**ck["model_config"])
jepa.load_state_dict(strip_compile_prefix(ck["model_state_dict"]))
jepa.eval()

fsq = AtariFSQVAE()
sd = torch.load(os.path.join(ROOT, "artifacts", "checkpoints", "autoencoders", "fsq_vae_decoder", "fsq_vae_epoch_125.pth"),
                map_location="cpu")
fsq.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
fsq.eval()

nature = nn.Sequential(  # NatureCNN conv trunk, random init (seeded)
    nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
    nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten()).eval()


def batches(x):
    for i in range(0, len(x), BS):
        yield torch.from_numpy(x[i:i + BS])


def to_chw(x):
    return np.ascontiguousarray(np.transpose(x, (0, 3, 1, 2)))


def feats(obs, game):
    """obs uint8 [N,4,84,84] -> dict of feature matrices."""
    gid = gamelib.GAME_TO_ID[game]
    out = {k: [] for k in ["pixels", "nature_random", "fsq", "jepa_flat", "jepa_pool"]}
    for xb in batches(obs):
        xf = xb.float() / 255.0
        out["pixels"].append(nn.functional.avg_pool2d(xf, 4).flatten(1))
        out["nature_random"].append(nature(xf))
        z = fsq.fsq(fsq.pre_quant_proj(fsq.encoder(xf)))[0]
        out["fsq"].append(z.flatten(1))
        e = jepa.env_embed(torch.full((len(xb),), gid))
        zt = jepa.context_encoder(xb, e)
        out["jepa_flat"].append(zt.flatten(1))
        out["jepa_pool"].append(zt.mean(1))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def probe(xtr, ytr, xte, yte):
    k = min(256, xtr.shape[1])
    clf = make_pipeline(StandardScaler(), PCA(k, random_state=0),
                        LogisticRegression(max_iter=3000, C=1.0))
    clf.fit(xtr, ytr)
    return float((clf.predict(xte) == yte).mean())


def jepa_losses(obs, act, nxt, term, game):
    """per-transition one-step latent MSE of the predictor and of the identity baseline."""
    gid = gamelib.GAME_TO_ID[game]
    lut = torch.from_numpy(gamelib.action_lut(game)).long()
    pl, il = [], []
    for i in range(0, len(obs), BS):
        s, s2 = torch.from_numpy(obs[i:i + BS]), torch.from_numpy(nxt[i:i + BS])
        a = lut[torch.from_numpy(act[i:i + BS]).long()]
        e = jepa.env_embed(torch.full((len(s),), gid))
        zt = jepa.context_encoder(s, e)
        tgt = jepa.target_encoder(s2, e)
        pred = jepa.predictor(zt, a, e)
        pl.append(((pred - tgt) ** 2).mean((1, 2)))
        il.append(((zt - tgt) ** 2).mean((1, 2)))
    keep = ~torch.from_numpy(term)
    return float(torch.cat(pl)[keep].mean()), float(torch.cat(il)[keep].mean())


def fsq_mse(obs):
    tot = []
    for xb in batches(obs):
        xf = xb.float() / 255.0
        tot.append(((fsq(xf)[0] - xf) ** 2).mean((1, 2, 3)))
    return float(torch.cat(tot).mean())


def random_transitions(game, n, seed=123):
    """random-policy transitions with the same wrappers as the expert collectors."""
    import ale_py
    import gymnasium as gym
    from stable_baselines3.common.env_util import make_atari_env
    from stable_baselines3.common.vec_env import VecFrameStack
    gym.register_envs(ale_py)
    env = VecFrameStack(make_atari_env(f"ALE/{game}-v5", n_envs=1, seed=seed,
                                       env_kwargs={"frameskip": 1, "repeat_action_probability": 0.0}),
                        n_stack=4)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    O, A, N, T = [], [], [], []
    for _ in range(n):
        a = rng.integers(env.action_space.n, size=1)
        nxt, _, done, info = env.step(a)
        N.append(info[0]["terminal_observation"] if done[0] else nxt[0])
        O.append(obs[0]); A.append(a[0]); T.append(bool(done[0]))
        obs = nxt
    env.close()
    return (to_chw(np.array(O, np.uint8)), np.array(A), to_chw(np.array(N, np.uint8)), np.array(T))


results = {}
for game in gamelib.TRAIN_GAMES:
    d = np.load(os.path.join(VAL, f"{game}NoFrameskip-v4_expert_10000_transitions.npz"))
    n = N_TRAIN + N_TEST
    obs, act = to_chw(d["obs"][:n]), d["actions"][:n].astype(int)
    nxt, term = to_chw(d["next_obs"][:n]), d["terminals"][:n]
    F = feats(obs, game)
    r = {"majority": float((act[N_TRAIN:] == np.bincount(act[:N_TRAIN]).argmax()).mean()),
         "n_actions": int(len(gamelib.action_lut(game)))}
    for k, x in F.items():
        r[f"probe_{k}"] = probe(x[:N_TRAIN], act[:N_TRAIN], x[N_TRAIN:], act[N_TRAIN:])
    # distribution shift
    r["jepa_pred_expert"], r["jepa_ident_expert"] = jepa_losses(obs[:N_RANDOM], act[:N_RANDOM],
                                                                nxt[:N_RANDOM], term[:N_RANDOM], game)
    r["fsq_mse_expert"] = fsq_mse(obs[:N_RANDOM])
    ro, ra, rn, rt = random_transitions(game, N_RANDOM)
    r["jepa_pred_random"], r["jepa_ident_random"] = jepa_losses(ro, ra, rn, rt, game)
    r["fsq_mse_random"] = fsq_mse(ro)
    results[game] = r
    print(game, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)

os.makedirs(os.path.join(ROOT, "evaluation", "outputs", "results"), exist_ok=True)
json.dump(results, open(os.path.join(ROOT, "evaluation", "outputs", "results", "jepa_failure_probe.json"), "w"), indent=1)

# summary
keys = ["majority", "probe_pixels", "probe_nature_random", "probe_fsq", "probe_jepa_flat", "probe_jepa_pool"]
mean = {k: np.mean([results[g][k] for g in results]) for k in keys}
print("mean probe accuracy:", {k: round(v, 3) for k, v in mean.items()})
for k in ["jepa_pred", "jepa_ident", "fsq_mse"]:
    ratio = [results[g][f"{k}_random"] / results[g][f"{k}_expert"] for g in results]
    print(f"{k}: random/expert ratio median {np.median(ratio):.2f} (range {min(ratio):.2f}-{max(ratio):.2f})")
