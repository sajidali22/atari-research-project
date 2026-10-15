"""Games section: one frame per game + measured per-game statistics.

Run from the repository root:
    python evaluation/report_scripts/games_figure.py
Writes evaluation/outputs/figures/games_grid.pdf and evaluation/outputs/tables/games_rows.tex.

Statistics per game:
  |A|        size of the game's minimal action set (ale-py registry, jepa/games.py)
  moving %   share of pixels that change within a frame stack (motion mask, delta = 0.05),
             averaged over 2000 random stacks of the expert frame corpus
  steps/term mean number of agent steps between two terminals in the transition corpus
             (the SB3 Atari wrapper also ends an episode on a life loss)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "jepa"))
import games as gamelib  # noqa: E402

TRAIN = ["Pong", "Breakout", "SpaceInvaders", "DemonAttack", "BeamRider", "Enduro",
         "MsPacman", "Qbert", "Riverraid", "Seaquest", "RoadRunner"]
HELD = ["Alien", "Asteroids", "Atlantis", "IceHockey"]
EXPERT = {"DemonAttack": "DQN", "RoadRunner": "A2C", "Alien": "DQN", "Asteroids": "DQN",
          "Atlantis": "DQN", "IceHockey": "DQN"}           # all others: PPO
KIND = {"Pong": "paddle vs.\\ paddle, one ball", "Breakout": "paddle, ball, brick wall",
        "SpaceInvaders": "shooter, grid of invaders", "DemonAttack": "shooter, flying demons",
        "BeamRider": "shooter, pseudo-3D beams", "Enduro": "racing, pseudo-3D road",
        "MsPacman": "maze, pellets, ghosts", "Qbert": "isometric pyramid",
        "Riverraid": "vertical scroller, fuel", "Seaquest": "underwater shooter, oxygen",
        "RoadRunner": "horizontal scroller, chase", "Alien": "maze, aliens",
        "Asteroids": "free-moving ship, flicker", "Atlantis": "static base defence",
        "IceHockey": "team sport, two players"}
SHOW_IDX = 25_000


def frames(game):
    split = "train" if game in TRAIN else "test"
    return np.load(os.path.join(ROOT, "artifacts", "data", "expert", split,
                                f"{game}NoFrameskip-v4_expert_50000_frames.npz"))["frames"]


def moving_share(fr, n=2000, seed=0):
    idx = np.random.default_rng(seed).choice(len(fr), n, replace=False)
    x = fr[idx].astype(np.float32) / 255.0               # (n, 84, 84, 4) HWC
    diff = np.abs(np.diff(x, axis=-1)).max(-1)
    return float((diff > 0.05).mean())


def steps_per_terminal(game):
    split = "val" if game in TRAIN else "test"
    n = 10_000 if game in TRAIN else 50_000
    t = np.load(os.path.join(ROOT, "artifacts", "data", "jepa", "custom_datasets_small", split,
                             f"{game}NoFrameskip-v4_expert_{n}_transitions.npz"))["terminals"]
    return len(t) / max(1, int(t.sum()))


stats, shown = {}, {}
for g in TRAIN + HELD:
    fr = frames(g)
    shown[g] = fr[SHOW_IDX, :, :, 3].copy()
    stats[g] = dict(n_actions=len(gamelib.action_lut(g)), moving=moving_share(fr),
                    spt=steps_per_terminal(g))
    del fr
    print(g, stats[g], flush=True)

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.family": "serif", "font.size": 7})
fig, axes = plt.subplots(3, 5, figsize=(7.0, 4.55))
for ax, g in zip(axes.flat, TRAIN + HELD):
    ax.imshow(shown[g], cmap="gray", vmin=0, vmax=255)
    ax.set_title(g + ("$^\\dagger$" if g in HELD else ""), fontsize=7.5, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout(h_pad=0.4, w_pad=0.3)
fig.savefig(os.path.join(ROOT, "evaluation", "outputs", "figures", "games_grid.pdf"), bbox_inches="tight")

# ------------------------------------------------------------------ table rows
lines = []
for g in TRAIN + HELD:
    s = stats[g]
    tag = "$^\\dagger$" if g in HELD else ""
    lines.append(f"{g}{tag} & {KIND[g]} & {s['n_actions']} & {EXPERT.get(g, 'PPO')} & "
                 f"{100 * s['moving']:.1f} & {s['spt']:.0f} \\\\")
    if g == TRAIN[-1]:
        lines.append(r"\midrule")
open(os.path.join(ROOT, "evaluation", "outputs", "tables", "games_rows.tex"), "w").write("\n".join(lines) + "\n")
print("written")
