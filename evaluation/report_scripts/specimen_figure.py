"""Games figure + table from the Data Specimen Sheet (docs/analysis/Atari JEPA — Data Specimen Sheet.html).

The sheet embeds, for each of the 15 games, the four channels of one observation stack,
the next frame and |frame(t+1) - frame(t)|, read from custom_datasets_small (50k transitions
per game), together with per-game statistics. This script extracts frame t and |delta| for
every game into figures/games_grid.pdf and writes tables/games_rows.tex from the sheet's
statistics plus the moving-pixel share measured by games_figure.py.

Run from the repository root:  python evaluation/report_scripts/specimen_figure.py
"""
import base64
import io
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHEET = os.path.join(ROOT, "docs", "analysis", "Atari JEPA — Data Specimen Sheet.html")
GAMES = ["BeamRider", "Breakout", "DemonAttack", "Enduro", "MsPacman", "Pong", "Qbert",
         "Riverraid", "RoadRunner", "Seaquest", "SpaceInvaders",
         "Alien", "Asteroids", "Atlantis", "IceHockey"]
HELD = {"Alien", "Asteroids", "Atlantis", "IceHockey"}

s = open(SHEET, encoding="utf-8").read()
imgs = [np.array(Image.open(io.BytesIO(base64.b64decode(m.group(1)))))
        for m in re.finditer(r'<img[^>]*src="data:image/\w+;base64,([^"]+)"', s)]
assert len(imgs) == 8 + 6 * len(GAMES), len(imgs)
# after the 8-image sample, each game has t-3, t-2, t-1, t, t+1, |delta|
frame_t = {g: imgs[8 + 6 * i + 3] for i, g in enumerate(GAMES)}
delta = {g: imgs[8 + 6 * i + 5] for i, g in enumerate(GAMES)}

plt.rcParams.update({"font.family": "serif", "font.size": 7})
fig, axes = plt.subplots(3, 10, figsize=(7.2, 2.55),
                         gridspec_kw=dict(wspace=0.04, hspace=0.32))
for k, g in enumerate(GAMES):
    r, c = divmod(k, 5)
    a1, a2 = axes[r, 2 * c], axes[r, 2 * c + 1]
    a1.imshow(frame_t[g], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    d = delta[g].astype(float)
    a2.imshow(d, cmap="gray_r", vmin=0, vmax=max(1.0, d.max()), interpolation="nearest")
    a1.set_title(g + ("$^\\dagger$" if g in HELD else ""), fontsize=6.8, pad=2, loc="left")
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#bbbbbb"); sp.set_linewidth(0.4)
axes[2, 8].axis("off"); axes[2, 9].axis("off")
fig.savefig(os.path.join(ROOT, "evaluation", "outputs", "figures", "games_grid.pdf"), bbox_inches="tight")

# ------------------------------------------------------------------ table
# statistics as printed in section 04 of the sheet (custom_datasets_small, 50k per game)
t = re.sub(r"<[^>]+>", "\n", s[s.find("How different the games are"):s.find("What comes out the other side")])
cells = [c.strip() for c in t.split("\n") if c.strip()]
start = cells.index("Most common action") + 1
rows = {}
for i in range(start, start + 8 * len(GAMES), 8):
    g, split, nact, pmean, black, term, ep, top = cells[i:i + 8]
    rows[g] = dict(nact=int(nact), pmean=float(pmean), black=black.rstrip("%"),
                   ep=ep.replace(",", "\\,"), top=re.search(r"\((\d+)%\)", top).group(1))

KIND = {"Pong": "paddle vs.\\ paddle", "Breakout": "paddle, ball, bricks",
        "SpaceInvaders": "fixed shooter", "DemonAttack": "fixed shooter",
        "BeamRider": "pseudo-3D shooter", "Enduro": "pseudo-3D racing",
        "MsPacman": "maze", "Qbert": "isometric platform",
        "Riverraid": "vertical scroller", "Seaquest": "underwater shooter",
        "RoadRunner": "horizontal scroller", "Alien": "maze",
        "Asteroids": "free-roaming shooter", "Atlantis": "fixed defence",
        "IceHockey": "team sport"}
EXPERT = {"DemonAttack": "DQN", "RoadRunner": "A2C", "Alien": "DQN", "Asteroids": "DQN",
          "Atlantis": "DQN", "IceHockey": "DQN"}
MOVING = {"Pong": 1.7, "Breakout": 1.4, "SpaceInvaders": 2.8, "DemonAttack": 3.7,
          "BeamRider": 33.5, "Enduro": 6.9, "MsPacman": 3.5, "Qbert": 2.2, "Riverraid": 10.3,
          "Seaquest": 3.8, "RoadRunner": 7.4, "Alien": 3.2, "Asteroids": 6.0,
          "Atlantis": 6.1, "IceHockey": 1.1}          # from games_figure.py (2000 stacks/game)

lines = []
for g in GAMES:
    r = rows[g]
    tag = "$^\\dagger$" if g in HELD else ""
    lines.append(f"{g}{tag} & {KIND[g]} & {r['nact']} & {EXPERT.get(g, 'PPO')} & "
                 f"{r['pmean']:.0f} & {r['black']} & {MOVING[g]:.1f} & {r['ep']} & {r['top']} \\\\")
    if g == "SpaceInvaders":
        lines.append(r"\midrule")
open(os.path.join(ROOT, "evaluation", "outputs", "tables", "games_rows.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
