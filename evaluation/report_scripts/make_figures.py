"""Generate every figure and results table used in paper/main.tex.

Run from the repository root:
    python evaluation/report_scripts/make_figures.py

Inputs:
  artifacts/wandb_export/               W&B export (source of truth for every training run), made by
                                     export_wandb.py; PPO scores are summarised by wandb_results.py
                                     into evaluation/outputs/tables/wandb_ppo_runs.csv + wandb_export/ppo_curves.pkl
  evaluation/outputs/results/*.json|log          eval_jepa_diagnostics.py output (seen / held-out games)
  artifacts/checkpoints/autoencoders/...                autoencoder checkpoints for the reconstruction figure
  artifacts/data/expert/{train,test}        expert frame stacks

Order: export_wandb.py inventory / history ... -> wandb_results.py -> make_figures.py
"""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "evaluation", "outputs", "figures")
TAB = os.path.join(ROOT, "evaluation", "outputs", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
sys.path.insert(0, ROOT)

# validated categorical palette (dataviz reference slots 1-5, light mode)
C = {"nature": "#2a78d6", "fsq_frozen": "#eb6834", "fsq_joint": "#1baf7a",
     "jepa_frozen": "#eda100", "jepa_joint": "#e87ba4"}
LS = {"nature": "-", "fsq_frozen": "--", "fsq_joint": "-", "jepa_frozen": "--", "jepa_joint": "-"}
LABEL = {"nature": "NatureCNN (scratch)", "fsq_frozen": "FSQ-VAE frozen",
         "fsq_joint": "FSQ-VAE joint", "jepa_frozen": "V-JEPA frozen", "jepa_joint": "V-JEPA joint"}
METHODS = list(LABEL)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "axes.edgecolor": "#8a8a8a", "savefig.bbox": "tight", "savefig.dpi": 300,
})

TRAIN_GAMES = ["BeamRider", "Breakout", "DemonAttack", "Enduro", "MsPacman", "Pong",
               "Qbert", "Riverraid", "RoadRunner", "Seaquest", "SpaceInvaders"]
HELDOUT = ["Alien", "Asteroids", "Atlantis", "IceHockey"]

# Random / human reference scores (Badia et al., 2020, Agent57 appendix)
RANDOM = {"Alien": 227.8, "Asteroids": 719.1, "Atlantis": 12850.0, "BeamRider": 363.9,
          "Breakout": 1.7, "DemonAttack": 152.1, "Enduro": 0.0, "IceHockey": -11.2,
          "MsPacman": 307.3, "Pong": -20.7, "Qbert": 163.9, "Riverraid": 1338.5,
          "RoadRunner": 11.5, "Seaquest": 68.4, "SpaceInvaders": 148.0}
HUMAN = {"Alien": 7127.7, "Asteroids": 47388.7, "Atlantis": 29028.1, "BeamRider": 16926.5,
         "Breakout": 30.5, "DemonAttack": 1971.0, "Enduro": 860.5, "IceHockey": 0.9,
         "MsPacman": 6951.6, "Pong": 14.6, "Qbert": 13455.0, "Riverraid": 17118.0,
         "RoadRunner": 7845.0, "Seaquest": 42054.7, "SpaceInvaders": 1668.7}


def hns(game, score):
    return (score - RANDOM[game]) / (HUMAN[game] - RANDOM[game])


# --------------------------------------------------------------------------- data
# Everything below comes from the full (unsampled) W&B histories.
# Final score = mean training return over steps 9-10M: for the FSQ runs the mean of
# individual episode returns, for SB3 the mean of the logged 100-episode running mean.
EXP = os.path.join(ROOT, "artifacts", "wandb_export")
runs = pd.read_csv(os.path.join(TAB, "wandb_ppo_runs.csv"))
curves = pd.read_pickle(os.path.join(EXP, "ppo_curves.pkl"))
scores = {(r.method, r.game): r.final for r in runs.itertuples()}
sb3_meta = {(r.method, r.game): r._asdict() for r in runs.itertuples()}

# trainable parameters are not logged to W&B; mean over games from the SB3 job logs
# ("trainable X M of Y M" in artifacts/logs/ppo_jepa_slurm_oldextractor)
TRAINABLE = {"nature": 2.21, "jepa_frozen": 9.44, "jepa_joint": 14.40}

# --------------------------------------------------------------------------- tables
games_all = TRAIN_GAMES + HELDOUT
rows = []
for g in games_all:
    row = {"game": g}
    for k in METHODS:
        row[k] = scores.get((k, g), np.nan)
    rows.append(row)
tab = pd.DataFrame(rows).set_index("game")
tab.to_csv(os.path.join(TAB, "final_scores.csv"))

hn = tab.copy()
for g in hn.index:
    for k in METHODS:
        hn.loc[g, k] = hns(g, tab.loc[g, k])
hn.to_csv(os.path.join(TAB, "hns.csv"))


def fmt(v):
    if np.isnan(v):
        return "--"
    if abs(v) >= 10000:
        return f"{v/1000:.1f}k"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


# LaTeX table of raw scores, best per row in bold (only among methods with data)
lines = []
for g in games_all:
    vals = [tab.loc[g, k] for k in METHODS]
    best = np.nanmax(vals)
    cells = []
    for v in vals:
        s = fmt(v)
        if not np.isnan(v) and v == best:
            s = r"\textbf{" + s + "}"
        cells.append(s)
    tag = r"$^\dagger$" if g in HELDOUT else ""
    lines.append(f"{g}{tag} & " + " & ".join(cells) + r" \\")
    if g == TRAIN_GAMES[-1]:
        lines.append(r"\midrule")


def agg(frame, games):
    sub = frame.loc[games]
    return sub.median(), sub.mean(), (sub >= 1.0).sum()

med, mean, _ = agg(hn, TRAIN_GAMES)
lines.append(r"\midrule")
lines.append("Median HNS (11 seen) & " + " & ".join(f"{med[k]:.2f}" for k in METHODS) + r" \\")
lines.append("Mean HNS (11 seen) & " + " & ".join(f"{mean[k]:.2f}" for k in METHODS) + r" \\")
medh = hn.loc[HELDOUT, ["fsq_frozen", "fsq_joint"]].median()
lines.append("Median HNS (4 held-out) & -- & " + f"{medh['fsq_frozen']:.2f} & {medh['fsq_joint']:.2f}" + r" & -- & -- \\")
open(os.path.join(TAB, "scores_rows.tex"), "w").write("\n".join(lines) + "\n")

# SB3 test-set table (30 held-out-seed episodes) + throughput
lines = []
for g in TRAIN_GAMES:
    cells = []
    for k in ["nature", "jepa_frozen", "jepa_joint"]:
        m = sb3_meta[(k, g)]
        cells.append(f"{fmt(m['test_final'])} $\\pm$ {fmt(m['test_final_std'])}")
    lines.append(f"{g} & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
for label, field, f in [("Mean FPS", "fps", "{:.0f}"), ("Mean wall-clock [h]", "runtime_h", "{:.2f}")]:
    cells = [f.format(np.mean([sb3_meta[(k, g)][field] for g in TRAIN_GAMES]))
             for k in ["nature", "jepa_frozen", "jepa_joint"]]
    lines.append(f"{label} & " + " & ".join(cells) + r" \\")
lines.append("Trainable params [M] & " + " & ".join(
    f"{TRAINABLE[k]:.2f}" for k in ["nature", "jepa_frozen", "jepa_joint"]) + r" \\")
open(os.path.join(TAB, "sb3_test_rows.tex"), "w").write("\n".join(lines) + "\n")

# Sample efficiency: env steps until the learning curve (100 bins, FSQ smoothed over 5)
# first reaches 50% / 80% of the way from NatureCNN's initial to its final return.
def binned_curve(k, g, n=100):
    xs, ys = curves[(k, g)]
    e = np.linspace(0, 10_000_000, n + 1)
    idx = np.digitize(xs, e) - 1
    cy = np.array([ys[idx == b].mean() if (idx == b).any() else np.nan for b in range(n)])
    cy = pd.Series(cy).ffill().rolling(5 if k.startswith("fsq") else 1, min_periods=1).mean().to_numpy()
    return (e[:-1] + e[1:]) / 2, cy


def steps_to(k, g, thr):
    x, y = binned_curve(k, g)
    hit = np.where(y >= thr)[0]
    return x[hit[0]] / 1e6 if len(hit) else np.inf


eff = []
for g in TRAIN_GAMES:
    _, yn = binned_curve("nature", g)
    lo, fin = yn[:3].mean(), yn[-10:].mean()
    row = {"game": g}
    for f in (0.5, 0.8):
        for k in ("nature", "fsq_joint", "fsq_frozen"):
            row[f"{k}@{int(f * 100)}"] = steps_to(k, g, lo + f * (fin - lo))
    eff.append(row)
eff = pd.DataFrame(eff).set_index("game")
eff.to_csv(os.path.join(TAB, "sample_efficiency.csv"))
lines = []
for g in TRAIN_GAMES:
    cells = []
    for f in ("50", "80"):
        n_, j_, z_ = (eff.loc[g, f"{k}@{f}"] for k in ("nature", "fsq_joint", "fsq_frozen"))
        best = min(n_, j_, z_)
        for v in (n_, j_, z_):
            s = "--" if np.isinf(v) else f"{v:.1f}"
            cells.append(r"\textbf{" + s + "}" if (v == best and not np.isinf(v)) else s)
    lines.append(f"{g} & " + " & ".join(cells) + r" \\")
lines.append(r"\midrule")
wins = {f: int((eff[f"fsq_joint@{f}"] < eff[f"nature@{f}"]).sum()) for f in ("50", "80")}
lines.append(r"FSQ joint faster & \multicolumn{3}{c}{" + f"{wins['50']}/11" + r"} & \multicolumn{3}{c}{"
             + f"{wins['80']}/11" + r"} \\")
open(os.path.join(TAB, "sample_eff_rows.tex"), "w").write("\n".join(lines) + "\n")
print("sample efficiency (M steps):\n", eff.round(2).to_string())

# V-JEPA feature ablation: flattened 9216-d tokens vs pooled 256-d feature (W&B *_newextractor runs)
ABL = [("jepa_frozen", "flat", "frozen", "0.01"), ("jepa_joint", "flat", "joint", "0.01"),
       ("jepa_frozen_pool", "pool", "frozen", "0.01"), ("jepa_joint_pool", "pool", "joint", "0.01"),
       ("jepa_frozen_pool_ent05", "pool", "frozen", "0.05")]
lines = []
for k, feat, mode, ent in ABL:
    sub = runs[(runs.method == k) & runs.game.isin(TRAIN_GAMES)].set_index("game")
    h = [hns(g, sub.loc[g, "final"]) for g in TRAIN_GAMES]
    ht = [hns(g, sub.loc[g, "test_final"]) for g in TRAIN_GAMES]
    wins = sum(sub.loc[g, "final"] > scores[("jepa_" + mode, g)] for g in TRAIN_GAMES) if feat == "pool" else None
    flab = r"9\,216 (flat)" if feat == "flat" else "256 (pooled)"
    lines.append(
        f"{flab} & {mode} & {ent} & "
        f"{np.median(h):.3f} & {np.median(ht):.3f} & "
        f"{sub['explained_var_end'].mean():.2f} & {sub['entropy_end'].mean():.2f} & "
        f"{sub['fps'].mean():.0f} & {'--' if wins is None else f'{wins}/11'} \\\\")
open(os.path.join(TAB, "jepa_ablation_rows.tex"), "w").write("\n".join(lines) + "\n")
abl_scores = pd.DataFrame({k: [scores[(k, g)] for g in TRAIN_GAMES] for k, *_ in ABL}, index=TRAIN_GAMES)
abl_scores.to_csv(os.path.join(TAB, "jepa_ablation_scores.csv"))

summary = {
    "median_hns_seen": med.to_dict(), "mean_hns_seen": mean.to_dict(),
    "median_hns_heldout": medh.to_dict(),
    "games_joint_beats_frozen_fsq": int((tab["fsq_joint"] > tab["fsq_frozen"]).sum()),
    "games_fsqjoint_ge_nature": int((tab.loc[TRAIN_GAMES, "fsq_joint"] >= tab.loc[TRAIN_GAMES, "nature"]).sum()),
    "games_fsqfrozen_ge_nature": int((tab.loc[TRAIN_GAMES, "fsq_frozen"] >= tab.loc[TRAIN_GAMES, "nature"]).sum()),
    "fps": {k: float(np.mean([sb3_meta[(k, g)]["fps"] for g in TRAIN_GAMES])) for k in ["nature", "jepa_frozen", "jepa_joint"]},
}
json.dump(summary, open(os.path.join(TAB, "summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
print(tab.round(1).to_string())

# --------------------------------------------------------------------------- Fig: learning curves
def smooth(xs, ys, key, n_bins=100):
    """bin to a common 100-point grid; FSQ curves are per-episode, SB3 already a running mean."""
    edges = np.linspace(0, 10_000_000, n_bins + 1)
    idx = np.digitize(xs, edges) - 1
    cx, cy = [], []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            cx.append((edges[b] + edges[b + 1]) / 2)
            cy.append(ys[sel].mean())
    cy = pd.Series(cy).rolling(5 if key.startswith("fsq") else 1, min_periods=1).mean().to_numpy()
    return np.array(cx), cy


fig, axes = plt.subplots(3, 5, figsize=(7.2, 4.6))
for ax, g in zip(axes.flat, games_all):
    for k in METHODS:
        if (k, g) not in curves:
            continue
        x, y = smooth(*curves[(k, g)], k)
        ax.plot(x / 1e6, y, color=C[k], ls=LS[k], lw=1.2, label=LABEL[k])
    ax.set_title(g + (" (held-out)" if g in HELDOUT else ""), pad=2)
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 5, 10])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, p: f"{v/1e6:g}M" if abs(v) >= 1e6 else (f"{v/1e3:g}k" if abs(v) >= 1e3 else f"{v:g}")))
for ax in axes[-1]:
    ax.set_xlabel("env steps [M]")
for ax in axes[:, 0]:
    ax.set_ylabel("episode return")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.6, w_pad=0.4)
fig.savefig(os.path.join(FIG, "learning_curves.pdf"))
plt.close(fig)

# --------------------------------------------------------------------------- Fig: HNS summary
fig, ax = plt.subplots(figsize=(3.4, 2.1))
xs = np.arange(len(METHODS))
w = 0.38
b1 = ax.bar(xs - w / 2 - 0.01, [med[k] for k in METHODS], w, color=[C[k] for k in METHODS],
            edgecolor="white", linewidth=0.8)
b2 = ax.bar(xs + w / 2 + 0.01, [mean[k] for k in METHODS], w, color=[C[k] for k in METHODS],
            edgecolor="white", linewidth=0.8, hatch="////", alpha=0.85)
for bars in (b1, b2):
    for r in bars:
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.02, f"{r.get_height():.2f}",
                ha="center", va="bottom", fontsize=6.2, color="#333333")
ax.set_xticks(xs)
ax.set_xticklabels([LABEL[k].replace(" (scratch)", "\n(scratch)").replace(" frozen", "\nfrozen")
                    .replace(" joint", "\njoint") for k in METHODS], fontsize=6.5)
ax.set_ylabel("human-normalised score")
ax.grid(axis="x", visible=False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#9a9a9a", label="median"),
                   Patch(facecolor="#9a9a9a", hatch="////", label="mean")],
          frameon=False, loc="upper right")
fig.savefig(os.path.join(FIG, "hns_summary.pdf"))
plt.close(fig)

# --------------------------------------------------------------------------- Fig: JEPA drift
def parse_drift(path):
    t = open(path).read()
    rows = re.findall(r"step (\d) \| predictor=([\d.]+)\s+identity=([\d.]+)", t)
    return np.array([[int(a), float(b), float(c)] for a, b, c in rows])

dv = parse_drift(os.path.join(ROOT, "evaluation", "outputs", "results", "val_true.log"))
dt = parse_drift(os.path.join(ROOT, "evaluation", "outputs", "results", "test_unk.log"))
fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.0), sharey=True)
for ax, d, title in [(axes[0], dv, "11 seen games (true env id)"),
                     (axes[1], dt, "4 held-out games (UNKNOWN slot)")]:
    ax.plot(d[:, 0], d[:, 1], "-o", color=C["nature"], lw=1.4, ms=4, label="JEPA predictor rollout")
    ax.plot(d[:, 0], d[:, 2], "--s", color="#7a7a7a", lw=1.2, ms=3.5, label=r"identity baseline $z_t$")
    ax.axvspan(3.5, 5.3, color="#f2f2f2", zorder=0)
    ax.set_title(title, pad=3)
    ax.set_xlabel("rollout horizon $k$")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlim(0.7, 5.3)
axes[0].set_ylabel("cosine sim. to target $\\bar z_{t+k}$")
axes[0].text(4.4, 0.815, "no shared\nframes", ha="center", fontsize=6.5, color="#555555")
axes[0].legend(frameon=False, loc="lower left")
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(FIG, "jepa_drift.pdf"))
plt.close(fig)

# --------------------------------------------------------------------------- Fig: pre-training curves (W&B)
HJ = os.path.join(EXP, "history", "atari-vjepa-world-model")
HA = os.path.join(EXP, "history", "atari-universal-feature-extractor")

fsq_test = pd.read_csv(os.path.join(HA, "64k242nd.csv")).dropna(subset=["Test Loss"])["Test Loss"].to_numpy()
jm = pd.read_csv(os.path.join(HJ, "kter723b.csv")).dropna(subset=["val/jepa_physics_loss"])
jm = jm.groupby("epoch").last().reset_index()

fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.85))
ax = axes[0]
ep = np.arange(1, len(fsq_test) + 1)
ax.plot(ep, fsq_test * 1e3, color=C["nature"], lw=1.4)
i = int(np.argmin(fsq_test))
ax.plot(ep[i], fsq_test[i] * 1e3, "o", color=C["nature"], ms=4)
ax.annotate(f"min, ep. {i + 1}", (ep[i], fsq_test[i] * 1e3), xytext=(18, -2), textcoords="offset points",
            fontsize=6.5, color="#555555")
ax.set_title("(a) FSQ-VAE: held-out MSE", pad=3)
ax.set_xlabel("epoch"); ax.set_ylabel(r"MSE [$\times10^{-3}$]")

ax = axes[1]
ax.plot(jm["epoch"], jm["val/jepa_physics_loss"] * 1e3, color=C["nature"], lw=1.4, label="seen (val)")
ax.plot(jm["epoch"], jm["heldout_unk/jepa_physics_loss"] * 1e3, color=C["fsq_frozen"], lw=1.4,
        ls="--", label="held-out (UNK)")
ax.set_title(r"(b) V-JEPA: $\mathcal{L}_{\mathrm{JEPA}}$", pad=3)
ax.set_xlabel("epoch"); ax.set_ylabel(r"latent MSE [$\times10^{-3}$]")
ax.legend(frameon=False, loc="lower right", fontsize=6.3)

ax = axes[2]
ax.plot(jm["epoch"], jm["health/effective_rank"], color=C["nature"], lw=1.4)
ax.axhline(float(jm["health_init/effective_rank"].dropna().iloc[0]) if "health_init/effective_rank" in jm
           and jm["health_init/effective_rank"].notna().any() else 2.53, color="#8a8a8a", lw=1, ls=":")
ax.text(92, 6, "untrained", ha="right", fontsize=6.5, color="#555555")
ax.set_title("(c) V-JEPA: effective rank", pad=3)
ax.set_xlabel("epoch"); ax.set_ylabel("eff. rank (of 256)")

ax = axes[3]
ax.plot(jm["epoch"], jm["val/inverse_accuracy"] * 100, color=C["nature"], lw=1.4, label="seen (val)")
ax.plot(jm["epoch"], jm["heldout_unk/inverse_accuracy"] * 100, color=C["fsq_frozen"], lw=1.4, ls="--",
        label="held-out (UNK)")
ax.axhline(float(jm["val/inverse_chance_accuracy"].iloc[-1]) * 100, color="#8a8a8a", lw=1, ls=":")
ax.text(92, 15, "chance (seen)", ha="right", fontsize=6.5, color="#555555")
ax.set_title("(d) V-JEPA: inverse accuracy", pad=3)
ax.set_xlabel("epoch"); ax.set_ylabel("accuracy [%]")
fig.tight_layout(w_pad=0.7)
fig.savefig(os.path.join(FIG, "pretrain_curves.pdf"))
plt.close(fig)
print("FSQ held-out MSE: min %.5f at epoch %d, final %.5f" % (fsq_test.min(), i + 1, fsq_test[-1]))
print("JEPA val loss epoch0 %.5f -> final %.5f; eff rank %.1f -> %.1f" % (
    jm["val/jepa_physics_loss"].iloc[0], jm["val/jepa_physics_loss"].iloc[-1],
    jm["health/effective_rank"].iloc[0], jm["health/effective_rank"].iloc[-1]))

# --------------------------------------------------------------------------- Table: inverse-loss weight sweep (W&B)
jruns = json.load(open(os.path.join(EXP, "atari-vjepa-world-model_runs_full.json")))
lines = []
for r in jruns:
    if not (r["name"].startswith("v2-sweep") or r["name"] == "v2-main") or r["state"] != "finished":
        continue
    s, c = r["summary"], r["config"]
    rnd = "A" if r["created_at"].startswith("2026-08-12") else "B"
    tag = "main" if r["name"] == "v2-main" else rnd
    lines.append((rnd, c["inverse_weight"], tag, s["final/global_step"], s["best/epoch"],
                  s["best/val_jepa_loss"], s["best/heldout_unk_jepa_loss"],
                  s["best/val_inverse_accuracy"], s["best/val_inverse_information_gain"],
                  s["best/effective_rank"]))
lines.sort(key=lambda t: (t[0], t[2] == "main", t[1]))
rows = []
for rnd, lam, tag, steps, bep, vj, hj, acc, ig, er in lines:
    lab = "main run" if tag == "main" else rnd
    rows.append(f"{lab} & {lam:g} & {steps/1000:.1f}\\,k & {vj*1e3:.2f} & {hj*1e3:.2f} & "
                f"{acc*100:.1f} & {ig:.2f} & {er:.1f} \\\\")
    if tag == "A" and lam == 0.1:
        rows.append(r"\midrule")
open(os.path.join(TAB, "sweep_rows.tex"), "w").write("\n".join(rows) + "\n")

# --------------------------------------------------------------------------- Fig: reconstructions
from models.fsq_vae_decoder import AtariFSQVAE
from models.residual_vqvae import AtariResidualVQVAE


def load(model, path):
    sd = torch.load(path, map_location="cpu")
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.eval()


models = {
    "Residual VQ-VAE": load(AtariResidualVQVAE(num_embeddings=512, embedding_dim=64),
                            os.path.join(ROOT, "artifacts/checkpoints/autoencoders/vq_vae_collab/residual_vqvae_epoch_15.pth")),
    "Residual VQ-VAE\n(sprite-weighted)": load(AtariResidualVQVAE(num_embeddings=512, embedding_dim=64),
                            os.path.join(ROOT, "artifacts/checkpoints/autoencoders/vq_vae_weighted_collab/residual_vqvae_epoch_15.pth")),
    "FSQ-VAE\n(resize-conv)": load(AtariFSQVAE(), os.path.join(ROOT, "artifacts/checkpoints/autoencoders/fsq_vae_decoder/fsq_vae_epoch_125.pth")),
}


def frames(game, split, n, seed=3):
    d = np.load(os.path.join(ROOT, "artifacts", "data", "expert", split, f"{game}NoFrameskip-v4_expert_50000_frames.npz"))["frames"]
    idx = np.random.RandomState(seed).choice(len(d), n, replace=False)
    return torch.tensor(np.transpose(d[idx], (0, 3, 1, 2)), dtype=torch.float32) / 255.0


# quantitative: plain MSE and moving-pixel MSE per model, seen vs held-out
def moving_mse(r, x, thr=0.05):
    diff = (x[:, 1:] - x[:, :-1]).abs().amax(1, keepdim=True)
    m = (diff > thr).float().expand_as(x)
    e = (r - x) ** 2
    return (e.mean().item(), ((e * m).sum() / m.sum().clamp_min(1)).item())

recon = {}
with torch.no_grad():
    for split, gl in [("train", TRAIN_GAMES), ("test", HELDOUT)]:
        for name, mdl in models.items():
            tot, mov = [], []
            for g in gl:
                x = frames(g, split, 256, seed=11)
                r, _ = mdl(x)
                a, b = moving_mse(r, x)
                tot.append(a); mov.append(b)
            recon[(name.replace("\n", " "), split)] = (np.mean(tot), np.mean(mov))
lines = []
for name in models:
    n = name.replace("\n", " ")
    s, h = recon[(n, "train")], recon[(n, "test")]
    lines.append(f"{n} & {s[0]*1e3:.2f} & {s[1]*1e3:.2f} & {h[0]*1e3:.2f} & {h[1]*1e3:.2f} \\\\")
open(os.path.join(TAB, "recon_rows.tex"), "w").write("\n".join(lines) + "\n")
print(recon)

show = [("Pong", "train"), ("SpaceInvaders", "train"), ("MsPacman", "train"), ("Breakout", "train"),
        ("Alien", "test"), ("IceHockey", "test")]
fig, axes = plt.subplots(len(models) + 1, len(show), figsize=(7.0, 4.9))
with torch.no_grad():
    for j, (g, split) in enumerate(show):
        x = frames(g, split, 1, seed=5 + j)
        axes[0, j].imshow(x[0, 3], cmap="gray", vmin=0, vmax=1)
        axes[0, j].set_title(g + ("\n(held-out)" if split == "test" else "\n(seen)"), fontsize=7.5)
        for i, (name, mdl) in enumerate(models.items(), start=1):
            r, _ = mdl(x)
            axes[i, j].imshow(r[0, 3], cmap="gray", vmin=0, vmax=1)
for i, name in enumerate(["Input frame"] + list(models)):
    axes[i, 0].set_ylabel(name, fontsize=7.5)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
fig.tight_layout(h_pad=0.3, w_pad=0.2)
fig.savefig(os.path.join(FIG, "reconstructions.pdf"))
plt.close(fig)
print("figures written to", FIG)
