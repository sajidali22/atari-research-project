"""Recompute every PPO result in the report from the exported W&B histories.

Run export_wandb.py first, then from the repository root:
    python evaluation/report_scripts/wandb_results.py

Writes evaluation/outputs/tables/wandb_*.csv and prints the comparison with the older
log/CSV-derived tables.

Some SB3 runs were resumed under the same W&B id, so one history can hold a
500k-step pilot, the full 10M run and a crashed restart. Segments are split
where global_step goes backwards and only the segment that reaches 10M is used.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP = os.path.join(ROOT, "artifacts", "wandb_export")
HIST = os.path.join(EXP, "history", "atari-universal-feature-extractor")
TAB = os.path.join(ROOT, "evaluation", "outputs", "tables")

TRAIN_GAMES = ["BeamRider", "Breakout", "DemonAttack", "Enduro", "MsPacman", "Pong",
               "Qbert", "Riverraid", "RoadRunner", "Seaquest", "SpaceInvaders"]
HELDOUT = ["Alien", "Asteroids", "Atlantis", "IceHockey"]
TOTAL = 10_000_000


def method_of(name):
    """W&B run name -> (method key, game)."""
    if name.startswith("PPO_FSQ-VAE_JOINT_"):
        return "fsq_joint", name[len("PPO_FSQ-VAE_JOINT_"):].split("NoFrameskip")[0]
    if name.startswith("PPO_FSQ-VAE_"):
        return "fsq_frozen", name[len("PPO_FSQ-VAE_"):].split("NoFrameskip")[0]
    p = name.split("_")                      # PPO_<arch>_<mode>_<Game>_s0[_suffix]
    key = {"nature_scratch": "nature"}.get(f"{p[1]}_{p[2]}", f"{p[1]}_{p[2]}")
    suffix = "_".join(p[5:])
    if suffix == "newextractor":
        key += "_pool"
    elif suffix == "newextractor_ent_coeff0.05":
        key += "_pool_ent05"
    return key, p[3]


def main_segment(df, col):
    """rows of the segment that reaches the full budget."""
    d = df.dropna(subset=["global_step"]).reset_index(drop=True)
    seg = (d["global_step"].diff() < 0).cumsum()
    best = d.groupby(seg)["global_step"].max().idxmax()
    return d[seg == best]


def summarise(run_id, name):
    key, game = method_of(name)
    df = pd.read_csv(os.path.join(HIST, f"{run_id}.csv"), low_memory=False)
    out = dict(method=key, game=game, run_id=run_id)
    if key.startswith("fsq"):
        r = df.dropna(subset=["charts/episodic_return"])
        xs, ys = r["global_step"].to_numpy(float), r["charts/episodic_return"].to_numpy(float)
        out["final"] = float(ys[xs >= 0.9 * TOTAL].mean())
        out["max_step"] = float(xs.max())
        out["runtime_h"] = float(df["_runtime"].max() / 3600)
        return out, (xs, ys)
    seg = main_segment(df, "rollout/ep_rew_mean")
    r = seg.dropna(subset=["rollout/ep_rew_mean"])
    xs, ys = r["global_step"].to_numpy(float), r["rollout/ep_rew_mean"].to_numpy(float)
    out["final"] = float(ys[xs >= 0.9 * TOTAL].mean())
    out["max_step"] = float(xs.max())
    fps = seg["time/fps"].dropna()
    out["fps"] = float(fps.iloc[-1])
    out["runtime_h"] = float((seg["_runtime"].max() - seg["_runtime"].min()) / 3600)
    ev = seg.dropna(subset=["train/explained_variance"])
    out["explained_var_end"] = float(ev["train/explained_variance"].iloc[-10:].mean())
    ent = seg.dropna(subset=["train/entropy_loss"])
    out["entropy_end"] = float(-ent["train/entropy_loss"].iloc[-10:].mean())
    ev_rew = seg.dropna(subset=["eval/mean_reward"])
    out["eval_best"] = float(ev_rew["eval/mean_reward"].max())
    out["eval_best_step"] = float(ev_rew.loc[ev_rew["eval/mean_reward"].idxmax(), "global_step"])
    # test scores logged once at the end of the full run
    t = df.dropna(subset=["test_final_mean"])
    t = t[t["timesteps"] >= TOTAL]
    if len(t):
        out["test_final"] = float(t["test_final_mean"].iloc[-1])
        out["test_final_std"] = float(t["test_final_std"].iloc[-1])
        out["test_best"] = float(t["test_best_mean"].iloc[-1])
    return out, (xs, ys)


def main():
    runs = json.load(open(os.path.join(EXP, "atari-universal-feature-extractor_runs_full.json")))
    rows, curves = [], {}
    for r in runs:
        if not r["name"].startswith("PPO_"):
            continue
        out, curve = summarise(r["id"], r["name"])
        out["state"] = r["state"]
        rows.append(out)
        curves[(out["method"], out["game"])] = curve
    df = pd.DataFrame(rows).sort_values(["method", "game"])
    df.to_csv(os.path.join(TAB, "wandb_ppo_runs.csv"), index=False)
    final = df.pivot(index="game", columns="method", values="final").reindex(TRAIN_GAMES + HELDOUT)
    final.to_csv(os.path.join(TAB, "wandb_final_scores.csv"))
    pd.to_pickle(curves, os.path.join(EXP, "ppo_curves.pkl"))
    pd.set_option("display.width", 250)
    print(final.round(1).to_string())
    print(df.groupby("method")[["max_step", "fps", "runtime_h", "explained_var_end", "entropy_end"]]
          .mean().round(3).to_string())


if __name__ == "__main__":
    main()
