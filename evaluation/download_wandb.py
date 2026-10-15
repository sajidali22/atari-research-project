"""Download episodic-return curves of the custom-PPO runs from W&B into CSVs.

Credentials come from WANDB_API_KEY or ~/.netrc (`wandb login`); never put a key
in this file.

    python evaluation/download_wandb.py                                   # FSQ-VAE joint runs
    python evaluation/download_wandb.py --match PPO_FSQ-VAE_ --out csv_data
"""

import argparse
import os

import wandb

HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--project", default="sajid-alisadruddin-/atari-universal-feature-extractor")
p.add_argument("--match", default="PPO_FSQ-VAE_JOINT", help="substring of the run names to download")
p.add_argument("--out", default="csv_data_joint", help="output folder, relative to evaluation/")
args = p.parse_args()

out_dir = os.path.join(HERE, args.out)
os.makedirs(out_dir, exist_ok=True)

api = wandb.Api()
runs = api.runs(args.project)

print("Locating runs and downloading data...")

for run in runs:
    if args.match in run.name:
        print(f"Downloading: {run.name}")

        # Pull the global step and the episodic return
        # samples=10000 ensures high resolution for the 10M steps
        history = run.history(keys=["global_step", "charts/episodic_return"], samples=10000)

        # Clean the dataframe and save it locally
        history = history.dropna()
        history.to_csv(os.path.join(out_dir, f"{run.name}.csv"), index=False)

print(f"All data successfully downloaded to {out_dir}")
