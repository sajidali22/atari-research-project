# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research project studying whether one pre-trained visual encoder (VAE / VQ-VAE / FSQ-VAE autoencoders, or an action- and game-conditioned V-JEPA world model) can replace the per-game NatureCNN inside PPO on Atari, used frozen or fine-tuned jointly. Stages: (1) collect expert gameplay, (2) train a representation, (3) train PPO on top of it, (4) evaluate and write up. `README.md` has the user-facing overview.

## Environment & running scripts

- Python `>=3.10`, deps in `pyproject.toml` / `requirements.txt`. On the author's machine the working env is the conda env `attari` (`~/miniconda3/envs/attari/bin/python`).
- No test suite, linter or formatter. Self-checks: `python jepa/JEPA.py`, `python -m models.vit_vqae`, `python jepa/jepa_train.py --smoke --wandb-mode disabled`.
- Training scripts log to Weights & Biases. Credentials come from `WANDB_API_KEY` / `~/.netrc`, never hardcode keys.
- GPU assumed, CPU fallback everywhere.
- **Working directories differ by stage**: `autoencoders/` and `ppo/custom/` run as modules from the repo root (`python -m autoencoders.train`, `python -m ppo.custom.train_custom --config ppo/custom/configs/fsq_vae.yaml`); `jepa/` and `ppo/` scripts are run from inside their own folder (flat imports like `from JEPA import ...`, `from factory import ...`).

## Layout

- `paths.py`: shared path constants (ROOT, `artifacts/` sub-folders, `evaluation/outputs/`). `jepa/config.py` mirrors the same layout without importing it.
- `artifacts/` (**git-ignored**) holds all checkpoints, datasets, W&B exports and logs. Code must reference artifacts through `paths.py` / `jepa/config.py` or paths relative to `__file__`, never cwd-relative strings.
- `data_collection/`: `collect_expert_data.py` rolls out pretrained HF-Hub SB3 agents (dicts `TRAIN_GAMES` / `TEST_GAMES`) into `artifacts/data/expert/{train,test}/*_frames.npz` (key `frames`, `(N,84,84,4)`). `collect_jepa_transitions.py` writes `obs/actions/next_obs/terminals` transitions to `artifacts/data/jepa/custom_datasets/`. `collect_random_frames.py` is an early random-policy collector.
- `models/`: autoencoder architectures, a package shared by `autoencoders/`, `ppo/factory.py` and `evaluation/report_scripts/`. `fsq_vae_decoder.py` (`AtariFSQVAE`) is the FSQ variant used in the paper; `fsq_vae.py` is the older one. `vit_vqae.py` reuses `EmaVectorQuantizer` from `ema_vqvae.py`. All take `[B,4,84,84]` in `[0,1]`.
- `autoencoders/`: `config.py` (experiment selector, `MODEL_TYPE`), `train.py`, `dataset.py` (in-RAM `.npz`), `dataset_mmap.py` (`.npy` memmap), `inspect_reconstructions.py`.
- `jepa/`: v2 environment-conditioned V-JEPA (`PaperAccurateJEPA` in `JEPA.py`: context encoder, EMA target encoder, action+env-conditioned predictor, inverse-dynamics head; SimNorm; `UNKNOWN` env slot for unseen games). `jepa_train.py` is the trainer that produced `vjepa_v2_best.pt` (it was `jepa_train_NEW.py`; the previous version is `jepa_train_legacy.py`). `eval_jepa_diagnostics.py`, `jepa_fewshot.py`, `games.py` (canonical ALE-18 action remap), `slurm/`. See `jepa/README.md`, `CHANGES.md`, `HPC.md`.
- `ppo/`: **the pipeline behind the paper's numbers**. `train_sb3.py` (Stable-Baselines3 PPO), `factory.py` (`build_policy_kwargs`, `ARCH_NAMES` = nature, jepa, vae, ema_vqvae, vqvae, fsq, vit_vqvae; strips `_orig_mod.` prefixes; `--frozen` / `--joint`), `configs/{nature,jepa}.yaml`, `job_ppo_jepa.sh` (SLURM array). Runs go to `ppo/runs/`.
- `ppo/custom/`: earlier hand-written PPO (`factory.py` with `UniversalFeatureExtractor` + `PPOAgent`, `train_custom.py`, `train_custom_basic.py`, `play.py` live viewer, `configs/`).
- `evaluation/`: `download_wandb.py`, `plot_baselines.py`, `csv_data*/`; `report_scripts/` regenerate the paper's figures and tables into `evaluation/outputs/` (they don't write into `paper/`). See `docs/reproducing_figures.md`.
- `paper/`: final LNCS report. `slides/`: final presentation.
- `docs/`: documentation, `analysis/` (design notes), `archive/` (superseded drafts; paths in there are historical, mapped in `docs/README.md`).

## Gotchas

- The JEPA transition datasets (`artifacts/data/jepa/custom_datasets{,_small}`) may not exist locally. Anything that loads them (JEPA training and diagnostics, some report scripts) will fail with a missing-dir error until they are generated or copied in.
- `ppo/custom/factory.py` loads checkpoints with `strict=False`, so a wrong checkpoint loads silently.
