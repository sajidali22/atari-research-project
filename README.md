# Towards a Universal Feature Extractor for Atari

TU Dresden research project. Can one visual encoder, pre-trained once on expert gameplay
from eleven Atari games, replace the per-game NatureCNN inside PPO?

We compare autoencoders (Gaussian VAE, VQ-VAE, FSQ-VAE) and an action- and game-conditioned
V-JEPA world model, each used **frozen** or **fine-tuned jointly** with PPO for 10 M steps on
11 pre-training games and 4 held-out games, against a NatureCNN trained from scratch.

**Headline result** (see [`paper/main.pdf`](paper/main.pdf)): the jointly fine-tuned FSQ-VAE
reaches a higher median human-normalised score than the NatureCNN on the seen games
(1.07 vs 0.74) and learns faster on several of them. Frozen encoders fall far behind, and the
V-JEPA features were barely usable by PPO (median 0.05). Pre-trained encoders work as an
initialisation, not yet as a frozen replacement.

## Pipeline

```
 data_collection/            autoencoders/  ─┐
 expert agents from     ──►                  ├──►  ppo/            ──►  evaluation/
 the HF Hub play Atari       jepa/          ─┘     PPO (SB3) on         W&B export, figures,
 → frame stacks /            representation        frozen / joint       tables for paper/
   transitions               learning              encoders
```

| Folder | What it contains |
|---|---|
| [`data_collection/`](data_collection) | Roll out pretrained SB3 agents (HuggingFace Hub) and save expert frame stacks (`collect_expert_data.py`) or `(obs, action, next_obs, terminal)` transitions for the JEPA (`collect_jepa_transitions.py`); viewers and validators |
| [`models/`](models) | Autoencoder architectures shared by every stage: `StandardVAE`, `EmaVqVae`, `AtariResidualVQVAE`, `AtariFSQVAE` (`fsq_vae_decoder.py`, the one used in the paper; `fsq_vae.py` is the older variant), `AdvancedViTVQVAE` |
| [`autoencoders/`](autoencoders) | Training loop for the models above (`config.py` selects the model) |
| [`jepa/`](jepa) | Environment-conditioned V-JEPA world model (v2): model, training, diagnostics, few-shot, SLURM jobs. Has its own [README](jepa/README.md), [CHANGES](jepa/CHANGES.md) and [HPC guide](jepa/HPC.md) |
| [`ppo/`](ppo) | **The PPO pipeline used for the paper**: Stable-Baselines3 PPO with a pluggable feature extractor (`factory.py`), YAML configs, SLURM array job |
| [`ppo/custom/`](ppo/custom) | Earlier hand-written CleanRL-style PPO (`train_custom.py`) and a live viewer (`play.py`) |
| [`evaluation/`](evaluation) | W&B download + plotting; `report_scripts/` regenerate every figure and table of the paper into `outputs/` |
| [`paper/`](paper) | Final report (LaTeX + PDF) |
| [`slides/`](slides) | Final presentation (LaTeX + PDF + speaker script) |
| [`docs/`](docs) | Technical documentation, design analyses, archived drafts (see [docs/README.md](docs/README.md)) |
| `artifacts/` | **Not in git.** Checkpoints, datasets, W&B exports, logs (see below) |

## Setup

Python ≥ 3.10, a GPU is recommended (everything falls back to CPU).

```bash
pip install -r requirements.txt        # or: uv sync
wandb login                            # all training scripts log to Weights & Biases
```

Run commands from the **repository root** unless a section says otherwise.

## How to run each stage

**1. Collect data**
```bash
python data_collection/collect_expert_data.py        # → artifacts/data/expert/{train,test}/*_frames.npz
python data_collection/collect_jepa_transitions.py   # → artifacts/data/jepa/custom_datasets/{train,test}/*_transitions.npz
python data_collection/view_data.py   [file.npz]     # look at frame stacks
python data_collection/validate_transitions.py [file.npz]
```
11 training games (BeamRider, Breakout, DemonAttack, Enduro, MsPacman, Pong, Qbert, Riverraid,
RoadRunner, Seaquest, SpaceInvaders) and 4 held-out games (Alien, Asteroids, Atlantis,
IceHockey), 50 k steps each.

**2a. Train an autoencoder**: edit `autoencoders/config.py` (`MODEL_TYPE`, hyperparameters), then
```bash
python -m autoencoders.train                       # → artifacts/checkpoints/autoencoders/new_run/
python -m autoencoders.inspect_reconstructions     # original vs reconstruction plot
```

**2b. Train the JEPA world model** (from inside `jepa/`, see [jepa/README.md](jepa/README.md))
```bash
cd jepa
python games.py                                    # build the action registry
python JEPA.py                                     # architecture self-test
python jepa_train.py --smoke --wandb-mode disabled # pipeline smoke test
python jepa_train.py --epochs 100                  # → artifacts/checkpoints/jepa/
python eval_jepa_diagnostics.py                    # drift, action sensitivity, collapse checks
```

**3. PPO on top of an encoder** (from inside `ppo/`)
```bash
cd ppo
python train_sb3.py --config configs/nature.yaml --game PongNoFrameskip-v4             # baseline
python train_sb3.py --config configs/jepa.yaml   --game PongNoFrameskip-v4 --frozen    # or --joint
python train_sb3.py --config configs/jepa.yaml   --arch fsq \
    --checkpoint ../artifacts/checkpoints/autoencoders/fsq_vae_decoder/fsq_vae_epoch_125.pth --joint
sbatch job_ppo_jepa.sh                                                                 # all games on SLURM
```
Available `--arch`: `nature`, `jepa`, `vae`, `ema_vqvae`, `vqvae`, `fsq`, `vit_vqvae`.

**4. Figures and tables**: see [docs/reproducing_figures.md](docs/reproducing_figures.md).

## Artifacts

Large files live in the git-ignored `artifacts/` folder:

```
artifacts/
  checkpoints/
    jepa/                 vjepa_v2_best.pt (used in the paper), per-10-epoch snapshots, old_v1/
    autoencoders/         fsq_vae_decoder/ (epoch 125 used in the paper), vq_vae_collab/,
                          vq_vae_weighted_collab/, fsq_vae_collab/
    ppo/                  custom-PPO agents (e.g. mspacman.pth)
  data/
    expert/{train,test}/  expert frame stacks (N, 84, 84, 4) uint8, key 'frames'
    jepa/                 custom_datasets/ and custom_datasets_small/{train,val,test}/
                          transitions with keys obs / actions / next_obs / terminals
  wandb_export/           run tables + full histories of both W&B projects
  logs/                   SLURM logs, local W&B run dirs, videos
```

If you got this repository without `artifacts/`, either ask the authors for a copy or
regenerate it: the data with `data_collection/`, the checkpoints with the training
scripts, and the W&B export with `evaluation/report_scripts/export_wandb.py`.

The JEPA training run used a larger 5-shard training split (`custom_datasets/train`, CHW)
and a smaller HWC split (`custom_datasets_small/{train,val,test}`) that were built on the
HPC cluster. `collect_jepa_transitions.py` produces the same format (one 50 k-step shard per
game), and `jepa/dataset_JEPA.py` auto-detects either layout.

## Known limitations

- The JEPA trains on single-step prediction only; see "Known limitation" in [jepa/README.md](jepa/README.md).
- `ppo/custom/` is the earlier hand-written PPO. The paper's numbers come from `ppo/train_sb3.py`.
- There is no automated test suite. `python jepa/JEPA.py` and `python -m models.vit_vqae` are self-checks.

## Credits

Sajid Ali Sadruddin, TU Dresden. Research topic by Oleh Shkalikov.
