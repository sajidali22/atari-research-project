"""Shared filesystem locations, anchored to the repository root.

Everything large (checkpoints, datasets, W&B exports, logs) lives under artifacts/,
which is git-ignored. See the "Artifacts" section of README.md for what goes where.

jepa/ does not import this file (its scripts run from inside jepa/ with flat
imports); jepa/config.py mirrors the same layout.
"""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))

ARTIFACTS = os.path.join(ROOT, "artifacts")
CHECKPOINTS = os.path.join(ARTIFACTS, "checkpoints")
AE_CHECKPOINTS = os.path.join(CHECKPOINTS, "autoencoders")
JEPA_CHECKPOINTS = os.path.join(CHECKPOINTS, "jepa")
PPO_CHECKPOINTS = os.path.join(CHECKPOINTS, "ppo")

DATA = os.path.join(ARTIFACTS, "data")
EXPERT_DATA = os.path.join(DATA, "expert")          # {train,test}/*_frames.npz   (frame stacks)
JEPA_DATA = os.path.join(DATA, "jepa")              # custom_datasets{,_small}/    (transitions)

WANDB_EXPORT = os.path.join(ARTIFACTS, "wandb_export")
LOGS = os.path.join(ARTIFACTS, "logs")

JEPA_DIR = os.path.join(ROOT, "jepa")

# what evaluation/report_scripts/ generate
OUTPUTS = os.path.join(ROOT, "evaluation", "outputs")
FIGURES = os.path.join(OUTPUTS, "figures")
TABLES = os.path.join(OUTPUTS, "tables")
RESULTS = os.path.join(OUTPUTS, "results")
