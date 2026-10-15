#!/bin/bash
# PPO on the jepa representation, frozen vs joint, over the 11 games the world
# model saw during training. 22 tasks = 11 games x 2 modes, 1 seed.
#
# submit from this dir! train_sb3.py imports factory.py from the cwd.
#
#     cd ppo && sbatch job_ppo_jepa.sh
#     SEED=1 sbatch job_ppo_jepa.sh
#     sbatch --array=5,16 job_ppo_jepa.sh                    # pong only, both modes
#     TIMESTEPS=500000 sbatch --array=5,16 job_ppo_jepa.sh   # short pilot to check fps
#
# for the baseline use 0-10, NOT 0-21 (see the nature check further down):
#
#     CONFIG=configs/nature.yaml sbatch --job-name=ppo_nature --array=0-10 job_ppo_jepa.sh
#
# requeuing is fine, every task runs with --resume which does nothing if there is
# no checkpoint yet.
#
#SBATCH --job-name=ppo_jepa
#SBATCH --partition=capella
#SBATCH --account=p_atari26
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-21
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail

cd "${UNIVERSAL_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
mkdir -p logs

# the 11 games from the checkpoints train_games list
GAMES=(BeamRider Breakout DemonAttack Enduro MsPacman Pong Qbert Riverraid \
       RoadRunner Seaquest SpaceInvaders)
MODES=(frozen joint)

# 0-10 frozen, 11-21 joint
TASK="${SLURM_ARRAY_TASK_ID:-0}"
GAME_NAME="${GAMES[$((TASK % 11))]}"
MODE="${MODES[$((TASK / 11))]}"
JOINT_FLAG="--frozen"
[ "$MODE" = "joint" ] && JOINT_FLAG="--joint"

GAME="${GAME_NAME}NoFrameskip-v4"
SEED="${SEED:-0}"
CONFIG="${CONFIG:-configs/jepa.yaml}"
ARCH=$(python3 -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['extractor']['arch'])")

# nature has no pretrained weights so frozen/joint means nothing, task N and N+11
# would get the same run name and overwrite each other. bail out instead.
if [ "$ARCH" = "nature" ]; then
    MODE_LABEL="scratch"
    if [ "$TASK" -ge 11 ]; then
        echo "ERROR: arch 'nature' has no frozen/joint distinction, so task $TASK is a" >&2
        echo "       duplicate of task $((TASK - 11)) and would overwrite it." >&2
        echo "       Submit the baseline with: sbatch --array=0-10 ..." >&2
        exit 1
    fi
else
    MODE_LABEL="$MODE"
fi

# slurm only gives us %x/%A/%a in the filename so the raw log says nothing about
# which game it is. symlink a readable name next to it, matching runs/.
DESC="${ARCH}_${MODE_LABEL}_${GAME_NAME}_s${SEED}"
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    ln -sfn "${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" "logs/${DESC}.out"
    ln -sfn "${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err" "logs/${DESC}.err"
fi

# don't put defaults here, they would silently beat the config. these only get
# appended if you actually set the env var.
OVERRIDES=()
[ -n "${CKPT:-}" ]       && OVERRIDES+=(--checkpoint "$CKPT")
[ -n "${TIMESTEPS:-}" ]  && OVERRIDES+=(--timesteps "$TIMESTEPS")

echo "Job ${SLURM_JOB_ID:-local} task ${TASK} on ${SLURMD_NODENAME:-$(hostname)} | $(date)"
echo "  ${DESC}  (config=${CONFIG})"
echo "  overrides: ${OVERRIDES[*]:-(none, all params from config)}"

# never commit the key, export it in ~/.bashrc (see jepa/HPC.md)
if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "WARNING: WANDB_API_KEY unset; logging offline. Run 'wandb sync' afterwards." >&2
    export WANDB_MODE=offline
fi

source "$VENV_PATH/bin/activate"
nvidia-smi   # goes in the log, handy when a node gives us a slower gpu

python train_sb3.py \
    --config "$CONFIG" \
    $JOINT_FLAG \
    --game "$GAME" \
    --seed "$SEED" \
    ${OVERRIDES[@]+"${OVERRIDES[@]}"} \
    --resume \
    --wandb

echo "Done $(date)"
