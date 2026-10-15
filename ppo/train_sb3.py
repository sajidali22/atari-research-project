#!/usr/bin/env python
"""PPO on top of a frozen (or jointly finetuned) feature extractor.

nothing here reimplements PPO, it's all stable-baselines3. this file just picks
the extractor (factory.py), builds the atari envs and calls learn().

    python train_sb3.py --config configs/nature.yaml --game PongNoFrameskip-v4
    python train_sb3.py --config configs/jepa.yaml --game PongNoFrameskip-v4 --joint
    python train_sb3.py --config configs/jepa.yaml --game AlienNoFrameskip-v4 \
        --env-mode unknown

hyperparams live in configs/*.yaml, cli only overrides the per-run stuff.
3 env splits with different seed offsets: train (seed), val (seed+10k, picks
best_model) and test (seed+20k). everything ends up in runs/<name>/.
"""

import argparse
import glob
import json
import os

import yaml
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (BaseCallback, CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder

from factory import ARCH_NAMES, build_policy_kwargs

# sample instead of argmax when evaluating, deterministic policies get stuck in
# loops on atari (paddle just stops moving). rl-zoo reports scores this way too.
DETERMINISTIC = False


def parse_args():
    # 2 pass parsing: read --config first so the yaml can supply the defaults
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="configs/jepa.yaml")
    cfg = yaml.safe_load(open(pre.parse_known_args()[0].config))
    ex, ppo, ev = cfg["extractor"], cfg["ppo"], cfg["eval"]

    p = argparse.ArgumentParser(parents=[pre], description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default=ex["arch"], choices=ARCH_NAMES)
    p.add_argument("--checkpoint", default=ex.get("checkpoint"))
    p.add_argument("--env-mode", default=ex.get("env_mode", "true"),
                   choices=["true", "unknown"],
                   help="jepa only: 'unknown' uses the UNKNOWN slot, for unseen games")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--joint", dest="freeze", action="store_false",
                      help="fine-tune the extractor with the policy")
    mode.add_argument("--frozen", dest="freeze", action="store_true",
                      help="keep the extractor fixed (default)")
    p.set_defaults(freeze=ex.get("freeze", True))

    p.add_argument("--game", default="PongNoFrameskip-v4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=ppo["total_timesteps"])
    p.add_argument("--n-envs", type=int, default=ppo["n_envs"])
    for k in ("eval_freq", "n_eval_episodes", "n_test_episodes", "save_freq",
              "video_freq", "video_length"):
        p.add_argument(f"--{k.replace('_', '-')}", type=int, default=ev[k])
    p.add_argument("--resume", action="store_true",
                   help="continue from the newest checkpoint in the run directory")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="Atari-Universal-Feature-Extractor")

    args = p.parse_args()
    args.ppo, args.net_arch = ppo, cfg.get("policy", {}).get("net_arch", [512])
    return args


def run_name(args):
    game = args.game.split("NoFrameskip")[0]
    mode = "scratch" if args.arch == "nature" else ("frozen" if args.freeze else "joint")
    unk = "-unk" if args.env_mode == "unknown" else ""
    return f"PPO_{args.arch}{unk}_{mode}_{game}_s{args.seed}"


def make_env(game, n_envs, seed, raw_rewards=False):
    """raw_rewards -> no reward clipping / no terminal on life loss, so we get the
    real game score. used for val + test, training keeps the clipped version."""
    kwargs = dict(clip_reward=False, terminal_on_life_loss=False) if raw_rewards else {}
    env = make_atari_env(game, n_envs=n_envs, seed=seed, wrapper_kwargs=kwargs)
    return VecFrameStack(env, n_stack=4)


def record_video(model, game, seed, out_dir, prefix, video_length):
    """dump one mp4 of the current policy to out_dir"""
    env = VecVideoRecorder(
        make_env(game, 1, seed, raw_rewards=True), out_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length, name_prefix=prefix)
    obs = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs, deterministic=DETERMINISTIC)
        obs, _, done, _ = env.step(action)
        if done.any():
            obs = env.reset()
    env.close()


class VideoCallback(BaseCallback):
    """videos every video_freq steps, just to eyeball how it's doing"""

    def __init__(self, game, seed, out_dir, freq, video_length):
        super().__init__()
        self.game, self.seed, self.out_dir = game, seed, out_dir
        self.freq, self.video_length = freq, video_length

    def _on_step(self):
        # num_timesteps jumps by n_envs so we can't test == , use the modulo window
        if self.num_timesteps % self.freq < self.training_env.num_envs:
            record_video(self.model, self.game, self.seed, self.out_dir,
                         f"step{self.num_timesteps}", self.video_length)
        return True


def main():
    args = parse_args()
    # fail early, a bad checkpoint path after 20 min of queue time is painful
    if args.arch != "nature":
        if not args.checkpoint:
            raise SystemExit(
                f"arch '{args.arch}' needs pretrained weights: set extractor.checkpoint "
                f"in {args.config}, or pass --checkpoint")
        if not os.path.exists(args.checkpoint):
            raise SystemExit(
                f"checkpoint not found: {args.checkpoint}\n"
                f"  (resolved from {args.config}; override with --checkpoint)")

    name = run_name(args)
    run_dir = os.path.join("runs", name)
    video_dir = os.path.join(run_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    env = make_env(args.game, args.n_envs, args.seed)

    callbacks = []
    if args.eval_freq > 0:
        # eval_freq is in env steps but SB3 counts per-env, hence the // n_envs
        callbacks.append(EvalCallback(
            make_env(args.game, 1, args.seed + 10_000, raw_rewards=True),
            n_eval_episodes=args.n_eval_episodes,
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            log_path=run_dir, best_model_save_path=run_dir,
            deterministic=DETERMINISTIC, verbose=1))
    if args.save_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=os.path.join(run_dir, "checkpoints"), name_prefix="ppo"))
    if args.video_freq > 0:
        callbacks.append(VideoCallback(args.game, args.seed + 30_000, video_dir,
                                       args.video_freq, args.video_length))

    if args.wandb:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        # id=name + resume=allow so a requeued slurm job reattaches to the same run
        wandb.init(project=args.wandb_project, name=name, config=vars(args),
                   sync_tensorboard=True, id=name, resume="allow")
        callbacks.append(WandbCallback(verbose=1))

    resume_from = None
    if args.resume:
        found = glob.glob(os.path.join(run_dir, "checkpoints", "*.zip"))
        resume_from = max(found, key=os.path.getmtime) if found else None

    if resume_from:
        model = PPO.load(resume_from, env=env, tensorboard_log=run_dir)
        print(f"resumed from {resume_from} at {model.num_timesteps:,} steps")
    else:
        c = args.ppo
        model = PPO(
            "CnnPolicy", env,
            policy_kwargs=build_policy_kwargs(
                arch=args.arch, checkpoint=args.checkpoint, freeze=args.freeze,
                game=args.game.split("NoFrameskip")[0], env_mode=args.env_mode,
                net_arch=args.net_arch),
            # lr and clip both decay linearly to 0, same as the ppo paper
            learning_rate=get_linear_fn(float(c["learning_rate"]), 0.0, 1.0),
            clip_range=get_linear_fn(float(c["clip_range"]), 0.0, 1.0),
            n_steps=c["n_steps"], batch_size=c["batch_size"], n_epochs=c["n_epochs"],
            gamma=c["gamma"], gae_lambda=c["gae_lambda"],
            ent_coef=c["ent_coef"], vf_coef=c["vf_coef"],
            max_grad_norm=c["max_grad_norm"],
            seed=args.seed, tensorboard_log=run_dir, verbose=1)

    # sanity check: frozen runs should print a much smaller trainable count
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.policy.parameters())
    print(f"{name} | trainable {trainable/1e6:.2f}M of {total/1e6:.2f}M")

    model.learn(total_timesteps=args.timesteps, callback=callbacks or None,
                reset_num_timesteps=resume_from is None, progress_bar=True)
    model.save(os.path.join(run_dir, "final_model"))
    env.close()

    # ---- final test, 3rd split. never used for training or for picking best_model ----
    test_env = make_env(args.game, 1, args.seed + 20_000, raw_rewards=True)
    results = {"run": name, "arch": args.arch, "game": args.game, "seed": args.seed,
               "mode": "frozen" if args.freeze else "joint",
               "timesteps": model.num_timesteps,
               "n_test_episodes": args.n_test_episodes}

    # report both, sometimes best_model is a lucky eval and final is the honest one
    for label, path in [("final", os.path.join(run_dir, "final_model.zip")),
                        ("best", os.path.join(run_dir, "best_model.zip"))]:
        if not os.path.exists(path):
            continue
        m = PPO.load(path, env=test_env)
        mean, std = evaluate_policy(m, test_env, n_eval_episodes=args.n_test_episodes,
                                    deterministic=DETERMINISTIC)
        results[f"test_{label}_mean"], results[f"test_{label}_std"] = mean, std
        print(f"TEST {label:5} | {mean:.1f} +/- {std:.1f} over {args.n_test_episodes} episodes")
        if args.video_freq > 0:
            record_video(m, args.game, args.seed + 30_000, video_dir, f"{label}_model",
                         args.video_length)

    test_env.close()
    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {run_dir}/test_results.json")

    if args.wandb:
        import wandb
        wandb.log({k: v for k, v in results.items() if isinstance(v, (int, float))})
        wandb.finish()


if __name__ == "__main__":
    main()
