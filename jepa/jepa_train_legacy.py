"""
Training loop for the environment-conditioned V-JEPA world model.

Changes over v1 beyond plumbing the game id through:

  * Env dropout. Each step, a fraction of samples have their game id replaced by
    UNKNOWN before the embedding lookup, so the UNKNOWN slot learns a generic-Atari
    prior and the encoder cannot fragment into 11 disjoint per-game subspaces.

  * Information gain, not raw cross-entropy. v1 logged inverse_action_loss alone,
    which is unreadable: a model that only recognises the game already scores
    H(a|game) = 1.886 nats against a chance of log(18) = 2.890 (finding D3). We log
    (masked chance - L_inv), so a positive number means genuine dynamics knowledge.

  * Per-game breakdown. v1 pooled 11 games into one number, so a game that never
    learned was invisible.

  * Representation health. SimNorm group entropy and effective rank are logged every
    epoch. A collapsed representation was previously indistinguishable from a healthy
    one in every metric the pipeline recorded (finding D10).

  * Validation runs twice, under true env ids and under UNKNOWN. The UNKNOWN pass on
    *seen* games is the control that separates "unseen game" from "unconditioned"
    when the held-out games are evaluated later.

Usage:
    python jepa_train.py --smoke                       # 2 epochs on the small split
    python jepa_train.py --inverse-weight 0.01         # one arm of the lambda sweep
    python jepa_train.py --data-dir custom_datasets/train --epochs 100
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import wandb

import config
import games
from JEPA import PaperAccurateJEPA
from dataset_JEPA import AtariTransitionDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None, help="train split (default: config.TRAIN_DIR)")
    p.add_argument("--val-dir", default=None, help="val split (default: config.VAL_DIR)")
    p.add_argument("--epochs", type=int, default=None,
                   help="explicit epoch count; overrides --target-hours auto-sizing")
    p.add_argument("--steps-per-epoch", type=int, default=None,
                   help="explicit steps per epoch; overrides auto-sizing")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--tau", type=float, default=0.996)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--inverse-weight", default=str(config.INVERSE_LOSS_WEIGHT),
                   help="float, or 'auto' to read the winning lambda from sweep_result.json")

    # --- wall-clock control -------------------------------------------------
    p.add_argument("--target-hours", type=float, default=None,
                   help="size the run to fill this many hours, measured from a short "
                        "throughput probe. Sets epochs, steps-per-epoch and the cosine horizon.")
    p.add_argument("--max-hours", type=float, default=None,
                   help="hard stop: finish cleanly and checkpoint at this wall-clock budget")
    p.add_argument("--probe-steps", type=int, default=200,
                   help="steps used to measure throughput for --target-hours. These are real "
                        "training steps; nothing is discarded.")

    # --- runtime ------------------------------------------------------------
    p.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "fp32"],
                   help="bf16 is the right default on H100: no GradScaler, no underflow")
    p.add_argument("--resume", nargs="?", const="__latest__", default=None,
                   help="resume from a checkpoint (default: vjepa_v2_latest.pt in --save-dir). "
                        "A no-op if the file does not exist, so the same script is safe to requeue.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep-tag", default=None,
                   help="label this run as a sweep arm; on completion it appends to "
                        "sweep_result.json so --inverse-weight auto can pick the winner")
    p.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    p.add_argument("--env-dropout", type=float, default=config.ENV_DROPOUT_P)
    p.add_argument("--simnorm-v", type=int, default=config.SIMNORM_V)
    p.add_argument("--no-simnorm", action="store_true", help="ablation arm")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-val-batches", type=int, default=None,
                   help="cap validation batches per pass (default: full sweep)")
    p.add_argument("--eval-heldout", action="store_true",
                   help="also evaluate the 4 held-out games (config.TEST_DIR) zero-shot under "
                        "UNKNOWN every epoch. This is the headline claim, so tracking it per "
                        "epoch shows whether transfer degrades as the model specialises to the "
                        "11 training games. NOTE: first run materialises a ~11 GB .npy_cache "
                        "next to custom_datasets_small/test.")
    p.add_argument("--max-heldout-batches", type=int, default=20)
    p.add_argument("--fig-every", type=int, default=10,
                   help="log the env-embedding heatmap every N epochs (0 disables)")
    p.add_argument("--save-dir", default=config.CHECKPOINT_DIR)
    p.add_argument("--run-name", default=None)
    p.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--smoke", action="store_true",
                   help="2 short epochs on the small split; for pipeline validation only")
    return p.parse_args()


def simnorm_health(z, V):
    """Group-wise max probability and entropy of a SimNorm latent.

    max_prob -> 1.0 and entropy -> 0 means the simplices have gone one-hot and the
    representation has lost capacity; max_prob -> 1/V means they are uniform and
    carry no information. Healthy training sits between the two.
    """
    g = z.view(*z.shape[:-1], z.shape[-1] // V, V)
    ent = -(g.clamp_min(1e-9).log() * g).sum(-1)
    return g.max(-1).values.mean().item(), ent.mean().item()


def _participation_ratio(x):
    x = (x - x.mean(0, keepdim=True)).float()
    s2 = torch.linalg.svdvals(x).pow(2)
    return (s2.sum().pow(2) / s2.pow(2).sum().clamp_min(1e-12)).item()


def effective_rank(z):
    """Participation ratio over all B*36 tokens. Same statistic as Section E of
    eval_jepa_diagnostics.py, kept identical for comparability.

    IMPORTANT CAVEAT, measured rather than assumed: on real Atari frames a randomly
    initialised, completely untrained model already scores ~3.4 / 256 here. Most
    84x84 Atari patches are empty black background, so the token distribution is
    intrinsically low-rank, and transformer token embeddings are strongly anisotropic
    at init. Section E's "warn below 25.6" threshold therefore fires on an untrained
    network and cannot by itself demonstrate collapse.

    Read this metric only against the init baseline logged as health/*_init, and
    alongside the dispersion statistics below -- a genuine collapse drives per-dim std
    and token spread DOWN, whereas concentrating variance onto a few dominant
    directions lowers the participation ratio while spread goes up.
    """
    return _participation_ratio(z.reshape(-1, z.shape[-1]))


def effective_rank_pooled(z):
    """Participation ratio over mean-pooled per-sample vectors -- i.e. over exactly
    the representation a downstream frozen-feature PPO would consume."""
    return _participation_ratio(z.mean(dim=1))


class WarmupCosine:
    """Per-step linear warmup then cosine decay, with a horizon fixed after auto-sizing.

    Per-step rather than per-epoch because --target-hours does not know the epoch count
    until the throughput probe has run. `total_steps` is written once, after the probe,
    which is safe because the probe finishes well inside the warmup phase where the
    schedule does not depend on the horizon at all.
    """

    def __init__(self, optimizer, base_lr, warmup_steps, eta_min=1e-5):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warmup_steps = max(1, warmup_steps)
        self.eta_min = eta_min
        self.total_steps = None
        self.step_num = 0

    def lr_at(self, step):
        if step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if self.total_steps is None:
            return self.base_lr
        prog = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return self.eta_min + (self.base_lr - self.eta_min) * cos

    def step(self):
        self.step_num += 1
        lr = self.lr_at(self.step_num)
        for group in self.opt.param_groups:
            group["lr"] = lr

    def get_last_lr(self):
        return [self.lr_at(self.step_num)]

    def state_dict(self):
        return {"total_steps": self.total_steps, "step_num": self.step_num,
                "base_lr": self.base_lr, "warmup_steps": self.warmup_steps,
                "eta_min": self.eta_min}

    def load_state_dict(self, sd):
        self.__dict__.update({k: v for k, v in sd.items() if k in
                              ("total_steps", "step_num", "base_lr", "warmup_steps", "eta_min")})


def build_param_groups(model, weight_decay):
    """Exempt LayerNorm gains, biases and every embedding from weight decay.

    Finding D4: decay applied to LayerNorm gains is an active collapse route -- driving
    gamma toward 0 drives z toward the constant beta, which is exactly the degenerate
    solution the JEPA objective rewards. v1 used a single param group, and the risk
    scales with training length; this run is ~10x longer than anything attempted so far.

    Decaying env_embed would also fight the max_norm=1 constraint, and decaying
    pos_embed shrinks the only signal telling the encoder where a patch sits on screen.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".pos_embed") or "embed.weight" in name:
            no_decay.append((name, p))
        else:
            decay.append((name, p))
    return (
        [{"params": [p for _, p in decay], "weight_decay": weight_decay},
         {"params": [p for _, p in no_decay], "weight_decay": 0.0}],
        [n for n, _ in decay], [n for n, _ in no_decay],
    )


def autosize(samples_per_sec, batch_size, seconds_left, eval_headroom=0.90):
    """Turn a measured throughput into (epochs, steps_per_epoch, total_steps).

    steps_per_epoch targets ~10-minute epochs, which bounds both how much progress a
    SLURM kill can destroy and how often validation runs.
    """
    total_steps = int(seconds_left * samples_per_sec / batch_size * eval_headroom)
    steps_per_epoch = int(min(20000, max(200, round(600 * samples_per_sec / batch_size))))
    epochs = max(4, total_steps // steps_per_epoch)
    return epochs, steps_per_epoch, epochs * steps_per_epoch


def env_embedding_stats(model, init_weight):
    """Health of the environment embedding table -- the v2 centerpiece.

    Nothing else in the pipeline would reveal a failure here. If the 11 game
    embeddings drift toward each other, environment conditioning silently becomes a
    no-op while every loss curve continues to look healthy, and the whole premise of
    the model quietly stops holding.

    mean_pairwise_cos is the metric to watch:
      ~0.0  the games occupy distinct directions -- conditioning is doing work
      ~1.0  the embeddings have collapsed onto one vector -- conditioning is inert

    cos_unknown_to_games says what UNKNOWN became. Near +1 means it degenerated into
    the average game embedding rather than a genuine "unspecified game" prior, which
    would undermine the zero-shot claim on the held-out games.
    """
    W = model.env_embed.weight.detach().float()
    G = games.NUM_TRAIN_GAMES
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-9)
    S = Wn @ Wn.T

    iu = torch.triu_indices(G, G, offset=1, device=W.device)
    pair = S[:G, :G][iu[0], iu[1]]
    trained = slice(0, G + 1)  # 11 games + UNKNOWN; held-out rows are untrained

    return {
        "mean_pairwise_cos": pair.mean().item(),
        "max_pairwise_cos": pair.max().item(),
        "min_pairwise_cos": pair.min().item(),
        "cos_unknown_to_games": S[games.UNKNOWN_ID, :G].mean().item(),
        "row_norm_min": W[trained].norm(dim=1).min().item(),
        "row_norm_max": W[trained].norm(dim=1).max().item(),
        "drift_from_init": (W - init_weight).norm(dim=1)[trained].mean().item(),
        "unknown_drift_from_init": (W[games.UNKNOWN_ID] - init_weight[games.UNKNOWN_ID]).norm().item(),
    }


def env_embedding_figure(model):
    """Cosine-similarity heatmap of the learned env embeddings, as a wandb.Image."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    W = model.env_embed.weight.detach().float().cpu()
    n = games.NUM_TRAIN_GAMES + 1
    Wn = W[:n] / W[:n].norm(dim=1, keepdim=True).clamp_min(1e-9)
    S = (Wn @ Wn.T).numpy()
    labels = [games.ID_TO_GAME[i] for i in range(n)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    fig.colorbar(im, label="cosine similarity")
    ax.set_title("Environment embedding similarity")
    fig.tight_layout()
    img = wandb.Image(fig)
    plt.close(fig)
    return img


def dispersion_stats(z):
    """Statistics that move monotonically under true collapse, unlike eff_rank."""
    flat = z.reshape(-1, z.shape[-1]).float()
    centered = flat - flat.mean(0, keepdim=True)
    return {
        "per_dim_std_mean": flat.std(0).mean().item(),
        "per_dim_std_min": flat.std(0).min().item(),
        "token_spread": centered.norm(dim=1).mean().item(),
        "mean_norm": flat.mean(0).norm().item(),
    }


def make_eval_loader(dataset, batch_size, num_workers, max_batches=None, seed=1234,
                     persistent=False):
    """Deterministic evaluation loader that stays representative when capped.

    Shards are grouped by game and the sampler is sequential, so taking the first
    `max_batches` of an unshuffled loader reads a single game -- the pooled "val loss"
    would silently become "val loss on BeamRider" and the per-game panel would cover
    one game. Instead we draw a FIXED seeded permutation of the requested size, which
    spans all games while producing the identical subset every epoch, so the curve
    stays comparable across epochs.
    """
    if max_batches is not None:
        n = min(len(dataset), max_batches * batch_size)
        idx = np.random.RandomState(seed).permutation(len(dataset))[:n]
        dataset = Subset(dataset, idx.tolist())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=persistent and num_workers > 0,
    )


AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def amp_autocast(device, amp_dtype):
    """bf16 is the H100 default: it needs no GradScaler and cannot underflow.

    That matters here specifically -- SimNorm makes the JEPA term small (~0.026 at init
    and falling), which is exactly the regime where fp16 gradients vanish.
    """
    enabled = device.type == "cuda" and amp_dtype != "fp32"
    return torch.amp.autocast("cuda", enabled=enabled, dtype=AMP_DTYPES[amp_dtype])


def run_epoch_eval(model, loader, device, env_mode, inverse_weight, amp_dtype, max_batches=None):
    """One validation pass under a fixed env mode. Returns pooled + per-game metrics."""
    model.eval()
    n_slots = games.NUM_ENV_SLOTS
    tot = dict(jepa=0.0, inv=0.0, gain=0.0, total=0.0, acc=0.0, chance_acc=0.0, n=0)
    per_game_loss = torch.zeros(n_slots, device=device)
    per_game_gain = torch.zeros(n_slots, device=device)
    per_game_acc = torch.zeros(n_slots, device=device)
    per_game_n = torch.zeros(n_slots, device=device)
    health = None

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            env_ids = model.resolve_env_ids(batch["g_t"], mode=env_mode)

            with amp_autocast(device, amp_dtype):
                z_pred, z_tgt, z_ctx, e = model(batch, return_latents=True, env_ids=env_ids)
                mse_item = F.mse_loss(z_pred, z_tgt, reduction="none").mean(dim=[1, 2])
                inv_item = model.compute_inverse_loss(z_ctx, z_tgt, batch["a_t"], e, env_ids)
                acc_item = model.inverse_correct(z_ctx, z_tgt, batch["a_t"], e, env_ids)

            m = batch["mask"]
            denom = m.sum().clamp(min=1.0)
            jepa = (mse_item * m).sum() / denom
            inv = (inv_item * m).sum() / denom
            gain_item = model.chance_inverse_loss(env_ids) - inv_item
            gain = (gain_item * m).sum() / denom

            tot["jepa"] += jepa.item(); tot["inv"] += inv.item()
            tot["gain"] += gain.item(); tot["total"] += (jepa + inverse_weight * inv).item()
            tot["acc"] += ((acc_item * m).sum() / denom).item()
            tot["chance_acc"] += ((model.chance_action_prob(env_ids) * m).sum() / denom).item()
            tot["n"] += 1

            # Per-game uses the TRUE id, so the breakdown is by game even under UNKNOWN.
            g = batch["g_t"]
            per_game_loss.scatter_add_(0, g, (mse_item * m).float())
            per_game_gain.scatter_add_(0, g, (gain_item * m).float())
            per_game_acc.scatter_add_(0, g, (acc_item * m).float())
            per_game_n.scatter_add_(0, g, m.float())

            if health is None:
                health = z_ctx.float()

    n = max(tot["n"], 1)
    out = {k: tot[k] / n for k in ("jepa", "inv", "gain", "total", "acc", "chance_acc")}
    cnt = per_game_n.clamp_min(1.0)
    out["per_game_loss"] = (per_game_loss / cnt).cpu().numpy()
    out["per_game_gain"] = (per_game_gain / cnt).cpu().numpy()
    out["per_game_acc"] = (per_game_acc / cnt).cpu().numpy()
    out["per_game_n"] = per_game_n.cpu().numpy()
    out["z_sample"] = health
    return out


SWEEP_RESULT = "sweep_result.json"
SWEEP_CRITERION = "val/inverse_information_gain (tie-break: val jepa loss)"


def record_sweep_arm(args, best_val_loss, val_true, val_unk, heldout, env_cos):
    """Append this arm's result and re-rank, so job_c can read the winner directly.

    Selection is on **information gain**, not raw inverse cross-entropy: a head that
    merely recognises the game already scores H(a|game)=1.886 against a log(18)=2.890
    floor (finding D3), so cross-entropy would reward game recognition over dynamics.

    Arms whose environment embeddings collapsed are disqualified outright -- a low loss
    with mean pairwise cosine near 1 means conditioning went inert, which is exactly the
    failure this architecture exists to avoid.
    """
    path = os.path.join(args.save_dir, SWEEP_RESULT)
    arms = []
    if os.path.exists(path):
        with open(path) as f:
            arms = json.load(f).get("arms", [])
    arms = [a for a in arms if a["tag"] != args.sweep_tag]
    arms.append({
        "tag": args.sweep_tag,
        "inverse_weight": args.inverse_weight,
        "batch_size": args.batch_size,
        "val_information_gain": val_true["gain"],
        "val_jepa_loss": val_true["jepa"],
        "val_inverse_accuracy": val_true["acc"],
        "val_inverse_chance_accuracy": val_true["chance_acc"],
        "val_unk_jepa_loss": val_unk["jepa"],
        "heldout_unk_jepa_loss": heldout["jepa"] if heldout else None,
        "env_mean_pairwise_cos": env_cos,
        "best_val_loss": best_val_loss,
        "collapsed": bool(env_cos > 0.9),
    })
    healthy = [a for a in arms if not a["collapsed"]] or arms
    healthy.sort(key=lambda a: (-a["val_information_gain"], a["val_jepa_loss"]))
    best = healthy[0]

    with open(path, "w") as f:
        json.dump({"best_inverse_weight": best["inverse_weight"],
                   "best_tag": best["tag"], "criterion": SWEEP_CRITERION,
                   "arms": sorted(arms, key=lambda a: a["tag"])}, f, indent=2)
    print(f"\nSweep arm '{args.sweep_tag}' recorded in {path}. "
          f"Current leader: lambda={best['inverse_weight']} "
          f"(gain {best['val_information_gain']:.4f} nats, jepa {best['val_jepa_loss']:.6f})")


def resolve_inverse_weight(spec, save_dir):
    """Accept a float, or 'auto' to consume the winning lambda written by the sweep."""
    if str(spec).lower() != "auto":
        return float(spec)
    path = os.path.join(save_dir, SWEEP_RESULT)
    if not os.path.exists(path):
        path = SWEEP_RESULT
    if not os.path.exists(path):
        raise SystemExit(
            f"--inverse-weight auto: no {SWEEP_RESULT} found in {save_dir} or cwd. "
            f"Run the sweep first (slurm/job_b_sweep.sh), or pass a number."
        )
    with open(path) as f:
        res = json.load(f)
    print(f"--inverse-weight auto -> {res['best_inverse_weight']} "
          f"(selected on {res.get('criterion', 'val information gain')} from {path})")
    return float(res["best_inverse_weight"])


def train_jepa():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.smoke:
        # Deliberately trains on the val split: it is the only split with a prebuilt
        # .npy_cache, and materialising one for custom_datasets_small/train would write
        # ~31 GB. Train/val therefore overlap -- this mode validates that the pipeline
        # runs and that every metric populates, and its loss numbers mean nothing.
        args.data_dir = args.data_dir or config.VAL_DIR
        args.epochs = args.epochs or 2
        args.target_hours = None  # smoke is explicitly sized; never auto-size
        args.run_name = args.run_name or "smoke-test"
        args.warmup_steps = min(args.warmup_steps, 10)
        if not torch.cuda.is_available():
            # 12M-param transformer on CPU: keep it to a couple of minutes.
            args.steps_per_epoch = args.steps_per_epoch or 20
            args.batch_size = min(args.batch_size, 32)
            args.max_val_batches = args.max_val_batches or 5
            args.num_workers = min(args.num_workers, 2)
        else:
            args.steps_per_epoch = args.steps_per_epoch or 100
            args.max_val_batches = args.max_val_batches or 20
        print(f"SMOKE MODE: train==val split, {args.epochs}x{args.steps_per_epoch} steps, "
              f"batch {args.batch_size}. Loss values are not meaningful.")

    data_dir = args.data_dir or config.TRAIN_DIR
    val_dir = args.val_dir or config.VAL_DIR
    use_simnorm = not args.no_simnorm
    os.makedirs(args.save_dir, exist_ok=True)
    args.inverse_weight = resolve_inverse_weight(args.inverse_weight, args.save_dir)
    run_name = args.run_name or f"v2-envcond-lam{args.inverse_weight}-drop{args.env_dropout}"

    # Resume: read the checkpoint before wandb.init so the run can be re-attached
    # rather than forking a second run for the same training trajectory.
    resume_path = None
    if args.resume:
        resume_path = (os.path.join(args.save_dir, "vjepa_v2_latest.pt")
                       if args.resume == "__latest__" else args.resume)
        if not os.path.exists(resume_path):
            print(f"--resume: {resume_path} not found; starting fresh.")
            resume_path = None
    ckpt_in = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path else None

    wandb.init(
        project="atari-vjepa-world-model",
        name=run_name,
        mode=args.wandb_mode,
        id=(ckpt_in or {}).get("wandb_id"),
        resume="allow" if ckpt_in and ckpt_in.get("wandb_id") else None,
        config={
            "architecture": "Env-Conditioned V-JEPA (TD-MPC2 task embedding + UNKNOWN slot)",
            "train_dir": data_dir, "val_dir": val_dir,
            "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
            "batch_size": args.batch_size, "learning_rate": args.lr,
            "weight_decay": args.weight_decay, "tau_ema": args.tau,
            "latent_dim": args.latent_dim, "max_grad_norm": 1.0, "mixed_precision": True,
            "inverse_weight": args.inverse_weight, "env_dropout_p": args.env_dropout,
            "use_simnorm": use_simnorm, "simnorm_V": args.simnorm_v,
            "num_env_slots": games.NUM_ENV_SLOTS, "unknown_id": games.UNKNOWN_ID,
            "canonical_actions": True,
        },
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = args.amp_dtype if device.type == "cuda" else "fp32"
    print(f"Launching env-conditioned V-JEPA training on: {device} (amp={amp_dtype})")

    train_dataset = AtariTransitionDataset(data_dir)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0, drop_last=True,
    )
    val_dataset = AtariTransitionDataset(val_dir)
    val_loader = make_eval_loader(
        val_dataset, args.batch_size, max(2, args.num_workers // 2),
        max_batches=args.max_val_batches, persistent=True,
    )

    # The 4 genuinely held-out games. Zero-shot performance here IS the claim, so it is
    # worth a curve rather than a single number at the end -- the interesting failure
    # mode is transfer peaking early and then decaying as the model overfits the 11
    # training games, which only a per-epoch trace can show.
    heldout_loader = None
    if args.eval_heldout:
        heldout_dataset = AtariTransitionDataset(config.TEST_DIR)
        heldout_loader = make_eval_loader(
            heldout_dataset, args.batch_size, max(2, args.num_workers // 2),
            max_batches=args.max_heldout_batches, seed=4321,
        )
        print(f"Held-out zero-shot eval enabled: {len(heldout_dataset)} transitions "
              f"across {sorted(set(heldout_dataset.file_games))}")
    else:
        print("Held-out zero-shot eval DISABLED. Enable with --eval-heldout "
              "(builds a ~11 GB cache on first run).")

    model_kwargs = dict(
        num_actions=games.NUM_CANONICAL_ACTIONS,
        num_env_slots=games.NUM_ENV_SLOTS,
        embed_dim=args.latent_dim,
        tau=args.tau,
        use_simnorm=use_simnorm,
        simnorm_V=args.simnorm_v,
        env_dropout_p=args.env_dropout,
    )
    model = PaperAccurateJEPA(**model_kwargs).to(device)

    param_groups, decay_names, no_decay_names = build_param_groups(model, args.weight_decay)
    optimizer = optim.AdamW(param_groups, lr=args.lr)
    # GradScaler only makes sense for fp16; bf16 has fp32 dynamic range.
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == "fp16"))
    scheduler = WarmupCosine(optimizer, args.lr, args.warmup_steps, eta_min=1e-5)
    print(f"Param groups: {len(decay_names)} decayed, {len(no_decay_names)} exempt "
          f"(LayerNorm gains, biases, pos_embed, env_embed, action_embed)")

    global_step = 0
    start_epoch = 0
    best_val_loss = float("inf")
    n_slots = games.NUM_ENV_SLOTS

    if ckpt_in is not None:
        model.load_state_dict(ckpt_in["model_state_dict"])
        optimizer.load_state_dict(ckpt_in["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt_in["scheduler_state_dict"])
        if ckpt_in.get("scaler_state_dict") and amp_dtype == "fp16":
            scaler.load_state_dict(ckpt_in["scaler_state_dict"])
        start_epoch = ckpt_in["epoch"] + 1
        global_step = ckpt_in.get("global_step", 0)
        best_val_loss = ckpt_in.get("best_val_loss", float("inf"))
        print(f"RESUMED from {resume_path}: epoch {start_epoch}, step {global_step}, "
              f"best_val {best_val_loss:.6f}, lr {scheduler.get_last_lr()[0]:.6g}")

    # torch.compile after loading, so the state_dict keys match the checkpoint. It
    # prefixes parameters with '_orig_mod.'; save_state_dict below unwraps it so the
    # checkpoint stays loadable by uncompiled code (eval, few-shot).
    raw_model = model
    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled (first steps will be slow while it traces)")

    wandb.watch(raw_model, log="all", log_freq=500)

    print(f"Train {len(train_dataset)} transitions | Val {len(val_dataset)} | "
          f"lambda_inv={args.inverse_weight} | env_dropout={args.env_dropout} | simnorm={use_simnorm}")

    # Representation-health baseline at initialisation, BEFORE any gradient step.
    # Without it the health numbers are uninterpretable: an untrained model already
    # scores ~3.4/256 effective rank on Atari frames, so "low" is not "collapsed".
    init_stats = run_epoch_eval(model, val_loader, device, "true", args.inverse_weight, args.amp_dtype, max_batches=4)
    init_health = {
        "effective_rank": effective_rank(init_stats["z_sample"]),
        "effective_rank_pooled": effective_rank_pooled(init_stats["z_sample"]),
        **dispersion_stats(init_stats["z_sample"]),
    }
    for k, v in init_health.items():
        wandb.run.summary[f"health_init/{k}"] = v
    print("Init health baseline (untrained reference): " +
          ", ".join(f"{k}={v:.4f}" for k, v in init_health.items()))

    # Reference copy of the env table at init, so drift is measurable rather than
    # merely absolute. Cloned before the first optimizer step.
    init_env_weight = raw_model.env_embed.weight.detach().clone()
    wandb.run.summary["params_total"] = sum(p.numel() for p in raw_model.parameters())
    wandb.run.summary["params_trainable"] = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)

    # --- run sizing -------------------------------------------------------------
    # Explicit --epochs/--steps-per-epoch win. Otherwise --target-hours sizes the run
    # from a measured throughput: the first epoch is a short probe, after which the
    # epoch count, steps-per-epoch and the cosine horizon are all fixed. The probe is
    # real training -- nothing is thrown away.
    run_start = time.time()
    autosizing = args.epochs is None and args.target_hours is not None
    if autosizing:
        total_epochs, steps_this_epoch = 10 ** 9, args.probe_steps
        print(f"Auto-sizing: probing throughput over {args.probe_steps} steps to fill "
              f"{args.target_hours:.2f} h")
    else:
        total_epochs = args.epochs if args.epochs is not None else 100
        steps_this_epoch = args.steps_per_epoch if args.steps_per_epoch is not None else 1000
        scheduler.total_steps = total_epochs * steps_this_epoch
        print(f"Fixed sizing: {total_epochs} epochs x {steps_this_epoch} steps "
              f"= {scheduler.total_steps} steps")

    def out_of_time():
        return args.max_hours is not None and (time.time() - run_start) / 3600.0 >= args.max_hours

    stop = False
    epoch = start_epoch
    while epoch < total_epochs and not stop:
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        n_steps = 0
        ep_game_loss = torch.zeros(n_slots, device=device)
        ep_game_gain = torch.zeros(n_slots, device=device)
        ep_game_n = torch.zeros(n_slots, device=device)

        desc = f"Epoch {epoch+1}/{total_epochs if total_epochs < 10**9 else '?'}"
        pbar = tqdm(train_loader, desc=desc)
        for step, batch in enumerate(pbar):
            if step >= steps_this_epoch:
                break
            if out_of_time():
                print(f"\n--max-hours {args.max_hours} reached; finishing epoch early.")
                stop = True
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Env dropout: resolved once, then used for the encoder, the target encoder
            # and the predictor, so all three see the same conditioning.
            env_ids = raw_model.resolve_env_ids(batch["g_t"], mode="auto")

            optimizer.zero_grad(set_to_none=True)
            with amp_autocast(device, amp_dtype):
                z_pred, z_tgt, z_ctx, e = model(batch, return_latents=True, env_ids=env_ids)

                mse_item = F.mse_loss(z_pred, z_tgt, reduction="none").mean(dim=[1, 2])
                m = batch["mask"]
                denom = m.sum().clamp(min=1.0)
                jepa_loss = (mse_item * m).sum() / denom

                inv_item = raw_model.compute_inverse_loss(z_ctx, z_tgt, batch["a_t"], e, env_ids)
                inv_loss = (inv_item * m).sum() / denom

                loss = jepa_loss + args.inverse_weight * inv_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # clip_grad_norm_ returns the PRE-clip total norm -- free information that
            # v1 discarded. If it sits persistently above max_norm the clip is active
            # every step, which silently rescales the effective learning rate.
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            raw_model.update_target_network()

            with torch.no_grad():
                gain_item = raw_model.chance_inverse_loss(env_ids) - inv_item
                gain = (gain_item * m).sum() / denom
                acc_item = raw_model.inverse_correct(z_ctx, z_tgt, batch["a_t"], e, env_ids)
                g = batch["g_t"]
                ep_game_loss.scatter_add_(0, g, (mse_item * m).float())
                ep_game_gain.scatter_add_(0, g, (gain_item * m).float())
                ep_game_n.scatter_add_(0, g, m.float())

            if global_step % 50 == 0:
                log = {
                    "train/total_loss": loss.item(),
                    "train/jepa_physics_loss": jepa_loss.item(),
                    "train/inverse_action_loss": inv_loss.item(),
                    "train/inverse_information_gain": gain.item(),
                    "train/inverse_chance_floor": raw_model.chance_inverse_loss(env_ids).mean().item(),
                    "train/inverse_accuracy": ((acc_item * m).sum() / denom).item(),
                    "train/inverse_chance_accuracy": ((raw_model.chance_action_prob(env_ids) * m).sum() / denom).item(),
                    "train/env_dropout_frac": (env_ids == games.UNKNOWN_ID).float().mean().item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/grad_clipped": float(grad_norm.item() > 1.0),
                    "train/loss_scale": scaler.get_scale() if amp_dtype == "fp16" else 1.0,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/global_step": global_step,
                    "train/epoch": epoch,
                }
                if use_simnorm:
                    mx, ent = simnorm_health(z_ctx.detach().float(), args.simnorm_v)
                    log["train/simnorm_max_prob"] = mx
                    log["train/simnorm_entropy"] = ent
                # The v2 centerpiece: if these embeddings converge, env conditioning
                # becomes inert while every loss curve still looks fine.
                log.update({f"env_embed/{k}": v
                            for k, v in env_embedding_stats(raw_model, init_env_weight).items()})
                wandb.log(log)

            scheduler.step()  # per-step: the horizon is not known until auto-sizing
            total_loss += loss.item()
            n_steps += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.5f}", jepa=f"{jepa_loss.item():.5f}")

        avg_loss = total_loss / max(n_steps, 1)
        epoch_seconds = time.time() - epoch_start
        samples_per_sec = n_steps * args.batch_size / max(epoch_seconds, 1e-6)

        # --- fix the run size from the measured throughput, once ---
        if autosizing:
            seconds_left = args.target_hours * 3600.0 - (time.time() - run_start)
            eps, spe, total_steps = autosize(samples_per_sec, args.batch_size, seconds_left)
            total_epochs = epoch + 1 + eps
            steps_this_epoch = spe
            scheduler.total_steps = global_step + total_steps
            autosizing = False
            print(f"\nAuto-sized from {samples_per_sec:.0f} samples/s over {n_steps} probe steps:\n"
                  f"  {eps} more epochs x {spe} steps = {total_steps:,} steps "
                  f"({total_steps * args.batch_size / 1e6:.0f}M samples)\n"
                  f"  epoch wall-clock ~{spe * args.batch_size / samples_per_sec / 60:.1f} min, "
                  f"cosine horizon {scheduler.total_steps:,} steps")
            wandb.run.summary.update({
                "autosize/samples_per_sec": samples_per_sec,
                "autosize/epochs": total_epochs,
                "autosize/steps_per_epoch": spe,
                "autosize/total_steps": scheduler.total_steps,
            })

        # --- validation: true env ids, then UNKNOWN on the same seen games ---
        # No max_batches here: make_eval_loader already sized the loader, and doing it
        # again would re-truncate the permuted subset back toward a single game.
        val_true = run_epoch_eval(model, val_loader, device, "true", args.inverse_weight, args.amp_dtype)
        val_unk = run_epoch_eval(model, val_loader, device, "unk", args.inverse_weight, args.amp_dtype)

        heldout = None
        if heldout_loader is not None:
            heldout = run_epoch_eval(model, heldout_loader, device, "unk", args.inverse_weight, args.amp_dtype)

        z_val = val_true["z_sample"]
        eff_rank = effective_rank(z_val)
        log = {f"health/{k}": v for k, v in dispersion_stats(z_val).items()}
        log.update({f"env_embed/{k}": v
                    for k, v in env_embedding_stats(raw_model, init_env_weight).items()})
        log.update({
            "train/epoch_avg_loss": avg_loss,
            "train/epoch_seconds": epoch_seconds,
            "train/samples_per_sec": n_steps * args.batch_size / max(epoch_seconds, 1e-6),
            "val/epoch_avg_loss": val_true["total"],
            "val/jepa_physics_loss": val_true["jepa"],
            "val/inverse_action_loss": val_true["inv"],
            "val/inverse_information_gain": val_true["gain"],
            "val/inverse_accuracy": val_true["acc"],
            "val/inverse_chance_accuracy": val_true["chance_acc"],
            "val_unk/epoch_avg_loss": val_unk["total"],
            "val_unk/jepa_physics_loss": val_unk["jepa"],
            "val_unk/inverse_information_gain": val_unk["gain"],
            "val_unk/inverse_accuracy": val_unk["acc"],
            # How much worse the model is when denied the game id, on games it has seen.
            # This is the control for the held-out-game experiment.
            "val/env_conditioning_benefit": val_unk["jepa"] - val_true["jepa"],
            "health/effective_rank": eff_rank,
            "health/effective_rank_pooled": effective_rank_pooled(z_val),
            "epoch": epoch,
        })
        if heldout is not None:
            log.update({
                "heldout_unk/jepa_physics_loss": heldout["jepa"],
                "heldout_unk/inverse_information_gain": heldout["gain"],
                "heldout_unk/inverse_accuracy": heldout["acc"],
                # The generalisation gap that matters: unseen games under UNKNOWN vs
                # seen games under UNKNOWN. Comparing against val/ instead would
                # conflate "unseen game" with "no game id", which is a different thing.
                "heldout_unk/gap_vs_seen_unk": heldout["jepa"] - val_unk["jepa"],
            })
            for gid in range(n_slots):
                if heldout["per_game_n"][gid] > 0:
                    name = games.ID_TO_GAME.get(gid, f"slot{gid}")
                    log[f"per_game_heldout/{name}/jepa_loss"] = float(heldout["per_game_loss"][gid])
                    log[f"per_game_heldout/{name}/info_gain"] = float(heldout["per_game_gain"][gid])
        if args.fig_every and (epoch % args.fig_every == 0 or epoch + 1 >= total_epochs):
            fig = env_embedding_figure(raw_model)
            if fig is not None:
                log["env_embed/cosine_matrix"] = fig
        if use_simnorm:
            mx, ent = simnorm_health(val_true["z_sample"], args.simnorm_v)
            log["health/simnorm_max_prob"] = mx
            log["health/simnorm_entropy"] = ent

        # Per-game train and val breakdowns.
        for gid in range(n_slots):
            name = games.ID_TO_GAME.get(gid, f"slot{gid}")
            if ep_game_n[gid] > 0:
                log[f"per_game_train/{name}/jepa_loss"] = (ep_game_loss[gid] / ep_game_n[gid]).item()
                log[f"per_game_train/{name}/info_gain"] = (ep_game_gain[gid] / ep_game_n[gid]).item()
            if val_true["per_game_n"][gid] > 0:
                log[f"per_game_val/{name}/jepa_loss"] = float(val_true["per_game_loss"][gid])
                log[f"per_game_val/{name}/info_gain"] = float(val_true["per_game_gain"][gid])
                log[f"per_game_val/{name}/inverse_accuracy"] = float(val_true["per_game_acc"][gid])
                log[f"per_game_val_unk/{name}/jepa_loss"] = float(val_unk["per_game_loss"][gid])
                # Sample counts: batches are shuffled across 11 games, so a game that
                # happens to be thinly represented produces a noisy curve. Without n
                # there is no way to tell noise from a real per-game regression.
                log[f"per_game_val/{name}/n"] = float(val_true["per_game_n"][gid])
        wandb.log(log)

        env_cos = log["env_embed/mean_pairwise_cos"]
        print(
            f"Epoch {epoch+1} | train {avg_loss:.5f} | val {val_true['total']:.5f} "
            f"(jepa {val_true['jepa']:.5f}, gain {val_true['gain']:.3f} nats, "
            f"acc {val_true['acc']:.3f} vs chance {val_true['chance_acc']:.3f}) | "
            f"val-UNK jepa {val_unk['jepa']:.5f}"
            + (f" | heldout-UNK jepa {heldout['jepa']:.5f}" if heldout else "")
            + f" | env_cos {env_cos:+.3f} | eff_rank {eff_rank:.1f} | {epoch_seconds:.0f}s"
        )
        if env_cos > 0.9:
            print("  WARNING: mean pairwise cosine between game embeddings is above 0.9 -- "
                  "the environment embeddings are collapsing onto a single direction, which "
                  "makes environment conditioning inert regardless of how the losses look.")

        # --- checkpointing ---
        # model_config is saved so eval and few-shot scripts can rebuild the exact
        # architecture instead of hardcoding a constructor call, as v1's eval scripts did.
        ckpt = {
            "epoch": epoch,
            # raw_model, never the torch.compile wrapper: compiled state_dicts carry an
            # '_orig_mod.' prefix on every key, which would break eval and few-shot.
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if amp_dtype == "fp16" else None,
            "model_config": model_kwargs,
            # Everything --resume needs to continue rather than restart.
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "wandb_id": wandb.run.id if wandb.run is not None else None,
            "train_loss": avg_loss,
            "val_loss": val_true["total"],
            "val_jepa_loss": val_true["jepa"],
            "val_unk_jepa_loss": val_unk["jepa"],
            "train_games": games.TRAIN_GAMES,
            "game_to_id": games.GAME_TO_ID,
        }
        torch.save(ckpt, os.path.join(args.save_dir, "vjepa_v2_latest.pt"))
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, os.path.join(args.save_dir, f"vjepa_v2_ep{epoch+1}.pt"))
        if val_true["total"] < best_val_loss:
            best_val_loss = val_true["total"]
            torch.save(ckpt, os.path.join(args.save_dir, "vjepa_v2_best.pt"))
            print(f"  New best validation loss: {best_val_loss:.5f} -> vjepa_v2_best.pt")
            wandb.run.summary.update({
                "best/epoch": epoch,
                "best/val_loss": val_true["total"],
                "best/val_jepa_loss": val_true["jepa"],
                "best/val_inverse_information_gain": val_true["gain"],
                "best/val_inverse_accuracy": val_true["acc"],
                "best/val_unk_jepa_loss": val_unk["jepa"],
                "best/env_mean_pairwise_cos": env_cos,
                "best/effective_rank": eff_rank,
                **({"best/heldout_unk_jepa_loss": heldout["jepa"]} if heldout else {}),
            })
            # best_val_loss changed after ckpt was built; keep latest consistent so a
            # resume does not re-declare an already-beaten loss as the new best.
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, os.path.join(args.save_dir, "vjepa_v2_latest.pt"))

        epoch += 1

    # Sweep arms record their own result so job_c can pick lambda up without a human
    # reading W&B. Ranked on information gain, not raw cross-entropy (finding D3).
    if args.sweep_tag:
        record_sweep_arm(args, best_val_loss, val_true, val_unk, heldout, env_cos)

    wandb.run.summary["final/epochs_completed"] = epoch
    wandb.run.summary["final/global_step"] = global_step
    wandb.finish()
    print(f"Training complete: {epoch} epochs, {global_step:,} steps, "
          f"{(time.time() - run_start)/3600:.2f} h.")


if __name__ == "__main__":
    train_jepa()
