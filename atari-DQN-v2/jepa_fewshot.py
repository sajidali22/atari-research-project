"""
Few-shot adaptation of a pretrained env-conditioned JEPA to an unseen Atari game.

This is TD-MPC2's new-task protocol -- "for new tasks, either initialize as a
semantically similar task's embedding or as a random vector", then finetune -- with
the addition that our UNKNOWN slot provides a genuine zero-shot starting point and a
principled initialisation, which TD-MPC2 lacks.

Five arms, so the headline number is interpretable rather than merely favourable:

  zero_shot    no training at all; the UNKNOWN slot. The floor the rest must beat.
  embed        freeze everything, train ONLY this game's 256-d embedding row,
               initialised from UNKNOWN. The claim under test.
  embed_rand   same, but the row starts as a random unit vector. Isolates how much of
               `embed` came from UNKNOWN carrying a learned prior versus from simply
               having any free 256-d vector to fit.
  full         finetune the whole network. The ceiling for this data budget.
  scratch      a freshly initialised model trained on the same budget. TD-MPC2 reports
               ~2x over this in the low-data regime.

Usage:
    python jepa_fewshot.py --game Alien --n-adapt 1000 5000 20000
    python jepa_fewshot.py --game Atlantis --arms zero_shot embed --output-json fs.json
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import games as gamelib
from JEPA import PaperAccurateJEPA
from dataset_JEPA import AtariTransitionDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.path.join(config.CHECKPOINT_DIR, "vjepa_v2_best.pt"))
    p.add_argument("--data-dir", default=config.TEST_DIR)
    p.add_argument("--game", required=True, help="held-out game name, e.g. Alien")
    p.add_argument("--n-adapt", type=int, nargs="+", default=[1000, 5000, 20000],
                   help="adaptation budgets in transitions")
    p.add_argument("--arms", nargs="+",
                   default=["zero_shot", "embed", "embed_rand", "full", "scratch"])
    p.add_argument("--eval-n", type=int, default=4000, help="held-out eval transitions")
    p.add_argument("--epochs", type=int, default=4, help="passes over the adaptation subset")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr-embed", type=float, default=1e-2,
                   help="lr for embedding-only arms; a single 256-d vector tolerates and "
                        "needs a much larger step than the full network")
    p.add_argument("--lr-full", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def game_indices(dataset, game):
    """Flat dataset indices belonging to one game, in order."""
    idx = []
    start = 0
    for shard, (g, n) in enumerate(zip(dataset.file_games, dataset.lengths)):
        if g == game:
            idx.extend(range(start, start + n))
        start += n
    if not idx:
        raise ValueError(f"No shards for '{game}' in the dataset. Present: {sorted(set(dataset.file_games))}")
    return idx


def freeze_all_but_env_row(model, row):
    """Train exactly one row of env_embed and nothing else.

    Two traps, both live:

    1. An nn.Embedding produces a DENSE gradient -- zero for unused rows, but present.
       AdamW's decoupled weight decay updates every parameter that has a .grad, so
       without intervention every other game's embedding would decay while we adapt.
       Handled by putting the tensor in a weight_decay=0 group.
    2. Momentum. Even with a zero gradient on other rows, Adam's exponential averages
       carry over from any earlier step. Handled by masking the gradient to the target
       row before the optimizer ever sees it.
    """
    for p in model.parameters():
        p.requires_grad = False
    w = model.env_embed.weight
    w.requires_grad = True

    mask = torch.zeros_like(w)
    mask[row] = 1.0
    handle = w.register_hook(lambda g: g * mask)
    return w, handle


@torch.no_grad()
def evaluate(model, loader, device, env_id):
    """JEPA prediction quality on held-out transitions of the target game."""
    model.eval()
    tot_mse, tot_sim, tot_adv, n = 0.0, 0.0, 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        ids = torch.full_like(batch["g_t"], env_id)
        e = model.env_embed(ids)
        z0 = model.context_encoder(batch["s_t"], e)
        z_tgt = model.target_encoder(batch["s_next"], e)
        z_pred = model.predictor(z0, batch["a_t"], e)
        sim = F.cosine_similarity(z_pred.flatten(1), z_tgt.flatten(1), dim=-1)
        ident = F.cosine_similarity(z0.flatten(1), z_tgt.flatten(1), dim=-1)
        b = batch["s_t"].shape[0]
        tot_mse += F.mse_loss(z_pred, z_tgt).item() * b
        tot_sim += sim.mean().item() * b
        tot_adv += (sim - ident).mean().item() * b
        n += b
    return dict(jepa_mse=tot_mse / n, pred_sim=tot_sim / n, advantage=tot_adv / n)


def build_model(ckpt, device, fresh=False):
    model = PaperAccurateJEPA(**ckpt["model_config"]).to(device)
    if not fresh:
        model.load_state_dict(ckpt["model_state_dict"])
    return model


def run_arm(arm, ckpt, adapt_loader, eval_loader, device, args, target_row):
    torch.manual_seed(args.seed)
    model = build_model(ckpt, device, fresh=(arm == "scratch"))

    if arm == "zero_shot":
        return evaluate(model, eval_loader, device, gamelib.UNKNOWN_ID), 0

    # Initialise the target row.
    with torch.no_grad():
        if arm == "embed":
            # From UNKNOWN: the learned generic-Atari prior.
            model.env_embed.weight[target_row] = model.env_embed.weight[gamelib.UNKNOWN_ID].clone()
        elif arm in ("embed_rand", "scratch", "full"):
            if arm == "embed_rand":
                v = torch.randn(model.embed_dim, device=device)
                model.env_embed.weight[target_row] = v / v.norm()
            elif arm == "full":
                model.env_embed.weight[target_row] = model.env_embed.weight[gamelib.UNKNOWN_ID].clone()

    handle = None
    if arm in ("embed", "embed_rand"):
        w, handle = freeze_all_but_env_row(model, target_row)
        # weight_decay=0: see freeze_all_but_env_row's docstring, trap 1.
        optimizer = torch.optim.AdamW([w], lr=args.lr_embed, weight_decay=0.0)
    else:
        for p in model.parameters():
            p.requires_grad = True
        for p in model.target_encoder.parameters():
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr_full, weight_decay=1e-5)

    env_id = target_row
    steps = 0
    model.train()
    for _ in range(args.epochs):
        for batch in adapt_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            ids = torch.full_like(batch["g_t"], env_id)
            optimizer.zero_grad(set_to_none=True)

            e = model.env_embed(ids)
            z0 = model.context_encoder(batch["s_t"], e)
            with torch.no_grad():
                z_tgt = model.target_encoder(batch["s_next"], e.detach())
            z_pred = model.predictor(z0, batch["a_t"], e)

            mse = F.mse_loss(z_pred, z_tgt, reduction="none").mean(dim=[1, 2])
            m = batch["mask"]
            loss = (mse * m).sum() / m.sum().clamp(min=1.0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            if arm in ("full", "scratch"):
                model.update_target_network()
            steps += 1

    if handle is not None:
        handle.remove()
    return evaluate(model, eval_loader, device, env_id), steps


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.game not in gamelib.GAME_TO_ID:
        raise SystemExit(f"Unknown game '{args.game}'. Known: {sorted(gamelib.GAME_TO_ID)}")
    target_row = gamelib.GAME_TO_ID[args.game]
    if target_row < gamelib.UNKNOWN_ID:
        print(f"NOTE: '{args.game}' is a TRAINING game (slot {target_row}), not held out. "
              f"Its embedding is already trained, so these numbers measure re-adaptation, "
              f"not few-shot transfer.")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_config" not in ckpt:
        raise SystemExit(f"{args.checkpoint} is a v1 checkpoint (no model_config).")
    print(f"Checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch','?')})")
    print(f"Target game '{args.game}' -> env slot {target_row}, initialised from "
          f"UNKNOWN (slot {gamelib.UNKNOWN_ID})\n")

    dataset = AtariTransitionDataset(args.data_dir)
    idx = game_indices(dataset, args.game)
    rng = np.random.RandomState(args.seed)

    # Adaptation and evaluation subsets are disjoint AND non-adjacent: consecutive
    # Atari transitions share 3 of 4 stacked frames, so an interleaved split would
    # leak almost the entire eval set into the adaptation set.
    split = len(idx) - args.eval_n
    if split < max(args.n_adapt):
        raise SystemExit(f"'{args.game}' has {len(idx)} transitions; not enough for "
                         f"eval_n={args.eval_n} plus n_adapt={max(args.n_adapt)}.")
    adapt_pool, eval_pool = idx[:split], idx[split:]
    eval_loader = DataLoader(Subset(dataset, eval_pool), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    print(f"{args.game}: {len(idx)} transitions -> adapt pool {len(adapt_pool)}, "
          f"eval {len(eval_pool)} (disjoint, non-adjacent)\n")

    results = {"game": args.game, "env_slot": target_row, "checkpoint": args.checkpoint, "runs": []}

    # zero_shot does not depend on the budget, so it is measured once.
    if "zero_shot" in args.arms:
        m, _ = run_arm("zero_shot", ckpt, None, eval_loader, device, args, target_row)
        print(f"{'zero_shot':<12} {'(no adaptation)':>10}  "
              f"mse={m['jepa_mse']:.6f}  sim={m['pred_sim']:.4f}  adv={m['advantage']:+.4f}")
        results["runs"].append(dict(arm="zero_shot", n_adapt=0, steps=0, **m))
        baseline = m["jepa_mse"]
    else:
        baseline = None

    for n in args.n_adapt:
        print()
        sel = rng.choice(adapt_pool, size=min(n, len(adapt_pool)), replace=False)
        adapt_loader = DataLoader(Subset(dataset, sel.tolist()), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers, drop_last=False)
        for arm in args.arms:
            if arm == "zero_shot":
                continue
            m, steps = run_arm(arm, ckpt, adapt_loader, eval_loader, device, args, target_row)
            rel = ""
            if baseline:
                rel = f"  ({(baseline - m['jepa_mse']) / baseline * 100:+.1f}% vs zero-shot)"
            print(f"{arm:<12} n={n:<8}  mse={m['jepa_mse']:.6f}  sim={m['pred_sim']:.4f}  "
                  f"adv={m['advantage']:+.4f}  [{steps} steps]{rel}")
            results["runs"].append(dict(arm=arm, n_adapt=int(n), steps=steps, **m))

    print("\nRead the arms against each other, not in isolation:")
    print("  embed vs zero_shot   -> did adapting a single 256-d vector buy anything?")
    print("  embed vs embed_rand  -> did the UNKNOWN initialisation carry a useful prior,")
    print("                          or was any free vector equally good?")
    print("  embed vs full        -> how much of the achievable gain is reachable without")
    print("                          touching the 12M-parameter backbone?")
    print("  full  vs scratch     -> the TD-MPC2 headline comparison (~2x in low data).")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
