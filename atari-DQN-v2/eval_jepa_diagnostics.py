"""
Rigorous JEPA evaluation harness.

Implements the evidentiary standard from THEORETICAL_VALIDATION_DEBATE.md (Sec 3.4),
which found that eval_JEPA_new.py's numbers cannot yet distinguish a working world
model from two specific degenerate solutions:

  (a) representation collapse (Sec 2.1) -- all embeddings converge toward a
      near-constant vector, which trivially yields high cosine similarity everywhere
      regardless of prediction quality;
  (b) the lazy-identity / frame-stack shortcut (Sec 2.2, D1) -- 3 of 4 stacked
      frames are bit-identical between s_t and s_{t+1}, so copying z_t forward
      already scores well without any dynamics understanding.

eval_JEPA_new.py already added the identity-baseline control at horizon 1. This
script fills the remaining gaps the debate names as missing instrumentation:

  D2/D3  action-index collision across games (Breakout=4 actions, Seaquest=18, ...
         folded into one 18-way embedding) contaminates any counterfactual test
         that cycles blindly through all 17 "wrong" actions -- most of which were
         never valid for that sample's game. Fixed here by restricting wrong-action
         sampling to each sample's actual per-game action count.
  D9     extract_unbroken_sequences only checks the terminal mask (which fires on
         ~0.06% of steps per D6) -- a window can silently splice across a game-shard
         boundary. Fixed here by also requiring the whole window stay inside one
         source .npz file.
  D10    no representation-health statistic (embedding std / effective rank) is
         logged anywhere, so a collapsed representation would look identical to a
         healthy one in every existing metric. Added here (Sec E).
  Sec2.2 diagnostic 2 (action-conditional dispersion) and diagnostic 3 (attention
         mass the spatial tokens assign to the action token) are the two
         diagnostics the debate specifies but the pipeline never implements.
         Added here (Sec C, Sec D).

Nothing here retrains or modifies the model architecture in JEPA.py. Section D
reimplements the predictor's forward pass layer-by-layer (reusing the same loaded
weights) purely to request attention weights, since PyTorch's fused/fast attention
path used by nn.TransformerEncoderLayer never materializes them.

Usage:
    python eval_jepa_diagnostics.py
    python eval_jepa_diagnostics.py --checkpoint ../old_checkpoints/JEPA/production_checkpoints/vjepa_atari_ep10.pt
    python eval_jepa_diagnostics.py --data-dir custom_datasets_small/test --num-sequences 64
    python eval_jepa_diagnostics.py --output-json results/ep10.json
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Imported as `gamelib` because this module already uses `games` as a local variable
# for the per-sequence list of game names.
import games as gamelib
from JEPA import PaperAccurateJEPA, strip_compile_prefix
from dataset_JEPA import AtariTransitionDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Verified via: gym.make(f"ALE/{game}-v5").action_space.n  (ale-py 0.11.2, gymnasium)
# in this repo's `attari` conda env. `full_action_space=True` is never used by the
# collection scripts (atari-replay-dataset_full.py / _small.py), so every game's
# actions are its own minimal, game-specific set -- NOT a shared canonical 18-set.
# This is D2 from the debate: PaperAccurateJEPA(num_actions=18, ...) trains one
# embedding table shared across all these incompatible index spaces.
GAME_ACTION_COUNTS = {
    "Breakout": 4,
    "Qbert": 6,
    "DemonAttack": 6,
    "SpaceInvaders": 6,
    "Pong": 6,
    "BeamRider": 9,
    "Enduro": 9,
    "MsPacman": 9,
    "RoadRunner": 18,
    "Riverraid": 18,
    "Seaquest": 18,
    # held-out games, present only in custom_datasets_small/test:
    "Asteroids": 14,
    "Alien": 18,
    "Atlantis": 4,
    "IceHockey": 18,
}


def game_name_from_path(path):
    base = os.path.basename(path)
    return base.split("NoFrameskip")[0]


def verify_action_counts(games):
    """Best-effort live check against gymnasium/ale-py; falls back to the
    hardcoded table above (with a warning) if the environment isn't available."""
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
    except ImportError:
        warnings.warn(
            "gymnasium/ale-py not importable -- using hardcoded GAME_ACTION_COUNTS "
            "without live verification. Counts may be stale if the ALE version changes."
        )
        return dict(GAME_ACTION_COUNTS)

    counts = {}
    for g in games:
        expected = GAME_ACTION_COUNTS.get(g)
        try:
            env = gym.make(f"ALE/{g}-v5")
            actual = env.action_space.n
            env.close()
        except Exception as e:
            warnings.warn(f"Could not verify action count for {g}: {e}. Using hardcoded {expected}.")
            actual = expected
        if expected is not None and actual != expected:
            warnings.warn(f"Action count mismatch for {g}: hardcoded={expected}, live={actual}. Using live value.")
        counts[g] = actual if actual is not None else expected
    return counts


@torch.no_grad()
def extract_unbroken_sequences(dataset, action_counts, num_sequences=64, horizon=5, seed=None):
    """
    Like eval_JEPA_new.py's extract_unbroken_sequences, but additionally rejects
    any window whose steps do not all belong to the same source .npz file (D9:
    the original version only checked the terminal mask, which per D6 fires on
    ~0.06% of steps and cannot be trusted to catch a splice between two games'
    shards). Also records which game each sequence came from, so downstream
    diagnostics can restrict counterfactual actions to that game's valid range
    (fixes the D2 contamination in the counterfactual test).
    """
    rng = np.random.RandomState(seed)
    print(f"Extracting {num_sequences} single-game, unbroken validation sequences (horizon={horizon})...")

    s_trajectories, a_trajectories, games = [], [], []
    cumulative = dataset.cumulative_lengths
    valid_count, attempts = 0, 0

    while valid_count < num_sequences and attempts < 200_000:
        attempts += 1
        start_idx = rng.randint(0, len(dataset) - horizon - 1)

        start_file = int(np.searchsorted(cumulative, start_idx, side="right"))
        end_file = int(np.searchsorted(cumulative, start_idx + horizon, side="right"))
        if start_file != end_file:
            continue  # window straddles a game-shard boundary

        game = game_name_from_path(dataset.file_paths[start_file])
        if game not in action_counts:
            continue  # unknown game, skip rather than guess an action count

        is_valid = True
        for step in range(horizon):
            if dataset[start_idx + step]["mask"].item() == 0.0:
                is_valid = False
                break
        if not is_valid:
            continue

        s_seq, a_seq = [], []
        for step in range(horizon):
            transition = dataset[start_idx + step]
            s_seq.append(transition["s_t"])
            a_seq.append(transition["a_t"])
        final_transition = dataset[start_idx + horizon - 1]
        s_seq.append(final_transition["s_next"])

        s_trajectories.append(torch.stack(s_seq))
        a_trajectories.append(torch.stack(a_seq))
        games.append(game)
        valid_count += 1

    if valid_count < num_sequences:
        print(f"WARNING: only found {valid_count}/{num_sequences} valid single-game sequences.")

    return {
        "s_trajectory": torch.stack(s_trajectories),  # [B, horizon+1, 4, 84, 84]
        "a_trajectory": torch.stack(a_trajectories),  # [B, horizon]
        "games": games,                               # len B, one game name per sequence
    }


def load_model(checkpoint_path, device):
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # v2 checkpoints carry the exact constructor kwargs, so the architecture never has
    # to be guessed here (v1 hardcoded num_actions=18, embed_dim=256 and would silently
    # mis-load any other configuration).
    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise ValueError(
            f"{checkpoint_path} has no 'model_config' -- it is a v1 checkpoint. v1 models "
            f"have no environment embedding and cannot be evaluated by this script. Use "
            f"JEPA_v1.py with the v1 eval scripts instead."
        )
    model = PaperAccurateJEPA(**model_config).to(device)
    model.load_state_dict(strip_compile_prefix(checkpoint["model_state_dict"]))
    model.eval()

    print(f"  epoch (0-indexed): {checkpoint.get('epoch', '?')}")
    print(f"  train loss {checkpoint.get('train_loss', '?')} | "
          f"val loss {checkpoint.get('val_loss', '?')} | "
          f"val-UNK jepa {checkpoint.get('val_unk_jepa_loss', '?')}")
    print(f"  config: {model_config}")
    return model


class EnvConditioner:
    """Resolves per-sequence environment ids under a chosen mode, and caches the
    canonical-action bookkeeping every section needs.

    env_mode "true" feeds each sequence its own game's embedding; "unk" feeds the
    UNKNOWN slot, which is the zero-shot setting used for the held-out games.
    """

    def __init__(self, game_names, env_mode, device):
        self.device = device
        self.env_mode = env_mode
        self.game_names = game_names

        # True game id -- always the real one. Used for action validity, which is a
        # property of the game, not of what we told the model it was playing.
        self.true_ids = torch.tensor(
            [gamelib.GAME_TO_ID[g] for g in game_names], dtype=torch.long, device=device
        )
        if env_mode == "unk":
            self.env_ids = torch.full_like(self.true_ids, gamelib.UNKNOWN_ID)
        elif env_mode == "true":
            self.env_ids = self.true_ids
        else:
            raise ValueError(f"env_mode must be 'true' or 'unk', got {env_mode!r}")

        mask = gamelib.valid_action_mask()  # [16, 18]
        n = gamelib.NUM_ENV_SLOTS
        A = gamelib.NUM_CANONICAL_ACTIONS
        valid_list = np.zeros((n, A), dtype=np.int64)
        rank_of = np.full((n, A), -1, dtype=np.int64)
        for g in range(n):
            idxs = np.nonzero(mask[g])[0]
            valid_list[g, : len(idxs)] = idxs
            for r, a in enumerate(idxs):
                rank_of[g, a] = r
        self.valid_list = torch.from_numpy(valid_list).to(device)
        self.rank_of = torch.from_numpy(rank_of).to(device)
        self.counts = torch.from_numpy(mask.sum(1).astype(np.int64)).to(device)

    def e(self, batch_size=None):
        """Environment embedding vectors for the sequences, [B, D]."""
        ids = self.env_ids if batch_size is None else self.env_ids[:batch_size]
        return self._model.env_embed(ids)

    def bind(self, model):
        self._model = model
        return self

    def counterfactual(self, true_action, offset):
        """The `offset`-th alternative canonical action within each sequence's own
        game, plus a validity mask. Replaces v1's `(a + offset) % 18`, which is
        meaningless now that actions live in the canonical ALE-18 space where most
        indices are invalid for most games.
        """
        g = self.true_ids
        n = self.counts[g]
        rank = self.rank_of[g, true_action]
        wrong_rank = (rank + offset) % n.clamp(min=1)
        wrong = self.valid_list[g, wrong_rank]
        return wrong, offset < n


# ---------------------------------------------------------------------------
# Section A: multi-step drift, WITH a per-step identity-copy baseline
# ---------------------------------------------------------------------------
@torch.no_grad()
def section_a_drift_with_identity_baseline(model, s_traj, a_traj, horizon, cond):
    print("\n" + "=" * 60)
    print("SECTION A: MULTI-STEP DRIFT vs PER-STEP IDENTITY BASELINE")
    print("=" * 60)
    print("The debate's Sec 3.4 diagnostic 1 requires the identity control reported")
    print("alongside EVERY drift number, not just horizon 1. At k=4 the frame stacks")
    print("s_t and s_{t+4} share zero overlapping frames (D1), so a persistently high")
    print("predictor similarity there is the strongest evidence available that the")
    print("model is not just exploiting frame-stack overlap.\n")

    e = cond.e()
    z0 = model.context_encoder(s_traj[:, 0], e)
    z_pred = z0
    results = []
    for step in range(horizon):
        z_true = model.target_encoder(s_traj[:, step + 1], e)
        actions = a_traj[:, step]
        z_pred = model.predictor(z_pred, actions, e)

        pred_sim = F.cosine_similarity(z_pred.flatten(1), z_true.flatten(1), dim=-1).mean().item()
        identity_sim = F.cosine_similarity(z0.flatten(1), z_true.flatten(1), dim=-1).mean().item()
        advantage = pred_sim - identity_sim

        frames_shared = max(0, 4 - (step + 1))
        results.append(dict(step=step + 1, pred_sim=pred_sim, identity_sim=identity_sim,
                             advantage=advantage, frames_shared_with_input=frames_shared))
        flag = " <-- zero frame overlap: overlap confound eliminated" if frames_shared == 0 else ""
        print(f"  step {step+1} | predictor={pred_sim:.4f}  identity={identity_sim:.4f}  "
              f"advantage={advantage:+.4f}  (shares {frames_shared}/4 frames with input){flag}")
    return results


# ---------------------------------------------------------------------------
# Section B: counterfactual action sensitivity, restricted to each sample's
# actual per-game action range (fixes D2 contamination)
# ---------------------------------------------------------------------------
@torch.no_grad()
def section_b_counterfactual_action_sensitivity(model, s_traj, a_traj, games, action_counts, device, cond):
    print("\n" + "=" * 60)
    print("SECTION B: COUNTERFACTUAL ACTION SENSITIVITY (per-game action range)")
    print("=" * 60)
    print("eval_JEPA_new.py tests all 17 offsets mod 18 regardless of game. For a")
    print("Breakout sequence (4 valid actions) that means 14 of 17 'wrong' actions")
    print("were never valid for Breakout at all -- borrowed embedding rows trained on")
    print("other games entirely (D2). This restricts wrong-action sampling to each")
    print("sequence's own game's valid action count.\n")

    # v2 note: actions are canonical ALE-18 indices, so `(a + offset) % n_actions` no
    # longer names a valid alternative -- for Breakout the valid canonical set is
    # {0,1,3,4} and index 2 (UP) is not reachable. cond.counterfactual walks each
    # sequence's own valid canonical action list instead.
    counts_tensor = cond.counts[cond.true_ids]
    max_actions = int(counts_tensor.max().item())

    e = cond.e()
    z_initial = model.context_encoder(s_traj[:, 0], e)
    true_action = a_traj[:, 0]
    true_next_flat = model.target_encoder(s_traj[:, 1], e).flatten(1)

    z_pred_true = model.predictor(z_initial, true_action, e).flatten(1)
    sim_true = F.cosine_similarity(z_pred_true, true_next_flat, dim=-1)

    wrong_sims, wrong_preds, valid_masks = [], [], []
    for offset in range(1, max_actions):
        wrong_action, valid = cond.counterfactual(true_action, offset)
        z_pred_wrong = model.predictor(z_initial, wrong_action, e).flatten(1)
        sims = F.cosine_similarity(z_pred_wrong, true_next_flat, dim=-1)
        wrong_sims.append(sims)
        wrong_preds.append(z_pred_wrong)
        valid_masks.append(valid)

    stacked_sims = torch.stack(wrong_sims, dim=0)          # [num_offsets, B]
    stacked_mask = torch.stack(valid_masks, dim=0).float()  # [num_offsets, B]
    n_valid_per_sample = stacked_mask.sum(dim=0).clamp(min=1)

    avg_sim_wrong = ((stacked_sims * stacked_mask).sum(dim=0) / n_valid_per_sample)
    masked_for_min = stacked_sims.clone()
    masked_for_min[stacked_mask == 0] = float("inf")
    min_sim_wrong = masked_for_min.min(dim=0).values

    sim_true_mean = sim_true.mean().item()
    avg_sim_wrong_mean = avg_sim_wrong.mean().item()
    min_sim_wrong_mean = min_sim_wrong.mean().item()
    avg_delta = sim_true_mean - avg_sim_wrong_mean
    max_delta = sim_true_mean - min_sim_wrong_mean

    n_offsets_tested = n_valid_per_sample.mean().item()
    print(f"  correct-action similarity      : {sim_true_mean:.4f}")
    print(f"  avg wrong-action similarity     : {avg_sim_wrong_mean:.4f}  "
          f"(avg {n_offsets_tested:.1f} valid alternatives tested per sample, range 1-{max_actions-1})")
    print(f"  max disruption (worst action)   : {min_sim_wrong_mean:.4f}")
    print(f"  avg action delta                : {avg_delta:+.4f}")
    print(f"  max action delta                : {max_delta:+.4f}")

    if avg_delta >= 0.005 or max_delta >= 0.01:
        verdict = "PASS: predictor is action-sensitive within each game's own valid action range."
    elif avg_delta > 0.001:
        verdict = "WARNING: weak, game-range-restricted action sensitivity."
    else:
        verdict = "FAIL: no action sensitivity once cross-game action-index aliasing is removed."
    print(f"  -> {verdict}")

    return dict(sim_true=sim_true_mean, avg_sim_wrong=avg_sim_wrong_mean, min_sim_wrong=min_sim_wrong_mean,
                avg_delta=avg_delta, max_delta=max_delta, avg_offsets_tested=n_offsets_tested,
                verdict=verdict, z_initial=z_initial, z_pred_true=z_pred_true, wrong_preds=wrong_preds,
                valid_masks=valid_masks)


# ---------------------------------------------------------------------------
# Section C: action-conditional dispersion (debate Sec 2.2, diagnostic 2)
# ---------------------------------------------------------------------------
@torch.no_grad()
def section_c_action_dispersion(section_b_out):
    print("\n" + "=" * 60)
    print("SECTION C: ACTION-CONDITIONAL DISPERSION")
    print("=" * 60)
    print("Debate Sec 2.2, diagnostic 2: compare how much predictions move BETWEEN")
    print("actions against how much the predictor moves the latent AT ALL. If the")
    print("action token were purely decorative, spread-across-actions would be small")
    print("relative to the total predicted displacement from z_t.\n")

    z_initial = section_b_out["z_initial"]
    z_pred_true = section_b_out["z_pred_true"]
    wrong_preds = section_b_out["wrong_preds"]
    valid_masks = section_b_out["valid_masks"]

    total_change = (z_pred_true - z_initial.flatten(1)).norm(dim=-1)  # [B]

    dispersions = []
    for z_pred_wrong, valid in zip(wrong_preds, valid_masks):
        d = (z_pred_wrong - z_pred_true).norm(dim=-1)  # [B]
        dispersions.append(torch.where(valid, d, torch.zeros_like(d)))
    stacked = torch.stack(dispersions, dim=0)
    stacked_mask = torch.stack(valid_masks, dim=0).float()
    n_valid = stacked_mask.sum(dim=0).clamp(min=1)
    action_dispersion = (stacked.sum(dim=0) / n_valid)  # [B], mean over valid wrong actions

    mean_dispersion = action_dispersion.mean().item()
    mean_total_change = total_change.mean().item()
    ratio = mean_dispersion / (mean_total_change + 1e-8)

    print(f"  mean ||p(z,a) - p(z,a')|| across actions : {mean_dispersion:.4f}")
    print(f"  mean ||p(z,a) - z||  (total predicted move): {mean_total_change:.4f}")
    print(f"  dispersion / total-change ratio            : {ratio:.4f}")
    if ratio < 0.05:
        print("  -> WARNING: action identity explains under 5% of how far the predictor")
        print("     moves the latent -- consistent with the action token being decorative.")
    else:
        print(f"  -> Action identity accounts for {ratio*100:.1f}% of the predictor's total")
        print("     displacement magnitude.")
    return dict(mean_dispersion=mean_dispersion, mean_total_change=mean_total_change, ratio=ratio)


# ---------------------------------------------------------------------------
# Section D: attention mass on the action token (debate Sec 2.2, diagnostic 3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _predictor_forward_with_attention(predictor, context_tokens, action, e):
    """
    Reimplements TransformerPredictor.forward layer-by-layer to request real
    attention weights. nn.TransformerEncoderLayer's forward calls self_attn with
    need_weights=False internally to use PyTorch's fused/fast attention path,
    which never materializes a weights tensor -- a plain forward hook cannot
    recover it. This calls each layer's own self_attn submodule directly with
    need_weights=True, average_attn_weights=False, reusing the exact same
    (already-loaded) weights, so results are unaffected by this rewrite; only the
    internal attention path differs.
    """
    a_token = predictor.action_embed(action).unsqueeze(1)          # [B, 1, D]
    x = torch.cat([e.unsqueeze(1), a_token, context_tokens], 1)    # [B, 38, D]

    per_layer_attn = []
    for layer in predictor.blocks.layers:
        normed = layer.norm1(x)
        attn_out, attn_weights = layer.self_attn(
            normed, normed, normed, need_weights=True, average_attn_weights=False
        )  # attn_weights: [B, heads, 37, 37]
        x = x + layer.dropout1(attn_out)

        normed2 = layer.norm2(x)
        ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(normed2))))
        x = x + layer.dropout2(ff)

        per_layer_attn.append(attn_weights)

    x = predictor.norm(x)
    return x[:, 2:, :], per_layer_attn


@torch.no_grad()
def section_d_attention_on_action_token(model, s_traj, a_traj, cond):
    print("\n" + "=" * 60)
    print("SECTION D: ATTENTION MASS ON THE ACTION AND ENVIRONMENT TOKENS")
    print("=" * 60)
    print("Debate Sec 2.2, diagnostic 3: 'near-zero attention mass [on the action")
    print("token] is dispositive'. v2's predictor sequence is [env, action, 36 patches],")
    print("so the uniform baseline is 1/38 = 0.0263 and the env token is measured")
    print("alongside the action token. Their relative mass says whether the predictor")
    print("leans on WHAT was pressed or merely on WHICH GAME it is playing -- the")
    print("latter would be the dynamics-side analogue of the D3 recognition shortcut.\n")

    e_vec = cond.e()
    context_tokens = model.context_encoder(s_traj[:, 0], e_vec)
    actions = a_traj[:, 0]

    try:
        _, per_layer_attn = _predictor_forward_with_attention(
            model.predictor, context_tokens, actions, e_vec)
    except AttributeError as e:
        print(f"  SKIPPED: predictor internals didn't match the expected nn.TransformerEncoderLayer "
              f"structure ({e}). This diagnostic depends on PyTorch internal attribute names and may "
              f"need updating if the torch version changed.")
        return None

    uniform_baseline = 1.0 / (context_tokens.shape[1] + 2)
    layer_results, env_results = [], []
    for i, attn_weights in enumerate(per_layer_attn):
        # mass the 36 spatial queries (positions 2:) assign to the env key (0) and the
        # action key (1)
        mass_on_env = attn_weights[:, :, 2:, 0].mean().item()
        mass_on_action = attn_weights[:, :, 2:, 1].mean().item()
        layer_results.append(mass_on_action)
        env_results.append(mass_on_env)
        flag = "" if mass_on_action > uniform_baseline * 1.5 else "  <-- at/below uniform baseline"
        print(f"  layer {i}: action token = {mass_on_action:.4f}   env token = {mass_on_env:.4f}"
              f"  (uniform would be {uniform_baseline:.4f}){flag}")

    avg_across_layers = float(np.mean(layer_results))
    avg_env = float(np.mean(env_results))
    print(f"\n  average across layers: action {avg_across_layers:.4f}  env {avg_env:.4f}  "
          f"(ratio env/action = {avg_env / max(avg_across_layers, 1e-9):.2f}x)")
    if avg_env > avg_across_layers * 2:
        print("  -> NOTE: the predictor attends to the environment token substantially more")
        print("     than to the action token. It may be modelling per-game average dynamics")
        print("     rather than the consequences of the specific action taken.")
    if avg_across_layers <= uniform_baseline * 1.2:
        print(f"\n  -> WARNING: average attention mass on the action token ({avg_across_layers:.4f}) is "
              f"close to or below the uniform baseline ({uniform_baseline:.4f}). Spatial tokens are not "
              f"preferentially attending to the action.")
    else:
        print(f"\n  -> Spatial tokens attend to the action token at {avg_across_layers/uniform_baseline:.2f}x "
              f"the uniform baseline rate on average.")
    return dict(per_layer=layer_results, per_layer_env=env_results,
                uniform_baseline=uniform_baseline, avg_across_layers=avg_across_layers,
                avg_env_across_layers=avg_env)


# ---------------------------------------------------------------------------
# Section E: representation health -- per-dim std and effective rank (D10)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _effective_rank_and_std(z):
    """z: [N, D] flattened embeddings (batch*tokens collapsed onto N)."""
    z = z - z.mean(dim=0, keepdim=True)
    per_dim_std = z.std(dim=0)
    # Effective rank via participation ratio of the covariance eigenvalues,
    # computed through SVD of the centered feature matrix for numerical stability.
    try:
        s = torch.linalg.svdvals(z.float())
        eigenvalues = s ** 2
        eff_rank = (eigenvalues.sum() ** 2 / (eigenvalues ** 2).sum()).item()
    except Exception:
        eff_rank = float("nan")
    return per_dim_std.mean().item(), per_dim_std.min().item(), eff_rank


@torch.no_grad()
def section_e_representation_health(model, s_traj, cond):
    print("\n" + "=" * 60)
    print("SECTION E: REPRESENTATION HEALTH (embedding std / effective rank)")
    print("=" * 60)
    print("D10: no representation-health statistic is logged anywhere in the training")
    print("or eval pipeline, so a fully collapsed representation (z -> constant) would")
    print("present as an excellent, high-similarity result in every test above. This")
    print("is the single cheapest measurement the debate names as decisive (Sec 4.3,")
    print("item 3) -- it distinguishes 'accurate predictions' from 'everything looks")
    print(f"similar because the embedding space has collapsed'. Full width = {256} dims.\n")

    e = cond.e()
    context_out = model.context_encoder(s_traj[:, 0], e)   # [B, 36, 256]
    target_out = model.target_encoder(s_traj[:, 0], e)     # [B, 36, 256]

    ctx_flat = context_out.reshape(-1, context_out.shape[-1])
    tgt_flat = target_out.reshape(-1, target_out.shape[-1])

    ctx_std_mean, ctx_std_min, ctx_rank = _effective_rank_and_std(ctx_flat)
    tgt_std_mean, tgt_std_min, tgt_rank = _effective_rank_and_std(tgt_flat)

    # UNTRAINED CONTROL. Measured, not assumed: on real Atari frames a randomly
    # initialised model already scores single-digit effective rank here, because most
    # 84x84 patches are empty black background and transformer token embeddings are
    # strongly anisotropic at init. Without this control the absolute thresholds below
    # fire on a network that has never seen a gradient, so "low" cannot be read as
    # "collapsed" -- only a value well BELOW the untrained control can.
    ref = PaperAccurateJEPA(**{**_model_config_of(model)}).to(s_traj.device).eval()
    with torch.no_grad():
        ref_out = ref.context_encoder(s_traj[:, 0], ref.env_embed(cond.env_ids))
    ref_std_mean, _, ref_rank = _effective_rank_and_std(ref_out.reshape(-1, ref_out.shape[-1]))

    print(f"  context_encoder | mean per-dim std={ctx_std_mean:.4f}  min per-dim std={ctx_std_min:.4f}  "
          f"effective rank={ctx_rank:.1f} / 256")
    print(f"  target_encoder  | mean per-dim std={tgt_std_mean:.4f}  min per-dim std={tgt_std_min:.4f}  "
          f"effective rank={tgt_rank:.1f} / 256")
    print(f"  UNTRAINED ref   | mean per-dim std={ref_std_mean:.4f}  "
          f"{'':<19}effective rank={ref_rank:.1f} / 256   <-- read the two rows above against THIS")

    for name, std_mean, rank in [("context_encoder", ctx_std_mean, ctx_rank), ("target_encoder", tgt_std_mean, tgt_rank)]:
        if rank < ref_rank * 0.5:
            print(f"  -> WARNING: {name} effective rank ({rank:.1f}) is less than half the untrained "
                  f"reference ({ref_rank:.1f}) -- training actively destroyed representational spread. "
                  f"This is the dimensional-collapse signature (Jing et al., 2022, debate T2).")
        if std_mean < ref_std_mean * 0.5:
            print(f"  -> WARNING: {name} mean per-dim std ({std_mean:.4f}) is less than half the untrained "
                  f"reference ({ref_std_mean:.4f}) -- consistent with the gamma-collapse mechanism in "
                  f"debate Sec 2.1 (LayerNorm gain -> 0).")

    return dict(context_std_mean=ctx_std_mean, context_std_min=ctx_std_min, context_eff_rank=ctx_rank,
                target_std_mean=tgt_std_mean, target_std_min=tgt_std_min, target_eff_rank=tgt_rank,
                untrained_ref_std_mean=ref_std_mean, untrained_ref_eff_rank=ref_rank)


def _model_config_of(model):
    """Reconstruct the constructor kwargs of a live model, for the untrained control."""
    return dict(
        num_actions=model.num_actions,
        num_env_slots=model.num_env_slots,
        embed_dim=model.embed_dim,
        tau=model.tau,
        use_simnorm=not isinstance(model.context_encoder.simnorm, torch.nn.Identity),
        simnorm_V=getattr(model.context_encoder.simnorm, "V", 8),
        enc_depth=len(model.context_encoder.blocks.layers),
        pred_depth=len(model.predictor.blocks.layers),
        env_dropout_p=model.env_dropout_p,
    )


# ---------------------------------------------------------------------------
# Section F: unseen-game behaviour under the UNKNOWN environment slot
# ---------------------------------------------------------------------------
@torch.no_grad()
def section_f_unseen_games(model, s_traj, a_traj, game_names, device):
    """Zero-shot behaviour on whatever games are loaded, under three conditionings.

    The comparison that matters is not "how good is the model on unseen games" in
    isolation -- it is the gap decomposition:

      true vs unk   on SEEN games   = the cost of being denied the game id
      unk           on UNSEEN games = the cost of the game id being unavailable at all
      unk vs random on UNSEEN games = whether the UNKNOWN slot learned anything, or is
                                      just an arbitrary vector

    Without the third column a good UNKNOWN number proves nothing: any fixed embedding
    would produce some number, and the question is whether THIS one carries a learned
    generic-Atari prior.
    """
    print("\n" + "=" * 60)
    print("SECTION F: UNSEEN-GAME / UNKNOWN-SLOT BEHAVIOUR")
    print("=" * 60)
    print("TD-MPC2 conditions every component on a learnable task embedding but states")
    print("it 'requires training on all target tasks simultaneously rather than enabling")
    print("zero-shot generalization to completely unseen tasks'. The UNKNOWN slot is the")
    print("mechanism meant to close that gap; this section is the test of whether it did.\n")

    seen = [g for g in set(game_names) if gamelib.GAME_TO_ID[g] < gamelib.UNKNOWN_ID]
    unseen = [g for g in set(game_names) if gamelib.GAME_TO_ID[g] > gamelib.UNKNOWN_ID]
    print(f"  seen games in this split   : {sorted(seen) or '(none)'}")
    print(f"  unseen games in this split : {sorted(unseen) or '(none)'}\n")

    ids_true = torch.tensor([gamelib.GAME_TO_ID[g] for g in game_names], device=device)
    ids_unk = torch.full_like(ids_true, gamelib.UNKNOWN_ID)

    # A random unit vector in the same space -- the control for "did UNKNOWN learn?".
    gen = torch.Generator(device="cpu").manual_seed(0)
    rand_e = torch.randn(model.embed_dim, generator=gen).to(device)
    rand_e = (rand_e / rand_e.norm()).unsqueeze(0).expand(len(game_names), -1)

    variants = {
        "true env id": model.env_embed(ids_true),
        "UNKNOWN slot": model.env_embed(ids_unk),
        "random vector": rand_e,
    }

    rows = {}
    for label, e in variants.items():
        z0 = model.context_encoder(s_traj[:, 0], e)
        z_true = model.target_encoder(s_traj[:, 1], e)
        z_pred = model.predictor(z0, a_traj[:, 0], e)
        mse = F.mse_loss(z_pred, z_true, reduction="none").mean(dim=[1, 2])
        sim = F.cosine_similarity(z_pred.flatten(1), z_true.flatten(1), dim=-1)
        ident = F.cosine_similarity(z0.flatten(1), z_true.flatten(1), dim=-1)
        rows[label] = dict(
            jepa_mse=mse.mean().item(), pred_sim=sim.mean().item(),
            identity_sim=ident.mean().item(), advantage=(sim - ident).mean().item(),
        )

    print(f"  {'conditioning':<16} {'JEPA mse':>10} {'pred sim':>10} {'identity':>10} {'advantage':>11}")
    for label, r in rows.items():
        print(f"  {label:<16} {r['jepa_mse']:>10.6f} {r['pred_sim']:>10.4f} "
              f"{r['identity_sim']:>10.4f} {r['advantage']:>+11.4f}")

    per_game = {}
    names = np.array(game_names)
    for g in sorted(set(game_names)):
        sel = torch.from_numpy((names == g).nonzero()[0]).to(device)
        if len(sel) == 0:
            continue
        e_unk = model.env_embed(torch.full((len(sel),), gamelib.UNKNOWN_ID, device=device))
        z0 = model.context_encoder(s_traj[sel, 0], e_unk)
        z_true = model.target_encoder(s_traj[sel, 1], e_unk)
        z_pred = model.predictor(z0, a_traj[sel, 0], e_unk)
        per_game[g] = dict(
            n=len(sel),
            jepa_mse=F.mse_loss(z_pred, z_true).item(),
            pred_sim=F.cosine_similarity(z_pred.flatten(1), z_true.flatten(1), dim=-1).mean().item(),
            seen=gamelib.GAME_TO_ID[g] < gamelib.UNKNOWN_ID,
        )

    print(f"\n  Per-game, under UNKNOWN:")
    print(f"  {'game':<16} {'n':>4} {'JEPA mse':>10} {'pred sim':>10}   status")
    for g, r in sorted(per_game.items(), key=lambda kv: (not kv[1]["seen"], kv[0])):
        print(f"  {g:<16} {r['n']:>4} {r['jepa_mse']:>10.6f} {r['pred_sim']:>10.4f}   "
              f"{'seen in training' if r['seen'] else 'HELD OUT'}")

    unk_vs_rand = rows["random vector"]["jepa_mse"] - rows["UNKNOWN slot"]["jepa_mse"]
    if unk_vs_rand <= 0:
        print(f"\n  -> WARNING: the UNKNOWN slot is no better than an arbitrary random vector "
              f"(delta {unk_vs_rand:+.6f}). It has not learned a generic-Atari prior, so the "
              f"zero-shot claim is not supported.")
    else:
        print(f"\n  -> UNKNOWN beats a random vector by {unk_vs_rand:.6f} MSE, so the slot carries "
              f"a learned prior rather than acting as an arbitrary constant.")

    return dict(by_conditioning=rows, by_game=per_game, unknown_vs_random=unk_vs_rand)


# ---------------------------------------------------------------------------
# Section G: geometry of the learned environment embedding space
# ---------------------------------------------------------------------------
@torch.no_grad()
def section_g_env_embedding_geometry(model, output_prefix=None):
    """Cosine-similarity matrix and 2-D projection of the learned env embeddings.

    TD-MPC2's Fig. 7 finds that task-embedding similarity tracks task DYNAMICS
    (embodiment, objects) rather than objective. The Atari analogue would be shooters
    clustering with shooters and maze games with maze games. This is the check.
    """
    print("\n" + "=" * 60)
    print("SECTION G: ENVIRONMENT EMBEDDING GEOMETRY")
    print("=" * 60)
    print("TD-MPC2 Fig. 7: 'embedding similarity appears to align more closely with task")
    print("dynamics (embodiment, objects) than objective'. If that holds here, games with")
    print("similar mechanics should sit close together in this matrix.\n")

    W = model.env_embed.weight.detach().float()
    Wn = W / W.norm(dim=1, keepdim=True).clamp_min(1e-9)
    sim = (Wn @ Wn.T).cpu().numpy()

    labels = [gamelib.ID_TO_GAME.get(i, f"slot{i}") for i in range(W.shape[0])]
    trained = list(range(gamelib.NUM_TRAIN_GAMES)) + [gamelib.UNKNOWN_ID]

    print(f"  {'':<15}" + "".join(f"{l[:6]:>7}" for l in [labels[i] for i in trained]))
    for i in trained:
        row = "".join(f"{sim[i, j]:>7.2f}" for j in trained)
        print(f"  {labels[i]:<15}{row}")

    print(f"\n  norms: " + ", ".join(f"{labels[i]}={W[i].norm():.3f}" for i in trained))

    # Nearest neighbour of each training game, and of UNKNOWN.
    print("\n  Nearest neighbour by cosine similarity:")
    for i in trained:
        others = [j for j in trained if j != i]
        j = max(others, key=lambda j: sim[i, j])
        print(f"    {labels[i]:<15} -> {labels[j]:<15} ({sim[i, j]:+.3f})")

    unk = gamelib.UNKNOWN_ID
    mean_to_unk = float(np.mean([sim[unk, i] for i in range(gamelib.NUM_TRAIN_GAMES)]))
    print(f"\n  mean cos(UNKNOWN, training game) = {mean_to_unk:+.3f}")
    print("    Near +1 would mean UNKNOWN collapsed onto the average game embedding;")
    print("    near 0 would mean it occupies its own direction in the space.")

    out = dict(similarity_matrix=sim.tolist(), labels=labels, mean_cos_unknown_to_games=mean_to_unk)

    if output_prefix:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 7))
            sub = sim[np.ix_(trained, trained)]
            im = ax.imshow(sub, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(trained)), [labels[i] for i in trained], rotation=90)
            ax.set_yticks(range(len(trained)), [labels[i] for i in trained])
            fig.colorbar(im, label="cosine similarity")
            ax.set_title("Learned environment embedding similarity")
            fig.tight_layout()
            path = f"{output_prefix}_env_embedding_cosine.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"\n  Wrote {path}")
            out["figure"] = path
        except Exception as exc:
            print(f"\n  (figure skipped: {exc})")

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=os.path.join(
        SCRIPT_DIR, "production_checkpoints", "vjepa_v2_best.pt"))
    parser.add_argument("--env-mode", default="true", choices=["true", "unk"],
                        help="conditioning for Sections A-E: 'true' feeds each sequence its own "
                             "game embedding, 'unk' feeds the UNKNOWN slot (zero-shot). Section F "
                             "always reports both regardless of this flag.")
    parser.add_argument("--data-dir", default=os.path.join(SCRIPT_DIR, "custom_datasets_small", "val"),
                         help="Directory of .npz transition files. Pass "
                              "custom_datasets_small/test to check generalization to the 4 held-out games.")
    parser.add_argument("--num-sequences", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None, help="Optional path to dump all results as JSON.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Mounting validation dataset...")
    dataset = AtariTransitionDataset(args.data_dir)

    games_present = sorted({game_name_from_path(p) for p in dataset.file_paths})
    print(f"Games found in {args.data_dir}: {games_present}")
    action_counts = verify_action_counts(games_present)
    unknown = [g for g in games_present if g not in action_counts]
    if unknown:
        warnings.warn(f"No action-count entry for {unknown} -- sequences from these games will be skipped.")

    val_batch = extract_unbroken_sequences(
        dataset, action_counts, num_sequences=args.num_sequences, horizon=args.horizon, seed=args.seed
    )
    print(f"Sequences by game: { {g: val_batch['games'].count(g) for g in set(val_batch['games'])} }")

    s_traj = val_batch["s_trajectory"].to(device)
    a_traj = val_batch["a_trajectory"].to(device)
    games = val_batch["games"]

    model = load_model(args.checkpoint, device)

    cond = EnvConditioner(games, args.env_mode, device).bind(model)
    print(f"\nSections A-E run under env-mode '{args.env_mode}'"
          + (" (UNKNOWN slot -- zero-shot conditioning)" if args.env_mode == "unk" else ""))

    results = {"env_mode": args.env_mode}
    results["drift_and_identity"] = section_a_drift_with_identity_baseline(
        model, s_traj, a_traj, args.horizon, cond)
    section_b_out = section_b_counterfactual_action_sensitivity(
        model, s_traj, a_traj, games, action_counts, device, cond)
    results["counterfactual_action_sensitivity"] = {k: v for k, v in section_b_out.items()
                                                     if k not in ("z_initial", "z_pred_true", "wrong_preds", "valid_masks")}
    results["action_dispersion"] = section_c_action_dispersion(section_b_out)
    results["attention_on_action_token"] = section_d_attention_on_action_token(model, s_traj, a_traj, cond)
    results["representation_health"] = section_e_representation_health(model, s_traj, cond)
    results["unseen_games"] = section_f_unseen_games(model, s_traj, a_traj, games, device)
    results["env_embedding_geometry"] = section_g_env_embedding_geometry(
        model, output_prefix=os.path.splitext(args.output_json)[0] if args.output_json else None)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("This is a diagnostic report, not a pass/fail gate. Read Sections D and E")
    print("first: if attention mass on the action token is near-uniform AND the")
    print("representation's effective rank / per-dim std is collapsed, the high")
    print("cosine similarities in Sections A-C are likely an artifact of a degenerate")
    print("embedding space rather than evidence of a working world model.")

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else str(o))
        print(f"\nWrote full results to {args.output_json}")


if __name__ == "__main__":
    main()
