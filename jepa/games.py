"""
Single source of truth for game identity and action-space canonicalisation.

Why this file exists
--------------------
The collection scripts (atari-replay-dataset_full.py / _small.py) build envs via
make_atari_env WITHOUT full_action_space=True, so every game's stored actions are
indices into that game's own *minimal* action set. Breakout has 4, Pong 6,
Seaquest 18. Stored action `3` therefore means LEFT in Breakout but RIGHT in
Seaquest. PaperAccurateJEPA folds all of them into one nn.Embedding(18, ...),
which makes that table not merely polysemous but contradictory -- finding D2 in
THEORETICAL_VALIDATION_DEBATE.md.

The fix needs no data re-collection. ALE's full action set has a fixed 18-entry
order, and every game's minimal set is a subset listed in that same order, so a
per-game lookup table maps minimal index -> canonical ALE-18 index. That LUT is
applied once at dataset load.

This module also owns the environment-embedding id space used by the JEPA model:

    ids  0..10  the 11 training games (alphabetical)
    id   11     UNKNOWN  -- the "generic Atari game" slot, trained by embedding
                           dropout so the model runs zero-shot on unseen games
    ids  12..15 the 4 held-out games (Alien, Asteroids, Atlantis, IceHockey)

The held-out rows are pre-allocated but receive no gradient during pretraining.
Reserving them up front means few-shot adaptation needs no checkpoint surgery:
the state_dict shape is identical before and after.

Building the registry requires gymnasium + ale-py; the result is cached to
game_registry.json so training and evaluation never need them.
"""

import json
import os
import warnings

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REGISTRY_PATH = os.path.join(SCRIPT_DIR, "game_registry.json")

# ALE's canonical full action set, in its fixed order.
# Verified against gym.make("ALE/Breakout-v5", full_action_space=True) on ale-py 0.11.2.
FULL_ACTION_MEANINGS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT",
    "UPFIRE", "RIGHTFIRE", "LEFTFIRE", "DOWNFIRE",
    "UPRIGHTFIRE", "UPLEFTFIRE", "DOWNRIGHTFIRE", "DOWNLEFTFIRE",
]
NUM_CANONICAL_ACTIONS = len(FULL_ACTION_MEANINGS)  # 18

# The 11 games in custom_datasets/train and custom_datasets_small/{train,val}.
TRAIN_GAMES = [
    "BeamRider", "Breakout", "DemonAttack", "Enduro", "MsPacman", "Pong",
    "Qbert", "Riverraid", "RoadRunner", "Seaquest", "SpaceInvaders",
]

# The 4 games that appear ONLY in custom_datasets_small/test. Never trained on.
HELDOUT_GAMES = ["Alien", "Asteroids", "Atlantis", "IceHockey"]

NUM_TRAIN_GAMES = len(TRAIN_GAMES)      # 11
UNKNOWN_ID = NUM_TRAIN_GAMES            # 11
NUM_HELDOUT_GAMES = len(HELDOUT_GAMES)  # 4
NUM_ENV_SLOTS = NUM_TRAIN_GAMES + 1 + NUM_HELDOUT_GAMES  # 16

GAME_TO_ID = {g: i for i, g in enumerate(TRAIN_GAMES)}
GAME_TO_ID.update({g: UNKNOWN_ID + 1 + i for i, g in enumerate(HELDOUT_GAMES)})
ID_TO_GAME = {i: g for g, i in GAME_TO_ID.items()}
ID_TO_GAME[UNKNOWN_ID] = "UNKNOWN"

# Minimal-action-set sizes, verified live against ale-py 0.11.2. Mirrors the table
# in eval_jepa_diagnostics.py:69-86; build_registry() cross-checks against it.
GAME_ACTION_COUNTS = {
    "Breakout": 4, "Qbert": 6, "DemonAttack": 6, "SpaceInvaders": 6, "Pong": 6,
    "BeamRider": 9, "Enduro": 9, "MsPacman": 9,
    "RoadRunner": 18, "Riverraid": 18, "Seaquest": 18,
    "Asteroids": 14, "Alien": 18, "Atlantis": 4, "IceHockey": 18,
}


def game_name_from_path(path):
    """'.../SeaquestNoFrameskip-v4_expert_part1.npz' -> 'Seaquest'.

    Same convention as eval_jepa_diagnostics.py:89-91.
    """
    return os.path.basename(path).split("NoFrameskip")[0]


def build_registry(verbose=True):
    """Query ale-py for each game's minimal action set and map it into ALE-18.

    Requires gymnasium + ale-py. Raises if unavailable -- we refuse to guess a
    mapping, since a wrong LUT would silently corrupt every action label.
    """
    import ale_py
    import gymnasium as gym

    gym.register_envs(ale_py)

    action_map = {}
    for game in TRAIN_GAMES + HELDOUT_GAMES:
        env = gym.make(f"ALE/{game}-v5")
        meanings = list(env.unwrapped.get_action_meanings())
        env.close()

        lut = []
        for m in meanings:
            if m not in FULL_ACTION_MEANINGS:
                raise ValueError(f"{game}: action '{m}' is not in the canonical ALE-18 set")
            lut.append(FULL_ACTION_MEANINGS.index(m))

        expected = GAME_ACTION_COUNTS.get(game)
        if expected is not None and len(lut) != expected:
            warnings.warn(
                f"{game}: live action count {len(lut)} != hardcoded {expected}. "
                f"Using the live value; update GAME_ACTION_COUNTS."
            )
        action_map[game] = lut
        if verbose:
            print(f"  {game:<15} {len(lut):>2} actions -> canonical {lut}")

    return {
        "full_action_meanings": FULL_ACTION_MEANINGS,
        "train_games": TRAIN_GAMES,
        "heldout_games": HELDOUT_GAMES,
        "unknown_id": UNKNOWN_ID,
        "game_to_id": GAME_TO_ID,
        "action_map": action_map,
    }


def load_registry(rebuild=False, verbose=False):
    """Load game_registry.json, building it via ale-py if missing."""
    if rebuild or not os.path.exists(REGISTRY_PATH):
        if verbose:
            print(f"Building action registry (requires ale-py) -> {REGISTRY_PATH}")
        reg = build_registry(verbose=verbose)
        with open(REGISTRY_PATH, "w") as f:
            json.dump(reg, f, indent=2)
        return reg

    with open(REGISTRY_PATH) as f:
        reg = json.load(f)

    # A stale registry is worse than none -- it would mislabel every action.
    if reg.get("train_games") != TRAIN_GAMES or reg.get("heldout_games") != HELDOUT_GAMES:
        raise ValueError(
            f"{REGISTRY_PATH} is stale (game lists no longer match games.py). "
            f"Delete it or call load_registry(rebuild=True)."
        )
    return reg


_REGISTRY = None


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = load_registry()
    return _REGISTRY


def action_lut(game):
    """int32 array mapping this game's minimal action index -> canonical ALE-18 index."""
    amap = _registry()["action_map"]
    if game not in amap:
        raise KeyError(f"Unknown game '{game}'. Known: {sorted(amap)}")
    return np.asarray(amap[game], dtype=np.int32)


def valid_action_mask():
    """Bool array [NUM_ENV_SLOTS, 18]; True where that env slot's game can emit
    the canonical action.

    Used to mask invalid logits in the inverse-dynamics cross-entropy, which is
    TD-MPC2's action-masking applied to Atari. This changes the chance floor from
    log(18) to log(|A_g|) -- the honest baseline, and what makes the information-gain
    metric (H(a|game) - L_inv) meaningful.

    The UNKNOWN row is all-True: with no game identity we cannot exclude anything.
    """
    mask = np.zeros((NUM_ENV_SLOTS, NUM_CANONICAL_ACTIONS), dtype=bool)
    amap = _registry()["action_map"]
    for game, gid in GAME_TO_ID.items():
        mask[gid, amap[game]] = True
    mask[UNKNOWN_ID, :] = True
    return mask


def action_entropy_floor(counts=None):
    """log|A_g| per env slot, in nats -- the masked-chance floor for the inverse head.

    Averaged over a batch's game ids this gives the H(a|game) upper bound that
    train/inverse_action_loss must beat to demonstrate any dynamics knowledge
    beyond game recognition (finding D3).
    """
    amap = _registry()["action_map"]
    floor = np.zeros(NUM_ENV_SLOTS, dtype=np.float32)
    for game, gid in GAME_TO_ID.items():
        floor[gid] = np.log(len(amap[game]))
    floor[UNKNOWN_ID] = np.log(NUM_CANONICAL_ACTIONS)
    return floor


if __name__ == "__main__":
    # Verification 1 from the plan: build the registry and assert it is coherent.
    print("Building registry from ale-py...\n")
    reg = load_registry(rebuild=True, verbose=True)

    print("\nValidating...")
    amap = reg["action_map"]
    for game, lut in amap.items():
        assert len(set(lut)) == len(lut), f"{game}: LUT is not injective: {lut}"
        assert all(0 <= a < NUM_CANONICAL_ACTIONS for a in lut), f"{game}: LUT out of range: {lut}"
        expected = GAME_ACTION_COUNTS.get(game)
        assert len(lut) == expected, f"{game}: {len(lut)} actions, table says {expected}"
    print(f"  [ok] {len(amap)} LUTs injective, in range, and matching GAME_ACTION_COUNTS")

    ids = sorted(GAME_TO_ID.values()) + [UNKNOWN_ID]
    assert sorted(ids) == list(range(NUM_ENV_SLOTS)), f"env id space has gaps/dupes: {sorted(ids)}"
    assert UNKNOWN_ID not in GAME_TO_ID.values()
    print(f"  [ok] env id space is exactly 0..{NUM_ENV_SLOTS - 1} with UNKNOWN={UNKNOWN_ID}")

    m = valid_action_mask()
    assert m.shape == (NUM_ENV_SLOTS, NUM_CANONICAL_ACTIONS)
    assert m[UNKNOWN_ID].all(), "UNKNOWN row must permit every action"
    for game, gid in GAME_TO_ID.items():
        assert m[gid].sum() == len(amap[game]), f"{game}: mask cardinality mismatch"
    print(f"  [ok] valid_action_mask {m.shape}, per-game cardinalities match")

    floor = action_entropy_floor()
    print(f"  [ok] entropy floors (nats): min={floor.min():.3f} max={floor.max():.3f}")

    print(f"\nWrote {REGISTRY_PATH}")
    print("\nEnv id space:")
    for i in range(NUM_ENV_SLOTS):
        tag = "  <- UNKNOWN" if i == UNKNOWN_ID else ("  (held out)" if i > UNKNOWN_ID else "")
        n = "-" if i == UNKNOWN_ID else len(amap[ID_TO_GAME[i]])
        print(f"  {i:>2}  {ID_TO_GAME[i]:<15} {n:>3} actions{tag}")
