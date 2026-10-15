"""
Pre-flight checks and cache builder. Run this BEFORE requesting a GPU.

Two jobs:

  --check   verify the data is present, correctly shaped and self-consistent, and
            that there is enough disk for the caches. Reads headers only; writes
            nothing. Cheap enough to run on a login node.
  --build   materialise the .npy_cache directories. This is the expensive part
            (~161 GB, tens of minutes) and it is pure CPU/IO -- doing it inside a
            GPU job would burn an H100 allocation on decompression.

Why the caches are so large: the .npz shards are only ~2.6 GB compressed, but
AtariTransitionDataset mmaps uncompressed .npy for random access, and obs/next_obs
are stored separately at 84*84*4 uint8 each:

    full train   2,750,000 x 28,224 x 2  = 144.6 GB
    val            110,000 x 28,224 x 2  =   5.8 GB
    test (4 games) 200,000 x 28,224 x 2  =  10.5 GB

Usage:
    python preflight.py --check
    python preflight.py --check --build
    python preflight.py --check --splits train val        # skip the held-out set
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import games

BYTES_PER_FRAME = 84 * 84 * 4  # uint8

# split name -> (path, expected shard count, expected frame layout)
SPLITS = {
    "train": (config.TRAIN_DIR, 55, "CHW"),
    "train_small": (config.TRAIN_DIR_SMALL, 11, "HWC"),
    "val": (config.VAL_DIR, 11, "HWC"),
    "test": (config.TEST_DIR, 4, "HWC"),
}


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def inspect(path):
    """Header-only inspection of every shard in a split. Does not decompress pixels."""
    if not os.path.isdir(path):
        return None
    shards = sorted(f for f in os.listdir(path) if f.endswith(".npz"))
    info, total, problems = [], 0, []
    for f in shards:
        full = os.path.join(path, f)
        game = games.game_name_from_path(full)
        if game not in games.GAME_TO_ID:
            problems.append(f"{f}: game '{game}' is not in games.py")
            continue
        with np.load(full) as z:
            missing = [k for k in ("obs", "actions", "next_obs", "terminals") if k not in z]
            if missing:
                problems.append(f"{f}: missing keys {missing}")
                continue
            shape = z["obs"].shape
            actions = z["actions"]
            a_max = int(actions.max()) if len(actions) else -1
            n_lut = len(games.action_lut(game))
            if a_max >= n_lut:
                problems.append(
                    f"{f}: action index {a_max} exceeds {game}'s minimal action set "
                    f"({n_lut}). Shard was collected with a different action space.")
        frame = tuple(shape[1:])
        layout = "HWC" if frame == (84, 84, 4) else "CHW" if frame == (4, 84, 84) else f"?{frame}"
        if layout.startswith("?"):
            problems.append(f"{f}: unrecognised frame shape {frame}")
        info.append(dict(file=f, game=game, n=int(shape[0]), layout=layout))
        total += int(shape[0])
    return dict(shards=info, total=total, problems=problems)


def cache_bytes(total_transitions):
    return total_transitions * BYTES_PER_FRAME * 2  # obs + next_obs


def cache_state(path):
    cache = os.path.join(path, ".npy_cache")
    if not os.path.isdir(cache):
        return 0, 0
    files = [os.path.join(cache, f) for f in os.listdir(cache) if f.endswith(".npy")]
    return len(files), sum(os.path.getsize(f) for f in files)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify data and disk (default)")
    ap.add_argument("--build", action="store_true", help="materialise the .npy_cache directories")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    choices=list(SPLITS), help="which splits to check/build")
    ap.add_argument("--margin-gb", type=float, default=10.0,
                    help="free space required beyond the computed cache size")
    args = ap.parse_args()
    if not (args.check or args.build):
        args.check = True

    print(f"Registry: {len(games.TRAIN_GAMES)} training games, "
          f"{len(games.HELDOUT_GAMES)} held out, UNKNOWN={games.UNKNOWN_ID}\n")

    ok = True
    needed = 0
    reports = {}
    for name in args.splits:
        path, exp_shards, exp_layout = SPLITS[name]
        rep = inspect(path)
        reports[name] = rep
        if rep is None:
            print(f"[MISSING] {name:<12} {path}")
            ok = False
            continue

        layouts = sorted({s["layout"] for s in rep["shards"]})
        n_files, have_bytes = cache_state(path)
        want_bytes = cache_bytes(rep["total"])
        built = have_bytes >= want_bytes * 0.98
        if not built:
            needed += want_bytes - have_bytes

        status = "cache OK" if built else (f"cache PARTIAL {human(have_bytes)}" if n_files
                                           else "cache MISSING")
        print(f"[{'ok' if not rep['problems'] else 'FAIL'}] {name:<12} "
              f"{len(rep['shards']):>3} shards  {rep['total']:>9,} transitions  "
              f"{','.join(layouts):<4}  cache {human(want_bytes):>9}  {status}")

        if len(rep["shards"]) != exp_shards:
            print(f"      WARNING: expected {exp_shards} shards, found {len(rep['shards'])}")
        if layouts != [exp_layout]:
            print(f"      NOTE: expected {exp_layout}, found {layouts}. "
                  f"dataset_JEPA detects layout per shard, so this is handled -- but "
                  f"an unexpected layout means the shard is not what you think it is.")
        for p in rep["problems"]:
            print(f"      PROBLEM: {p}")
            ok = False

        by_game = {}
        for s in rep["shards"]:
            by_game[s["game"]] = by_game.get(s["game"], 0) + s["n"]
        seen = [g for g in by_game if games.GAME_TO_ID[g] < games.UNKNOWN_ID]
        held = [g for g in by_game if games.GAME_TO_ID[g] > games.UNKNOWN_ID]
        if seen:
            print(f"      training games ({len(seen)}): {', '.join(sorted(seen))}")
        if held:
            print(f"      HELD-OUT games ({len(held)}): {', '.join(sorted(held))}")

    free = shutil.disk_usage(os.path.dirname(os.path.abspath(__file__))).free
    margin = args.margin_gb * 2 ** 30
    print(f"\nDisk: {human(free)} free | {human(needed)} still needed for caches "
          f"| margin {human(margin)}")
    if needed + margin > free:
        print(f"INSUFFICIENT DISK: need {human(needed + margin)}, have {human(free)}.")
        print("  Options: drop 'test' from --splits (-10.5 GB), or train on "
              "custom_datasets_small/train instead of the full set (-115.7 GB).")
        ok = False
    if not ok:
        print("\nPREFLIGHT FAILED -- fix the above before submitting a GPU job.")
        return 1
    print("\nPREFLIGHT OK")

    if args.build:
        # Importing here keeps --check free of torch.
        from dataset_JEPA import AtariTransitionDataset
        print("\nBuilding caches. This is IO-bound and can take tens of minutes.")
        for name in args.splits:
            path = SPLITS[name][0]
            print(f"\n--- {name}: {path}")
            ds = AtariTransitionDataset(path)
            sample = ds[0]
            print(f"    {len(ds):,} transitions | s_t {tuple(sample['s_t'].shape)} "
                  f"{sample['s_t'].dtype} | a_t {int(sample['a_t'])} | "
                  f"g_t {int(sample['g_t'])} ({games.ID_TO_GAME[int(sample['g_t'])]})")
            n_files, have = cache_state(path)
            print(f"    cache: {n_files} files, {human(have)}")
        print("\nCACHE BUILD COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
