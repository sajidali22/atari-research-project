import os

import numpy as np
import torch
from torch.utils.data import Dataset

import games


class AtariTransitionDataset(Dataset):
    """Transition tuples (s_t, a_t, s_{t+1}) pooled across Atari games.

    Three things changed relative to v1:

    1. Frame layout is DETECTED, not assumed. v1 applied np.transpose(raw, (2,0,1))
       unconditionally. That is correct for the HWC custom_datasets_small/* shards
       but silently reshapes the CHW custom_datasets/train shards from (4,84,84)
       into (84,4,84), which then feeds Conv2d(in_channels=4) an 84-channel tensor.
       The full 2.75M-transition set was therefore unusable by the v1 loop.

    2. Game identity survives. v1 computed file_idx and threw it away, so the model
       could not tell 11 games apart. We now emit `g_t`, the environment-embedding
       slot id from games.py.

    3. Actions are canonicalised to the ALE-18 index space. Stored actions index
       each game's own *minimal* action set, so raw index 3 is LEFT in Breakout and
       RIGHT in Seaquest (finding D2). A per-shard lookup table remaps them once at
       load, making the shared action embedding coherent.

    Emits: s_t [4,84,84], a_t long (canonical), s_next [4,84,84],
           mask float (0 at terminals), g_t long (env slot id).

    Frames are emitted as **uint8** by default and normalised on the GPU (see
    JEPA.to_float). Converting to float32 here would quadruple the bytes crossing the
    worker->main-process IPC boundary and the host->device copy (56 KB -> 226 KB per
    sample) and spend worker CPU doing it. This job is data-bound, so that conversion
    is the single most expensive avoidable thing in the pipeline. Pass dtype="float32"
    to restore the old behaviour for debugging.
    """

    def __init__(self, data_dir, require_known_games=True, dtype="uint8"):
        if dtype not in ("uint8", "float32"):
            raise ValueError(f"dtype must be 'uint8' or 'float32', got {dtype!r}")
        self.dtype = dtype
        self.data_dir = data_dir

        # Sorted so shard order -- and therefore index -> game mapping -- is
        # reproducible across machines. v1 relied on os.listdir order.
        self.file_paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz")
        )
        if not self.file_paths:
            raise FileNotFoundError(f"No .npz shards found in {data_dir}")

        self.cache_dir = os.path.join(data_dir, ".npy_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.obs_arrays = []
        self.action_arrays = []
        self.next_obs_arrays = []
        self.terminal_arrays = []
        self.lengths = []
        self.file_games = []      # game name per shard
        self.file_game_ids = []   # env slot id per shard
        self.file_luts = []       # minimal action index -> canonical ALE-18 index
        self.file_is_hwc = []     # frame layout per shard

        print(f"Verifying memory-mapped cache for {len(self.file_paths)} files...")

        for path in self.file_paths:
            base_name = os.path.basename(path).replace(".npz", "")

            game = games.game_name_from_path(path)
            if game not in games.GAME_TO_ID:
                if require_known_games:
                    raise KeyError(
                        f"{base_name}: game '{game}' is not in games.py. Add it to "
                        f"TRAIN_GAMES or HELDOUT_GAMES, or pass require_known_games=False."
                    )
                continue

            obs_path = os.path.join(self.cache_dir, f"{base_name}_obs.npy")
            act_path = os.path.join(self.cache_dir, f"{base_name}_actions.npy")
            next_obs_path = os.path.join(self.cache_dir, f"{base_name}_next_obs.npy")
            term_path = os.path.join(self.cache_dir, f"{base_name}_terminals.npy")

            if not all(os.path.exists(p) for p in [obs_path, act_path, next_obs_path, term_path]):
                print(f"First-time setup: unpacking {base_name}.npz into raw binary cache...")
                data = np.load(path)
                np.save(obs_path, data["obs"])
                np.save(act_path, data["actions"])
                np.save(next_obs_path, data["next_obs"])
                np.save(term_path, data["terminals"])
                del data

            obs_arr = np.load(obs_path, mmap_mode="r")
            action_arr = np.load(act_path, mmap_mode="r")
            next_obs_arr = np.load(next_obs_path, mmap_mode="r")
            term_arr = np.load(term_path, mmap_mode="r")

            # --- layout detection ---
            # HWC shards are (N,84,84,4); CHW shards are (N,4,84,84). 84 != 4 so the
            # two are unambiguous, and anything else is a corrupt shard we refuse.
            frame_shape = obs_arr.shape[1:]
            if frame_shape == (84, 84, 4):
                is_hwc = True
            elif frame_shape == (4, 84, 84):
                is_hwc = False
            else:
                raise ValueError(
                    f"{base_name}: unrecognised frame shape {frame_shape}; "
                    f"expected (84,84,4) HWC or (4,84,84) CHW"
                )

            lut = games.action_lut(game)
            observed_max = int(np.asarray(action_arr).max())
            if observed_max >= len(lut):
                raise ValueError(
                    f"{base_name}: stored action {observed_max} exceeds {game}'s minimal "
                    f"action set of size {len(lut)}. The shard was likely collected with a "
                    f"different action space than games.py assumes."
                )

            self.obs_arrays.append(obs_arr)
            self.action_arrays.append(action_arr)
            self.next_obs_arrays.append(next_obs_arr)
            self.terminal_arrays.append(term_arr)
            self.lengths.append(len(obs_arr))
            self.file_games.append(game)
            self.file_game_ids.append(games.GAME_TO_ID[game])
            self.file_luts.append(lut)
            self.file_is_hwc.append(is_hwc)

        self.cumulative_lengths = np.cumsum(self.lengths)
        self.total_length = int(self.cumulative_lengths[-1])

        n_hwc = sum(self.file_is_hwc)
        print(
            f"Cache verified. {self.total_length} transitions across "
            f"{len(set(self.file_games))} games "
            f"({n_hwc} HWC / {len(self.file_is_hwc) - n_hwc} CHW shards)."
        )

    def __len__(self):
        return self.total_length

    def game_of(self, idx):
        """Game name for a flat index, without loading any frame data."""
        return self.file_games[int(np.searchsorted(self.cumulative_lengths, idx, side="right"))]

    def __getitem__(self, idx):
        file_idx = int(np.searchsorted(self.cumulative_lengths, idx, side="right"))
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]

        # Slicing an mmap returns a view tied to the whole file; .copy() detaches a
        # standalone (84,84,4) block so worker processes don't hold the mapping open.
        raw_obs = self.obs_arrays[file_idx][local_idx].copy()
        raw_action = self.action_arrays[file_idx][local_idx].copy()
        raw_next_obs = self.next_obs_arrays[file_idx][local_idx].copy()
        is_terminal = self.terminal_arrays[file_idx][local_idx].copy()

        if self.file_is_hwc[file_idx]:
            raw_obs = np.transpose(raw_obs, (2, 0, 1))
            raw_next_obs = np.transpose(raw_next_obs, (2, 0, 1))

        # Minimal action index -> canonical ALE-18 index.
        canonical_action = int(self.file_luts[file_idx][int(raw_action)])

        s_t = torch.from_numpy(np.ascontiguousarray(raw_obs))
        s_next = torch.from_numpy(np.ascontiguousarray(raw_next_obs))
        if self.dtype == "float32":
            s_t = s_t.float().div_(255.0)
            s_next = s_next.float().div_(255.0)

        return {
            "s_t": s_t,
            "a_t": torch.tensor(canonical_action, dtype=torch.long),
            "s_next": s_next,
            "mask": torch.tensor(0.0 if is_terminal else 1.0, dtype=torch.float32),
            "g_t": torch.tensor(self.file_game_ids[file_idx], dtype=torch.long),
        }
