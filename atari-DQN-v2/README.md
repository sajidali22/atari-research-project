# atari-DQN-v2 — environment-conditioned JEPA

v2 of the action-conditioned V-JEPA world model, adding TD-MPC2's learnable task
embedding (here: one per Atari game) plus an `UNKNOWN` slot that makes zero-shot
evaluation on unseen games possible.

`../atari-DQN/` is the untouched v1 subproject and remains the reference for the v1
architecture and its checkpoints. The two `custom_datasets*` entries here are
symlinks back to it — the data is shared, not duplicated.

See [CHANGES.md](CHANGES.md) for the full v1 → v2 record.

## What changed, and why

| Change | Motivation |
|---|---|
| Per-game env embedding, injected as a token into encoder **and** predictor | TD-MPC2 conditions `h(s,e)`, `d(z,a,e)`, … on a learnable task embedding |
| `UNKNOWN` slot + 25% env dropout | TD-MPC2 "requires training on all target tasks … rather than enabling zero-shot generalization". The `UNKNOWN` slot is the mechanism that closes that |
| Canonical ALE-18 action remap + masked inverse head | Stored actions index each game's *minimal* set, so index 3 was LEFT in Breakout and RIGHT in Seaquest (finding D2). TD-MPC2's zero-pad + action masking |
| SimNorm on all three latents | TD-MPC2 Eq. 5; removes the scale route to representation collapse |
| Layout auto-detection in the dataset | v1 transposed unconditionally, corrupting the CHW `custom_datasets/train` into `(84,4,84)`. The full 2.75M set was unusable |
| Per-game metrics + information gain + health baseline | v1 pooled 11 games into one number and logged no health statistic (D3, D10) |

## Order of operations

```bash
# 1. Build the action registry (needs ale-py; caches to game_registry.json)
python games.py

# 2. Verifications — each is self-checking and exits non-zero on failure
python JEPA.py                      # architecture, SimNorm, gradient isolation

# 3. Pipeline smoke test (no GPU needed, ~2 min)
python jepa_train.py --smoke --wandb-mode disabled

# 4. lambda sweep — SimNorm shrinks the JEPA term ~10x while the inverse
#    cross-entropy is unchanged, so v1's 0.1 is far too large. Pick on val
#    information gain, NOT raw cross-entropy.
for L in 0.003 0.01 0.03 0.1; do
  python jepa_train.py --inverse-weight $L --epochs 5 --run-name sweep-lam$L
done

# 5. Full run on the 2.75M-transition set
python jepa_train.py --epochs 100

# 6. Diagnostics — seen games, then the 4 held-out games zero-shot
python eval_jepa_diagnostics.py --output-json results/val.json
python eval_jepa_diagnostics.py --data-dir custom_datasets_small/test \
       --env-mode unk --output-json results/test_unk.json

# 7. Few-shot adaptation, one held-out game at a time
python jepa_fewshot.py --game Alien --n-adapt 1000 5000 20000
```

Step 5 builds a ~10 GB `.npy_cache` next to `custom_datasets/train` on first run.
Only `custom_datasets_small/val` is currently cached.

## What lands in W&B

~125 keys in 9 groups: `train/`, `val/`, `val_unk/`, `heldout_unk/`, `health/`,
`env_embed/`, `per_game_train/`, `per_game_val/`, `per_game_val_unk/`, plus `best/`
and `health_init/` in the run summary.

Three panels to put on the front of the dashboard:

1. **`env_embed/mean_pairwise_cos`** — the v2 premise. Near 0 means the game
   embeddings are distinct and conditioning is doing work; approaching 1 means they
   have collapsed and conditioning is inert *while every loss curve still looks fine*.
   Nothing else in the run would catch this.
2. **`val/inverse_accuracy` vs `val/inverse_chance_accuracy`** — chance differs per
   game (Breakout has 4 actions, Seaquest 18), so the floor is computed per sample.
   Above the floor means real inverse dynamics; at the floor means the head is
   guessing regardless of what the cross-entropy looks like.
3. **`heldout_unk/gap_vs_seen_unk`** — the headline. Unseen games under `UNKNOWN`
   compared against *seen games under `UNKNOWN`*, not against `val/`, so it isolates
   "unseen game" from "no game id". Requires `--eval-heldout`.

Held-out evaluation is off by default because it materialises a ~11 GB `.npy_cache`
on first use. Turn it on for the real runs — the interesting failure mode is
zero-shot transfer peaking early and then decaying as the model specialises to the 11
training games, which only a per-epoch trace reveals.

## Reading the results

**Do not read effective rank as an absolute number.** Measured, not assumed: an
untrained model already scores ~2–9 / 256 on real Atari frames, because most 84×84
patches are empty background and transformer tokens are anisotropic at init. v1's
Section E threshold (`warn below 25.6`) fires on a network that has never seen a
gradient. Both `jepa_train.py` and Section E now emit an untrained reference
alongside; only a value well *below* that reference indicates collapse.

**The three Section F columns are the point**, not the `UNKNOWN` number alone:

- `true` vs `unk` on **seen** games — the cost of being denied the game id
- `unk` on **unseen** games — the cost of the id being unavailable at all
- `unk` vs `random` — whether `UNKNOWN` learned a prior, or is just some vector

Without the third column a good `UNKNOWN` score proves nothing.

**Few-shot arms are comparisons**: `embed` vs `embed_rand` isolates whether the
`UNKNOWN` initialisation carried a useful prior; `full` vs `scratch` is TD-MPC2's
headline (~2× in the low-data regime).

## Known limitation carried over from v1

Training is still single-step, and `next_obs[..., :3] == obs[..., 1:]` holds at a
measured fraction of **exactly 1.0** — three of four target frames are a bit-identical
copy of the input (finding D1). The lazy-identity confound at k=1 is therefore
**not** addressed. Section A reports the identity baseline at every horizon; at k≥4
the stacks share zero frames and the confound is gone. Quote those numbers, not k=1.

The multi-step consistency loss (TD-MPC2 Eq. 3's `Σ λ^t`) is the fix and was
deliberately deferred. PPO integration is also out of scope — `PPO/factory.py` still
has no `JEPA` branch.
