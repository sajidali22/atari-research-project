# v1 → v2: what changed and why

Baseline is `../atari-DQN/` (unmodified). Every file named below exists in both
folders, so any change here can be diffed directly:

```bash
diff ../atari-DQN/JEPA.py JEPA.py
diff ../atari-DQN/jepa_train.py jepa_train.py
diff ../atari-DQN/dataset_JEPA.py dataset_JEPA.py
diff ../atari-DQN/config.py config.py
```

Motivating request: *add an environment representation, and an unknown environment
for unseen games.* Translated into TD-MPC2 (Hansen et al., 2024) terms, that is a
learnable per-task embedding conditioning every component, plus a reserved slot that
covers tasks the model was never trained on — the case TD-MPC2 explicitly does not
handle: *"requires training on all target tasks simultaneously rather than enabling
zero-shot generalization to completely unseen tasks."*

---

## 1. Environment embedding (the main feature)

**v1:** no notion of which game a sample came from, anywhere in the pipeline.

**v2:** `env_embed = nn.Embedding(16, 256, max_norm=1.0)`, injected as a token.

| | v1 | v2 |
|---|---|---|
| encoder | `forward(x)` → 36 tokens | `forward(x, e)` → `[env] + 36` = 37, env sliced off |
| predictor | `forward(z, action)` → `[act] + 36` = 37 | `forward(z, action, e)` → `[env, act] + 36` = 38 |
| inverse head | `Linear(512 → 256 → 18)` | `Linear(768 → 256 → 18)`, env concatenated |

Slot layout — 16 rows:

```
 0..10  the 11 training games (alphabetical)
 11     UNKNOWN
 12..15 Alien, Asteroids, Atlantis, IceHockey (pre-reserved, untrained)
```

Two decisions worth recording:

- **`env_embed` lives on `PaperAccurateJEPA`, not inside `TransformerEncoder`.**
  `copy.deepcopy(self.context_encoder)` (v1 line 133) would otherwise produce a
  second, frozen, EMA-lagged copy of the table. The encoders receive the embedding
  *vector*, never the id.
- **Rows 12–15 are pre-allocated.** Few-shot adaptation therefore needs no checkpoint
  surgery — the `state_dict` shape is identical before and after.

Init detail: `nn.Embedding`'s default `N(0,1)` gives each row a norm of ≈16 at
`d=256`, and `max_norm` only renormalises rows *as they are looked up* — so an
untouched row would sit at 16 while trained rows sit at 1, and its first lookup would
be a 16× discontinuity. v2 initialises every row on the unit sphere.

## 2. UNKNOWN slot and env dropout

25% of training samples have their game id replaced by `UNKNOWN_ID` before the
lookup (classifier-free-guidance style). The same possibly-dropped id feeds the
context encoder, the target encoder and the predictor, so the prediction target
stays consistent with the conditioning.

This buys three things: `UNKNOWN` learns a real generic-Atari prior; the encoder
cannot fragment into 11 disjoint per-game subspaces because it must still work
unconditioned; and one checkpoint is evaluable in both modes, so the "does env
conditioning help?" ablation is free (`--env-mode true|unk`).

`resolve_env_ids(g_t, mode)` is the single entry point — `"auto"` (dropout while
training), `"true"`, `"unk"`, or a specific slot for few-shot.

## 3. Canonical ALE-18 action space

**The v1 bug.** Collection never passed `full_action_space=True`, so stored actions
index each game's own *minimal* set. One shared `nn.Embedding(18, 256)` served all
of them. Concretely, from the generated registry:

```
Breakout  minimal [0,1,2,3] -> canonical [0, 1, 3, 4]    index 3 = LEFT
Seaquest  minimal [0..17]   -> canonical [0..17]         index 3 = RIGHT
```

Same embedding row, opposite direction. Not polysemous — contradictory.

**v2.** `games.py` builds a per-game minimal→ALE-18 lookup from
`gym.make(...).unwrapped.get_action_meanings()`, cached to `game_registry.json` so
training never needs `gym`. The LUT is applied at dataset load. No data
re-collection was required.

The inverse-dynamics head now masks logits to the acting environment's valid actions
(TD-MPC2's action masking). Masking uses the **effective** env id, not the true one,
so a sample dropped to `UNKNOWN` faces the full 18-way problem — the mask is part of
the conditioning and must not leak identity the encoder was denied.

## 4. SimNorm

TD-MPC2 Eq. 5, applied per token: `[B,36,256] → [B,36,32,8]`, softmax each group of
8, reshape back. Applied to the context encoder, the target encoder **and** the
predictor — all three must land in the same space.

Bounds the latent on a product of simplices, removing the *scale* route to
representation collapse. **It does not remove the constant-vector route**: a fixed
non-zero simplex vector is still a valid collapse solution. SimNorm is a partial
mitigation, not a proof.

**Consequence that changes a hyperparameter.** Simplex entries average `1/V = 0.125`,
so the JEPA MSE term shrinks by roughly an order of magnitude while the inverse
cross-entropy (in nats) does not move. Measured at init: JEPA loss **0.026** against
an inverse loss of **~2.2 nats**. v1's `loss = jepa_loss + 0.1 * inv_loss`
(`jepa_train.py:119`) would let the inverse term dominate by ~8×. Default is now
`0.01`, and the README documents a sweep.

## 5. Dataset — three fixes

**Layout.** v1 applied `np.transpose(raw, (2,0,1))` unconditionally
(`dataset_JEPA.py:92`). Correct for the HWC `custom_datasets_small/*` shards, but it
reshapes the CHW `custom_datasets/train` shards `(4,84,84)` into `(84,4,84)`, which
then feeds `Conv2d(in_channels=4)` an 84-channel tensor. **The full 2.75M-transition
set was unusable by the v1 loop.** v2 detects layout per shard from `arr.shape[1:]`
and refuses anything that is neither.

**Game identity.** v1 computed `file_idx` (line 75) and discarded it. v2 emits `g_t`.

**Shard order.** v1 used `os.listdir` order (arbitrary); v2 sorts, so the
index→game mapping is reproducible across machines.

Returned dict: `{s_t, a_t, s_next, mask}` → `{s_t, a_t, s_next, mask, g_t}`, with
`a_t` now canonical.

## 6. config.py

- Paths anchored on `__file__`. v1 had `TRAIN_DIR = "atari-DQN/custom_datasets/train"`
  but `VAL_DIR = "custom_datasets_small/val"` — **no single working directory made
  both resolve.**
- `TEST_DIR` pointed at `custom_datasets/test`, which does not exist. It now points
  at the 4 held-out games, which v1 could not reach from config at all.
- Dropped the unused VAE block (`MODEL_TYPE`, `FSQ_LEVELS`, `NUM_EMBEDDINGS`,
  `COMMITMENT_COST`, `BETA`, …) — no JEPA file ever read it.

## 7. Metrics

v1 logged 12 keys; v2 logs ~125 across 9 groups.

| v1 | v2 |
|---|---|
| pooled loss over 11 games | per-game train/val breakdown, **plus per-game `n`** |
| raw inverse cross-entropy | **information gain** `log\|A_g\| − L_inv`, plus **top-1 accuracy against its own per-game chance floor** |
| no health statistic (D10) | SimNorm entropy, per-dim std, token spread, effective rank |
| — | **untrained-reference baseline** for every health metric |
| — | **`env_embed/*`** — the v2 centerpiece; see below |
| — | `val_unk/*`: validation re-run under `UNKNOWN` on seen games |
| — | `heldout_unk/*`: the 4 held-out games zero-shot, per epoch (`--eval-heldout`) |
| grad norm discarded | `train/grad_norm`, `train/grad_clipped`, `train/loss_scale` |
| — | `train/epoch_seconds`, `train/samples_per_sec` |
| — | `best/*` and `params_*` in run summary |
| `'loss'` = train loss | `train_loss` and `val_loss` stored separately |
| state_dict only | `model_config` saved, so eval rebuilds the exact architecture |

### Why `env_embed/*` is the one to watch

Nothing else would reveal a failure of the v2 premise. If the 11 game embeddings
drift toward a single direction, environment conditioning silently becomes a no-op
while every loss curve continues to look healthy.

- `env_embed/mean_pairwise_cos` — **~0.0** = games occupy distinct directions,
  conditioning is doing work; **~1.0** = collapsed, conditioning is inert. The
  training loop prints a warning above 0.9.
- `env_embed/cos_unknown_to_games` — near +1 means `UNKNOWN` degenerated into the
  average game embedding rather than a genuine "unspecified game" prior, which would
  undermine the zero-shot claim.
- `env_embed/cosine_matrix` — heatmap image, logged every `--fig-every` epochs.

### Bug found and fixed while auditing this

`make_eval_loader`. Shards are grouped by game and the val sampler was sequential, so
capping validation with `--max-val-batches` read only the **first shard** — the pooled
"val loss" silently became "val loss on BeamRider", and the per-game panel covered one
game. Now a fixed seeded permutation is drawn, spanning all games while staying
identical across epochs. Measured effect: reported chance accuracy moved 0.111 → 0.130,
where 0.111 was exactly `1/9`, BeamRider's action count.

### The effective-rank correction

v1's Section E warns when effective rank `< 25.6` (10% of 256). **Measured here: a
randomly initialised, untrained model already scores ~2–9 / 256 on real Atari
frames** — most 84×84 patches are empty black background, and transformer token
embeddings are strongly anisotropic at init. That threshold fires on a network that
has never seen a gradient, so it cannot demonstrate collapse.

v2 emits an untrained control alongside and warns only below *half the untrained
reference*. Read the number against that control, and against the dispersion stats:
genuine collapse drives per-dim std and token spread **down**, whereas merely
concentrating variance onto a few directions lowers the participation ratio while
spread goes up.

## 8. New capabilities

- `eval_jepa_diagnostics.py --env-mode true|unk`, threaded through Sections A–E.
- **Section F** — unseen-game behaviour under three conditionings: true id,
  `UNKNOWN`, and a *random unit vector*. The third is essential: without it a good
  `UNKNOWN` number proves nothing, since any fixed vector produces some number.
- **Section G** — env-embedding geometry (cosine matrix, nearest neighbours, PNG).
  TD-MPC2 Fig. 7 found similarity tracks task *dynamics* rather than objective.
- **Section D** now measures attention on the env token beside the action token —
  if the predictor leans on *which game* rather than *what was pressed*, that is the
  dynamics-side analogue of the game-recognition shortcut.
- Section B's counterfactuals walk each game's valid canonical action list;
  `(a + offset) % 18` is meaningless once actions are canonical, because most indices
  are invalid for most games.
- `jepa_fewshot.py` — five arms: `zero_shot`, `embed`, `embed_rand`, `full`,
  `scratch`.

### Two traps handled in few-shot

1. `nn.Embedding` produces a **dense** gradient — zero for unused rows but present —
   and AdamW's decoupled weight decay updates every parameter that has a `.grad`.
   Without intervention every other game's embedding decays while you adapt one.
   Handled with a `weight_decay=0` group.
2. Adam's momentum carries over even under a zero gradient. Handled by masking the
   gradient to the target row via a backward hook.

Verified: during an `embed` run exactly one tensor changes, and within it exactly one
row; every other row is bit-identical.

---

## Verified

Registry (15 LUTs injective, in range, matching the live ale-py counts) · dataset
(CHW and HWC shards yield byte-identical `s_t`; corrupt layouts refused; 3000 val
samples all land inside their game's valid action mask) · model (SimNorm groups sum
to 1, `‖e‖ ≤ 1`, dropout fires at 0.257, disabled at eval) · gradient isolation
(target encoder receives none; `env_embed` gradient confined to rows present in the
batch, held-out rows untouched) · a 2-epoch smoke run · diagnostics A–G on a seen
split and on a mixed split where Alien and Atlantis are correctly flagged HELD OUT ·
all five few-shot arms.

## Deliberately not done

- **Multi-step consistency loss** (TD-MPC2 Eq. 3's `Σ λ^t`). Training is still
  single-step, and `next_obs[..., :3] == obs[..., 1:]` holds at a measured fraction
  of **exactly 1.0** — three of four target frames are a bit-identical copy of the
  input. The lazy-identity confound at k=1 is **not** addressed. Section A reports
  the identity baseline at every horizon; at k ≥ 4 the stacks share zero frames.
  Quote those numbers, not k=1.
- **PPO integration.** `PPO/factory.py` still has no `JEPA` branch; there remains no
  code path from a JEPA checkpoint to a PPO number.
- **Env-inference network** (predicting `e` from pixels). The `UNKNOWN` slot is the
  substitute.

## 9. HPC readiness (added for the capella run)

| Change | Why |
|---|---|
| Dataset emits **uint8**; normalisation moved into `TransformerEncoder.forward` via `to_float` | The job is data-bound. Converting in the worker pushed 226 KB/sample through IPC instead of 56 KB. One normalisation point means no caller can diverge — verified bit-identical to the float32 path. |
| `--amp-dtype`, default **bf16** | `autocast` defaulted to fp16. bf16 on H100 needs no GradScaler and cannot underflow, which matters because SimNorm keeps the JEPA term small (~0.026 at init). |
| `--resume` | A 12–18 h job that hit its wall clock previously lost everything. Restores model, optimizer, scheduler, epoch, `global_step`, `best_val_loss` and the W&B run id. No-op when no checkpoint exists, so a job script is safe to requeue verbatim. |
| `--target-hours` auto-sizing | A ~200-step probe (real training) measures throughput, then fixes epochs, steps-per-epoch and the cosine horizon. The horizon must be known upfront or the LR never fully decays; this removes the need to compute it by hand on the cluster. |
| `--max-hours` | Clean stop with a final checkpoint, instead of being killed mid-write. |
| Linear **warmup** then cosine, stepped **per step** | Standard at this batch size and previously absent. Per-step because the epoch count is unknown until auto-sizing. |
| **Weight-decay param groups** | Finding D4: decay on LayerNorm gains is an active collapse route (γ→0 ⇒ z→β). Decaying `env_embed` would also fight its `max_norm=1`. Exempts gains, biases, `pos_embed` and both embeddings. |
| `--seed`, `--compile`, `strip_compile_prefix` | Reproducibility; compiled checkpoints stay loadable by eval and few-shot. |
| `--sweep-tag` / `--inverse-weight auto` | Sweep arms write `sweep_result.json`; the main run reads the winner. Ranked on **information gain**, and arms with collapsed env embeddings are disqualified — a low loss with inert conditioning is not a win. |
| `preflight.py`, `slurm/*.sh`, `HPC.md`, `verify_hpc_ready.py` | Cache building is 161 GB of CPU/IO and must not run inside a GPU job. 20 local checks cover every path before it ships. |

Trap found while writing the job scripts: `job_c_main.sh` documented
`sbatch --time=12:00:00` as a shorter allocation, but a hardcoded `--max-hours 15.6`
would then overrun it and be killed. The script now reads its own SLURM end time via
`scontrol` and derives the budget, so changing `--time` needs no other edit.

## Files removed from this folder

Not deleted from the project — all still present in `../atari-DQN/`:

| File | Reason |
|---|---|
| `JEPA_v1.py` | byte-identical to `../atari-DQN/JEPA.py`; the snapshot was only needed while editing in place |
| `eval_JEPA.py`, `eval_JEPA_new.py` | call the v1 signature `context_encoder(s)` with no `e`; broken against v2 and superseded by `eval_jepa_diagnostics.py` |
| `atari-replay-dataset_{full,small}.py` | data collection; the datasets are already built and symlinked |
| `dataset_validation.py` | asserts HWC only, which the layout-detection change makes wrong |
| `extract_val.py` | one-off val-batch dump, unused |
