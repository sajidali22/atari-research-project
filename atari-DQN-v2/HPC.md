# Running on capella (1× H100)

Three `sbatch` commands. No arithmetic on your side — the training script measures its
own throughput and sizes the run, and the sweep hands its winning λ to the main run
through `sweep_result.json`.

## 0. Once, before anything

**Rotate your W&B key.** The old one is in git history (`git show c05ac30:run_job.sh`)
and hardcoded at `evaluation/download_wandb.py:8`.

```bash
# in ~/.bashrc on capella -- never in a committed script
export WANDB_API_KEY=<new key>
export VENV_PATH=$HOME/DIR/horse/attari/atari-research-project/.venv
```

`VENV_PATH` is read by all three job scripts. Your previous runs used a `uv` venv with
no `module load`, so nothing else is needed.

## 1. Transfer

Only the compressed shards move — 2.6 GB. The 161 GB of `.npy_cache` is built on the
cluster in Job A.

```bash
DEST=$USER@capella.hpc.tu-dresden.de:$HOME/DIR/horse/attari/atari-research-project
rsync -avP --exclude '.npy_cache' atari-DQN-v2/ $DEST/atari-DQN-v2/
rsync -avP --exclude '.npy_cache' \
      atari-DQN/custom_datasets atari-DQN/custom_datasets_small \
      $DEST/atari-DQN-v2/
```

The local `custom_datasets*` are symlinks into `../atari-DQN/`; the second command
copies the real directories, so they become plain directories on the cluster.

## 2. Job A — verify and build caches (CPU only, ~30–60 min)

```bash
cd $HOME/DIR/horse/attari/atari-research-project/atari-DQN-v2
sbatch slurm/job_a_cache.sh
```

No GPU is requested: this is 161 GB of decompression, and doing it inside a GPU job
would spend an H100 allocation on zlib. It refuses to write anything if disk is short
or a shard is malformed. To check without building:

```bash
python preflight.py --check
```

Expected: 55 train shards (CHW), 11 val, 4 test (HWC), and a disk requirement of
~161 GB. If space is tight, drop the held-out set (`--splits train val`, −10.5 GB) or
train on `custom_datasets_small/train` (−115.7 GB).

## 3. Job B — λ sweep (4 arms in parallel, ~1.5 h)

```bash
sbatch slurm/job_b_sweep.sh          # array 0-3 over {0.003, 0.01, 0.03, 0.1}
```

SimNorm shrank the JEPA term ~10× while the inverse cross-entropy is unchanged
(measured: 0.026 vs 2.2 nats at init), so v1's 0.1 is far off and the right value is
genuinely unknown.

Each arm appends to `production_checkpoints/sweep_result.json` and re-ranks. Selection
is on **information gain**, not raw cross-entropy — a head that merely recognises the
game already scores H(a|game)=1.886 against a log(18)=2.890 floor. Arms whose env
embeddings collapsed (`mean_pairwise_cos > 0.9`) are disqualified outright: a low loss
with inert conditioning is not a win.

Wait for all four before Job C:

```bash
squeue -u $USER
cat production_checkpoints/sweep_result.json | python -m json.tool | head -20
```

## 4. Job C — main run (12–16 h)

```bash
sbatch slurm/job_c_main.sh
sbatch --time=12:00:00 slurm/job_c_main.sh   # shorter allocation is fine
```

The script reads its own SLURM end time and sizes the budget from it, so changing
`--time` needs no other edit. `--inverse-weight auto` picks up the sweep winner.

**If it dies or hits the wall clock, resubmit the identical script.** `--resume`
restores epoch, global step, optimizer, LR schedule, best-so-far and the W&B run id,
so the run continues rather than restarting or forking a second W&B run.

## 5. After

```bash
python eval_jepa_diagnostics.py --output-json results/val.json
python eval_jepa_diagnostics.py --data-dir custom_datasets_small/test \
       --env-mode unk --output-json results/test_unk.json
for G in Alien Asteroids Atlantis IceHockey; do
  python jepa_fewshot.py --game $G --n-adapt 1000 5000 20000 \
         --output-json results/fewshot_$G.json
done
```

---

## What to watch on W&B

In priority order:

1. **`env_embed/mean_pairwise_cos`** — near 0 means the 11 game embeddings occupy
   distinct directions and conditioning is doing work; approaching 1 means they
   collapsed and conditioning went inert *while every loss curve still looks healthy*.
   Nothing else in the run catches this. The loop prints a warning above 0.9.
2. **`val/inverse_accuracy` vs `val/inverse_chance_accuracy`** — chance differs per
   game (Breakout 4 actions, Seaquest 18), so the floor is per-sample. At the floor
   means the inverse head is guessing, whatever the cross-entropy shows.
3. **`heldout_unk/gap_vs_seen_unk`** — the headline. Unseen games under UNKNOWN against
   *seen* games under UNKNOWN, so it isolates "unseen game" from "no game id". Watch
   for zero-shot transfer peaking early then decaying as the model specialises.

Also: `train/samples_per_sec` (is the GPU fed?), `train/grad_clipped` (persistently 1.0
means the clip is silently rescaling the LR), and `health/effective_rank` **against the
`health_init/*` summary baseline**, never against an absolute threshold — an untrained
model already scores ~2–9 / 256 on Atari frames.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `--inverse-weight auto` exits at startup | Sweep has not completed. Check `production_checkpoints/sweep_result.json`, or pass a number. |
| `samples_per_sec` far below expectation | Dataloader-bound. Raise `--cpus-per-task` and `--num-workers` together; the job is data-bound, not compute-bound. |
| `PREFLIGHT FAILED: INSUFFICIENT DISK` | Drop `test` from `--splits`, or use the small train split. |
| Job killed at the wall clock | Resubmit the same script — `--resume` continues. If it happens repeatedly, epochs are too long; lower `--target-hours`. |
| W&B hangs on a compute node | Add `--wandb-mode offline`, then `wandb sync production_checkpoints/wandb/offline-run-*` from a login node. |
| Second W&B run appears after a resume | The checkpoint had no `wandb_id` (pre-resume-support checkpoint). Harmless; the metrics are intact. |
| `grad_clipped` pinned at 1.0 | The clip is active every step. Lower `--lr`, or raise the clip in the loop. |

## Configuration notes

**Batch size stays 512.** Memory is irrelevant (12 M params on 94 GB), but `tau=0.996`
is applied *per optimizer step* — a ~250-step EMA horizon. Doubling the batch halves
the steps for the same data and therefore halves how far the target encoder tracks. If
you benchmark B=1024 as faster, also raise `--lr` by √2 (≈4.2e-4) and lower `--tau` to
~0.992 to preserve that horizon.

**`--compile`** is off by default. It should help a fixed-shape transformer, but it
adds trace time and has not been tested on this model — try it in a sweep arm first.
Checkpoints stay compatible either way; the training loop saves the uncompiled module.
