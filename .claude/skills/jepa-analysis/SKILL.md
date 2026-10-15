---
name: jepa-analysis
description: Run an adversarial two-pass review of the action-conditioned JEPA world model (jepa/JEPA.py) — a jepa-researcher pass proposes cross-game Atari representation improvements, a skeptical-reviewer pass challenges each for compute cost, representation collapse, action collapse, and missing baselines — then write the transcribed debate and roadmap to a new JEPA_ANALYSIS_AND_PLAN.md. Read-only: never edits existing project files. Use when asked to analyze, refine, critique, or plan changes to the JEPA world model / architecture, or to run the jepa-researcher / skeptical-reviewer dialectic.
---

# Action-conditioned JEPA analysis & planning (read-only)

Adapted from `docs/archive/prompts/prompt_v2.md`.

> Note: file/line citations below were written against the v1 code (`atari-DQN/`). The
> current code is v2 in `jepa/` (see `jepa/CHANGES.md`); re-check line numbers before quoting.

**Research context (Oleh Shkalikov topic):** train a single **Universal Feature Extractor**
that works across multiple/all Atari games, via **action-conditioned** self-supervised
pretraining — latent-space prediction plus an **Inverse Dynamics Regularizer** whose job is to
prevent *action collapse* (the predictor learning to ignore the action token). The standing
open question this skill exists to keep sharpening: should the JEPA encoder be pretrained
**independently** (offline SSL) or **jointly** with the downstream policy/value network
(auxiliary loss)?

**Input contract:** `(B, 4, 84, 84)` stacked grayscale frames **and** a discrete integer action
`a_t`. Both halves of that contract are load-bearing — a proposal that quietly drops the action
input is out of scope, not a simplification.

## Hard constraints (do not violate)

- **Read-only.** Do not edit `jepa/JEPA.py`, `jepa/jepa_train.py`,
  `jepa/dataset_JEPA.py`, or any other pre-existing project file during this skill. The
  only file this skill writes is the new `docs/analysis/JEPA_ANALYSIS_AND_PLAN.md` (default, unless the
  user names another location). Implementation happens only if the user asks in a later turn.
- **Input invariance.** Every proposal must preserve `(B, 4, 84, 84)` + discrete action input
  (a documented channel-last equivalent is fine).
- **No pixel reconstruction.** JEPA stays strictly latent/embedding space. Reject proposals
  that add a pixel-space decoder — the VAE / VQ-VAE / FSQ-VAE models in `models/` already
  occupy that design point and exist as the *comparison baselines*, not as things to graft on.
- **Gradient-leak check.** Verify the target encoder is EMA-updated or frozen via `.detach()` /
  `torch.no_grad()` and never receives direct gradients.

## Current-state anchors (verify, don't assume)

These were true at the time of writing; the files are under active iteration, so re-read before
relying on them. If any has drifted, that drift is itself a finding for the report.

- `PaperAccurateJEPA` (`jepa/JEPA.py:121`) = `context_encoder` (trainable ViT-style
  `TransformerEncoder`, `depth=6`; `PatchEmbedding` turns 84×84×4 into a 36-token sequence at
  `patch_size=14`, `embed_dim=256`) + `target_encoder` (EMA copy, `tau≈0.996`, updated by
  `update_target_network()`) + `predictor` (`TransformerPredictor`, `depth=3`) + `inverse_head`.
- **Action injection** is a prepended token: `action_embed(action).unsqueeze(1)` makes sequence
  length 37, and the predictor output slices the action token back off
  (`jepa/JEPA.py:99-119`). This is the exact mechanism action collapse would defeat —
  the model can learn to route around a prepended token cheaply.
- **The inverse-dynamics loss is wired in**, at a fixed weight: `loss = jepa_loss + 0.1 *
  inv_loss` (`jepa/jepa_train.py:119`, mirrored in validation at line 168), with
  `compute_inverse_loss` predicting `a_t` from `[z_current, z_next]`
  (`jepa/JEPA.py:225`). The `0.1` is an unjustified constant — treat it as a live
  question (too low → action collapse; too high → the encoder becomes an action classifier and
  discards scene structure).
- Terminal masking comes from `AtariTransitionDataset` (`jepa/dataset_JEPA.py`), which
  now unpacks each `.npz` once into a `.npy_cache/` and reads it back `mmap_mode='r'` — so RAM
  is no longer the binding constraint; disk I/O and per-sample work are.
- `jepa/eval_jepa_diagnostics.py` already implements two diagnostics: multi-step autoregressive latent
  drift, and a counterfactual action-sensitivity test (true vs. deliberately wrong action).
  Prefer extending these over inventing new metrics from scratch.

## Workflow

### Step 1 — Source audit

Read `jepa/JEPA.py` in full first, as the entry point. Pull in `jepa_train.py`,
`dataset_JEPA.py`, and `eval_JEPA.py` as the analysis touches training dynamics, data flow, or
evaluation. Map out: patch embedding, context encoder, target encoder, predictor with its
action-injection path, the inverse regularizer, and the stop-gradient/EMA mechanics. Re-read
rather than trusting a prior read or the anchors above.

### Step 2 — `jepa-researcher` pass (Architectural Explorer & SSL Specialist)

Propose concrete improvements to cross-game representation quality. Each proposal must name the
file, class, and hyperparameter it touches — concrete enough for step 3 to attack. Fertile
ground:

- Spatial vs. temporal masking ratios (I-JEPA / V-JEPA style) at the 36-token granularity.
- The inverse-dynamics regularizer: weight schedule vs. the fixed `0.1`, where the head reads
  from (pooled latents vs. per-token), whether it should share or split from the predictor path.
- Predictor depth/capacity relative to encoder depth (currently 3 vs. 4) — an
  overparameterized predictor can absorb dynamics the encoder should be learning.
- Stronger action conditioning than a prepended token (e.g. FiLM/conditioning at every block)
  if counterfactual sensitivity turns out weak.
- How latents feed downstream policy/value heads: linear probing on a frozen encoder vs.
  fine-tuning vs. joint auxiliary-loss training.

### Step 3 — `skeptical-reviewer` pass (Devil's Advocate & Benchmarking Guard)

Take step 2's proposals **individually** and attack them. This must be a genuine adversarial
pass — not generic caution restated, and not a rubber stamp. For each proposal:

- Ask outright: *"Does this improve cross-game generalization, or does it just add compute
  overhead?"*
- Counter-propose the leaner alternative: frozen encoder over joint updates; and before
  trusting JEPA is better at all, demand the comparison against the existing VAE / VQ-VAE /
  FSQ-VAE extractors already plumbed into `ppo/factory.py`'s `UniversalFeatureExtractor`.
- Audit for RL failure modes: representation collapse, **action collapse** (predictor ignoring
  the action token — the specific thing the inverse loss is supposed to prevent, and the
  reason `eval_JEPA.py`'s counterfactual test exists), gradient leaks into the target encoder,
  and dataloader/compute bottlenecks.
- Reject any proposal shipped without a concrete evaluation metric. Acceptable: downstream PPO
  sample efficiency, counterfactual action sensitivity, or linear-probe accuracy across **3+
  distinct games** (e.g. Breakout, Pong, Seaquest — expert data already exists under
  `artifacts/data/expert/`).

When a proposal survives, say *which specific objection* it survived and why.

### Step 4 — Write `JEPA_ANALYSIS_AND_PLAN.md`

A brand new file (touch nothing existing), containing:

1. **Current architecture summary** from step 1, including the verified state of the
   gradient-leak check and the action-injection path.
2. **Transcribed debate** — the actual back-and-forth, proposal by proposal: researcher
   position → reviewer critique → resolution (kept / dropped / modified, and why). Show the
   reasoning, don't just report the verdict.
3. **Roadmap** of surviving proposals in execution order, each with its required evaluation
   metric and baseline comparison stated explicitly.
4. **Open questions** where the debate genuinely didn't converge (independent vs. joint
   pretraining is the usual one). Don't manufacture consensus.

### Step 5 — Stop

Planning only. Do not implement anything against the existing files unless the user explicitly
asks in a follow-up turn.

## Running the dialectic

There are no standing `jepa-researcher` / `skeptical-reviewer` subagents in this project — run
both personas yourself, clearly labeled, in sequence. Step 3 must read and respond to step 2's
specific proposals. If the source audit is large, the Agent tool may be used for step 1's
legwork to keep raw exploration out of context, but keep the debate (steps 2–3) and the
synthesis where both sides are visible at once.
