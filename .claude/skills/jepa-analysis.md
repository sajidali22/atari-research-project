---
name: jepa-analysis
description: Run an adversarial two-pass review of the JEPA world model (jepa/JEPA.py) — a researcher pass proposes cross-game Atari representation improvements, a skeptical-reviewer pass challenges them for compute cost, collapse risk, and missing baselines — then write the synthesized findings to a new JEPA_ANALYSIS_AND_PLAN.md. Never edits existing project files.
when_to_use: Use when asked to analyze, refine, critique, or plan changes to the JEPA world model / architecture, or when asked to run the jepa-researcher / skeptical-reviewer dialectic from prompt.md.
---

# JEPA analysis & planning (read-only)

Adapted from `docs/archive/prompts/prompt.md`. Research context: the goal (Oleh Shkalikov topic) is
a single **Universal Feature Extractor** that generalizes across multiple Atari games, trained
via JEPA-style latent-space self-supervised prediction (no pixel reconstruction), on
`(B, 4, 84, 84)` stacked grayscale frame inputs. The open question this skill exists to keep
sharpening is: should the JEPA encoder be pretrained **independently** (offline SSL), or
**jointly** with the downstream policy/value network (auxiliary loss)?

## Hard constraints (do not violate)

- **Read-only.** Do not edit `jepa/JEPA.py`, `jepa/dataset_JEPA.py`, or any other
  existing project file during this skill. The only file this skill writes is the new
  `docs/analysis/JEPA_ANALYSIS_AND_PLAN.md` (default, unless the user asks for another location).
- **Input invariance.** Any proposal must keep the input contract `(B, 4, 84, 84)` (or a
  documented channel-last equivalent) intact.
- **No pixel reconstruction.** JEPA must stay strictly latent-space. Reject/flag any proposal
  that adds a pixel-space decoder unless it's explicitly framed as a benchmarked comparison
  against the VAE/VQ-VAE baselines already in `models/`.
- **Gradient-leak check.** Confirm the target encoder stays EMA-updated or `.detach()`/
  `torch.no_grad()`-frozen — never receiving direct gradients. (As of this writing,
  `PaperAccurateJEPA.target_encoder` in `jepa/JEPA.py` is EMA-updated via
  `update_target_network()` and encoded under `torch.no_grad()` — verify this hasn't drifted
  before assuming it's still true.)

## Workflow

1. **Source audit.** Read `jepa/JEPA.py` in full (and `jepa/dataset_JEPA.py` /
   `jepa/eval_jepa_diagnostics.py` if the analysis touches data flow or evaluation). Map out: patch
   embedding, context encoder, target encoder, predictor (action-conditioning mechanism), and
   the stop-gradient/EMA mechanics. Don't rely on memory of a prior read — re-read, since this
   file is under active iteration.

2. **Researcher pass (`jepa-researcher` persona) — Architectural Explorer & SSL Specialist.**
   Produce concrete proposals for cross-game Atari representation quality, e.g.:
   - Spatial vs. temporal masking ratio changes (I-JEPA/V-JEPA style).
   - Auxiliary loss terms (VICReg variance/covariance terms, the already-defined but unused
     `compute_inverse_loss` inverse-dynamics head — is it worth wiring in?).
   - Predictor depth/capacity tradeoffs relative to the encoder.
   - How the resulting latents should feed downstream policy/value heads: linear probing
     (frozen encoder) vs. fine-tuning vs. joint auxiliary-loss training.
   Each proposal should be concrete enough to critique — name the file/class/hyperparameter it
   touches, not just a general direction.

3. **Skeptical-reviewer pass (`skeptical-reviewer` persona) — Devil's Advocate & Benchmarking
   Guard.** Take the researcher's proposals from step 2 and challenge each one explicitly:
   - "Does this improve cross-game generalization, or just add compute overhead?"
   - Push for the simpler alternative (frozen encoder vs. joint updates; compare against the
     existing VAE/VQ-VAE baselines in `models/` before trusting JEPA is better at all).
   - Audit for RL failure modes: representation collapse, any gradient leak into the target
     encoder, dataloader/compute bottlenecks (`jepa/dataset_JEPA.py` loads full `.npz`
     arrays into RAM per file — flag if a proposal would make that worse).
   - Reject any proposal that doesn't ship with a concrete evaluation metric: downstream PPO
     sample efficiency, or linear-probe accuracy across **3+ distinct games** (e.g. Breakout,
     Pong, Seaquest — games with expert data already collected under `artifacts/data/expert/`).
   Do this as a genuine adversarial pass — don't rubber-stamp step 2's proposals. If a proposal
   survives, say why it survived the specific objection raised.

4. **Synthesize into `JEPA_ANALYSIS_AND_PLAN.md`.** Write a new file (do not touch any existing
   file) containing:
   - Current architecture summary (from step 1).
   - Each proposal from step 2, paired with the step-3 critique and the resolution (kept /
     dropped / modified, and why).
   - A step-by-step roadmap of the surviving proposals, each with its required evaluation
     metric and baseline comparison called out explicitly.
   - Open questions still unresolved (e.g. independent vs. joint pretraining) if the debate
     didn't converge — don't force false consensus.

5. **Stop.** This skill is planning-only. Do not implement any of the surviving proposals
   against `jepa/JEPA.py` unless the user explicitly asks in a follow-up turn.

## Notes on running the dialectic

There's no standing `jepa-researcher` / `skeptical-reviewer` subagent defined in this project —
run both passes yourself, clearly labeled, in sequence (step 3 must actually read and respond to
step 2's specific proposals, not restate generic caution). If the analysis is large enough to
warrant offloading, you may use the Agent tool's `fork` subagent type for the source-audit
research legwork (step 1) to keep raw exploration out of the main context — but the adversarial
debate (steps 2–3) and the final synthesis should happen where you can see both sides at once.
