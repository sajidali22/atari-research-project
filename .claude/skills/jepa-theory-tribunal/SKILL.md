---
name: jepa-theory-tribunal
description: Convene a three-agent academic tribunal (Validator / Skeptic / Visionary) that theoretically validates the action-conditioned JEPA model in jepa/JEPA.py and its training pipeline in jepa/jepa_train.py against SOTA literature (V-JEPA, AMI, World Models, ICM, DreamerV3, VICReg, InfoNCE), then transcribes the debate and a next-gen architecture proposal into THEORETICAL_VALIDATION_DEBATE.md. Theory only — no refactoring, no bug fixes, no edits to existing files. Use when asked for a theoretical/academic validation, a thesis-defense-style critique, a literature grounding of the JEPA design, or to run the Validator/Skeptic/Visionary tribunal.
---

# JEPA theoretical validation tribunal

Adapted from `docs/archive/prompts/prompt_v3.md`. You preside as an **ICLR/NeurIPS Area Chair**

> Note: file/line citations below were written against the v1 code (`atari-DQN/`). The
> current code is v2 in `jepa/` (see `jepa/CHANGES.md`); re-check line numbers before quoting.
over a theoretical review tribunal, conducting a strict academic validation of a custom
Action-Conditioned JEPA and its training pipeline, written for a Master's thesis.

**What the author is claiming to build:** a "Universal Feature Extractor" for Atari that fuses
passive video-prediction (V-JEPA) with active, action-aware World Models, stabilized by an
Intrinsic Curiosity / Inverse Dynamics module. The tribunal's job is to decide whether the code
actually earns that claim, in theory.

## Scope discipline (this is what separates this skill from `jepa-analysis`)

- **Theory only.** Do **not** suggest code refactoring, style changes, or bug fixes. If you
  spot a bug, mention it only where it changes the *theoretical* argument (e.g. a gradient path
  that isn't what the design intends), and frame it as an architectural consequence — not a
  patch to apply.
- **Read-only.** The only file written is the new `docs/analysis/THEORETICAL_VALIDATION_DEBATE.md` (default
  location, unless the user says otherwise). Never edit `jepa/JEPA.py`,
  `jepa/jepa_train.py`, or any other pre-existing file.
- **Formal academic register.** Write as if for a thesis defense: precise, cited, mathematical
  where mathematics is warranted. Every claim mapped to literature gets a real citation
  (author, venue, year). Do not invent citations — if unsure a paper says what you need, say
  the connection is *your* reading rather than attributing it.
- Sibling skill `jepa-analysis` handles the engineering-side dialectic (compute cost,
  benchmarks, evaluation metrics) and writes `JEPA_ANALYSIS_AND_PLAN.md`. Don't duplicate it.

## Entry points

`prompt_v3.md` names `train_JEPA.py`; the file in this repo is **`jepa/jepa_train.py`**.
Ignore the stale root-level `jepa_train.py` (broken imports — see `CLAUDE.md`). Read both
`jepa/JEPA.py` and `jepa/jepa_train.py` in full before any agent speaks. Pull in
`jepa/dataset_JEPA.py` when the argument turns on what a transition actually contains.

## Current-state anchors (verify against the files; report drift as a finding)

- `PaperAccurateJEPA` (`jepa/JEPA.py:121`): `context_encoder` = `TransformerEncoder`
  **depth 6**, `predictor` = `TransformerPredictor` **depth 3**, `embed_dim=256`,
  `tau=0.996`. `PatchEmbedding` at `patch_size=14` → **36 tokens** from `(B, 4, 84, 84)`.
- **Action injection**: `action_embed(a_t).unsqueeze(1)` prepended as token 0 (sequence 37),
  sliced back off the predictor output (`JEPA.py:99-119`).
- **Anti-collapse machinery is double-locked**: `target_encoder` params carry
  `requires_grad=False` (`JEPA.py:142-143`) *and* the target forward runs under
  `torch.no_grad()` (`JEPA.py:207`). EMA update `φ ← τφ + (1-τ)θ` runs after every
  `optimizer.step()` (`jepa_train.py:130`), at **fixed τ** — no momentum schedule toward 1.0,
  unlike BYOL/V-JEPA.
- **Objective**: `loss = jepa_loss + 0.1 * inv_loss` (`jepa_train.py:119`), where `jepa_loss`
  is MSE over `[B, 36, 256]` meaned across tokens and dims, terminal-masked; `inv_loss` is
  cross-entropy over 18 actions, terminal-masked. The `0.1` is a hardcoded constant with no
  stated derivation.
- **A gradient-path asymmetry worth arguing about**: `compute_inverse_loss` is called as
  `model.compute_inverse_loss(z_context, z_target, batch["a_t"])` (`jepa_train.py:115`) — its
  second argument is the **target encoder's** output, produced under `no_grad`. So the inverse
  loss shapes only `z_t` (context encoder) and `inverse_head`; it can exert no gradient on the
  representation of `s_{t+1}`. Pathak et al.'s ICM (2017) derives its features from a single
  trainable encoder applied to both states. Whether this asymmetry preserves or breaks the ICM
  anti-action-collapse argument is a first-class question for the tribunal, not a footnote.
- **No variance/covariance term anywhere** — anti-collapse rests entirely on EMA asymmetry plus
  the predictor bottleneck. No VICReg, no InfoNCE, no stop-grad predictor head beyond the
  action-conditioned predictor itself.
- **Frame-stack overlap**: `s_t` and `s_next` are both 4-frame stacks drawn from
  `obs` / `next_obs` in `jepa/dataset_JEPA.py`. If they follow the standard replay
  convention they share 3 of 4 frames — verify this in the dataset/collection code before
  building the "lazy identity" argument on it, because the strength of that attack scales
  directly with the overlap.
- Optimizer: AdamW + `CosineAnnealingLR`, AMP with `GradScaler`, grad-norm clipping
  (`jepa_train.py:69-83`).

## The tribunal

All three agents read both files first. Run the personas yourself, clearly labeled — there are
no standing subagents by these names in this project. The Agent tool may carry the step-1 code
audit if context is tight, but the debate itself must happen where all three positions are
visible at once.

### Agent 1 — The Validator (Proponent)

Defend the architecture by mapping code to SOTA literature:

- Map Context/Target encoders to LeCun's *A Path Towards Autonomous Machine Intelligence*
  (2022) and Meta's **V-JEPA** (Bardes et al., 2024); map the EMA target to BYOL
  (Grill et al., 2020) lineage.
- Map action-injection to Ha & Schmidhuber, **World Models** (2018), and the
  action-conditioned latent dynamics line (PlaNet / Dreamer).
- Argue mathematically that the EMA update plus `no_grad`/`requires_grad=False` on the target
  branch prevents representation collapse — state the argument, don't just assert it (why is
  the degenerate constant solution not a fixed point of *this* update?).
- Connect the inverse head to **ICM** (Pathak et al., 2017) and explain the mechanism by which
  inverse dynamics is supposed to prevent action collapse.

### Agent 2 — The Skeptic (Brutal Reviewer)

Attack the theoretical vulnerabilities. At minimum, prosecute these three:

1. **Loss incommensurability.** Is cross-entropy over a discrete 18-way action space
   mathematically compatible with MSE over continuous latent embeddings under naive scalar
   addition? Their gradients differ in scale and in units; the MSE term shrinks as the model
   improves while CE is lower-bounded by environment stochasticity. Does `λ=0.1` implicitly
   define a time-varying, uncontrolled task weighting — and do the two terms produce conflicting
   gradients on the shared context encoder?
2. **The lazy identity function.** With a 4-frame stack, `s_t` and `s_{t+1}` overlap heavily.
   Argue why MSE next-step prediction may be minimized by an approximate identity map rather
   than by learned dynamics, and what observable signature would distinguish the two.
3. **The Noisy-TV vulnerability.** Inverse dynamics presumes the action is recoverable from the
   state transition. Where does it fail — uncontrollable moving entities, stochastic sprites,
   frame-skip aliasing, no-op-equivalent actions (Atari's 18-action space contains actions with
   identical effects in many games)? What does the inverse head learn when the action is
   *not* identifiable, and what does that do to the shared encoder?

Also press the Validator's collapse argument: fixed τ with no schedule, and no
variance/covariance regularizer — is the anti-collapse guarantee actually a guarantee, or an
empirical observation borrowed from a different training regime?

### Agent 3 — The Visionary (SOTA Architect)

Resolve the debate by proposing a strictly superior **Next-Gen Action-Conditioned JEPA**, built
from the flaws the Skeptic established (not from generic novelty). Cover at least:

- **Latent space**: should the continuous latent be quantized — FSQ (Mentzer et al., 2023) or
  the VQ/symlog-discretized latents of **DreamerV3** (Hafner et al., 2023) — to prevent drift
  over autoregressive rollouts? Note this repo already has FSQ machinery in
  `models/fsq_vae.py`, so the comparison is concrete rather than hypothetical.
- **Objective**: should MSE be replaced or augmented by **VICReg** (Bardes et al., 2022)
  variance–invariance–covariance terms, or **InfoNCE** contrastive prediction (CPC, Oord et
  al., 2018), to force embeddings to span the manifold instead of contracting?
- **Loss balancing**: a principled alternative to the hardcoded `0.1` — uncertainty weighting
  (Kendall et al., 2018), gradient surgery/PCGrad (Yu et al., 2020), or normalizing each term.
- **The exact mathematical formulation** of the proposal: write the full objective with all
  terms, symbols defined, and state what each term is doing and which Skeptic objection it
  answers.

## Deliverable — `THEORETICAL_VALIDATION_DEBATE.md`

A brand new file containing, in formal academic prose:

1. **Architecture under review** — a precise description of what the code actually implements
   (as verified in step 1), including the gradient-flow graph in words: which branches receive
   gradients from which loss terms.
2. **Transcript of the debate** — actual dialogue between the Validator and the Skeptic across
   the three contested fronts: the target encoder / collapse argument, the action-injection
   mechanism, and the joint loss function. Let each side answer the other; a rebuttal that
   concedes a point is a stronger transcript than one that doesn't.
3. **The Visionary's proposal** — the next-gen architecture with its full mathematical
   formulation, each component traced to the specific objection it resolves.
4. **Verdict** — as Area Chair: what is theoretically sound as written, what is unjustified but
   defensible, and what is a genuine theoretical liability for the thesis. Where the tribunal
   didn't converge, record the disagreement rather than manufacturing consensus.
5. **References** — full citation list for every work invoked.

Stop there. Implementing anything from the Visionary's proposal requires an explicit follow-up
request from the user.
