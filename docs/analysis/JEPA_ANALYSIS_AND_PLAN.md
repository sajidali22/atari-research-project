# JEPA Analysis & Plan — Action-Conditioned World Model for a Universal Atari Feature Extractor

*Read-only analysis produced by the `jepa-analysis` skill. No existing project files were modified.*
*Research context: a single Universal Feature Extractor across many/all Atari games, via
action-conditioned self-supervised pretraining (latent prediction + inverse-dynamics
regularizer against action collapse). Standing open question: pretrain the encoder
**independently** (offline SSL) or **jointly** with the PPO policy/value net (auxiliary loss)?*

---

## 1. Current architecture summary (verified against source)

### Components (`atari-DQN/JEPA.py`)

| Piece | Class / line | What it is |
|---|---|---|
| Patch tokenizer | `PatchEmbedding` (`:6`) | `Conv2d(4, 256, k=14, s=14)` on an 84×84×4 stack → **36 tokens** (6×6) of dim 256, plus learnable spatial pos-embed. No temporal/CLS token. |
| Context encoder | `TransformerEncoder` (`:41`), instantiated `depth=6` (`:133`) | Trainable ViT, pre-LN, 8 heads, `dim_ff=1024`. Gradients flow here. |
| Target encoder | `copy.deepcopy` of context (`:138`) | EMA copy, `requires_grad=False` (`:142-143`), `tau=0.996` fixed, updated by `update_target_network()` (`:152`, `@torch.no_grad()`). |
| Predictor | `TransformerPredictor` (`:74`), instantiated `depth=3` (`:134`) | Action-conditioned. Embeds `a_t` (`nn.Embedding(18,256)`), **prepends** it as one token (seq 36→37), runs 3 attention layers, then **slices the action token back off** (`:117`). |
| Inverse head | `nn.Sequential` (`:145`) | `Linear(512→256)→GELU→Linear(256→18)`. Reads pooled `[z_current ‖ z_next]`. |

### Data & loss flow (`jepa_train.py`, `dataset_JEPA.py`)

- Input contract **preserved**: `(B, 4, 84, 84)` float in `[0,1]` + discrete `a_t` in `[0,18)`
  (`dataset_JEPA.py:96-99`). Channels-last `.npz` transposed HWC→CHW.
- Loss (`jepa_train.py:119`): `loss = jepa_loss + 0.1 * inv_loss`, mirrored in validation (`:168`).
  - `jepa_loss` = masked MSE between `z_next_pred` (predictor) and `z_next_target` (EMA target), meaned over the 36×256 grid, terminal-masked (`:110-112`).
  - `inv_loss` = masked CE of `compute_inverse_loss(z_context, z_target, a_t)` (`:115`).
- Terminal masking: `mask=0` on terminal transitions (`dataset_JEPA.py:99`) — loss never crosses episode boundaries.
- AdamW `3e-4`, cosine schedule, AMP, `grad_clip=1.0`, batch 512, **`steps_per_epoch=1000`** cap
  (only ~512k of the 2.75M transitions seen per epoch), 100 epochs.

### Gradient-leak check — **PASS**
Target encoder is severed three ways: `requires_grad=False` at construction (`:142-143`), forward
pass wraps it in `torch.no_grad()` (`:207`), and the EMA update is `@torch.no_grad()` (`:152`). No
gradient reaches the target network. This is correct and not a finding.

### Action-injection path — the collapse-vulnerable seam
Action enters as a **single prepended token** into a depth-3 predictor over 37 tokens, then is
discarded from the output (`:104-117`). A single soft token is cheap for attention to route
around; this is precisely the surface `eval_JEPA.py`'s counterfactual test (`:105-136`) exists to
police (`delta = sim_true − sim_wrong`, PASS if `> 0.02`).

### Anchor drift discovered (findings in their own right)
1. **Predictor/encoder depth is 3 vs 6**, not "3 vs 4" (skill hint) nor the class default 4/4 — the
   instantiation at `JEPA.py:133-134` overrides the defaults. Asymmetry ratio is 1:2.
2. **Dataset no longer holds full arrays in RAM.** It extracts each `.npz` once, then serves from a
   memory-mapped `.npy_cache` (`mmap_mode='r'`) with a per-sample `.copy()` to sever the mmap view
   (`dataset_JEPA.py:51-89`). The RAM-bottleneck anchor is stale; the live bottleneck is random-access
   SSD I/O on 84×84×4 slices, not memory.
3. **Inverse head's "next" state is the EMA target, detached** (`z_target`, `jepa_train.py:115`), not a
   second context-encoder pass. Inverse-dynamics gradient reaches the encoder **only through the
   current-state path** (`z_context`). Half the intended signal is severed.
4. **JEPA is not plumbed into the downstream at all.** `PPO/factory.py`'s `UniversalFeatureExtractor`
   knows only `NatureCNN`, `FSQ-VAE`, `Residual-VQVAE` (`:43-50`). **There is currently no code path
   from a trained JEPA checkpoint to a PPO number.** Every downstream claim below is, today,
   unmeasurable. This is the single most important gap.

### Data reality (grounds every "3+ games" requirement below)
`atari-DQN/custom_datasets/train/` contains **11 games**: BeamRider, Breakout, DemonAttack, Enduro,
MsPacman, Pong, Qbert, Riverraid, RoadRunner, Seaquest, SpaceInvaders. Cross-game train/probe/eval is
fully feasible with data already on disk.

---

## 2. Transcribed debate (researcher → reviewer → resolution)

Both personas are the analyst, run in sequence. The reviewer reads and answers the researcher's
*specific* proposals; the standing benchmark demand is: **any change must beat the FSQ-VAE /
Residual-VQVAE extractors already in `PPO/factory.py` on downstream PPO across ≥3 games, or it does
not ship.**

---

### Proposal P0 — Wire JEPA into `UniversalFeatureExtractor` (prerequisite)

**🔬 Researcher.** Nothing else can be evaluated until a `"JEPA"` branch exists in
`PPO/factory.py:43-50`. Add it: instantiate `context_encoder`, load the checkpoint (strip
`_orig_mod.`, `strict=False` like the others), and in `forward` return **mean-pooled** tokens
`z.mean(dim=1)` → `feature_dim=256` (flattening the 36×256 grid gives 9216-d, which bloats the
actor/critic `Linear` and biases the comparison toward JEPA on raw width). Respect `freeze_weights`.

**🥊 Reviewer.** No objection to *existence* — it's the precondition for the whole research question,
and without it P1–P7 are unfalsifiable. Two constraints on the *how*: (a) mean-pool vs flatten is not
a free choice — it changes what "JEPA features" even means, so **fix one pooling and hold it constant
across all downstream comparisons**, and report it. (b) The baselines (`FSQ-VAE`, `Residual-VQVAE`)
flatten their quantized grid (`factory.py:80,85`); to keep the comparison honest, either give every
extractor the same pooling or explicitly control for feature width. Otherwise you'll measure pooling,
not representation quality.

**✅ Resolution — KEPT (must-do first).** Add the branch; default to **mean-pool → 256**; document
pooling; when comparing to FSQ/Residual, note their feature widths and, if they differ materially,
run an ablation with matched widths. Survives because it's a precondition, not a claim.

---

### Proposal P1 — Fix the inverse-dynamics target, then tune its weight

**🔬 Researcher.** Two coupled issues at `jepa_train.py:115` and `JEPA.py:225`. First, the inverse
head is fed `z_target` (EMA, detached) as the "next" state (drift finding #3), so gradients only
shape the encoder through the current frame — the regularizer is running at half strength. Feed it a
**context-encoder pass of `s_next`** so both states carry gradient. Second, the `0.1` weight
(`:119,168`) is an unjustified constant. Replace with a **warmup schedule**: 0 for the first ~5
epochs (let JEPA physics stabilize), then ramp to a tuned value, sweeping `{0.05, 0.1, 0.25, 0.5}`.

**🥊 Reviewer.** I'll split this. The **weight sweep is fine** — cheap, and the `0.1` genuinely is a
magic number; but it is **not shippable without the counterfactual gate**: for each weight, report
`eval_JEPA.py`'s action-delta *and* downstream PPO. Too high and the encoder becomes an action
classifier that discards scene structure (the reviewer's standing fear); the sweep is only meaningful
if both metrics are on the table.

The **gradient-path change I partly resist.** Feeding a fresh `context_encoder(s_next)` doubles
encoder forward passes per step (a real ~1.5× compute hit on the dominant cost) and risks a shortcut:
if inverse dynamics can pull *both* latents, the easiest solution is to make `z` linearly action-
decodable everywhere, which is closer to action collapse's cousin — over-encoding the action at the
expense of transferable scene features. The current asymmetric form (grad only through `s_t`) is
arguably a *feature*. **Counter-proposal:** keep the detached target; instead, first *measure* whether
the regularizer is doing anything at all — ablate `0.1` vs `0.0` and read the action-delta. If
`0.0` already passes the counterfactual test, the whole regularizer is redundant compute and the
gradient-path debate is moot.

**✅ Resolution — MODIFIED.** Ship the **weight sweep `{0.0, 0.05, 0.1, 0.25}` with warmup**, gated on
**both** counterfactual action-delta and downstream PPO. **Drop** the dual-gradient change for now;
promote it only if the `0.0` ablation shows the regularizer matters *and* the sweep shows the detached
form saturating below a useful action-delta. Survives the "just adds compute" objection because the
`0.0` arm makes the regularizer's value *falsifiable* before any compute is spent scaling it.

---

### Proposal P2 — Stronger action conditioning (FiLM at every predictor block)

**🔬 Researcher.** A single prepended token (`JEPA.py:104-117`) is the weakest possible conditioning
and the most collapse-prone. Replace/augment with **FiLM**: project `action_embed(a_t)` to per-block
`(γ, β)` and modulate every `TransformerPredictor` layer's hidden state. The action then cannot be
routed around — it multiplicatively gates every token at every depth.

**🥊 Reviewer.** **Reject as a first move — this is a solution to a problem you have not shown you
have.** The counterfactual test (`eval_JEPA.py:105-136`) exists precisely to tell you whether action
conditioning is weak. If the current prepended-token model already scores action-delta `> 0.02`,
FiLM is pure architecture-tourism compute with no target to beat. Sequencing: run the counterfactual
diagnostic on the *existing* checkpoint **first**. Only if delta is in the `⚠️ WARNING`/`❌ FAIL`
band does FiLM earn a place. And even then, the leaner fix is P1's regularizer weight, not new
parameters — try turning the knob you already have before bolting on FiLM.

**✅ Resolution — CONDITIONAL / DEFERRED.** Gate strictly on the diagnostic. Run counterfactual on the
current best checkpoint; **if** action-delta `< 0.02`, escalate — but P1 (regularizer weight) is tried
*before* P2 (new architecture), and FiLM ships only if it beats the prepended-token baseline on
action-delta **and** doesn't regress downstream PPO. Survives only as a contingency, explicitly
subordinate to a measurement it hasn't passed.

---

### Proposal P3 — Fix the trivial-prediction leak from overlapping frame stacks

**🔬 Researcher.** This is the sharpest representational issue and it's structural. `s_t` and `s_next`
are **4-frame stacks that overlap by 3 frames** — three of the four target frames are already present
in the input. The predictor can score high cosine similarity by near-identity, so `jepa_loss` and the
drift metric are both inflated and the learning signal is weak. Options: (a) predict a **farther
horizon** (`s_{t+k}`, k≥4, non-overlapping) — needs a `k`-step target in `dataset_JEPA.py`; or
(b) predict only the **newest frame's** latent rather than the whole stack.

**🥊 Reviewer.** **This one I endorse, and I'll sharpen it into the most important representational
change on the board.** It's not compute overhead — it's a validity fix: today's headline drift/cosine
numbers may largely reflect frame-stack overlap, not learned dynamics. Diagnostic to prove it exists
*before* building anything: compute the cosine similarity of a **null predictor** (`z_pred := z_t`,
identity) against `z_target` on the val set. If identity already scores high, the task is partly
trivial and every metric in `eval_JEPA.py` is confounded. Caveats on the fix: (a) multi-step targets
raise per-sample I/O on the mmap dataset (drift finding #2) — index carefully so you don't fetch two
full stacks per item; (b) don't over-correct into a horizon so long the target is unpredictable and
you just reintroduce noise. Prefer **k = 4** (first fully non-overlapping stack) over exotic values.

**✅ Resolution — KEPT, HIGH PRIORITY.** First run the **identity-predictor confound check**; if
confirmed, switch training + eval to a **k=4 non-overlapping target** in `dataset_JEPA.py`. Evaluation:
re-baseline the drift/counterfactual numbers and, decisively, downstream PPO across ≥3 games (Breakout,
Pong, Seaquest — data present). Survives the "just compute" objection because the reviewer *co-signed*
it as a validity fix, and the identity check makes the payoff falsifiable up front.

---

### Proposal P4 — Add I-JEPA-style latent masking of context tokens

**🔬 Researcher.** There is currently **no masking** — the context encoder sees the full frame and the
predictor predicts the full next frame. Classic I-JEPA masks a subset of context tokens and predicts
them, which is what forces *representation* rather than *copying*. Add a mask ratio (e.g. 0.4–0.75) over
the 36 tokens in `TransformerEncoder`/`TransformerPredictor`.

**🥊 Reviewer.** **Careful — this conflates two different JEPAs.** I-JEPA masks *within a single image*
(spatial context→target). This project is **V-JEPA-style action-conditioned temporal prediction**
(`s_t, a_t → s_next`); the predictive signal already comes from the *temporal* gap, not from spatial
masking. Bolting spatial masking on top is a second, partially redundant pretext task, and at 36 tokens
the granularity is coarse (each token is a 14×14 patch — masking 40% throws away most of the tiny
screen). Bigger objection: **it's untestable in isolation** stacked on P3. Sequence it — if P3 (a
non-trivial temporal target) already yields good linear-probe accuracy across games, spatial masking is
solving a problem P3 already solved. **Reject for the first roadmap cut.**

**✅ Resolution — DROPPED (for now).** Not a first-cut change. Revisit only if, after P3, linear-probe
accuracy across games remains weak *and* an ablation isolates masking's contribution. Did **not** survive:
redundant with the temporal task, coarse at 36 tokens, and untestable stacked on P3.

---

### Proposal P5 — EMA `tau` warmup schedule

**🔬 Researcher.** `tau=0.996` is fixed (`JEPA.py:127,133`). I-JEPA/V-JEPA ramp `tau` from ~0.996→1.0
over training so the target stabilizes as the encoder matures. Add a cosine `tau` schedule in
`update_target_network()`.

**🥊 Reviewer.** Low-risk, low-reward, and **low-priority**. It won't cause action collapse or a gradient
leak, and it won't move cross-game transfer enough to matter before P0/P1/P3 land. Accept it only as a
**free rider** bundled into another run — never as a standalone experiment consuming a training slot.
No dedicated evaluation; just don't let it regress the P3 re-baseline.

**✅ Resolution — KEPT as a free rider.** Fold a `0.996→1.0` cosine `tau` ramp into the P3 training run;
no standalone experiment. Survives only because it's nearly free and can't introduce a failure mode.

---

### Proposal P6 — Downstream usage: linear probe vs fine-tune vs joint auxiliary

**🔬 Researcher.** This *is* the research question. Three regimes on top of the P0 branch: (a) **frozen
+ linear probe** (`freeze_weights=True`), (b) **fine-tune** the encoder end-to-end with PPO, (c)
**joint** — keep the JEPA loss as an auxiliary term alongside PPO. Propose building toward (c) since
joint training could let the representation adapt to reward-relevant features.

**🥊 Reviewer.** Order matters and the researcher has it backwards. **Start frozen (a).** The entire
premise of a "Universal Feature Extractor" is a *frozen* representation reused across games — if frozen
JEPA can't beat frozen FSQ-VAE/Residual-VQVAE (`PPO/factory.py`) on linear-probe/PPO across ≥3 games,
the method has *failed on its own terms* and joint training is just an expensive way to launder a bad
representation into a NatureCNN-with-extra-steps. Joint training (c) also silently reintroduces a
**gradient-leak surface** and confounds "is the representation good?" with "did PPO fix it?" — you can
no longer attribute the win. **Frozen linear-probe is the honest, cheap, decisive test and it comes
first.** Only escalate to fine-tune/joint if frozen is competitive-but-trailing.

**✅ Resolution — KEPT, RE-ORDERED.** Evaluate in the order **frozen probe → fine-tune → joint**, each
gated on beating the frozen VAE-family baselines across ≥3 games. Joint is the *last* resort, not the
goal. This directly informs the open question in §4. Survives, but with its priority inverted by the
reviewer.

---

### Proposal P7 — Predictor/encoder depth sweep (3:6 asymmetry)

**🔬 Researcher.** Depth is 3:6 (drift finding #1). An over-strong predictor can absorb dynamics the
encoder should learn, hollowing out the representation. Sweep predictor depth `{2,3,4}` against encoder
depth `{4,6}`.

**🥊 Reviewer.** **Reject as premature.** A 2×2 depth grid is 4 full pretraining runs to chase a
second-order effect while P0 (no downstream path), P1 (untuned regularizer), and P3 (confounded metric)
are unresolved. Depth tuning on top of a confounded objective (P3) just fits hyperparameters to noise.
Defer until the objective is valid and the downstream harness exists; then, if at all, sweep *one* axis
(predictor depth `{3,4}`) not a grid.

**✅ Resolution — DEFERRED.** Off the first-cut roadmap. Revisit after P3, single-axis only, if
downstream results plateau. Did not survive as a near-term item — cost too high, effect too small,
objective not yet trustworthy.

---

## 3. Roadmap (surviving proposals, execution order)

Each item states its evaluation metric and baseline. **The universal baseline for every downstream
claim: frozen FSQ-VAE and Residual-VQVAE extractors already in `PPO/factory.py`, plus the NatureCNN
control, on PPO across ≥3 games (Breakout, Pong, Seaquest — data on disk).**

| # | Change | Files touched | Evaluation metric | Baseline / gate |
|---|---|---|---|---|
| **1. P0** | Add `"JEPA"` branch to `UniversalFeatureExtractor`; mean-pool → 256; respect `freeze_weights` | `PPO/factory.py` | Sanity: forward runs, shapes match; pooling documented | Precondition — unblocks all downstream metrics |
| **2. P3-check** | Identity-predictor confound check (`z_pred:=z_t` vs `z_target`) | new eval script (extends `eval_JEPA.py`) | Cosine sim of null predictor | If high → objective is confounded, proceed to P3-fix |
| **3. P3-fix** | k=4 **non-overlapping** target | `dataset_JEPA.py` (+ re-train) | Re-baselined drift + counterfactual; **downstream PPO ×3 games** | Must beat frozen FSQ/Residual-VQVAE |
| **4. P1** | Inverse-weight sweep `{0.0, 0.05, 0.1, 0.25}` + ~5-epoch warmup | `jepa_train.py:119,168` | `eval_JEPA.py` action-delta **and** downstream PPO | `0.0` arm proves regularizer earns its compute |
| **5. P6** | Downstream regimes **frozen → fine-tune → joint** | `PPO/factory.py`, PPO train loop | Linear-probe acc + PPO sample-efficiency ×3 games | Frozen must be competitive before escalating |
| **6. P5** | `tau` `0.996→1.0` cosine ramp (free rider on step 3) | `JEPA.py:152` | Must not regress the P3 re-baseline | Bundled, no standalone run |
| — **P2** | FiLM conditioning | `JEPA.py:74-118` | Action-delta vs prepended-token | **Contingent** on P3-era counterfactual `< 0.02` |
| — **P4** | Spatial token masking | `JEPA.py` | Linear-probe contribution ablation | **Deferred** — redundant with temporal task |
| — **P7** | Depth sweep | `JEPA.py:133-134` | Downstream PPO | **Deferred** — revisit only if results plateau |

**Critical-path reading:** P0 → P3-check → P3-fix → P1 → P6. Everything else is contingent or bundled.
Do **not** run architecture experiments (P2/P4/P7) before the objective is de-confounded (P3) and a
downstream number exists (P0), or you will be tuning hyperparameters against a confounded metric.

---

## 4. Open questions (genuine non-convergence)

1. **Independent vs joint pretraining — unresolved by design.** The reviewer forced the *order*
   (frozen probe first, joint last) but not the *answer*. Frozen best serves the "Universal Feature
   Extractor across all games" thesis and keeps attribution clean; joint could capture reward-relevant
   features a frozen SSL encoder misses. This only resolves empirically at step 5 (P6), and the honest
   prior is: if frozen JEPA cannot match frozen FSQ-VAE, the joint result — even if good — no longer
   answers "is JEPA a better representation?" The debate did **not** manufacture a winner here.

2. **Is the inverse-dynamics regularizer necessary at all?** The `0.0` arm (P1) may reveal the
   prepended-token predictor already passes the counterfactual test, making the whole regularizer —
   and the P2 FiLM contingency — redundant compute. Undetermined until measured.

3. **Does JEPA beat the VAE family on downstream RL at all?** The reviewer's structural challenge stands
   unanswered: FSQ-VAE and Residual-VQVAE are already plumbed, frozen, and cheaper. JEPA's entire
   justification rests on a cross-game PPO comparison that **has never been run** (drift finding #4).
   Until P0+P3 land and that number exists, JEPA's advantage is a hypothesis, not a result.

4. **Pooling choice (mean vs flatten) as a confound.** Fixed by fiat for comparability (P0), but whether
   mean-pooling discards spatially-localized information that PPO needs is untested — an ablation, not a
   settled decision.
