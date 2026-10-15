# Theoretical Validation Tribunal: An Action-Conditioned JEPA for Atari Representation Learning

**Convened as:** ICLR/NeurIPS-style Area Chair review
**Subject:** `atari-DQN/JEPA.py` (`PaperAccurateJEPA`) and `atari-DQN/jepa_train.py`
**Supporting evidence:** `atari-DQN/dataset_JEPA.py`, `atari-DQN/atari-replay-dataset_full.py`, `atari-DQN/atari-replay-dataset_small.py`, `atari-DQN/config.py`, `atari-DQN/eval_JEPA_new.py`
**Author's claim under review:** that this pipeline constitutes a *Universal Feature Extractor* for Atari, fusing passive video-prediction (V-JEPA) with active, action-aware World Models, stabilised by an Intrinsic-Curiosity / Inverse-Dynamics module.

**Scope.** This is a theoretical adjudication. No code was modified. Where a defect is named, it is named because it changes what the architecture *is* — not as a patch to apply. Implementation of anything proposed in §3 requires a separate, explicit instruction.

**Evidentiary note.** Three contested questions in this review are empirical, not interpretive, and were settled by direct measurement of the artefacts on disk rather than by argument. Those measurements are reported in §1.4 and are load-bearing for §2.2 and §2.3. One claim in this document — the *semantic* mapping of ALE action indices to joystick commands — could not be executed in this environment (no `gymnasium`/`ale_py` installed) and is drawn from ALE's documented action enumeration; it is flagged as such at the point of use.

---

## 1. The Architecture Under Review

### 1.1 What the code actually implements

Let $s_t \in [0,1]^{4\times84\times84}$ be a four-frame Atari stack, $a_t \in \{0,\dots,17\}$ a discrete action, and $s_{t+1}$ the successor stack. The model defines three parameterised maps and one auxiliary head:

| Symbol | Module | Depth | Trainable |
|---|---|---|---|
| $f_\theta$ | `context_encoder` (`TransformerEncoder`) | 6 | yes |
| $f_\varphi$ | `target_encoder` (deep copy of $f_\theta$) | 6 | **no** |
| $p_\psi$ | `predictor` (`TransformerPredictor`) | 3 | yes |
| $g_\xi$ | `inverse_head` (2-layer MLP) | — | yes |

**Tokenisation.** `PatchEmbedding` (`JEPA.py:6`) applies a `Conv2d(4, 256, kernel_size=14, stride=14)` — a non-overlapping partition of the $84\times84$ field into a $6\times6$ grid — yielding $N=36$ tokens of width $D=256$, plus a learned positional embedding. Note the temporal axis is *not* tokenised: the four stacked frames enter as convolutional input channels, so time is collapsed into the patch projection at layer zero. The sequence the transformer reasons over is purely spatial.

**Encoders.** Both $f_\theta$ and $f_\varphi$ are pre-LN transformer stacks (`norm_first=True`) terminated by an `nn.LayerNorm(256)` with learnable affine parameters $(\gamma,\beta)$. Output: $z \in \mathbb{R}^{B\times36\times256}$.

**Action-conditioned prediction.** `TransformerPredictor.forward` (`JEPA.py:99–118`) embeds $a_t$ via `nn.Embedding(18, 256)`, prepends it as token $0$ to give a length-37 sequence, applies three pre-LN blocks with *full bidirectional self-attention*, then slices the action token off the output:

$$\hat z_{t+1} = \Pi_{1:37}\big(p_\psi([\,e(a_t)\,;\,z_t\,])\big) \in \mathbb{R}^{B\times36\times256}.$$

Conditioning is therefore *soft and unstructured*: the action is a peer token in an unmasked attention set, free to be attended to or ignored by every spatial token. There is no FiLM-style modulation, no gating, no architectural guarantee that $a_t$ influences the output at all.

**Objective** (`jepa_train.py:110–119`), with terminal mask $m_i\in\{0,1\}$:

$$\mathcal{L} = \underbrace{\frac{\sum_i m_i \cdot \frac{1}{ND}\|\hat z^{(i)}_{t+1} - z^{(i),\text{tgt}}_{t+1}\|_2^2}{\sum_i m_i}}_{\mathcal{L}_{\text{JEPA}}} \;+\; \lambda \underbrace{\frac{\sum_i m_i \cdot \mathrm{CE}\big(g_\xi([\bar z^{(i)}_t ; \bar z^{(i),\text{tgt}}_{t+1}]),\, a^{(i)}_t\big)}{\sum_i m_i}}_{\mathcal{L}_{\text{inv}}}, \qquad \lambda = 0.1$$

where $\bar z = \frac{1}{36}\sum_{n} z_{:,n,:}$ is a uniform mean-pool over spatial tokens, and $z^{\text{tgt}}_{t+1} = f_\varphi(s_{t+1})$.

**Target update.** $\varphi \leftarrow \tau\varphi + (1-\tau)\theta$ with $\tau = 0.996$ held **fixed**, executed after every `optimizer.step()` (`jepa_train.py:130`).

**Optimisation.** AdamW ($\text{lr}=3\times10^{-4}$, weight decay $10^{-5}$), `CosineAnnealingLR`, AMP with `GradScaler`, global grad-norm clip at 1.0. Trainable parameters are collected as a **single group** — `filter(lambda p: p.requires_grad, model.parameters())` (`jepa_train.py:70`) — with no exclusion list.

### 1.2 The gradient-flow graph, in words

This is the crux of the review, so it is stated exhaustively.

- $s_t \to f_\theta \to z_t$. The context encoder receives gradient from **both** terms: from $\mathcal{L}_{\text{JEPA}}$ through the predictor, and from $\lambda\mathcal{L}_{\text{inv}}$ through the first half of the inverse head's input.
- $z_t, a_t \to p_\psi \to \hat z_{t+1}$. The predictor — **including its `action_embed` table** — receives gradient from $\mathcal{L}_{\text{JEPA}}$ **only**.
- $s_{t+1} \to f_\varphi \to z^{\text{tgt}}_{t+1}$. No gradient, by two independent mechanisms: `requires_grad=False` on every target parameter (`JEPA.py:142–143`) and a `torch.no_grad()` context around the forward call (`JEPA.py:207`). $\varphi$ changes only by EMA.
- $[\bar z_t ; \bar z^{\text{tgt}}_{t+1}] \to g_\xi$. The inverse head receives gradient from $\lambda\mathcal{L}_{\text{inv}}$ only, and propagates it backwards **only into $\theta$**, because its second input argument is the detached target (`jepa_train.py:115`).

Two consequences follow immediately and are not, as far as this review can determine, acknowledged anywhere in the codebase:

> **(G1) The inverse-dynamics loss can exert no gradient on the predictor or on the action embedding.** The module introduced to prevent the model from ignoring actions is structurally incapable of touching the only two components that consume the action.
>
> **(G2) The model contains two disjoint, never-tied representations of "action":** the 18×256 `action_embed` table inside $p_\psi$ (trained by $\mathcal{L}_{\text{JEPA}}$), and the 18-way output layer of $g_\xi$ (trained by $\mathcal{L}_{\text{inv}}$). Nothing in the objective couples them.

### 1.3 Anchor verification

Every structural anchor supplied to this tribunal was checked against the source and **confirmed**: depths 6/3; `embed_dim=256`; `tau=0.996` fixed with no schedule; `patch_size=14` → 36 tokens; action prepend-and-slice at `JEPA.py:99–118`; double-locked target branch; EMA after every optimiser step; $\mathcal{L} = \mathcal{L}_{\text{JEPA}} + 0.1\,\mathcal{L}_{\text{inv}}$ at `jepa_train.py:119`; inverse loss called with the detached target as its second argument at `jepa_train.py:115`; no variance or covariance term anywhere; AdamW/cosine/AMP/clipping at `jepa_train.py:69–83`. (The class *defaults* are `depth=4` for both encoder and predictor; the 6/3 asymmetry is imposed at construction in `PaperAccurateJEPA.__init__`, `JEPA.py:133–134`.)

### 1.4 Drift and findings beyond the anchors

Measured directly from the artefacts on disk:

**(D1) The frame-stack overlap is exactly 3/4, bit-identical.** For every transition sampled, `next_obs[..., :3] == obs[..., 1:]` holds with fraction $1.0$ — not approximately, but exactly. Three of the four channels of the prediction target are a verbatim memcopy of three of the four channels of the predictor's input. This is the standard `VecFrameStack` convention (Mnih et al., 2015) and is confirmed in both the CHW and HWC variants of the dataset. §2.2 rests on this number.

**(D2) The action space is per-game minimal and is being folded into a single 18-way embedding.** Measured index ranges across the 11 training games:

| Game | $|\mathcal{A}|$ | observed indices | $H(a\mid\text{game})$ [nats] |
|---|---|---|---|
| Breakout | 4 | 0–3 | 1.379 |
| Qbert | 6 | 0–5 | 1.373 |
| DemonAttack | 6 | 0–5 | 1.670 |
| SpaceInvaders | 6 | 0–5 | 1.732 |
| Pong | 6 | 0–5 | 1.771 |
| BeamRider | 9 | 0–8 | 1.933 |
| Enduro | 9 | 0–8 | 1.966 |
| MsPacman | 9 | 0–8 | 1.983 |
| RoadRunner | 18 | 0–17 | 1.421 |
| Riverraid | 18 | 0–17 | 2.677 |
| Seaquest | 18 | 0–17 | 2.845 |

The collection scripts call `make_atari_env` without `full_action_space=True`, so each game emits its *minimal* action set, compactly re-indexed. Row $k$ of `action_embed` is therefore shared between games in which index $k$ denotes different joystick commands. Per ALE's documented action enumeration — the full 18-set ordered `NOOP, FIRE, UP, RIGHT, LEFT, DOWN, …`, with minimal sets being that list filtered in place — index 3 denotes `LEFT` in Pong's and Breakout's 6- and 4-action sets but `RIGHT` in Seaquest's full-18 set. *(This semantic mapping is from ALE documentation; it could not be executed here. The index-range collision itself is measured and certain, and is sufficient for the argument.)* The dataset carries no game identifier, so the model cannot disambiguate even in principle.

**(D3) Shortcut floors for the inverse loss, measured.** Pooled action marginal $H(a) = 2.508$ nats; game-conditional $H(a\mid\text{game}) = 1.886$ nats; chance $\log 18 = 2.890$ nats. A model that learns *only* to recognise which game it is looking at — trivially separable from raw pixels — and emits that game's marginal action distribution attains $\mathcal{L}_{\text{inv}} \approx 1.886$, capturing $1.004$ of the $2.890$ available nats **with zero knowledge of dynamics**. Nothing in the training loop measures against this baseline.

**(D4) Weight decay is applied to the LayerNorm gains, positional embeddings, and action embedding.** The single-parameter-group construction at `jepa_train.py:70` gives AdamW's decoupled decay (Loshchilov & Hutter, 2019) authority over $\gamma$. §2.1 shows this opens a direct, actively-driven route to representation collapse.

**(D5) Model selection is performed against a non-stationary target.** `vjepa_best.pt` is written whenever `val_avg_loss` improves (`jepa_train.py:203–207`). But $\mathcal{L}_{\text{JEPA}}$ measures distance to $f_\varphi$, which is itself changing. The quantity is not comparable across epochs, and — per §2.1 — it is *minimised* by the degenerate solution. The checkpoint-selection criterion is aligned with the failure mode.

**(D6) The terminal mask is nearly inert.** Measured terminal fraction $\approx 6\times10^{-4}$. The masking machinery, though correct, conditions roughly one sample in 1,700; it cannot be credited with any material effect on the learned representation.

**(D7) The inverse head mean-pools over all 36 spatial tokens.** `z.mean(dim=1)` (`JEPA.py:232–233`). Atari action effects are spatially localised to a handful of patches; uniform averaging attenuates a localised signal relative to global scene content. This is examined in §2.3.

**(D8) Train and validation splits have incompatible memory layouts.** `custom_datasets/train` stores `obs` as $(N,4,84,84)$ (CHW — the full collector transposes at write time, `atari-replay-dataset_full.py`), while `custom_datasets_small/{train,val,test}` store $(N,84,84,4)$ (HWC). `dataset_JEPA.py:92` applies `np.transpose(raw_obs, (2,0,1))` unconditionally. On the HWC files this is correct; on the CHW files it maps $(4,84,84)\mapsto(84,4,84)$, which cannot enter a 4-channel convolution. Compounding this, `config.py` sets `TRAIN_DIR = "atari-DQN/custom_datasets/train"` (resolvable from the repo root) and `VAL_DIR = "custom_datasets_small/val"` (resolvable from `atari-DQN/`), so no single working directory resolves both. The theoretical consequence is one of *provenance*: the 2.75 M-transition corpus named in the W&B config is not the corpus the loader can consume, and any reported result must be attributed to the small HWC splits before it can be interpreted. Further, `custom_datasets_small/val` contains the same eleven games as training — it measures interpolation within the training distribution, not generalisation. The four held-out games (`Alien`, `Asteroids`, `Atlantis`, `IceHockey`) sit in `test`, unused by `jepa_train.py`.

**(D9) Evaluation sequences may splice across game boundaries.** `extract_unbroken_sequences` (`eval_JEPA_new.py`) draws `start_idx … start_idx+horizon` from the concatenated multi-file index and validates only the terminal mask (`mask == 0`), which per (D6) fires on $0.06\%$ of steps. A window straddling the boundary between, say, Pong's shard and Qbert's shard passes the check. The drift metric's sample is contaminated by transitions that are not transitions.

**(D10) No representation-health statistic is logged anywhere.** The training loop records total, JEPA, and inverse losses. It records no embedding variance, no rank, no off-diagonal correlation, no cosine similarity between $z_t$ and $z_{t+1}$. A total collapse of the representation would present in the logs as an excellent, monotonically-decreasing loss curve.

**(D11) Environment determinism departs from the standard ALE protocol.** Both collectors set `repeat_action_probability=0.0`, disabling sticky actions. Machado et al. (2018) introduced stickiness precisely to prevent open-loop memorisation on the deterministic ALE. This makes the corpus maximally favourable to the degenerate solutions prosecuted below.

---

## 2. Transcript of the Debate

### 2.1 First front — the target encoder and the collapse argument

**VALIDATOR.** The architecture is a faithful instantiation of the joint-embedding predictive programme set out in LeCun (2022) and realised in I-JEPA (Assran et al., 2023) and V-JEPA (Bardes et al., 2024). The essential move — predict in *representation* space rather than pixel space — is precisely what those works argue lets the model discard unpredictable low-level detail and retain semantic structure. The asymmetric encoder/predictor depth (6 vs 3) reproduces the standard JEPA ratio, forcing representational work into the encoder rather than the dynamics head. The EMA target is the BYOL mechanism (Grill et al., 2020), and it is here double-locked: `requires_grad=False` *and* `torch.no_grad()`. Gradient cannot reach the target branch by any path.

The anti-collapse argument runs as follows. Write the objective as $\mathcal{L}(\theta,\psi;\varphi) = \mathbb{E}\|p_\psi(f_\theta(s),a) - f_\varphi(s')\|^2$. If the target were $f_\theta$ itself, the degenerate solution $f_\theta \equiv c$ would be reachable by direct descent — the gradient would actively pull predictions and targets toward each other, and the shortest path is mutual contraction. Under the EMA construction, $\varphi$ is not a free variable of the minimisation; it is a low-pass filtered trajectory of $\theta$. The gradient $\nabla_\theta \mathcal{L}$ therefore chases a target that lags by an effective horizon of $(1-\tau)^{-1} = 250$ steps. The optimisation is a pursuit problem, not a mutual-contraction problem, and Tian et al. (2021) show for the linear-predictor case that the predictor's alignment with the target's correlation eigenspace induces dynamics under which the collapsed subspace is not attracting.

**SKEPTIC.** You have argued that collapse is *not the fastest descent direction*. You have not argued that it is not a solution. Let me state the fixed point precisely, because your own construction concedes it.

Take $\theta^\star$ such that $f_{\theta^\star}(\cdot) \equiv c$ for all inputs. Set $\varphi = \theta^\star$ — which the EMA reaches exactly, being a convex combination of identical points. Choose $\psi^\star$ such that $p_{\psi^\star}(c, a) = c$ for every $a$; a transformer with a final LayerNorm whose gain is zero and bias is $c$ does this exactly. Then $\mathcal{L}_{\text{JEPA}}(\theta^\star,\psi^\star;\varphi^\star) = 0$, the global infimum, and $\nabla_\theta\mathcal{L}_{\text{JEPA}} = \nabla_\psi\mathcal{L}_{\text{JEPA}} = 0$. The EMA update is also stationary, since $\varphi = \theta$. It is a fixed point of the *entire* system, target update included. So the honest claim is not "EMA prevents collapse" but "EMA may make collapse dynamically unattractive under conditions." That is a materially weaker thesis, and the conditions matter.

Now — the conditions. Tian et al. (2021) analyse a **linear** predictor, and their result depends explicitly on the interaction of predictor learning rate, EMA rate, and *weight decay* on the eigenvalue dynamics of the correlation matrix. Your predictor is three layers of pre-LN transformer with GELU, sitting far outside that analysis. You inherit the conclusion without inheriting the hypotheses.

And here is the part I want on the record. Both encoders terminate in `nn.LayerNorm` with learnable affine parameters. The encoder output is
$$z = \gamma \odot \mathrm{LN}(h) + \beta.$$
As $\gamma \to 0$, $z \to \beta$ — a constant, for every input, every patch, every game. That is exactly the collapsed solution $c = \beta$. And $\mathcal{L}_{\text{JEPA}}$, being a raw squared error on unnormalised embeddings, *shrinks quadratically in $\|\gamma\|$*. There is no scale-invariance in this loss; there is no cosine, no normalisation, no variance floor.

Now recall (D4): `jepa_train.py:70` builds a single AdamW parameter group with no exclusion list, so decoupled weight decay applies to $\gamma$. The decay dynamics are
$$\gamma \leftarrow (1-\eta\lambda_{\text{wd}})\,\gamma - \eta\,\widehat{\nabla_\gamma \mathcal{L}}.$$
The first term is an unopposed geometric contraction toward the collapse manifold. The second term is the only thing resisting it — and $\nabla_\gamma\mathcal{L}_{\text{JEPA}}$ *also* points toward smaller $\gamma$, because shrinking the latent scale reduces the squared error. Both terms push the same way. The sole counterpressure in the entire objective is $\lambda\nabla_\gamma\mathcal{L}_{\text{inv}}$, since cross-entropy through the inverse head genuinely requires a non-degenerate $\bar z_t$.

So the true statement about this architecture is: **the representation is held away from collapse by $\lambda = 0.1$ and by nothing else.** The EMA is not the anti-collapse mechanism; a hardcoded constant with no stated derivation is. And per (D10), no logged metric would reveal it if that balance failed — and per (D5), `vjepa_best.pt` would preferentially save the most collapsed checkpoint, because collapse *is* the minimum of the selection criterion.

**VALIDATOR.** I concede the fixed-point characterisation; "prevents collapse" was too strong, and I will restate it as "renders collapse dynamically disfavoured under conditions not verified here." I also concede the weight-decay channel, which I had not considered and which I agree is a real gradient path rather than a hypothetical one.

I will contest one thing and qualify another. The contest: $\lambda\mathcal{L}_{\text{inv}}$ is not a *weak* counterpressure. Cross-entropy diverges as the representation degenerates — if $\bar z_t$ carries no information, CE is pinned at its shortcut floor and the gradient with respect to $\gamma$ grows relative to a vanishing $\mathcal{L}_{\text{JEPA}}$. There is a genuine restoring force, and it strengthens exactly as collapse deepens. The equilibrium is a balance, not a slide.

The qualification: $\lambda_{\text{wd}} = 10^{-5}$ is small, and 100 epochs × 1000 steps is 10⁵ updates, giving a contraction factor of order $(1 - 3\times10^{-4}\cdot10^{-5})^{10^5}$ — negligible on its own. The decay channel is real but is not, by itself, fast. My concern is now the *other* term you identified: $\nabla_\gamma \mathcal{L}_{\text{JEPA}}$ pointing toward smaller $\gamma$ is not small, and it operates at the full learning rate.

**SKEPTIC.** Accepted, and that concession sharpens the finding rather than blunting it: the dominant collapse pressure is the loss itself, not the regulariser. Which means it cannot be dismissed as a hyperparameter accident. And note what your restoring-force argument actually establishes — that the equilibrium latent scale is *set by* $\lambda$. Change $\lambda$ and you change the norm of the learned representation. That is a strange property for a quantity the author describes as a minor auxiliary weight, and it is the subject of the third front.

**AREA CHAIR.** Recorded. The tribunal converges on: collapse is a global minimum and a fixed point of the full system including the EMA; the EMA argument is dynamical, conditional, and imported from a regime (linear predictor) that does not obtain here; the only term in the objective with an interest in a non-degenerate representation is $\lambda\mathcal{L}_{\text{inv}}$; and no instrument in the pipeline would detect the failure. The disputed residual — whether the restoring force is sufficient in practice — is *not resolvable by argument* and is recorded as an open empirical question in §4.

### 2.2 Second front — action injection and the lazy identity function

**VALIDATOR.** Conditioning latent dynamics on action is the defining move of the World Models line: Ha & Schmidhuber (2018) train an action-conditioned MDN-RNN over a learned latent; PlaNet (Hafner et al., 2019) and Dreamer (Hafner et al., 2020) make the action-conditioned latent transition the object of learning. Injecting the action as a sequence token lets attention route the action's influence to the spatial patches it actually affects, rather than broadcasting it uniformly — which is more expressive than concatenation and is the natural transformer idiom. The 36-token spatial output preserves locality, so the model can in principle represent "this action moved the sprite in patch 14."

**SKEPTIC.** In principle. Let me put the measurement on the table.

Between $s_t$ and $s_{t+1}$, **three of four frames are bit-identical** — verified exactly, fraction $1.0$, not approximately (D1). The target the model must produce is an encoding of a stack that is 75% a verbatim copy of the stack it was given. Consider the candidate solution $p_\psi(z, a) = z$ — the identity, ignoring the action token entirely, which the architecture permits since the action is merely one of 37 tokens in an unmasked attention set with nothing forcing attention onto it. Under this solution the residual error is
$$\mathbb{E}\big\|f_\theta(s_t) - f_\varphi(s_{t+1})\big\|^2,$$
which is small for any encoder that is Lipschitz in its input, precisely because the inputs differ in one channel of four. Now add: $f_\varphi$ is an EMA copy of $f_\theta$, so the two encoders are near-identical; and $\tau = 0.996$ makes them *more* alike, not less. The identity solution is not a pathology I am inventing — it is a low-loss basin that the data geometry, the encoder coupling, and the architecture jointly carve out.

Ask what the action must actually explain. The information content of $a_t$ is confined to its effect on the *single* new frame, further attenuated by frame-skip 4 with max-pooling, and then averaged with three frames of unchanged history inside the patch projection at layer zero — because, note, this architecture does not tokenise time at all; the four frames are input channels, so the temporal difference is smeared into the very first convolution. The signal the predictor would need is at its weakest exactly where the architecture needs it strongest.

**VALIDATOR.** The identity is not a *zero*-loss solution, and the gradient of the residual is what drives learning. And the counterfactual test in `eval_JEPA_new.py` is designed to detect precisely this — it compares prediction under the true action against predictions under all 17 alternatives.

**SKEPTIC.** It compares them, and reports two cosine similarities. What it does not report is the control that would make those numbers interpretable: the cosine similarity between $f_\theta(s_t)$ and $f_\varphi(s_{t+1})$ — the identity baseline. Without it, a drift curve reading $0.98$ is consistent with an excellent world model and equally consistent with an encoder that has learned nothing but that consecutive frames look alike. The diagnostic and the pathology produce the same number. Per (D9), the sequences feeding that diagnostic can additionally splice across game boundaries.

Let me state the observable signature the tribunal should demand, since you asked what would distinguish the two hypotheses:

1. **The identity control.** $\cos(\hat z_{t+1}, z^{\text{tgt}}_{t+1})$ must be reported against $\cos(z_t, z^{\text{tgt}}_{t+1})$. If the learned predictor does not beat copying, there is no dynamics model. This single number decides the question and costs nothing.
2. **Action-conditional dispersion.** $\mathbb{E}_{a\neq a'}\|p_\psi(z,a) - p_\psi(z,a')\|$ against $\mathbb{E}\|p_\psi(z,a) - z\|$. If the spread across actions is small relative to the total predicted change, the action token is decorative.
3. **Attention mass on token 0.** Directly measurable: the fraction of attention weight the 36 spatial queries assign to the action key, per layer. Near-zero mass is dispositive.
4. **The disjoint-stack test.** Because the stack depth is 4, $s_t$ and $s_{t+k}$ share $\max(0, 4-k)$ frames — **zero** at $k=4$. Evaluating at $k=4$ removes the overlap confound entirely. This is a property of the data, not a modelling choice, and it is exactly why the fix in §3 is a horizon, not a regulariser.

**VALIDATOR.** I accept all four diagnostics, and I accept that the current evaluation cannot discriminate the hypotheses. I want to record one point in the architecture's favour that survives: the *residual* between $z_t$ and $z^{\text{tgt}}_{t+1}$, though small, is the part of the signal that is genuinely action-dependent, and MSE does place gradient on it. The concern is not that the model *cannot* learn dynamics — it is that the loss is dominated by a component it can obtain for free, so the learning signal for dynamics is a small fraction of a small number. That is a signal-to-noise argument, and it is the correct form of the objection.

**SKEPTIC.** Then we agree on the substance. I add only (D11): sticky actions are disabled, so the environment is deterministic. Machado et al. (2018) introduced stickiness specifically because deterministic ALE rewards open-loop memorisation. Every degenerate solution discussed here is *easier* on this corpus than on a standard-protocol one.

**AREA CHAIR.** Converged. The lazy-identity hypothesis is not refuted by anything currently in the pipeline, and the measured 3/4 bit-identical overlap gives it a quantitative foundation. The four diagnostics are adopted as the tribunal's required evidentiary standard for the claim "this is a world model."

### 2.3 Third front — the joint objective, the inverse head, and $\lambda$

**VALIDATOR.** The inverse head implements the ICM mechanism (Pathak et al., 2017). The argument is information-theoretic and clean: if the latent pair $(z_t, z_{t+1})$ suffices to recover $a_t$, then the representation must retain action-relevant, agent-controllable structure. ICM's central insight is that inverse dynamics filters out precisely the parts of the observation the agent cannot influence, because those parts carry no information about the action and are therefore not preserved by an inverse-dynamics-trained encoder. Applied here, this is a principled defence against action collapse and against the encoder wasting capacity on uncontrollable scenery. Schwarzer et al. (2021) demonstrate that this family — BYOL-style latent prediction with action conditioning — is effective on precisely this benchmark.

**SKEPTIC.** Three prosecutions. I will take them in order of increasing severity.

**(i) Incommensurability, and what $\lambda$ actually controls.** $\mathcal{L}_{\text{JEPA}}$ is a squared error in units of (latent scale)². $\mathcal{L}_{\text{inv}}$ is in nats. Their sum is dimensionally meaningless, but that alone is a pedantic objection; here is the substantive one. Consider the reparameterisation $z \mapsto \alpha z$, which the LayerNorm gains implement exactly. Under it, $\mathcal{L}_{\text{JEPA}} \mapsto \alpha^2 \mathcal{L}_{\text{JEPA}}$, while $\mathcal{L}_{\text{inv}}$ is essentially invariant — the inverse head's first linear layer absorbs $\alpha$ by rescaling its weights. Therefore $\lambda$ is **not** a task weight in any scale-free sense: it *selects the operating scale of the latent space*. It is the parameter that fixes the equilibrium the Validator identified in §2.1. A constant introduced as a minor auxiliary coefficient turns out to be the quantity that determines the norm of the representation, and it has no stated derivation.

Now the time-varying part, which I can bound rigorously. At initialisation, both branches are LayerNormed with unit gain, so per-dimension variance $\approx 1$ and the two are near-independent, giving $\mathcal{L}_{\text{JEPA}} \approx 2$; meanwhile $\lambda\mathcal{L}_{\text{inv}} = 0.1\log 18 \approx 0.289$. The objective starts at roughly $7:1$ in favour of prediction. But $\inf \mathcal{L}_{\text{JEPA}} = 0$ — attainable, per §2.1 — whereas $\mathcal{L}_{\text{inv}} \geq H(a \mid s_t, s_{t+1}) > 0$ strictly, whenever actions are aliased, which prosecution (iii) establishes they are. **The ratio therefore provably inverts.** The objective silently transitions from a world-model objective to an inverse-dynamics objective, and the transition point is set by nothing anyone chose. The gradients on the shared encoder $\theta$ are meanwhile in genuine tension: $\mathcal{L}_{\text{JEPA}}$ rewards contracting the latent scale, $\mathcal{L}_{\text{inv}}$ rewards expanding it. This is a textbook case for gradient surgery (Yu et al., 2020) or uncertainty weighting (Kendall et al., 2018), and neither is present.

**(ii) The gradient asymmetry.** `compute_inverse_loss(z_context, z_target, ...)` is called with the *target encoder's* output as its second argument (`jepa_train.py:115`), produced under `no_grad`. ICM derives $\phi(s_t)$ and $\phi(s_{t+1})$ from a single trainable encoder; the inverse loss shapes both. Here it shapes only $f_\theta(s_t)$ and $g_\xi$. The constraint "retain action-relevant information" is imposed on the representation of $s_t$ alone.

**VALIDATOR.** This one I will answer rather than concede. The asymmetry is largely self-healing. $f_\varphi$ is an EMA of $f_\theta$ — the same function, delayed by $\approx 250$ steps. Any action-relevant structure the inverse loss induces in $f_\theta$ propagates into $f_\varphi$ within a few hundred updates. The gradient path is absent; the *representational* consequence largely is not. I would rate this the weakest of your three.

**SKEPTIC.** I accept that, and I withdraw it as a first-order objection — with one substitution, which I consider stronger than the claim I am withdrawing. Look at what the inverse head is actually fed: `z.mean(dim=1)`, a uniform average over all 36 spatial tokens (D7). An Atari action displaces one sprite across a few patches. Averaging over 36 tokens attenuates a localised signal by roughly $1/36$ relative to global scene content that is identical in both frames — and since the head sees $[\bar z_t ; \bar z_{t+1}]$, the shared scene content is common-mode and the action signal is the difference of two heavily-diluted averages. **The inverse head is architecturally starved of the very signal it exists to extract.** That is not a gradient-routing subtlety; it is a pooling operator destroying the evidence.

And compounding it, per (G1): no gradient from $\mathcal{L}_{\text{inv}}$ reaches the predictor or the `action_embed` table. The stated purpose of this module is to prevent the model from ignoring actions. The module cannot reach either component that consumes an action. Per (G2), the model maintains two entirely separate notions of "action," trained by different losses, never tied. If the predictor learns to ignore token 0, the inverse head will not notice and could not object.

**(iii) The Noisy-TV vulnerability, in its Atari-specific form.** Inverse dynamics presumes $a_t$ is recoverable from $(s_t, s_{t+1})$. Burda et al. (2019) documented the failure mode when the observation contains variation the agent does not control. Here the environment is deterministic (D11), so the classical stochastic noisy-TV does not arise — but a more damaging variant does, and it is *measurable*:

- **Action aliasing.** ALE's action set contains actions with identical effects in most games. In Breakout after the ball is in play, `NOOP` and `FIRE` produce the same transition. In Pong, likewise. The inverse head is being asked to distinguish causes that produced identical effects. The Bayes-optimal answer is the conditional marginal over the aliased set, and the irreducible CE floor is strictly positive.
- **Uncontrollable dynamics dominate the pixel budget.** In Enduro and BeamRider, scrolling background and enemy motion account for most inter-frame change; the agent's contribution is a few pixels. The inverse head must find a small controllable signal inside a large uncontrollable one — and per (D7), it must do so after uniform spatial averaging has mixed them.
- **The action-space catastrophe (D2).** This is the one I want the committee to see. The eleven games contribute action sets of sizes 4, 6, 9, and 18, compactly re-indexed *per game*, folded into one 18-way embedding and one 18-way classification head, with no game identifier anywhere in the data. Embedding row 3 must simultaneously mean whatever index 3 means in Breakout's 4-action set, in Pong's 6-action set, and in Seaquest's full-18 set — and per ALE's enumeration those are not merely different commands but, between the minimal and full sets, *opposed* ones. The action embedding is not polysemous. It is contradictory.
- **And the shortcut is quantified (D3).** $H(a) = 2.508$ nats pooled; $H(a \mid \text{game}) = 1.886$ nats. A model that learns *only* to recognise the game — trivially separable from pixels, requiring no dynamics whatsoever — and emits that game's marginal action distribution captures $1.004$ of the $2.890$ available nats. That is what a falling `train/inverse_action_loss` curve is consistent with. There is no baseline logged against which to distinguish "the model learned inverse dynamics" from "the model learned to recognise Seaquest."

What does the head learn when the action is unidentifiable? It regresses to the conditional marginal — and because the gradient flows into $f_\theta$, it shapes the shared encoder to encode *whatever predicts the action prior*, which is game identity and coarse scene statistics. The mechanism intended to enforce controllable-state representation instead rewards game classification.

**VALIDATOR.** I concede (iii) substantially and (i) in full. On (iii) I would enter one mitigation: encoding game identity is not worthless for a *universal* extractor, and the ICM argument survives *within* a game even if it fails across the mixture. But I accept that as stated — single embedding table, no game conditioning, no per-game action canonicalisation — the multi-game training regime actively undermines the mechanism rather than merely diluting it. And I accept that $H(a\mid\text{game}) = 1.886$ is the number against which the reported inverse loss must be judged, and that no such comparison exists.

**AREA CHAIR.** Recorded, including the Skeptic's withdrawal of (ii) as originally framed and its replacement with the mean-pooling objection, which the Validator did not contest. The tribunal treats the action-space collision (D2) and the unmeasured shortcut floor (D3) as the most serious findings of this review, on the grounds that they are certain, quantified, and go directly to the central claim.

---

## 3. The Visionary's Proposal — A Next-Generation Action-Conditioned JEPA

Each component below answers a specific objection established above. Nothing is proposed for novelty's sake.

### 3.1 Two prerequisites that are not modelling choices

**P1 — Canonicalise the action space (answers D2).** Collect with `full_action_space=True`, or store the per-game index→ALE-18 map alongside the transitions and remap at load time. Without this, every action-conditioned quantity in the model is trained on contradictory labels, and no loss function can repair it. This is the single highest-value change in this document and it is a data-collection change, not an architectural one.

**P2 — Condition on game identity, or hold it out deliberately (answers D3).** Store a game id $c$ per transition. Then the inverse-dynamics baseline $H(a\mid c)$ becomes computable per batch, and the reportable quantity becomes the **information gain over the shortcut**, $H(a\mid c) - \mathcal{L}_{\text{inv}}$, rather than $\mathcal{L}_{\text{inv}}$ itself. A metric that cannot go up when the model learns nothing is not a metric.

### 3.2 The objective

Let $f_\theta$ be the online encoder, $f_\varphi$ its EMA target, $p_\psi$ the action-conditioned predictor, $q_\zeta$ an attention-pooling head, $g_\xi$ the inverse head. Let $\mathcal{K} \subseteq \{1,\dots,K\}$ be a set of prediction horizons with $\max \mathcal{K} \geq 4$. Write $\mathrm{sg}[\cdot]$ for stop-gradient and $B$ for batch size.

**Latent rollout.** Define $\hat z^{(0)} = f_\theta(s_t)$ and, autoregressively, $\hat z^{(k)} = p_\psi(\hat z^{(k-1)}, a_{t+k-1})$.

**(T1) Multi-step cosine prediction** — *answers §2.2 (lazy identity) and the $\gamma$-collapse channel of §2.1.*

$$\mathcal{L}_{\text{pred}} = \frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}} \frac{1}{N}\sum_{n=1}^{N} \Big(1 - \cos\big(\hat z^{(k)}_n,\; \mathrm{sg}[f_\varphi(s_{t+k})_n]\big)\Big)$$

Two changes from the current form, each doing specific work. *Cosine replaces MSE*: the loss becomes invariant to $z \mapsto \alpha z$, which closes the $\gamma \to 0$ route entirely — shrinking the latent scale no longer reduces the loss, so the dominant collapse pressure identified in §2.1 is removed at the source, not regularised against. *Multi-step with $\max\mathcal{K} \geq 4$*: since the frame stack has depth 4, $s_t$ and $s_{t+4}$ share **zero** frames (D1). At $k=4$ the identity solution is not merely disfavoured, it is unavailable. Including small $k$ as well retains the easy signal for early optimisation; the schedule $\mathcal{K} = \{1,2,4\}$ is the natural minimum. Multi-step latent rollout with an EMA target is the SPR construction (Schwarzer et al., 2021), which is the closest published relative of this thesis and should be cited and baselined against.

**(T2) Variance–covariance regularisation** — *answers §2.1 (collapse is a fixed point and nothing detects it).* With $z = f_\theta(s)$ flattened to $\mathbb{R}^{B\times D}$, $\tilde z = z - \bar z$, and $C(z) = \frac{1}{B-1}\tilde z^\top \tilde z$:

$$\mathcal{L}_{\text{var}} = \frac{1}{D}\sum_{j=1}^{D}\max\big(0,\; 1 - \sqrt{C(z)_{jj} + \epsilon}\,\big), \qquad \mathcal{L}_{\text{cov}} = \frac{1}{D}\sum_{i \neq j} C(z)_{ij}^2$$

This is VICReg (Bardes et al., 2022). $\mathcal{L}_{\text{var}}$ makes collapse not merely disfavoured but **infeasible**: the hinge imposes a hard floor of unit standard deviation on every embedding dimension across the batch, converting the anti-collapse property from a dynamical conjecture into a constraint. $\mathcal{L}_{\text{cov}}$ addresses dimensional collapse (Jing et al., 2022) — the milder failure in which the representation survives but occupies a low-rank subspace, which no metric in the current pipeline would detect either. Applied to the *online* branch only, this converts the Validator's conditional argument into an unconditional one and makes the fixed point of §2.1 unreachable rather than merely unattractive.

**(T3) Attention-pooled, symmetric, multi-step inverse dynamics** — *answers (D7), (G1), (G2), and §2.3(ii).*

$$\mathcal{L}_{\text{inv}} = \frac{1}{|\mathcal{K}|}\sum_{k \in \mathcal{K}}\mathrm{CE}\Big(g_\xi\big[\,q_\zeta(f_\theta(s_t))\,;\,q_\zeta(f_\theta(s_{t+k}))\,\big],\; a^{\text{canon}}_{t:t+k}\Big)$$

Four changes. *Attention pooling* $q_\zeta$ replaces `mean(dim=1)`, so the head can select the few patches where the action's effect lives instead of averaging it away — this is the objection the Validator did not contest. *Both arguments run through the trainable $f_\theta$*, restoring ICM's symmetric construction (Pathak et al., 2017) rather than the current detached-target form. *Multi-step targets*: predicting which action sequence separates $s_t$ from $s_{t+k}$ is the multi-step inverse-dynamics objective that Lamb et al. (2022) argue recovers control-endogenous latent state while discarding uncontrollable exogenous noise — which is precisely the Enduro/BeamRider failure of §2.3(iii). *Canonical action labels* per P1. Optionally, to answer (G1) directly, tie the inverse head's output layer to the predictor's `action_embed` table by weight sharing; this is the minimal change that makes $\mathcal{L}_{\text{inv}}$ capable of shaping the action representation the predictor actually consumes.

**(T4) Principled loss balancing** — *answers §2.3(i).* Replace the hardcoded $\lambda = 0.1$ with learned homoscedastic uncertainty weighting (Kendall et al., 2018) over the free parameters $\{\sigma_i\}$:

$$\boxed{\;\mathcal{L} = \sum_{i \in \{\text{pred},\,\text{var},\,\text{cov},\,\text{inv}\}} \left(\frac{1}{2\sigma_i^2}\,\mathcal{L}_i + \log \sigma_i\right)\;}$$

The $\log\sigma_i$ term prevents the trivial escape $\sigma_i \to \infty$. Because (T1) is now scale-invariant, the pathology of §2.3(i) — where $\lambda$ covertly fixed the latent scale — cannot recur: no weight can be reduced by shrinking the representation. If a learned weighting is judged too unstable for a thesis timeline, the defensible fallback is normalising each term by a running estimate of its own gradient norm (GradNorm, Chen et al., 2018) or applying PCGrad (Yu et al., 2020) to the encoder gradients, since §2.3(i) established that $\mathcal{L}_{\text{pred}}$ and $\mathcal{L}_{\text{inv}}$ impose genuinely conflicting pressure on the shared $\theta$. A hardcoded constant is the one option this review cannot endorse.

**(T5) EMA momentum schedule.** Anneal $\tau_t$ from $0.996$ to $1.0$ on a cosine schedule, as in BYOL (Grill et al., 2020), I-JEPA (Assran et al., 2023), and V-JEPA (Bardes et al., 2024). The fixed-$\tau$ choice in the current code is the one place where the implementation departs from every work it cites, and the departure is in the direction of a less stable target late in training, when stability matters most.

### 3.3 On quantisation — a qualified recommendation

The tribunal was asked whether the continuous latent should be quantised via FSQ (Mentzer et al., 2024) or the categorical latents of DreamerV2/V3 (Hafner et al., 2021; 2023) to bound drift over autoregressive rollout. The honest answer distinguishes two problems that quantisation is often credited with solving jointly.

*Collapse* is fully addressed by (T1)+(T2); adding FSQ for that purpose is redundant, and stacking a hard quantisation bottleneck on top of a variance hinge risks an over-determined objective whose failure modes are harder to diagnose than either alone — a poor trade in a thesis where diagnosis is the deliverable.

*Rollout drift* is a genuinely separate problem that (T1)+(T2) do not solve. Under autoregressive composition $\hat z^{(k)} = p_\psi(\hat z^{(k-1)}, \cdot)$, small per-step errors compound and the iterate leaves the manifold of realisable encodings. A finite latent alphabet bounds this by construction: the state cannot drift to a point that is not a code. This is the argument DreamerV2 makes for categorical latents and that IRIS (Micheli et al., 2023) exploits for transformer world models on Atari specifically.

**Recommendation:** treat FSQ as a distinct experimental arm, not a default. This repository already contains `models/fsq_vae.py` with a working `FSQBottleneck` and `config.FSQ_LEVELS = [8,5,5,3]` (codebook size 600), so the comparison is a genuinely cheap ablation rather than a research programme — and FSQ's chief advantage over VQ, that it needs no commitment loss, codebook-EMA, or dead-code handling, means it adds no new hyperparameters to confound the comparison. Report drift at $k \in \{1,2,4,8,16\}$ for continuous versus FSQ latents. A thesis chapter that answers "does quantisation bound latent drift in an action-conditioned JEPA?" with a controlled experiment is worth more than one that assumes either answer.

### 3.4 The evidentiary standard

No configuration should be reported as a world model without, at minimum: the identity control $\cos(z_t, z^{\text{tgt}}_{t+1})$ reported alongside every drift number (§2.2, diagnostic 1); action-conditional dispersion (diagnostic 2); attention mass on the action token (diagnostic 3); inverse-dynamics information gain over $H(a\mid c)$ rather than raw CE (P2); per-dimension embedding standard deviation and effective rank logged every epoch (D10); and evaluation windows constrained to lie within a single game shard (D9). The last four items are instrumentation, not research, and their absence is what currently makes the pipeline's outputs uninterpretable rather than merely uncertain.

---

## 4. Verdict of the Area Chair

### 4.1 Theoretically sound as written

- **Predicting in representation space rather than pixel space.** The core JEPA commitment is correctly implemented and correctly motivated (LeCun, 2022; Assran et al., 2023; Bardes et al., 2024). The thesis's framing of this as the central methodological choice is defensible.
- **The asymmetric branch construction.** The context/target split with a shallower predictor is faithful to the literature, and the double-locked gradient severance is correct and unambiguous.
- **Action conditioning as a sequence token.** Expressive, transformer-idiomatic, and a reasonable design; the objections in §2.2 concern whether the training signal exercises it, not whether the mechanism is sound.
- **Terminal masking.** Correctly derived and correctly implemented, even if (D6) shows it conditions almost nothing.
- **Engineering substrate.** AMP, gradient clipping, memory-mapped loading, cosine LR — competent and not at issue.

### 4.2 Unjustified but defensible

- **Fixed $\tau = 0.996$ without a momentum schedule.** Departs from BYOL, I-JEPA, and V-JEPA, all of which anneal $\tau \to 1$. Defensible for a shorter training budget; must be stated as a deliberate deviation and, ideally, ablated.
- **The 6/3 depth ratio and $D=256$.** Reasonable, unjustified, and cheap to defend with a small ablation.
- **The detached second argument to the inverse loss.** The Skeptic's original framing was successfully rebutted: EMA coupling means the representational consequence largely propagates even though the gradient path is absent. Deviates from ICM; survives as defensible.
- **Mean-pooling in the inverse head (D7).** Uncontested by the Validator and probably harmful, but it degrades an auxiliary term rather than invalidating the architecture. Fix it if (T3) is implemented; disclose it either way.

### 4.3 Genuine theoretical liabilities

These are the findings this tribunal regards as load-bearing for the thesis defence, in descending order of severity.

1. **The action-space collision (D2).** Eleven games contribute per-game minimal action sets of sizes 4, 6, 9, and 18, compactly re-indexed and folded into a single 18-way embedding and a single 18-way classification head, with no game identifier in the data. Every action-conditioned claim in the thesis is presently trained on labels whose semantics vary by game — and, per ALE's documented enumeration, in at least one case invert. This is measured, certain, and defeats the "action-aware" half of the central claim at the level of the data, before any modelling question arises. **It is also the cheapest to fix** (P1), which is why it heads the list.

2. **The inverse-dynamics shortcut is unmeasured, and large (D3).** $H(a) = 2.508$ nats, $H(a\mid\text{game}) = 1.886$ nats, chance $2.890$ nats. Pure game recognition — requiring no dynamics knowledge whatsoever — captures $1.004$ of $2.890$ available nats. A falling `inverse_action_loss` curve is therefore not evidence for the mechanism the thesis attributes it to. Any claim resting on that curve must be re-grounded against the $H(a\mid c)$ baseline.

3. **Collapse is a global minimum and a fixed point of the full system, and the pipeline is blind to it (§2.1, D4, D5, D10).** The Validator conceded the fixed-point characterisation. The dominant pressure toward it is $\nabla_\gamma\mathcal{L}_{\text{JEPA}}$ — the scale-sensitivity of an unnormalised MSE on affine-LayerNormed embeddings — operating at the full learning rate, with weight decay on $\gamma$ (D4) adding a smaller unopposed contraction. The sole counterpressure is $\lambda\mathcal{L}_{\text{inv}}$ with $\lambda$ hardcoded and underived. Compounding: no representation-health statistic is logged (D10), and checkpoint selection minimises a criterion that collapse also minimises (D5) — so `vjepa_best.pt` preferentially saves the most degenerate checkpoint reached. **The tribunal did not converge on whether the restoring force suffices in practice**, and it should not be asked to: this is decidable in one afternoon by logging per-dimension embedding standard deviation from an existing checkpoint, and that measurement should precede any further theoretical argument.

4. **The lazy-identity hypothesis is unrefuted and quantitatively grounded (§2.2, D1).** Three of four frames are bit-identical between input and target — verified exactly. The current evaluation cannot distinguish a learned dynamics model from an encoder exploiting that overlap, because the identity control is not reported. Until diagnostic (1) of §3.4 is run, the phrase "world model" is not supported by evidence in this pipeline. Note that this is a claim about *evidence*, not about the model: the model may well have learned dynamics. Nothing currently reported would tell us.

5. **The joint objective's task weighting provably inverts over training (§2.3(i)).** $\inf\mathcal{L}_{\text{JEPA}} = 0$ while $\mathcal{L}_{\text{inv}} \geq H(a\mid s,s') > 0$ under action aliasing. The objective silently migrates from prediction-dominated ($\approx 7{:}1$ at initialisation) to inverse-dynamics-dominated, at a crossover no one selected. Separately, because $\mathcal{L}_{\text{JEPA}}$ is scale-sensitive and $\mathcal{L}_{\text{inv}}$ is not, $\lambda$ covertly sets the operating scale of the latent space rather than trading off two tasks.

6. **The inverse module cannot reach the components it exists to regulate (G1, G2).** No gradient from $\mathcal{L}_{\text{inv}}$ reaches the predictor or `action_embed`. The model carries two disjoint, never-tied representations of "action." The thesis should not claim the inverse head prevents *action* collapse; on the gradient graph it can only make $z_t$ action-informative, which is a different and weaker claim — and one worth making, since it is true.

7. **Data provenance is unresolved (D8).** The 250k-frame-per-game CHW corpus and the loader's unconditional HWC transpose are incompatible, and no single working directory resolves both `TRAIN_DIR` and `VAL_DIR`. Before any number enters the thesis, the corpus that actually produced it must be identified. Additionally, the validation split shares all eleven games with training — it measures interpolation, not generalisation — while the four genuinely held-out games sit unused. For a thesis whose claim is a *universal* extractor, transfer to held-out games is not an optional extra experiment; it is the claim.

### 4.4 Points on which the tribunal did not converge

Recorded rather than resolved, per the tribunal's remit:

- **Whether $\lambda\mathcal{L}_{\text{inv}}$ in fact holds the representation away from collapse.** The Skeptic holds that the mechanism is fragile and undetected; the Validator holds that the restoring force strengthens as collapse deepens and is therefore self-stabilising. Both arguments are sound; the question is empirical (§4.3, item 3).
- **Whether encoding game identity is a defect or a feature.** The Validator argues it has value for a universal extractor; the Skeptic argues it is the shortcut that hollows out the ICM mechanism. Both may be true, and the disagreement is really about what the thesis claims the representation is *for*.
- **The severity of the weight-decay channel on $\gamma$.** Agreed to be real; disputed to be fast. Superseded in practice by the agreement that $\nabla_\gamma\mathcal{L}_{\text{JEPA}}$ dominates it.

### 4.5 Recommendation

The architecture is a competent, literature-faithful implementation of a joint-embedding predictive world model, and the thesis's core methodological commitment is sound. It is **not currently in a position to support the "Universal Feature Extractor" claim** — not primarily because the design is wrong, but because the training corpus contains a defect that makes action conditioning incoherent across games (D2), and because the pipeline lacks the instrumentation that would distinguish success from the two specific degenerate solutions this tribunal identified as reachable. The first is a data-collection fix. The second is logging. Neither is a research risk.

The ordering this review recommends is: canonicalise the action space (P1); log the four diagnostics of §3.4 against the existing checkpoint; and only then decide how much of §3.2 the thesis timeline can absorb. It is entirely possible that the existing model is healthier than the Skeptic fears — but on the present evidence that is a hope, not a finding, and the distinction is what a defence turns on.

---

## 5. References

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA). *CVPR 2023*.

Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer Normalization. *arXiv:1607.06450*.

Bardes, A., Ponce, J., & LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. *ICLR 2022*.

Bardes, A., Garrido, Q., Ponce, J., Rabbat, M., LeCun, Y., Assran, M., & Ballas, N. (2024). Revisiting Feature Prediction for Learning Visual Representations from Video (V-JEPA). *Transactions on Machine Learning Research*.

Bellemare, M. G., Naddaf, Y., Veness, J., & Bowling, M. (2013). The Arcade Learning Environment: An Evaluation Platform for General Agents. *Journal of Artificial Intelligence Research*, 47, 253–279.

Burda, Y., Edwards, H., Pathak, D., Storkey, A., Darrell, T., & Efros, A. A. (2019). Large-Scale Study of Curiosity-Driven Learning. *ICLR 2019*.

Burda, Y., Edwards, H., Storkey, A., & Klimov, O. (2019). Exploration by Random Network Distillation. *ICLR 2019*.

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging Properties in Self-Supervised Vision Transformers (DINO). *ICCV 2021*.

Chen, X., & He, K. (2021). Exploring Simple Siamese Representation Learning (SimSiam). *CVPR 2021*.

Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks. *ICML 2018*.

Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.

Gelada, C., Kumar, S., Buckman, J., Nachum, O., & Bellemare, M. G. (2019). DeepMDP: Learning Continuous Latent Space Models for Representation Learning. *ICML 2019*.

Grill, J.-B., et al. (2020). Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning (BYOL). *NeurIPS 2020*.

Guo, Z. D., Pires, B. A., Piot, B., Grill, J.-B., Altché, F., Munos, R., & Azar, M. G. (2020). Bootstrap Latent-Predictive Representations for Multitask Reinforcement Learning. *ICML 2020*.

Ha, D., & Schmidhuber, J. (2018). Recurrent World Models Facilitate Policy Evolution (World Models). *NeurIPS 2018*.

Hafner, D., Lillicrap, T., Fischer, I., Villegas, R., Ha, D., Lee, H., & Davidson, J. (2019). Learning Latent Dynamics for Planning from Pixels (PlaNet). *ICML 2019*.

Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination (Dreamer). *ICLR 2020*.

Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2021). Mastering Atari with Discrete World Models (DreamerV2). *ICLR 2021*.

Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering Diverse Domains through World Models (DreamerV3). *arXiv:2301.04104*.

Hansen, N., Su, H., & Wang, X. (2024). TD-MPC2: Scalable, Robust World Models for Continuous Control. *ICLR 2024*.

Jing, L., Vincent, P., LeCun, Y., & Tian, Y. (2022). Understanding Dimensional Collapse in Contrastive Self-Supervised Learning. *ICLR 2022*.

Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics. *CVPR 2018*.

Lamb, A., Islam, R., Efroni, Y., Didolkar, A., Misra, D., Foster, D., Molu, L., Chari, R., Krishnamurthy, A., & Langford, J. (2022). Guaranteed Discovery of Control-Endogenous Latent States with Multi-Step Inverse Models. *arXiv:2207.08229*.

LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. *OpenReview* (position paper, version 0.9.2).

Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). *ICLR 2019*.

Machado, M. C., Bellemare, M. G., Talvitie, E., Veness, J., Hausknecht, M., & Bowling, M. (2018). Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents. *Journal of Artificial Intelligence Research*, 61, 523–562.

Mentzer, F., Minnen, D., Agustsson, E., & Tschannen, M. (2024). Finite Scalar Quantization: VQ-VAE Made Simple. *ICLR 2024*.

Micheli, V., Alonso, E., & Fleuret, F. (2023). Transformers are Sample-Efficient World Models (IRIS). *ICLR 2023*.

Mnih, V., et al. (2015). Human-Level Control through Deep Reinforcement Learning. *Nature*, 518, 529–533.

van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 2017*.

van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation Learning with Contrastive Predictive Coding (CPC/InfoNCE). *arXiv:1807.03748*.

Pathak, D., Agrawal, P., Efros, A. A., & Darrell, T. (2017). Curiosity-Driven Exploration by Self-Supervised Prediction (ICM). *ICML 2017*.

Robine, J., Höftmann, M., Uelwer, T., & Harmeling, S. (2023). Transformer-Based World Models Are Happy with 100k Interactions. *ICLR 2023*.

Schwarzer, M., Anand, A., Goel, R., Hjelm, R. D., Courville, A., & Bachman, P. (2021). Data-Efficient Reinforcement Learning with Self-Predictive Representations (SPR). *ICLR 2021*.

Tian, Y., Chen, X., & Ganguli, S. (2021). Understanding Self-Supervised Learning Dynamics without Contrastive Pairs. *ICML 2021*.

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient Surgery for Multi-Task Learning (PCGrad). *NeurIPS 2020*.

Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. *ICML 2021*.

Zhang, W., Wang, G., Sun, J., Yuan, Y., & Huang, G. (2023). STORM: Efficient Stochastic Transformer Based World Models for Reinforcement Learning. *NeurIPS 2023*.
