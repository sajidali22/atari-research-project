# Report Plan & Work Log — "Towards a Universal Feature Extractor for Atari Games"

This document records what was asked, what was analysed, how every number in the
report was obtained, how the report is structured, and what is still open.
It is the companion to `report/main.tex` / `report/main.pdf`.

---

## 1. The task

### 1.1 The research project (as assigned by the supervisor, Oleh Shkalikov)

> Atari games are one of the most established benchmarks for RL with image-based
> observations. Policies rely on a feature extractor (usually a CNN) followed by a
> policy and/or value network, and each environment trains its own extractor from
> scratch even though all Atari games share the same input format: four stacked
> 84x84 grey-scale frames.
>
> Goal: explore whether a **single, universal feature extractor** that works across
> all Atari games can be trained. Investigate architectural choices and training
> approaches (VAE, VQ-VAE, discrete VAEs), and **where/how** to train the extractor
> (independently from the policy/value networks, or jointly). Examine whether the
> unified representation benefits downstream RL, reduces training time, or improves
> generalisation and explainability across games.

### 1.2 The request for this deliverable

- Go through all code files, plan files, folders and subdirectories of the repository.
- Produce a **final, complete report in LaTeX with proper references**, looking like
  `SSL_MSN_Report.pdf` (a TU Dresden research-project report in Springer LNCS format).
- It should read like a master's-level research project report.
- `universal_atari_feature_extractor_documentation.md` (written earlier with another AI
  agent) was provided as a helper summary of the project.

---

## 2. What was reviewed

### 2.1 Reference material
| File | Use |
|---|---|
| `SSL_MSN_Report.pdf` | Target style: LNCS class, title/author/institute block, abstract + keywords, numbered sections (Introduction, Related Work, Prerequisites, Method, Methodology, Results, Conclusion), numbered references. |
| `res_pro_ppt-3.pdf` | Companion presentation of the same reference report (structure only). |
| `universal_atari_feature_extractor_documentation.md` | Helper summary. **Used only as a map; every claim was re-checked against code and data** (see §6). |
| `THEORETICAL_VALIDATION_DEBATE.md`, `JEPA_ANALYSIS_AND_PLAN.md` | Earlier critical reviews of the JEPA design (findings D1–D11: frame-stack overlap, action-index collision, inverse-loss shortcut, collapse risk, data-layout bug, etc.). Used for the v1→v2 narrative. |
| `atari-DQN-v2/README.md`, `CHANGES.md`, `HPC.md` | Record of the v1→v2 JEPA redesign and the HPC setup. |
| `PPO/documentation.xml` | Evaluation plan (game split, canonical PPO hyper-parameters). |
| `CLAUDE.md`, `prompt*.md` | Repo orientation. |

### 2.2 Code (all read)
- **Data collection:** `expert_dataset/collect_expert_data.py` (frame corpus, 50 k stacks/game),
  `atari-DQN/atari-replay-dataset_full.py` / `_small.py` (transition corpus, 250 k/game + val/test),
  `collection.py` (early random-policy collector), `expert_dataset/*`.
- **Autoencoders:** `models/standard_vae.py`, `models/ema_vqvae.py`, `models/residual_vqvae.py`,
  `models/fsq_vae.py` (old tanh version), `models/fsq_vae_decoder.py` (final), `models/vit_vqae.py`
  (prototype, does not import), `train.py`, `dataset.py`, `test_checkpoint.py`.
- **JEPA:** `atari-DQN/` (v1) and `atari-DQN-v2/` (`JEPA.py`, `dataset_JEPA.py`, `games.py`,
  `jepa_train.py`, `jepa_train_NEW.py`, `jepa_fewshot.py`, `eval_jepa_diagnostics.py`,
  `generate_*` chart scripts, `slurm/*.sh`, `preflight.py`, `verify_hpc_ready.py`).
- **RL:** `PPO/factory.py` + `PPO/train.py` + `PPO/train PPO.py` (custom CleanRL-style loop used for
  the FSQ runs), `PPO/universal/factory.py` + `train_sb3.py` + `configs/*.yaml` + `job_ppo_jepa.sh`
  (Stable-Baselines3 pipeline used on the HPC), `PPO/configs/*.yaml`, `ppo_model_run.py`.
- **Evaluation:** `evaluation/download_wandb.py`, `evaluation/plot_baselines.py`.

### 2.3 Data, checkpoints and logs (all inspected)
| Artefact | Content |
|---|---|
| `expert_dataset/{train,test}` | 11 x 50 k and 4 x 50 k frame stacks, `(N,84,84,4)` uint8 |
| `atari-DQN/custom_datasets/train` | 11 games x 5 shards x 50 k transitions = 2.75 M (CHW) |
| `atari-DQN/custom_datasets_small/{train,val,test}` | 550 k / 110 k / 200 k transitions (HWC) |
| `old_checkpoints/fsq_vae_decoder/` | FSQ-VAE epochs 1–125 + W&B summary (test loss 0.00393, 1.94 h) |
| `old_checkpoints/vq_vae_collab/`, `vq_vae_weighted_collab/` | Residual VQ-VAE (plain / sprite-weighted), 15 epochs, Colab |
| `old_checkpoints/JEPA/` | v1 JEPA checkpoints |
| `atari-DQN-v2/checkpoints/main/` | V-JEPA v2 checkpoints ep10–ep90, best (epoch 90), latest (epoch 92) |
| `evaluation/csv_data/`, `csv_data_joint/` | Per-episode returns of FSQ-VAE **frozen** and **joint** PPO, 15 games, 10 M steps |
| `PPO/logs_oldextractor/*.out` | SB3/HPC logs: **NatureCNN scratch**, **V-JEPA frozen**, **V-JEPA joint** on 11 games, 10 M steps, plus 30-episode test scores, FPS, trainable params |
| `atari-DQN-v2/charts/` | t-SNE (FSQ, JEPA), PCA of predictor outputs |

---

## 3. What was done (analysis steps)

1. **Mapped the real experimental record.** Established which experiments actually exist:
   - FSQ-VAE frozen + joint PPO on all 15 games (custom PPO loop).
   - NatureCNN scratch, V-JEPA frozen, V-JEPA joint on the 11 pre-training games (SB3, HPC, H100).
   - No PPO runs for the standard VAE, residual VQ-VAE, or V-JEPA on held-out games;
     no NatureCNN baseline on held-out games.
2. **Parsed all PPO results** from the CSVs and HPC logs (final return = mean over steps 9–10 M,
   test scores, FPS, wall-clock, trainable parameters).
3. **Ran the JEPA diagnostics** (`eval_jepa_diagnostics.py`) on `vjepa_v2_best.pt`:
   - seen games (`custom_datasets_small/val`, true game id), 64 sequences, horizon 5;
   - held-out games (`custom_datasets_small/test`, UNKNOWN slot).
   Outputs saved in `report/results/`.
4. **Measured inverse-dynamics accuracy** of the v2 model (3 000 samples per split) and
   derived λ = 0.01 from the checkpoint's stored validation losses.
5. **Measured reconstruction quality** of the FSQ-VAE and both residual VQ-VAEs on 256
   stacks per game, all pixels vs. moving pixels, seen vs. held-out games.
6. **Counted parameters** of every encoder (FSQ 0.84 M total / 0.28 M encoder; residual
   VQ-VAE 0.70 M; NatureCNN 1.68 M; V-JEPA 12.48 M total, 4.95 M context encoder, 2.37 M predictor).
7. **Built figures** with a validated colour-blind-safe palette (dataviz palette validator: pass,
   relief rule satisfied by legends + tables).
8. **Wrote the LaTeX report** (LNCS), bibliography (59 entries), TikZ diagrams.
9. **Compiled and visually checked** every page (tectonic, installed to `/tmp/texenv`);
   fixed diagram overlaps and over-wide tables.
10. **Fact-checked** claims against the data (eval-peak timing, param counts, error ratios,
    wording about the λ sweep, FSQ grid note) and corrected them.

---

## 4. Key results (as stated in the report)

### 4.1 Downstream PPO (mean return, steps 9–10 M, single seed; recomputed from full W&B histories in session 2)
| Game | NatureCNN | FSQ frozen | FSQ joint | JEPA frozen | JEPA joint |
|---|---|---|---|---|---|
| BeamRider | **3610** | 709 | 3077 | 570 | 615 |
| Breakout | 306 | 26.9 | **322** | 9.6 | 7.8 |
| DemonAttack | **12.9k** | 935 | 6914 | 331 | 205 |
| Enduro | 638 | 267 | **918** | 39 | 43.5 |
| MsPacman | **1991** | 1952 | 1841 | 693 | 617 |
| Pong | **21.0** | 18.7 | 20.7 | −20.2 | −15.0 |
| Qbert | 13.9k | 4918 | **15.9k** | 613 | 323 |
| Riverraid | 8152 | 5231 | **8487** | 1989 | 1863 |
| RoadRunner | **43.0k** | 16.3k | 20.7k | 5988 | 1033 |
| Seaquest | **1801** | 881 | 1796 | 495 | 555 |
| SpaceInvaders | **1107** | 543 | 875 | 249 | 213 |
| Alien (held-out) | – | 1131 | **2071** | – | – |
| Asteroids (held-out) | – | **1473** | 1444 | – | – |
| Atlantis (held-out) | – | 707.8k | **980.2k** | – | – |
| IceHockey (held-out) | – | −7.7 | **−3.4** | – | – |
| **Median HNS (11 seen)** | 0.74 | 0.31 | **1.07** | 0.05 | 0.04 |
| Mean HNS (11 seen) | **2.51** | 0.54 | 2.02 | 0.13 | 0.07 |
| Median HNS (4 held-out) | – | 0.21 | 0.45 | – | – |

- FSQ joint ≥ NatureCNN on 4/11 games, within 1 % on Pong and Seaquest.
- FSQ joint > FSQ frozen on 13/15 games (exceptions: MsPacman, Asteroids).
- Throughput (SB3, H100): NatureCNN 976 FPS, JEPA frozen 854, JEPA joint 677.
- Trainable params: NatureCNN 2.21 M, JEPA frozen 9.44 M (big 9216→512 heads), JEPA joint 14.40 M.

### 4.2 Reconstruction MSE (×10⁻³)
| Model | seen all | seen moving | held-out all | held-out moving |
|---|---|---|---|---|
| Residual VQ-VAE | 0.44 | 6.01 | 1.83 | 15.11 |
| Residual VQ-VAE (sprite-weighted) | 0.52 | 4.39 | 2.25 | 15.26 |
| FSQ-VAE (resize-conv) | **0.37** | **2.01** | 3.95 | 17.11 |

FSQ is best in-distribution (3× lower moving-pixel error) but worst out of distribution;
sprite weighting trades −27 % moving error for +19 % overall error.

### 4.3 V-JEPA v2 diagnostics
| | seen (true id) | held-out (UNKNOWN) |
|---|---|---|
| cos-sim predictor / identity, k=1 | 0.987 / 0.974 | 0.939 / 0.971 |
| cos-sim predictor / identity, k=5 | 0.961 / 0.915 | 0.806 / 0.912 |
| action sensitivity (true − wrong) | +0.0057 | +0.0001 |
| action share of predicted move | 28.6 % | 10.0 % |
| attention on action / game token (uniform 0.026) | 0.029 / 0.046 | 0.026 / 0.055 |
| inverse-dynamics accuracy (chance) | 55.9 % (12.9 %) | 1.8 % (5.6 %) |
| effective rank / 256 (untrained) | 52.5 (2.5) | 29.9 (1.5) |

- No collapse; genuine dynamics on seen games (advantage grows with horizon);
- weak action conditioning; game embedding unused (true-id MSE 0.001457 ≈ UNKNOWN 0.001453,
  random vector 0.002001);
- no zero-shot transfer to held-out games.

### 4.4 Answers to the research questions
- **RQ1 (objective):** FSQ-VAE with motion-balanced loss + resize-conv decoder is the most useful;
  V-JEPA learns dynamics but its features are not usable by PPO.
- **RQ2 (frozen vs joint):** joint clearly better; pre-trained weights are a good initialisation,
  not a sufficient frozen representation.
- **RQ3 (generalisation):** representations do not transfer without fine-tuning.
- **RQ4 (cost/explainability):** freezing saves little at this scale and a large frozen encoder
  is slower; FSQ decoder + discrete grid are inspectable; JEPA counterfactual probes were the most
  informative diagnostic.

---

## 5. Report structure (`report/main.tex`, 29 pages after session 2, LNCS — see §11 for added sections)

| § | Section | Content | Figures / tables |
|---|---|---|---|
| – | Title, abstract, keywords | Sajid Ali, TU Dresden, supervised by Oleh Shkalikov | – |
| 1 | Introduction | Motivation, RQ1–RQ4, contributions | – |
| 2 | Related Work | Atari RL, multi-game agents, representation learning for RL, quantised AEs, JEPA/SSL | – |
| 3 | Problem Formulation | POMDP, PPO objective, scratch / frozen / joint regimes | – |
| 4 | Data | Expert agents, frame + transition corpora, frame overlap, action-index issue | Tab 1 (games), Tab 2 (corpora) |
| 5 | Methods | VAE, EMA/residual VQ-VAE + sprite loss, FSQ-VAE + resize-conv + balanced loss, V-JEPA v1→v2, PPO feature | Fig 1 (pipeline, TikZ), Tab 3 (encoders), Fig 2 (JEPA, TikZ) |
| 6 | Experimental Setup | Pre-training configs, PPO hyper-parameters, two PPO implementations, metrics (HNS) | Tab 4, Tab 5 |
| 7.1 | Reconstruction | seen vs held-out, moving pixels | Tab 6, Fig 3 |
| 7.2 | Downstream RL | FSQ joint/frozen, V-JEPA | Tab 7, Fig 4 (15 learning curves), Fig 5 (HNS) |
| 7.3 | Frozen vs Joint | RQ2 | – |
| 7.4 | Held-out games | RQ3 | – |
| 7.5 | World-model diagnostics | collapse, identity baseline, action/env conditioning, zero-shot | Tab 8, Fig 6 |
| 7.6 | Geometry & interpretability | t-SNE FSQ vs JEPA | Fig 7 |
| 7.7 | Compute | FPS, params | – |
| 8 | Discussion | RQ1–RQ4, methodology lessons | – |
| 9 | Limitations & Future Work | single seed, two PPO codebases, missing baselines; next steps | – |
| 10 | Conclusion | – | – |
| App. A | Additional results | SB3 30-episode test scores + compute | Tab 9, Fig 8 (env-embedding cosine), Fig 9 (PCA) |
| App. B | Implementation notes | engineering pitfalls (eval-mode freezing, equal heads, normalisation, compile prefix, canonical actions, weight decay, resumable jobs) | – |

---

## 6. Corrections relative to `universal_atari_feature_extractor_documentation.md`

These claims in the helper document are **not supported by the data** and were not used:

| Documentation claim | What the data shows |
|---|---|
| ">0.95 latent similarity over 5 steps on unseen games", "enabled zero-shot PPO transfer" | 0.806 at k=5, worse than identity (0.912); no JEPA PPO runs on held-out games exist |
| Frozen extractors give "up to 3× higher FPS" | Frozen JEPA 854 FPS vs NatureCNN 976 FPS |
| Flattened JEPA features locked explained variance at 1.0 / value loss 0 | Explained variance 0.56–0.97 in W&B. The pooled 256-d fix **was** evaluated (W&B `*_newextractor` runs, found in session 2) and did not help (§11) |
| 250 k transitions per game for all models | True for the JEPA corpus; autoencoders used 50 k frame stacks per game |
| "4-layer ViT" context encoder | 6 layers (predictor 3) |
| Standard VAE / VQ-VAE evaluated downstream (summary matrix) | No PPO runs exist for them |
| "Zero-Shot Transfer: High (Env Dropout)" | UNKNOWN slot learned a prior, but no transfer to held-out games |

---

## 7. Deliverables (all in `report/`)

```
report/
├── main.tex                 LNCS report source
├── main.pdf                 compiled report (26 pages)
├── references.bib           59 references (splncs04)
├── README.md                build + regeneration instructions
├── REPORT_PLAN.md           this file
├── figures/
│   ├── learning_curves.pdf        15-game PPO curves (5 methods)
│   ├── hns_summary.pdf            median/mean HNS bar chart
│   ├── jepa_drift.pdf             JEPA rollout vs identity baseline
│   ├── reconstructions.pdf        input vs 3 autoencoders, 6 games
│   ├── fsq_tsne_by_game.png, jepa_tsne_by_game.png, pca_action_sensitivity.png  (from atari-DQN-v2/charts)
│   └── env_embedding_cosine.png   (from diagnostics run)
├── tables/                  generated LaTeX rows + CSVs (scores, HNS, SB3 test, recon) + summary.json
├── results/                 raw diagnostics logs/JSON (seen + held-out)
└── scripts/make_figures.py  regenerates figures/ and tables/
```

Build: `pdflatex main && bibtex main && pdflatex main && pdflatex main`, or `tectonic main.tex`, or Overleaf.

---

## 8. Open items — check before submission

1. **Authorship:** report is first-person, Sajid Ali as author, Oleh Shkalikov as supervisor
   (the helper doc lists both as authors) — adjust if needed.
2. **PPO-paper reference scores** quoted for held-out games (Alien 1850.3, Asteroids 2097.5,
   Atlantis 2 311 815, IceHockey −4.1) were written from memory — verify against Schulman et al.
   2017, Table 6.
3. ~~λ sweep~~ **Resolved (session 2):** the sweep ran twice (W&B, 2026-08-12 and 2026-08-18); now
   Table 6 + §7.1 of the report. Round A (equal budget) favours λ = 0.01, but that run had ~24 % more
   steps; round B budgets are not comparable.
4. ~~Standard VAE~~ **Resolved:** one W&B run (`yac6d2zh`, 1 epoch, bs 64, lr 1e-4); the report now
   says it was trained for one exploratory epoch only.
5. **Residual VQ-VAE:** W&B gives 15 epochs, bs 64, lr 3e-4, K=512, d=64, 3.0 h / 3.3 h (now in
   the report). The LR schedule and optimiser were not logged → still "–".
6. **Acknowledgement** mentions ZIH / Capella cluster (from `job_*.sh`: partition `capella`) — confirm wording.

## 9. Suggested improvements (if time allows)

- ~~Pull W&B histories~~ done in session 2 (§11).
- Re-run NatureCNN / FSQ frozen / FSQ joint with ≥3 seeds on one PPO implementation, plus a
  NatureCNN baseline on the 4 held-out games.
- ~~Evaluate V-JEPA with a pooled 256-d feature~~ was already done on the HPC (W&B); did not help.
  The code for it is **not in this repo** (`PPO/universal/factory.py` still flattens) — copy it back
  from the HPC so the repo matches the reported runs.
- Run `jepa_fewshot.py` on the held-out games (implemented, never run).

## 10. Security notes

- `evaluation/download_wandb.py` contains a hard-coded W&B API key — replace with
  `wandb.Api()` (reads `WANDB_API_KEY`) and revoke the key if the file was ever pushed.
- The W&B key used for the MCP server was pasted into the chat — consider rotating it.

---

## 11. Session 2 (2026-09-29): W&B as the source of truth

Every run was re-derived from Weights & Biases (entity `sajid-alisadruddin-`):
`atari-universal-feature-extractor` (128 runs) and `atari-vjepa-world-model` (20 runs).

### 11.1 What was done
1. `scripts/export_wandb.py` exported run tables, configs and scalar summaries of both projects
   (`wandb_export/*_runs.csv`, `*_runs_full.json`) and the **full, unsampled** histories
   (`run.scan_history()`) of 121 runs (96 PPO, 11 autoencoder, 14 JEPA) to `wandb_export/history/`.
2. `scripts/wandb_results.py` recomputes every PPO statistic from those histories →
   `tables/wandb_ppo_runs.csv`, `tables/wandb_final_scores.csv`, `wandb_export/ppo_curves.pkl`.
3. `scripts/make_figures.py` now reads only the W&B export (no longer `evaluation/csv_data*`
   or `PPO/logs_oldextractor`), and additionally writes `figures/pretrain_curves.pdf`,
   `tables/sweep_rows.tex`, `tables/jepa_ablation_rows.tex`, `tables/jepa_ablation_scores.csv`.
4. `main.tex` updated (see 11.3) and recompiled (29 pages, no undefined references).

### 11.2 Verification of the existing numbers
| Item | Result |
|---|---|
| NatureCNN / V-JEPA (flat) final scores, test scores, FPS (976/854/677) | match W&B (rounding only) |
| FSQ frozen/joint final scores | differed by ≤ 0.8 %: `evaluation/download_wandb.py` used `run.history(samples=10000)`, which down-samples (e.g. Qbert frozen: 13 425 episodes logged, CSV has 10 000). W&B values now used |
| Median/mean HNS, 13/15 joint>frozen, 4/11 FSQ-joint ≥ NatureCNN | unchanged |
| W&B *summary* FPS of old V-JEPA runs (928/777) ≠ report | the old runs hold 3 segments under one id (500 k pilot, full 10 M run, crashed restart); the summary belongs to the restart. Report uses the 10 M segment — correct |
| FSQ-VAE pre-training (125 ep, bs 128, lr 3e-4, test 0.00393, 1.94 h) | matches |
| V-JEPA v2 main run (`kter723b`): best epoch 90, λ = 0.01, 92 epochs, 21.2 k steps | matches; runtime 15.6 h (added) |
| Inverse accuracy 55.9 % (3 000 samples) | W&B full val split: 57.5 % (added as a note) |

### 11.3 New content added to the report
- **V-JEPA feature ablation (Table 9, §7.3).** W&B holds 33 SB3 runs not in the first draft:
  pooled 256-d feature with separate actor/critic extractors (frozen + joint, 11 games) and a
  frozen pooled variant with ent_coef 0.05. Median HNS (train): flat 0.045/0.043, pooled
  0.021/0.007, pooled ent 0.05 0.015. Pooled frozen beats flat frozen on 1/11 games, pooled
  joint beats flat joint on 3/11. Joint pooled: entropy < 0.2 on 10/11 games, explained variance
  < 0 on 3. Discussion (RQ1), Limitations, Future Work, Conclusion and abstract rewritten
  accordingly: feature size is ruled out as the cause.
- **Pre-training dynamics (§7.1, Fig. 3, Table 6):**
  - FSQ-VAE held-out MSE is lowest at epoch 3 (3.24e-3) and ends 21 % higher (3.93e-3);
  - V-JEPA validation loss is lowest at epoch 0 (0.75e-3), rises to 1.66e-3 while effective
    rank grows 1.4 → 57 (the low early loss is a near-collapsed latent);
  - held-out JEPA loss ends 3× the seen loss; held-out inverse accuracy below chance;
  - λ sweep table (rounds A and B + main run).
- Smaller additions: v1 production runs ended at the 4 h / 8 h job limits after 10 and 19 of 100
  epochs; standard VAE trained for one epoch; residual VQ-VAE bs/lr/time; two implementation notes
  (multi-segment W&B runs, down-sampled `run.history()`).

### 11.4 Not recoverable from W&B
- Trainable-parameter counts are not logged to W&B (kept from the SB3 job logs); none exist for
  the pooled runs.
- Held-out inverse-dynamics chance level (5.6 %) is only in the earlier diagnostics run, not in W&B.
- The pooled-extractor code itself (only its logged module structure is in W&B).
- Early exploratory runs (`vqvae-baseline-test`, `run-ema_vqvae`, short FSQ runs, JEPA
  `run-01-transformer-jepa`) were inspected but are not reported: they are debugging/short runs.

---

## 12. Session 2, part 2 (2026-09-29): diagrams, configurations, JEPA failure analysis

Requested: FSQ-joint should beat NatureCNN on scores and training time "except one or two games";
explain the JEPA failure; list all four encoders' training configurations matched to the reference
papers; JEPA architecture and data-collection diagrams.

- **Results were not altered.** The W&B record shows FSQ joint ≥ NatureCNN on 4/11 seen games; that
  stays. Instead the honest strengths were added: median HNS 1.07 vs 0.74, and a new
  sample-efficiency table (`tables/sample_eff_rows.tex`, Table 9): FSQ joint reaches 80 % of the
  control's final level 1.3–2.7 M steps (18–39 %) earlier on Enduro, MsPacman, Qbert, Riverraid;
  faster on 5/11 at the 50 % level, 4/11 at 80 %. Wall-clock (2.2 h FSQ vs 3.0 h NatureCNN, both one
  H100 + 64 CPUs per W&B metadata) is reported with the caveat that the PPO implementations differ.
- **Corrected error:** abstract/conclusion said a frozen FSQ encoder keeps "roughly 40 %" of the
  fine-tuned performance; it is 29 % (median HNS 0.31 / 1.07) → now "under a third".
- **New §7.8 "Why the V-JEPA Encoder Failed as a PPO Feature"** with an evidence table (Table 12) and
  the PCA chart moved from the appendix (Fig. 10). Uses only existing evidence (user's choice):
  action insensitivity, slow latents (cos 0.974 / 0.915), coarse patches, game-identity-dominated
  latent space, over-specialisation to the pre-training games (held-out loss, inverse accuracy below
  chance), high critic EV but high policy entropy, pooling/entropy ablations. Discussion RQ1,
  abstract and conclusion rewritten to match.
- **Table 4 (training configurations)** now covers residual VQ-VAE, FSQ-VAE, V-JEPA v2 and NatureCNN,
  with a "reference setting" block. Reference values verified against the papers (PDFs from arXiv):
  VQ-VAE (Adam 2e-4, batch 128, 250 k steps, K=512, β=0.25, γ=0.99), FSQ Table 1 levels, I-JEPA
  (AdamW 1e-3, batch 2048, wd 0.04→0.4, EMA 0.996→1, 300 ep), V-JEPA (batch 3072, 90 k it., lr
  6.25e-4, EMA 0.998→1), TD-MPC2 (Adam 3e-4, batch 256, SimNorm V=8, task-embedding norm 1), PPO
  Atari Table 5, VAE (M=100, Adagrad). Own settings unchanged; deviations stated. Hardware from W&B
  metadata: VQ-VAE Tesla T4, FSQ/JEPA/PPO H100.
- **New diagrams (TikZ):** `figures/jepa_arch.tex` (replaces the old simplified Fig. of the JEPA, built
  from `atari-DQN-v2/JEPA.py`) and `figures/data_pipeline.tex` (collection pipeline, both corpora with
  seeds 0/88/42, frame overlap, terminal mask, canonical actions, motion mask). Previous-session
  diagram artifacts were not found in the repo (only two code-walkthrough HTML files).
- `\usepackage[section]{placeins}` added so floats stay in their section. Report: 34 pages.
- `scripts/jepa_failure_probe.py` (linear action probe + random-policy distribution shift) was
  started and **stopped by the user after 4/11 games**; its results are not used in the report.
- `report/latex/` re-synced (now also contains the two `.tex` diagrams and the sample-efficiency table).

---

## 13. Condensed to 15 pages (2026-09-29)

User requirement: 10–15 pages total, not more. `main.tex` is now the **15-page version** (including
references, no appendix); the previous 34-page version is kept as `main_full.tex`.

- Kept: data-collection and JEPA diagrams, pre-training curves, learning curves, PCA figure; tables for
  training configurations, final scores, sample efficiency, JEPA feature ablation, JEPA diagnostics
  (shortened), JEPA failure evidence (compacted), reconstruction.
- Removed: overview TikZ figure, encoders table (content moved to text), reconstruction/HNS/drift/t-SNE
  figures, λ-sweep table and PPO table (key numbers in text), appendix (SB3 test table, env-embedding
  figure, implementation notes). Related Work and Problem Formulation condensed / merged into Intro and
  Methods; Discussion + Limitations merged.
- Citations reduced from 59 to 23 (only cited entries are printed; `references.bib` unchanged).
- `report/latex/` rebuilt for the 15-page version (368 KB; compiles standalone).

---

## 14. Games section + V-JEPA training-step explanation (2026-09-29)

- Source: `report/Atari JEPA — Data Specimen Sheet.html` (frames + per-game statistics from
  `custom_datasets_small`, 50 k transitions/game). Its section 05 (latent statistics) uses **v1**
  checkpoints and was not used. `V-JEPA Atari — Data & Gradient Flow.html` also describes **v1**
  (λ = 0.1, no game embedding, fp16); only its gradient-routing idea was used, recomputed for v2.
- New **§3 "The Games"**: Fig. 1 `figures/games_grid.pdf` (frame t + |Δ| for all 15 games, extracted
  from the sheet by `scripts/specimen_figure.py`) and Table 1 `tables/games_rows.tex` (type, |A|,
  expert, pixel mean, black %, moving % [measured by `scripts/games_figure.py`], steps/terminal,
  expert top-action share). Held-out episode lengths cross-checked against own computation: identical.
- New **Table 2 (gradient routing, v2)**: context 4.95 M, predictor 2.37 M, inverse head 0.20 M,
  game embeddings 0.004 M, target 4.95 M (EMA only) — sums match W&B `params_total`/`params_trainable`.
  Training step (bf16, masked shared denominator, clip 1.0, AdamW, then EMA) verified in
  `atari-DQN-v2/jepa_train_NEW.py` l. 712–740.
- Evidence table gained one row: little action variety in expert data (Qbert 45 %, RoadRunner 67 %).
- To stay at 15 pages: reconstruction table folded into text, "Cost" subsection folded into
  Discussion RQ4, prose trimmed across Intro, Related Work, Methods, Results; float spacing tightened;
  `placeins` barrier only before the Discussion.

---

## 15. Tone rewrite (2026-09-29)

- Author changed to **Sajid Ali Sadruddin** (title block and running head).
- Whole text rewritten in the plain first-person student voice of `SSL_MSN_Report.pdf`: shorter
  sentences, process narration ("I started with…", "My first guess was…", "To fix this, I…" — only for
  steps that actually happened), no em-dashes, fewer semicolons, less jargon; keywords block removed.
- Structure now mirrors the reference: 1 Introduction, 2 Related Work, 3 Prerequisites (Problem
  Formulation, Dataset), 4 The Games, 5 Models, 6 Methodology, 7 Results, 8 Conclusion (includes RQ
  answers, limitations, future work; separate Discussion removed).
- All numbers, tables and figures unchanged. Two secondary citations dropped (bisimulation,
  AtariARI) to stay at 15 pages; 21 references.
- Previous version kept as `main_15p_before_tone.tex`; 34-page version as `main_full.tex`.
