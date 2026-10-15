# docs/

| Path | What it is |
|---|---|
| [`reproducing_figures.md`](reproducing_figures.md) | How to rebuild the paper and regenerate its figures and tables |
| [`universal_atari_feature_extractor_documentation.md`](universal_atari_feature_extractor_documentation.md) | Long-form technical documentation of the whole project |
| `analysis/` | Design notes written during the project: JEPA critique and roadmap (`JEPA_ANALYSIS_AND_PLAN.md`), theoretical validation (`THEORETICAL_VALIDATION_DEBATE.md`), report plan, and HTML explainers (SB3 integration, JEPA→PPO integration, data specimen sheet, gradient flow) |
| `archive/` | Superseded material, kept for reference |
| `archive/report_v1/` | Earlier report version (Sep 29) with its intermediate `*_before_*.tex` drafts, LaTeX build, backup and slides |
| `archive/experiment/` | Another report/slides working copy, including the Overleaf exports |
| `archive/report_drafts/` | Drafts of the final paper |
| `archive/prompts/` | The prompts used to generate the analysis documents |
| `archive/SSL_MSN_Report.pdf` | LNCS template reference report |
| `archive/res_pro_ppt-3.pdf` | Earlier presentation |

**Paths in these documents are historical.** They were written before the repository was
reorganised:

| Old path | Now |
|---|---|
| `atari-DQN-v2/` | `jepa/` |
| `atari-DQN/` (JEPA v1) | removed; never committed (git history only has an earlier prototype: root `models/JEPA.py` / `jepa_train.py`, commit `f5c93e1`); checkpoints in `artifacts/checkpoints/jepa/old_v1/`; transition collector → `data_collection/collect_jepa_transitions.py` |
| `atari-DQN-v2/checkpoints/main/`, `production_checkpoints/` | `artifacts/checkpoints/jepa/` |
| `custom_datasets*/` | `artifacts/data/jepa/custom_datasets*/` |
| `expert_dataset/{train,test}` | `artifacts/data/expert/{train,test}`; scripts in `data_collection/` |
| `old_checkpoints/` | `artifacts/checkpoints/autoencoders/` |
| `PPO/universal/` | `ppo/` |
| `PPO/factory.py`, `PPO/train*.py`, `ppo_model_run.py` | `ppo/custom/` |
| `train.py`, `dataset.py`, `config.py` (root) | `autoencoders/` |
| `report/scripts/` | `evaluation/report_scripts/` |
| `report/{figures,tables,results}` | `evaluation/outputs/` |
| `report/wandb_export/`, `wandb_export/` | `artifacts/wandb_export/` |
| `report_1/` | `paper/` |
| `jepa_train_NEW.py` | `jepa/jepa_train.py` (the old `jepa_train.py` is now `jepa_train_legacy.py`) |

`.zip` and LaTeX build files (`*.log`, `*.blg`) in `archive/` are git-ignored and exist only on
the original machine.
