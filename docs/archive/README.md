# docs/archive

Historical project documents, kept verbatim for provenance. **They are not current
specifications.** Where they disagree with the code, `configs/`, or `docs/results_*.md`, the
latter win. Known superseded details:

| Document | What is historical |
|---|---|
| `research_audit.md` | Written against `research-next @ 34cfafe`. Its checkpoint/ground-truth inventory describes the old repo; the `pretrain_data*` directories and `models/*` it lists are not on this branch. Its proposed PPO config used `economy_reward_mode: log_relative`; the frozen config uses `linear` (the only mode with 5 seeds). |
| `repo_pipeline_plan.md` | The plan this branch is being built from. Actual layout differs in places: `hssbeamgen/` package instead of `src/`; `02_train_agent.py` / `03_evaluate_agent.py` / `09_compare_arms.py` instead of `02_train_ppo.py` / `03_evaluate_policy.py` / `09_build_paper_tables.py`; `ppo_final.yaml` is now `rl_final.yaml`; `repair` is now `apply_operator` / "adjusted"; runs live in `runs/` (gitignored), not `models/`. |
| `results_validation_draft.md` | Pre-rebuild draft. Its numbers come from the old pipeline and are NOT results of this branch (e.g. 7.5% headline gap vs 8.9% in `docs/results_2a_reward_mode.md`; it says `log_relative`, the frozen config is `linear`). Do not cite figures from it without re-deriving them. |

`full_experimental_history.md` (the consolidated `perplexity_docs/` narrative mentioned in the
plan) is **not** in this archive; `research_audit.md` is the closest available substitute.
