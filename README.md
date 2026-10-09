# hssbeamgen

EC3-compliant generative design of high-strength-steel (S355-S690) I-beams. An amortized design
policy (reinforcement learning) is compared against conventional per-context optimization (GA, DE,
random search) and a simpler learned mapping (kNN), across varying demands (span x factored UDL).

> **Status.** All experiments are complete and logged in `docs/`; the figures and tables are generated from
> the saved results. The manuscript is being rewritten from them; the logs are interim, not the paper.

**Research question.** Can an amortized generative design approach efficiently produce EC3-compliant
HSS beam designs across varying demands, and what does RL actually contribute compared with
conventional optimization and simpler learned mappings? The evidence is reported wherever it points,
including where it favours GA/DE or kNN.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # exact pins used for the reported results
pip install -e .
pytest tests/                            # most tests need only numpy/pandas
```

RL training and checkpoint evaluation were run on Colab; checkpoints (`runs/`, `models/`) are
gitignored and not part of the repo. Everything else needed to reproduce the logged
results from the saved per-run outputs is under `results/`.

## Layout

```
hssbeamgen/            package: EC3 environment (envs/), GA/DE/search + operators (algo/), training utils
pipeline/              numbered scripts, run in order (see below)
configs/rl_final.yaml  frozen RL hyperparameters (not edited after the pilots)
data/ground_truth/     pooled best-known reference designs per objective (cost, mass, CO2):
                       main_grid, ood_span, ood_load, ood_joint (each with a *_pooled version)
results/               per-seed / per-run evaluation outputs behind every log in docs/
figures/               paper figures (png + pdf), the plotted numbers (*_data.csv) and captions.md
paper/tables/          paper tables (md + csv), built from results/
docs/                  interim results logs; docs/archive/ holds the original project briefing
tests/                 pytest suite
```

Always evaluate against the `_pooled` ground-truth directories; the scripts refuse the raw ones.

## Pipeline

| Script | Purpose |
|---|---|
| `01_generate_ground_truth.py`, `01b_verify_ground_truth.py`, `01c_pool_ground_truth.py` | GA ground truth per objective, schema/re-evaluation checks, pooling of the best-known reference |
| `02_train_agent.py` | PPO and SAC/TD3/DDPG training with full resume |
| `03_evaluate_agent.py` | evaluate one checkpoint against the pooled reference under three post-hoc operators (`none`, `scale`, `scale+thin`) |
| `04_baseline_search.py` | GA / DE / random search at matched evaluation budgets |
| `05_baseline_knn.py` | kNN amortized mapping (leave-one-out, subsample, OOD; ground-truth or search-generated labels) |
| `09_compare_arms.py` | exact permutation test with Holm correction across arms (5 seeds each) |
| `10_make_figures.py` | seven figures (overview, quality vs evaluations, in-dist vs OOD, ablations, kNN label cost, operator ablation, reference landscape) from the saved `results/` only |
| `11_build_paper_tables.py` | tables (`paper/tables/`) from the saved `results/` only, including the PPO-vs-search significance tests and the operator ablation; no significance test across objectives (different references) |
| `12_time_inference.py` | wall-clock per design, policy vs search, on one machine (run locally; needs a trained checkpoint) |
| `13_export_training_curves.py` | TensorBoard logs of complete runs -> small CSVs in `results/training_curves/` (run locally; needs `tensorboard`, not torch); feeds fig7 |

Example (evaluation of one checkpoint, kNN, and search baseline):

```bash
python pipeline/03_evaluate_agent.py --model runs/<run>/final_model \
    --ground_truth_dir data/ground_truth/main_grid_pooled --economy_metric cost --out results/eval.csv
python pipeline/05_baseline_knn.py --protocol loo --economy_metric cost \
    --train_gt_dir data/ground_truth/main_grid_pooled --out results/knn_loo_cost.csv
python pipeline/04_baseline_search.py --economy_metric cost \
    --gt_dir data/ground_truth/main_grid_pooled --workers 4 --out results/search_main_cost.csv
```

`03` defaults `--economy_metric` to `cost`; pass it explicitly for mass and CO2 models.

Training runs (seeds 42-46; PPO from 2a is reused as the PPO arm of 2b and as the cost arm of 2c):

```bash
for s in 42 43 44 45 46; do
  for mode in feasibility_gated lagrangian shaped; do          # 2a
    python pipeline/02_train_agent.py --algo ppo --reward_mode $mode --seed $s \
      --out runs/2a_reward_mode/$mode/seed$s; done
  for algo in sac td3 ddpg; do                                  # 2b
    python pipeline/02_train_agent.py --algo $algo --reward_mode feasibility_gated --seed $s \
      --out runs/2b_algo/$algo/seed$s; done
  for metric in mass co2; do                                    # 2c
    python pipeline/02_train_agent.py --algo ppo --reward_mode feasibility_gated --seed $s \
      --economy_metric $metric --out runs/2c_objective/$metric/seed$s; done
done
python pipeline/09_compare_arms.py --out results/2a_comparison.csv
python pipeline/09_compare_arms.py --arms ppo=2a_feasibility_gated sac=2b_sac td3=2b_td3 ddpg=2b_ddpg \
  --out results/2b_comparison.csv
python pipeline/10_make_figures.py && python pipeline/11_build_paper_tables.py

# wall-clock per design on this machine (single process; run once per objective)
python pipeline/12_time_inference.py --model runs/2a_reward_mode/feasibility_gated/seed42/final_model \
  --ground_truth_dir data/ground_truth/main_grid_pooled --economy_metric cost --n_contexts 20 --repeats 3 \
  --out results/timing_cost.csv
```

Claim-by-claim evidence and the known limitations are listed in [`docs/claims_and_evidence.md`](docs/claims_and_evidence.md).

## Main results

Main grid, `scale+thin`, gap to the best-known reference in % (RL: mean of 5 seeds; lower is better):

| Method | Cost | Mass | CO2 | EC3 evals per design | Needed beforehand |
|---|---|---|---|---|---|
| PPO (`feasibility_gated`) | 8.9 | 7.2 | 4.6 | about 100-112 | 1M training steps, no labels |
| kNN, pooled labels (leave-one-out) | 1.1 | 1.0 | 0.5 | 41-63 | solved contexts (about 96,000 evaluations each) |
| kNN, DE labels B = 1000 (LOO) | 3.2 | 3.2 | 1.6 | 42-70 | 141 searches of 1,000 evaluations |
| kNN, DE labels B = 4800 (LOO) | 0.6 | 0.7 | 0.2 | 39-61 | 141 searches of 4,800 evaluations |
| DE search, budget 40 | 39.2 | 17.7 | 26.5 | about 108-112 | nothing |

At equal evaluations the policy beats per-context search: with the `scale+thin` operator in all nine
(objective, search method) comparisons (exact permutation test, Holm p = 0.024, the smallest attainable
with five seeds per arm), and without any operator (40 vs 40 evaluations) in seven of nine (mass vs DE
and vs random: Holm p = 0.063); `paper/tables/t10_ppo_vs_search.md`. The kNN beats the policy on gap and cost
per design but needs solved labelled contexts and is less often feasible. Details, statistics and
caveats are in the logs below; the tables are in `paper/tables/`.

## Experiments and logs

Interim results are logged in `docs/` as each stage completes, separate from the eventual manuscript.
Gaps are relative to the best-known reference (0% = matches it; lower is better), measured after
the `scale+thin` operator unless stated.

| Log | Question | Headline (interim) |
|---|---|---|
| [2a reward mode](docs/results_2a_reward_mode.md) | which PPO reward mode? | `feasibility_gated` 8.9%, lagrangian 9.4%, shaped 15.8%; no pairwise difference survives Holm correction at n = 5 |
| [2b algorithm](docs/results_2b_algorithm.md) | PPO vs SAC / TD3 / DDPG | PPO 8.9%, SAC 24.9%, TD3 36.1%, DDPG 32.1%; only PPO vs TD3 is significant after Holm; off-policy seeds are far less stable |
| [2c objective transfer](docs/results_2c_objective_transfer.md) | same configuration for mass and CO2 | mass 7.2%, CO2 4.6% (descriptive; objectives are scored against different references) |
| [OOD](docs/results_ood.md) | PPO outside the training range | OOD gaps similar to in-distribution after the operators (partly an operator effect); `ood_joint` has no feasible reference and is untestable |
| [search baselines](docs/results_search_baselines.md) | GA / DE / random at equal evaluations | the policy wins at about 100 evaluations per design; GA/DE need about 2.5-10x more to match it and beat it at full budget (4800) |
| [kNN baseline](docs/results_knn_baseline.md) | non-RL amortized mapping | 0.5-1.7% gap in distribution, ahead of PPO on gap, but lower feasibility (especially cost, and OOD) |
| [kNN, realistic labels](docs/results_knn_cheap_labels.md) | does the kNN survive cheap labels? | search-generated labels at 4800 evaluations are as good as the pooled ones; the kNN is cheaper than RL training; OOD results are noisier |

Caveats that apply across the logs: five RL seeds per arm (the kNN is deterministic); the reference
is the best-known, not proven, optimum (DE at full budget slightly undercuts it); post-hoc
operators contribute much of the quality of every method; the context space has two parameters.
The original project briefing is archived in [`docs/archive/`](docs/archive/README.md).

## License

MIT. See `LICENSE`.
