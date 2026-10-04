# 2c: objective transfer (mass and CO2 under the cost configuration)

Status: **complete**: 10 new runs (mass x 5 seeds, CO2 x 5 seeds), each evaluated on the pooled
main-grid ground truth for its own objective. Interim results log, not the manuscript.

## Setup

- PPO, `feasibility_gated`, `configs/rl_final.yaml`, seeds 42-46. Only `--economy_metric` differs
  (`mass` or `co2`); nothing else was retuned.
- The cost arm is the 2a `feasibility_gated` runs (not retrained).
- All 10 new runs: single uninterrupted pass of 1,007,616 steps (1M rounded up to whole rollouts),
  `--resume` not used, `git_dirty: false` for training and evaluation (see `results/2c_*_eval_meta.json`).
- Evaluation: `pipeline/03_evaluate_agent.py` with `--economy_metric` set to the trained objective,
  against `data/ground_truth/main_grid_pooled` (142 contexts). The reference is the best-known
  optimum for that objective. In the eval CSVs the column `cost_ratio_mean` holds the ratio on the
  evaluated objective (1.00 = matches the reference optimum).
- Descriptive comparison only. The three objectives are scored against different references with
  different optimal-grade structure, so no cross-objective significance test is reported.

## Results (gap to best-known optimum, %, mean ± sd over 5 seeds)

| Objective | none | scale | scale+thin | Per-seed (scale+thin) | Within 110% | Within 125% | Grade match |
|---|---|---|---|---|---|---|---|
| Cost (2a) | 49.4 ± 20.8 | 20.8 ± 9.8 | 8.9 ± 2.4 | 12.5 / 7.1 / 9.7 / 6.3 / 8.8 | 71.1% | 94.4% | 94.4% |
| Mass | 48.2 ± 5.7 | 24.3 ± 4.1 | 7.2 ± 3.2 | 12.3 / 5.7 / 4.2 / 5.7 / 8.0 | 74.4% | 97.6% | 47.6% |
| CO2 | 47.6 ± 8.5 | 23.1 ± 5.6 | 4.6 ± 2.6 | 5.3 / 2.8 / 2.2 / 8.8 / 3.8 | 83.2% | 99.2% | 98.2% |

Feasibility is 100% for every run and every operator mode (all three objectives). Policy
evaluations per design: 40 (none), 57 (scale), 100-112 (scale+thin).

Optimal-grade structure of the references differs by objective: cost S355 in 134/142 contexts;
mass S690 in 75, S550 in 25, S500 in 17, S620 in 13, S460 in 12; CO2 S690 in all 142.

## Reading

- The cost configuration transfers: with no changes, mass and CO2 reach scale+thin gaps of 7.2% and
  4.6%, in the same range as cost (8.9%), with every run feasible. Seed spread is similar to
  cost (sd 3.2 and 2.6 vs 2.4 at scale+thin; smaller before thinning).
- Before the thinning step the three objectives are close (47.6-49.4% policy-only, 20.8-24.3% after
  scale); the objective differences appear mainly after thinning.
- CO2 has the lowest gap (4.6%); its reference uses S690 in every context, and the agent matches
  the grade in 98% of contexts. Mass has the lowest grade-match (47.6%) while still reaching a
  7.2% gap.
- Five seeds per objective; the differences between objectives are descriptive and not tested.
- No GA/DE or kNN baseline has been run for mass or CO2 yet, so these gaps have no non-RL
  reference.

## Audit trail

No 2c run was resumed or replaced.
