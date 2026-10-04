# 2b: algorithm comparison (PPO / SAC / TD3 / DDPG under feasibility_gated)

Status: **complete**: 20 runs (4 algorithms x 5 seeds), evaluated against the pooled main-grid
ground truth. Interim results log, not the manuscript.

## Setup

- Reward mode `feasibility_gated` and `configs/rl_final.yaml` (`ppo` section for PPO, `offpolicy`
  section for SAC/TD3/DDPG), seeds 42-46 for every algorithm. Only the algorithm differs.
- PPO arm: the five 2a `feasibility_gated` runs, not retrained (1,007,616 steps, 1M rounded up to
  whole rollouts). SAC, TD3, DDPG: 15 new runs, each a single uninterrupted pass of 1,000,000
  steps; `--resume` is not used for any reported run.
- Evaluation: `pipeline/03_evaluate_agent.py` on `data/ground_truth/main_grid_pooled`
  (142 contexts). Headline number is the `scale+thin` operator mode, as in 2a.
- Provenance: every run has `git_dirty: false` for training and evaluation
  (see `results/*_eval_meta.json`). The PPO arm is read from `results/2a_feasibility_gated_seed*`
  via `--arms ppo=2a_feasibility_gated`.

## Results (cost gap to best-known optimum, %, mean ± sd over 5 seeds)

| Algorithm | none (best of 40 policy steps) | scale | scale+thin | Per-seed range (scale+thin) | Within 110% | Within 125% | Grade match |
|---|---|---|---|---|---|---|---|
| PPO | 49.4 ± 20.8 | 20.8 ± 9.8 | **8.9 ± 2.4** | 6.3 - 12.5 | 71.1% | 94.4% | 94% |
| SAC | 68.0 ± 32.6 | 61.0 ± 31.3 | 24.9 ± 15.1 | 8.0 - 38.2 | 32.3% | 56.1% | 50% |
| TD3 | 89.9 ± 44.4 | 80.7 ± 43.1 | 36.1 ± 14.8 | 13.4 - 53.6 | 23.4% | 47.6% | 61% |
| DDPG | 119.1 ± 88.1 | 64.3 ± 55.3 | 32.1 ± 32.6 | 10.6 - 88.5 | 37.9% | 64.4% | 72% |

Feasibility is 100% in every algorithm and seed after either operator. Before any operator
(`none`) it is 100% for PPO, 99.9% for DDPG, 96.9% for SAC and 96.2% for TD3 (lowest single
seed 87.3%, TD3).

Per-seed `scale+thin` gaps (%), seeds 42-46: PPO 12.5 / 7.1 / 9.7 / 6.3 / 8.8; SAC 8.0 / 34.1 /
35.5 / 38.2 / 8.8; TD3 35.1 / 34.9 / 53.6 / 43.4 / 13.4; DDPG 14.4 / 10.6 / 88.5 / 14.5 / 32.6.
SAC splits into two groups (seeds 42 and 46 near PPO, the other three at 34-38%); DDPG is
dominated by one seed (44).

## Statistical comparison

Same procedure as 2a (`pipeline/09_compare_arms.py`): independent-sample exact permutation test
on the difference of means (252 splits), percentile bootstrap 95% CI, Holm correction, now over
6 pairwise comparisons. The smallest attainable Holm p is 6 x 2/252 = 0.048, so only a complete
separation of two arms' five seeds can reach significance.

scale+thin (`results/2b_comparison.csv`):

| Comparison | Mean diff (pp) | 95% CI (pp) | p (raw) | p (Holm) |
|---|---|---|---|---|
| PPO vs SAC | -16.0 | [-27.5, -4.1] | 0.095 | 0.381 |
| PPO vs TD3 | -27.2 | [-38.2, -14.7] | 0.008 | **0.048** |
| PPO vs DDPG | -23.2 | [-52.7, -3.8] | 0.016 | 0.079 |
| SAC vs TD3 | -11.2 | [-27.8, +5.3] | 0.214 | 0.643 |
| SAC vs DDPG | -7.2 | [-38.3, +17.3] | 0.714 | 1.000 |
| TD3 vs DDPG | +4.0 | [-27.1, +27.7] | 0.881 | 1.000 |

PPO vs TD3 is the only comparison that survives correction (every PPO seed is below every TD3
seed). The other PPO comparisons point the same way but do not reach significance at n = 5; the
three off-policy algorithms are not distinguishable from one another. Without the thinning step
(`2b_comparison_scale.csv`) no pair is significant (Holm p: PPO vs TD3 0.095, vs SAC 0.119, vs
DDPG 0.159), and policy-only (`2b_comparison_none.csv`) none is either (smallest Holm p 0.476).
Bootstrap CIs with n = 5 per arm are descriptive only.

## Interpretation and next step

- **PPO has the lowest mean gap and the smallest seed spread in every operator mode**, and the
  highest grade-match rate (94% vs 50-72%). PPO's worst seed (12.5%) is better than the median
  seed of every off-policy algorithm.
- **The statistical evidence for "PPO beats the alternatives" is limited to PPO vs TD3** after
  the thinning step; the PPO vs SAC and PPO vs DDPG differences are consistent in direction but
  not established at five seeds.
- **Off-policy algorithms are less stable across seeds rather than uniformly worse**: SAC seeds
  42 and 46 reach PPO's range, and DDPG has four seeds at 10.6-32.6%.
- **PPO is carried forward** to 2c (mass/CO2 transfer).

Suggested manuscript wording:

> Under the same feasibility-gated reward, PPO achieved the lowest mean cost gap (8.9%) against
> SAC (24.9%), TD3 (36.1%) and DDPG (32.1%); the difference was significant against TD3 (Holm
> p = 0.048) but not against SAC or DDPG (Holm p = 0.38 and 0.08; n = 5 seeds each). Off-policy
> algorithms showed large seed-to-seed variability (per-seed gaps 8-88%).

## Audit trail

No 2b run was resumed or replaced.
