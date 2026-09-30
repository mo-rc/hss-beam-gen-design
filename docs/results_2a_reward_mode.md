# 2a: reward-mode ablation (PPO, feasibility_gated / lagrangian / shaped)

Status: **complete**: 15 runs (3 reward modes x 5 seeds), evaluated against the pooled
main-grid ground truth. This is an interim results log, updated as each pipeline stage lands;
it is not the manuscript (which is rewritten from scratch once all evidence is frozen).

## Setup

- Algorithm: PPO with `configs/rl_final.yaml`, identical across the three arms; only
  `reward_mode` differs.
- Seeds 42-46 for every arm. Each run is a single uninterrupted training pass of 1,007,616
  steps (1M rounded up to whole 8192-step rollouts); `--resume` is not used for any reported run.
- Evaluation: `pipeline/03_evaluate_agent.py` on `data/ground_truth/main_grid_pooled`
  (142 contexts). Headline number is the `scale+thin` operator mode.
- Provenance: every run has `git_dirty: false` for training and evaluation and the same
  `hss_env.py` hash as the ground truth (see `meta.json` / `*_eval_meta.json`).

## Results (cost gap to best-known optimum, %, mean ± sd over 5 seeds)

| Reward mode | none (best of 40 policy steps) | scale | scale+thin | Per-seed range (scale+thin) | Within 110% | Within 125% | Feasibility |
|---|---|---|---|---|---|---|---|
| feasibility_gated | 49.4 ± 20.8 | 20.8 ± 9.8 | **8.9 ± 2.4** | 6.3 - 12.5 | 71.1% | 94.4% | 100% |
| lagrangian | 61.3 ± 8.1 | 30.3 ± 4.2 | 9.4 ± 1.4 | 8.1 - 11.4 | 75.4% | 90.6% | 100% |
| shaped | 67.4 ± 4.9 | 46.6 ± 6.3 | 15.8 ± 6.6 | 9.4 - 26.4 | 54.4% | 75.2% | 100% |

Feasibility is 100% in every arm, seed and operator mode, so reward mode affects cost quality,
not compliance. The post-hoc operator does most of the work in every arm (policy-only gaps are
49-67%); report the policy-only and operator-adjusted numbers together.

## Statistical comparison

`pipeline/09_compare_arms.py` treats the arms as independent 5-seed samples: exact two-sample
permutation test on the difference of means (all 252 splits), percentile bootstrap 95% CI, and
Holm correction over the 3 pairwise comparisons. A paired-by-seed test was not used: the shared
seed only fixes initial weights and context sampling, the observed across-arm correlation of
per-seed gaps is about zero, and a paired exact test at n = 5 cannot reach p < 0.0625 (0.19
after Holm), so it could never report significance.

scale+thin (`results/2a_comparison.csv`):

| Comparison | Mean diff (pp) | 95% CI (pp) | p (raw) | p (Holm) |
|---|---|---|---|---|
| feasibility_gated vs lagrangian | -0.5 | [-2.6, +1.8] | 0.706 | 0.706 |
| feasibility_gated vs shaped | -6.9 | [-12.8, -1.9] | 0.048 | 0.119 |
| lagrangian vs shaped | -6.4 | [-12.1, -1.7] | 0.040 | 0.119 |

Before the thinning step (`results/2a_comparison_scale.csv`) shaped is separated more clearly:
+25.8 pp vs feasibility_gated and +16.4 pp vs lagrangian, both p = 0.008 (Holm 0.024). Policy-only
(`_none.csv`) shows no significant differences (raw p 0.11-0.25). The bootstrap CIs with n = 5
per arm are narrow relative to the true uncertainty and are descriptive only.

## Hyperparameter sensitivity pilot (seed 99)

A single-seed pilot varied PPO's clip range, entropy coefficient and learning rate one at a
time against the frozen `feasibility_gated` config. Seed 99 is not one of the headline seeds
(42-46), but it was evaluated on the same 142-context main grid as the headline runs, so the
pilot is not a held-out test of any tuning: it can only inform whether the frozen config is
fragile, and no value from it is adopted. All five runs are single-pass (1,007,616 steps,
commit `8440488`, `git_dirty: false`, pooled ground truth); files in `results/hp_pilot/`.

| Variant | scale+thin gap | Delta vs baseline | none (best of 40 policy steps) |
|---|---|---|---|
| baseline (frozen: clip 0.15, ent 0.03, lr 3e-4) | 6.8% | -- | 30.2% |
| clip_range = 0.10 | 7.0% | +0.2 pp | 55.0% |
| ent_coef = 0.05 | 7.5% | +0.6 pp | 42.6% |
| ent_coef = 0.01 | 5.0% | -1.8 pp | 36.6% |
| lr = 1e-4 | 6.5% | -0.3 pp | 51.1% |

After the operator, every variant is within the spread of the five headline gated seeds
(6.3-12.5%, sd 2.4 pp); the largest move (ent_coef 0.01, -1.8 pp) is under one seed-sd and comes
from one seed, so it is not evidence for a better setting. Policy-only gaps vary more (30-55%),
but the operator absorbs most of it; that is a single-seed observation and is not interpreted.
The frozen configuration was retained.

## Interpretation and next step

- **feasibility_gated and lagrangian are tied** after the full operator (8.9% vs 9.4%,
  p = 0.71). Lagrangian has the tighter seed spread (sd 1.4 vs 2.4); gated is better before the
  operator (20.8 vs 30.3 after `scale`, raw p = 0.087).
- **shaped is worse**: 15.8% mean with a wide seed spread (9.4-26.4%). The scale+thin difference
  is p ~ 0.04-0.05 raw and does not survive Holm (0.12); the `scale` difference does (0.024).
- **feasibility_gated is carried forward** to 2b (SAC/TD3/DDPG) and 2c (mass/CO2 transfer) as the
  arm with the lowest mean gap at every operator level and the simplest mechanism (no dual
  multipliers to tune). This is a tie-break on point estimates, not a confirmed ranking against
  lagrangian. Its five seeds are reused as 2b's PPO arm.

Suggested manuscript wording:

> feasibility_gated and lagrangian achieved indistinguishable mean cost gaps (8.9% and 9.4%;
> exact permutation p = 0.71, n = 5 seeds each); shaped was worse (15.8%), a difference that is
> significant before the thinning step (Holm p = 0.024) and marginal after it (Holm p = 0.12).
> feasibility_gated was carried forward as the simplest arm with the lowest point estimate.

## Audit trail

Three runs (lagrangian seed 45, shaped seeds 43 and 45) were first trained through interrupted,
resumed sessions. Under the rule that every reported run is a single uninterrupted pass, they
were replaced by single-pass trainings with the same seed, config and commit. The original
files are archived outside git and appear in no table here.
