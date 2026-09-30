# 2a: reward-mode ablation (PPO, feasibility_gated / lagrangian / shaped)

Status: **complete** -- 15/15 runs (3 reward modes x 5 seeds), evaluated against the pooled
main-grid ground truth. This is an interim results log, updated as each pipeline stage lands;
it is not the manuscript (see repo_pipeline_plan.md: "Manuscript to be rewritten from scratch
only after final evidence is frozen").

## Setup

- Algorithm: PPO, `rl_final.yaml` (frozen config, unchanged across the three arms -- only
  `reward_mode` differs).
- Seeds: 42, 43, 44, 45, 46 for every arm.
- Evaluation: `pipeline/03_evaluate_agent.py` against `data/ground_truth/main_grid_pooled`
  (142 contexts), `scale+thin` operator mode is the headline number.
- Provenance: all 15 training runs reached >=1,004,800 steps (target 1,000,000; the small
  overshoot is PPO's rollout-boundary rounding, see pipeline/02_train_agent.py), clean git
  state (`git_dirty: false`) on both training and evaluation, identical `hss_env.py` hash
  across every run. Seed 45 (lagrangian) and one hyperparameter-pilot run were interrupted
  and resumed via `--resume`; both were checked and reached the correct final step count
  proportionately -- see the discussion below.

## Results (scale+thin, cost, mean of 5 seeds)

| Reward mode | Mean gap | Per-seed range | Within 110% | Within 125% | Feasibility |
|---|---|---|---|---|---|
| **feasibility_gated** | **8.9%** | 6.3% - 12.5% | 71.1% | 94.4% | 100% |
| lagrangian | 10.5% | 8.1% - 16.9% | 70.1% | 90.4% | 100% |
| shaped | 15.8% | 9.4% - 26.4% | 54.4% | 75.2% | 100% |

Feasibility is 100% across every arm, every seed, every operator mode (including raw,
unrepaired policy output) -- reward mode affects cost quality, not compliance.

## Statistical comparison

`pipeline/09_compare_arms.py`: seeds paired by seed number across arms (the same seed value
drives PPO's weight initialisation and the env's context-sampling sequence in every arm, so
pairing is more informative than treating the 15 runs as independent), exact paired sign-flip
permutation test (2^5 = 32 permutations, appropriate at n=5), percentile bootstrap 95% CI,
Holm-Bonferroni correction across the 3 pairwise comparisons.

| Comparison | Mean diff (pp) | 95% CI (pp) | p (raw) | p (Holm) | Significant? |
|---|---|---|---|---|---|
| feasibility_gated vs lagrangian | -1.6 | [-6.2, +1.6] | 0.813 | 0.813 | No |
| feasibility_gated vs shaped | -6.9 | [-12.5, -2.4] | 0.125 | 0.375 | No |
| lagrangian vs shaped | -5.3 | [-12.4, +0.9] | 0.188 | 0.375 | No |

**None of the pairwise differences are significant after Holm correction.** feasibility_gated
vs shaped has the strongest raw signal (p=0.125 is close to the smallest p-value achievable
at n=5, and 4/5 seed pairs are directionally consistent; its 95% CI excludes zero before
correction), but does not survive the multiple-comparison penalty. This is a known limitation
of a 5-seed ablation, not evidence the underlying effect is absent -- see "how to report this"
below.

## Seed 45 (lagrangian) -- checked, not an anomaly

Seed 45's lagrangian run has the highest gap of any 2a arm/seed (16.9%) and was the only 2a
run that used `--resume` (interrupted at 800k/1M steps). Checked and ruled out as a data-
quality issue: `wall_time_min` for the resumed segment (5.91 min) is proportional to the
fraction of steps remaining (20% of a ~29-30 min full run), the training git commit and
`hss_env.py` hash are identical to every other run, and `log_std_anneal` is designed to stay
correctly synced across a resume (anneals against the grand total, re-read from
`model.num_timesteps` on restart). The elevated gap is most likely genuine seed variance,
plausibly larger than usual because lagrangian's dual-ascent multipliers are more sensitive to
the exact early-training trajectory than a fixed reward shape -- a hypothesis, not confirmed
(would need the actual training curves to check further). The seed was kept in every
aggregate above; it was not dropped.

## Hyperparameter sensitivity pilot (seed 99, held out of the headline batch)

Before committing to this ablation's config, a single-seed pilot (seed 99, never used in any
headline arm) varied PPO's clip range (0.15 -> 0.10), entropy coefficient (0.03 -> 0.05 and
-> 0.01) and learning rate (3e-4 -> 1e-4) independently against the frozen baseline:

| Variant | Gap | Delta vs. baseline |
|---|---|---|
| baseline (frozen config) | 6.8% | -- |
| clip_range=0.10 | 12.5% | +5.7 pp |
| ent_coef=0.05 | 7.4% | +0.6 pp |
| ent_coef=0.01 | 5.0% | -1.8 pp |
| lr=0.0001 | 6.5% | -0.3 pp |

No variant exceeded the ~6 pp seed-to-seed variance already observed across the five headline
seeds. Decision rule was set before running the pilot: only a change clearly larger than that
noise band would trigger adopting a new frozen config and retraining. None did; the carried-
over configuration was retained.

## Interpretation and next step

**feasibility_gated is carried forward** as the reward mode for 2b (algorithm comparison:
SAC/TD3/DDPG) and 2c (mass/CO2 transfer), on the basis of the best point estimate and lowest
seed-to-seed variance of the three arms -- explicitly a descriptive choice, not one backed by
a significant pairwise test at this sample size. Suggested manuscript wording:

> feasibility_gated achieved the lowest mean cost gap (8.9% vs. 10.5% lagrangian, 15.8%
> shaped), consistent with the constraint-classification-error hypothesis motivating its
> design; however, with 5 seeds per arm, none of the pairwise differences reach significance
> after Holm correction (all p > 0.05). feasibility_gated is carried forward to the algorithm
> and objective comparisons as the arm with the best point estimate and lowest seed-to-seed
> variance, while noting this choice rests on a descriptive rather than a statistically
> confirmed ranking.

PPO's own 5 feasibility_gated seeds (this batch) are reused as 2b's PPO arm; no retraining
needed there.
