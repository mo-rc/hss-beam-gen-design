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

*Note: lagrangian seed 45 was originally trained with `--resume` (interrupted at 800k/1M
steps). Retraining it fresh (`--force`, full 1M steps in one run) changed its gap from 16.9%
to 11.4% and is the number used below. See "Seed 45" below for the full before/after and why
this was corrected rather than silently overwritten.*

| Reward mode | Mean gap | Per-seed range | Within 110% | Within 125% | Feasibility |
|---|---|---|---|---|---|
| feasibility_gated | 8.9% | 6.3% - 12.5% (6.2pp) | 71.1% | 94.4% | 100% |
| lagrangian | 9.4% | 8.1% - 11.4% (3.3pp) | 75.4% | 90.6% | 100% |
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
| feasibility_gated vs lagrangian | -0.5 | [-2.9, +1.6] | 0.813 | 0.813 | No |
| feasibility_gated vs shaped | -6.9 | [-12.5, -2.4] | 0.125 | 0.375 | No |
| lagrangian vs shaped | -6.4 | [-12.4, -1.4] | 0.125 | 0.375 | No |

**None of the pairwise differences are significant after Holm correction.**
feasibility_gated-vs-shaped and lagrangian-vs-shaped now show equally strong raw signal
(p=0.125 each, near the smallest p achievable at n=5; both 95% CIs exclude zero before
correction, neither survives Holm). **feasibility_gated vs lagrangian is effectively a tie**
(0.5pp difference, CI [-2.9, +1.6]) -- see "Seed 45" below for why this is smaller than the
1.6pp originally reported. This is a known limitation of a 5-seed ablation, not evidence the
shaped-vs-others effect is absent -- see "how to report this" below.

## Seed 45 (lagrangian) -- resumed run was noisier than a fresh run, now corrected

**Original run** (`--resume`, interrupted at 800k/1M steps): gap 16.9%, `within_110pct` 31%.
Checked at the time and no data-quality issue was found -- `wall_time_min` for the resumed
segment (5.91 min) was proportional to the fraction of steps remaining (20% of a full run),
training commit and `hss_env.py` hash matched every other run, `log_std_anneal` is designed
to stay synced across a resume. The elevated gap looked like genuine (if unusually large)
seed variance.

**Retrained fresh** (`--force`, full 1,007,616 steps in one uninterrupted run, same seed,
same commit `8440488`, same config, `git_dirty: false` on both training and evaluation):
gap dropped to **11.4%**, `within_110pct` rose to 57%. This changes two things:
- lagrangian's 5-seed mean gap: 10.5% -> **9.4%** (now within 0.5pp of feasibility_gated,
  not 1.6pp)
- lagrangian's per-seed spread: 8.8pp -> **3.3pp** (now TIGHTER than feasibility_gated's
  6.2pp, not wider)

**Interpretation:** the resume itself was very likely a real source of extra noise for this
specific run, not just an unlucky seed. A plausible mechanism: lagrangian's dual-ascent
multipliers depend on a running buffer of recent constraint violations
(`hssbeamgen/algo/lagrangian.py:LagrangianCallback`), and while the resume logic correctly
restores the multiplier values and callback state (see `pipeline/02_train_agent.py --resume`
tests), the buffer's contents at the moment of interruption may not represent as
representative a sample as one built up continuously -- this is a hypothesis, not confirmed,
and would need the actual training curves to check further. Practically: **prefer training
lagrangian arms straight through when the session allows it**; a resumed lagrangian run isn't
wrong, but this one data point suggests it may be a noisier estimate than a fresh run at the
same seed.

The corrected value is used throughout this document and in `results/`. The original
(resumed) run's files are not deleted -- see the audit-trail convention already used
elsewhere in this project (archive, don't delete).

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

With the corrected seed 45, **feasibility_gated and lagrangian are statistically and
practically indistinguishable** (8.9% vs 9.4%, 0.5pp apart, CI [-2.9, +1.6]) -- this is a
genuine tie, not a near-win. Both clearly separate from shaped, which remains the worst arm
by a wide margin on every metric.

**feasibility_gated is still carried forward** to 2b (algorithm comparison: SAC/TD3/DDPG) and
2c (mass/CO2 transfer), but on tiebreaker grounds only: a marginally better point estimate,
and a simpler mechanism (no dual-ascent multipliers to tune, and no resume-sensitivity
concern of the kind observed in lagrangian seed 45). This is a weaker basis than originally
reported and should be stated as such. Suggested manuscript wording:

> feasibility_gated and lagrangian achieved statistically indistinguishable mean cost gaps
> (8.9% and 9.4% respectively; 95% CI on the difference [-2.9, +1.6] percentage points,
> n=5 seeds each), both clearly outperforming shaped (15.8%, though this difference also does
> not survive Holm correction at n=5). feasibility_gated was carried forward to the algorithm
> and objective comparisons on tiebreaker grounds (marginally better point estimate, simpler
> mechanism); this choice is not a statistically confirmed ranking against lagrangian
> specifically, and a reviewer preferring lagrangian on other grounds (e.g. its explicit
> constraint-budget interpretability) would not be contradicted by this data.

PPO's own 5 feasibility_gated seeds (this batch) are reused as 2b's PPO arm; no retraining
needed there.
