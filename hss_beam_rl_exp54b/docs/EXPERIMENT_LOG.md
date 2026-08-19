# Experiment Log: exp46 → exp62

This document summarizes the full development history of the HSS beam
generative-design environment, condensed from the project's working
sessions. It is intended as source material for the paper's Methods /
Ablation / Limitations sections — every claim below is backed by either
a validation run, a targeted diagnostic, or a direct check against the
brute-force EC3-optimal dataset (`pretrain_data/ec3_optimal_designs.csv`).

## Related work

Jeong & Jo (2021), *"Deep reinforcement learning for automated design of
reinforced concrete structures,"* Computer-Aided Civil and Infrastructure
Engineering, trained a DDPG agent with a CNN actor-critic to design ACI
318-compliant RC beams (5-step episodes, single fixed material). This
project extends that line of work in three respects: (1) PPO rather than
DDPG — explicitly named as unexplored future work in that paper's
conclusion; (2) EC3 steel design rather than ACI 318 concrete, involving
substantially more complex constraints (lateral-torsional buckling,
grade-dependent section classification via `epsilon = sqrt(235/fy)`,
shear-moment interaction); (3) a learned material-grade selection action
(S355–S690) with no analogue in the RC case, since concrete strength there
is fixed rather than chosen by the agent.

## Environment overview (final / reference version)

- **State**: 25-dim — span, load, storey, current geometry (h, b, tf, tw),
  h/b ratio, grade, section type, utilization, moment ratio, mass, cost,
  CO₂, χ_LT, flange/web slenderness, EC3 section class (one-hot),
  utilization/mass deltas, normalized design moment (`Med_norm`).
- **Action**: 6-dim continuous, tanh-bounded — 4 geometry deltas, 1 grade
  selector (softmax-snapped over 6 grade centres, temperature 0.15), 1
  section-type selector.
- **Reward**: utilization-targeting Gaussian score (peak at util=0.96),
  economy penalty (mass + cost), demand-conditioned CO₂ lifecycle bonus,
  demand-driven HSS bonus, underutilization penalty, feasibility penalties
  (moment/shear/deflection/section-class violations, geometry sanity
  checks), mass-improvement shaping, geometry-novelty bonus.
- **Episode**: up to 40 steps per randomly sampled (span, load, storey)
  scenario; terminates early after 3 consecutive steps in the feasible
  band (util ∈ [0.90, 1.05], EC3 class ≤ 3).
- **Training**: PPO (Stable-Baselines3), 8 parallel envs, `VecNormalize`
  on both observations and reward, net_arch=[256,256,128] (pi+vf),
  `ent_coef=0.03`, evaluated every 25k steps.

## exp46 → exp56: reward and sampling iteration

Early experiments (exp46–exp50, not preserved in detail) established the
core EC3 mechanics and a baseline reward structure. exp47 corrected a
grade-reward bug that penalized S690 for its higher *absolute* mass even
when it was the structurally efficient choice, and decoupled cost/CO₂
reward treatment to match real LCA behavior (HSS has lower CO₂ per unit
capacity despite higher CO₂ per kg).

exp48 (97.5% feasibility) was the first strong baseline. exp51–53
attempted a persistent ("incremental") grade action, which coupled grade
to geometry and collapsed feasibility to 44–49% — the policy learned
"start at S690, hold it" as a stable but demand-insensitive local optimum.
exp54 fixed this with a **softmax-snapped grade action**, re-selected
independently every step, restoring exp48-level feasibility (96.5%) while
correctly reintroducing demand-driven grade diversity. **exp54 is the
architecture reported as final in this project** (see the ablation below).

exp55 added `h_target_norm`/`geometry_gap` observation features (a
physics-informed depth hint, `h_target = span_m × 42`, recomputed every
step) — kept in some later branches, not present in the final reference
version, and shown by the final ablation to not be necessary for the
reported result. exp56 added a light (15%) oversampling of the
heavy-demand corner, validated in isolation as **net-negative**
(feasibility 57.5% vs 96.5%, grade diversity collapsed to two grades).

## The auto-reset diagnostic bug (methodologically important)

Both `validate.py`'s grid runner and an earlier diagnostic script had a
subtle bug: SB3's `VecEnv` contract auto-resets the underlying environment
*inside* any `step()` call that returns `done=True`, before control
returns to the caller. Code that read geometry via live environment
attributes (`env.h`, `env.fy`, etc.) immediately after such a call was
silently reading the auto-reset's fresh random draw, not the policy's
actual terminal design. Because episode stepping consumes no randomness
(only `reset()` does) and every diagnostic scenario re-seeded the same
RNG state, this produced *bit-identical* "converged geometry" across every
tested scenario regardless of span/load — which looked exactly like total
policy collapse. Verified directly (`info['h']` vs `env.h` read
immediately after the same `step()` call returned different values).
**Root-caused and fixed by sourcing all geometry from the `info` dict
returned by `step()`, never from live environment attributes, in both
`validate.py` and the diagnostic scripts included in this package.** This
fix substantially changed the picture of which checkpoints were actually
working — several "collapsed" checkpoints later diagnosed cleanly were, in
fact, performing well.

## exp57 → exp62: chasing residual oscillation in the extreme demand corner

After the auto-reset fix, clean diagnostics on exp54 showed the policy
*does* condition correctly on (span, load) and finds good designs in 4 of
5 targeted hard scenarios — but in two specific cases (13m span, load ≥100
kN/m) the trained policy touches the feasible band without holding 3
consecutive steps, so the episode never terminates early (though the
*design itself* is genuinely feasible, confirmed independently by grid
validation). Six experiments attempted to fix this residual issue:

| Exp | Change | Result |
|---|---|---|
| exp57 | Reward term penalizing grade changes after a successful step | Feasibility regressed 96.5%→80.5% broadly (not just the target scenarios) — the reward term over-generalized past its narrow trigger condition |
| exp58 | Grade-selection hysteresis: block small-margin grade switches | Fixed small-amplitude grade jitter; feasibility 86.5% |
| exp59 | Geometry-action damping once previous step succeeded | Fixed geometry blow-up; feasibility 88.5%; oscillation *relocated* to grade (larger-amplitude swings) |
| exp60 | Grade tier-adjacency restriction once successful | Fixed the specific S355↔S620 swing; feasibility 77.0% — oscillation relocated to geometry again in the same scenario |
| exp61 | `tf_target`/`tw_target` physics-informed observation hints (see below) | Feasibility 94.5%, best HSS-grade diversity of the exp57–61 series (30.1%) — a genuine, dataset-verified fix, independent of the oscillation question |
| exp62 | Sliding-window (3-of-5) termination + correction of an accidental oversampling holdover | Feasibility collapsed to 69.0% — the sliding window is exploitable: an episode can terminate on a currently-failing step as long as 3-of-last-5 were ever successful, which PPO learned to exploit at light demand |

**Sampling documentation/code mismatch**: exp56's 15% heavy-demand
oversampling, believed removed starting at exp57, was in fact still
present (unremoved in code) through exp61 despite changelogs claiming
otherwise. Discovered and corrected in exp62. This means exp57–61's
results were not clean isolated tests of only their stated mechanism.

## The tf/tw finding (exp61, validated against ground truth)

`inspect_obs_normalization.py` on exp60 showed the policy driving both
flange and web thickness to their absolute design-limit ceiling (35mm,
25mm) at a scenario the brute-force `pretrain_data/ec3_optimal_designs.csv`
sweep shows needs only `tf≈26–30.5mm`, `tw≈12.3–21.8mm`. Checked broadly,
not just this one case: **across 99 span/load contexts in the dataset,
only 1% ever require `tw ≥ 20mm` even in the thinnest feasible design** —
the policy's ceiling-pinning behavior was a general miscalibration, not a
one-off. The existing `h_target_norm`/`geometry_gap` observation features
(exp55) gave the network an explicit, physics-informed target for `h`
every step; no equivalent existed for `tf`/`tw`, which only ever received
a crude linear heuristic at episode reset, never revisited. Fitted
quadratic `tf_target(span, load)` / `tw_target(span, load)` regressions
against the same dataset (R²=0.836 and 0.766 respectively, vs. the
existing heuristic's R²=−0.295 and −2.348 — i.e. actively worse than
predicting the mean) and added them as observation features in exp61.
This is a genuine, validated finding — independent of whether the grade/
geometry oscillation mechanisms in exp58–60/62 were ultimately kept.

## The decisive ablation: training budget vs. architectural engineering

exp61 (5 stacked changes across exp57–61, trained 1M timesteps) was
initially adopted as the reference model. Before finalizing, the
confound of **training budget** (exp54 was originally trained for only
500k timesteps, half of exp61's budget) was identified and tested
directly: **exp54's unmodified environment, retrained for the same 1M
timesteps ("exp54b"), matched or exceeded every one of exp61's headline
metrics:**

| Metric | exp54 (500k) | exp61 (1M, 5 changes) | exp54b (1M, 0 changes) |
|---|---|---|---|
| Feasibility (200 episodes) | 96.5% | 94.5% | **97.0%** |
| Infeasible grid cells (/64) | 3 | 5 | **2** |
| HSS usage (S620+S690) | 30.0% | 30.1% | **40.2%** |

exp54b still exhibits the same residual grade oscillation in the extreme
demand corner as every other checkpoint tested, including the ones
purpose-built to fix it — direct evidence this is training-budget-
independent PPO advantage-estimate noise specific to an underrepresented
region of the sampled demand space, not a defect in the environment's
reward or action design that further engineering was going to resolve.

**Conclusion: the unmodified exp54 environment, trained for 1M timesteps,
is the project's final reference model ("exp54b").** The tf/tw finding
remains a real, separately-validated contribution worth reporting, but
the five-change exp57–61 stack is not adopted, since a strictly simpler
and better-performing alternative (more training, same environment) was
directly demonstrated.

## Implications for the paper

- **Report exp54's environment as the final architecture.** exp54b
  (1M timesteps) as the headline result.
- **The exp57–62 chain is strong ablation material**, not a discarded
  detour — it directly demonstrates that (a) training budget explains
  more of the observed variation than five rounds of environment
  engineering, and (b) each of several plausible-looking fixes for the
  same symptom (grade hysteresis, geometry damping, tier restriction,
  reward shaping, relaxed termination) either only partially worked or
  caused a broader regression when validated beyond the two scenarios
  that motivated it — a useful, honest methodological point about
  validation scope in RL for engineering design.
- **Disclose the residual oscillation as a limitation**, not a solved
  problem — it does not affect reported feasibility or design quality,
  only training-time efficiency in one narrow, underrepresented region.
- **The auto-reset diagnostic bug** is worth a brief methods-section
  note if the paper discusses its own validation tooling, since it
  materially changed which checkpoints looked like they were "working."
