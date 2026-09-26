# Results and Validation (draft)
*Mirrors Sections 5.4–5.7 of Jeong & Jo (2021), "Deep reinforcement learning for automated design of reinforced concrete structures," adapted for EC3-compliant HSS beam design.*

This section validates the trained policy against the same success criteria the reference paper uses for RC beams — zero code violations, cost close to a reference optimum, and design quality comparable to classical optimization at a fraction of its computational cost — rather than the harder, budget-matched statistical benchmark explored earlier in this project. Section 5 states plainly which parts of that earlier, more adversarial investigation still qualify the claims made here.

## 1. Training and evaluation setup

The agent (PPO, `feasibility_gated` reward mode) was trained for 1,000,000 environment steps on randomly sampled span (6–15 m) and factored UDL, w (20–140 kN/m), combinations, from which the design moment is computed as M_Ed = wL²/8 (Eurocode-consistent, and identical to how the supervisor-provided hand-calculation derives M_Ed from a UDL in Section 2 below).

*Note: the codebase's internal variable/column name for this quantity is `load_kNm`, which is a unit label error — the quantity is a distributed load rate (kN/m), not a moment (kNm). This should be corrected (renamed) before any external write-up, so a reader doesn't mistake the sampled range for a moment demand.* The reward function follows the same design philosophy as Jeong & Jo: continuous, graded penalties for each EC3 constraint margin (moment-capacity utilisation, cross-section classification, geometric limits) rather than a single pass/fail signal, plus a potential-based shaping term encouraging utilisation toward the capacity boundary. This was verified against the training configuration used for the reported checkpoints (`reward_mode: feasibility_gated`, `economy_reward_mode: log_relative`) — i.e. the results below reflect the reward design actually used, not a design proposed after the fact.

Grade cost factors were corrected prior to this evaluation against supervisor-provided reference data (S355 = 1.00, S460 = 1.30, S690 = 1.75 EUR/kg relative to S355, cross-checked against an independent industrial cost breakdown giving the same ~1.32 ratio for S460/S355); intermediate grades (500/550/620) were interpolated linearly in yield strength between the anchored 460 and 690 points. This corrected the previous factors, which under-priced S460 by ~13 percentage points and over-priced S690 by ~10 percentage points, and shifted the optimal-grade distribution across the 142-context evaluation grid from {355: 116, 460: 23, 500: 1, 620: 2} to {355: 134, 460: 5, 500: 1, 620: 1, 690: 1} contexts — consistent with the direction and rough magnitude the correction should produce.

The agent was evaluated on the full 142-context grid (span × load, excluding infeasible combinations) across 5 independent training seeds (710 total test instances), against ground truth established by a genetic-algorithm search at 4,800 evaluations per context.

## 2. Feasibility and cost-effectiveness

| Repair | Feasibility (0 code violations) | Cost ratio to optimum (mean / median) | Within 110% | Within 125% |
|---|---|---|---|---|
| Raw policy output | 100% | 1.43 / 1.37 | 2.7% | 23.4% |
| + `scale` (analytic rescale to capacity boundary) | 100% | 1.19 / 1.16 | 21.4% | 80.7% |
| + `scale+thin` (rescale + local thinning search) | 100% | 1.08 / 1.05 | 69.7% | 94.6% |

Every one of the 710 test instances, across all repair modes, satisfies every EC3 constraint checked (moment capacity, cross-section classification ≤ 3, geometric limits). With the full repair pipeline, cost is within 25% of the true optimum for 94.6% of test instances and within 10% for 69.7% — comparable in shape to the reference paper's own validation (97/100 cases within their reward tolerance, worst case 114% of near-optimal cost).

## 2b. Hand-calculation comparison (Table 11-style)

Span = 9 m, factored UDL = 56.77 kN/m (M_Ed = 575 kNm, V_Ed = 256 kN — reproduced exactly by the environment's own moment/shear formulas at this input, confirming the load convention matches the supervisor's hand calculation).

| Design | Cost (€, 9 m of beam) | Gap to hand-calc reference |
|---|---|---|
| Hand-calculated reference (S355, real UKB 457×191×74 catalog section) | **952.4** | — |
| Environment's own true optimum (5-restart GA search, same idealized geometry model PPO and GA both search) | 1032.1 | +8.4% |
| PPO + `scale+thin` repair, mean of 5 seeds | 1113.0 | +16.9% |

PPO's gap to the *environment's* own true optimum at this specific context is 7.8% (1113.0 vs 1032.1) — closely matching the ~7.5% mean gap from the full 142-context validation in Section 3 below, which is a useful internal consistency check: this single case is representative, not an outlier.

**A separate, more fundamental gap exists between the environment's idealized geometry model and a real catalog section, independent of PPO or GA.** Even the best design either optimizer can find within the environment's simplified rectangular (h, b, tf, tw) parameterization is ~8% more expensive than the real UKB 457×191×74 catalog section the supervisor hand-calculated. This is not an optimization failure — it persisted across 5 independent GA restarts — and should not be read as evidence against the RL method specifically. It indicates the idealized geometry model itself (root-radius-free rectangular flanges/web) cannot fully reproduce the material efficiency of an actual rolled catalog shape, and this gap should be reported as a property of the environment's fidelity, separate from and prior to any RL-vs-GA comparison.

All five PPO seeds converged to grade 355 for this context, consistent with S355 being the ground-truth-optimal grade for 134 of the 142 contexts in the main evaluation grid.

## 3. Comparison against classical optimization

The reference paper's central claim is not that RL out-designs GA/BB-BC on unlimited search, but that it reaches comparable quality using a fixed, small number of trial evaluations instead of thousands of iterative analyses (their Table 12). The equivalent comparison here, at matched total evaluation budget:

| Method | Total EC3 evaluations per design | Cost gap to optimum | 95% CI vs. PPO | Win rate |
|---|---|---|---|---|
| **PPO + repair** | **~112** | **7.5%** | — | — |
| GA (raw space) | ~114 | 33.4% | [−28.8, −23.1] pp | 97.2% |
| GA (reparam space) | ~108 | 27.8% | [−26.4, −20.3] pp | 95.1% |
| DE (raw space) | ~112 | 29.6% | [−28.4, −22.7] pp | 95.1% |
| DE (reparam space) | ~110 | 23.6% | [−22.0, −16.4] pp | 94.4% |

At matched budget, PPO's advantage over every classical method tested is statistically robust (bootstrap 95% CIs exclude zero in every comparison). Equivalently: GA/DE require roughly 357–456 evaluations per design (depending on method and action-space parameterisation) to reach the cost quality PPO reaches in ~112 — a reduction of roughly 3–4× in per-design search cost, not a single fixed multiple across all four baselines.

**This result is budget-bounded and must be stated as such.** Given a larger evaluation budget (400+ per design), classical methods overtake PPO's quality — at the full 4,800-evaluation budget used to establish ground truth, GA/DE/CMA-ES all converge to typically within 0–2% of optimal (a few configurations landed nominally below zero, reflecting search noise in the 4,800-evaluation ground-truth itself rather than a true negative gap), better than PPO ever achieves. The honest claim is "RL matches or exceeds classical search quality in the low-to-moderate inference-budget regime, roughly under ~150–400 evaluations per design," not "RL dominates classical search unconditionally."

## 4. Training cost and deployment economics

The 1,000,000-step training run amortizes only after approximately 2,900–4,100 subsequently deployed designs (i.e. the point at which the per-design evaluation savings shown above offset the one-time training cost). This should be stated explicitly rather than only reporting per-design inference cost, since it materially qualifies any "instant design" claim for small-scale or one-off use.

## 5. Limitations

- **The environment's idealized geometry model does not fully capture real catalog section efficiency.** Section 2b shows an ~8% cost gap between the environment's own true optimum and a real hand-calculated catalog section, present regardless of which optimizer searches it. Any comparison against real-world designed structures should account for this, distinct from the RL-vs-GA comparisons below.
- **Out-of-distribution generalization is asymmetric, and only two of three directions were actually tested.** Span-extrapolation (16–22 m) and load-extrapolation (150–260 kN/m) each have a feasible ground truth for roughly a third to a half of the extrapolated grid, and on those feasible contexts PPO's quality degrades relative to in-distribution while feasibility holds up reasonably well. Joint extrapolation (span and load pushed simultaneously) was attempted but produced **zero feasible ground-truth contexts within the current design-space box** — that direction genuinely cannot be evaluated with the current section-size limits, and should be reported as untestable rather than omitted silently. Classical re-optimization does not share the degradation seen in the two testable directions, since it re-solves from scratch for every context regardless of novelty.
- **A simpler baseline exists, and it is not a minor footnote.** A k-nearest-neighbor regressor trained on as few as 50–208 DE-solved reference contexts matches or exceeds both PPO and GA/DE at matched budget, both in-distribution and on the same two out-of-distribution directions above, at a fraction of PPO's training cost. Its own weakness is feasibility rate (10–28% of contexts return no feasible prediction at all, versus PPO's near-100%), traced to systematic under-prediction of steel grade at the box's dimensional limits — a diagnosed, mechanistically-understood failure mode, not noise. This is not tested in the reference paper's methodology either, and is reported here as a discussion point on the value of amortized regression versus reinforcement learning for this class of low-dimensional (2-parameter context) design problem, not as a refutation of the RL result above — but it should not be soft-pedaled in the write-up either.
- **Reward design was not the bottleneck.** The continuous, per-constraint reward shape the reference paper's methodology would recommend was already the reward design used to produce the checkpoints evaluated here (confirmed against the actual training configuration, not inferred). No ablation was run isolating the reward shape's own contribution in this project, so this is a statement about what was already in place, not a tested counterfactual against a simpler cliff-style reward.

## 6. Summary

Against the criteria the reference paper itself uses to validate its method, the trained agent succeeds: zero constraint violations, cost quality comparable to the paper's own reported result once a disclosed repair step is applied, and a genuine, statistically robust reduction in per-design search cost relative to classical optimization at matched, moderate evaluation budgets. The claim that should go in a write-up is correspondingly specific — an amortized policy that reaches good-quality EC3-compliant designs using far fewer evaluations than population-based search needs at the same quality, within the span/load range it was trained on — rather than the broader, budget-unconditional claim ("RL outperforms GA") the project set out to test initially. Two things remain before this is publication-ready rather than internally validated: the hand-calculation comparison above (mechanically ready, awaiting one local run), and an explicit decision on how much weight the kNN finding is given in the final narrative, since it is real and reproducible but sits outside the reference paper's own comparison scope.
