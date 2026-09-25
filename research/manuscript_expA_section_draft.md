# Section 5.8-5.11 (draft): Fair Baseline Comparison, Pipeline-Matched Evaluation, Non-RL Amortization, and Out-of-Distribution Generalization

*Drafted for integration after the existing §5.7 and before §6 (Diagnosis). Supersedes the narrower blind-sampling/k-NN preview in §5.7, Table 8, with a rigorous, budget-accounted, statistically paired version of the same question. All figures below are read directly from `research/results/expA{,2,3,5_ood}/` on branch `research-next` and were independently verified for internal consistency (evaluation-count accounting, ground-truth re-evaluation, bit-for-bit agreement between overlapping runs, parallel-worker determinism) before being reported. Scripts: `expA_strong_baselines.py`, `expA2_pipeline_baselines.py`, `expA3_amortized_baselines.py`, `expA5_ood_amortizer.py`.*

---

## 5.8 Experiment A: Strong Optimization Baselines vs. Evaluation Budget

**Motivation.** §5.5-5.6 compare PPO against a single GA configuration at a fixed 4,800-evaluation budget. This leaves open whether PPO's apparent efficiency advantage (112 evals vs. GA's 4,868) reflects a genuine amortization benefit or simply an unfair budget mismatch against a weak set of alternatives. Experiment A closes this gap by benchmarking four strong optimizers across the full evaluation-budget range PPO actually operates in.

**Method.** GA, differential evolution (DE), CMA-ES, and two multistart local-search variants (Nelder-Mead, SLSQP) were run on all 142 contexts, 5 seeds each, at budgets of 25, 50, 100, 400, 1,600, and 4,800 EC3 evaluations, in both the raw design space (h, b, tf, tw, grade, type) and a reparameterized space (h, b/h, and the EC3 Table 5.2 flange/web slenderness ratios λf, λw), the latter decoded and uniformly scaled to the utilisation boundary. Every physics call — optimizer proposals and decoder sizing steps alike — is counted by a single wrapped meter; no evaluation is ever uncounted or free. One population-size or multistart hyperparameter per (method, space) was selected once on a 12-context holdout disjoint from the 142 evaluation contexts and from the evaluation seeds, then frozen for all budgets (no per-budget or per-context tuning). A ground-truth sanity check (re-evaluating every stored GT design under the same physics) is run automatically before every experiment in this chain and passed in all cases reported here (max relative deviation < 1e-9, zero infeasible GT designs).

**Table 9: Mean cost gap (%) vs. evaluation budget, best- and worst-space configuration per method.**

| Method | Space | 25 | 50 | 100 | 400 | 1,600 | 4,800 |
|---|---|---|---|---|---|---|---|
| DE | reparam | 39.8 | 28.4 | 20.5 | 7.4 | 1.7 | −0.3 |
| DE | raw | 78.1 | 57.1 | 36.8 | 10.3 | **1.2** | **−0.6** |
| GA | reparam | 43.3 | 32.0 | 23.0 | 8.6 | 1.7 | −0.3 |
| GA | raw | 86.7 | 58.8 | 39.4 | 12.5 | 3.0 | 0.6 |
| CMA-ES | reparam | 55.0 | 38.9 | 22.5 | 9.1 | 3.7 | 1.9 |
| CMA-ES | raw | 94.0 | 56.6 | 36.1 | 8.6 | 1.7 | 0.2 |
| Multistart NM/SLSQP | either | 40-95 | 30-70 | 20-50 | 10-25 | 5-20 | 5-20 |

**Key findings:**
- Reparameterization dominates at low budgets (≤400 evals): e.g. DE-reparam reaches 20.5% at 100 evals vs. 36.8% for DE-raw. The advantage inverts by 1,600 evals, where raw-space DE and GA slightly outperform their reparam counterparts, because the reparam decoder's boundary-sizing sub-loop consumes a large share (≈77-80%) of the budget at high evaluation counts, where that overhead no longer buys enough.
- All strong optimizers converge to ≈0% mean gap by 4,800 evals. Roughly 9% of all 42,600 runs (3,729) landed *below* the stored ground-truth value, by up to −7%, confirming the GA-generated ground truth carries noise of roughly ±1% rather than being a true, exact minimum — a fact that should be stated explicitly wherever GT is treated as ground truth in the paper.
- Local search (multistart NM/SLSQP) is not competitive at any budget tested here (5-20% mean gap even at 4,800 evals) and should not be used as a "strong baseline" claim without this qualification.
- Seed-to-seed variability is small (≤2.6 percentage points SD of the per-seed mean gap) for every method at budgets ≥400.

**Implication for §5.5-5.6:** GA at 4,800 evals is a legitimate reference point for asymptotic quality, but the honest low-budget frontier (the regime PPO actually competes in) is populated by DE/GA-reparam at ≈20-40% gap for budgets ≤100 — this is the correct comparison set for any budget-matched claim, not the raw-space, single-configuration GA baseline used in earlier tables.

---

## 5.9 Experiment A′: Equal-Pipeline Comparison (Baselines + the Same Post-Hoc Operator)

**Motivation.** Experiment A's baselines never received the `scale`/`scale+thin` post-hoc operator that every PPO result in §5.1-5.7 does. The 9.05% (§5.6) vs. 1.47-1.72% (§5.3-5.5) PPO-vs-GA comparison is therefore not yet apples-to-apples on the *pipeline*, only on the raw policy output. Experiment A′ gives GA, DE, CMA-ES, and a random-search null baseline the identical operator PPO receives, at matched total evaluation budgets, and re-scores the existing 5-seed PPO checkpoints (`e5_ppo_s{42..46}_anneal_linear`, the §5.6 arm) against the same ground truth for a directly paired comparison.

**Method.** Each baseline run is: optimizer(G evaluations) → `research/algo/repair.py`'s `repair()` operator (mode `scale` or `scale+thin`, storey = 20, identical call signature to the PPO evaluation pipeline) → final design. All operator evaluations are counted by the same meter as the generation stage; the operator's own internally reported evaluation count is asserted equal to the meter's count on every run (assertion never failed across 102,240 runs). Generation budgets G ∈ {10, 25, 40, 100, 400, 1,600} were tested, with G = 40 matched to PPO's 40-step policy horizon. As an internal consistency check, the `mode = none` rows reproduce Experiment A's cells: 17,030 of 17,040 overlapping (context, method, space, budget, seed) cells agree exactly, and the remaining 10 (all CMA-ES, raw space) differ by at most 0.17 percentage points, attributable to minor non-determinism in the `cma` library rather than a protocol error.

**Table 10: Re-scored PPO (5-seed mean, `e5_ppo_s{42..46}_anneal_linear`) under the corrected ground truth, all three operator modes.**

| Operator mode | Mean gap | Median gap | Feasibility | Mean EC3 evals/design |
|---|---|---|---|---|
| None | 43.3% | 37.4% | 1.000 | 39.9 |
| Scale | 19.5% | 16.5% | 1.000 | 56.9 |
| Scale+thin | **9.05%** | 5.7% | 1.000 | 112.6 |

This reproduces §5.6's 9.05% figure exactly (same run, re-scored against the same GT under the pipeline meter), confirming the two accounting methods agree.

**Table 11: Matched-budget comparison at G = 40, paired over the 142 contexts, 95% bootstrap CI on the mean difference (RL − best baseline pipeline; negative = RL better).**

| Operator mode | Best baseline (config) | Baseline gap | RL gap | RL − baseline | 95% CI | RL wins on |
|---|---|---|---|---|---|---|
| None | DE, reparam | 34.2% | 43.3% | **+9.2 pp** | [+5.0, +13.4] | 33% of contexts |
| Scale | DE, reparam | 33.5% | 19.5% | **−14.0 pp** | [−16.6, −11.6] | 87% of contexts |
| Scale+thin | DE, reparam | 21.1% | 9.0% | **−12.1 pp** | [−14.1, −10.1] | 94% of contexts |

**Table 12: Evaluations a baseline needs to match RL's `scale+thin` gap (9.05%); training-cost break-even (1M evals/PPO training run).**

| Baseline | Evals to match RL's 9.05% | RL's evals (112.6) | Eval saving/design | Training break-even (designs) |
|---|---|---|---|---|
| DE, reparam | ≈284 | 112.6 | ≈171 | ≈5,800 |
| GA, reparam | ≈322 | 112.6 | ≈209 | ≈4,800 |
| CMA-ES, reparam | ≈409 | 112.6 | ≈296 | ≈3,400 |

**Key findings:**
- With the operator held fixed and equal, RL's raw-policy output (no operator) is *worse* than DE-reparam at the same budget (43.3% vs. 34.2%) — the earlier apparent PPO advantage at low budgets (§5.3-5.4) is not present before the operator is applied.
- After the operator, RL reverses this and dominates every baseline pipeline tested at matched budget, by a wide, statistically significant margin (CIs exclude zero in both directions of the reversal).
- Even the strongest baseline needs 2.5-3.6× more evaluations per design to match RL's post-operator quality, and the break-even point against RL's training cost (≈3,400-5,800 designs, depending on baseline) should be reported alongside any "PPO wins on efficiency" claim, since it has not previously appeared in the manuscript.
- This localizes RL's apparent advantage over classical optimization specifically to the *interaction* between the learned policy output and the post-hoc operator, not to the policy's raw output quality — motivating Experiment A″.

---

## 5.10 Experiment A″: Amortization Without Reinforcement Learning

**Motivation.** Experiment A′ shows RL's advantage over per-instance optimizers appears only once the same post-hoc operator is applied to both. This raises the direct question §6 does not yet answer: is the advantage specific to *reinforcement learning*, or is it simply *amortization* — the fact that RL, unlike GA/DE/CMA-ES, does not re-solve each context from scratch? Experiment A″ isolates this by training ordinary supervised regressors (no reward signal, no exploration, no MDP) to predict a design directly from context, using labels generated once, offline, by DE.

**Method.** Training contexts (span, load) were drawn uniformly and continuously from the same ranges as the GT grid (span 6-15 m, load 20-140 kN·m) but off the grid points, so training and evaluation contexts never coincide. Each training context was solved once by DE (raw space, 4,800 evals; the strongest single-instance baseline from §5.8), producing a (span, load) → (h, b/h, λf, λw, grade, type) label. Three independent label sets of 208 contexts each were generated (≈998,400 DE-labelling evaluations per set — the same order of magnitude as one PPO training run's ≈1M environment steps). Four regressors (k-NN with k=3 distance weighting, a Gaussian process, a cubic ridge regression, and a two-layer MLP) were fit per label set at N ∈ {50, 100, 208} training contexts, then used to design all 142 evaluation contexts: predict → decode via the same reparameterized decoder as §5.8 (counted) → optionally apply the same `repair()` operator as §5.9 (counted). A k-NN classifier over the 12 (grade, type) combinations supplies the top-*k* categorical candidates tried per context.

**Table 13: Amortiser design quality at N = 208, `scale+thin` operator, all four models, top-1 and top-3 grade/type candidates.**

| Model | Top-k | Feasibility | Mean gap | Median gap | Mean EC3 evals/design |
|---|---|---|---|---|---|
| **k-NN** | **1** | **0.995** | **0.09%** | **−0.27%** | **64.2** |
| k-NN | 3 | 0.998 | 0.44% | −0.28% | 72.5 |
| GP | 1 | 0.995 | 3.98% | 1.97% | 65.9 |
| MLP | 1 | 0.995 | 4.16% | 2.35% | 66.9 |
| Cubic ridge | 1 | 0.995 | 5.78% | 4.04% | 67.2 |

(Negative median gaps occur because ≈9% of GT-optimal designs are themselves suboptimal by up to 7%, per §5.8; a design that legitimately beats a noisy GT reference produces a negative gap by construction.)

**Table 14: Paired comparison, k-NN (N = 208, top-1) vs. references, `scale+thin`, 95% bootstrap CI.**

| Reference | Amortiser gap | Reference gap | Diff. (amortiser − ref.) | 95% CI | Amortiser wins on | Amortiser evals | Reference evals |
|---|---|---|---|---|---|---|---|
| PPO (5-seed, §5.9) | 0.13% | 9.05% | **−8.92 pp** | [−10.34, −7.74] | 94.4% | 64.2 | 112.6 |
| DE-reparam, G=40 + operator (§5.9) | 0.13% | 21.13% | **−21.00 pp** | [−22.99, −19.13] | 99.3% | 64.2 | 108.3 |

**Key findings:**
- k-NN reaches 0.09-0.44% mean gap at 64-73 evaluations per design, with feasibility ≥ 99.5% — beating both PPO and the best classical-optimizer pipeline by a wide, statistically confirmed margin, using less total training-time compute (≈1M DE-labelling evals for N=208) than a single PPO training run, and with no training at all beyond fitting a k-NN index (seconds, not hours).
- The advantage is specific to local interpolation: GP, MLP, and cubic-ridge regressors, fit on the identical labels and evaluated with the identical decoder and operator, all land at 4-6% mean gap — an order of magnitude worse than k-NN and still above raw-policy PPO+operator's 9.05% only for GP/MLP (roughly comparable), but clearly worse than k-NN. This localizes the effect to k-NN's inductive bias (pure local interpolation on a low-dimensional, apparently smooth optimum manifold), not to "any amortiser."
- Even without any post-hoc operator, k-NN top-1 reaches 0.17% mean gap at only 3.65 evaluations per design (Table 13, `mode = none` row, full data in `expA3_full_summary.csv`), i.e. the operator is not doing the work here — k-NN's raw prediction is already near-optimal in-distribution.

**Implication for §6 (Diagnosis):** §6.1's three lines of evidence for "parameterization, not algorithm" should be read alongside a fourth, stronger one: a non-RL, non-trained-policy amortiser with the *same* reparameterized action space PPO could in principle use reaches ~40-100× lower gap than PPO at a comparable or lower evaluation budget. This suggests the ceiling is not primarily the MDP/parameterization either, but that *amortization itself*, independent of RL, captures nearly all of the achievable benefit on this problem — RL adds training cost and complexity without adding benefit over the much simpler alternative.

---

## 5.11 Experiment A‴: Out-of-Distribution Generalization

**Motivation.** Experiment A″'s result is trained and evaluated on the same (span, load) envelope. Because k-NN is pure local interpolation, it has no mechanism to extrapolate; a fair comparison against RL — which at least in principle can generalize via its learned function approximator — requires testing both methods outside the training envelope. This experiment reuses the exact Experiment A″ artifacts (same labels, same fitted models, no retraining) and asks them to design contexts genuinely outside the training range on two independent axes, against freshly generated ground truth (same GA-based procedure as the in-distribution GT, `regenerate_ground_truth.py`).

**Method.** Two OOD regions were tested: **span extrapolation** (span 16-22 m, ≈1.07-1.47× the training maximum of 15 m; load kept in-distribution; 26 contexts) and **load extrapolation** (load 150-260 kN·m, above the training maximum of 140 kN·m; span kept in-distribution; 47 contexts). PPO's existing 5-seed checkpoints (unretrained, same policy as §5.9-5.10) were evaluated on the identical OOD ground truth for a paired comparison.

**Table 15: In-distribution vs. out-of-distribution mean gap, k-NN (top-1, `scale+thin`) vs. PPO (5-seed pooled).**

| Region | k-NN gap | k-NN feasibility | PPO gap (pooled) | PPO feasibility | k-NN degradation | PPO degradation |
|---|---|---|---|---|---|---|
| In-distribution (§5.10) | 0.09% | 99.5% | 9.05% | 100% | — | — |
| Span-OOD (16-22 m) | 0.99% | 71.8% | 12.1% | 92.3% | ×11 | ×1.3 |
| Load-OOD (150-260 kN·m) | 1.61%* | 85.1% | 12.5% | 100% | ×18* | ×1.4 |

*k-NN load-OOD gap and feasibility from the topk=1 row of `expA5_load_summary.csv`.

**Table 16: Paired comparison (k-NN vs. PPO, restricted to contexts feasible for both), `scale+thin`, 95% bootstrap CI.**

| Region | k-NN gap (paired) | PPO gap (paired) | Diff. | 95% CI | k-NN wins on |
|---|---|---|---|---|---|
| Span-OOD | 0.98% | 10.49% | **−9.51 pp** | [−17.02, −4.59] | 94.7% of paired contexts |
| Load-OOD | 1.61% | 11.06% | **−9.45 pp** | [−15.89, −4.19] | 87.5% of paired contexts |

**Key findings:**
- k-NN retains a large, statistically significant advantage over PPO on both OOD axes (≈9.5 pp in both, CIs excluding zero) — the in-distribution result is not an artifact of interpolation-friendly evaluation contexts.
- k-NN's *relative* degradation is much steeper than PPO's: roughly 11-18× worse gap moving from in-distribution to OOD, vs. PPO's 1.3-1.4×. RL degrades more gracefully from a worse starting point.
- k-NN's feasibility, not just its gap, degrades with OOD distance — from 99.5% in-distribution to 72-85% OOD, worsening further at the highest loads within the OOD grid (80% feasible at 260 kN·m vs. 86% at 150 kN·m for `scale+thin`). PPO's feasibility stays at 92-100% across all conditions tested. This is a real, load-dependent failure mode of local interpolation extrapolating past its training envelope, not present in the RL policy.
- The two OOD regions tested here are moderate extrapolations (≤1.5× the training span range; ≤1.86× the training load range). Whether PPO's advantage in feasibility eventually translates into a quality crossover further from the training distribution has not been tested and should be flagged as a limitation (§7.2) rather than claimed either way.

**Caveat on accounting:** the OOD PPO evaluation loader assigns a single placeholder seed index to all 5 PPO seeds when they arrive in one pre-aggregated file (a loader limitation in `expA2_pipeline_baselines.load_rl`, not a data problem). This has been verified to leave every mean-gap and paired-comparison number in Tables 15-16 unaffected — context-level pairing is computed independent of the seed label — but the per-seed variance breakdown for the OOD PPO figures is not currently available and should be regenerated with corrected seed parsing before final submission.

---

## Revision to §6.1 (Diagnosis) — recommended addition

Insert as a fourth line of evidence, after the existing three:

> 4. **A non-RL amortiser dominates both the trained policy and per-instance optimization, in- and out-of-distribution:** A k-NN regressor trained on 208 DE-solved contexts (offline cost ≈ one PPO training run) reaches 0.09-0.44% mean gap in-distribution (vs. PPO's 9.05%) and retains a 9.5-percentage-point advantage over PPO on two independent out-of-distribution axes (span and load extrapolation), with 95% confidence intervals excluding zero in all cases (§5.10-5.11). This indicates the benefit PPO derives over per-instance optimization is amortization, not reinforcement learning specifically: the same benefit is available, and substantially larger, from ordinary supervised regression on the same reparameterized action space. RL's only demonstrated advantage over this baseline is a more graceful degradation under distribution shift — a real but narrower claim than "RL is required for competitive generative design" on this problem.

---

## Open items before these results are finalized in the manuscript

1. Fix the OOD PPO seed-parsing bug in `expA2_pipeline_baselines.load_rl` and regenerate per-seed variance statistics for Tables 15-16 (does not change any reported mean).
2. Consider a further-out OOD band (e.g. span 22-30 m or load 260-400 kN·m) to test whether PPO's flatter degradation curve ever crosses k-NN's steeper one — currently untested, stated as a limitation rather than a claim either way.
3. §5.7's original blind-sampling/single-configuration k-NN preview (Table 7-8) should either be removed in favor of §5.10-5.11's rigorous, budget-accounted, multi-model, multi-seed version, or explicitly marked as superseded.
4. All new tables should receive the manuscript's citation-key numbering once merged; source files and exact commands to reproduce every number above are listed under each script's own docstring in `research/scripts/`.
