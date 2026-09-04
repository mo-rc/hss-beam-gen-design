# E3 — Conclusions

Draft conclusions section for the paper. Built from §6/§7 of
`E3_corrected-cost-retrain_RESULTS.md`, plus the corrected-costing re-run of
the reparameterisation probe and the leave-one-out k-NN predictor
(`results/e3_reparam_probe_costing_ab_n{20,40}.csv`,
`results/e3_knn_oneshot_corrected_summary.csv`).

All gaps are relative to the corrected manufacturability-aware ground truth
(`research/pretrain_data_corrected`), over the same 142 feasible
span × load contexts, unless stated otherwise. The GA reference on that ground
truth attains 1.72% mean / 0.73% median unrepaired and 1.47% mean with the
`scale+thin` repair; readings below roughly 1.7% are therefore at or inside the
optimiser's own noise floor and should not be interpreted as improvements on it.

---

## 1. What the experiment was, and what it settles

E1 established that the original cost model let the agent label a plate girder
"rolled" and collect a rolled fabrication factor it could not have earned in a
mill, and replaced it with a manufacturability-aware cost. That correction
raised the measured gap of the best existing policy from 5.41% to 7.53%
(`scale+thin`, seed 43). Two readings of that were available and E1 §6.5 could
not distinguish them: either the policy was *mis-specified* — it had learned to
exploit costing that no longer existed, and retraining under the corrected cost
would recover the difference — or it was *transfer shock*, an artefact of
scoring a policy against a reward it was never trained on, which retraining
would remove entirely.

E3 retrained the same arm, same seed, same architecture, same 1,302,528 steps,
changing only the cost model, and evaluated it against the corrected ground
truth. The retrained policy reaches **22.65% unrepaired and 6.63% with the
`scale+thin` repair**, at 1.000 feasibility.

The comparison that answers the question is the paired same-seed control: the
pre-correction policy, unchanged, scored against the same corrected ground
truth, which gives 23.20% and 7.53%. Retraining therefore recovered **0.90 pp
of the 2.12 pp that the correction opened up in the repaired arm, and 0.55 pp
of 1.35 pp unrepaired — about 40% in both cases.** The remaining ~60% is not
recoverable by retraining under this formulation.

E1 §6.5 is thus resolved in favour of **mis-specification, with a floor**. The
correction was not merely a scoring artefact: a policy trained under the
corrected cost genuinely does better than one that was not. But it is also not
purely mis-specification, because a majority of the widened gap survives
retraining. The honest statement is that the corrected cost model exposed a
real deficiency in the policy *and* a real deficiency in the formulation, and
only the first is fixable by training.

## 2. The single-seed status of these numbers

**22.65% and 6.63% are single-seed point estimates and should be read as
such.** Seed 43 was run once. There is no seed variance around either figure,
and the paper should not present them with the apparent precision of the
five-seed means reported elsewhere.

The paired bootstrap over the 142 contexts, comparing the retrained policy to
the same-seed pre-correction policy, gives:

| Repair mode | Pre-correction | Retrained | Δ | 95% CI | p |
|---|---|---|---|---|---|
| `none` | 23.20% | 22.65% | −0.55 pp | [−2.70, +1.28] | 0.090 |
| `scale` | 11.39% | 11.34% | −0.05 pp | — | 0.017 |
| `scale+thin` | 7.53% | 6.63% | −0.90 pp | [−2.92, +0.73] | 0.018 |
| `scale+thin_rolled` | 9.42% | 9.37% | −0.05 pp | — | 0.0004 |

**Both confidence intervals on the headline improvements include zero.** The
per-context paired sign and rank tests reach conventional significance for
`scale+thin` (p = 0.018) because the improvement is consistent in direction
across contexts, but the magnitude is not separated from zero by the bootstrap,
and the retrained policy wins on only **51 of 142 contexts** — fewer than half.
The 0.90 pp figure is a shift in the mean produced by shrinking a small number
of very bad contexts, not a broad improvement.

Two further constraints on how these numbers may be used. First, seed 43 was
the **best** of the five pre-correction seeds, so 6.63% is an optimistic
estimate of what this formulation achieves under corrected costing, not a
central one. Second, 6.63% must **not** be compared against the 8.53%
five-seed transfer mean; that comparison mixes a single best seed against a
multi-seed average and would overstate the effect of retraining by more than a
factor of two. The only defensible comparison is the same-seed 7.53%.

A five-seed corrected-cost retrain, which would give a genuine confidence
interval on 6.63% and place seed 43 within a distribution, is **future work
that is blocked on compute budget**. It is not an oversight and not a question
we consider settled: four additional 1.3M-step runs were scoped and costed and
are the first thing to run when budget allows. We report the single-seed result
because the direction of the finding is stable under the paired test and
because the mechanism in §4 does not depend on the magnitude.

## 3. Mean and median disagree, and the median is the more informative one

The distributional statistics show that retraining **redistributed** the gap
more than it reduced it. Across the 142 contexts, going from the
pre-correction to the retrained policy under `scale+thin`:

| Statistic | Pre-correction | Retrained |
|---|---|---|
| mean | 7.53% | 6.63% |
| median | 4.08% | 4.75% |
| sd | 14.31% | 8.11% |
| p95 | 16.54% | 13.20% |
| worst | 89.94% | 62.50% |
| within 1% | 10.6% | 4.9% |
| within 5% | 58.5% | 52.1% |

The mean improves by 0.90 pp while the **median gets worse by 0.67 pp**, the
fraction of contexts solved to within 1% drops by more than half, and the
fraction within 5% also falls. What improves is the tail: the standard
deviation nearly halves, p95 falls 3.3 pp, and the worst context improves by
27 pp.

Retraining under corrected costing therefore bought **tail robustness at the
cost of peak accuracy**. This matters for how the contribution is described. A
paper that reports only the mean would claim the retrained policy is better; a
paper that reports only the median would claim it is worse. Both are true of
different parts of the distribution, and the full quantile table is the only
honest presentation. It also means the improvement is not the kind a
practitioner would notice on a typical member — it is insurance against the
worst member in a schedule.

## 4. The policy learned the corrected economics almost perfectly, and still cannot reach util = 1.0

The clearest positive result in E3 is that the corrected cost model was learned
essentially completely. The share of designs that are labelled rolled but
actually cost as welded under the manufacturability rules falls from
**20.4% to 3.5%**, against 2.8% for the ground-truth optima themselves — so
the retrained policy is within 0.7 pp of the reference on the specific
behaviour the correction was designed to penalise. Web slenderness moves from
72.1ε to 66.3ε, web thickness from 10.67 mm to 11.75 mm, and mean section class
from 1.61 to 1.45: the policy shifted toward genuinely rolled-manufacturable
proportions rather than relabelling.

That adaptation is worth **0.90 pp**.

The reason such a complete economic adaptation buys so little is that the
binding constraint is elsewhere. Mean utilisation converges to **0.841** in the
retrained run and **0.838** in the pre-correction run. Two runs, different cost
models, different reward signals, arriving at the same sub-unity ceiling; every
one of the 142 ground-truth optima sits at utilisation exactly 1.000. The
residual gap is capacity slack, and capacity slack is a property of the action
parameterisation, not of the cost model.

This is the same mechanism the earlier gap decomposition identified — roughly
9–13 pp of capacity slack and 11–15 pp of cross-section proportioning error,
against only about 1 pp attributable to grade and section-type choice. E3
confirms it from the opposite direction: we changed the economics as completely
as the cost model allows, the policy tracked the change almost exactly, and the
gap barely moved. E1 §6.3's negative result also replicates — the
`scale+thin_rolled` repair remains worse than `scale+thin` (9.37% vs 6.63%),
so forcing rolled-admissible proportions after the fact does not help.

## 5. Why the ceiling exists: the policy is pinned against its own exploration limit

The learning curves and the exploration diagnostics locate the ceiling
concretely.

Evaluating every checkpoint against the corrected ground truth shows the
retrained run was **still improving monotonically at termination** — 7.91% at
1.0M, 6.75% at 1.2M, 6.63% at 1.3M — while the pre-correction run had already
plateaued and was oscillating over the same window (7.84%, 8.36%, 7.53%). But
the retrained improvement is decelerating sharply: 1.16 pp over 1.0M→1.2M,
then 0.12 pp over the final 100k, with the unrepaired arm flat
(22.55% → 22.65%). Extrapolating the deceleration, further steps buy tenths of
a percentage point, not the ~5 pp that would be needed to approach the GA.
6.63% is therefore very slightly conservative as a converged value, and
emphatically not a truncation artefact. The corrected cost model also made
learning *easier* early — the retrained run leads at 200k (26.8% vs 48.6%) and
400k (9.2% vs 20.5%), and reaches 0.79 grade-match by 400k against 0.45 — so
the corrected economics are a cleaner training signal, not a harder problem.

Utilisation tells the same story: 0.506, 0.665, 0.657, 0.758, 0.803, 0.839,
0.841. It converges, and it converges below 1.0.

The exploration diagnostics say why. The annealed action standard deviation
`log_std_anneal/actual_mean_std` sat **exactly at its ceiling for 100% of the
final 30%** of the retrained run, against 57% for the pre-correction run. The
policy wanted more action variance than the schedule permitted for the entire
back half of training. Terminal `clip_fraction` averaged 0.39 with a maximum of
0.49, and `approx_kl` reached 0.027 — both high, indicating the policy was
repeatedly proposing updates large enough to be clipped. Meanwhile
`explained_variance` reached 0.998, so the critic was accurate and the
bottleneck is on the policy side, not in value estimation.

This is the signature the diagnosis predicted: the cost optimum is a vertex of
the feasible set where two or three constraints are simultaneously active, and
one side of that vertex is a discontinuous cliff — crossing into section
Class 4 returns `util ≥ 2.0` and `mass = 4000`. A Gaussian policy with
non-vanishing variance cannot place its mean on such a boundary; it must
retreat by several standard deviations, which is precisely the observed
under-utilisation and over-thick flanges and webs. The `log_std` trace shows
the policy pressing against that trade-off and being held there by the
schedule.

We report this as a mechanism consistent with the evidence, not as a
demonstrated cause. Establishing causation would require an anneal-schedule
variant, which is a new training run and is out of budget for the same reason
as the five-seed CI. We flag specifically that the observation is confounded:
a policy pinned at its variance ceiling could be under-exploring *or* could be
correctly signalling that the reward landscape rewards variance the schedule
denies, and the two are not separable from a single run.

## 6. Recommended next direction: the reparameterisation still works under corrected costing

§7 item 5 of the E3 results doc recorded that the reparameterisation direction
was untouched by the retrain. We have now measured it against the same
corrected reference that everything else in E3 uses, so it is no longer
untouched, and it survives.

Two things were re-run with **no training of any kind** — uniform random
sampling and scalar EC3 evaluations only — under both cost models at identical
contexts and identical budgets, so that the only thing varying is the costing.
The original-costing arm of both re-runs reproduces the previously reported
tables to every published digit, which is what licenses reading the corrected
arm as a costing effect rather than a reimplementation artefact.

**The reparameterisation probe** samples `(h, b/h, λ_f, λ_w, grade, type)`
instead of `(h, b, t_f, t_w, grade, type)`, with the EC3 Table 5.2 slenderness
ratios capped at the Class-3 limits so section class is admissible by
construction, then scales the section to `util = 1.0` by bisection so the
capacity constraint holds by construction too:

| Space | Proposals | EC3 evals | Original costing | Corrected costing |
|---|---|---|---|---|
| RAW | 4,000 | 4,000 | 17.5% | 22.6% |
| REPARAM | 40 | 960 | 9.4% | 14.5% |
| REPARAM | 200 | 4,800 | 3.6% | 5.9% |
| REPARAM | 1,000 | 24,000 | **1.6%** | **3.1%** |

*(40 contexts for the first two rows, 20 for the last two, matching the
budgets of the original tables.)*

The probe **does not** still reach 1.6%: at the same 1,000-proposal budget it
reaches 3.1% mean / 2.4% median under corrected costing, roughly double. Some
of that is the reference moving — the GA floor is 1.72% under corrected costing
against about 1.31% before — but not all of it. Relative to its own floor the
probe goes from 1.2× to 1.8×. Corrected costing makes the reparameterised
search genuinely harder, which is expected: the manufacturability rules make
the rolled/welded decision boundary sharper, and uniform sampling over the
shape parameters no longer places proposals as favourably.

The qualitative conclusion is nevertheless unchanged and arguably stronger.
Blind uniform sampling in the reparameterised space reaches **5.9% at 4,800 EC3
evaluations**, which already beats the fully retrained PPO policy's 6.63% after
1.3M steps and roughly 108.7 EC3 evaluations per design across 142 contexts.
Feasibility in the reparameterised space is 1.00 by construction at every
budget from 20 proposals upward, against 0.95 for raw random search at 4,000
samples. The parameterisation, not the learner, remains the bottleneck.

**The leave-one-out k-NN one-shot predictor** — inverse-distance-weighted
regression on `(h, b/h, λ_f, λ_w)` over normalised `(log span, log load)`,
nearest-neighbour grade and section type, decoded and analytically sized, with
strict leave-one-out over the 142 labels — holds up better still:

| k | Labels | Feasible | mean | median | p90 | worst | ≤1% | ≤5% |
|---|---|---|---|---|---|---|---|---|
| 1 | 142 (LOO) | 0.99 | **2.00%** | 0.64% | 3.88% | 71.3% | 0.60 | 0.93 |
| 3 | 142 (LOO) | 0.99 | 2.56% | 0.83% | 4.85% | 67.7% | 0.54 | 0.91 |
| 5 | 142 (LOO) | 0.99 | 2.83% | 1.01% | 4.98% | 66.5% | 0.48 | 0.90 |
| 1 | 71 (50%) | 0.99 | 2.05% | 0.54% | 4.45% | 71.3% | 0.59 | 0.91 |
| 1 | 36 (25%) | 0.99 | 2.28% | 0.42% | 3.76% | 91.1% | 0.62 | 0.95 |

Against a GA floor of 1.72% mean / 0.73% median, the predictor's **median gap
of 0.64% is below the GA's own median**, and its mean of 2.00% is 0.28 pp above
the GA mean — from a single weighted average plus 24 scalar EC3 evaluations per
context, with no training and no new labels. Under original costing the
corresponding figures were 1.87% mean against a 1.31% floor, i.e. +0.56 pp; the
predictor is therefore **closer to the floor under corrected costing than it
was before**. It also still degrades gracefully, losing only 0.28 pp of mean
when three-quarters of the labels are removed. Note that k = 1 is now the best
setting, where k = 3 was best under original costing — corrected costing makes
the neighbourhood less smooth, so averaging over neighbours helps less.

One new and important failure mode. The predictor's mean is now dominated by a
small set of edge-of-envelope contexts. Partitioning the 142 contexts at
span ≥ 14 m and load ≥ 100 kN/m:

| | Contexts | Original costing, k=1 | Corrected costing, k=1 |
|---|---|---|---|
| Interior | 136 | 1.55% | **0.97%** |
| Envelope corner | 6 | 12.24% | **37.16%** |

In the interior the predictor is **better** under corrected costing — 0.97%
mean, comfortably inside the GA noise floor. The entire apparent degradation is
six contexts adjacent to the structurally infeasible region already documented
at span ≥ 14.7 m and load ≥ 127.6 kN/m, where nearest-neighbour extrapolation
picks the wrong grade or section type and corrected costing punishes that
choice much harder than the original costing did. Two of those contexts are the
ones responsible for the 0.99 feasibility.

This is an extrapolation failure at a known boundary, not a general weakness,
and it is fixable without new training: a trained grade/type classifier with a
fallback grade sweep, or simply excluding the infeasible corner explicitly as
the ground-truth generation already does.

**Recommendation.** The amortised one-shot designer over a feasible-by-
construction design space remains the direction the evidence supports, and it
now stands on the same corrected reference as the rest of E3. The incremental
continuous-action PPO formulation should be reported as a rigorous negative
result with §4 and §5 as the explanation: the policy learned the corrected
economics to within 0.7 pp of the reference, and still could not reach
`util = 1.0`, because a Gaussian policy cannot sit on the vertex where the
optimum lives.

The standing caveat on the k-NN result is unchanged and must be stated
wherever it is reported: leave-one-out k-NN on a dense 12 × 12 grid is
interpolation in a two-dimensional context space, which is an easy regime. It
becomes a publishable claim only after a held-out split on a larger GA label
set, a richer context space, and the out-of-distribution tiers. The corrected
costing re-run does not change that — it only establishes that the direction
survives the E1 correction, which was the open question.
