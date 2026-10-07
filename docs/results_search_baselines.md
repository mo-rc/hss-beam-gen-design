# GA / DE / random-search baselines and head-to-head comparison

Status: **complete**: 3 objectives x 3 sets (main grid, OOD span, OOD load) x 3 methods x 8 budgets
x 5 seeds. Interim results log, not the manuscript.

## Setup

- `pipeline/04_baseline_search.py`: per-context search re-solved from scratch, free grade and
  section type. GA, differential evolution (DE/rand/1/bin) and random search, each with a fixed
  budget B of EC3 evaluations (20, 40, 60, 112, 200, 400, 1000, 4800). Population size is
  `clip(round(sqrt(B)), 6, 60)` for GA and DE; no per-budget tuning. Five seeds; results are
  summarised per seed over contexts and reported as mean ± sd over seeds.
- Each search result is scored under `none` and under the same post-hoc operators as the policy
  (`scale+thin`). Operator analyses are added to the evaluation count, so "total evals" below is
  B plus about 40-75 for `scale+thin`. The policy's `scale+thin` costs about 100-112 evaluations per
  design, so the matched row is B = 40 with operators (about 106-112 total).
- Provenance: all 9 runs at commit `2c9edfc`, `git_dirty: false`, 142 / 26 / 47 contexts.
- Reference: pooled best-known optimum, as for the policy and the kNN. Gaps are `scale+thin`
  cost-ratio minus 1, in %, over contexts that have a feasible result.

## Matched budget on the main grid (`scale+thin`, about 105-112 evaluations per design)

| Objective | PPO (5 seeds) | kNN k=3 (LOO / n=36) | GA, B=40 | DE, B=40 | Random, B=40 |
|---|---|---|---|---|---|
| Cost | 8.9 (100% feas.) | 1.1 (95.1%) / 1.7 | 43.6 (95%) | 39.2 (95%) | 34.6 (97%) |
| Mass | 7.2 (100%) | 1.0 (100%) / 1.2 | 17.9 (94.8%) | 17.7 (95.1%) | 15.4 (97.2%) |
| CO2 | 4.6 (100%) | 0.5 (100%) / 1.1 | 25.9 (94.8%) | 26.5 (95.1%) | 23.0 (97.2%) |

The kNN uses 40-63 evaluations per design at `scale+thin` (cost 63, mass and CO2 about 41; one
prediction plus operators) after its labelled data are in place; PPO about 100-112 after training; the searches have no prior and pay B plus operators.

## Gap versus evaluations on the main grid (`scale+thin`, gap %)

| Objective | Method | B=112 (about 180 evals) | B=200 (about 265) | B=400 (about 460) | B=1000 (about 1050) | B=4800 (about 4850) |
|---|---|---|---|---|---|---|
| Cost | GA | 27.1 | 19.7 | 12.3 | 5.9 | 1.6 |
| Cost | DE | 22.5 | 14.9 | 8.3 | 2.9 | -0.5 |
| Cost | Random | 27.3 | 24.1 | 20.1 | 15.5 | 11.3 |
| Mass | GA | 12.5 | 10.1 | 7.3 | 4.3 | 1.8 |
| Mass | DE | 14.1 | 11.1 | 6.9 | 2.7 | -0.1 |
| Mass | Random | 13.9 | 12.3 | 10.4 | 8.8 | 6.8 |
| CO2 | GA | 16.6 | 12.0 | 7.4 | 3.5 | 0.9 |
| CO2 | DE | 18.9 | 13.1 | 7.5 | 2.2 | -0.3 |
| CO2 | Random | 19.2 | 17.0 | 14.2 | 11.7 | 7.8 |

Feasibility after `scale+thin` is 95-98% at B = 40-112 and 100% from B = 400 for GA and DE; random
search reaches 99-100% only from B = 400-1000.

Evaluations needed to reach PPO's gap, first budget on the grid at or below it (the true crossover
lies between the previous budget and this one):
- Cost (PPO 8.9%): DE at B = 400 (about 470 evals; B = 200 is 14.9%), GA at B = 1000 (about 1,065).
- Mass (PPO 7.2%): DE at B = 400 (about 450; B = 200 is 11.1%), GA at B = 1000 (about 1,040;
  B = 400 is 7.3%, just above).
- CO2 (PPO 4.6%): GA and DE both at B = 1000 (about 1,040; B = 400 is 7.4-7.5%).
So the searches need roughly 2.5-4x (DE on cost and mass) to about 10x (GA on cost, GA and DE on
CO2) the policy's evaluations to reach its gap.

## OOD (`scale+thin`; gap %, feasibility in brackets)

Matched budget, B = 40 (about 106-112 evaluations; PPO and kNN feasibility as in 2a/2c/OOD and the kNN log):

| Set | Objective | PPO | kNN k=3 | GA | DE | Random |
|---|---|---|---|---|---|---|
| OOD span | Cost | 9.1 (90.0%) | 1.5 (69.2%) | 62.3 (72%) | 62.4 (74%) | 56.7 (83%) |
| OOD span | Mass | 7.4 (100%) | 4.8 (88.5%) | 16.6 (72%) | 21.0 (74%) | 17.1 (83%) |
| OOD span | CO2 | 7.3 (100%) | 3.8 (88.5%) | 23.7 (72%) | 28.8 (74%) | 24.6 (83%) |
| OOD load | Cost | 9.7 (100%) | 1.2 (78.7%) | 38.9 (87%) | 34.1 (85%) | 36.1 (88%) |
| OOD load | Mass | 4.5 (100%) | 2.2 (100%) | 17.4 (87%) | 19.0 (85%) | 15.7 (88%) |
| OOD load | CO2 | 5.0 (100%) | 1.7 (100%) | 25.4 (87%) | 27.2 (85%) | 22.3 (88%) |

At B = 400 (about 455 evals) DE reaches 8.5% (OOD span) and 5.6% (OOD load) on cost, with 99%
feasibility; at B = 4800 DE is at or just below the reference everywhere (-0.4% to 0.0%) and GA at
0.9-2.4%. Random search is far behind at B = 4800 (6.3-20.5%).

Budget needed to reach PPO's gap on the OOD sets (first budget on the grid): DE at B = 400 (about
470 evals) for cost on both sets and for mass and CO2 on `ood_span`; DE at B = 1000 (about 1,050)
for mass and CO2 on `ood_load`; GA at B = 1000 everywhere. Random search never reaches PPO's gap
on the OOD sets, and on the main grid only for mass at B = 4800.

## Reading

- At matched evaluations the trained policy beats GA, DE and random search by a wide margin on
  every objective and every set, in-distribution and OOD. The searches need several times the
  policy's evaluation count (about 2.5-10x, depending on method and objective) to reach its gap,
  and at full budget (B = 4800) GA and DE are better than the policy (-0.5% to 1.8% versus
  4.6-8.9%). The honest claim is a budget-bounded one: the policy is better below roughly 250-450
  evaluations per design and worse above.
- The kNN is ahead of the policy at every matched point in gap, but its feasibility is lower on
  the OOD sets (and on cost in distribution); see `results_knn_baseline.md` for the labelling-cost
  caveat. On the main grid the searches are not ahead of the kNN on gap at any budget below B = 4800; on
  the OOD sets DE at B = 1000 (about 1,050 evals) is at or below the kNN gap for mass and CO2
  (for example `ood_span` CO2 1.4% vs 3.8%), but not for cost.
- The operators matter for every method. At about 105 evaluations, random search with operators
  (15.4-34.6%) is no worse than GA and DE (17.7-43.6%), and policy-only/raw search (`none`) is much
  worse (for example cost GA at B = 40: 104.9% with 77% feasibility). The search comparison is
  therefore also a comparison of how much each method gains from the shared post-hoc operators.
- The reference is not a global optimum. DE at B = 4800 lands below it on average (cost -0.5%,
  mass -0.1%, CO2 -0.3%; OOD up to -0.4%), which means every gap reported for the policy, kNN and
  searches slightly understates the distance to the true optimum, by an amount of the same order as
  the pooling gain measured earlier (about 0.3-0.5%). The independent DE/CMA-ES cross-check of the
  ground truth remains open.
- Population size follows one fixed rule across budgets; neither GA nor DE was tuned. Better
  per-budget settings, or a reparameterised search space, could change the low-budget numbers;
  that was not tested.

## Audit trail

No search run was replaced.
