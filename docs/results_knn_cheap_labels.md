# kNN with realistic (search-generated) labels

Status: **complete**: 3 objectives x 3 label budgets x 3 label seeds x 4 protocols (leave-one-out,
subsample n = 36 / 72 / 108 with 10 repetitions, OOD span, OOD load). Interim results log, not the
manuscript.

## Setup

- Same kNN (k = 3, fixed rule) and same operators / reference / metrics as `results_knn_baseline.md`.
  Only the labels change: instead of the pooled best-known optimum (the "best-case" labels), each
  labelled context is solved by ONE free-grade differential-evolution search of B evaluations
  (`--label_source search --label_method de`, B = 400, 1000, 4800). A context whose label search ends
  infeasible is dropped from the labelled pool, and its evaluations are still counted.
- Three label seeds (0, 1, 2) per budget; the tables give mean ± sd over the three seeds, which is
  descriptive (n = 3). For each budget the same label file is shared by all four protocols, and the
  meta files record its SHA-256. Dropped labels: one context at B = 400 for label seeds 0 and 2 in
  every objective, none otherwise.
- Provenance: all 108 result sets (3 objectives x 3 budgets x 3 label seeds x 4 protocols) are at
  commit `bdd214a` with `git_dirty: false`; each meta's labels hash matches its committed label CSV.
- Labelling cost = number of labelled contexts x B. For leave-one-out and OOD the pool is 142 (the
  meta's `labelling_evals`); for the subsample runs the cost is n x B.

Labelling cost in EC3 evaluations:

| Labelled contexts | B = 400 | B = 1000 | B = 4800 |
|---|---|---|---|
| 36 | 14,400 | 36,000 | 172,800 |
| 72 | 28,800 | 72,000 | 345,600 |
| 108 | 43,200 | 108,000 | 518,400 |
| 142 | 56,800 | 142,000 | 681,600 |

For comparison: RL training is 1,000,000 environment steps (about one EC3 evaluation each); the
best-case pooled labels cost about 96,000 evaluations per context and objective.

## Results in distribution (`scale+thin`, gap % to best-known optimum, k = 3)

Mean ± sd over three label seeds; feasibility in brackets (mean over seeds). "Best-case" is the
kNN trained on the pooled-optimum labels (single deterministic run); PPO is the 5-seed mean.

| Objective | Protocol | Best-case labels | B = 400 | B = 1000 | B = 4800 | PPO |
|---|---|---|---|---|---|---|
| Cost | LOO (141 labelled) | 1.1 (95%) | 5.9 ± 0.3 (96%) | 3.2 ± 0.3 (96%) | 0.6 ± 0.1 (95%) | 8.9 (100%) |
| Cost | subsample n = 36 | 1.7 (97%) | 6.5 ± 0.1 (97%) | 3.8 ± 0.4 (97%) | 1.3 ± 0.3 (96%) | 8.9 (100%) |
| Cost | subsample n = 72 | 1.5 (96%) | 6.4 ± 0.4 (97%) | 3.6 ± 0.4 (96%) | 0.9 ± 0.2 (96%) | 8.9 (100%) |
| Cost | subsample n = 108 | 1.5 (94%) | 6.2 ± 0.4 (96%) | 3.6 ± 0.6 (95%) | 0.9 ± 0.3 (94%) | 8.9 (100%) |
| Mass | LOO (141 labelled) | 1.0 (100%) | 6.6 ± 0.2 (100%) | 3.2 ± 0.2 (100%) | 0.7 ± 0.1 (100%) | 7.2 (100%) |
| Mass | subsample n = 36 | 1.2 (100%) | 7.0 ± 0.4 (100%) | 3.4 ± 0.1 (100%) | 1.2 ± 0.0 (100%) | 7.2 (100%) |
| Mass | subsample n = 72 | 1.1 (100%) | 6.9 ± 0.1 (100%) | 3.2 ± 0.2 (100%) | 1.0 ± 0.0 (100%) | 7.2 (100%) |
| Mass | subsample n = 108 | 0.9 (100%) | 6.8 ± 0.4 (100%) | 3.2 ± 0.2 (100%) | 0.9 ± 0.0 (100%) | 7.2 (100%) |
| CO2 | LOO (141 labelled) | 0.5 (100%) | 5.6 ± 0.4 (100%) | 1.6 ± 0.1 (100%) | 0.2 ± 0.0 (100%) | 4.6 (100%) |
| CO2 | subsample n = 36 | 1.1 (100%) | 5.8 ± 0.3 (100%) | 1.9 ± 0.1 (100%) | 0.6 ± 0.1 (100%) | 4.6 (100%) |
| CO2 | subsample n = 72 | 0.9 (100%) | 5.5 ± 0.4 (100%) | 1.8 ± 0.1 (100%) | 0.5 ± 0.0 (100%) | 4.6 (100%) |
| CO2 | subsample n = 108 | 0.5 (100%) | 5.5 ± 0.3 (100%) | 1.7 ± 0.1 (100%) | 0.3 ± 0.0 (100%) | 4.6 (100%) |

## Results out of distribution (`scale+thin`, gap % and feasibility)

Trained on all 142 grid contexts; test on the feasible OOD contexts.

| Objective | Set | Best-case labels | B = 400 | B = 1000 | B = 4800 | PPO |
|---|---|---|---|---|---|---|
| Cost | OOD span | 1.5 (69%) | 5.7 ± 3.6 (71%) | 1.6 ± 0.3 (69%) | 0.7 ± 0.1 (69%) | 9.1 (90%) |
| Cost | OOD load | 1.2 (79%) | 3.1 ± 0.9 (81%) | 1.4 ± 0.1 (79%) | 0.6 ± 0.1 (79%) | 9.7 (100%) |
| Mass | OOD span | 4.8 (88%) | 8.0 ± 1.6 (87%) | 8.3 ± 2.5 (90%) | 5.2 ± 0.3 (88%) | 7.4 (100%) |
| Mass | OOD load | 2.2 (100%) | 8.7 ± 1.2 (100%) | 3.1 ± 0.3 (99%) | 2.0 ± 0.1 (100%) | 4.5 (100%) |
| CO2 | OOD span | 3.8 (88%) | 6.2 ± 2.0 (92%) | 5.7 ± 0.5 (94%) | 4.5 ± 1.3 (88%) | 7.3 (100%) |
| CO2 | OOD load | 1.7 (100%) | 5.7 ± 3.5 (100%) | 2.6 ± 1.3 (100%) | 1.8 ± 0.0 (100%) | 5.0 (100%) |

## Reading

- In distribution the label budget matters, and B = 4800 labels are as good as the best-case ones:
  leave-one-out gaps are 0.6 / 0.7 / 0.2% (cost / mass / CO2) at B = 4800 against 1.1 / 1.0 / 0.5%
  with the pooled labels. DE searches of 4800 evaluations slightly undercut the pooled reference, so
  they are, if anything, better labels; this costs about 5% of the pooled-label effort.
- Lower budgets degrade the kNN in a roughly monotone way: about 3-4% (cost, mass) and 1.6-1.9% (CO2) at
  B = 1000, about 5.5-7% at B = 400. Spread across label seeds is small in distribution
  (sd 0.0-0.6 points).
- Against the policy, in distribution: with n = 36 labelled contexts at B = 1000 (36,000 labelling
  evaluations) the kNN is at 3.8 / 3.4 / 1.9% versus PPO's 8.9 / 7.2 / 4.6%, and at B = 400
  (14,400 evaluations) it is at 6.5 / 7.0 / 5.8%, better than PPO for cost, comparable for mass and
  worse for CO2. At B = 4800 and n = 36 (172,800 evaluations) it is at 1.3 / 1.2 / 0.6%. All of these are
  below RL's 1,000,000 training steps. This is a comparison of training/labelling cost only: the kNN
  needs labelled contexts and RL does not, the context space has two parameters, and the kNN's
  feasibility is lower than the policy's on cost in distribution (94-97% versus 100%).
- Out of distribution the picture is mixed and noisier. Cost: the kNN beats PPO's gap at B >= 1000
  (1.6 / 1.4% versus 9.1 / 9.7%) but with 69% / 79% feasibility against 90% / 100%. Mass on OOD span:
  the kNN is no better than PPO at B <= 1000 (8.0-8.3% versus 7.4%), and only 5.2% at B = 4800
  (88% feasibility). CO2 is better than PPO at B >= 1000 (5.7 / 2.6% versus 7.3 / 5.0%). The label
  seed matters more here: sd up to 3.6 points at B = 400 (cost OOD span 5.7 ± 3.6, CO2 OOD load
  5.7 ± 3.5).
- In distribution, grade match with cheap labels is lower for mass at small budgets (seed means
  33-49% at B <= 1000, 63-64% at B = 4800, versus 65-67% with pooled labels), consistent with noisier grade labels where several
  grades give near-equal mass.
- The labels come from a single search method (DE) with the fixed population rule of
  `pipeline/04`; other search settings or methods were not tried. Three label seeds bound the label
  noise only roughly.

## Audit trail

No cheap-label run was replaced.
