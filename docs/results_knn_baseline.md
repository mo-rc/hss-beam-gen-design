# kNN baseline (non-RL amortized mapping), all three objectives

Status: **complete**: leave-one-out, random-subsample and OOD protocols for cost, mass and CO2.
Deterministic (no training seeds); the numbers are reproducible exactly. Interim results log, not
the manuscript.

## Setup

- `pipeline/05_baseline_knn.py`: (span, load) -> design by a weighted vote on (grade, section type)
  among the k nearest solved contexts and a log-space weighted mean of (h, b, tf, tw) over the
  neighbours that share the winning pair. Headline k = 3, fixed in advance; k = 1 and 5 are a
  sensitivity table. Labels are the best-known optimum per context of the pooled ground truth.
- The predicted design goes through the same operators (`none`, `scale`, `scale+thin`) and the same
  summary function as the policy (`03_evaluate_agent.py`), so rows are directly comparable with
  2a-2c and the OOD log.
- Protocols: `loo` (each of the 142 grid contexts predicted from the other 141; secondary, optimistic
  because grid neighbours surround every held-out point), `subsample` (train on n = 36 / 72 / 108
  random grid contexts, test on the held-out rest, 10 repetitions; primary in-distribution),
  `ood` (train on all 142, test on `ood_span_pooled` / `ood_load_pooled`; no leakage).
- Ground-truth labelling cost: each labelled context is the best over 12 fixed (grade, section type)
  GA searches x 2 restarts x (pop 50 x 80 generations), i.e. roughly 96,000 EC3 evaluations per
  context and objective. This is the kNN counterpart of RL's one-time training cost
  (1,000,000 environment steps).

## Results (`scale+thin`, k = 3)

PPO columns are the 5-seed means from 2a/2c and the OOD log. "Within 110% (all)" counts a context
with no feasible design as a failure.

| Objective | Set | kNN gap (%) | kNN feasible | kNN within 110% (all) | PPO gap (%) | PPO feasible | PPO within 110% (all) |
|---|---|---|---|---|---|---|---|
| Cost | in-dist, LOO (141 train) | 1.1 | 95.1% | 94.4% | 8.9 | 100% | 71.1% |
| Cost | in-dist, subsample n=36 | 1.7 ± 0.8 | 96.5% | 94.8% | | | |
| Cost | OOD span | 1.5 | 69.2% | 69.2% | 9.1 | 90.0% | 66.9% |
| Cost | OOD load | 1.2 | 78.7% | 78.7% | 9.7 | 100% | 73.2% |
| Mass | in-dist, LOO | 1.0 | 100% | 98.6% | 7.2 | 100% | 74.4% |
| Mass | OOD span | 4.8 | 88.5% | 73.1% | 7.4 | 100% | 63.1% |
| Mass | OOD load | 2.2 | 100% | 100% | 4.5 | 100% | 93.6% |
| CO2 | in-dist, LOO | 0.5 | 100% | 98.6% | 4.6 | 100% | 83.2% |
| CO2 | OOD span | 3.8 | 88.5% | 76.9% | 7.3 | 100% | 60.0% |
| CO2 | OOD load | 1.7 | 100% | 100% | 5.0 | 100% | 89.4% |

Sample-size curve (subsample, mean ± sd over 10 repetitions, gap %): cost 1.7 ± 0.8 / 1.5 ± 0.9 /
1.5 ± 0.8 for n = 36 / 72 / 108; mass 1.2 / 1.1 / 0.9; CO2 1.1 / 0.9 / 0.5. The gap is essentially
flat from 36 to 108 labelled contexts; leave-one-out (141) gives 1.1 / 1.0 / 0.5.

k sensitivity (LOO gap, k = 1 / 3 / 5): cost 0.9 / 1.1 / 0.8, mass 1.3 / 1.0 / 0.7, CO2 1.0 / 0.5 /
0.2. On OOD span, k = 5 is lower than k = 3 for mass (3.8 vs 4.8) and CO2 (3.4 vs 3.8) but higher
for cost (1.9 vs 1.5). The conclusions do not depend on k.

Evaluations per design: kNN needs 1 (`none`), about 18 (`scale`) and about 60 (`scale+thin`)
EC3 evaluations; the policy needs about 40, 57 and 100-112 respectively.

Before any operator, a single kNN prediction is feasible for only 39-44% of in-distribution
contexts (cost 44.4%, mass 39.4%, CO2 42.3%) and about 0-11% on OOD sets; the operators supply
the feasibility. This `none` figure is one evaluation, whereas the policy's `none` is the best of
about 40 steps, so the two are not comparable.

## Reading

- Under the same operators and the same reference, the kNN gives a lower mean gap than PPO on
  every objective and every set tested (about 0.5-1.7% in distribution versus 4.6-8.9% for PPO;
  1.2-4.8% on OOD versus 4.5-9.7%), with about 40% fewer evaluations per design. On the
  within-110%-of-all-contexts measure the kNN is also ahead everywhere, though by only 2-6 points
  on cost OOD.
- The kNN's weakness is feasibility, not quality. On cost it fails in 4.9% of in-distribution
  contexts (LOO) and in 21-31% of OOD contexts, where PPO reaches 90-100%. The infeasible
  in-distribution cost contexts (LOO, k = 3) lie in the high-demand corner and the kNN predicts a
  grade one step too low there; uniform scaling cannot recover within the box limits.
- Mass and CO2 are easier for the kNN: their references are mostly or entirely S690, and it is 100%
  feasible in distribution and on OOD load; on OOD span 88.5%.
- The labelled data are not free. At about 96,000 evaluations per labelled context and objective,
  n = 36 labelled contexts cost roughly 3.5 million evaluations, more than RL's 1,000,000 training
  steps. The kNN is cheaper only if labels can be produced more cheaply (a lower-budget search
  would give noisier labels; not tested). RL needs no labels.
- LOO is optimistic; the subsample numbers at n = 36 are almost as good, so for this 2-parameter
  context space the grid structure is not what drives the kNN's result.
- No significance tests: the kNN has no seed variance, so only the 10-repetition spread of the
  subsample protocol is reported.
