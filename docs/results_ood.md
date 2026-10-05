# OOD evaluation of the trained PPO agents (cost, mass, CO2)

Status: **complete**: 30 evaluations (3 objectives x 2 OOD sets x 5 seeds), checkpoints from 2a
(cost) and 2c (mass, CO2), no retraining. Interim results log, not the manuscript.

## Setup

- Training range: span 6-15 m, factored UDL 20-140 kN/m. OOD sets (pooled ground truth):
  `ood_span_pooled` (spans 16-22 m, 26 of 64 contexts feasible for cost), `ood_load_pooled`
  (loads 150-260 kN/m, 47 of 64 feasible), `ood_joint_pooled` (span and load both extrapolated:
  **0 of 64 feasible, untestable**, not evaluated).
- Only contexts with a feasible reference design are scored (26 and 47 for cost; the same counts
  were used for mass and CO2). Gaps are therefore measured on the feasible subset of the OOD grid.
- Provenance: all 30 evaluations are of the existing 2a/2c checkpoints (single-pass, 1,007,616
  steps, never resumed), scored at commit `4060dc3` with `git_dirty: false`; each
  `results/ood_*_eval_meta.json` records the checkpoint and the OOD ground truth used.
- Evaluation: `pipeline/03_evaluate_agent.py`, `--economy_metric` equal to the trained objective,
  `scale+thin` as headline, as in 2a-2c. The span observation saturates at 15 m and the load
  observation at 210 kN/m, so the policy's span input is identical for every span above 15 m
  (the design-moment and utilization inputs still change).

## Results (`scale+thin`, mean ± sd over 5 seeds)

| Objective | Set | Gap to optimum (%) | Per-seed range | Feasibility | Within 110% (feasible) | Within 125% (feasible) | Within 110% (all contexts) | Grade match |
|---|---|---|---|---|---|---|---|---|
| Cost | in-distribution (2a) | 8.9 ± 2.4 | 6.3-12.5 | 100.0% | 71.1% | 94.4% | 71.1% | 94.4% |
| Cost | OOD span | 9.1 ± 3.5 | 5.3-12.9 | 90.0% | 74.4% | 94.9% | 66.9% | 80.0% |
| Cost | OOD load | 9.7 ± 2.1 | 7.4-12.3 | 100.0% | 73.2% | 88.5% | 73.2% | 77.0% |
| Mass | in-distribution (2c) | 7.2 ± 3.2 | 4.2-12.3 | 100.0% | 74.4% | 97.6% | 74.4% | 47.6% |
| Mass | OOD span | 7.4 ± 1.1 | 5.8-8.8 | 100.0% | 63.1% | 100.0% | 63.1% | 69.2% |
| Mass | OOD load | 4.5 ± 1.0 | 3.3-6.0 | 100.0% | 93.6% | 99.1% | 93.6% | 89.4% |
| CO2 | in-distribution (2c) | 4.6 ± 2.6 | 2.2-8.8 | 100.0% | 83.2% | 99.2% | 83.2% | 98.2% |
| CO2 | OOD span | 7.3 ± 0.9 | 6.2-8.4 | 100.0% | 60.0% | 100.0% | 60.0% | 96.2% |
| CO2 | OOD load | 5.0 ± 1.2 | 3.6-6.8 | 100.0% | 89.4% | 100.0% | 89.4% | 100.0% |

"Within 110% (all contexts)" counts a context with no feasible design as a failure (per-seed
within-110% share times feasibility, averaged over seeds); the other within-% columns use feasible
contexts only, as in earlier results logs.

Before the operators, feasibility on OOD span is below 100%: policy-only (`none`) is 86.2% for
cost, 92.3% for mass and 93.8% for CO2; `scale` raises mass and CO2 to 100% and cost to 90.0%.
OOD load is 100% feasible in every mode. With `scale+thin` the only run set that is not fully
feasible is cost on OOD span (90.0%, lowest seed 84.6%).

## Reading

- After `scale+thin` the OOD gaps are in the same range as in-distribution for every objective
  (cost 9.1-9.7% vs 8.9%; mass 4.5-7.4% vs 7.2%; CO2 5.0-7.3% vs 4.6%), and feasibility is 100% except
  cost on OOD span. This differs from the earlier project's OOD tests, which reported degradation; those
  used a superseded checkpoint batch and pre-cost-fix ground truth (see
  `docs/archive/research_audit.md`), so the two are not comparable. The evidence here is five
  seeds on the feasible subset only.
- The gaps are not a pure policy effect. The operators project the policy's design onto the
  EC3 capacity boundary using the physics, independent of the policy's training range, so the
  post-operator gaps partly reflect the operator rather than the policy.
- Only OOD contexts for which the design box still admits a reference solution are scored, so
  the OOD context set is a selected subset and its gap is not directly comparable to the
  in-distribution gap on a like-for-like context set.
- The one visible OOD cost is on cost-optimal span extrapolation: some contexts have no feasible
  design from the agent even after `scale+thin` (10% on average), and grade match drops
  (94% -> 77-80%) for cost.
- `ood_joint` cannot be evaluated: no feasible reference design exists inside the current design box.

## Audit trail

No OOD evaluation was rerun or replaced.
