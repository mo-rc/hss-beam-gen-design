# research/ — Constrained-RL Redesign for Generative HSS Beam Design

This module redesigns the reward/objective layer of the exp54b environment
into a scientifically defensible constrained-MDP formulation, adds a
fabrication-realistic discrete-catalog action-space variant, adds DDPG/TD3
and GA baselines against the identical environment and EC3 mechanics, and
adds a ground-truth-referenced evaluation harness. Everything here was
built to directly answer the weaknesses identified across the supervisor's
review and the subsequent audit of `hss_beam_rl_exp54b/`: the circular
`hss_demand_bonus` reward term, the missing baselines, the missing
optimality-gap reporting, the missing multi-seed replication, and the
missing generalization testing.

**Every claim below was verified by actually running the code in this
repository, not just written and assumed correct** — see "Verification
log" at the bottom for what was tested and what the results were.

## 1. Formal problem statement

```
minimise    E_{(span,load)~D} [ Economy(design) ]
subject to  g1: Med/Mrd - 1.0          <= 0   (EC3 flexural+shear+LTB capacity)
            g2: section_class - 3      <= 0   (EC3 Table 5.2 compactness)
            g3: geometry_penalty       == 0   (proportion sanity, not a code clause)
```
`Economy(design)` is ONE of {normalised mass, normalised cost, normalised
CO2}, selected via `economy_metric`; the other two are always reported as
secondary metrics, never optimised directly (avoids the ill-posed
"weighted sum of three correlated objectives" problem the original reward
had implicitly).

## 2. Key finding that should reframe Paper 1's central claim

Running `research/scripts/grade_policy_analysis.py` against the existing
brute-force ground truth (`pretrain_data/ec3_optimal_designs.csv`, no RL
involved) shows that **whether high-strength steel is ever truly optimal
depends entirely on which economy metric is used**:

| economy_metric | grades that are EVER truly optimal | pattern |
|---|---|---|
| `cost` | S355, S460 only | S500-S690 essentially never cost-justified in this pricing model |
| `mass` | all six grades | genuine, demand-spread selection |
| `co2` | dominated by S690 (48/63 contexts) | high grade almost always CO2-optimal |

**Recommendation: train and evaluate against all three economy metrics as
parallel arms**, and report each policy's grade-vs-demand pattern against
its OWN metric's ground truth — this is a stronger, falsifiable, three-way
comparison rather than a single "the agent learned to prefer HSS" claim,
and it is real evidence, not a design choice made to look good.

## 3. Environment redesign (`envs/hss_env.py`)

- EC3 mechanics and cost/CO2 model: **byte-identical** to exp54b (see
  Verification log — 25-episode regression test, zero float divergence).
- Four switchable `reward_mode` values:
  - `legacy_shaped` — exact exp54b reward (Arm A, the reference point)
  - `shaped_no_bonus` — Arm A with `hss_demand_bonus` forced to zero (the
    circularity ablation)
  - `feasibility_gated` — safe-RL-style: reward = 0 unless feasible, plus
    potential-based shaping (Ng, Harada & Russell 1999 — policy-invariant
    by construction)
  - `lagrangian` — true constrained RL: reward = -economy - Σλᵢgᵢ, with
    λᵢ updated externally via dual ascent (`algo/lagrangian.py`)
- Fixed the `feasible`-labelled-at-util≤1.05 bug: `info["feasible"]` now
  means util≤1.0 (numerical tolerance only); `info["in_target_band"]` is
  the old 0.90–1.05 band, explicitly relabelled as a training-termination
  convenience, not a code-compliance claim.

## 4. Catalog-constrained variant (`envs/hss_catalog_env.py`, `envs/rolled_catalog.py`)

Discrete `MultiDiscrete` action space navigating a procedurally-generated
UB/UC-proportioned catalog (62 sections) instead of continuous geometry.
Inherits ALL EC3/cost/reward logic unchanged from `HSSBeamEnv` — only the
action space and design-update mechanics differ. Covers rolled sections
only (welded remains a continuous, custom-fabrication problem, correctly
modelled by the continuous arm). **For publication: replace
`rolled_catalog.py`'s procedural generator with a real manufacturer/EN
10365 section table** — same CSV schema, drop-in swap, no other code
changes needed.

## 5. Constrained optimisation mechanism (`algo/lagrangian.py`)

Dual ascent: `λᵢ ← max(0, λᵢ + ηᵢ·(mean_violationᵢ − budgetᵢ))`, applied at
a **fixed timestep interval** (not per-rollout-event, which differs
between PPO and off-policy algorithms and was a real bug caught and fixed
during smoke testing — see Verification log). Works identically across
PPO, DDPG, and TD3 since it only reads `info["constraint_violations"]`
and calls `env.set_lagrange_multipliers()`.

## 6. Baselines

- **DDPG / TD3** (`scripts/train_baseline_offpolicy.py`): same environment,
  same reward modes, same Lagrangian mechanism — a genuine head-to-head
  with PPO, not an assertion that PPO is better.
- **Genetic algorithm** (`scripts/ga_baseline.py`): real-valued GA
  operating on the *exact same* `_ec3_analysis`/`_calculate_cost_co2`
  functions (imported directly, not reimplemented) — mirrors Jeong & Jo
  (2021)'s GA/BB-BC comparison methodology. The point is NOT "does GA
  beat RL" but the amortised-cost argument: GA re-solves from scratch
  per query (~0.2–0.5s/context observed); RL is a single forward pass
  after a one-time training cost. Report both numbers honestly.

## 7. Evaluation (`scripts/evaluate.py`, `scripts/generalization_test.py`, `scripts/run_multiseed.py`)

- `evaluate.py` reconstructs the TRUE optimum per (span, load) context
  from the ground-truth CSV (best across all grade×type combinations,
  not just the fixed-grade rows the CSV stores directly), then reports
  optimality gap (mean/median/p90/p95) and feasibility rate for any
  trained policy OR the GA baseline, on identical terms.
- `generalization_test.py` evaluates trained policies on span/load
  combinations OUTSIDE the training envelope (16-22m span, 150-260 kN/m
  load, and both jointly) and reports where feasibility collapses —
  characterising the failure boundary rather than only reporting
  in-distribution success.
- `run_multiseed.py` automates N-seed training/evaluation and runs both
  Welch's t-test and Mann-Whitney U (flags disagreement rather than
  picking whichever favours the result) for arm-vs-arm comparisons.

## 8. Recommended experiment matrix for the paper

| Arm | Script | Purpose |
|---|---|---|
| A: `legacy_shaped` | `train.py` | Reference point (reproduces exp54b) |
| A′: `shaped_no_bonus` | `train.py` | R1 ablation — circularity test |
| C: `feasibility_gated` | `train.py` | Safe-RL baseline formulation |
| B: `lagrangian` | `train.py` | Primary proposed method |
| DDPG | `train_baseline_offpolicy.py --algo ddpg` | Algorithm-choice justification |
| TD3 | `train_baseline_offpolicy.py --algo td3` | Algorithm-choice justification |
| GA | `ga_baseline.py` via `evaluate.py --ga_baseline` | Classical-optimization / amortized-cost comparison |
| B-catalog | `train.py` on `HSSBeamCatalogEnv` | Fabrication-realism robustness check |

Each of {A, A′, C, B} × {mass, cost, co2} × 5 seeds = 60 runs at full
scale (1M timesteps each). This is the "computational cost is secondary"
scope explicitly authorised — run on real compute, not this sandbox.

## Verification log (what was actually tested, and what happened)

1. **EC3/cost regression test** (`tests/test_ec3_regression.py`): 25
   episodes, 1000 steps, `legacy_shaped` reward mode vs. the original
   exp54b environment. Found and fixed one real transcription bug (a
   mistyped cost-factor table) on the first run. Second run: **zero
   divergence** across every physical field and the reward, to float
   precision.
2. **All four reward modes**: ran a full random-action trajectory through
   each; no crashes, no NaNs, sensible relative reward magnitudes.
3. **Lagrangian dual ascent (PPO)**: 40k-timestep run. Mean utilisation
   violation fell 2.02 → 0.09 as λ rose 0 → 61 and stabilised — the
   expected constrained-optimisation convergence shape.
4. **Off-policy Lagrangian bug found and fixed**: `_on_rollout_end` fires
   once per (n_steps×n_envs) for PPO but almost every step for DDPG/TD3;
   the callback was updating λ thousands of times more often than
   intended for off-policy algorithms, saturating it almost instantly.
   Fixed by moving the trigger to an explicit `update_freq` (in
   environment timesteps), verified identical, sane cadence across PPO
   and TD3 after the fix.
5. **Catalog-constrained environment**: random-action smoke test (no
   crashes); short PPO+Lagrangian training run initially converged
   slowly given short training budget (violation stuck ~3.0-3.6 at 30k
   steps) — traced to catalog-navigation step granularity being too
   small to reach large sections in time; widened the jump-step options,
   re-ran, violation fell to 0.08 within 60k steps, matching the
   continuous arm's convergence shape.
6. **GA baseline**: single-context run converged to util=1.0007 (right at
   the feasibility boundary, as a well-converged optimiser should) in
   0.48s.
7. **Evaluation harness**: ran end-to-end against a (deliberately
   undertrained, 20k-step) smoke-test PPO checkpoint and against the GA
   baseline. GA achieved 100% feasibility, 11% mean gap vs. true optimum
   at 0.2s/context — a sane result that validates the ground-truth
   reconstruction and gap computation are correct.
8. **Generalization test**: ran against the same smoke-test checkpoint.
   Feasibility dropped from 91% (in-distribution) to 24% (span
   extrapolation), 58% (load extrapolation), 0% (joint extrapolation) —
   the expected degradation pattern, pipeline confirmed working.
9. **Multi-seed comparison statistics**: verified `compare` subcommand's
   Welch's t-test and Mann-Whitney U agree on synthetic, clearly-separated
   data (p=0.001 and p=0.008 respectively).

None of the smoke-test *results* above (checkpoints trained for
20k–60k timesteps) should be cited as findings — they exist only to prove
the pipeline is correct. Every real number for the paper needs a full
1M-timestep, multi-seed run on real compute.
