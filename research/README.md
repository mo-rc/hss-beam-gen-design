# research/ — Constrained-RL Framework for Generative HSS Beam Design

Reinforcement-learning framework for generative design of high-strength
steel (HSS) I-section beams to Eurocode 3 (EN 1993-1-1). This file
describes the current methodology and is kept in exact sync with the
code -- if you change `hss_env.py`'s reward/constraint logic or
regenerate ground truth, update this file in the same commit.

## 1. Formal problem statement

```
minimise    E_{(span,load)~D} [ Economy(design) ]
subject to  g1: Med/Mrd - 1.0          <= 0   (EC3 flexural+shear+LTB capacity)
            g2: section_class - 3      <= 0   (EC3 Table 5.2 compactness)
            g3: geometry_penalty       <= 0   (one-sided: 0 when b<=h, >0 when
                                                b>h; proportion sanity, not an
                                                EC3 code clause)
```
`Economy(design)` is ONE of {normalised mass, normalised cost, normalised
CO2}, selected via `economy_metric` -- trained and evaluated as **three
parallel arms**, not a single default, because (see Section 3) whether HSS
grade selection is demand-appropriate turns out to depend entirely on which
one is chosen.

## 2. Reward modes (`envs/hss_env.py`)

- **`shaped`** -- weighted-sum reward shaping (economy + utilisation-target
  Gaussian + feasibility penalties + mass-improvement shaping). Reference/
  control arm.
- **`feasibility_gated`** -- safe-RL style: reward=0 unless feasible, plus
  potential-based shaping (Ng, Harada & Russell 1999 -- policy-invariant).
- **`lagrangian`** -- reward = -economy - Sum(lambda_i * g_i), lambda_i
  updated externally via dual ascent (`algo/lagrangian.py`). Primary
  proposed method.

No reward term anywhere in this codebase references a specific grade or
grade threshold. Every grade-dependent number the agent experiences comes
from `_calculate_cost_co2`'s market-rate tables and `_ec3_analysis`'s
Fy-dependent capacity calculation -- both physically/economically
grounded, neither reward-engineered. This is a deliberate structural
choice: if demand-appropriate grade selection emerges under training, it
is evidence of a genuine relationship between grade, EC3 mechanics, and
economics, not an artefact of reward design, and can be tested three
separate ways (Section 3) rather than asserted once.

`shaped`'s economy term uses `self._economy(mass, cost, co2)`, identical
to the other two modes -- all three reward modes optimise the SAME
objective and differ only in *how* they are incentivised toward it, which
is the precondition for a valid reward-formulation ablation. Its
utilisation-score curve gives full reward only for util<=1.0 (the actual
`feasible` boundary used everywhere else in this codebase), blends down
through the (1.0, 1.05] training-termination band, then applies a steep
quadratic penalty beyond 1.05 -- so the reward signal never tells the
agent that mild EC3 non-compliance is nearly as good as compliance.

## 3. Objective-specific ground truth (`scripts/regenerate_ground_truth.py`)

Ground truth is produced by an independent GA search per (span, load,
grade, section_type, **economy_metric**) -- three full searches per
context, not one, since the mass-optimal geometry for a given context is
generally not the cost-optimal or CO2-optimal geometry for that same
context (fabrication cost/CO2 factors weight grade and section_type
differently than mass alone). Produces three separate files:
`pretrain_data/ec3_optimal_designs_{mass,cost,co2}.csv` (1,623 rows
each, out of 1,728 possible span x load x grade x type combinations --
the remaining 105 are structurally infeasible, see Section 5).

**Validation.** Cross-checked against a substantially larger-budget GA
search (n=100 contexts per metric, ~9x the generation budget) and against
an independent algorithm (scipy `differential_evolution`, a different
library entirely). Result: all three ground-truth files carry a similar,
small residual optimality gap relative to a much larger-budget search --
**mean 0.86-0.89%, max 3.86%, remarkably stable across all three
metrics.** This is the practical noise floor of population-based GA
search on this 4D nonlinear-constrained problem, not a bug and not
metric-specific; a substantially larger validation budget did not shrink
it further, so this is treated as a stated, quantified methodological
bound rather than something further compute is expected to eliminate.

**Why mass/cost/co2 show near-identical per-context gaps (mechanistic
explanation, verified computationally):** for a FIXED grade and section
type, `_calculate_cost_co2` computes cost and CO2 as **exactly linear
functions of mass** for rolled sections (verified: cost/mass and co2/mass
ratios are identical to 9 decimal places at different mass values, same
grade). This means, for rolled sections, minimising mass/cost/CO2 at a
fixed grade is mathematically the *same optimisation problem*, just
rescaled -- so the same GA geometry-convergence residual necessarily
produces the same %-gap in all three metrics simultaneously. (Welded
high-grade sections have a small thickness-dependent fabrication term
that breaks exact proportionality, but it's a second-order effect on top
of this.)

**State this uniformly in the paper's Methods section:** ground truth
(all three objectives) carries an estimated ~0.86-0.89% mean / ~3.86%
max residual optimality gap from the reference optimizer itself, on top
of whatever gap the RL/baseline arms show relative to it. This is an
order of magnitude smaller than any gap number Experiment 1 is likely to
report for actual RL policies.

**How much can the grade-selection finding (below) bear at the
per-context level?** The noise floor only matters for grade-selection
claims when the top-1 vs. top-2 grade margin at a given context is
smaller than the noise floor -- computed directly from the ground-truth
files via `grade_policy_analysis.py`'s `grade_selection_margins()`:

| economy_metric | median top-1 vs. top-2 margin | contexts with margin < 2% (within/near the noise floor) |
|---|---|---|
| co2 | 7.95% | 0% |
| cost | 3.37% | 32.4% |
| mass | 1.19% | 76.8% |

**Implication for how to report each finding:**
- **CO2 (S690 universal): report with full confidence, including at the
  per-context level.** Margins are never within the noise floor.
- **Cost (S355-dominant, S690 never optimal): report the aggregate/
  distributional pattern with confidence** (the headline is driven by
  large-margin contexts) but do not present any single context's
  "optimal grade" as certain -- about a third of individual labels are
  close calls.
- **Mass (genuine demand-spread across grades): report ONLY as a
  distributional/statistical claim** (e.g. the Spearman correlation
  across all 142 contexts, or a grade-vs-demand histogram), which is
  robust to individual-point noise given n=142. Do not claim any single
  context's mass-optimal grade is definitively X -- more than three-
  quarters of contexts have a top-1/top-2 margin smaller than the
  reference optimizer's own noise floor, so individual per-context grade
  labels for mass are frequently near-ties, not confident determinations.

**Key finding:**

| economy_metric | grades ever truly optimal | pattern |
|---|---|---|
| `cost` | S355 (72%), S460, S500, S550, S620 -- never S690 | overwhelmingly favours standard grade |
| `mass` | S460, S500, S550, S620, S690 (not S355) | genuinely demand-spread, spearman=0.47 |
| `co2` | **S690, 142/142 contexts (100%)** | universal, not demand-dependent at all |

Train and evaluate against all three `economy_metric` values as parallel
arms; report each arm's grade-vs-demand pattern against its *own*
metric's ground truth. This is a stronger, three-way falsifiable claim
than a single "the agent learned to prefer HSS" statement.

## 4. EC3 mechanics -- independently verified

`_ec3_analysis()`/`_calculate_cost_co2()` are cross-checked in two
independent ways: `tests/ec3_independent_verification.py` re-derives the
EC3 clauses from the specification directly (not copied from the
environment) and cross-checks 5 representative cases, matching to
<0.5%; `tests/test_ec3_golden_values.py` pins hardcoded reference values
for 5 representative geometries as a fast regression check against
future accidental changes. Key formulas implemented, with citation:

- Section classification: EN1993-1-1 Table 5.2 (internal compression
  parts / web; outstand flanges).
- Plastic/elastic moment resistance: EN1993-1-1 6.2.5.
- Shear resistance: EN1993-1-1 6.2.6.
- Elastic critical moment for LTB (doubly symmetric I-section,
  non-destabilising load): NCCI SN003, C1=1.13 for the simply-supported
  UDL case being modelled.
- LTB reduction factor: EN1993-1-1 6.3.2.2 ("General case"), imperfection
  factor from Table 6.3, buckling curve selection from **Table 6.5**:
  rolled h/b<=2 -> curve a (0.21); rolled h/b>2 -> curve b (0.34);
  welded h/b<=2 -> curve c (0.49); welded h/b>2 -> curve d (0.76).
- Deflection: elementary beam theory, delta = 5wL^4/(384EI).

**What is not independently verified**: no third-party structural
software (IDEA StatiCa, Robot, etc.) cross-check -- see Section 8.

## 5. Structurally infeasible demand contexts -- quantified

`tests/quantify_infeasible_contexts.py`: a fast, deterministic,
GA-independent scan (try every grade/type at *maximum-bound* geometry --
an exact necessary-condition check, not an approximation) finds **0.7%**
of a 30x30 (span, load) grid has no feasible design under any grade or
section type, tightly localised to span>=14.7m **and** load>=127.6kN/m
simultaneously (the extreme corner of the training envelope). These
contexts are correctly and automatically excluded from ground-truth-based
gap metrics (no row exists to compare against) -- this section exists so
that exclusion is stated, not silently inferred from a row count.

## 6. Termination vs. feasibility

`info["feasible"]`: util<=1.0 (tolerance 1e-3) AND section_class<=3 AND
geometry_penalty<=0 -- the only field that should ever be called
"feasible" in analysis or plots.

`info["in_target_band"]`: the 0.90-1.05 band, used *only* to decide early
termination during training (3 consecutive steps in-band). This is
purely a training-efficiency stopping rule, not a reporting mechanism --
every evaluation script uses `run_policy_episode(...,
return_best_feasible=True)` (default), which tracks the best **feasible**
(util<=1.0) design found at *any* point in the episode, not whichever
design the trajectory happened to end on. This means the training-
termination band's exact width does not bias reported results.

## 7. Catalog, cost, and CO2 assumptions

**Rolled-section catalog** (`envs/rolled_catalog.py`): procedurally
generated from typical UB/UC proportion rules, not a real manufacturer
table. It spans the same dimensional envelope as the continuous arm (so
the two are a fair comparison) and is used consistently across
everything that touches it. None of the planned comparisons (reward
formulation, algorithm choice, ground-truth gap, generalisation) depend
on the catalog's specific entries being real, purchasable sections --
only on the catalog and continuous arms exploring comparable space.
Swapping in a real manufacturer table (same CSV schema) would strengthen
a fabrication-realism claim specifically. State as a limitation; revisit
before any claim about real procurement.

**Cost/CO2 unit tables** (`_calculate_cost_co2`): assumed market-rate
figures, not independently sourced or cited (e.g. against EPD data for
structural steel). The qualitative finding in Section 3 (cost devalues
HSS, CO2 rewards it) is a *directional* result unlikely to flip under
plausible +/-15-20% price variation, though this has not been explicitly
stress-tested. A cheap post-hoc sensitivity check (perturb the unit
tables, re-run the grade-policy analysis) is recommended if reviewers
push on this.

## 8. Deferred, with justification

- **Categorical grade-action mechanism comparison** (continuous softmax-
  snap vs. true `MultiDiscrete` stepping): the catalog arm already uses a
  genuine categorical mechanism; the continuous arms use softmax-snap. A
  true hybrid-actor-head PPO policy (separate continuous+categorical
  heads in one network) would need a custom SB3 policy class. None of
  the core claims (reward formulation, algorithm choice, ground-truth
  gap, generalisation) depend on which grade-selection mechanism is
  used, only on it being used *consistently* within an arm.
- **Full RL hyperparameter sensitivity** (Lagrangian eta, PPO clip range,
  etc.): can only be meaningfully assessed from trained-agent behaviour,
  which is a post-hoc analysis of Experiment 1's own output, not
  something to resolve beforehand.
- **Independent third-party FE/software verification**: the from-spec
  re-derivation in Section 4, cross-validated further by GA reliably
  converging to util~1.0 (the constraint boundary) from multiple
  independent search methods, is the verification basis for this
  project. A commercial software cross-check would be additional
  confirmation, not a blocking requirement.
- **Cost/CO2 unit-price sensitivity sweep**: see Section 7. Cheap,
  recommended, not blocking.

## 9. Experiment matrix

| Arm | Script | Purpose |
|---|---|---|
| `shaped` | `train.py` | Reward-shaping baseline, no grade-specific term |
| `feasibility_gated` | `train.py` | Safe-RL baseline formulation |
| `lagrangian` | `train.py` | Primary proposed method |
| DDPG | `train_baseline_offpolicy.py --algo ddpg` | Algorithm-choice justification |
| TD3 | `train_baseline_offpolicy.py --algo td3` | Algorithm-choice justification |
| GA | `ga_baseline.py` via `evaluate.py --ga_baseline` | Classical-optimisation / amortised-cost comparison |
| Random search | `ga_baseline.py:random_search_design` | Equal-budget control for GA |
| Rule-based | `ga_baseline.py:rule_based_design` | Naive-practice baseline |
| Catalog (`lagrangian`) | `train.py --env_type catalog` | Fabrication-realism robustness check |

Each of {`shaped`, `feasibility_gated`, `lagrangian`} x {mass, cost, co2}
x 5 seeds = 45 PPO runs, + DDPG/TD3 x 5 seeds = 10, + catalog arm x 3
metrics x 5 seeds = 15 (or fewer if scoped down). GA/random/rule-based
are deterministic-enough per context to not need seed replication at
the same scale (GA already uses fixed seeds per evaluation context).

## Directory contents

```
research/
  envs/
    hss_env.py            Constrained-MDP environment, 3 reward modes
    hss_catalog_env.py     Discrete-catalog action-space variant
    rolled_catalog.py      Procedural rolled-section catalog generator
  algo/
    lagrangian.py           Dual-ascent Lagrange multiplier callback
  scripts/
    train.py                        PPO training (all reward modes, both env types)
    train_baseline_offpolicy.py     DDPG/TD3 training
    ga_baseline.py                  GA, random search, rule-based baselines
    regenerate_ground_truth.py      Objective-specific ground-truth generator
    evaluate.py                     Optimality-gap evaluation vs. ground truth
    generalization_test.py          Out-of-distribution stress test
    grade_policy_analysis.py        Grade-vs-demand analysis, ground truth only
    run_multiseed.py                Multi-seed orchestration + significance tests
  tests/
    ec3_independent_verification.py    From-spec EC3 re-derivation
    test_ec3_golden_values.py           Pinned-value regression test
    quantify_infeasible_contexts.py     Structural infeasibility scan
    validate_ground_truth.py            GA-vs-finer-GA ground truth cross-check
    validate_optimizer_independence.py  GA-vs-scipy-DE cross-check
  pretrain_data/
    ec3_optimal_designs_{mass,cost,co2}.csv   Objective-specific ground truth
```
