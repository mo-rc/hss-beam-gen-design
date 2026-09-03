# E1 — Resolving the Rolled-versus-Welded Costing Inconsistency

**Status:** complete. **Date:** 2026-09-02. **Repo commit at start:** `a11e01d`.

> **This document supersedes every cost and CO₂ number in the E0 repair-operator
> report and in the earlier PPO reward-comparison results.** Mass results are
> unaffected and are unchanged (verified byte-identical, §5). The headline change:
> the best repaired arm moves from **5.79%** to **8.53%** mean cost gap, and the
> best unrepaired arm from **25.51%** to **30.15%**. Nothing in the qualitative
> ranking of arms changes, but one previously-closed direction (the discrete
> catalog) has to be reopened (§6).

---

## 1. The defect

`HSSBeamEnv._calculate_cost_co2` picked the fabrication cost factor from the
`section_type` *label* alone:

| label | `fab_factor` | `fab_co2_factor` |
|---|---|---|
| `rolled` | 0.15 | 0.08 |
| `welded` | 0.42 (× grade multipliers, + thickness penalty ≥ S550) | 0.22 (× multipliers) |

`section_type` is a **free decision variable**: `ga_baseline.py` line 60 exposes
`SECTION_TYPES = ["rolled", "welded"]` as the sixth gene of the genome
`[h, b, tf, tw, fy_index, section_type_index]`, and the environment exposes it to
the policy as well. Nothing anywhere coupled that label to whether the geometry
could be hot-rolled.

The optimiser found the obvious exploit: claim the 2.8×-cheaper rolled
fabrication factor while using plate-girder proportions. In the stored cost
ground truth, **all 142 context optima are labelled `rolled`**, yet their median
web slenderness is **123.2 ε** and their median web thickness is **6.30 mm**.
No mill rolls that section. The "optimum" the RL arms were being scored against
was not a manufacturable beam, and the reported gaps were measured against a
fictitious reference.

This is a **costing/labelling defect, not a mechanics defect**. No EC3 capacity
equation, class check, or constraint definition was touched by the fix.

## 2. Grounding the manufacturability envelope

Two independent published constraints, deliberately used instead of invented
thresholds.

**(a) Geometry — what hot-rolled sections actually look like.** All 132 universal
beam sections in the [British Steel universal beams datasheet](https://www.britishsteel.co.uk/wp-content/uploads/2026/02/british-steel-universal-beams-datasheet-190724.pdf)
(h = 127–1026 mm) were evaluated with *this project's own* slenderness
definition (root radius r = 0.1·t_f, ε at S355):

| ratio | min | median | p99 | max |
|---|---|---|---|---|
| d/t_w (ε) | 19.11 | 49.95 | 72.23 | **73.32** |
| c/t_f (ε) | 3.25 | 6.63 | — | **10.27** |
| t_w/t_f | 0.53 | 0.61 | — | 0.84 |
| b/h | 0.30 | 0.41 | — | 0.66 |

**Zero of 132 real sections exceed the EC3 Class-3 web limit of 124 ε; only 2 of
132 exceed 72 ε.** A web at 123 ε is categorically not a rolled product.

**(b) Grade availability.** [EN 10025-6:2019](https://standards.iteh.ai/catalog/standards/cen/77ea2e36-0020-4c39-afef-337cb89046ca/en-10025-6-2019)
— the standard covering S460–S960 — is titled *"Technical delivery conditions
for **flat products** of high yield strength structural steels in the quenched
and tempered condition"* and scopes plate of 3–200 mm nominal thickness. It does
**not** cover hot-rolled sections. Hot-rolled I-sections are supplied in the
EN 10025-2/-3/-4 grades, i.e. up to S460 (S460 sections do exist, e.g. HISTAR
460). **S500 and above are therefore a plate product: a beam in those grades
must be a welded plate girder.**

## 3. The fix

New module `research/envs/manufacturability.py`. `section_type` remains a free
decision variable, but it no longer decides cost by itself: a design may only be
**costed** as rolled if it is inside the rolled envelope. Anything outside is
costed as the welded plate girder it physically is.

```python
ROLLED_LIMITS = {
    "web_slenderness_max":     75.0,   # d/tw in eps   (observed real max 73.3)
    "flange_slenderness_max":  11.0,   # c/tf in eps   (observed real max 10.3)
    "tw_over_tf_min":          0.50,   #               (observed real min 0.53)
    "b_over_h_max":            1.05,   # covers the near-square UC family
    "b_over_h_min":            0.25,   #               (observed real min 0.30)
    "fy_max":                  460.0,  # EN 10025-6 is a flat-product standard
}
```

Thresholds are the **observed real-section extremes rounded outward**, so the
envelope is permissive rather than tight: a design rejected as rolled is one no
real section comes close to. All limits live in one dict so sensitivity is
testable in one place (§7).

`_calculate_cost_co2` now computes `eff_type = effective_section_type(...)` and
uses it for the fabrication factor, the grade multipliers and the ≥S550
thickness penalty. The `debug` dict gained `effective_section_type` and
`reclassified` for auditability.

Guarded by a new constructor flag `enforce_rolled_manufacturability` (default
**True**). Setting it `False` reproduces the pre-correction behaviour exactly:

> **Regression test.** With the flag off, re-costing 400 randomly sampled stored
> ground-truth rows reproduces the stored `cost` to **max relative error
> 6.16 × 10⁻¹⁶** — i.e. bit-level identical modulo floating point. Every historical
> number in the project remains exactly reproducible.

### Scale of the mislabelling in the old ground truth

| | count | share |
|---|---|---|
| Stored GT rows labelled `rolled` | 837 | — |
| …of which **not** rolled-manufacturable | **827** | **98.8%** |
| The 142 cost-optimal designs, reclassified to welded | **139/142** | **97.9%** |

Violation frequency among rolled-labelled rows: web too slender 790, flange too
slender 606, grade plate-only 564, web too thin vs flange 538.

Re-costing those 142 optima **at unchanged geometry** inflates them by
**+19.3% mean / +18.9% median / +44.5% max**. That is an upper bound on the
damage, not the answer — the correct fix requires re-optimising.

## 4. Corrected ground truth

Regenerated with `research/scripts/regenerate_ground_truth.py` into
`research/pretrain_data_corrected/`: 1728 contexts × 2 restarts, pop 50, 80
generations, for **cost** and **CO₂**. **1623/1728 feasible for each — exactly
matching the original row count**, which is a useful consistency signal.
Wall time 21.6 min (cost) and 21.1 min (CO₂), run concurrently on 2 vCPUs.

| metric | reference inflation (mean / median / max) | optima costed as welded | median d/t_w | median t_w | cost-optimal grades |
|---|---|---|---|---|---|
| **cost** — old | — | 139/142 | 122.6 ε | 6.3 mm | 355:106, 460:30, 620:3, 500:2, 550:1 |
| **cost** — corrected | **+7.9% / +7.9% / +44.5%** | **4/142** | **72.9 ε** | **8.8 mm** | 355:116, 460:23, 620:2, 500:1 |
| **CO₂** — old | — | 142/142 | 123.2 ε | 7.0 mm | 690:142 |
| **CO₂** — corrected | **+20.9% / +20.5% / +26.7%** | 142/142 | 122.8 ε | 7.1 mm | 690:142 |

The two metrics respond in **opposite and economically sensible ways**, which is
the strongest evidence that the fix is doing the right thing:

- **Cost.** The GA re-optimises *into* the rolled envelope: only 4/142 optima
  now pay the welded factor, median web slenderness collapses 122.6 ε → 72.9 ε
  (sitting hard against the 75 ε limit, so the constraint is active), and webs
  thicken 6.3 → 8.8 mm. Because it can dodge the 0.42 factor by becoming
  stockier, the reference only rises **+7.9%**, far below the +19.3% naive
  re-costing bound.
- **CO₂.** Every optimum is S690, which is plate-only, so the welded factor is
  **unavoidable**. The GA cannot escape it, geometry barely moves
  (123.2 ε → 122.8 ε), and the reference absorbs the full **+20.9%**. Notably 23
  optima are now explicitly labelled `welded` — the label finally carries no
  economic advantage, so the optimiser stops gaming it.

**Corrected optimiser noise floor.** The independent GA control re-run scores
**1.72%** against the corrected GT (vs 1.31% against the old GT). The corrected
cost landscape is slightly harder to optimise, plausibly because the 75 ε
envelope boundary is now an active constraint adjacent to the optimum vertex.
**All corrected gaps should be read against a ~1.7% floor, not ~1.3%.**

## 5. Mass ground truth is unaffected

Mass is computed in `_ec3_analysis` from geometry and density alone and never
consults `_calculate_cost_co2`. `ec3_optimal_designs_mass.csv` was verified
**byte-identical** between the two directories. No mass result in the project
changes.

## 6. Re-evaluation of every arm

All arms re-scored against the corrected GT with corrected costing. Mean gap over
per-seed means, ±95% CI across 5 seeds (single seed → no CI), n = 142 contexts.

| arm | mode | old gap | **corrected gap** | Δ | feas |
|---|---|---|---|---|---|
| **GA control** | none | 1.31% | **1.72%** | +0.41 | 1.000 |
| | scale | 1.23% | **1.70%** | +0.47 | 1.000 |
| | scale+thin | 0.83% | **1.47%** | +0.65 | 1.000 |
| | scale+thin_rolled | — | **1.58%** | — | 1.000 |
| **gated_cost_merged** (1.3M, best) | none | 25.51% ±4.28 | **30.15% ±4.50** | +4.64 | 1.000 |
| | scale | 13.88% ±1.87 | **17.33% ±4.41** | +3.44 | 1.000 |
| | **scale+thin** | 5.79% ±0.58 | **8.53% ±1.47** | +2.75 | 1.000 |
| | scale+thin_rolled | — | **11.98% ±2.38** | — | 1.000 |
| **gated_cost** (linear, 1M) | none | 38.71% ±6.60 | **47.53% ±7.93** | +8.82 | 1.000 |
| | scale+thin | 6.31% ±0.33 | **12.32% ±0.89** | +6.01 | 1.000 |
| **lagrangian_cost** | none | 63.67% ±9.28 | **74.26% ±6.97** | +10.59 | 1.000 |
| | scale+thin | 9.21% ±1.01 | **15.14% ±2.47** | +5.92 | 1.000 |
| **shaped_cost** | none | 70.50% ±12.26 | **66.83% ±10.32** | **−3.67** | 1.000 |
| | scale+thin | 13.77% ±5.25 | **17.79% ±3.39** | +4.02 | 1.000 |
| **gated_cost_catalog** | none | 48.14% | **42.62%** | **−5.53** | 0.908 |
| | catalog_snap | 28.55% | **23.52%** | **−5.02** | 0.908 |
| | **catalog_snap+grade** | 25.82% | **19.26%** | **−6.56** | 0.908 |
| **rule_based** | none | 26.22% | **35.41%** | +9.19 | 0.944 |
| | scale+thin | 13.67% | **19.17%** | +5.50 | 0.944 |
| **random_search_40** (matched budget) | none | 73.15% | **78.56%** | +5.40 | 0.838 |
| | scale+thin | 24.53% | **33.22%** | +8.70 | 0.979 |
| **random_search_114** (matched to scale+thin) | scale+thin | 17.60% | **26.23%** | +8.63 | 0.972 |
| **random_search_4800** (120× budget) | none | 19.13% | **23.77%** | +4.63 | 0.979 |
| | scale+thin | 6.80% | **11.60%** | +4.79 | 0.993 |

Full per-arm/per-seed tables: `research/results/corrgt_{gated_merged,otherarms,baselines,catalog}_{summary,per_context}.csv`.

### 6.1 My prior expectation was wrong, and why

I expected gaps to **shrink**, because the correction reclassifies 95.8% of GA
designs but only 87.7% of PPO's thicker designs. That reasoning ignored
re-optimisation. The GA *moves* — it finds stocky manufacturable sections only
+7.9% more expensive. PPO's checkpoints cannot move; they emit the geometry they
were trained to emit, most of it still outside the envelope, and pay the full
0.42 factor. **The reference improved relative to the policies, so almost every
gap widened.** Recorded explicitly because it was a falsified prediction.

### 6.2 Part of E0's headline was exploiting the bug

The `scale+thin` operator thins plates toward EC3 class limits. Under the old
costing, thinning to a 124 ε Class-3 web was free; under corrected costing it
converts the section into a welded plate girder. The operator's internal variant
choice shifts accordingly:

| winning variant | old | corrected |
|---|---|---|
| `thin_c3` (slender web) | 2551 | 184 |
| `thin_c2` | 884 | 132 |
| `thin_c1` (stocky) | 106 | **289** |
| `scale` | 232 | 105 |
| `original` | 61 | 0 |

Repair now prefers **Class 1 stocky** sections. `within_10pct` for the best arm
falls 90.8% → 76.8%, and mean section class rises 1.39 → 1.64. The operator is
still strongly beneficial (30.15% → 8.53%, a 21.6 pp improvement) and still
100% feasible — but roughly 2.7 pp of E0's reported benefit was fabrication-cost
arbitrage rather than engineering.

### 6.3 `scale+thin_rolled` — a clean negative result

I added a variant that caps thinning at the rolled envelope instead of the EC3
class limits. It is **uniformly worse** than unconstrained `scale+thin`
(11.98% vs 8.53% on the best arm; 1.58% vs 1.47% on GA). The reason is
instructive: once fabrication cost is computed correctly, welded plate girders
genuinely *are* the cheaper choice in some contexts, and the corrected cost model
already makes that trade-off endogenously. Hard-constraining to rolled removes a
legitimate option. **Recommendation: keep the corrected cost model, do not add an
explicit manufacturability constraint.** The variant is retained in `repair.py` as
the documented control.

### 6.4 The discrete catalog direction must be reopened

The catalog arm is the **only** arm that improves, and it improves the most
(25.82% → 19.26%, −6.56 pp), because its designs are drawn from a real section
table and are genuinely manufacturable — only 21.1% of them get reclassified,
versus 95.8% of GA designs. E0 §6a closed this direction as a negative result on
the grounds that 96.5% of GT optima lay outside the catalog's slenderness
envelope. **That argument was itself an artefact of the bug**: those optima were
non-manufacturable fictions. Against the corrected GT the median optimum sits at
72.9 ε, and the catalog spans 27.0–59.9 ε — still short, but a gap of a factor
1.2 rather than 2.1. The catalog remains *behind* the continuous arm
(19.26% vs 8.53%) so it is not the recommended path, but "closed as a negative
result" is no longer a defensible statement and E0 §6a is retracted on this point.

`shaped_cost` also improves pre-repair (−3.67 pp), consistent with the earlier
finding that its `+5.0` Class-3 and `+10.0` web-slenderness reward terms — which
I had characterised as penalising exactly the optimum's features — were in fact
pushing the policy toward manufacturable geometry. It was accidentally right for
a reason the reward author did not state.

### 6.5 Caveat on interpretation — these are transfer numbers

Every arm here is a checkpoint **trained under the old costing** and rolled out
in the corrected environment. Because economy enters the observation and reward,
the corrected environment changes the rollout trajectory itself (the best arm's
unrepaired mean section class shifts 1.93 → 1.56 and its reclassification rate
falls 87.7% → 33.1% relative to the stored E0 rollouts). **These are off-training
transfer results, not retrained results.** Retraining under corrected costing is
a separate experiment and should be expected to recover some of the widened gap.
No claim here should be read as "PPO's ceiling under correct costing".

## 7. Threshold sensitivity

Because the GA cannot be re-run per threshold setting (21 min each), sensitivity
uses a proxy: for each setting, the reference is the cheapest feasible design per
context from the **union of both GA design pools** (3246 designs, spanning both
the slender and stocky regimes), and the best arm's `scale+thin` designs are
re-costed against it. The proxy reproduces the measured baseline value
**8.54% vs 8.53% measured**, which validates it near the baseline setting.

| setting | best-arm gap (proxy) |
|---|---|
| tighter web, d/t_w ≤ 65 ε | 11.64% |
| **baseline: d/t_w ≤ 75 ε, c/t_f ≤ 11 ε, f_y ≤ 460** | **8.54%** |
| looser web, d/t_w ≤ 90 ε | 8.10% |
| no web limit (124 ε) | 8.04% |
| tighter flange, c/t_f ≤ 9 ε | 5.45% |
| looser flange, c/t_f ≤ 14 ε | 8.56% |
| no grade limit (f_y ≤ 690) | 7.67% |
| grade limit only, no geometry limits | 12.95% |

**The headline sits in 5.5–13.0% across every variant tested**, so the conclusion
"the best repaired arm is in the high single digits to low teens against a
manufacturable reference" does not depend on the threshold choice. The web limit
matters most in the tightening direction (65 ε costs +3.1 pp); loosening it
beyond 75 ε barely matters, confirming 75 ε is close to where the envelope stops
binding.

*Proxy limitation:* the "no correction at all" setting returns 10.99% by proxy
versus 5.79% measured. The proxy is only valid near the baseline, because the
repair operator's variant choices were made under baseline costing and are not
re-optimised per setting. Cite the measured 5.79% for that case, not the proxy.

## 8. What is now superseded

| claim | old | corrected |
|---|---|---|
| Best PPO arm, unrepaired | 25.51% | **30.15%** |
| Best PPO arm + `scale+thin` | 5.79% | **8.53%** |
| Catalog exact ceiling | 25.82% | **19.26%** |
| GA / optimiser noise floor | 1.31% | **1.72%** |
| Equal-budget random search (4800) | 19.13% | **23.77%** |
| All CO₂ reference values | — | **+20.9%** |
| All mass results | — | **unchanged** |
| E0 §6a "catalog direction closed" | asserted | **retracted (§6.4)** |

Unchanged conclusions: `feasibility_gated` still beats `shaped` and `lagrangian`;
the merged reward + log_std annealing still beats the linear variant; the repair
operator still produces a large, statistically robust, 100%-feasible improvement;
PPO still vastly beats matched-budget random search (30.15% vs 78.56%) while
still losing to the 120×-budget control before repair and beating it after
(8.53% vs 11.60%).

## 9. Reproduction

```bash
# corrected ground truth (cost and CO2; mass is unaffected)
python research/scripts/regenerate_ground_truth.py \
    --out_dir research/pretrain_data_corrected --metrics cost
python research/scripts/regenerate_ground_truth.py \
    --out_dir research/pretrain_data_corrected --metrics co2
cp research/pretrain_data/ec3_optimal_designs_mass.csv research/pretrain_data_corrected/

# re-evaluation (continuous arms)
python research/scripts/evaluate_with_repair.py \
    --ground_truth_dir research/pretrain_data_corrected \
    --modes none scale scale+thin scale+thin_rolled \
    --models research/models/gated_cost_merged_seed4{2,3,4,5,6}/final_model \
    --include_ga --out_prefix research/results/corrgt_gated_merged

# re-evaluation (discrete catalog arm)
python research/scripts/evaluate_with_repair.py \
    --ground_truth_dir research/pretrain_data_corrected --env_type catalog \
    --modes none catalog_snap catalog_snap+grade \
    --models research/models/gated_cost_catalog_seed42/final_model \
    --out_prefix research/results/corrgt_catalog

# reproduce ALL pre-correction numbers
HSSBeamEnv(..., enforce_rolled_manufacturability=False)
```

## 10. Open items

1. **Retrain the best arm under corrected costing** (single seed, 1.3M steps) to
   separate "the reference got harder" from "the policy is mis-specified". This is
   the highest-information next run and the only way to state a defensible
   post-correction PPO number.
2. **Unresolved discrepancy, flagged not reconciled:** GA wall time measured here
   is 0.451 s/context, but the stored `research/results/ga_cost_summary.json`
   records `wall_time_s_per_context = 1.213`. A 2.7× disagreement in the
   baseline's cost-per-solution directly affects every equal-budget claim in the
   paper and needs resolving before publication.
3. **Grade limit is conservative.** `fy_max = 460` admits S460 sections
   (HISTAR 460 etc.). If the intended product scope is UK-market UB/UC only,
   S460 sections are not commonly stocked and 355 would be the honest limit;
   §7 shows that direction is untested and should be if the paper claims a
   specific supply market.
4. **The catalog itself is procedurally generated** (`rolled_catalog.py`, 62
   members) rather than taken from a published table. Given §6.4 reopens this
   direction, replacing it with the 132 real British Steel UB sections
   (`real_ub.csv`, already extracted) is now a cheap and worthwhile change.

## 11. Files changed

| path | change |
|---|---|
| `research/envs/manufacturability.py` | **new** — `ROLLED_LIMITS`, `rolled_violations`, `is_rolled_manufacturable`, `effective_section_type` |
| `research/envs/hss_env.py` | `enforce_rolled_manufacturability` flag (default True); `_calculate_cost_co2` uses `eff_type`; `debug` gains `effective_section_type` / `reclassified` |
| `research/algo/repair.py` | `_thin_to_limits()`; `scale+thin_rolled` mode |
| `research/pretrain_data_corrected/` | corrected cost + CO₂ GT (mass copied unchanged) |
| `research/results/corrgt_*` | re-evaluation outputs |

The sandbox has no push credentials for `github.com/mo-rc/hss-beam-gen-design`;
these changes must be committed locally.
