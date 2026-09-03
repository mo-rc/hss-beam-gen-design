# E0 — Post-hoc Repair Operator: Implementation and Results

**Date:** 2026-09-02
**Repo:** `github.com/mo-rc/hss-beam-gen-design` @ `a11e01d`, `research/` module
**Metric:** cost, all 142 usable ground-truth contexts, 5 seeds per PPO arm
**New code:** `research/algo/repair.py`, `research/scripts/evaluate_with_repair.py`
**New results:** `research/results/e0_repair_{gated_merged,otherarms,baselines,catalog}_{summary,per_context}.csv`

---

## 1. What was implemented

Two deterministic, training-free operators applied to a design *after* the policy has produced it.

**`scale` — this is E0 as specified.** Multiply `(h, b, tf, tw)` by a single scalar `s` and bisect on `s` until utilisation reaches 1.0 from the feasible side. Two structural properties make it safe rather than merely convenient:

- Uniform scaling leaves `c/tf` and `d/tw` **exactly invariant**, so it cannot change the section class and cannot push a Class-3 section over the Class-4 cliff.
- Bisection only ever accepts the feasible bracket end, so the output is feasible by construction.

It also repairs *infeasible* inputs by growing `s` instead of shrinking it, which is why it improves feasibility rates on the weaker arms.

**`scale+thin` — reported separately, and it is NOT "repair".** First push the plate slenderness ratios out toward an EC3 Table 5.2 class limit (thinning `tf` and `tw` — the direction a stochastic policy structurally refuses to go), then re-apply uniform scaling to return to utilisation 1.0. Class targets 3, 2 and 1 are all tried, because Class 3 uses `Wel` while Classes 1–2 use `Wpl`, so the cheapest admissible class is context-dependent. This is a **local improvement operator**, and I have kept it in a separate column so the E0 claim is not inflated by it.

**Non-degradation guarantee.** Both operators return the original design unless the repaired one is feasible *and* strictly cheaper. Audited over all 3,834 (arm × context) pairs: **zero regressions on a feasible baseline, zero feasible→infeasible regressions.** The 64 / 36 cases where cost rose all had an *infeasible* baseline being grown into compliance, where a cost increase is the correct outcome.

Every operator call reports `n_ec3`, the number of EC3 analyses it consumed, so its cost is visible next to its benefit.

---

## 2. Falsification test: repair applied to GA

Before trusting any PPO improvement: GA already converges to utilisation 0.9995, so a *correct* operator must do almost nothing to GA designs. If repair "improved" GA substantially, it would be exploiting a modelling inconsistency rather than removing genuine policy slack.

| GA | feasibility | mean gap | median | p90 | worst | EC3/design | s/design |
|---|---|---|---|---|---|---|---|
| none | 1.000 | **1.313%** | 0.808% | 3.89% | 7.19% | 1 | 0.451 |
| scale | 1.000 | **1.233%** | 0.683% | 3.79% | 6.72% | 18 | 0.452 |
| scale+thin | 1.000 | **0.826%** | 0.338% | 3.09% | 5.81% | 55 | 0.453 |

The operator moves GA by 0.08 pp (`scale`) and 0.49 pp (`scale+thin`) — both inside the documented ~0.9% GA optimiser noise floor. **Test passed.** The large PPO gains below are therefore removal of real policy slack, not an artefact.

---

## 3. Main result — PPO arms

Mean cost gap over 142 contexts, averaged across 5 seeds, ±95% CI over seeds:

| Arm | none | `scale` (E0) | `scale+thin` |
|---|---|---|---|
| `gated_cost_merged` (log_relative + log_std anneal, 1.3M) | 25.51% ± 6.06 | **13.88% ± 2.65** | **5.79% ± 0.82** |
| `gated_cost` (linear economy, 1M) | 38.71% ± 9.35 | 16.20% ± 2.28 | 6.31% ± 0.47 |
| `lagrangian_cost` (1M) | 63.67% ± 13.14 | 28.28% ± 4.06 | 9.21% ± 1.43 |
| `shaped_cost` (1M) | 70.50% ± 17.36 | 45.61% ± 15.81 | 13.77% ± 7.44 |
| `gated_cost_catalog` (1 seed) | 48.15% | 26.87% | 9.08% |

The `none` column reproduces the previously reported numbers (25.5% for the best arm, 38.7% for gated-linear), confirming the harness is consistent with the existing results tables.

Paired t-tests across the 5 seeds — every improvement is significant:

| Arm | none → scale | none → scale+thin | scale → scale+thin |
|---|---|---|---|
| `gated_cost_merged` | t = 7.41, p = 1.8e-3 | t = 9.36, p = 7.3e-4 | t = 11.17, p = 3.7e-4 |
| `gated_cost` | t = 7.70, p = 1.5e-3 | t = 9.78, p = 6.1e-4 | t = 13.98, p = 1.5e-4 |
| `lagrangian_cost` | t = 9.35, p = 7.3e-4 | t = 11.39, p = 3.4e-4 | t = 14.65, p = 1.3e-4 |
| `shaped_cost` | t = 9.43, p = 7.1e-4 | t = 10.60, p = 4.5e-4 | t = 8.47, p = 1.1e-3 |

Full statistics for the best arm (5 seeds, cost, n = 142 each):

| | none | `scale` | `scale+thin` |
|---|---|---|---|
| feasibility | 1.000 | 1.000 | 1.000 |
| mean gap | 25.51% | 13.88% | 5.79% |
| median gap | 24.0% | 13.1% | 5.0% |
| p90 | 34.1% | 20.4% | 10.2% |
| p95 | 40.1% | 24.1% | 12.8% |
| worst | 99.4% | 54.5% | 35.1% |
| within 5% | 0.4% | 6.3% | **50.0%** |
| within 10% | 2.1% | 29.3% | **90.8%** |
| mean utilisation | 0.847 | 1.001 | 1.000 |
| mean section class | 1.93 | 1.90 | 2.32 |
| EC3 analyses / design | 1 | 18 | 73 |
| seconds / design | 0.021 | 0.021 | 0.023 |

Mean utilisation moves from 0.847 to 1.001 — the operator does exactly what the diagnosis predicted, and mean section class rises from 1.93 to 2.32 as `scale+thin` pushes plates toward the Class-3 limit where 87% of the true optima live.

---

## 4. Equal-budget baselines get the same operator

This is the comparison that decides whether the learned policy contributes anything. A PPO rollout costs 40 EC3 analyses; `+scale` ≈ 58 total, `+scale+thin` ≈ 114 total. Random search was run at each of those budgets, plus at the GA-matched 4800, and given the identical operators.

| Method | EC3 evals | feasibility | mean gap (none) | mean gap (`scale`) | mean gap (`scale+thin`) | s/design |
|---|---|---|---|---|---|---|
| Random search | 40 | 0.838 → 0.979 | 73.15% | 52.36% | 24.53% | 0.005 |
| Random search | 58 | 0.852 → 0.979 | 65.00% | 45.13% | 22.24% | 0.006 |
| Random search | 114 | 0.887 → 0.972 | 54.02% | 36.03% | **17.60%** | 0.009 |
| Random search | 4800 | 0.979 → 0.993 | **19.13%** | 14.13% | 6.80% | 0.257 |
| Rule-based EC3 | ~10 | 0.944 | 26.22% | 18.45% | 13.67% | 0.003 |
| **PPO best arm** | 40 / 58 / 114 | **1.000** | 25.51% | 13.88% | **5.79%** | 0.023 |
| GA | 4800 | 1.000 | 1.31% | 1.23% | 0.83% | 0.451 |

**A correction to the project record.** The historical claim that PPO is "not clearly better than the equal-budget random-search baseline (~19%)" does not survive checking. `research/README.md` line 251 documents `random_search_design` as an *"equal-budget control for GA"*, and its default is `n_evaluations=4800`. The ~19% figure is therefore random search with **4800** EC3 analyses, compared against a PPO rollout using **40** — a 120× budget advantage for the baseline. At genuinely matched budget, random search scores 73.15% (40 evals) / 65.00% (58) / 54.02% (114) against PPO's 25.51%. PPO was always substantially better than equal-budget random search; the comparison was mislabelled.

With repair applied to both, PPO at 114 evaluations (5.79%) beats random search at the same 114 evaluations (17.60%) by 11.8 pp, and beats random search at 4800 evaluations (6.80%) while using **42× fewer** EC3 analyses and **11× less** wall-clock time. The learned policy is now doing demonstrable work.

---

## 5. The uncomfortable finding: repair erases most of the reward engineering

The `log_relative` economy reward plus `log_std` annealing plus extended training moved the gated arm from 38.71% to 25.51% — a 13.20 pp gain, significant at p = 0.011. After repair, that same comparison:

| Repair mode | merged (log_relative + anneal, 1.3M) | linear, 1M | difference | p |
|---|---|---|---|---|
| none | 25.51% | 38.71% | **+13.20 pp** | **0.011** |
| `scale` | 13.88% | 16.20% | +2.32 pp | 0.102 |
| `scale+thin` | 5.79% | 6.31% | +0.52 pp | 0.164 |

**The advantage collapses to 0.52 pp and becomes statistically indistinguishable.** Nearly the entire measured benefit of that reward-engineering effort was moving designs closer to the active constraint boundary — which an 18-evaluation deterministic bisection does for free. The same pattern holds across arms: `scale+thin` compresses four reward formulations spanning 25.5%–70.5% into a band of 5.8%–13.8%, and the three non-`shaped` arms into 5.8%–9.2%.

This strongly reinforces the §3 diagnosis in the bottleneck report: these reward variants were all fighting the same structural problem — a stochastic policy retreating from a cliff-adjacent vertex optimum — and none of them addressed its cause. **Further reward shaping in the current action space is not worth compute.**

A related code-level finding, for the record: `_reward_shaped` (lines ~603–607 of `hss_env.py`) adds `+5.0` penalty when `section_class == 3` and `+10.0` when `hw/tw > 72ε`. Both penalise exactly the features that characterise the true optimum — 87% of optima are Class 3, and the median optimum has `d/tw = 122.6ε`. This is consistent with the `shaped` arm's pre-repair mean section class of 1.02, and it explains why that arm is the worst of the four. The best arm (`_reward_feasibility_gated`) does **not** contain these penalties — its `g3_geom` is only the benign `b/h > 1` term — so this is an explanation for `shaped`'s failure specifically, not for the 25% plateau. Separately, the gated arm's potential-based shaping targets `target_util = 0.96` rather than 1.00; potential-based shaping is theoretically policy-invariant, so this should not bias the optimum, but it is worth removing given that the true optimum is at exactly 1.000.

---

## 6. Limitations and caveats

- **`scale+thin` is not a feasibility repair.** It changes the design's slenderness and section class to improve cost. Only the `scale` column supports a claim of the form "we project the policy output onto the active constraint boundary". Present them as two separate operators; do not report 5.79% as "PPO with a repair operator" without saying which one.
- **The catalog arm's repair breaks catalog membership.** Uniform scaling produces continuous geometry, so the repaired catalog designs are no longer rolled-catalog members. The 48.15% → 9.08% improvement there is therefore **not valid** as a catalog result. For that arm the operator must instead snap to the next-larger admissible catalog member. Fix before reporting.
- **`scale+thin` costs ~73 EC3 analyses per design** versus 1 for the raw policy, raising total inference cost from 40 to ~114 analyses. Reported throughout, and the matched-budget comparison in §4 accounts for it. Wall-clock impact is small (0.021 → 0.023 s/design) because an EC3 analysis is cheap.
- **The bisection is crude.** 16 iterations on a scalar whose capacity scales roughly as `s³`; 3–4 Newton steps would achieve the same. Do this before publishing inference-cost figures.
- **Eight contexts remain infeasible** for the weaker baselines after repair, all at span ≥ 13.4 m with load ≥ 96 kN/m — consistent with the documented 0.7% structurally infeasible envelope plus its immediate neighbourhood. The PPO arms reach 100% feasibility on all 142.
- **Repair does not close the gap to GA** (5.79% vs 0.83% repaired). It converts an uncompetitive result into an arguable speed/quality trade-off — ~20× faster per design than GA for ~5 pp more cost — but it does not by itself make the RL approach publication-winning. The reformulation in E1/E2 is still the path to a headline result.
- Utilisation slightly exceeds 1.0 (1.019–1.021) for the random-search arms because the mean includes the few contexts that remain infeasible; feasibility is reported separately so this is not hidden.

---

## 7. What this changes about the plan

1. **Adopt `scale` everywhere immediately.** It is free, provably non-degrading, improves every arm, lifts the catalog arm's feasibility from 0.908 to 1.000, and satisfies reviewer comment 6's request for a feasibility projection operator. There is no argument for reporting un-repaired numbers again.
2. **Report `scale+thin` as a second, clearly-labelled operator**, and use it to make the honest argument that most of the prior reward-engineering gains were redundant. That is a stronger and more interesting paper section than presenting those gains at face value.
3. **Correct the random-search comparison in the paper and in the project notes.** Report random search at 40, 114, and 4800 evaluations so the budget axis is explicit.
4. **Stop reward engineering in the current action space.** §5 is direct evidence that it is not where the remaining gap lives.
5. **E1 (feasible-by-construction one-shot reformulation) remains the priority.** Note that `scale+thin`'s winning variant was `thin_c3` in 2,551 of 3,834 cases and `thin_c2` in 884 — i.e. the operator is effectively *searching over the class target*, which is exactly the `(λ_f, λ_w)` action dimension E1 proposes handing to the policy. E0's success is corroborating evidence for E1's design, not a substitute for it.
6. **Fix the catalog repair** to snap to catalog members before that arm is reported.

---

## 8. Reproduction

```bash
# PPO best arm + GA falsification control
python research/scripts/evaluate_with_repair.py \
  --models research/models/gated_cost_merged_seed4{2,3,4,5,6}/final_model \
  --include_ga --out_prefix research/results/e0_repair_gated_merged

# equal-budget baselines
python research/scripts/evaluate_with_repair.py --models \
  --random_evals 40 58 114 4800 --include_rule_based \
  --out_prefix research/results/e0_repair_baselines

# remaining arms
python research/scripts/evaluate_with_repair.py \
  --models research/models/{shaped_cost,lagrangian_cost,gated_cost}_seed4{2,3,4,5,6}/final_model \
  --out_prefix research/results/e0_repair_otherarms

# catalog arm (see §6 caveat)
python research/scripts/evaluate_with_repair.py \
  --models research/models/gated_cost_catalog_seed42/final_model --env_type catalog \
  --out_prefix research/results/e0_repair_catalog
```

Environment note: `pip install "stable-baselines3[extra]"` fails (pygame/SDL build error); use `stable-baselines3==2.3.2`. Model loading emits a harmless `FloatSchedule` deserialisation warning.
