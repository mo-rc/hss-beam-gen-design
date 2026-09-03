# E2 — Resolving the GA Timing Discrepancy, and Restating the Equal-Budget Claims

**Status:** complete. **Date:** 2026-09-02.
**Closes:** E1 §10 open item 2.

> **Verdict on the timing discrepancy: there is no bug and nothing to reconcile
> in the code.** Both numbers are correct measurements of an identical
> computation on different hardware. The discrepancy was never the real problem.
> Chasing it did, however, expose a **genuine instrumentation bug in my own
> evaluation harness** that invalidated every equal-budget comparison in the
> project — including the "PPO is not clearly better than random search" claim
> that motivated this whole line of work. **That claim was wrong.** Corrected, PPO
> plus repair beats the random-search control by 15.2 pp while using **43× fewer**
> structural evaluations.

---

## 1. The discrepancy

| source | value |
|---|---|
| `research/results/ga_cost_summary.json` → `wall_time_s_per_context` | 1.213 s |
| Measured in this sandbox (E0/E1 runs) | 0.451–0.471 s |
| Ratio | **2.57×** |

## 2. Resolution: same computation, different host

Four pieces of evidence, in order of decisiveness.

**(a) The time was genuinely spent inside the GA, not in harness overhead.**
`ga_cost.csv` stores a per-context `wall_time_s` measured *inside* `ga_design`
itself (`ga_baseline.py` sets `t0` before the generation loop and returns
`wall_time_s`). Summing that column gives **171.94 s** against the harness-level
`wall_time_s_total` of **172.25 s** — the harness accounts for only **0.32 s
(0.2%)**. So the 1.213 s is 99.8% GA internals. No overhead explanation survives.

**(b) The GA settings are identical in both scripts.** `evaluate.py` defaults
`--ga_pop_size 60 --ga_n_generations 80` (lines 336–337); `evaluate_with_repair.py`
defaults `--ga_pop 60 --ga_gen 80` (lines 197–198). Both call the same
`ga_design`. Confirmed by the returned `n_evaluations = 4800 = 60 × 80` in both.
There is no hidden settings difference.

**(c) The measurement in this sandbox is extremely stable, and reproduces
exactly.** Twelve serial `ga_design` runs at pop 60 / gen 80:
**0.4711 s ± 0.0080 (CV 1.7%)**. The re-run harness independently reports
**0.4674 s ± 0.0084**. These agree to within 0.8%.

**(d) The original host was slow *and* noisy; this host is fast and quiet.**
`ga_design` performs a **fixed** amount of work — exactly `pop_size ×
n_generations` fitness evaluations, with no early exit and no caching (elites are
re-evaluated each generation). Runtime variation across contexts is therefore
pure measurement noise, and it is diagnostic:

| host | mean | sd | **CV** |
|---|---|---|---|
| original (stored `ga_cost.csv`, n=142) | 1.2108 s | 0.2889 s | **23.9%** |
| this sandbox (n=12) | 0.4711 s | 0.0080 s | **1.7%** |

A 24% coefficient of variation on a fixed-work computation is the signature of a
**contended, shared host** (the max, 2.20 s, is 2.4× the min, 0.91 s, for
byte-identical work). The 2.57× mean ratio is consistent with a host that is both
slower per-operation and heavily time-shared.

**Conclusion.** The stored 1.213 s is a valid measurement on the original
authors' machine; 0.471 s is a valid measurement here. Neither is wrong. They are
not comparable, and **neither should appear in the paper as a cost figure.**

### 2.1 Why wall-clock cannot be the budget currency at all

Beyond non-portability, wall-clock actively misattributes cost:

- Of the GA's 0.4674 s/context, only **0.177 s (37.9%)** is EC3 evaluation
  (4800 evals × 0.0369 ms/eval, measured over 18,000 warmed evaluations). The
  other **62%** is the GA's own Python selection / tournament / BLX-α crossover /
  mutation bookkeeping.
- Decisive demonstration: `random_search_4800` performs **exactly the same 4800
  EC3 evaluations** and completes in **0.2637 s** — the GA is **1.77× slower for
  an identical evaluation count**.

Reporting wall-clock would therefore penalise the GA for being written in pure
Python rather than for anything algorithmic, and would flatter any method whose
per-evaluation overhead happens to be lower. **EC3 evaluation count is the correct
currency**: it is hardware-independent, implementation-independent, and exactly
deterministic for every arm here.

## 3. The real bug this uncovered

While establishing what the budget currency should be, I checked whether the
harness was actually measuring it. It was not.

`evaluate_with_repair.py` recorded `n_ec3 = res["n_ec3"]`, taken from
`repair()`. That counter covers **only the repair operator's** analyses — the
baseline check plus its bisection calls. The cost of *generating* the design was
never counted. Consequences:

| arm, mode `none` | previously reported | actual |
|---|---|---|
| PPO rollout | 1.0 | **40.4** (39 policy steps + 1) |
| GA | 1.0 | **4801** |
| `random_search_4800` | 1.0 | **4801** |
| `rule_based` | 1.0 | **1.0** (closed-form; correct by luck) |

Every arm was reported as costing one structural analysis to produce a design.
The `ec3_evals_per_design` column — the only budget number in the results — was
meaningless for all but one arm, and the GA's 4800-fold cost advantage was
invisible.

A second, smaller defect: `sec_per_design` was computed as
`t_gen / len(opt) + mean(t_repair)`, i.e. the *shared* total generation time
divided by n, plus the arm-mean repair time. That yields one constant per arm and
hides all per-context variation. It is also why the E1 run showed `sec_per_design`
varying 21× across five identical PPO seeds (0.4535 / 0.1308 / 0.0210 / 0.0212 /
0.0210): I had run two evaluation processes concurrently on 2 vCPUs, and the
early arms absorbed the contention. **Those E1 timing figures were measuring CPU
contention, not algorithm cost.** The E1 gap figures are unaffected.

### Fixes applied

- `rollout_design` now returns `(design, was_feasible, n_ec3)`, counting one
  analysis per `env.step` (HSSBeamEnv.step calls `_ec3_analysis` exactly once).
- `ga_design` and `random_search_design` already returned `n_evaluations`; the
  harness now propagates it instead of discarding it. `rule_based_design` is
  wired as 0 search evaluations.
- Per-context rows now carry `n_ec3_gen`, `n_ec3_repair` and `n_ec3`
  (= gen + repair); summaries report all three.
- `sec_per_design` is now each design's **own** generation time plus its **own**
  repair time, and summaries report its mean **and standard deviation**.
- The `--random_evals` help text, which asserted the wrong matched budget
  (§4.1), was rewritten.

**The accounting fix changed no result.** All gap statistics are bit-identical to
the E1 run (GA `none` 1.72%, PPO `none` 30.15%, etc.), confirming this is pure
instrumentation. All runs below were executed **serially** on an otherwise idle
sandbox.

## 4. Corrected equal-budget results

Corrected ground truth (E1), cost metric, n = 142 contexts, ±95% CI over 5 seeds.
`gen` / `repair` / `TOTAL` are mean EC3 analyses per design.

| arm | mode | gap | gen | repair | **TOTAL** | s/design |
|---|---|---|---|---|---|---|
| **GA** (reference) | none | 1.72% | 4800 | 1.0 | **4801** | 0.4674 ±0.0084 |
| | scale+thin | 1.47% | 4800 | 67.8 | 4868 | 0.4697 ±0.0085 |
| **gated_cost_merged** | none | 30.15% ±4.50 | 39 | 1.0 | **40** | 0.0207 ±0.0030 |
| | scale | 17.33% ±4.41 | 39 | 18.0 | **57** | 0.0213 ±0.0030 |
| | scale+thin_rolled | 11.98% ±2.38 | 39 | 37.3 | **77** | 0.0220 ±0.0030 |
| | **scale+thin** | **8.53% ±1.47** | 39 | 72.2 | **112** | 0.0232 ±0.0030 |
| **rule_based** | none | 35.41% | 0 | 1.0 | **1** | 0.0006 ±0.0002 |
| | scale+thin | 19.17% | 0 | 73.3 | **73** | 0.0031 ±0.0006 |
| **random_search_40** | none | 78.56% | 40 | 1.0 | **41** | 0.0024 ±0.0000 |
| | scale | 61.77% | 40 | 18.1 | **58** | 0.0030 ±0.0001 |
| | scale+thin_rolled | 43.93% | 40 | 36.7 | **77** | 0.0036 ±0.0002 |
| | scale+thin | 33.22% | 40 | 73.2 | **113** | 0.0049 ±0.0004 |
| **random_search_4800** | none | 23.77% | 4800 | 1.0 | **4801** | 0.2637 ±0.0065 |
| | scale+thin | 11.60% | 4800 | 71.0 | **4871** | 0.2661 ±0.0065 |

Over-budget random controls retained for continuity: `random_search_58`
(59 evals) 72.29% / (131) 32.35% `scale+thin`; `random_search_114` (115) 59.97% /
(186) 26.23% `scale+thin`.

Full tables: `research/results/budget_{gated_merged,baselines}_{summary,per_context}.csv`.

### 4.1 The matched control was the wrong one

The `--random_evals 40 58 114` design intent was that `random_search_58` matches
"PPO + scale" (58 total) and `random_search_114` matches "PPO + scale+thin" (114
total). **This is a double-count.** Those totals already include PPO's repair
cost; when the same repair operator is then applied to the random arm, the random
arm pays its repair cost a second time:

- `random_search_58` + `scale` = 58 + 18 = **76** evals, vs PPO's 57.
- `random_search_114` + `scale+thin` = 114 + 71.5 = **186** evals, vs PPO's 112.

The correctly matched control is **`random_search_40`** — matching PPO's
*generation* budget — evaluated under the *same* repair mode, so both arms pay the
same generation cost and the same repair cost:

| matched comparison | total evals | random | **PPO** | PPO advantage |
|---|---|---|---|---|
| `none` | 41 vs 40 | 78.56% | **30.15%** | **48.4 pp** |
| `scale` | 58 vs 57 | 61.77% | **17.33%** | **44.4 pp** |
| `scale+thin_rolled` | 77 vs 77 (exact) | 43.93% | **11.98%** | **32.0 pp** |
| `scale+thin` | 113 vs 112 | 33.22% | **8.53%** | **24.7 pp** |

The `scale+thin_rolled` row is an exact budget match (76.7 evals on both sides).
PPO wins decisively at every matched budget, by 25–48 pp.

### 4.2 The "PPO is not clearly better than random search" claim was wrong

That claim compared PPO's ~25% against a random-search figure of ~19%
(23.77% on the corrected ground truth). Those two numbers sit at **112** and
**4801** EC3 evaluations respectively — a **43× budget advantage** to random
search that the broken `ec3_evals_per_design` column concealed.

Stated at honest budgets, PPO + `scale+thin` reaches **8.53%** at **112** evals
while `random_search_4800` reaches **23.77%** at **4801** evals: **15.2 pp better
using 43× fewer structural analyses.** And when random search is given the same
112-eval budget it manages only 33.22%.

**This claim should be reversed in the paper.** It was the single most damaging
number in the project's self-assessment, and it was an artefact of unmeasured
budgets — not evidence about PPO.

### 4.3 The GA reference is validated at equal budget

GA and `random_search_4800` consume **identical** budgets (4801 evals) and
identical bounds. GA achieves **1.72%**, random search **23.77%**. The GA's
evolutionary structure is worth **22.1 pp at matched budget**, which is exactly
the control `random_search_design`'s docstring says it exists to provide. The GA
is a legitimately strong reference, not merely an expensive one.

### 4.4 Honest caveats

- **`rule_based` is a strong, very cheap baseline.** With `scale+thin` it reaches
  **19.17% at 73 evals** — better than PPO's unrepaired 30.15% at 40 evals, and
  at a comparable budget to PPO + `scale+thin_rolled` (77 evals), where PPO wins
  11.98% vs 22.76%. PPO wins every matched comparison, but a closed-form sizer
  plus the repair operator recovers a large share of the benefit for almost no
  search. The paper should report this rather than only the random controls.
- **Most of the repaired arm's budget is the repair operator, not the policy.**
  PPO + `scale+thin` spends 39 evals generating and 72 repairing: **65% of the
  inference budget is the operator.** The headline 8.53% is a property of
  policy-plus-operator, and E1 §6.2 already showed the operator erases the
  differences between reward formulations.
- **Training cost is not in any of these figures.** The 1.3M-step policy consumed
  ~1.3 × 10⁶ EC3 evaluations to train, versus GA's 4801 per design with no
  training. Amortised, PPO + repair (1.3M + 112/design) undercuts a
  4801-evals/design method only beyond **N ≈ 277 designs**. Beyond that
  threshold PPO + repair is both cheaper and better than `random_search_4800`
  (8.53% vs 23.77%); it is cheaper but **5× worse** than the GA (8.53% vs 1.72%).
  Any efficiency claim in the paper must state the amortisation threshold.

## 5. Recommendations for the paper

1. **Report EC3 evaluation counts, not wall-clock.** Wall-clock is
   non-portable (2.57× between two hosts running identical code) and
   misattributes 62% of the GA's cost to Python overhead.
2. **If a wall-clock number is required**, report it with a standard deviation
   and a host specification, measured serially on an idle machine, and note that
   the reference `ga_cost_summary.json` figure has a 23.9% CV and should not be
   quoted.
3. **Reverse the random-search claim** (§4.2) and use `random_search_40` under
   matched repair modes as the equal-budget control.
4. **Add the amortisation threshold** (N ≈ 277) wherever inference efficiency is
   claimed.
5. **Regenerate `ga_cost_summary.json`**, which is stale on two counts: its key
   names (`gap_mean`, not `gap_mean_pct`) predate the units refactor in
   `evaluate.py:summarize`, and its gap statistics are against the pre-correction
   ground truth (E1).

## 6. Reproduction

```bash
# GA micro-benchmark (per-eval cost + serial GA timing, 12 repeats)
#   see §2(c)/§2.1; requires an idle machine

# corrected budget accounting, run SERIALLY
python research/scripts/evaluate_with_repair.py \
    --ground_truth_dir research/pretrain_data_corrected \
    --modes none scale scale+thin scale+thin_rolled \
    --models research/models/gated_cost_merged_seed4{2,3,4,5,6}/final_model \
    --include_ga --out_prefix research/results/budget_gated_merged

python research/scripts/evaluate_with_repair.py \
    --ground_truth_dir research/pretrain_data_corrected \
    --modes none scale scale+thin scale+thin_rolled --models \
    --include_rule_based --random_evals 40 58 114 4800 \
    --out_prefix research/results/budget_baselines
```

## 7. Files changed

| path | change |
|---|---|
| `research/scripts/evaluate_with_repair.py` | `rollout_design` returns `n_ec3`; generation evals propagated from all four generators; `n_ec3_gen` / `n_ec3_repair` / `n_ec3` recorded per context; `sec_per_design` made per-design with an sd; `--random_evals` help corrected |
| `research/results/budget_*` | corrected-accounting evaluation outputs |

Superseded: the `ec3_evals_per_design` column in every
`research/results/e0_repair_*` and `research/results/corrgt_*` file, and all
`sec_per_design` values in `corrgt_*` (contended). Gap statistics in those files
remain valid.

The sandbox has no push credentials for `github.com/mo-rc/hss-beam-gen-design`;
these changes must be committed locally.
