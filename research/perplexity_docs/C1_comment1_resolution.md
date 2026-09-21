# Comment 1 — Internal numerical inconsistencies: resolution

Manuscript checked: `research/manuscript_final_comprehensive.pdf` (18 pp), later replaced by
`research/manuscript_final_comprehensive.md`. **All corrections in §C have been applied to the `.md`
(41 edits, see §G).**
Everything below was recomputed from files in this repository; nothing is taken from prose.

**Acceptance criterion:** every quoted number traces to exactly one evaluation run, and every
use of "repair" names the operator.

- Numbers: `python research/scripts/c1_verify_manuscript_numbers.py` recomputes 70 quantities
  from named source files → `results/c1_number_ledger.csv`
  (**58 PASS, 3 PASS within 1 unit of last digit, 4 FAIL, 5 INFO**). The 4 failures are
  items B2, B3, B4 and B6 below; all other manuscript numbers trace.
- Repair wording: all 32 lines of the manuscript text that contain "repair" are classified in §D.

---

## A. What changed in the repository

| File | Change |
|---|---|
| `scripts/diag_reparam_probe_costing_ab.py` | **New.** The driver the E3 manifest cites but was never committed. Pairs each costing with its own ground truth, evaluates all budgets on one context set in one run, writes mean/median/p90/worst/feasibility together plus per-context gaps. |
| `scripts/diag_reparam_probe.py` | **Fixed.** Default paired *corrected* costing with the *original* ground truth (gave 13.5% instead of 5.9% at 200 proposals). Now `--costing {corrected,original}` selects both together; CSV stores full statistics, not just the mean. |
| `scripts/c1_verify_manuscript_numbers.py` | **New.** Re-runnable number ledger. |
| `algo/repair.py` | Header no longer calls both operators "repair"; states what each is. |
| `README.md` | New §6a (exact step schedule), §6b (operator terminology), §6c (what each "GA gap" is). |
| `results/c1_*` | Evidence files (see below). |

---

## B. Findings and evidence

### B1. Step-size schedule — manuscript §3.2 is wrong
Manuscript: *"decreases linearly from coarse (σ0 = 0.4) to fine (σT = 0.05) over T = 40 steps"*.
Code (`envs/hss_env.py::_update_design`): half-cosine, `s_t = 0.30 + 0.70·½(1+cos(πt/40))`,
1.0 → 0.30 (step 1 = 0.999, step 20 = 0.650, step 40 = 0.300), multiplying full-scale
increments of 50 / 28 / 3 / 2.5 mm for h / b / tf / tw. **Neither σ0 = 0.4 nor σT = 0.05 exists
anywhere in the code.** All models use horizon 40 (train.py default, no overrides).

### B2. Table 7 — mean and median are from different sources
| Proposals | Manuscript mean / median | Committed corrected-costing file | Committed original-costing file |
|---|---|---|---|
| 40 | 14.5% / **8.2%** | 14.47% / 16.13% (40 ctx) | 9.41% / 8.52% |
| 200 | 5.9% / **3.1%** | 5.86% / 3.99% (20 ctx) | 3.58% / 3.02% |
| 1000 | 3.1% / 2.4% | 3.09% / 2.43% (20 ctx) | 1.59% / 1.25% |

Means are corrected-costing; the 8.2% and 3.1% medians match neither costing. The table also
mixes a 40-context file (row 1) with a 20-context file (rows 2–3). **Replacement** (one run,
all 142 contexts, corrected costing, `results/c1_reparam_budget_table.csv`):

| Proposals | EC3 evals | Mean | Median | p90 | Worst | Feasibility |
|---|---|---|---|---|---|---|
| 40 | 960 | 11.67% | 10.63% | 21.65% | 37.9% | 1.000 |
| 200 | 4,800 | 5.34% | 4.16% | 12.04% | 20.0% | 1.000 |
| 1,000 | 24,000 | 2.30% | 1.49% | 5.67% | 12.3% | 1.000 |

The original-costing rows of the same run are in the CSV. **Driver validated:** in `--rng sequential`
mode it reproduces all 18 rows of the committed E3 A/B files with maximum difference 0.0
(`results/c1_driver_legacy_repro_n{20,40}.csv`).

### B3. GA baseline — five values, four configurations
| Value | Configuration | Source |
|---|---|---|
| **1.72% / 0.73%** | GA 4,800 evals, no operator, corrected costing + corrected GT | `e4_nonrl_baselines_summary.csv` |
| **1.47% / 0.54%** | same GA + `scale+thin` | same |
| 1.31% / 0.81% | GA 4,800, no operator, original costing + original GT | `ga_cost_summary.json` |
| 0.83% | original costing + `scale+thin` | E0 doc |
| 0.86–0.89% | ground truth vs ~9× larger GA (n = 100) | README prose only; no stored output |

**Independent verification (your question 3):** re-running `evaluate_with_repair.py --include_ga`
from scratch on the corrected ground truth gave 1.7200% / 0.7257% (none) and 1.4739% / 0.5396%
(scale+thin) — **bit-identical to the stored file (max difference 0.0)**
(`results/c1_ga_baseline_repro_*.csv`). So 1.72% and 1.47% are both *correct for their own
configuration*; the problem is only labelling.

What the number means: the reference searches each (span, load, grade, type) with pop 50 × 80 ×
2 restarts and takes the best of 12 combinations (≈ 96,000 evaluations); the GA baseline uses
4,800. It is therefore a *fixed-budget GA vs. a ~20×-larger-budget reference*, not a repeat-run
"noise floor". Distribution: 37/142 contexts have negative gap (GA beat the reference); 14 exceed
5%; 3 exceed 10%; worst 36.6% (span 15 m, load 107 kN/m); mean excluding the worst 5 is 1.17%.
**Recommendation:** compare like-with-like — PPO+`scale+thin` vs GA+`scale+thin` (1.47%),
unrepaired vs unrepaired (1.72%) — and state which in every table.

### B4. Other numerical/label errors found while tracing
| Loc. | Problem | Evidence |
|---|---|---|
| T5 | PPO median **6.41%** is untraceable | files give 5.66% (mean of per-seed medians) or 5.42% (pooled 426) |
| §5.7 | "k-NN median 0.64% **below** GA median 0.54%" is false | 0.64 > 0.54; it is below the *unrepaired* GA median 0.73% |
| §4.3 | "138/142 optima costed as **welded**" is inverted | recomputed: 138 rolled, 4 welded (agrees with T2's 4/142) |
| T2 | "Optima costed as welded 139/142 → 4/142" mislabelled | pre-correction, all 142 were *costed as rolled*; 139 *would have to be welded* under the rule |
| §4.3, §1.3 | "GA (pop 60, 80 gen, 4,800 evals) … 1,728 span-load contexts (12×12×**12 grades**)" | ground truth is pop 50 × 80 × 2 restarts; 1,728 = 12 spans × 12 loads × **6** grades × 2 types; 4,800 is the *baseline* GA |
| §1.3 | "1M steps, **three seeds each**" for four algorithms | T5: DDPG/SAC/TD3 are seed 42 only |
| §6.1 | "PPO's 6.34% after **1.3M** steps" | T6 / E5: 6.34% is seed 43 at **1M** steps (6.63% is the 1.3M run) |
| §5.3 | "beats random search (23.77%) by 15.2 pp" mixes configs | 23.77% is *unrepaired* random search; like-for-like (both `scale+thin`) is 11.60% → 3.06 pp |
| T1 vs T5 | "±" means different things | T1: 95% t-CI half-width; T5: sample SD (E1 doc: population SD) |
| Abstract / T6 | "5.50 pp" annealing effect vs headline 6.34% | 5.50 pp is E5-B (linear economy); 6.34% is E5-A, which also changes the economy reward (Δ = 6.01 pp) |
| T1 | no costing label | T1 is *pre-correction* costing; §3.5/§7 quote 30.15 → 8.53 (*corrected*) |
| §5.2 / §7.1 | "98.8% of designs violated physical feasibility" | 827/837 rolled-labelled *rows* (not designs) lie outside the *manufacturability* envelope; EC3 feasibility is unaffected |

Table 7's 5.9% vs PPO 6.34% (Abstract, §5.7, §6.1) compares a 20-context figure with a 142-context
figure. On 142 contexts the probe gives 5.34% at 4,800 evals — still below 6.34%, but PPO used
112 evals (43× fewer); state both budgets.

### B5. Repair operators — decomposition the manuscript lacks
Corrected costing, 5 seeds, 142 contexts (`corrgt_gated_merged_summary.csv`):
unrepaired **30.15%** → `scale` **17.33%** → `scale+thin` **8.53%**. So `scale` (projection) supplies
12.8 pp and the cost-improving thinning a further 8.8 pp. The manuscript reports only the endpoints,
attributing all of it to "the repair operator". `scale` is itself only a strict *feasibility* repair
for infeasible inputs; for feasible, under-utilised designs (mean util 0.84) it removes capacity
slack. The project's own E0 document already states `scale+thin` "is NOT repair"; the E3/E4 docs
and the manuscript do not follow it.

---

## C. Replacement text for the manuscript

**§3.2, last sentence.** *"Actions are increments to (h, b, t_f, t_w) with full-scale magnitudes of
50, 28, 3 and 2.5 mm, multiplied by a step scale that follows a half-cosine from 1.0 to 0.30 over
the T = 40-step episode, s_t = 0.30 + 0.70·½[1 + cos(πt/T)] (s_1 = 0.999, s_20 = 0.650, s_40 = 0.300).
The remaining two components select the grade (nearest of six grade centres) and section type (sign)."*

**§3.5 (retitle "Post-hoc operators").** *"We apply two deterministic post-hoc operators of different
kinds. `scale` is a constraint-boundary projection: it scales (h, b, t_f, t_w) uniformly until
utilisation = 1.0, restoring feasibility for infeasible designs and removing capacity slack from
feasible ones; it never changes section class. `scale+thin` is a cost-improving local search: it
evaluates `scale` plus candidates with plates thinned toward the Class 1–3 slenderness limits and
returns the cheapest feasible one; it changes section class and is not a feasibility repair.
Under corrected costing the mean cost gap of the best PPO arm falls from 30.15% (unrepaired) to
17.33% (`scale`) to 8.53% (`scale+thin`), all at 100% feasibility."*

**§4.3, GA sentences.** *"Ground truth: one GA per (span, load, grade, section type) —
12 × 12 × 6 × 2 = 1,728 combinations, 1,623 feasible — with population 50, 80 generations and 2
restarts, under manufacturability-aware costing; a span-load context's reference is the cheapest of
its 12 (grade, type) results (142 of 144 contexts feasible). Of the 142 reference optima, 138 are
rolled and 4 welded. The GA baseline (population 60 × 80 generations = 4,800 evaluations, all six
variables) reaches a 1.72% mean / 0.73% median gap unrepaired and 1.47% / 0.54% with `scale+thin`;
the reference used roughly 20× more evaluations, so we report this as the GA reference gap rather
than a repeat-run noise floor."*

**Abstract / §5.3 / T5 / Conclusion:** write "GA + `scale+thin` (4,800 evaluations): 1.47%".
**T2 row 4:** "GA reference gap, unrepaired: 1.31% → 1.72%". **T2 row 5:** "Optima that must be
welded (fail rolled envelope): 139/142 (costed as rolled under the original model) → 4/142".
**T5 PPO median:** 5.66% (mean of per-seed medians). **T1 caption:** add "pre-correction costing;
± = 95% t-interval half-width over 5 seeds". **T5 caption:** add "± = sample SD over 3 seeds".
**§5.7 finding 2:** *"k-NN k = 1 median 0.64% is below the unrepaired GA median (0.73%) but above
the `scale+thin` GA median (0.54%); its designs are analytically sized to utilisation 1.0."*
**§5.3 finding 1:** *"…reaches 8.53% at 112 evals; random search with the same operator and 4,800
evals reaches 11.60% (3.06 pp worse, 43× more evals); unrepaired 4,800-eval random search: 23.77%."*
**§1.3 (algorithms):** "PPO, three seeds; DDPG, SAC and TD3, one seed each". **§6.1(1):** "after 1M steps".

## D. Every use of "repair" (32 lines in the PDF text)
Rule: "post-hoc operators" collectively; "`scale` (projection)" or "`scale+thin` (cost-improving
search)" individually; "repair" only for `scale` acting on an infeasible design. "Unrepaired" may stay
if defined once as "no post-hoc operator applied".

**Must change:** §1.3 contribution bullet 3 (title and both clauses) · §3.5 heading ·
§3.5 first sentence · §3.5 last paragraph · §4.4 ("under scale+thin repair") · §5.1 heading ·
§5.1 lead sentence · Table 1 caption · Table 1 column header "Repair mode" · §5.1 findings 1, 2
("Repair erases…", "Repair prefers Class 1…") · §5.3 amortisation bullet · Table 5 caption ·
§7.1(2) · §7.2(5) (three lines) · §7.2(6) · §7.3(7) (two lines) · §7.4(2) · §7.4(6) ·
Conclusion bullet 2 · Data availability ("repair operator in code/repair.py") · Appendix A.4 heading ·
Appendix B `scale+thin` row ("Repair operator variant").

**May stay:** reference [12] (title of the E0 document) · file names `repair.py`,
`evaluate_with_repair.py` · "unrepaired" as a defined term.

## E. Not changed — flagged for later comments
* Headline 6.34% is a single seed (the best of five) → **Comment 2**.
* Data availability cites `code/…` paths; the repo layout is `research/…`; SB3/Gymnasium versions in
  §4.1 differ from E0's note (`stable-baselines3==2.3.2`) — verify before submission.
* The E0/E3/E4/E5 docs still say "`scale+thin` repair" (historical records; left as written).

## F. Reproduce
```bash
export PYTHONPATH=$PWD
python research/scripts/diag_reparam_probe_costing_ab.py --n_contexts 142 \
  --budgets 10 20 40 200 1000 --raw_budgets 40 400 4000 4800 \
  --out research/results/c1_reparam_budget_table.csv          # ~6.5 min, 1 CPU
python research/scripts/evaluate_with_repair.py --include_ga --ground_truth_dir \
  research/pretrain_data_corrected --modes none scale scale+thin --n_contexts 142 \
  --seed 0 --out_prefix /tmp/ga_repro                          # ~70 s
python research/scripts/c1_verify_manuscript_numbers.py
```

## G. Manuscript edits applied (`manuscript_final_comprehensive.md`)
41 string replacements, each asserted to match exactly once. Beyond §C, the same pass also:
* replaced Table 7 with the single-run table (added p90 and feasibility columns);
* restated the §5.1 variant-choice claim with fractions (2,551/3,834 = 67% all-arm pre-correction vs
  289/710 = 41% best-arm corrected — the original compared counts over different denominators);
* corrected Appendix A.3/A.4: they trained/evaluated the 1.3M-step run (6.63%) while the headline
  6.34% is `e5_ppo_s43_anneal_logrel` at 1M steps (both commands now given);
* fixed the Data-availability paths (`code/…` → `research/…`) and added `scale` to the nomenclature.
Audit after patching: no remaining "5.9%", "linearly", σ0/σT, "6.41%", "costed as welded",
"12 grades", or "three seeds each"; the only remaining "repair" strings are the defined term
"unrepaired", file names, and reference [12]'s title.
