# E3 — Corrected-Cost PPO Retrain: Results

**Status:** complete. **Date:** 2026-09-03.
**Closes:** E1 §10 open item 1 and the E1 §6.5 transfer caveat.
**Run:** `corrcost_gated_merged_seed43`, seed 43, 1.3M steps, single seed,
trained by the user locally under `enforce_rolled_manufacturability=True`.
Evaluated against the corrected cost ground truth, n = 142 contexts.

> **Headline: retraining under corrected costing recovers only ~40% of the gap
> that the correction opened, but it changes the policy's behaviour decisively.
> The retrained policy learns to stay inside the rolled envelope — its
> costed-as-welded rate collapses 20.4% → 3.5%, essentially matching the
> corrected GA optima's 2.8% — and yet its mean gap improves by only 0.90 pp.
> E1 §6.5's caveat is therefore resolved as a *finding*: the widened gap was
> **not** transfer shock. The policy adapts to the corrected economics almost
> perfectly and still cannot close the gap, which is direct confirmation that
> the bottleneck is the action parameterisation (diagnosis §3), not the costing.**

**New authoritative post-correction PPO numbers: 22.65% unrepaired, 6.63% with
`scale+thin`, 100% feasible, 108.7 EC3 evaluations per design.** These supersede
the E1 §6 transfer figures as the number the paper should quote for PPO.

---

**Artifact status: complete.** The full run was pushed to the repo at `c7b3c35`
(2026-09-03) — `final_model.zip`, `vecnormalize.pkl`, six intermediate
checkpoints with matching VecNormalize statistics, `training_config.json`, and
the TensorBoard event file. Limitation 3 of the first draft of this document is
withdrawn. §1c and §5 below are new analysis enabled by those artifacts.

---

## 1. Integrity checks (all three pass)

Before interpreting anything, three checks that the run is the controlled
experiment it claims to be.

**(a) The run was scored against the authoritative E1 corrected ground truth.**
The 142 per-context `optimal` values in the uploaded results were matched
against the per-context minima of `results/gt_corrected_cost.csv`:
**142/142 contexts matched, max relative difference 2.22e-16, zero grade
mismatches, zero section-type mismatches.** Not a rebuilt or drifted reference —
the same one.

**(b) The GA control reproduces E1 exactly.** The uploaded `ga` rows give
1.7200% / 1.4739% (none / `scale+thin`) against E1 §6's 1.72% / 1.47%, agreeing
to all reported digits. The optimiser noise floor is unchanged, so gaps here are
directly comparable to every E1 number.

**(c) The environment and hyperparameters are confirmed against the artifacts,
not taken on trust.** Two checks against `c7b3c35`:

- `research/envs/hss_env.py` at `c7b3c35` is **byte-identical** to the
  reconstruction in `code/hss_env.py` (`diff` reports no differences). The lost
  E1 cost patch was reconstructed correctly, so every statement in the runbook
  about what the environment does is verified rather than inferred.
- `training_config.json` matches the runbook run spec **field for field**:
  `feasibility_gated` / `cost` / `log_relative`, seed 43, 1,300,000 steps,
  8 envs, lr 3e-4, `n_steps` 1024, batch 256, 8 epochs, γ 0.99, λ 0.95,
  clip 0.15, `ent_coef` 0.03, `vf_coef` 0.5, `max_grad_norm` 0.5, ltb 0.40,
  sls 0.50, `log_std_anneal` true with ceiling 8.0 → 1.0 at `start_frac` 0.5.
  No undocumented deviation. TensorBoard confirms **1,302,528 steps** actually
  ran (159 rollout logs), so the run completed rather than being truncated.

Re-evaluating `final_model` here reproduces the uploaded CSVs exactly
(22.6508% / 6.6314%), so the uploaded results are the artifact's results.

**Correction to the E3 runbook §1.3:** I recorded the corrected ground truth as
lost. It was not — it survived as `results/gt_corrected_cost.csv` /
`gt_corrected_co2.csv`, and as of `c7b3c35` it is restored in the repo at
`research/pretrain_data_corrected/`. Runbook Step 2's 21-minute regeneration is
unnecessary. Only the `hss_env.py` patch was genuinely lost, and it is now
restored too.

---

## 2. The controlled comparison

The right control is **the same seed's pre-correction checkpoint evaluated in
the corrected environment** — same hyperparameters, same seed, same contexts,
only the training-time costing differs. I re-ran that transfer baseline here
(`gated_cost_merged_seed43/final_model`), and it reproduces E1 §6's stored
`corrgt` seed-43 values to 5 decimal places (23.2022% vs 23.2009% none;
7.5269% vs 7.5269% `scale+thin`), so the comparison below is paired and clean.

**Do not compare 6.63% against E1's 8.53%.** That 8.53% is a five-seed mean, and
seed 43 was the *best* of the five. The like-for-like number is 7.53%.

| repair mode | transfer (seed 43) | **retrained (seed 43)** | Δ | 95% CI (paired bootstrap, 20k) | Wilcoxon p | contexts improved |
|---|---|---|---|---|---|---|
| `none` | 23.20% | **22.65%** | −0.55 pp | [−2.70, +1.28] | 0.090 | 38% |
| `scale` | 11.39% | **11.34%** | −0.05 pp | [−2.05, +1.65] | 0.017 | 39% |
| **`scale+thin`** | 7.53% | **6.63%** | **−0.90 pp** | [−2.92, +0.73] | 0.018 | 36% |
| `scale+thin_rolled` | 9.42% | **9.37%** | −0.05 pp | [−2.05, +1.56] | 0.0004 | 32% |

### 2.1 The mean and the median disagree, and that is the real result

Every paired bootstrap CI on the mean straddles zero, yet the Wilcoxon tests are
significant — **in the opposite direction.** The retrained policy is better on
only 32–39% of contexts, so the *median* context gets slightly worse, while the
*mean* improves. It is not a wash; it is a redistribution:

| `scale+thin` | transfer | retrained |
|---|---|---|
| mean | 7.53% | **6.63%** |
| median | **4.08%** | 4.75% |
| sd | 14.31 pp | **8.11 pp** |
| p90 | 12.32% | 12.37% |
| p95 | 16.54% | **13.20%** |
| **worst** | 89.94% | **62.50%** |
| within 1% | **10.6%** | 4.9% |
| within 5% | **58.5%** | 52.1% |
| within 10% | **86.6%** | 85.2% |

Retraining **cuts the tail and loses the peak.** Worst case falls 27.4 pp,
dispersion nearly halves, p95 falls 3.3 pp — but the policy stops nailing the
easy contexts (within-1% more than halves). The mean improvement is bought
entirely from the contexts the old policy handled catastrophically.

For a design tool this is the *preferable* trade: a 62% worst case is bad, a 90%
worst case is unusable. Report mean **and** worst-case, not mean alone.

### 2.2 Fraction of the correction's damage recovered

Three-point chain for seed 43, all n = 142:

| | `none` | `scale+thin` |
|---|---|---|
| old costing, old GT (E0) | 21.85% | 5.41% |
| old-costing policy → corrected GT (transfer, E1 §6) | 23.20% | 7.53% |
| **retrained under corrected costing (E3)** | **22.65%** | **6.63%** |
| widening caused by the correction | +1.35 pp | +2.12 pp |
| **recovered by retraining** | **0.55 pp (41%)** | **0.90 pp (42%)** |
| **unrecovered** | 0.80 pp | 1.22 pp |

The 41% / 42% agreement across two independent repair modes is a useful
consistency signal. **Roughly 40% of the correction's cost to PPO was transfer
shock and is recoverable by retraining; roughly 60% is a genuine, permanent loss
because the corrected reference is a better optimiser than the policy.**

---

## 3. What the retrained policy actually learned

This is the most informative part of the run, and it is not visible in the gap
numbers at all. Median values over the 142 unrepaired designs:

| | d/t_w | c/t_f | t_w | t_f | **costed as welded** | mean util | mean class |
|---|---|---|---|---|---|---|---|
| transfer seed 43 (`none`) | 72.1 ε | 8.0 ε | 10.67 mm | 18.00 mm | **20.4%** | 0.838 | 1.61 |
| **retrained seed 43 (`none`)** | **66.3 ε** | 8.8 ε | **11.75 mm** | 16.53 mm | **3.5%** | 0.841 | **1.45** |
| corrected GA optima (E1 §4) | 72.9 ε | — | 8.80 mm | — | **2.8%** | 1.000 | — |

The policy moved **exactly the way the GA moved in E1 §4**: webs thickened
(10.67 → 11.75 mm), slenderness pulled back off the envelope boundary
(72.1 → 66.3 ε), section class dropped (1.61 → 1.45), and the
costed-as-welded rate collapsed **20.4% → 3.5%**, landing on the corrected
optima's own 2.8%. It internalised the manufacturability economics essentially
completely, with no explicit manufacturability constraint anywhere in the reward
— which independently vindicates E1 §6.3's recommendation to fix the cost model
rather than add a hard constraint.

**And it bought 0.90 pp.** The policy now agrees with the reference about *what
kind of section is cheap* and still misses the optimum by 6.63% after repair,
because it does not reach `util = 1.0` (mean utilisation 0.841, unchanged from
0.838). The failure is capacity slack and plate proportioning — the retreat from
the discontinuity cliff described in diagnosis §3.2–§3.4 — and correcting the
economics does nothing about that. **This is a clean, decisive confirmation of
the diagnosis by a route the diagnosis did not use.**

### 3.1 `scale+thin_rolled` stays a negative result

9.37% vs 6.63% for unconstrained `scale+thin` — still uniformly worse, now under
retraining as well as transfer (E1 §6.3 measured 11.98% vs 8.53%). Confirmed on
a policy that was actually trained under the corrected cost model, so it is no
longer attributable to a train/test mismatch. **E1 §6.3's recommendation stands:
keep the corrected cost model, do not add an explicit manufacturability
constraint.**

---

## 4. Convergence — was 1.3M steps enough?

The checkpoints make this answerable. Every saved checkpoint of both the
retrained run and the pre-correction seed-43 run was evaluated against the
corrected ground truth, giving two learning curves in *gap space* rather than
reward space:

| steps | retrained `none` | retrained `scale+thin` | pre-corr `none` | pre-corr `scale+thin` |
|---|---|---|---|---|
| 200k | 148.48% | 26.82% | 199.93% | 48.64% |
| 400k | 63.62% | 9.16% | 68.74% | 20.49% |
| 600k | 51.93% | 10.99% | 48.54% | 10.44% |
| 800k | 33.31% | 8.89% | 46.27% | 11.67% |
| 1.0M | 27.80% | 7.91% | 27.36% | 7.84% |
| 1.2M | 22.55% | 6.75% | — | — |
| **1.3M (final)** | **22.65%** | **6.63%** | **23.20%** | **7.53%** |

Three things follow.

**The corrected cost model did not make learning harder — it made it easier
early.** The retrained run is ahead at 200k (26.8% vs 48.6%) and 400k (9.2% vs
20.5%), and grade-matching reaches 0.79 by 400k versus 0.45 for the
pre-correction run. The corrected economics are evidently a cleaner signal. By
600k–1.0M the two curves are indistinguishable, and they separate again only at
the end.

**The run was still improving when the budget ran out, but only marginally.**
Over the final 300k steps the retrained curve moved 7.91% → 6.63% (−1.28 pp)
versus −0.31 pp for the pre-correction run over the same window — so it had not
fully converged. But the improvement is decelerating hard: −1.16 pp over
1.0M→1.2M, then −0.12 pp over the final 100k, and unrepaired `none` is flat over
that last window (22.55% → 22.65%). **Extending the budget would plausibly buy a
few tenths of a percentage point, not the ~5 pp needed to approach the GA.** The
6.63% headline should be read as a slightly conservative estimate of this
configuration's asymptote, not as a truncated run.

**Utilisation is the binding constraint and it converged.** Mean utilisation
climbs 0.506 → 0.665 → 0.758 → 0.803 → 0.839 → 0.841 and flattens — at 0.841,
identical to the pre-correction policy's 0.838. Both runs converge to the same
utilisation ceiling well below 1.0 from opposite training conditions. That is a
property of the parameterisation, not of the cost model, and it is the direct
mechanical cause of the residual gap (§3).

### 4.1 Exploration was pinned at the anneal ceiling

The TensorBoard traces show `log_std_anneal/actual_mean_std` sitting **exactly
at the ceiling for 100% of the final 30% of training** in the retrained run,
versus 57% for the pre-correction run. The policy wanted more action variance
than the schedule allowed for the whole back half. Terminal `clip_fraction`
averaged 0.39 (max 0.49) and `approx_kl` 0.027 — both high, indicating the
updates were being clipped aggressively throughout.

Explained variance reached 0.998, so the critic is fine; this is a policy-side
constraint. It is consistent with, but does not by itself prove, the diagnosis
§3 picture of a Gaussian policy pressed against a constraint boundary it cannot
sit on. **Flagging as an observation, not acting on it** — changing the anneal
schedule would be a new experiment.

---

## 5. Efficiency, with corrected E2 accounting

| arm | mean gap | EC3 evals / design |
|---|---|---|
| retrained PPO, unrepaired | 22.65% | **39.4** |
| **retrained PPO + `scale+thin`** | **6.63%** | **108.7** |
| GA control + `scale+thin` | 1.47% | 4,868 |

PPO plus repair reaches 6.63% using **45× fewer** structural evaluations than the
GA, and the GA is still **4.5× more accurate**. Both halves must be stated
together. E2 §5's amortisation-threshold requirement is unaffected.

---

## 6. What this supersedes

| claim | previous | now |
|---|---|---|
| Best PPO arm, unrepaired, corrected GT | 30.15% ±4.50 (5-seed transfer) | **22.65%** (seed 43, retrained, point estimate) |
| Best PPO arm + `scale+thin`, corrected GT | 8.53% ±1.47 (5-seed transfer) | **6.63%** (seed 43, retrained, point estimate) |
| E1 §6.5 "these are transfer numbers, retraining should recover some of the gap" | open caveat | **resolved: ~40% recoverable, ~60% permanent** |
| "Is the widened gap the reference or the policy?" | unanswered | **the reference got harder AND the policy is parameterisation-limited; costing was never the policy's problem** |
| Runbook §1.3 "corrected GT was lost" | asserted | **retracted — it is `results/gt_corrected_cost.csv`** |

Unchanged: the GA noise floor (1.72%), `scale+thin_rolled` as a negative result,
the E2 budget accounting, and every mass result.

---

## 7. Limitations, stated plainly

1. **Single seed, no confidence interval.** 22.65% / 6.63% are point estimates.
   Seed 43 was the *best* of the five pre-correction seeds, so both numbers are
   optimistic as estimates of the arm's expected performance. The paired
   comparison against the same seed's transfer baseline is sound; the absolute
   levels are not a five-seed mean and must not be presented as one.
2. **All four paired bootstrap CIs on the mean include zero.** The −0.90 pp
   `scale+thin` improvement is not established as non-zero on n = 142. The
   distributional changes (§2.1, §3) are much larger than the mean shift and are
   where the evidence actually is.
3. ~~No training artefacts were uploaded.~~ **Withdrawn** — all artifacts are in
   the repo at `c7b3c35` and the run is fully verified (§1c, §4).
4. **The 1.3M budget is near but not at convergence** (§4). The reported 6.63% is
   mildly conservative for this configuration.
5. ~~The one-shot reparameterised direction (diagnosis §4–§5: 3.6% from blind
   random search, 1.87% from leave-one-out 3-NN) remains far ahead of this
   result and untouched by it.~~ **No longer untouched** — both were re-measured
   against corrected costing and the corrected ground truth, at matched contexts
   and matched budgets, with the original-costing arm reproducing the diagnosis
   tables to every published digit. The probe no longer reaches 1.6%: at the same
   1,000-proposal budget it gets 3.1% mean / 2.4% median, against a GA floor that
   also moved (1.72% vs ~1.31%), so it goes from 1.2× to 1.8× its own floor. The
   direction still dominates this result — blind sampling reaches 5.9% at 4,800
   EC3 evaluations, beating the retrained policy's 6.63% after 1.3M steps. The
   leave-one-out k-NN predictor holds up better: 2.00% mean / 0.64% median at
   k = 1, i.e. **closer to the GA floor under corrected costing (+0.28 pp) than it
   was under original costing (+0.56 pp)**, with a median below the GA's own. Its
   degradation is confined to six edge-of-envelope contexts (span ≥ 14 m,
   load ≥ 100 kN/m) where the mean is 37.16%; over the other 136 it is 0.97%,
   inside the noise floor. See `experiments/E3_conclusions_for_paper.md` §6 and
   `results/e3_{reparam_probe_costing_ab_n20,reparam_probe_costing_ab_n40,knn_oneshot_corrected_summary}.csv`.

---

## 8. Files

| path | content |
|---|---|
| `results/e3_corrcost_retrain_summary.csv` | user's run — per-arm/per-mode summary |
| `results/e3_corrcost_retrain_per_context.csv` | user's run — 142 contexts × 4 modes + GA |
| `results/e3_transfer_seed43_summary.csv` | same-seed pre-correction transfer control, re-run here |
| `results/e3_transfer_seed43_per_context.csv` | per-context values for the paired tests in §2 |
| `results/e3_curve_retrained_seed43_summary.csv` | §4 learning curve, all 7 retrained checkpoints |
| `results/e3_curve_precorrection_seed43_summary.csv` | §4 learning curve, all 7 pre-correction checkpoints |
| `results/E3_RESULTS_MANIFEST.md` | row/column/MD5 manifest + per-file column documentation |
| `results/e3_reparam_probe_costing_ab_n40.csv` | §7.5 probe A/B, 40 contexts, both cost models |
| `results/e3_reparam_probe_costing_ab_n20.csv` | §7.5 probe A/B, 20 contexts, large budgets |
| `results/e3_knn_oneshot_corrected_summary.csv` | §7.5 LOO k-NN, corrected costing, k × label fraction |
| `results/e3_knn_oneshot_corrected_per_context.csv` | per-context k-NN detail (142 × 9 configurations) |
| `results/e3_knn_oneshot_originalcosting_summary.csv` | k-NN reproduction check vs the diagnosis report |
| `results/e3_knn_oneshot_originalcosting_per_context.csv` | per-context detail for the reproduction check |
| `experiments/E3_conclusions_for_paper.md` | draft conclusions section |
| `code/diag_knn_oneshot.py` | LOO k-NN one-shot predictor |
| `code/diag_reparam_probe_costing_ab.py` | matched two-costing probe driver |

The two curve files gained explicit `checkpoint` and `steps` columns in this
pass, and the pre-correction curve gained its 1,200,000-step row (the checkpoint
existed but had not been evaluated), putting both curves on identical step
grids. Details and verification hashes are in `results/E3_RESULTS_MANIFEST.md`.

Upstream artifacts (repo `c7b3c35`): `research/models/corrcost_gated_merged_seed43/`
and `research/runs/corrcost_gated_merged_seed43_1/`.

As of `c7b3c35` the repo is self-contained — `research/envs/hss_env.py`,
`research/algo/repair.py`, `research/envs/manufacturability.py`,
`research/scripts/evaluate_with_repair.py` and `research/pretrain_data_corrected/`
are all present, so both commands below run against a clean clone with no
patching or ground-truth regeneration.

Reproduce the §2 paired transfer control:

```bash
python research/scripts/evaluate_with_repair.py \
    --ground_truth_dir research/pretrain_data_corrected \
    --modes none scale scale+thin scale+thin_rolled \
    --models research/models/gated_cost_merged_seed43/final_model \
    --out_prefix research/results/e3_transfer_seed43
```

Reproduce the §4 convergence curve:

```bash
python research/scripts/evaluate_with_repair.py \
    --ground_truth_dir research/pretrain_data_corrected \
    --modes none scale+thin \
    --models research/models/corrcost_gated_merged_seed43/checkpoint_{200000,400000,600000,800000,1000000,1200000}_steps \
             research/models/corrcost_gated_merged_seed43/final_model \
    --out_prefix research/results/e3_curve_retrained_seed43
```

No methodology, reward, action space, hyperparameter or costing decision was
changed in producing this analysis.
