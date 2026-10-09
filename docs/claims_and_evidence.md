# Claims, evidence and known limitations

Every headline claim of the paper, where its number comes from, and what it does **not** show. All
numbers are in `paper/tables/` (built by `pipeline/11_build_paper_tables.py` from `results/`) and
`figures/`; `tests/` fail if a table drifts from the logged value. Gap = achieved / best-known
optimum − 1, over feasible contexts; the headline operator mode is `scale+thin`.

## Claims

| # | Claim | Evidence | Strength |
|---|---|---|---|
| C1 | With PPO, `feasibility_gated` (8.9%) is not distinguishable from `lagrangian` (9.4%); `shaped` (15.8%) is worse on the point estimate; no pairwise difference survives Holm (n = 5). | t1, t4, Fig. 3; `results_2a_reward_mode.md` | descriptive |
| C2 | PPO has the lowest gap of PPO / SAC / TD3 / DDPG (8.9 vs 24.9 / 36.1 / 32.1%); only PPO vs TD3 survives Holm (p = 0.048). SAC/TD3/DDPG are seed-unstable (per-seed 8–88%). | t2, t4, Fig. 3; `results_2b_algorithm.md` | one significant pair; instability is visible per seed |
| C3 | The same configuration transfers to mass (7.2%) and CO2 (4.6%), 100% feasible in all 10 runs. | t3; `results_2c_objective_transfer.md` | descriptive (different references per objective) |
| C4 | Out of distribution the gaps stay in the same range (cost 9.1–9.7%, mass 4.5–7.4%, CO2 5.0–7.3%); the one real degradation is cost on `ood_span` (feasibility 90%). | t5, Fig. 2; `results_ood.md` | descriptive; 26 and 47 contexts |
| C5 | At equal evaluations PPO beats GA, DE and random search: with `scale+thin` in all 9 (objective, method) comparisons (Holm p = 0.024, the smallest attainable at 5 vs 5); with no operator (40 vs 40 evaluations) in 7 of 9 (mass vs DE and vs random: Holm 0.063). | t10, Fig. 1, Fig. 5 | exact test, complete separation of seeds; power-limited by n = 5 |
| C6 | Per-context search overtakes the policy with more evaluations: DE needs roughly 4–10x the policy's evaluations to match its gap, and at B = 4800 is at or just below the reference on the main grid (−0.5%, −0.1%, −0.3% for cost, mass, CO2). | t6, Fig. 1; `results_search_baselines.md` | descriptive, budget grid |
| C7 | A kNN over stored optima is more accurate than the policy (0.5–1.1% vs 4.6–8.9%) at fewer evaluations per design (41–63), but needs solved labelled contexts and is less often feasible (cost 95%, OOD down to 69%). | t7, Fig. 2, Fig. 4; `results_knn_baseline.md` | deterministic; 2-parameter context only |
| C8 | With realistic labels (one DE search each) the kNN stays ahead of PPO in distribution from B = 1000, and on 5 of 6 OOD sets (mass on `ood_span` excepted) (36 labelled contexts cost 36,000 evaluations vs PPO's 1,000,000 training steps); OOD results are label-seed dependent. | t8, Fig. 4; `results_knn_cheap_labels.md` | 3 label seeds; OOD reported as ranges |
| C9 | Much of every method's quality at `scale+thin` is the operator's: PPO without an operator has a 48–49% gap (100% feasible); the kNN is 39–44% and the searches (B = 40) 77–83% feasible without it. | t9, Fig. 5 | descriptive, same operator for all methods |
| C10 | In wall-clock time the policy's advantage over search is small: 29-30 ms per design; DE matches its gap at 1.5-1.6x the time (cost, mass) and 4.4x (CO2); at matched evaluations searches are 3.5-4.6x faster. | t11 | single checkpoint per objective, one thread, kNN not timed |

## What a reviewer can legitimately ask, and the status

1. **Context dimension.** The context is (span, load). A kNN over stored optima is competitive because
   the space is two-dimensional. The experiments say nothing about higher-dimensional contexts, where
   interpolation degrades and an amortized policy is expected to matter more. *Not tested;* state it
   as a limitation, and as the reason the policy is of interest (no labels, one model for all contexts).
   A small extension (one more context parameter, with its ground truth) would answer it.
2. **Wall-clock time.** Measured (t11, `pipeline/12_time_inference.py`, one process, one thread, design plus
   `scale+thin` operator): the policy takes 29-30 ms per design, searches at B = 40 take 6.6-8.4 ms for a
   similar number of evaluations. A policy step costs 0.25-0.28 ms against 0.06-0.15 ms for a search
   evaluation, so the policy is about 3.5-4.6x *slower* at matched evaluations. The gap advantage at matched
   evaluations therefore does not carry over to seconds. DE matches the policy's gap at B = 400 for cost
   and mass (about 47 ms, 1.5-1.6x the policy's time) and at B = 1000 for CO2 (about 130 ms, 4.4x; the
   budget grid is coarse). The kNN was not timed. One checkpoint per objective was timed, and the policy
   forward pass is unoptimised (single-sample PyTorch call per step). Claim evaluation-count efficiency;
   state the wall-clock result as measured.
3. **Operator dependence.** Headlines use `scale+thin`. Fig. 5 / t9 report all three operator modes for
   every method, and t10 tests PPO vs search with no operator at all.
4. **Reference.** The reference is the best design found by pooled GA searches, not a proven global
   optimum; DE at B = 4800 lands up to 0.55% below it. Gaps are relative to this reference.
5. **Feasible-only gaps.** Gaps are over feasible contexts; every table and Fig. 2 / Fig. 5 show feasibility
   next to the gap. The searches and the kNN are infeasible more often than PPO, so their feasible-only
   gaps are, if anything, favourable to them.
6. **Five seeds.** Exact permutation tests are used because n is small; with five seeds per arm the
   smallest attainable two-sided p is 0.0079 (Holm over three: 0.024). Confidence intervals are
   bootstrap and descriptive. The seeds replicate training / search, not the contexts.
7. **Baseline fairness.** Search budgets are matched in total EC3 evaluations including the operator's
   analyses; population size follows a fixed rule (no per-budget tuning); the kNN uses the same
   operators. A reviewer may ask for tuned GA/DE; the searches are not tuned.
8. **RL hyperparameters.** One frozen configuration is used for all arms (a seed-99 pilot found no
   effect and the configuration was kept); no per-algorithm tuning is reported, so SAC/TD3/DDPG results
   are for this configuration.
9. **Not evaluated.** `ood_joint` has no feasible reference design; learning curves are not reported
   (training runs are not tracked in the repository).
10. **Context dimension.** The context is (span, effective UDL load); the serviceability limit is fixed at L/250
    (`hss_env.py`, `delta / (L/250)`; utilisation is the maximum of moment and deflection utilisation). Storey is a
    training-time variable that scales the load by 1 to 1.5 and is part of the observation; in every evaluation and in
    all ground truth the load is used directly and storey is fixed at 20. `pipeline/14_deflection_sensitivity.py` (t12)
    shows that a tighter limit would move the reference: at L/360, 75 of 142 cost optima and all 142 mass and CO2
    optima violate. This is a re-check of stored designs, not a result for a policy trained over limits.
