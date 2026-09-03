"""
research/scripts/evaluate_with_repair.py
================================================================
Evaluates trained policies against the objective-specific EC3 ground
truth WITH and WITHOUT the deterministic post-hoc repair operators in
research/algo/repair.py (experiment "E0").

CONTROLLED COMPARISON
---------------------
Each (model, context) episode is rolled out EXACTLY ONCE. The design that
rollout produced is then handed to every repair mode in turn, so
"none" / "scale" / "scale+thin" differ only in the operator applied --
not in the policy rollout, the seed, the context, or the ground truth.
Any difference between the columns is therefore attributable to the
operator alone.

WHICH DESIGN IS REPAIRED
------------------------
`evaluate.py:run_policy_episode(return_best_feasible=True)` already reports
the best FEASIBLE step of an episode, which is what the existing results
tables use, so that is the design repaired here. If an episode never
reaches a feasible step, the terminal design is repaired instead -- the
uniform-scale operator grows an infeasible section until it complies, so
feasibility rate is itself something the operator can improve, which
matters for the shaped / lagrangian / catalog arms.

GA CONTROL
----------
`--include_ga` runs the same repair operators on GA designs. GA already
converges to utilisation 0.9995, so a correct operator must produce
approximately ZERO improvement there. If repair "improves" GA, the
operator is exploiting a modelling inconsistency rather than removing
genuine policy slack, and the PPO numbers cannot be trusted either.
This is a falsification test, not a formality.

REPORTED STATISTICS
-------------------
mean / median / sd / p90 / p95 / worst optimality gap, share within
1% / 5% / 10% of the true optimum, feasibility rate, mean utilisation,
mean section class, EC3 analyses consumed per design, and wall-clock
inference time per design -- i.e. the full set requested in reviewer
comments 2 and 3.

USAGE
-----
    python research/scripts/evaluate_with_repair.py \
        --models research/models/gated_cost_merged_seed4{2,3,4,5,6}/final_model \
        --economy_metric cost --modes none scale scale+thin \
        --out_prefix research/results/e0_repair
================================================================
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.envs.hss_catalog_env import HSSBeamCatalogEnv
from research.algo.repair import (repair, analyse, catalog_repair,
                                  load_catalog, verify_membership)
from research.scripts.evaluate import (
    load_ground_truth, ground_truth_path_for_metric, ground_truth_optimum,
)


def rollout_design(env, policy_fn, span_m, load_kNm, storey=20, max_steps=40, seed=0):
    """One episode at a forced context.

    Returns (design_dict, was_feasible, n_ec3) where `n_ec3` is the number
    of EC3 analyses the GENERATOR consumed -- exactly one per env.step(),
    since HSSBeamEnv.step calls _ec3_analysis once. This is the policy's
    share of the evaluation budget and MUST be counted: an earlier version
    of this harness reported only the repair operator's evals, which made
    every arm look like it cost 1 EC3 analysis to produce a design and so
    silently invalidated every equal-budget comparison.

    Mirrors evaluate.py:run_policy_episode's contract (best feasible step,
    terminal step as fallback) but returns the raw geometry rather than the
    env's info dict, because the repair operators need to re-analyse the
    design at explicitly chosen scale factors.
    """
    env.reset(seed=seed)
    env.use_storey_load_scaling = False
    env.span = float(span_m) * 1000.0
    env.load = float(load_kNm)
    env.storey = int(storey)
    obs = env._get_obs()

    best, best_econ, last = None, np.inf, None
    n_ec3 = 0
    for _ in range(max_steps):
        action = policy_fn(obs)
        obs, _r, terminated, truncated, info = env.step(action)
        n_ec3 += 1
        last = info
        if info["feasible"] and info[env.economy_metric] < best_econ:
            best_econ = info[env.economy_metric]
            best = info
        if terminated or truncated:
            break
    src = best if best is not None else last
    keys = ("h", "b", "tf", "tw", "fy", "section_type")
    return {k: src[k] for k in keys}, (best is not None), n_ec3


def summarise(df, metric, label):
    """Full gap statistics. Infeasible designs are EXCLUDED from gap
    statistics (an infeasible design has no meaningful optimality gap) but
    are counted in `feasibility`, so a method cannot buy a better-looking
    gap by abandoning hard contexts."""
    g = df["gap"].to_numpy(dtype=float)
    f = np.isfinite(g)
    gv = g[f]
    return {
        "arm": label,
        "n": int(len(df)),
        "feasibility": float(df["feasible"].mean()),
        "gap_mean": float(np.mean(gv)) if len(gv) else np.nan,
        "gap_median": float(np.median(gv)) if len(gv) else np.nan,
        "gap_sd": float(np.std(gv, ddof=1)) if len(gv) > 1 else np.nan,
        "gap_p90": float(np.percentile(gv, 90)) if len(gv) else np.nan,
        "gap_p95": float(np.percentile(gv, 95)) if len(gv) else np.nan,
        "gap_worst": float(np.max(gv)) if len(gv) else np.nan,
        "within_1pct": float(np.mean(gv <= 0.01)) if len(gv) else np.nan,
        "within_5pct": float(np.mean(gv <= 0.05)) if len(gv) else np.nan,
        "within_10pct": float(np.mean(gv <= 0.10)) if len(gv) else np.nan,
        "util_mean": float(df["utilization"].mean()),
        "util_median": float(df["utilization"].median()),
        "class_mean": float(df["section_class"].mean()),
        "grade_match": float(df["grade_match"].mean()),
        "repaired_frac": float(df["repaired"].mean()),
        "ec3_gen_per_design": float(df["n_ec3_gen"].mean()),
        "ec3_repair_per_design": float(df["n_ec3_repair"].mean()),
        "ec3_evals_per_design": float(df["n_ec3"].mean()),
        "sec_per_design": float(df["sec_per_design"].mean()),
        "sec_per_design_sd": float(df["sec_per_design"].std(ddof=1)) if len(df) > 1 else np.nan,
    }


CATALOG_MODES = ("catalog_snap", "catalog_snap+grade")


def eval_one(env, get_design, opt, metric, modes, label, storey=20, seed=0,
             catalog=None):
    """Roll out / generate once per context, then apply every repair mode."""
    rows = {m: [] for m in modes}
    t_gen = 0.0
    n_member_violations = 0
    for i, r in opt.iterrows():
        t0 = time.time()
        design, _was_feas, n_gen = get_design(r["span_m"], r["load_kNm"], seed + i)
        t_gen_i = time.time() - t0
        t_gen += t_gen_i
        for m in modes:
            t1 = time.time()
            if m in CATALOG_MODES:
                if catalog is None:
                    raise ValueError(f"mode {m} requires --env_type catalog")
                res = catalog_repair(env, r["span_m"] * 1000.0, r["load_kNm"],
                                     design, catalog, metric=metric,
                                     storey=storey, mode=m)
                if not verify_membership(res, catalog):
                    n_member_violations += 1
            else:
                res = repair(env, r["span_m"] * 1000.0, r["load_kNm"], design,
                             metric=metric, storey=storey, mode=m)
            t_rep = time.time() - t1
            gap = ((res[metric] - r[metric]) / r[metric]) if res["feasible"] else np.nan
            rows[m].append(dict(
                span_m=r["span_m"], load_kNm=r["load_kNm"],
                optimal=r[metric], optimal_grade=r["grade"], optimal_type=r["section_type"],
                achieved=res[metric], gap=gap, feasible=res["feasible"],
                feasible_before=res["feasible_before"],
                utilization=res["utilization"], section_class=res["section_class"],
                h=res["h"], b=res["b"], tf=res["tf"], tw=res["tw"],
                fy=res["fy"], section_type=res["section_type"],
                grade_match=float(res["fy"] == r["grade"]),
                repaired=res["repaired"], variant=res["variant"], scale=res["scale"],
                n_ec3_gen=n_gen, n_ec3_repair=res["n_ec3"],
                n_ec3=n_gen + res["n_ec3"],
                t_gen=t_gen_i, t_repair=t_rep,
            ))
    if n_member_violations:
        raise AssertionError(
            f"{label}: {n_member_violations} catalog-membership violations")
    out = {}
    for m in modes:
        d = pd.DataFrame(rows[m])
        # Per-design wall clock = this design's own generation time plus its
        # own repair time. Previously the SHARED total generation time was
        # divided by n and added to the mean repair time, which produced a
        # single constant for the whole arm and hid per-context variation.
        d["sec_per_design"] = d["t_gen"] + d["t_repair"]
        out[m] = d
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=[])
    p.add_argument("--algo", default="ppo")
    p.add_argument("--economy_metric", default="cost")
    p.add_argument("--reward_mode_for_env", default="feasibility_gated")
    p.add_argument("--env_type", default="continuous", choices=["continuous", "catalog"])
    p.add_argument("--ground_truth_dir", default="research/pretrain_data")
    p.add_argument("--modes", nargs="+", default=["none", "scale", "scale+thin"])
    p.add_argument("--n_contexts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include_ga", action="store_true")
    p.add_argument("--ga_pop", type=int, default=60)
    p.add_argument("--ga_gen", type=int, default=80)
    p.add_argument("--include_rule_based", action="store_true")
    p.add_argument("--random_evals", type=int, nargs="*", default=[],
                   help="Random-search controls, specified as the GENERATION "
                        "budget in EC3 analyses. A PPO rollout also costs 40 "
                        "generation evals, so `--random_evals 40` is the "
                        "correctly matched control at every repair mode: both "
                        "arms then pay the same 40 generation evals plus the "
                        "same repair evals. Do NOT use 58/114 as the matched "
                        "control -- those were chosen to match PPO's "
                        "generation+repair total, but the random arm then pays "
                        "its own repair cost on top, overshooting the budget. "
                        "Larger values (e.g. 4800) are deliberate "
                        "over-budget controls.")
    p.add_argument("--out_prefix", default="research/results/e0_repair")
    a = p.parse_args()

    metric = a.economy_metric
    gt = load_ground_truth(ground_truth_path_for_metric(metric, a.ground_truth_dir))
    opt = ground_truth_optimum(gt, metric)
    if a.n_contexts is not None and a.n_contexts < len(opt):
        opt = opt.sample(n=a.n_contexts, random_state=a.seed).reset_index(drop=True)

    cls = HSSBeamCatalogEnv if a.env_type == "catalog" else HSSBeamEnv
    catalog = load_catalog() if a.env_type == "catalog" else None
    env = cls(reward_mode=a.reward_mode_for_env, economy_metric=metric)

    summaries, per_ctx = [], []
    for mp in a.models:
        name = os.path.basename(os.path.dirname(mp))
        from stable_baselines3 import PPO, DDPG, TD3
        model = {"ppo": PPO, "ddpg": DDPG, "td3": TD3}[a.algo].load(mp)

        def policy_fn(obs):
            act, _ = model.predict(obs, deterministic=True)
            return act

        def get_design(span_m, load_kNm, s):
            return rollout_design(env, policy_fn, span_m, load_kNm, seed=s)

        res = eval_one(env, get_design, opt, metric, a.modes, name, seed=a.seed, catalog=catalog)
        for m, d in res.items():
            d.insert(0, "arm", name); d.insert(1, "repair_mode", m)
            per_ctx.append(d)
            s = summarise(d, metric, name); s["repair_mode"] = m
            summaries.append(s)
        print(f"[done] {name}")

    if a.include_ga:
        from research.scripts.ga_baseline import ga_design

        def get_design_ga(span_m, load_kNm, s):
            g = ga_design(span_mm=span_m * 1000.0, load_kNm=load_kNm, storey=20,
                          economy_metric=metric, pop_size=a.ga_pop,
                          n_generations=a.ga_gen, seed=s)
            # ga_design reports fitness_evals as n_evaluations
            # (= pop_size * n_generations); this is the GA's budget and it
            # dwarfs every other arm's, which is the whole point.
            return ({k: g[k] for k in ("h", "b", "tf", "tw", "fy", "section_type")},
                    bool(g["feasible"]), int(g["n_evaluations"]))

        res = eval_one(env, get_design_ga, opt, metric, a.modes, "ga", seed=a.seed, catalog=catalog)
        for m, d in res.items():
            d.insert(0, "arm", "ga"); d.insert(1, "repair_mode", m)
            per_ctx.append(d)
            s = summarise(d, metric, "ga"); s["repair_mode"] = m
            summaries.append(s)
        print("[done] ga")

    for nev in a.random_evals:
        from research.scripts.ga_baseline import random_search_design

        def get_design_rs(span_m, load_kNm, s, _n=nev):
            g = random_search_design(span_mm=span_m * 1000.0, load_kNm=load_kNm, storey=20,
                                     economy_metric=metric, n_evaluations=_n, seed=s)
            return ({k: g[k] for k in ("h", "b", "tf", "tw", "fy", "section_type")},
                    bool(g["feasible"]), int(g["n_evaluations"]))

        lbl = f"random_search_{nev}"
        res = eval_one(env, get_design_rs, opt, metric, a.modes, lbl, seed=a.seed, catalog=catalog)
        for m, d in res.items():
            d.insert(0, "arm", lbl); d.insert(1, "repair_mode", m)
            per_ctx.append(d)
            s = summarise(d, metric, lbl); s["repair_mode"] = m
            summaries.append(s)
        print(f"[done] {lbl}")

    if a.include_rule_based:
        from research.scripts.ga_baseline import rule_based_design

        def get_design_rb(span_m, load_kNm, s):
            g = rule_based_design(span_mm=span_m * 1000.0, load_kNm=load_kNm, storey=20,
                                  economy_metric=metric)
            # rule_based_design reports the yield strength as "grade", not "fy".
            return (dict(h=g["h"], b=g["b"], tf=g["tf"], tw=g["tw"],
                         fy=g["grade"], section_type=g["section_type"]),
                    bool(g["feasible"]), 0)  # closed-form sizer: no search evals

        res = eval_one(env, get_design_rb, opt, metric, a.modes, "rule_based", seed=a.seed, catalog=catalog)
        for m, d in res.items():
            d.insert(0, "arm", "rule_based"); d.insert(1, "repair_mode", m)
            per_ctx.append(d)
            s = summarise(d, metric, "rule_based"); s["repair_mode"] = m
            summaries.append(s)
        print("[done] rule_based")

    S = pd.DataFrame(summaries)
    cols = ["arm", "repair_mode", "n", "feasibility", "gap_mean", "gap_median", "gap_sd",
            "gap_p90", "gap_p95", "gap_worst", "within_1pct", "within_5pct", "within_10pct",
            "util_mean", "class_mean", "grade_match", "repaired_frac",
            "ec3_gen_per_design", "ec3_repair_per_design",
            "ec3_evals_per_design", "sec_per_design", "sec_per_design_sd"]
    S = S[cols]
    os.makedirs(os.path.dirname(a.out_prefix), exist_ok=True)
    S.to_csv(f"{a.out_prefix}_summary.csv", index=False)
    pd.concat(per_ctx, ignore_index=True).to_csv(f"{a.out_prefix}_per_context.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print("\n" + S.to_string(index=False))
    print(f"\nwrote {a.out_prefix}_summary.csv and {a.out_prefix}_per_context.csv")


if __name__ == "__main__":
    main()
