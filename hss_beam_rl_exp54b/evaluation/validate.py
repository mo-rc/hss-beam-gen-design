"""
================================================================
validate.py
----------------------------------------------------------------
Post-Training Validation Script — HSS Beam Generative Design
HKU Research Project

Loads a trained PPO model and runs it deterministically across
a structured grid of span / load combinations, then:
  - Reports EC3 compliance statistics
  - Exports all results to CSV
  - Generates TensorBoard validation scalars
  - Prints a console summary table

USAGE:
    python validate.py --model models/<run_name>/best_model.zip
    python validate.py --model models/<run_name>/best_model.zip --episodes 500
    python validate.py --model models/<run_name>/best_model.zip --grid

================================================================
"""

import os
import csv
import json
import argparse
import datetime
import numpy as np
from collections import defaultdict
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from env.high_rise_generative_env import HighRiseGenerativeEnv


# ================================================================
# ARGUMENT PARSER
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate trained PPO — HSS Beam"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model zip (e.g. models/run/best_model.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=200,
        help="Number of random validation episodes (default: 200)"
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Also run a structured span×load grid (8×8 = 64 episodes)"
    )
    parser.add_argument(
        "--ltb-factor", type=float, default=0.25,
        help="Must match the training ltb_restraint_factor"
    )
    parser.add_argument(
        "--sls-factor", type=float, default=0.50,
        help="Must match the training sls_load_factor"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for random validation episodes (default: 0)"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for CSV/logs (auto from model path)"
    )
    return parser.parse_args()


# ================================================================
# SINGLE EPISODE RUNNER
# ================================================================
def run_episode(model, env, deterministic: bool = True) -> dict:
    """
    Run one full episode with the trained model.
    Returns a flat dict of the best step (lowest valid util).
    """
    obs = env.reset()
    done = False

    # Track per-step data
    steps = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, dones, infos = env.step(action)
        
        # Unpack vectorized step arrays (index 0 for a single parallel env)
        done = dones[0]
        info = infos[0]
        step_reward = reward[0]
        
        rt = info.get("reward_terms", {})

        steps.append({
            "step": len(steps) + 1,

            "util": info["utilization"],
            "mass": info["mass"],
            "cost": info["cost"],
            "co2": info["co2"],
            "chi_lt": info["chi_lt"],

            "h": info["h"],
            "b": info["b"],
            "tf": info["tf"],
            "tw": info["tw"],

            "fy": info["fy"],
            "section_type": info["section_type"],

            "span": info["span"],
            "load": info["load"],
            "storey": info["storey"],

            "reward": step_reward,

            # --- Reward Components ---
            "economy_reward":      rt.get("economy_reward", 0.0),
            "utilization_reward":  rt.get("utilization_reward", 0.0),
            "co2_lca_reward":rt.get("co2_lca_reward", 0.0),
            "improvement_reward":  rt.get("improvement_reward", 0.0),
            "hss_demand_bonus": rt.get("hss_demand_bonus", 0.0),
            "novelty_reward":      rt.get("novelty_reward", 0.0),
            "feasibility_penalty": rt.get("feasibility_penalty", 0.0),
            "underutil_penalty":   rt.get("underutil_penalty", 0.0),
            
            "section_class": info["ec3"].get("section_class", 0) if info.get("ec3") else 0,
            "Mrd":           info["ec3"].get("Mrd", 0) if info.get("ec3") else 0,
            "Med":           info["ec3"].get("Med", 0) if info.get("ec3") else 0,
            "lambda_lt":     info["ec3"].get("lambda_lt", 0) if info.get("ec3") else 0,
            "defl_util":     info["ec3"].get("deflection_util", 0) if info.get("ec3") else 0,
            "moment_util":   info["ec3"].get("moment_util", 0) if info.get("ec3") else 0,
        })

    if not steps:
        return {}

    # Best step = closest util to 0.95 among feasible steps
    feasible = [
        s for s in steps
        if 0.90 <= s["util"] <= 1.05
        and 0 < s["section_class"] < 4
        
    ]

    if feasible:
        best = min(feasible, key=lambda s: abs(s["util"] - 0.95))
    else:
        # No feasible step found — return the last step
        best = steps[-1]

    best["n_steps"] = len(steps)
    best["had_feasible"] = len(feasible) > 0
    best["ep_reward"]    = sum(s["reward"] for s in steps)
    
    reward_keys = [
        "economy_reward",
        "utilization_reward",
        "co2_lca_reward",
        "improvement_reward",
        "hss_demand_bonus",
        "novelty_reward",
        "feasibility_penalty",
        "underutil_penalty",
    ]

    for k in reward_keys:
        best[f"ep_{k}"] = sum(s.get(k, 0.0) for s in steps)

    return best


# ================================================================
# GRID VALIDATION
# ================================================================
def run_grid(model, args) -> list[dict]:
    """
    Structured 8×8 span × load grid.
    Spans: 6, 7, 8, 9, 10, 11, 13, 15 m
    Loads: 20, 30, 40, 60, 80, 100, 120, 140 kN/m
    """
    spans = [6000, 7000, 8000, 9000, 10000, 11000, 13000, 15000]
    loads = [20, 30, 40, 60, 80, 100, 120, 140]

    results = []
    total = len(spans) * len(loads)
    print(f"\n  Running {total}-point span×load grid...")


    vecnorm_path = args.model.replace("best_model.zip", "vecnormalize.pkl").replace(
                   "final_model.zip", "vecnormalize.pkl")
    
    for span in spans:
        for load in loads:
            raw_env = HighRiseGenerativeEnv(
                use_storey_load_scaling=True,   # fixed grid, no scaling
                sls_load_factor=args.sls_factor,
                ltb_restraint_factor=args.ltb_factor,
            )
            # Force fixed demand
            obs, _ = raw_env.reset(seed=42)
            raw_env.span  = float(span)
            raw_env.load  = float(load)
            raw_env.storey = 20

            # ── FIX [grid diagnostic bug] ─────────────────────────────
            # reset(seed=42) is called with the SAME seed for every one
            # of the 64 grid cells, so h/b/tf/tw/fy were drawn identically
            # every time (before this span/load overwrite) — every grid
            # cell started the policy from the exact same geometry
            # (h≈490mm, S620) regardless of the actual span/load being
            # tested. That contaminates "mass barely changes across the
            # load sweep" as evidence, since it may just reflect a shared,
            # non-representative starting point rather than genuine
            # failure to condition on context.
            #
            # Reseed geometry deterministically (no randomness) from the
            # ACTUAL grid cell's span, matching reset()'s own span-
            # proportional formula but without h_noise/grade-curriculum
            # randomization, so every cell starts from its own comparable,
            # context-appropriate point.
            span_m_now = raw_env.span / 1000.0
            h_target   = float(np.clip(span_m_now * 42.0, 250.0, 750.0))
            raw_env.h  = h_target
            raw_env.b  = float(np.clip(h_target / 3.0, 120.0, 300.0))
            raw_env.tf = 20.0
            raw_env.tw = 12.0
            raw_env.fy = 500.0   # neutral mid-grade start (was: fixed-seed leftover)
            # ──────────────────────────────────────────────────────────

            # Wrap it into a DummyVecEnv matching the random evaluation pipeline
            venv = DummyVecEnv([lambda: raw_env])
            if os.path.exists(vecnorm_path):
                eval_env = VecNormalize.load(vecnorm_path, venv)
                eval_env.training = False
                eval_env.norm_reward = False
                eval_env.clip_obs = 10.0
            else:
                eval_env = venv
            
            obs = raw_env._get_obs()
                
            # Normalize the manually altered observation vector
            obs = eval_env.normalize_obs(obs)
            obs = np.expand_dims(obs, axis=0) if obs.ndim == 1 else obs

            done = False
            steps = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = eval_env.step(action)
                
                done = dones[0]
                info = infos[0]
                
                if info.get("ec3"):
                    steps.append(info)

            if steps:
                feasible = [
                    s for s in steps
                    if 0.90 <= s["utilization"] <= 1.05
                    and 0 < s.get("ec3", {}).get("section_class", 4) < 4
                ]
                best = (
                    min(feasible, key=lambda s: abs(s["utilization"] - 0.95))
                    if feasible else steps[-1]
                )
                results.append({
                    "span_m":      span / 1000,
                    "load_kNm":    load,
                    "util":        best["utilization"],
                    "mass_kg":     best["mass"],
                    "cost":        best["cost"],
                    "co2_kg":      best["co2"],
                    "chi_lt":      best["chi_lt"],
                    "h_mm":        best["h"],
                    "b_mm":        best["b"],
                    "tf_mm":       best["tf"],
                    "tw_mm":       best["tw"],
                    "fy_MPa":      best["fy"],
                    "section_type": best["section_type"],
                    "section_class": best.get("ec3", {}).get("section_class", 0),
                    "lambda_lt":   best.get("ec3", {}).get("lambda_lt", 0),
                    "feasible":    len(feasible) > 0,
                    "n_steps":     len(steps),
                })
            eval_env.close()

    print(f"  Grid complete: {len(results)} / {total} converged")
    return results


# ================================================================
# STATISTICS HELPERS
# ================================================================
def compute_stats(results: list[dict], key: str) -> dict:
    vals = [r[key] for r in results if key in r]
    if not vals:
        return {}
    return {
        "mean": float(np.mean(vals)),
        "std":  float(np.std(vals)),
        "min":  float(np.min(vals)),
        "max":  float(np.max(vals)),
        "p25":  float(np.percentile(vals, 25)),
        "p50":  float(np.percentile(vals, 50)),
        "p75":  float(np.percentile(vals, 75)),
    }


# ================================================================
# MAIN VALIDATION FUNCTION
# ================================================================
def validate(args):

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Output directory ─────────────────────────────────────
    if args.out_dir:
        out_dir = args.out_dir
    else:
        model_dir = os.path.dirname(args.model)
        out_dir   = os.path.join(model_dir, f"validation_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    tb_dir = os.path.join(out_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    print("=" * 64)
    print("  PPO Validation — HSS Beam Generative Design")
    print("=" * 64)
    print(f"  Model             : {args.model}")
    print(f"  Random episodes   : {args.episodes}")
    print(f"  Grid validation   : {args.grid}")
    print(f"  Output dir        : {out_dir}")
    print(f"  ltb_factor        : {args.ltb_factor}")
    print("=" * 64)
    
    
    raw_env = HighRiseGenerativeEnv(
        use_storey_load_scaling=True,
        sls_load_factor=args.sls_factor,
        ltb_restraint_factor=args.ltb_factor,
    )

    # ── Load model ────────────────────────────────────────────
    model = PPO.load(args.model)
    print(f"\n  Model loaded: {args.model}")

    # Load VecNormalize stats if present (normalises observations)
    env = DummyVecEnv([lambda: raw_env])
    vecnorm_path = args.model.replace("best_model.zip","vecnormalize.pkl").replace(
                   "final_model.zip","vecnormalize.pkl")
    _vecnorm = None
    if os.path.exists(vecnorm_path):
        print(f"  VecNormalize stats loaded: {vecnorm_path}")
        _vecnorm = VecNormalize.load(vecnorm_path, env)
        _vecnorm.training = False
        _vecnorm.norm_reward = False
        _vecnorm.clip_obs = 10.0
    else:
        print("  WARNING: vecnormalize.pkl not found! Evaluating on raw inputs.")

    # ── Random episode validation ─────────────────────────────
    print(f"\n  Running {args.episodes} random validation episodes...")

    

    random_results = []
    for ep in range(args.episodes):
        # Access original un-wrapped environment method to seed setup configs securely
        raw_env.reset(seed=args.seed + ep)
        result = run_episode(model, _vecnorm, deterministic=True)
        if result:
            random_results.append(result)

        if (ep + 1) % 50 == 0:
            done = ep + 1
            feasible_so_far = sum(1 for r in random_results if r.get("had_feasible"))
            print(f"  [{done}/{args.episodes}]  "
                  f"feasible rate: {feasible_so_far/done:.1%}")

    env.close()

    # ── Compute statistics ────────────────────────────────────
    feasible_results = [r for r in random_results if r.get("had_feasible")]
    feasibility_rate = len(feasible_results) / max(len(random_results), 1)

    util_stats   = compute_stats(feasible_results, "util")
    mass_stats   = compute_stats(feasible_results, "mass")
    cost_stats   = compute_stats(feasible_results, "cost")
    co2_stats    = compute_stats(feasible_results, "co2")
    chi_lt_stats = compute_stats(feasible_results, "chi_lt")
    reward_stats = compute_stats(random_results,   "ep_reward")

    economy_stats = compute_stats(random_results, "ep_economy_reward")
    util_reward_stats = compute_stats(random_results, "ep_utilization_reward")
    co2_lca_reward_stats = compute_stats(random_results, "ep_co2_lca_reward")
    improvement_reward_stats = compute_stats(random_results, "ep_improvement_reward")
    hss_demand_bonus_stats = compute_stats(random_results, "ep_hss_demand_bonus")
    underutil_stats = compute_stats(random_results, "ep_underutil_penalty")
    feasibility_penalty_stats = compute_stats(random_results, "ep_feasibility_penalty")
    
    # Target band: 0.90 ≤ util ≤ 1.05
    
    in_target = sum(1 for r in feasible_results if 0.90 <= r.get("util", 0) <= 1.05)
    target_rate = in_target / max(len(feasible_results), 1)

    # Grade distribution
    grade_counts = defaultdict(int)
    type_counts  = defaultdict(int)
    class_counts = defaultdict(int)
    for r in feasible_results:
        grade_counts[int(r.get("fy", 0))] += 1
        type_counts[r.get("section_type", "?")] += 1
        class_counts[r.get("section_class", 0)] += 1

    # ── TensorBoard validation scalars ────────────────────────
    writer.add_scalar("val/feasibility_rate",    feasibility_rate,           0)
    writer.add_scalar("val/target_rate",         target_rate,                0)
    writer.add_scalar("val/util_mean",           util_stats.get("mean", 0),  0)
    writer.add_scalar("val/util_std",            util_stats.get("std",  0),  0)
    writer.add_scalar("val/mass_mean_kg",        mass_stats.get("mean", 0),  0)
    writer.add_scalar("val/cost_mean",           cost_stats.get("mean", 0),  0)
    writer.add_scalar("val/co2_mean_kg",         co2_stats.get("mean", 0),   0)
    writer.add_scalar("val/chi_lt_mean",         chi_lt_stats.get("mean",0), 0)
    writer.add_scalar("val/episode_reward_mean", reward_stats.get("mean",0), 0)
    writer.add_scalar("val/economy_reward_mean", economy_stats.get("mean",0),0)
    writer.add_scalar("val/utilization_reward_mean", util_reward_stats.get("mean",0),0)
    writer.add_scalar("val/co2_lca_reward_mean", co2_lca_reward_stats.get("mean",0),0)
    writer.add_scalar("val/improvement_reward_mean", improvement_reward_stats.get("mean",0),0)
    writer.add_scalar("val/hss_demand_bonus_mean", hss_demand_bonus_stats.get("mean",0),0)
    writer.add_scalar("val/underutil_penalty_mean", underutil_stats.get("mean",0),0)
    writer.add_scalar("val/feasibility_penalty_mean", feasibility_penalty_stats.get("mean",0),0)
    # Per-grade mass (shows how well HSS reduces mass)
    for grade, count in sorted(grade_counts.items()):
        grade_mass = [
            r["mass"] for r in feasible_results
            if int(r.get("fy", 0)) == grade
        ]
        if grade_mass:
            writer.add_scalar(
                f"val/mass_by_grade/S{grade}",
                float(np.mean(grade_mass)), 0
            )

    # Utilisation histogram
    if feasible_results:
        utils = np.array([r["util"] for r in feasible_results], dtype=np.float32)
        writer.add_histogram("val/utilization_distribution", utils, 0)

        masses = np.array([r["mass"] for r in feasible_results], dtype=np.float32)
        writer.add_histogram("val/mass_distribution", masses, 0)

        chi_lts = np.array([r["chi_lt"] for r in feasible_results], dtype=np.float32)
        writer.add_histogram("val/chi_lt_distribution", chi_lts, 0)

    # ── Console summary ───────────────────────────────────────
    SEP  = "=" * 64
    SEP2 = "-" * 64

    print(f"\n{SEP}")
    print("  VALIDATION RESULTS — Random Episodes")
    print(SEP)
    print(f"  Total episodes          : {len(random_results)}")
    print(f"  Feasible episodes       : {len(feasible_results)}  "
          f"({feasibility_rate:.1%})")
    print(f"  In target [0.90–1.05]   : {in_target}  "
          f"({target_rate:.1%} of feasible)")
    print()
    print(f"  {'Metric':<22}  {'Mean':>8}  {'Std':>8}  "
          f"{'P25':>8}  {'P50':>8}  {'P75':>8}")
    print(f"  {SEP2}")

    rows = [
        ("Utilisation",      util_stats),
        ("Mass [kg]",        mass_stats),
        ("Cost [£]",         cost_stats),
        ("CO2 [kg]",         co2_stats),
        ("chi_lt",           chi_lt_stats),
        ("Episode reward",   reward_stats),
    ]
    for label, s in rows:
        if s:
            print(f"  {label:<22}  {s['mean']:>8.3f}  {s['std']:>8.3f}  "
                  f"{s['p25']:>8.3f}  {s['p50']:>8.3f}  {s['p75']:>8.3f}")

    print(f"\n{SEP}")
    print("  REWARD BREAKDOWN")
    print(SEP2)

    reward_rows = [
        ("Economy Reward", economy_stats),
        ("Utilization Reward", util_reward_stats),
        ("CO₂ LCA Reward", co2_lca_reward_stats),
        ("Improvement Reward", improvement_reward_stats),
        ("HSS Demand Bonus", hss_demand_bonus_stats),
        ("Underutil Penalty", underutil_stats),
        ("Feasibility Penalty", feasibility_penalty_stats),
    ]

    for label, s in reward_rows:
        if s:
            print(
                f"  {label:<22}"
                f"{s['mean']:>10.2f}"
                f"{s['std']:>10.2f}"
            )
    
    print(f"\n{SEP}")
    print("  Grade Distribution (feasible episodes)")
    print(SEP2)
    total_f = max(len(feasible_results), 1)
    for grade in sorted(grade_counts.keys()):
        cnt = grade_counts[grade]
        bar = "█" * int(cnt / total_f * 40)
        print(f"  S{grade:<4}  {cnt:>4} ({cnt/total_f:>5.1%})  {bar}")

    print(f"\n  Section type:")
    for t, cnt in sorted(type_counts.items()):
        print(f"    {t:<8}  {cnt:>4} ({cnt/total_f:>5.1%})")

    print(f"\n  Section class:")
    for cls in sorted(class_counts.keys()):
        cnt = class_counts[cls]
        print(f"    Class {cls}  {cnt:>4} ({cnt/total_f:>5.1%})")
    
    print(f"\n{SEP}")
    print("  SECTION GEOMETRY ANALYSIS")
    print(SEP2)

    if feasible_results:
        print(f"  Mean h  = {np.mean([r['h'] for r in feasible_results]):.1f} mm")
        print(f"  Mean b  = {np.mean([r['b'] for r in feasible_results]):.1f} mm")
        print(f"  Mean tf = {np.mean([r['tf'] for r in feasible_results]):.1f} mm")
        print(f"  Mean tw = {np.mean([r['tw'] for r in feasible_results]):.1f} mm")

    print(f"\n{SEP}")
    print("  GRADE PERFORMANCE")
    print(SEP2)

    for grade in sorted(grade_counts.keys()):

        subset = [
            r for r in feasible_results
            if int(r["fy"]) == grade
        ]

        if not subset:
            continue

        print(
            f"  S{grade}"
            f" | Util={np.mean([r['util'] for r in subset]):.3f}"
            f" | Mass={np.mean([r['mass'] for r in subset]):.1f}"
            f" | Cost={np.mean([r['cost'] for r in subset]):.1f}"
            f" | CO2={np.mean([r['co2'] for r in subset]):.1f}"
            f" | n={len(subset)}"
        )
    # ── Grid validation ───────────────────────────────────────
    grid_results = []
    if args.grid:
        grid_results = run_grid(model, args)

        if grid_results:
            grid_feasible = [r for r in grid_results if r["feasible"]]
            grid_rate = len(grid_feasible) / max(len(grid_results), 1)

            writer.add_scalar("val/grid_feasibility_rate", grid_rate, 0)

            print(f"\n{SEP}")
            print("  GRID VALIDATION RESULTS  (span × load)")
            print(SEP)
            print(f"  {'Span [m]':<10} {'Load [kN/m]':<12} {'Util':<8} "
                  f"{'Mass [kg]':<11} {'fy [MPa]':<10} {'Type':<8} "
                  f"{'Class':<7} {'Feasible'}")
            print(f"  {SEP2}")

            for r in grid_results:
                feas_str = "YES" if r["feasible"] else "NO "
                print(f"  {r['span_m']:<10.0f} {r['load_kNm']:<12.0f} "
                      f"{r['util']:<8.3f} {r['mass_kg']:<11.1f} "
                      f"{r['fy_MPa']:<10.0f} {r['section_type']:<8} "
                      f"{r['section_class']:<7} {feas_str}")
            
            print(f"\n{SEP}")
            print("  GRADE TRANSITION ANALYSIS")
            print(SEP2)

            for r in sorted(grid_results,
                            key=lambda x: (x["span_m"], x["load_kNm"])):

                print(
                    f"{r['span_m']:>2.0f}m "
                    f"{r['load_kNm']:>3.0f}kN/m "
                    f"-> S{int(r['fy_MPa'])}"
                )
            
            print(f"\n{SEP}")
            print("  DEMAND VS GRADE")
            print(SEP2)

            bins = defaultdict(list)

            for r in grid_results:

                demand = r["span_m"] * r["load_kNm"]

                bins[int(r["fy_MPa"])].append(demand)

            for grade in sorted(bins):

                print(
                    f"S{grade}"
                    f" | mean demand = {np.mean(bins[grade]):.1f}"
                    f" | max demand = {np.max(bins[grade]):.1f}"
                    f" | n={len(bins[grade])}"
                )
                
            grade_stats = defaultdict(list)

            for r in grid_results:

                grade_stats[int(r["fy_MPa"])].append(r)

            print(f"\n{SEP}")
            print("  GRADE EFFICIENCY")
            print(SEP2)

            for grade in sorted(grade_stats):

                rows = grade_stats[grade]

                print(
                    f"S{grade}"
                    f" | Mass={np.mean([x['mass_kg'] for x in rows]):.1f}"
                    f" | Cost={np.mean([x['cost'] for x in rows]):.1f}"
                    f" | CO2={np.mean([x['co2_kg'] for x in rows]):.1f}"
                    f" | Util={np.mean([x['util'] for x in rows]):.3f}"
                )

    # ── Export to CSV ─────────────────────────────────────────
    # Random episodes CSV
    random_csv = os.path.join(out_dir, "validation_random.csv")
    if random_results:
        fieldnames = list(random_results[0].keys())
        with open(random_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(random_results)
        print(f"\n  Random results CSV  : {random_csv}")

    # Grid CSV
    if grid_results:
        grid_csv = os.path.join(out_dir, "validation_grid.csv")
        fieldnames = list(grid_results[0].keys())
        with open(grid_csv, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(grid_results)
        print(f"  Grid results CSV    : {grid_csv}")

    # Summary JSON
    summary = {
        "model":              args.model,
        "timestamp":          timestamp,
        "n_episodes":         len(random_results),
        "feasibility_rate":   feasibility_rate,
        "target_rate_90_105": target_rate,
        "util":               util_stats,
        "mass_kg":            mass_stats,
        "cost":               cost_stats,
        "co2_kg":             co2_stats,
        "chi_lt":             chi_lt_stats,
        "episode_reward":     reward_stats,
        "grade_distribution": {str(k): v for k, v in grade_counts.items()},
        "type_distribution":  dict(type_counts),
        "class_distribution": {str(k): v for k, v in class_counts.items()},
    }
    summary_path = os.path.join(out_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON        : {summary_path}")
    print(f"  TensorBoard logs    : {tb_dir}")
    print(f"\n  Run:  tensorboard --logdir {tb_dir}")
    print(f"{SEP}\n")

    writer.flush()
    writer.close()

    return summary


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    validate(args)
