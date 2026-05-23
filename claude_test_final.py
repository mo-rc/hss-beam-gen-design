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

from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

from env.high_rise_generative_env_claude_final import HighRiseGenerativeEnv


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
    obs, _ = env.reset()
    done = False

    # Track per-step data
    steps = []

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps.append({
            "step":         len(steps) + 1,
            "util":         info["utilization"],
            "mass":         info["mass"],
            "cost":         info["cost"],
            "co2":          info["co2"],
            "chi_lt":       info["chi_lt"],
            "h":            info["h"],
            "b":            info["b"],
            "tf":           info["tf"],
            "tw":           info["tw"],
            "fy":           info["fy"],
            "section_type": info["section_type"],
            "span":         info["span"],
            "load":         info["load"],
            "storey":       info["storey"],
            "reward":       reward,
            "section_class": info["ec3"].get("section_class", 0) if info.get("ec3") else 0,
            "Mrd":          info["ec3"].get("Mrd", 0) if info.get("ec3") else 0,
            "Med":          info["ec3"].get("Med", 0) if info.get("ec3") else 0,
            "lambda_lt":    info["ec3"].get("lambda_lt", 0) if info.get("ec3") else 0,
            "defl_util":    info["ec3"].get("deflection_util", 0) if info.get("ec3") else 0,
            "moment_util":  info["ec3"].get("moment_util", 0) if info.get("ec3") else 0,
        })

    if not steps:
        return {}

    # Best step = closest util to 0.95 among feasible steps
    feasible = [
        s for s in steps
        if s["util"] <= 1.05
        and s["section_class"] < 4
        and s["section_class"] > 0
    ]

    if feasible:
        best = min(feasible, key=lambda s: abs(s["util"] - 0.95))
    else:
        # No feasible step found — return the last step
        best = steps[-1]

    best["n_steps"] = len(steps)
    best["had_feasible"] = len(feasible) > 0
    best["ep_reward"]    = sum(s["reward"] for s in steps)

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

    for span in spans:
        for load in loads:
            env = HighRiseGenerativeEnv(
                use_storey_load_scaling=False,   # fixed grid, no scaling
                sls_load_factor=args.sls_factor,
                ltb_restraint_factor=args.ltb_factor,
            )
            # Force fixed demand
            obs, _ = env.reset(seed=42)
            env.span  = float(span)
            env.load  = float(load)
            env.storey = 20

            done = False
            steps = []
            while not done:
                action, _ = model.predict(env._get_obs(), deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                if info.get("ec3"):
                    steps.append(info)

            if steps:
                feasible = [
                    s for s in steps
                    if s["utilization"] <= 1.05
                    and s.get("ec3", {}).get("section_class", 4) < 4
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
            env.close()

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
    
    
    env = HighRiseGenerativeEnv(
        use_storey_load_scaling=True,
        sls_load_factor=args.sls_factor,
        ltb_restraint_factor=args.ltb_factor,
    )

    # ── Load model ────────────────────────────────────────────
    model = PPO.load(args.model)
    print(f"\n  Model loaded: {args.model}")

    # Load VecNormalize stats if present (normalises observations)
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    vecnorm_path = args.model.replace("best_model.zip","vecnormalize.pkl").replace(
                   "final_model.zip","vecnormalize.pkl")
    _vecnorm = None
    if os.path.exists(vecnorm_path):
        print(f"  VecNormalize stats loaded: {vecnorm_path}")
        _vecnorm = VecNormalize.load(vecnorm_path, DummyVecEnv([lambda: env]))
        _vecnorm.training = False
        _vecnorm.norm_reward = False

    # ── Random episode validation ─────────────────────────────
    print(f"\n  Running {args.episodes} random validation episodes...")

    

    random_results = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        result = run_episode(model, env, deterministic=True)
        if result:
            random_results.append(result)

        if (ep + 1) % 50 == 0:
            done = ep + 1
            feasible_so_far = sum(1 for r in random_results if r.get("had_feasible"))
            # print(f"  [{done}/{args.episodes}]  "
            #       f"feasible rate: {feasible_so_far/done:.1%}")

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

    # Target band: 0.90 ≤ util ≤ 1.05
    in_target = sum(
        1 for r in feasible_results
        if 0.90 <= r.get("util", 0) <= 1.05
    )
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
    # print(f"  Summary JSON        : {summary_path}")
    # print(f"  TensorBoard logs    : {tb_dir}")
    # print(f"\n  Run:  tensorboard --logdir {tb_dir}")
    # print(f"{SEP}\n")

    writer.flush()
    writer.close()

    return summary


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    validate(args)