"""
research/scripts/evaluate.py
================================================================
Evaluates a trained policy (PPO/DDPG/TD3 from research/scripts/train*.py)
OR the GA baseline against the brute-force EC3 ground truth
(pretrain_data/ec3_optimal_designs.csv), producing the metric your
supervisor's Comment #3 asked for and paper.md never actually computed:
optimality gap (mean/median/p90/p95) relative to a TRUE optimum, not a
self-referential feasibility rate.

GROUND TRUTH RECONSTRUCTION
------------------------------
pretrain_data/ec3_optimal_designs.csv has one row per (span, load, grade,
section_type) -- i.e. it is the minimum-mass design FOR A FIXED GRADE.
The TRUE optimum for a (span, load) context, allowing the optimizer to
also choose grade and section type (exactly what the RL agent and GA are
free to choose), is the best row across all grade x section_type
combinations at that (span, load). This module computes that directly
from the CSV (`_ground_truth_optimum`) -- no new EC3 evaluation needed,
since the CSV already covers the full combinatorial space.

As a side effect, this also directly answers "does the true EC3+cost
optimum prefer higher grade at higher demand" from ground truth alone,
independent of any RL result -- see `grade_vs_demand_from_ground_truth()`.
This is worth reporting in the paper regardless of how the RL arms turn
out, since it's the actual mechanics-derived answer the RL policy is
trying to approximate.

USAGE
------
    python research/scripts/evaluate.py \\
        --model_path research/models/arm_B_lagrangian/final_model \\
        --vecnorm_path research/models/arm_B_lagrangian/vecnormalize.pkl \\
        --algo ppo --economy_metric cost --run_name arm_B_lagrangian \\
        --out_csv research/results/arm_B_lagrangian_eval.csv

    python research/scripts/evaluate.py --ga_baseline --economy_metric cost \\
        --run_name ga_baseline --out_csv research/results/ga_baseline_eval.csv \\
        --n_contexts 100   # subsample for speed; omit for the full 745-context set
================================================================
"""

import argparse
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv


# ================================================================
# Ground truth
# ================================================================
def load_ground_truth(csv_path="pretrain_data/ec3_optimal_designs.csv"):
    return pd.read_csv(csv_path)


def ground_truth_optimum(df: pd.DataFrame, economy_metric: str) -> pd.DataFrame:
    """One row per (span_m, load_kNm): the best (grade, section_type, geometry)
    among all combinations tested at that context, i.e. the TRUE optimum the
    RL agent / GA are being compared against."""
    idx = df.groupby(["span_m", "load_kNm"])[economy_metric].idxmin()
    return df.loc[idx].reset_index(drop=True)


def grade_vs_demand_from_ground_truth(df: pd.DataFrame, economy_metric: str = "cost") -> pd.DataFrame:
    """Direct, RL-independent evidence: at the TRUE optimum, does selected
    grade correlate with demand (span*load, a proxy for design moment)?
    Report this in the paper's Results section regardless of RL outcome --
    it's the ground truth the RL policy is trying to approximate, and it
    settles the "is grade-appropriate selection even physically justified
    under this cost model" question independent of any learned policy."""
    opt = ground_truth_optimum(df, economy_metric)
    opt["demand_proxy"] = opt["span_m"] * opt["load_kNm"]
    return opt[["span_m", "load_kNm", "demand_proxy", "grade", "section_type", economy_metric]].sort_values("demand_proxy")


# ================================================================
# Policy loading (algorithm-agnostic: PPO, DDPG, TD3 all share predict())
# ================================================================
def load_policy(model_path: str, algo: str):
    from stable_baselines3 import PPO, DDPG, TD3
    cls = {"ppo": PPO, "ddpg": DDPG, "td3": TD3}[algo]
    model = cls.load(model_path)

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy_fn


# ================================================================
# Running one policy episode at a FORCED (span, load) context
# ================================================================
def run_policy_episode(env: HSSBeamEnv, policy_fn, span_m: float, load_kNm: float,
                        storey: int = 20, max_steps: int = 40, seed: int = 0):
    obs, _ = env.reset(seed=seed)
    # Override the curriculum-sampled context with the EXACT ground-truth
    # context we want to compare against. use_storey_load_scaling is left
    # at the env's configured value, but we bypass it here by directly
    # setting the post-scaling load the ground-truth CSV already represents
    # (the CSV has no storey column -- it is span/load only), so we disable
    # storey scaling for this forced context to avoid double-applying it.
    env.use_storey_load_scaling = False
    env.span = float(span_m) * 1000.0
    env.load = float(load_kNm)
    env.storey = int(storey)
    obs = env._get_obs()

    info = None
    for t in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


# ================================================================
# Aggregate evaluation against ground truth
# ================================================================
def evaluate_policy_vs_ground_truth(policy_fn, economy_metric: str, ground_truth_csv: str,
                                     n_contexts: int | None = None, seed: int = 0,
                                     reward_mode_for_env: str = "lagrangian"):
    df = load_ground_truth(ground_truth_csv)
    opt = ground_truth_optimum(df, economy_metric)
    if n_contexts is not None and n_contexts < len(opt):
        opt = opt.sample(n=n_contexts, random_state=seed).reset_index(drop=True)

    env = HSSBeamEnv(reward_mode=reward_mode_for_env, economy_metric=economy_metric)

    rows = []
    t0 = time.time()
    for i, r in opt.iterrows():
        info = run_policy_episode(env, policy_fn, r["span_m"], r["load_kNm"], seed=seed + i)
        agent_economy = info[economy_metric]
        optimal_economy = r[economy_metric]
        gap = (agent_economy - optimal_economy) / optimal_economy if info["feasible"] else np.nan
        rows.append(dict(
            span_m=r["span_m"], load_kNm=r["load_kNm"],
            optimal_economy=optimal_economy, optimal_grade=r["grade"], optimal_type=r["section_type"],
            agent_economy=agent_economy, agent_grade=info["fy"], agent_type=info["section_type"],
            agent_util=info["utilization"], feasible=info["feasible"],
            in_target_band=info["in_target_band"], gap=gap,
        ))
    wall_time = time.time() - t0
    result = pd.DataFrame(rows)
    return result, wall_time


def evaluate_ga_vs_ground_truth(economy_metric: str, ground_truth_csv: str,
                                 n_contexts: int | None = None, seed: int = 0,
                                 pop_size: int = 60, n_generations: int = 80):
    from research.scripts.ga_baseline import ga_design
    df = load_ground_truth(ground_truth_csv)
    opt = ground_truth_optimum(df, economy_metric)
    if n_contexts is not None and n_contexts < len(opt):
        opt = opt.sample(n=n_contexts, random_state=seed).reset_index(drop=True)

    rows = []
    t0 = time.time()
    for i, r in opt.iterrows():
        ga_result = ga_design(span_mm=r["span_m"] * 1000.0, load_kNm=r["load_kNm"], storey=20,
                               economy_metric=economy_metric, pop_size=pop_size,
                               n_generations=n_generations, seed=seed + i)
        agent_economy = ga_result[economy_metric]
        optimal_economy = r[economy_metric]
        gap = (agent_economy - optimal_economy) / optimal_economy if ga_result["feasible"] else np.nan
        rows.append(dict(
            span_m=r["span_m"], load_kNm=r["load_kNm"],
            optimal_economy=optimal_economy, optimal_grade=r["grade"], optimal_type=r["section_type"],
            agent_economy=agent_economy, agent_grade=ga_result["fy"], agent_type=ga_result["section_type"],
            agent_util=ga_result["util"], feasible=ga_result["feasible"],
            wall_time_s=ga_result["wall_time_s"], gap=gap,
        ))
    wall_time = time.time() - t0
    result = pd.DataFrame(rows)
    return result, wall_time


def summarize(result: pd.DataFrame, wall_time: float, label: str) -> dict:
    feasible_mask = result["feasible"]
    gaps = result.loc[feasible_mask, "gap"].dropna()
    return dict(
        label=label,
        n_contexts=len(result),
        feasibility_rate=float(feasible_mask.mean()),
        gap_mean=float(gaps.mean()) if len(gaps) else np.nan,
        gap_median=float(gaps.median()) if len(gaps) else np.nan,
        gap_p90=float(gaps.quantile(0.90)) if len(gaps) else np.nan,
        gap_p95=float(gaps.quantile(0.95)) if len(gaps) else np.nan,
        wall_time_s_total=wall_time,
        wall_time_s_per_context=wall_time / max(len(result), 1),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--algo", choices=["ppo", "ddpg", "td3"], default="ppo")
    p.add_argument("--ga_baseline", action="store_true")
    p.add_argument("--economy_metric", choices=["mass", "cost", "co2"], default="cost")
    p.add_argument("--ground_truth_csv", type=str, default="pretrain_data/ec3_optimal_designs.csv")
    p.add_argument("--n_contexts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--ga_pop_size", type=int, default=60)
    p.add_argument("--ga_n_generations", type=int, default=80)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    if args.ga_baseline:
        result, wall_time = evaluate_ga_vs_ground_truth(
            args.economy_metric, args.ground_truth_csv, args.n_contexts, args.seed,
            args.ga_pop_size, args.ga_n_generations)
    else:
        assert args.model_path, "--model_path required unless --ga_baseline"
        policy_fn = load_policy(args.model_path, args.algo)
        result, wall_time = evaluate_policy_vs_ground_truth(
            policy_fn, args.economy_metric, args.ground_truth_csv, args.n_contexts, args.seed)

    result.to_csv(args.out_csv, index=False)
    summary = summarize(result, wall_time, args.run_name)
    summary_path = args.out_csv.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== {args.run_name} ===")
    for k, v in summary.items():
        print(f"  {k:28s}: {v}")
    print(f"\nPer-context results -> {args.out_csv}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
