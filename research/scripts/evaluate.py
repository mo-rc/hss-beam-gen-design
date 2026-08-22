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
def load_ground_truth(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def ground_truth_path_for_metric(economy_metric: str, ground_truth_dir: str = "pretrain_data") -> str:
    """
    Post-audit convention: ground truth is OBJECTIVE-SPECIFIC. There are
    three files, ec3_optimal_designs_{mass,cost,co2}.csv, each produced by
    an independent GA search optimising THAT metric directly (see
    research/scripts/regenerate_ground_truth.py) -- not one file with
    cost/co2 read off a mass-optimal geometry, which was the pre-audit
    (biased) behaviour. This function is the single place that maps an
    economy_metric to its correct ground-truth file; every evaluation
    script should go through it rather than hardcoding a filename.
    """
    return os.path.join(ground_truth_dir, f"ec3_optimal_designs_{economy_metric}.csv")


def ground_truth_optimum(df: pd.DataFrame, economy_metric: str) -> pd.DataFrame:
    """One row per (span_m, load_kNm): the best (grade, section_type, geometry)
    among all combinations tested at that context, i.e. the TRUE optimum the
    RL agent / GA are being compared against. `df` MUST already be the
    metric-specific ground truth (loaded via `ground_truth_path_for_metric`),
    not a mass-optimised file being reused for a different metric."""
    idx = df.groupby(["span_m", "load_kNm"])[economy_metric].idxmin()
    return df.loc[idx].reset_index(drop=True)


def ground_truth_optimum_all_metrics(ground_truth_dir: str = "pretrain_data") -> dict:
    """Ground-truth optimum for EACH of mass/cost/co2 independently, keyed
    by (span_m, load_kNm) -> {metric: optimal_value}, each drawn from ITS
    OWN objective-specific file. Used so a policy trained on ONE
    economy_metric can still be scored on its incidental gap in the other
    two -- directly answers the supervisor's request for a 'composite-
    objective gap' without reintroducing an arbitrary-weight scalarisation
    (Comment #5's objection to weighted-sum reward design applies equally
    to a weighted-sum GAP metric), and without the pre-audit bug of
    reading cost/co2 off a mass-optimal geometry."""
    out = {}
    for metric in ["mass", "cost", "co2"]:
        path = ground_truth_path_for_metric(metric, ground_truth_dir)
        if not os.path.exists(path):
            continue
        df = load_ground_truth(path)
        opt = ground_truth_optimum(df, metric)
        for _, r in opt.iterrows():
            key = (r["span_m"], r["load_kNm"])
            out.setdefault(key, {})[metric] = r[metric]
    return out


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
                        storey: int = 20, max_steps: int = 40, seed: int = 0,
                        return_best_feasible: bool = True):
    """
    Pre-Experiment-1 audit fix (supervisor Comment #14: "retain the best
    feasible design encountered in the episode"): the termination rule
    (3 consecutive steps in the 0.90-1.05 target band) is a TRAINING
    convenience, not a guarantee that the episode's LAST step is the best
    (or even a feasible) design. A policy can visit a genuinely good,
    code-compliant design mid-episode and then continue exploring/refining
    past it, ending on something worse or infeasible. Reporting only the
    terminal info dict would then understate the policy's true achievable
    performance -- and understates it in a way that has nothing to do with
    design quality, only with where the episode happened to stop.

    With return_best_feasible=True (the default, and what all evaluation
    scripts should use for reported results), this tracks every step's
    info dict and returns the one with the lowest economy_metric value
    AMONG FEASIBLE steps, matching how a real generative-design tool would
    actually be used ("show me the best candidate you found"), not "trust
    whatever the trajectory happened to end on". If no step in the episode
    was feasible, returns the terminal step's info (honestly infeasible).

    Set return_best_feasible=False to get the raw terminal-step behaviour
    (useful only for diagnosing termination-rule dynamics themselves, not
    for reporting economy/feasibility results).
    """
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
    best_info = None
    best_economy = np.inf
    for t in range(max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if return_best_feasible and info["feasible"]:
            econ = info[env.economy_metric]
            if econ < best_economy:
                best_economy = econ
                best_info = info
        if terminated or truncated:
            break

    if return_best_feasible and best_info is not None:
        return best_info
    return info


# ================================================================
# Aggregate evaluation against ground truth
# ================================================================
def evaluate_policy_vs_ground_truth(policy_fn, economy_metric: str, ground_truth_dir: str,
                                     n_contexts: int | None = None, seed: int = 0,
                                     reward_mode_for_env: str = "lagrangian"):
    df = load_ground_truth(ground_truth_path_for_metric(economy_metric, ground_truth_dir))
    opt = ground_truth_optimum(df, economy_metric)
    gt_all = ground_truth_optimum_all_metrics(ground_truth_dir)
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

        # Secondary-metric gaps: how the policy does on mass/cost/co2 it was
        # NOT directly optimising, relative to EACH metric's own ground-truth
        # optimum (not the primary metric's optimal design) -- see
        # ground_truth_optimum_all_metrics() docstring.
        gt_key = (r["span_m"], r["load_kNm"])
        secondary_gaps = {}
        for m in ["mass", "cost", "co2"]:
            if m == economy_metric or not info["feasible"]:
                continue
            gt_m = gt_all.get(gt_key, {}).get(m)
            if gt_m:
                secondary_gaps[f"gap_{m}"] = (info[m] - gt_m) / gt_m

        rows.append(dict(
            span_m=r["span_m"], load_kNm=r["load_kNm"],
            optimal_economy=optimal_economy, optimal_grade=r["grade"], optimal_type=r["section_type"],
            agent_economy=agent_economy, agent_grade=info["fy"], agent_type=info["section_type"],
            agent_util=info["utilization"], feasible=info["feasible"],
            in_target_band=info["in_target_band"], gap=gap, **secondary_gaps,
        ))
    wall_time = time.time() - t0
    result = pd.DataFrame(rows)
    return result, wall_time


def evaluate_ga_vs_ground_truth(economy_metric: str, ground_truth_dir: str,
                                 n_contexts: int | None = None, seed: int = 0,
                                 pop_size: int = 60, n_generations: int = 80):
    from research.scripts.ga_baseline import ga_design
    df = load_ground_truth(ground_truth_path_for_metric(economy_metric, ground_truth_dir))
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
    summary = dict(
        label=label,
        n_contexts=len(result),
        feasibility_rate=float(feasible_mask.mean()),
        gap_mean=float(gaps.mean()) if len(gaps) else np.nan,
        gap_median=float(gaps.median()) if len(gaps) else np.nan,
        gap_std=float(gaps.std()) if len(gaps) else np.nan,
        gap_p90=float(gaps.quantile(0.90)) if len(gaps) else np.nan,
        gap_p95=float(gaps.quantile(0.95)) if len(gaps) else np.nan,
        gap_worst=float(gaps.max()) if len(gaps) else np.nan,
        pct_within_1pct=float((gaps.abs() <= 0.01).mean()) if len(gaps) else np.nan,
        pct_within_5pct=float((gaps.abs() <= 0.05).mean()) if len(gaps) else np.nan,
        pct_within_10pct=float((gaps.abs() <= 0.10).mean()) if len(gaps) else np.nan,
        wall_time_s_total=wall_time,
        wall_time_s_per_context=wall_time / max(len(result), 1),
    )
    # Secondary-metric gaps (see evaluate_policy_vs_ground_truth docstring),
    # only present for RL-policy evaluations, not the GA baseline path.
    for col in [c for c in result.columns if c.startswith("gap_") and c != "gap"]:
        vals = result.loc[feasible_mask, col].dropna()
        if len(vals):
            summary[f"{col}_mean"] = float(vals.mean())
            summary[f"{col}_median"] = float(vals.median())
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--algo", choices=["ppo", "ddpg", "td3"], default="ppo")
    p.add_argument("--ga_baseline", action="store_true")
    p.add_argument("--economy_metric", choices=["mass", "cost", "co2"], default="cost")
    p.add_argument("--ground_truth_dir", type=str, default="pretrain_data",
                    help="Directory containing ec3_optimal_designs_{mass,cost,co2}.csv "
                         "(objective-specific ground truth; see regenerate_ground_truth.py)")
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
            args.economy_metric, args.ground_truth_dir, args.n_contexts, args.seed,
            args.ga_pop_size, args.ga_n_generations)
    else:
        assert args.model_path, "--model_path required unless --ga_baseline"
        policy_fn = load_policy(args.model_path, args.algo)
        result, wall_time = evaluate_policy_vs_ground_truth(
            policy_fn, args.economy_metric, args.ground_truth_dir, args.n_contexts, args.seed)

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
