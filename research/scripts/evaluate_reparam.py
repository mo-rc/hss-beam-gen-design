"""
research/scripts/evaluate_reparam.py
================================================================
Evaluates a trained HSSReparamEnv policy (from train_reparam_es.py)
against the SAME objective-specific ground truth and SAME gap
definition as evaluate.py's evaluate_policy_vs_ground_truth, so its
number is directly comparable to e4_ppo_seed*_summary.csv and
e4_nonrl_baselines_summary.csv (ga row). Deterministic action = the
policy network's raw output (no exploration noise), matching how
evaluate.py uses model.predict(deterministic=True) for the raw arm.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_reparam_env import HSSReparamEnv
from research.scripts.evaluate import (
    ground_truth_path_for_metric, ground_truth_optimum, ground_truth_optimum_all_metrics,
    load_ground_truth, summarize,
)
from research.scripts.train_reparam_es import forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy_path", required=True)
    p.add_argument("--economy_metric", default="cost")
    p.add_argument("--ground_truth_dir", default="research/pretrain_data_corrected")
    p.add_argument("--n_contexts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", required=True)
    a = p.parse_args()

    payload = np.load(a.policy_path, allow_pickle=True).item()
    theta, hidden = payload["theta"], payload["hidden"]

    df = load_ground_truth(ground_truth_path_for_metric(a.economy_metric, a.ground_truth_dir))
    opt = ground_truth_optimum(df, a.economy_metric)
    gt_all = ground_truth_optimum_all_metrics(a.ground_truth_dir)
    if a.n_contexts is not None and a.n_contexts < len(opt):
        opt = opt.sample(n=a.n_contexts, random_state=a.seed).reset_index(drop=True)

    env = HSSReparamEnv(reward_mode="feasibility_gated", economy_metric=a.economy_metric)

    rows = []
    t0 = time.time()
    for i, r in opt.iterrows():
        obs, _ = env.reset(seed=a.seed + i)
        env.use_storey_load_scaling = False
        env.span = float(r["span_m"]) * 1000.0
        env.load = float(r["load_kNm"])
        obs = env._get_obs()
        action = forward(theta, hidden, obs[None, :])[0]
        _, reward, _, _, info = env.step(action)

        agent_economy = info[a.economy_metric]
        optimal_economy = r[a.economy_metric]
        gap = (agent_economy - optimal_economy) / optimal_economy if info["feasible"] else np.nan

        gt_key = (r["span_m"], r["load_kNm"])
        secondary_gaps = {}
        for m in ["mass", "cost", "co2"]:
            if m == a.economy_metric or not info["feasible"]:
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
    summary = summarize(result, wall_time, "reparam_es")
    print(summary)

    os.makedirs(os.path.dirname(a.out_csv), exist_ok=True)
    result.to_csv(a.out_csv, index=False)
    pd.DataFrame([summary]).to_csv(a.out_csv.replace(".csv", "_summary.csv"), index=False)


if __name__ == "__main__":
    main()
