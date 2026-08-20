"""
research/scripts/generalization_test.py
================================================================
Every result so far (exp54b's validation, this project's evaluate.py) is
IN-DISTRIBUTION: spans 6-15m, loads 20-140 kN/m -- exactly the training
envelope. This script evaluates a trained policy OUTSIDE that envelope
and characterizes WHERE and HOW it fails, which is a standard and
expected thing for a top-journal reviewer to ask for and which costs
almost nothing to add (same policy, same env class, different test
points) yet meaningfully strengthens the paper's honesty/rigor.

Three extrapolation directions are tested independently, since a policy
might generalize well in one and fail in another:
    - span extrapolation:  16-22m (training max was 15m)
    - load extrapolation:  150-260 kN/m (training max was ~140 kN/m post
                            storey-scaling; 210 kN/m raw was the absolute
                            upper bound of the ORIGINAL env's LOAD_MAX
                            constant, so this also stress-tests near and
                            somewhat past that boundary)
    - joint extrapolation:  both span AND load pushed simultaneously
                            (compounding effect, more representative of a
                            genuinely out-of-scope real design case than
                            either alone)

For each region, reports feasibility rate and (where a policy succeeds)
whether the DESIGN STRATEGY still makes structural sense -- e.g. does
utilization stay near the training-time target band, does grade
selection still follow a sane pattern, or does the policy silently
produce nonsensical designs while still reporting "success"? This last
check matters because a policy can satisfy `in_target_band` (a training
convenience, not a correctness guarantee) on inputs it was never designed
to see.

USAGE
------
    python research/scripts/generalization_test.py \\
        --model_path research/models/arm_B_lagrangian_seed42/final_model \\
        --algo ppo --economy_metric cost \\
        --out_csv research/results/arm_B_generalization.csv
================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.scripts.evaluate import load_policy, run_policy_episode


TRAIN_SPAN_MAX_M = 15.0
TRAIN_LOAD_MAX_KNM = 140.0  # base_load sampling cap in reset()

REGIONS = {
    "in_distribution":     dict(span_range=(6.0, 15.0),  load_range=(20.0, 140.0)),
    "span_extrapolation":  dict(span_range=(16.0, 22.0), load_range=(20.0, 140.0)),
    "load_extrapolation":  dict(span_range=(6.0, 15.0),  load_range=(150.0, 260.0)),
    "joint_extrapolation": dict(span_range=(16.0, 22.0), load_range=(150.0, 260.0)),
}


def sample_grid(span_range, load_range, n_span=10, n_load=10):
    spans = np.linspace(*span_range, n_span)
    loads = np.linspace(*load_range, n_load)
    return [(s, l) for s in spans for l in loads]


def evaluate_region(policy_fn, env, span_range, load_range, seed, economy_metric):
    contexts = sample_grid(span_range, load_range)
    rows = []
    for i, (span_m, load_kNm) in enumerate(contexts):
        info = run_policy_episode(env, policy_fn, span_m, load_kNm, seed=seed + i)
        rows.append(dict(
            span_m=span_m, load_kNm=load_kNm, feasible=info["feasible"],
            in_target_band=info["in_target_band"], util=info["utilization"],
            grade=info["fy"], section_type=info["section_type"], economy=info[economy_metric],
            section_class=info["ec3"]["section_class"],
        ))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--algo", choices=["ppo", "ddpg", "td3"], default="ppo")
    p.add_argument("--economy_metric", choices=["mass", "cost", "co2"], default="cost")
    p.add_argument("--reward_mode_for_env", default="lagrangian")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    policy_fn = load_policy(args.model_path, args.algo)
    env = HSSBeamEnv(reward_mode=args.reward_mode_for_env, economy_metric=args.economy_metric)

    all_rows = []
    print(f"{'region':22s} {'feasible%':>10s} {'in_band%':>10s} {'mean_util':>10s} {'n':>5s}")
    for region_name, ranges in REGIONS.items():
        df = evaluate_region(policy_fn, env, ranges["span_range"], ranges["load_range"],
                              args.seed, args.economy_metric)
        df["region"] = region_name
        all_rows.append(df)
        print(f"{region_name:22s} {df.feasible.mean()*100:9.1f}% {df.in_target_band.mean()*100:9.1f}% "
              f"{df.util.mean():10.3f} {len(df):5d}")

    result = pd.concat(all_rows, ignore_index=True)
    result.to_csv(args.out_csv, index=False)

    print(f"\nPer-context generalization results -> {args.out_csv}")
    print("\nDegradation relative to in-distribution baseline:")
    baseline_feas = result[result.region == "in_distribution"].feasible.mean()
    for region_name in REGIONS:
        if region_name == "in_distribution":
            continue
        feas = result[result.region == region_name].feasible.mean()
        print(f"  {region_name:22s}: {feas*100:5.1f}% feasible "
              f"({(feas - baseline_feas)*100:+.1f} pts vs in-distribution)")


if __name__ == "__main__":
    main()
