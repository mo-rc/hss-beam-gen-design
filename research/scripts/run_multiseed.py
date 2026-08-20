"""
research/scripts/run_multiseed.py
================================================================
Launches N seeds for a given arm (sequentially, to keep peak resource
usage predictable on a single machine -- edit to launch in parallel
background processes if you have the cores/GPUs to spare), then
aggregates results and runs significance tests between arms.

This directly implements the supervisor's Comment #10 (multi-seed
replication) and gives the paper real statistics ("X vs Y, p=...")
instead of single-run point estimates.

USAGE
------
  # Train 5 seeds of the primary Lagrangian arm:
  python research/scripts/run_multiseed.py train \\
      --reward_mode lagrangian --economy_metric cost \\
      --run_prefix arm_B_lagrangian --seeds 42 43 44 45 46 --timesteps 1000000

  # Evaluate all 5 seeds against ground truth and aggregate:
  python research/scripts/run_multiseed.py evaluate \\
      --run_prefix arm_B_lagrangian --seeds 42 43 44 45 46 --economy_metric cost

  # Compare two arms statistically (e.g. Lagrangian vs. legacy shaped):
  python research/scripts/run_multiseed.py compare \\
      --arm_a research/results/arm_B_lagrangian_seed*_summary.json \\
      --arm_b research/results/arm_A_legacy_seed*_summary.json \\
      --metric gap_mean
================================================================
"""

import argparse
import glob
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from scipy import stats


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def cmd_train(args):
    for seed in args.seeds:
        run_name = f"{args.run_prefix}_seed{seed}"
        print(f"\n{'='*70}\nTraining {run_name} (reward_mode={args.reward_mode})\n{'='*70}")
        cmd = [
            sys.executable, os.path.join(THIS_DIR, "train.py"),
            "--reward_mode", args.reward_mode, "--economy_metric", args.economy_metric,
            "--run_name", run_name, "--seed", str(seed), "--timesteps", str(args.timesteps),
            "--n_envs", str(args.n_envs),
        ]
        subprocess.run(cmd, check=True)


def cmd_evaluate(args):
    for seed in args.seeds:
        run_name = f"{args.run_prefix}_seed{seed}"
        model_path = os.path.join("research", "models", run_name, "final_model")
        out_csv = os.path.join("research", "results", f"{run_name}_eval.csv")
        print(f"\nEvaluating {run_name}...")
        cmd = [
            sys.executable, os.path.join(THIS_DIR, "evaluate.py"),
            "--model_path", model_path, "--algo", args.algo,
            "--economy_metric", args.economy_metric, "--run_name", run_name,
            "--out_csv", out_csv,
        ]
        subprocess.run(cmd, check=True)

    # Aggregate across seeds
    summaries = []
    for seed in args.seeds:
        run_name = f"{args.run_prefix}_seed{seed}"
        summary_path = os.path.join("research", "results", f"{run_name}_eval_summary.json")
        with open(summary_path) as f:
            s = json.load(f)
        s["seed"] = seed
        summaries.append(s)

    df = pd.DataFrame(summaries)
    agg_path = os.path.join("research", "results", f"{args.run_prefix}_multiseed_summary.csv")
    df.to_csv(agg_path, index=False)

    print(f"\n=== {args.run_prefix}: {len(args.seeds)}-seed summary ===")
    for col in ["feasibility_rate", "gap_mean", "gap_median", "gap_p90", "gap_p95"]:
        if col in df.columns:
            print(f"  {col:20s}: mean={df[col].mean():.4f}  std={df[col].std():.4f}  "
                  f"(seeds: {df[col].round(4).tolist()})")
    print(f"\nPer-seed summaries -> {agg_path}")


def cmd_compare(args):
    files_a = sorted(glob.glob(args.arm_a))
    files_b = sorted(glob.glob(args.arm_b))
    assert files_a and files_b, "No files matched --arm_a / --arm_b glob patterns"

    def load_metric(files, metric):
        vals = []
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            vals.append(d[metric])
        return np.array(vals)

    a = load_metric(files_a, args.metric)
    b = load_metric(files_b, args.metric)

    print(f"Arm A ({len(a)} seeds): mean={a.mean():.4f} std={a.std():.4f}  values={a.tolist()}")
    print(f"Arm B ({len(b)} seeds): mean={b.mean():.4f} std={b.std():.4f}  values={b.tolist()}")

    # Welch's t-test (unequal variance) -- standard choice when comparing
    # RL seed distributions, which frequently violate equal-variance
    # assumptions. Also report Mann-Whitney U as a non-parametric check,
    # since RL seed distributions are often non-normal (bimodal seed
    # failures are common) -- report BOTH, note if they disagree.
    t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
    try:
        u_stat, u_p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        u_stat, u_p = np.nan, np.nan

    print(f"\nWelch's t-test:      t={t_stat:.3f}  p={t_p:.4f}")
    print(f"Mann-Whitney U test: U={u_stat:.3f}  p={u_p:.4f}")
    if (t_p < 0.05) != (u_p < 0.05):
        print("\n  NOTE: the two tests disagree on significance at alpha=0.05 -- "
              "report both in the paper rather than picking the favourable one, "
              "and consider whether n_seeds is large enough for either to be reliable.")
    else:
        verdict = "SIGNIFICANT" if t_p < 0.05 else "NOT significant"
        print(f"\n  Both tests agree: difference is {verdict} at alpha=0.05.")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--reward_mode", required=True)
    pt.add_argument("--economy_metric", default="cost")
    pt.add_argument("--run_prefix", required=True)
    pt.add_argument("--seeds", type=int, nargs="+", required=True)
    pt.add_argument("--timesteps", type=int, default=1_000_000)
    pt.add_argument("--n_envs", type=int, default=8)
    pt.set_defaults(func=cmd_train)

    pe = sub.add_parser("evaluate")
    pe.add_argument("--run_prefix", required=True)
    pe.add_argument("--seeds", type=int, nargs="+", required=True)
    pe.add_argument("--economy_metric", default="cost")
    pe.add_argument("--algo", default="ppo", choices=["ppo", "ddpg", "td3"])
    pe.set_defaults(func=cmd_evaluate)

    pc = sub.add_parser("compare")
    pc.add_argument("--arm_a", required=True, help="glob pattern matching arm A's *_summary.json files")
    pc.add_argument("--arm_b", required=True, help="glob pattern matching arm B's *_summary.json files")
    pc.add_argument("--metric", default="gap_mean")
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
