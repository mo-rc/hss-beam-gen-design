"""
Comment 5 -- aggregate out-of-distribution generalization results across the
5 headline seeds (e5_ppo_s{42..46}_anneal_linear).

Reads the per-design raw output already produced by generalization_test.py
(research/results/c2_generalization_seed{seed}.csv, one row per (span, load)
grid point x region) and aggregates across seeds, per region:
    - feasible % (mean, std, min, max across the 5 seeds)
    - in_target_band % (mean, std)
    - mean utilisation among feasible designs (mean, std)

This does NOT re-run the policy -- it only aggregates per-design output that
already exists on disk (from generalization_test.py, which does need torch/
stable-baselines3 and was run externally). Aggregation itself has no such
dependency.

Run from the repo root:
    PYTHONPATH=$PWD python research/scripts/c2_aggregate_generalization.py

Writes research/results/c2_generalization_summary.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

R = "research/results"
SEEDS = [42, 43, 44, 45, 46]
REGIONS = ["in_distribution", "span_extrapolation", "load_extrapolation", "joint_extrapolation"]


def main():
    per_seed = []
    for seed in SEEDS:
        path = os.path.join(R, f"c2_generalization_seed{seed}.csv")
        df = pd.read_csv(path)
        for region in REGIONS:
            sub = df[df.region == region]
            if len(sub) == 0:
                raise ValueError(f"seed {seed}: no rows for region {region} in {path}")
            feas = sub.feasible.astype(bool)
            per_seed.append(dict(
                seed=seed, region=region, n=len(sub),
                feasible_pct=feas.mean() * 100,
                in_band_pct=sub.in_target_band.astype(bool).mean() * 100,
                mean_util_feasible=sub.loc[feas, "util"].mean() if feas.any() else np.nan,
            ))
    per_seed_df = pd.DataFrame(per_seed)
    per_seed_df.to_csv(os.path.join(R, "c2_generalization_per_seed.csv"), index=False)

    out_rows = []
    for region in REGIONS:
        sub = per_seed_df[per_seed_df.region == region]
        out_rows.append(dict(
            region=region,
            n_seeds=len(sub),
            n_points_per_seed=int(sub.n.iloc[0]),
            feasible_pct_mean=sub.feasible_pct.mean(),
            feasible_pct_std=sub.feasible_pct.std(ddof=1),
            feasible_pct_min=sub.feasible_pct.min(),
            feasible_pct_max=sub.feasible_pct.max(),
            in_band_pct_mean=sub.in_band_pct.mean(),
            in_band_pct_std=sub.in_band_pct.std(ddof=1),
            mean_util_feasible_mean=sub.mean_util_feasible.mean(),
        ))
    out = pd.DataFrame(out_rows).set_index("region").reindex(REGIONS).reset_index()
    out_path = os.path.join(R, "c2_generalization_summary.csv")
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
