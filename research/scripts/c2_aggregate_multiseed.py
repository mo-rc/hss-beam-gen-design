"""
Comment 2 -- aggregate the 5-seed headline result (e5_ppo_s{42..46}_anneal_linear).

Reads the per-seed evaluation summaries already produced by evaluate_with_repair.py
(research/results/e5_ppo_s{seed}_anneal_linear_curve_1000000_summary.csv, one row
per repair_mode: none / scale / scale+thin) and aggregates across the 5 seeds:
mean, sample std (ddof=1), 95% CI (t-distribution, n=5), best/median/worst seed,
mean of the per-seed gap_median, and feasibility rate.

This does NOT re-run evaluation -- it only aggregates numbers that already exist
on disk, so re-running is instant and has no dependency on stable-baselines3/torch.

Run from the repo root:
    PYTHONPATH=$PWD python research/scripts/c2_aggregate_multiseed.py

Writes research/results/c2_multiseed_summary.csv.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from scipy import stats

R = "research/results"
SEEDS = [42, 43, 44, 45, 46]
RUN_PREFIX = "e5_ppo_s{seed}_anneal_linear"
REPAIR_MODES = ["none", "scale", "scale+thin"]


def main():
    per_seed = []
    for seed in SEEDS:
        path = os.path.join(R, f"{RUN_PREFIX.format(seed=seed)}_curve_1000000_summary.csv")
        df = pd.read_csv(path)
        for mode in REPAIR_MODES:
            row = df[df.repair_mode == mode]
            if len(row) != 1:
                raise ValueError(f"seed {seed}, mode {mode}: expected 1 row in {path}, got {len(row)}")
            r = row.iloc[0]
            per_seed.append(dict(seed=seed, repair_mode=mode, gap_mean=r.gap_mean,
                                  gap_median=r.gap_median, feasibility=r.feasibility))
    per_seed_df = pd.DataFrame(per_seed)
    per_seed_df.to_csv(os.path.join(R, "c2_multiseed_per_seed.csv"), index=False)

    out_rows = []
    for mode in REPAIR_MODES:
        sub = per_seed_df[per_seed_df.repair_mode == mode]
        vals = sub.gap_mean.to_numpy()
        n = len(vals)
        mean = vals.mean()
        std = vals.std(ddof=1)
        ci_lo, ci_hi = stats.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
        best_i, worst_i = vals.argmin(), vals.argmax()
        out_rows.append(dict(
            repair_mode=mode,
            n_seeds=n,
            gap_mean_of_means=mean,
            gap_mean_std=std,
            gap_mean_ci95_lo=ci_lo,
            gap_mean_ci95_hi=ci_hi,
            best_seed=int(sub.seed.iloc[best_i]),
            best_gap_mean=vals[best_i],
            median_gap_mean=float(np.median(vals)),
            worst_seed=int(sub.seed.iloc[worst_i]),
            worst_gap_mean=vals[worst_i],
            mean_of_gap_median=sub.gap_median.mean(),
            feasibility_mean=sub.feasibility.mean(),
        ))
    out = pd.DataFrame(out_rows)
    out_path = os.path.join(R, "c2_multiseed_summary.csv")
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path}")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
