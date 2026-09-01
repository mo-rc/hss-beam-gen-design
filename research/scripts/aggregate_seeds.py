"""
research/scripts/aggregate_from_explicit_paths.py
================================================================
Avoids the failure mode found in chat: aggregate_seeds.py infers each
seed's eval-summary filename from --run_prefix + seed, which silently
matched a STALE file (from an earlier, unrelated run at a different
step count) when the run naming didn't exactly match what was
expected. This script takes explicit paths instead -- no inference,
no silent mismatch possible.

Usage:
    python research/scripts/aggregate_from_explicit_paths.py \
        --json_paths \
            research/results/gated_cost_merged_seed42_1p3M_eval_summary.json \
            research/results/gated_cost_merged_seed43_1p3M_eval_summary.json \
            research/results/gated_cost_merged_seed44_1p3M_eval_summary.json \
            research/results/gated_cost_merged_seed45_1p3M_eval_summary.json \
            research/results/gated_cost_merged_seed46_1p3M_eval_summary.json \
        --seeds 42 43 44 45 46 \
        --label gated_cost_merged_1p3M \
        --out_csv research/results/gated_cost_merged_1p3M_multiseed_summary.csv
        --compare_json_paths \
            research/results/gated_cost_merged_seed42_eval_summary.json \
            research/results/gated_cost_merged_seed43_eval_summary.json \
            research/results/gated_cost_merged_seed44_eval_summary.json \
            research/results/gated_cost_merged_seed45_eval_summary.json \
            research/results/gated_cost_merged_seed46_eval_summary.json

================================================================
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json_paths", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--compare_json_paths", nargs="+", default=None,
                    help="Optional: matching-order list of summary jsons from a PRIOR "
                         "run to compute Welch t-test / Mann-Whitney U against (e.g. the "
                         "1.0M-step results), so significance is computed automatically "
                         "rather than by hand.")
    args = p.parse_args()

    assert len(args.json_paths) == len(args.seeds), "one json path per seed, same order"
    for path in args.json_paths:
        assert os.path.exists(path), f"file does not exist: {path}"

    rows = []
    for seed, path in zip(args.seeds, args.json_paths):
        with open(path) as f:
            d = json.load(f)
        d["seed"] = seed
        d["source_file"] = path  # kept explicitly so the provenance is never ambiguous later
        rows.append(d)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    gaps = df["gap_mean_pct"].to_numpy()
    print(f"{args.label}: n_seeds={len(gaps)}")
    print(f"  mean={gaps.mean():.2f}  std={gaps.std(ddof=1):.2f}  "
          f"min={gaps.min():.2f}  max={gaps.max():.2f}")
    print(f"  per-seed: {dict(zip(args.seeds, np.round(gaps, 2)))}")

    if args.compare_json_paths is not None:
        assert len(args.compare_json_paths) == len(args.seeds)
        compare_gaps = []
        for path in args.compare_json_paths:
            assert os.path.exists(path), f"file does not exist: {path}"
            with open(path) as f:
                compare_gaps.append(json.load(f)["gap_mean_pct"])
        compare_gaps = np.array(compare_gaps)
        print(f"\n  comparison set: mean={compare_gaps.mean():.2f}  std={compare_gaps.std(ddof=1):.2f}")
        t, p_t = stats.ttest_ind(compare_gaps, gaps, equal_var=False)
        u, p_u = stats.mannwhitneyu(compare_gaps, gaps, alternative="two-sided")
        print(f"  Welch t-test:      t={t:.3f}  p={p_t:.4f}")
        print(f"  Mann-Whitney U:    U={u:.3f}  p={p_u:.4f}")
        if p_t > 0.10 and p_u > 0.10:
            print("  -> NOT statistically significant at n={} -- report as a trend, "
                  "not a proven effect.".format(len(gaps)))

    print(f"\nWritten -> {args.out_csv}")


if __name__ == "__main__":
    main()