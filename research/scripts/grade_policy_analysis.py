"""
research/scripts/grade_policy_analysis.py
================================================================
Produces the mass/cost/CO2 grade-selection comparison directly from the
brute-force ground truth (no RL involved) -- this is the evidence behind
the finding that "does HSS become optimal at higher demand" has a
different, economy-metric-dependent answer, and should be reported as a
Results-section table/figure regardless of how the RL arms perform.

Also computes, for each economy metric, the correlation between demand
(span_m * load_kNm, a proxy for design moment magnitude) and selected
grade among TRUE optima -- for the RL comparison, this same function can
be pointed at an evaluated policy's output (research/scripts/evaluate.py's
per-context CSV, using `agent_grade` instead of `grade`) to check whether
the trained policy's grade-selection pattern matches the correct ground-
truth pattern for whichever economy_metric it was trained on.

USAGE
------
    python research/scripts/grade_policy_analysis.py \\
        --ground_truth_csv pretrain_data/ec3_optimal_designs.csv \\
        --out_dir research/results/grade_policy_analysis
================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.scripts.evaluate import load_ground_truth, ground_truth_optimum


def analyze_metric(df: pd.DataFrame, economy_metric: str) -> dict:
    opt = ground_truth_optimum(df, economy_metric)
    opt = opt.copy()
    opt["demand_proxy"] = opt["span_m"] * opt["load_kNm"]

    grade_counts = opt["grade"].value_counts().sort_index()
    demand_by_grade = opt.groupby("grade")["demand_proxy"].agg(["min", "max", "count"])
    spearman_corr = opt["demand_proxy"].corr(opt["grade"], method="spearman")

    # Monotonicity check: sort by demand, is grade non-decreasing (within
    # noise)? Report the fraction of adjacent pairs that are consistent
    # (grade doesn't decrease as demand increases) as a simple monotonicity
    # score, rather than assuming strict monotonicity is even the right
    # ground-truth expectation (Concern #21 from the supervisor comments --
    # deflection/compactness governance can legitimately make monotonicity
    # NOT hold, and this script reports that honestly instead of assuming
    # it away).
    sorted_opt = opt.sort_values("demand_proxy")
    grade_seq = sorted_opt["grade"].to_numpy()
    non_decreasing = np.diff(grade_seq) >= 0
    monotonicity_score = float(non_decreasing.mean()) if len(non_decreasing) else np.nan

    return dict(
        economy_metric=economy_metric,
        n_contexts=len(opt),
        grades_ever_optimal=sorted(grade_counts.index.tolist()),
        grade_counts=grade_counts.to_dict(),
        demand_range_by_grade=demand_by_grade.to_dict(orient="index"),
        spearman_demand_grade_corr=float(spearman_corr),
        monotonicity_score=monotonicity_score,
        governing_check_distribution=opt["governing"].value_counts().to_dict() if "governing" in opt.columns else {},
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground_truth_csv", type=str, default="pretrain_data/ec3_optimal_designs.csv")
    p.add_argument("--out_dir", type=str, default="research/results/grade_policy_analysis")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_ground_truth(args.ground_truth_csv)
    print(f"Loaded {len(df)} ground-truth rows, {df[['span_m','load_kNm']].drop_duplicates().shape[0]} "
          f"unique (span, load) contexts, grades {sorted(df.grade.unique())}\n")

    summary_rows = []
    for metric in ["mass", "cost", "co2"]:
        result = analyze_metric(df, metric)
        print(f"=== economy_metric = {metric} ===")
        print(f"  grades ever optimal      : {result['grades_ever_optimal']}")
        print(f"  grade counts             : {result['grade_counts']}")
        print(f"  spearman(demand, grade)  : {result['spearman_demand_grade_corr']:.3f}")
        print(f"  monotonicity score       : {result['monotonicity_score']:.3f}  "
              f"(fraction of demand-sorted adjacent pairs with non-decreasing grade)")
        print()
        summary_rows.append({k: v for k, v in result.items()
                              if k not in ("demand_range_by_grade", "governing_check_distribution")})

        opt = ground_truth_optimum(df, metric)
        opt["demand_proxy"] = opt["span_m"] * opt["load_kNm"]
        opt.sort_values("demand_proxy").to_csv(
            os.path.join(args.out_dir, f"ground_truth_optimum_{metric}.csv"), index=False)

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.out_dir, "summary_all_metrics.csv"), index=False)
    print(f"Per-metric ground-truth-optimum tables and summary saved to {args.out_dir}/")
    print("\n>>> KEY FINDING TO REPORT IN THE PAPER: grade-selection behaviour is fundamentally")
    print(">>> different depending on which economy metric is optimised. Train/evaluate RL")
    print(">>> policies against all three (--economy_metric mass|cost|co2) and check whether")
    print(">>> each policy's LEARNED grade-vs-demand pattern matches ITS OWN metric's ground")
    print(">>> truth pattern above -- that is a stronger and more falsifiable claim than a")
    print(">>> single 'the agent learned to prefer HSS' statement.")


if __name__ == "__main__":
    main()
