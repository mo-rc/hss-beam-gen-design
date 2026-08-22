"""
research/scripts/regenerate_ground_truth.py
================================================================
Replaces pretrain_data/ec3_optimal_designs.csv with THREE objective-
specific ground-truth datasets (mass/cost/co2), using the fixed EC3 code
and a GA-based search instead of the original coarse, pre-fix grid.

WHY THREE SEPARATE FILES, NOT ONE (pre-Experiment-1 audit fix)
------------------------------------------------------------------
The first version of this script (and the original
generate_ec3_pretrain_dataset.py before it) optimised geometry for MASS
only, then reported the resulting design's cost and CO2 alongside it "for
free". That is not objective-specific ground truth: the mass-optimal
geometry for a given (span, load, grade, type) is generally NOT the same
geometry that minimises cost or CO2 for that same combination (fabrication
cost/CO2 factors and material unit prices weight grade and section_type
differently than mass alone does). Using the mass-optimal design's cost/
CO2 values as if they were the cost-optimal/CO2-optimal ground truth
silently biases every reported gap for those two metrics.

This script now runs THREE independent GA searches per (span, load,
grade, section_type) context -- one per economy_metric -- and writes
three separate CSVs (ec3_optimal_designs_mass.csv, _cost.csv, _co2.csv),
each containing that metric's own optimal geometry. research/scripts/
evaluate.py's ground-truth loading was updated to match (see that file).

Same context grid as before (12 spans x 12 loads x 6 grades x 2 section
types = 1,728 contexts per metric, 5,184 GA searches total), same CSV
schema per file, so evaluate.py / grade_policy_analysis.py need only a
path change, not a logic change.

RUNTIME: ~0.42s per (context, restart) measured during the audit. At
pop_size=50, n_generations=80, n_restarts=2: ~1,728 x 3 x 2 x 0.42s =
~72 minutes single-threaded for the full 3-metric regeneration.

USAGE
------
    python research/scripts/regenerate_ground_truth.py \\
        --out_dir pretrain_data --pop_size 50 --n_generations 80 --n_restarts 2
        # writes pretrain_data/ec3_optimal_designs_{mass,cost,co2}.csv

    # Small-scale validation run:
    python research/scripts/regenerate_ground_truth.py \\
        --out_dir /tmp/gt_smoketest --n_spans 3 --n_loads 3 \\
        --pop_size 30 --n_generations 40 --n_restarts 1
================================================================
"""

import argparse
import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.scripts.ga_baseline import ga_design_fixed_grade

GRADES = [355, 460, 500, 550, 620, 690]
SECTION_TYPES = ["rolled", "welded"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="pretrain_data")
    p.add_argument("--span_min_m", type=float, default=6.0)
    p.add_argument("--span_max_m", type=float, default=15.0)
    p.add_argument("--load_min", type=float, default=20.0)
    p.add_argument("--load_max", type=float, default=140.0)
    p.add_argument("--n_spans", type=int, default=12)
    p.add_argument("--n_loads", type=int, default=12)
    p.add_argument("--pop_size", type=int, default=50)
    p.add_argument("--n_generations", type=int, default=80)
    p.add_argument("--n_restarts", type=int, default=2)
    p.add_argument("--metrics", nargs="+", default=["mass", "cost", "co2"],
                    choices=["mass", "cost", "co2"])
    p.add_argument("--grade_filter", type=float, default=None,
                    help="If given, only process this single grade -- for chunked execution "
                         "within a sandbox's per-call time limit. Writes to a per-chunk file "
                         "(ec3_optimal_designs_{metric}_g{grade}.csv); merge chunks afterward "
                         "with --merge_only.")
    p.add_argument("--merge_only", action="store_true",
                    help="Skip computation; just merge existing per-grade chunk files for "
                         "each metric in --metrics into the final ec3_optimal_designs_{metric}.csv.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge_only:
        for economy_metric in args.metrics:
            chunk_paths = sorted(glob.glob(os.path.join(args.out_dir, f"ec3_optimal_designs_{economy_metric}_g*.csv")))
            if not chunk_paths:
                print(f"[{economy_metric}] no chunk files found, skipping")
                continue
            dfs = [pd.read_csv(p_) for p_ in chunk_paths]
            merged = pd.concat(dfs, ignore_index=True)
            out_path = os.path.join(args.out_dir, f"ec3_optimal_designs_{economy_metric}.csv")
            merged.to_csv(out_path, index=False)
            print(f"[{economy_metric}] merged {len(chunk_paths)} chunks ({len(merged)} rows) -> {out_path}")
        return

    spans = np.linspace(args.span_min_m, args.span_max_m, args.n_spans)
    loads = np.linspace(args.load_min, args.load_max, args.n_loads)
    grades_to_run = [args.grade_filter] if args.grade_filter is not None else GRADES
    contexts = [(s, l, g, t) for s in spans for l in loads for g in grades_to_run for t in SECTION_TYPES]

    total_estimate_min = len(contexts) * len(args.metrics) * args.n_restarts * 0.42 / 60
    print(f"Regenerating ground truth for metrics={args.metrics}, grades={grades_to_run}")
    print(f"{len(contexts)} contexts x {len(args.metrics)} metrics x {args.n_restarts} restarts, "
          f"pop={args.pop_size}, gens={args.n_generations}")
    print(f"Estimated runtime this call: ~{total_estimate_min:.1f} minutes\n")

    grand_t0 = time.time()
    for economy_metric in args.metrics:
        rows = []
        t0 = time.time()
        for i, (span_m, load_kNm, grade, section_type) in enumerate(contexts):
            best_result, best_val = None, np.inf
            for restart in range(args.n_restarts):
                result = ga_design_fixed_grade(
                    span_mm=span_m * 1000.0, load_kNm=load_kNm, storey=20,
                    grade=grade, section_type=section_type, economy_metric=economy_metric,
                    pop_size=args.pop_size, n_generations=args.n_generations,
                    seed=args.seed * 100000 + i * 10 + restart,
                )
                if result["feasible"] and result[economy_metric] < best_val:
                    best_val, best_result = result[economy_metric], result

            if best_result is None:
                continue

            rows.append(dict(
                span_m=span_m, load_kNm=load_kNm, grade=grade, section_type=section_type,
                h=best_result["h"], b=best_result["b"], tf=best_result["tf"], tw=best_result["tw"],
                util=best_result["util"], mass=best_result["mass"], cost=best_result["cost"],
                co2=best_result["co2"], governing=f"GA_{economy_metric}",
            ))

        df = pd.DataFrame(rows)
        if args.grade_filter is not None:
            out_path = os.path.join(args.out_dir, f"ec3_optimal_designs_{economy_metric}_g{int(args.grade_filter)}.csv")
        else:
            out_path = os.path.join(args.out_dir, f"ec3_optimal_designs_{economy_metric}.csv")
        df.to_csv(out_path, index=False)
        print(f"[{economy_metric}] grade={grades_to_run} done: {len(df)}/{len(contexts)} feasible, "
              f"{(time.time()-t0)/60:.2f} min -> {out_path}")

    print(f"\nThis call's total wall time: {(time.time() - grand_t0)/60:.2f} minutes")


if __name__ == "__main__":
    main()
