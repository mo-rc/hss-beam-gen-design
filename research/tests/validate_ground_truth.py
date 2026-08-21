"""
research/tests/validate_ground_truth.py
================================================================
Every optimality-gap number in this project is relative to
pretrain_data/ec3_optimal_designs.csv, produced by a 9^4-point coarse grid
search per (span, load, grade, section_type) context
(generate_ec3_pretrain_dataset.py). That file's own generation-time
comments note the coarse-to-fine refinement strategy had known quality
issues (mass-ordering inversions changed between refinement passes). If
the grid search is NOT actually close to the true (grade, type)-fixed
optimum, every downstream gap metric in this project is biased and the
paper's central "X% mean gap vs. optimum" claims are not defensible.

This script cross-checks a random sample of ground-truth rows against a
fine-grained, GRADE-AND-TYPE-FIXED genetic algorithm search (only the 4
continuous geometry genes vary -- an apples-to-apples comparison against
what the grid search was doing at each of its 1,728 fixed-grade/fixed-
type contexts) with a larger population and more generations than any
"per-instance" GA baseline elsewhere in this project needs, specifically
BECAUSE this run only has to happen once, ever, as a validation check,
not per training run.

DECISION RULE
--------------
For each sampled context, compute:
    improvement = (grid_mass - ga_mass) / grid_mass
If GA finds a MEANINGFULLY better design (improvement > 1%) on more than
a small fraction of sampled contexts, the grid ground truth is NOT
reliable enough to report gap metrics against, and should be regenerated
with a finer grid (or replaced by GA-refined values context-by-context,
which this script can also produce, see --write_refined).

USAGE
------
    python research/tests/validate_ground_truth.py --n_samples 30 --seed 0
================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.scripts.evaluate import load_ground_truth
from research.scripts.ga_baseline import ga_design_fixed_grade


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground_truth_csv", default="pretrain_data/ec3_optimal_designs.csv")
    p.add_argument("--n_samples", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pop_size", type=int, default=120)
    p.add_argument("--n_generations", type=int, default=200)
    p.add_argument("--n_restarts", type=int, default=3,
                    help="Independent GA restarts per context (different seeds); "
                         "keep the best, to reduce the chance GA itself gets stuck "
                         "in a local optimum and produces a false 'grid is fine' verdict.")
    p.add_argument("--write_refined", type=str, default=None,
                    help="If given, write a GA-refined version of the sampled rows to this CSV.")
    args = p.parse_args()

    df = load_ground_truth(args.ground_truth_csv)
    rng = np.random.default_rng(args.seed)
    sample = df.sample(n=min(args.n_samples, len(df)), random_state=args.seed).reset_index(drop=True)

    print(f"Cross-checking {len(sample)} ground-truth rows against grade-fixed GA "
          f"(pop={args.pop_size}, gens={args.n_generations}, restarts={args.n_restarts})...\n")
    print(f"{'span_m':>7s} {'load':>6s} {'grade':>5s} {'type':>7s} {'grid_mass':>10s} "
          f"{'ga_mass':>10s} {'improve%':>9s}")

    improvements = []
    refined_rows = []
    for i, r in sample.iterrows():
        best_ga_mass = np.inf
        best_ga_result = None
        for restart in range(args.n_restarts):
            result = ga_design_fixed_grade(
                span_mm=r["span_m"] * 1000.0, load_kNm=r["load_kNm"], storey=20,
                grade=r["grade"], section_type=r["section_type"], economy_metric="mass",
                pop_size=args.pop_size, n_generations=args.n_generations,
                seed=args.seed * 1000 + i * 10 + restart,
            )
            if result["feasible"] and result["mass"] < best_ga_mass:
                best_ga_mass = result["mass"]
                best_ga_result = result

        if best_ga_result is None:
            print(f"{r['span_m']:7.2f} {r['load_kNm']:6.1f} {r['grade']:5.0f} {r['section_type']:>7s}   "
                  f"GA found NO feasible design (grid says {r['mass']:.1f} kg -- investigate)")
            continue

        improvement = (r["mass"] - best_ga_mass) / r["mass"]
        improvements.append(improvement)
        print(f"{r['span_m']:7.2f} {r['load_kNm']:6.1f} {r['grade']:5.0f} {r['section_type']:>7s} "
              f"{r['mass']:10.2f} {best_ga_mass:10.2f} {improvement*100:8.2f}%")

        refined_rows.append(dict(r) | dict(
            ga_mass=best_ga_mass, ga_h=best_ga_result["h"], ga_b=best_ga_result["b"],
            ga_tf=best_ga_result["tf"], ga_tw=best_ga_result["tw"], improvement=improvement,
        ))

    improvements = np.array(improvements)
    print(f"\n{'='*60}")
    print(f"n contexts checked      : {len(improvements)}")
    print(f"mean improvement        : {improvements.mean()*100:.3f}%")
    print(f"median improvement      : {np.median(improvements)*100:.3f}%")
    print(f"max improvement         : {improvements.max()*100:.3f}%")
    print(f"contexts with >1% improvement found: {(improvements > 0.01).sum()} / {len(improvements)}")

    if (improvements > 0.01).mean() > 0.10:
        print("\n=> WARNING: GA found >1% better designs on more than 10% of sampled")
        print("   contexts. The grid-search ground truth is NOT reliable enough to")
        print("   report gap metrics against as-is. Options: (a) regenerate with a")
        print("   finer grid, (b) use --write_refined to produce GA-corrected ground")
        print("   truth for the full dataset, (c) use GA itself (many restarts) as")
        print("   the reference optimum instead of the grid.")
    else:
        print("\n=> Grid ground truth validated: GA improvement <1% on >90% of sampled")
        print("   contexts. Safe to report gap metrics against pretrain_data/ec3_optimal_designs.csv")
        print("   as-is for Experiment 1.")

    if args.write_refined and refined_rows:
        pd.DataFrame(refined_rows).to_csv(args.write_refined, index=False)
        print(f"\nGA-refined sample written to {args.write_refined}")


if __name__ == "__main__":
    main()
