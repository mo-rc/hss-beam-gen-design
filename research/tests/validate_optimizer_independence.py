"""
research/tests/validate_optimizer_independence.py
================================================================
research/tests/validate_ground_truth.py cross-checked the GA against a
LARGER-BUDGET run of the SAME GA implementation -- useful for catching
grid-coarseness error, but not genuine algorithmic independence: a
systematic bug or blind spot in ga_baseline.py's fitness/constraint
handling would replicate in both runs and this check would not catch it.

This script cross-checks a sample of contexts against scipy.optimize.
differential_evolution -- a different algorithm, different library,
independent implementation of the mutation/selection/convergence logic
-- using the IDENTICAL EC3 physics (research.envs.hss_env.HSSBeamEnv,
imported directly, same as the GA) and the identical constraint/fitness
definition, so any disagreement reflects a genuine optimizer-choice
effect, not a different problem definition.

USAGE
------
    python research/tests/validate_optimizer_independence.py --n_samples 15 --seed 0
================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.optimize import differential_evolution

from research.envs.hss_env import HSSBeamEnv
from research.scripts.evaluate import load_ground_truth
from research.scripts.ga_baseline import BOUNDS, ga_design_fixed_grade, _make_probe_env


def de_design_fixed_grade(span_mm, load_kNm, storey, grade, section_type, economy_metric,
                            seed=0, maxiter=150, popsize=20):
    env = _make_probe_env(economy_metric)

    def objective(x):
        h, b, tf, tw = x
        env.h, env.b, env.tf, env.tw = float(h), float(b), float(tf), float(tw)
        env.fy, env.section_type = float(grade), section_type
        env.span, env.load, env.storey = float(span_mm), float(load_kNm), int(storey)
        util, mass, penalty, class_loss, chi_lt, dbg = env._ec3_analysis()
        cost, co2, _ = env._calculate_cost_co2(mass)
        violations = env._constraint_violations(util, class_loss, penalty)
        feasible = all(v <= 1e-3 for v in violations.values())
        economy = env._economy(mass, cost, co2)
        if feasible:
            return economy
        return 50.0 + 20.0 * violations["g1_util"] + 10.0 * violations["g2_class"] + 5.0 * violations["g3_geom"]

    bounds = [BOUNDS["h"], BOUNDS["b"], BOUNDS["tf"], BOUNDS["tw"]]
    result = differential_evolution(objective, bounds, seed=seed, maxiter=maxiter,
                                      popsize=popsize, tol=1e-7, polish=True, workers=1)

    h, b, tf, tw = result.x
    env.h, env.b, env.tf, env.tw = float(h), float(b), float(tf), float(tw)
    env.fy, env.section_type = float(grade), section_type
    env.span, env.load = float(span_mm), float(load_kNm)
    util, mass, penalty, class_loss, chi_lt, dbg = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    violations = env._constraint_violations(util, class_loss, penalty)
    feasible = all(v <= 1e-3 for v in violations.values())
    return dict(h=h, b=b, tf=tf, tw=tw, util=util, mass=mass, cost=cost, co2=co2, feasible=feasible)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ground_truth_csv", default="pretrain_data/ec3_optimal_designs.csv",
                    help="Any per-metric ground truth CSV works; this script only needs "
                         "(span_m, load_kNm, grade, section_type) rows to sample from -- "
                         "it re-derives its own reference value via DE, it does not trust "
                         "the CSV's stored mass/cost/co2 columns.")
    p.add_argument("--n_samples", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--economy_metric", default="mass")
    args = p.parse_args()

    df = load_ground_truth(args.ground_truth_csv)
    sample = df.sample(n=min(args.n_samples, len(df)), random_state=args.seed).reset_index(drop=True)

    print(f"Independent-algorithm cross-check: GA vs. scipy differential_evolution, "
          f"{len(sample)} contexts, economy_metric={args.economy_metric}\n")
    print(f"{'span_m':>7s} {'load':>6s} {'grade':>5s} {'type':>7s} {'GA_val':>9s} {'DE_val':>9s} "
          f"{'GA_feas':>7s} {'DE_feas':>7s} {'diff%':>8s}")

    diffs = []
    disagreements = 0
    for i, r in sample.iterrows():
        ga = ga_design_fixed_grade(span_mm=r["span_m"] * 1000.0, load_kNm=r["load_kNm"], storey=20,
                                     grade=r["grade"], section_type=r["section_type"],
                                     economy_metric=args.economy_metric,
                                     pop_size=60, n_generations=100, seed=args.seed + i)
        de = de_design_fixed_grade(span_mm=r["span_m"] * 1000.0, load_kNm=r["load_kNm"], storey=20,
                                     grade=r["grade"], section_type=r["section_type"],
                                     economy_metric=args.economy_metric, seed=args.seed + i)

        ga_val = ga[args.economy_metric] if ga["feasible"] else np.nan
        de_val = de[args.economy_metric] if de["feasible"] else np.nan

        if ga["feasible"] != de["feasible"]:
            disagreements += 1
            print(f"{r['span_m']:7.2f} {r['load_kNm']:6.1f} {r['grade']:5.0f} {r['section_type']:>7s} "
                  f"{ga_val!s:>9s} {de_val!s:>9s} {str(ga['feasible']):>7s} {str(de['feasible']):>7s}   "
                  f"FEASIBILITY DISAGREEMENT")
            continue

        if ga["feasible"] and de["feasible"]:
            diff_pct = 100 * abs(ga_val - de_val) / min(ga_val, de_val)
            diffs.append(diff_pct)
            print(f"{r['span_m']:7.2f} {r['load_kNm']:6.1f} {r['grade']:5.0f} {r['section_type']:>7s} "
                  f"{ga_val:9.2f} {de_val:9.2f} {'True':>7s} {'True':>7s} {diff_pct:7.2f}%")

    diffs = np.array(diffs)
    print(f"\n{'='*60}")
    print(f"n contexts both feasible : {len(diffs)}/{len(sample)}")
    print(f"feasibility disagreements: {disagreements}/{len(sample)}")
    if len(diffs):
        print(f"mean GA-vs-DE difference : {diffs.mean():.3f}%")
        print(f"max GA-vs-DE difference  : {diffs.max():.3f}%")
    if disagreements == 0 and len(diffs) and diffs.mean() < 2.0:
        print("\n=> GA and an independent optimizer (scipy differential_evolution) agree to "
              "<2% on average with zero feasibility disagreements. The GA-based reference "
              "optimizer is independently validated.")
    else:
        print("\n=> Disagreement above the 2% / feasibility-consistency threshold -- "
              "investigate before trusting GA-based ground truth.")


if __name__ == "__main__":
    main()
