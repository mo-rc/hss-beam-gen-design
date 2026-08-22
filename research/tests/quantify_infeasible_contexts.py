"""
research/tests/quantify_infeasible_contexts.py
================================================================
Some (span, load) demand contexts within the nominal training/evaluation
range cannot be satisfied by ANY (grade, section_type) combination, even
at maximum-bound geometry (h=750, b=300, tf=35, tw=25) -- confirmed
during the audit at span=15m/load=140kN/m (util=1.10 rolled, 1.46 welded
even at max bounds, S690). These contexts are automatically excluded from
ground-truth-based gap metrics (a context with zero feasible rows simply
has no entry to compare against), which is CORRECT behaviour, but the
exclusion needs to be quantified and stated explicitly rather than left
implicit -- a reviewer should not have to infer "why are there only ~745
of a possible 1728 rows" on their own.

This is a fast, deterministic, GA-independent check: for every (span,
load) grid point, try ALL 6 grades x 2 types at MAXIMUM-BOUND geometry
(the best any design at that grade/type could conceivably achieve) and
report whether at least one combination is feasible. No optimisation
involved -- this is a necessary-condition check (if even the max-bound
geometry fails, no smaller geometry could possibly succeed), so it is
exact, not an approximation like the GA-based ground truth.

USAGE
------
    python research/tests/quantify_infeasible_contexts.py --resolution 30
================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.envs.hss_env import HSSBeamEnv

GRADES = [355, 460, 500, 550, 620, 690]
SECTION_TYPES = ["rolled", "welded"]
MAX_GEOM = dict(h=750.0, b=300.0, tf=35.0, tw=25.0)


def max_bound_feasible_any_grade(env, span_m, load_kNm):
    for section_type in SECTION_TYPES:
        for grade in GRADES:
            env.h, env.b, env.tf, env.tw = MAX_GEOM["h"], MAX_GEOM["b"], MAX_GEOM["tf"], MAX_GEOM["tw"]
            env.fy, env.section_type = float(grade), section_type
            env.span, env.load = span_m * 1000.0, load_kNm
            env.use_storey_load_scaling = False
            util, mass, penalty, class_loss, chi_lt, dbg = env._ec3_analysis()
            if util <= 1.0 + 1e-6 and dbg["section_class"] <= 3:
                return True, grade, section_type, util
    return False, None, None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--span_min_m", type=float, default=6.0)
    p.add_argument("--span_max_m", type=float, default=15.0)
    p.add_argument("--load_min", type=float, default=20.0)
    p.add_argument("--load_max", type=float, default=140.0)
    p.add_argument("--resolution", type=int, default=30)
    args = p.parse_args()

    env = HSSBeamEnv(reward_mode="lagrangian")
    spans = np.linspace(args.span_min_m, args.span_max_m, args.resolution)
    loads = np.linspace(args.load_min, args.load_max, args.resolution)

    infeasible_points = []
    n_total = 0
    n_infeasible = 0
    for span_m in spans:
        for load_kNm in loads:
            n_total += 1
            feasible, grade, stype, util = max_bound_feasible_any_grade(env, span_m, load_kNm)
            if not feasible:
                n_infeasible += 1
                infeasible_points.append((span_m, load_kNm))

    frac = n_infeasible / n_total
    print(f"Scanned {n_total} (span, load) grid points "
          f"({args.resolution}x{args.resolution}, span {args.span_min_m}-{args.span_max_m}m, "
          f"load {args.load_min}-{args.load_max} kN/m)")
    print(f"Structurally infeasible (no grade/type feasible even at max-bound geometry): "
          f"{n_infeasible}/{n_total} ({frac*100:.1f}%)")

    if infeasible_points:
        spans_inf = [p[0] for p in infeasible_points]
        loads_inf = [p[1] for p in infeasible_points]
        print(f"\nInfeasible region boundary (within the scanned grid):")
        print(f"  span range where infeasibility occurs : {min(spans_inf):.1f} - {max(spans_inf):.1f} m")
        print(f"  load range where infeasibility occurs : {min(loads_inf):.1f} - {max(loads_inf):.1f} kN/m")
        # Report the single least-demanding infeasible point, to characterise the boundary precisely.
        min_demand_point = min(infeasible_points, key=lambda p: p[0] * p[1])
        print(f"  least-demanding infeasible point       : span={min_demand_point[0]:.2f}m, "
              f"load={min_demand_point[1]:.1f}kN/m (demand_proxy={min_demand_point[0]*min_demand_point[1]:.0f})")

    print(f"\n>>> Report this figure explicitly in the paper's scope/data section: "
          f"{frac*100:.1f}% of the nominal (span, load) training/evaluation envelope has NO "
          f"feasible design under ANY grade or section type, and is therefore correctly excluded "
          f"from ground-truth-based gap metrics, not silently dropped.")


if __name__ == "__main__":
    main()
