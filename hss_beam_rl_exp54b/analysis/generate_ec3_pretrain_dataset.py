"""
================================================================
generate_ec3_pretrain_dataset.py
----------------------------------------------------------------
Generates a supervised dataset of EC3-optimal beam geometries for
behavior-cloning pretraining of the PPO policy network.

WHY THIS EXISTS:
    Eight RL-only experiments (exp47-56) converged to a policy that
    learned roughly ONE fixed geometry that scores acceptably when
    averaged over the training distribution, rather than a genuine
    (span, load, grade) -> geometry mapping. Since the correct
    geometry IS a deterministic function of context (that's what
    EC3 gives us), this is a case where supervised pretraining
    should establish real context-conditioning before RL is used
    to refine cost/CO2/grade tradeoffs on top of it.

WHAT THIS SCRIPT PRODUCES:
    A CSV where each row is:
        span_m, load_kNm, grade, section_type,
        h, b, tf, tw,          <- EC3-optimal (min mass, feasible)
        util, mass, cost, co2  <- resulting performance

    For each (span, load, grade, section_type) combination, a finer
    grid search (default 9^4 = 6561 points, vs the diagnostic's
    quick 5^4=625) finds the minimum-mass feasible geometry.

USAGE:
    python generate_ec3_pretrain_dataset.py \
        --env ./env/high_rise_generative_env_claude_final.py \
        --out ./pretrain_data/ec3_optimal_designs.csv \
        --resolution 9

NEXT STEP (not in this script):
    Use this CSV to pretrain the PPO policy's mean-action head via
    supervised MSE regression (normalise h/b/tf/tw the same way
    _get_obs()/_update_design() do), then continue with PPO
    fine-tuning from that warm-started policy.
================================================================
"""

import argparse
import importlib.util
import csv
import os
import numpy as np


def load_env_class(env_path: str):
    spec   = importlib.util.spec_from_file_location("env_module", env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HighRiseGenerativeEnv


def make_bare_env(EnvClass, span_m, load_kNm, section_type, ltb_factor):
    env = EnvClass(use_storey_load_scaling=False, ltb_restraint_factor=ltb_factor)
    env.np_random = np.random.default_rng(0)
    env.span         = span_m * 1000.0
    env.load         = load_kNm
    env.storey       = 20
    env.section_type = section_type
    env.current_util = env.current_mass = env.current_cost = env.current_co2 = 0.0
    env.current_chi_lt = 1.0
    env.current_class  = 1
    env.current_Mrd = env.current_area = 1.0
    env.prev_util = env.prev_mass = env.moment_ratio = 0.0
    env.memory = []
    env.curr_step = 0
    env.success_counter = 0
    return env


def ec3_eval(env, h, b, tf, tw, fy, section_type):
    env.h, env.b, env.tf, env.tw = h, b, tf, tw
    env.fy = float(fy)
    env.section_type = section_type
    util, mass, penalty, class_loss, chi_lt, ec3 = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    return util, mass, cost, co2, ec3, penalty


def find_optimal_geometry(env, span_m, load_kNm, fy, section_type, resolution):
    """
    Coarse-to-fine sweep: minimum-mass geometry with util in [0.90, 1.02].

    STAGE 1 (coarse): full grid at `resolution` points/dimension — cheap,
    finds the approximate region of the optimum.

    STAGE 2 (refine): a second, finer grid search restricted to a small
    window around the coarse optimum (±1.5 coarse grid-steps in each
    dimension, resolved at `resolution` points within that window).
    This is a standard coarse-to-fine search: it fixes the discretization
    noise that caused ~32% mass-ordering inversions at resolution=7 alone,
    without the full n^4 blowup of simply raising global resolution
    (going 7->11 globally costs ~6x; this refine pass costs ~1x extra
    on top of the coarse pass, since the window is small).
    """
    limits = {"h": (250.0, 750.0), "b": (120.0, 300.0),
              "tf": (8.0, 35.0),   "tw": (6.0, 25.0)}
    n = resolution

    def sweep(h_range, b_range, tf_range, tw_range):
        best_mass = np.inf
        best = None
        for h in np.linspace(*h_range, n):
            for b in np.linspace(*b_range, n):
                for tf in np.linspace(*tf_range, n):
                    for tw in np.linspace(*tw_range, n):
                        util, mass, cost, co2, ec3, penalty = ec3_eval(
                            env, h, b, tf, tw, fy, section_type)
                        if (0.90 <= util <= 1.02
                                and ec3["section_class"] <= 2
                                and penalty == 0
                                and mass < best_mass):
                            best_mass = mass
                            governing = ("DEFL" if ec3["deflection_util"]
                                        > ec3["moment_util"] else "MOMENT")
                            best = (h, b, tf, tw, util, mass, cost, co2, governing)
        return best

    # Single-stage sweep. The two-stage coarse-to-fine refinement tried
    # previously made cross-grade comparability WORSE (42% vs 32%
    # inversions) because each grade's fine window anchored independently
    # to its own coarse optimum, decoupling grades from a shared search
    # region. Reverted — the real issue is governance-regime confounding
    # (see main() / inspect script), not search precision.
    result = sweep(limits["h"], limits["b"], limits["tf"], limits["tw"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="../env/high_rise_generative_env.py")
    parser.add_argument("--out", default="../pretrain_data/ec3_optimal_designs.csv")
    parser.add_argument("--resolution", type=int, default=9,
                        help="Grid points per geometry dimension (9 -> 6561 evals/context)")
    parser.add_argument("--ltb-factor", type=float, default=0.25)
    parser.add_argument("--n-spans", type=int, default=12,
                        help="Number of span points across [6,15]m")
    parser.add_argument("--n-loads", type=int, default=12,
                        help="Number of load points across [20,140]kN/m")
    args = parser.parse_args()

    EnvClass = load_env_class(args.env)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    spans  = np.linspace(6.0, 15.0, args.n_spans)
    loads  = np.linspace(20.0, 140.0, args.n_loads)
    grades = [355, 460, 500, 550, 620, 690]
    types  = ["rolled", "welded"]

    total = len(spans) * len(loads) * len(grades) * len(types)
    print(f"Generating {total} contexts, {args.resolution**4} geometry "
          f"evaluations each ({total * args.resolution**4:,} total EC3 calls)")
    print(f"This will take a while for high resolution — consider running "
          f"resolution=7 or 9 first to check output quality before scaling up.\n")

    rows = []
    done = 0
    for span_m in spans:
        for load_kNm in loads:
            for section_type in types:
                env = make_bare_env(EnvClass, span_m, load_kNm,
                                    section_type, args.ltb_factor)
                for fy in grades:
                    result = find_optimal_geometry(
                        env, span_m, load_kNm, fy, section_type,
                        args.resolution)
                    done += 1
                    if result is not None:
                        h, b, tf, tw, util, mass, cost, co2, governing = result
                        rows.append({
                            "span_m": span_m, "load_kNm": load_kNm,
                            "grade": fy, "section_type": section_type,
                            "h": h, "b": b, "tf": tf, "tw": tw,
                            "util": util, "mass": mass,
                            "cost": cost, "co2": co2,
                            "governing": governing,
                        })
                    if done % 50 == 0:
                        print(f"  [{done}/{total}] contexts processed, "
                              f"{len(rows)} feasible so far")

    with open(args.out, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\nDone. {len(rows)}/{total} contexts had a feasible EC3-optimal "
          f"geometry (rest are outside the design envelope, e.g. 15m/140kN·m "
          f"at low grades — expected, matches earlier diagnostic finding).")
    print(f"Saved to: {args.out}")


if __name__ == "__main__":
    main()
