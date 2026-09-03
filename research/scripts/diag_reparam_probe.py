"""
Zero-training probe: is the ~25% PPO cost gap caused by the ACTION
PARAMETERISATION rather than by the learning algorithm?

Evidence motivating this probe (research/results/diag_decomp_*.csv):
  * 100% of GA optima have util == 1.000     -> capacity constraint ACTIVE
  * 87%  of GA optima are section Class 3    -> classification limit ACTIVE
  * 57%  sit exactly on d/tw == 124*eps      -> web slenderness limit ACTIVE
  * 24% / 29% / 44% sit on h=750 / b=300 / tw=6 box bounds
  => the cost optimum is a VERTEX of the feasible set, and one side of the
     Class-3 limit is a discontinuous cliff (hss_env returns util>=2.0 and
     mass=4000 for Class 4). A Gaussian stochastic policy must keep its mean
     several sigma away from that cliff, which is exactly the observed
     failure signature (tf and tw ~1.3-1.4x too thick, util ~0.86).

The probe compares TWO search spaces under an IDENTICAL, tiny evaluation
budget, with no learning at all:

  RAW      : sample (h, b, tf, tw, grade, type) uniformly -- the space the
             PPO agent actually acts in.
  REPARAM  : sample (h, b/h, lambda_f, lambda_w, grade, type), where
             lambda_f/lambda_w are the EC3 Table 5.2 slenderness ratios.
             tf and tw are DERIVED from them, so section class <= 3 holds
             BY CONSTRUCTION, then the section is uniformly scaled until
             util == 1.0 (bisection), so the capacity constraint holds
             BY CONSTRUCTION too.

If REPARAM at ~20-40 samples approaches the GA reference (~1.3%) while RAW
needs thousands, the bottleneck is the parameterisation, not PPO.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from research.envs.hss_env import HSSBeamEnv

LIM = dict(h=(250., 750.), b=(120., 300.), tf=(8., 35.), tw=(6., 25.))
GRADES = [355., 460., 500., 550., 620., 690.]
TYPES = ["rolled", "welded"]


def ec3(env, span_mm, load, h, b, tf, tw, fy, st):
    env.use_storey_load_scaling = False
    env.span, env.load = span_mm, load
    env.h, env.b, env.tf, env.tw, env.fy, env.section_type = h, b, tf, tw, fy, st
    util, mass, pen, _, _, dbg = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    feas = (util <= 1.0 + 1e-3) and dbg["section_class"] <= 3 and pen <= 1e-9
    return feas, dict(util=util, mass=mass, cost=cost, co2=co2, cls=dbg["section_class"])


def decode(h, b_over_h, lam_f, lam_w, fy, st):
    """Feasible-by-construction geometry decoder.

    lam_f, lam_w are the EC3 Table 5.2 slenderness ratios (c/tf)/eps and
    (d/tw)/eps. Capping them at the Class-3 limits (14 and 124) makes the
    section-classification constraint structurally unreachable rather than
    something the agent has to learn to avoid via a penalty cliff.
    Fixed-point iteration because the rolled-section root radius r = 0.1*tf
    makes c and d depend on tf.
    """
    eps = float(np.sqrt(235.0 / fy))
    b = float(np.clip(b_over_h * h, *LIM["b"]))
    tf, tw = 15.0, 8.0
    for _ in range(12):
        r = 0.1 * tf if st == "rolled" else 0.0
        tf_new = ((b - tw) / 2.0 - r) / (lam_f * eps)
        tf_new = float(np.clip(tf_new, *LIM["tf"]))
        d_web = (h - 2.0 * tf_new) - 2.0 * (0.1 * tf_new if st == "rolled" else 0.0)
        tw_new = float(np.clip(d_web / (lam_w * eps), *LIM["tw"]))
        if abs(tf_new - tf) < 1e-6 and abs(tw_new - tw) < 1e-6:
            tf, tw = tf_new, tw_new
            break
        tf, tw = tf_new, tw_new
    return h, b, tf, tw


def size_to_boundary(env, span_mm, load, geom, fy, st, n_bisect=18):
    """Uniformly scale the decoded section until util == 1.0.
    Uniform scaling leaves every slenderness ratio (and hence the section
    class) invariant, so this cannot re-break the class constraint."""
    h, b, tf, tw = geom
    # find a feasible upper scale
    s_hi = None
    for s in (1.0, 1.15, 1.35, 1.6, 2.0, 2.6):
        g = [np.clip(h * s, *LIM["h"]), np.clip(b * s, *LIM["b"]),
             np.clip(tf * s, *LIM["tf"]), np.clip(tw * s, *LIM["tw"])]
        ok, res = ec3(env, span_mm, load, *g, fy, st)
        if ok:
            s_hi, best = s, res
            break
    if s_hi is None:
        return None, None, None
    s_lo = 0.25
    for _ in range(n_bisect):
        s = 0.5 * (s_lo + s_hi)
        g = [np.clip(h * s, *LIM["h"]), np.clip(b * s, *LIM["b"]),
             np.clip(tf * s, *LIM["tf"]), np.clip(tw * s, *LIM["tw"])]
        ok, res = ec3(env, span_mm, load, *g, fy, st)
        if ok:
            s_hi, best, best_g = s, res, g
        else:
            s_lo = s
    g = [np.clip(h * s_hi, *LIM["h"]), np.clip(b * s_hi, *LIM["b"]),
         np.clip(tf * s_hi, *LIM["tf"]), np.clip(tw * s_hi, *LIM["tw"])]
    ok, best = ec3(env, span_mm, load, *g, fy, st)
    return (best if ok else None), g, s_hi


def search_raw(env, span_m, load, budget, rng, metric="cost"):
    best = np.inf
    for _ in range(budget):
        h = rng.uniform(*LIM["h"]); b = rng.uniform(*LIM["b"])
        tf = rng.uniform(*LIM["tf"]); tw = rng.uniform(*LIM["tw"])
        fy = GRADES[rng.integers(len(GRADES))]; st = TYPES[rng.integers(2)]
        ok, res = ec3(env, span_m * 1000.0, load, h, b, tf, tw, fy, st)
        if ok:
            best = min(best, res[metric])
    return best


def search_reparam(env, span_m, load, budget, rng, metric="cost", n_bisect=18):
    """budget = number of DECODER proposals. Each proposal costs
    (1 + n_bisect + a few) EC3 evaluations for the analytic sizing step;
    both counts are reported so the comparison stays honest."""
    best = np.inf
    ec3_calls = 0
    for _ in range(budget):
        h = rng.uniform(*LIM["h"])
        b_over_h = rng.uniform(0.20, 0.75)
        lam_f = rng.uniform(5.0, 14.0)
        lam_w = rng.uniform(60.0, 124.0)
        fy = GRADES[rng.integers(len(GRADES))]; st = TYPES[rng.integers(2)]
        geom = decode(h, b_over_h, lam_f, lam_w, fy, st)
        res, g, s = size_to_boundary(env, span_m * 1000.0, load, geom, fy, st, n_bisect)
        ec3_calls += n_bisect + 6
        if res is not None:
            best = min(best, res[metric])
    return best, ec3_calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="cost")
    p.add_argument("--gt_dir", default="research/pretrain_data")
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 40, 100])
    p.add_argument("--raw_budgets", type=int, nargs="+", default=[40, 400, 4000])
    p.add_argument("--n_contexts", type=int, default=142)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    gt = pd.read_csv(os.path.join(a.gt_dir, f"ec3_optimal_designs_{a.metric}.csv"))
    opt = gt.loc[gt.groupby(["span_m", "load_kNm"])[a.metric].idxmin()].reset_index(drop=True)
    if a.n_contexts < len(opt):
        opt = opt.sample(a.n_contexts, random_state=a.seed).reset_index(drop=True)
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=a.metric)

    print(f"n_contexts = {len(opt)}   metric = {a.metric}\n")
    print(f"{'space':<10}{'proposals':>10}{'EC3 evals':>11}{'gap mean':>10}{'median':>9}{'p90':>8}{'worst':>8}{'feas':>7}")
    print("-" * 66)
    out = []
    for bud in a.raw_budgets:
        rng = np.random.default_rng(a.seed)
        gaps = []
        for _, r in opt.iterrows():
            c = search_raw(env, r["span_m"], r["load_kNm"], bud, rng, a.metric)
            gaps.append(np.nan if not np.isfinite(c) else c / r[a.metric] - 1)
        g = np.array(gaps); f = np.isfinite(g)
        print(f"{'RAW':<10}{bud:>10}{bud:>11}{np.nanmean(g)*100:>9.1f}%{np.nanmedian(g)*100:>8.1f}%"
              f"{np.nanpercentile(g[f],90)*100:>7.1f}%{np.nanmax(g[f])*100:>7.1f}%{f.mean():>7.2f}")
        out.append(dict(space="raw", proposals=bud, ec3=bud, gap_mean=np.nanmean(g)))
    for bud in a.budgets:
        rng = np.random.default_rng(a.seed)
        gaps = []; calls = 0
        for _, r in opt.iterrows():
            c, n = search_reparam(env, r["span_m"], r["load_kNm"], bud, rng, a.metric)
            calls = n
            gaps.append(np.nan if not np.isfinite(c) else c / r[a.metric] - 1)
        g = np.array(gaps); f = np.isfinite(g)
        print(f"{'REPARAM':<10}{bud:>10}{calls:>11}{np.nanmean(g)*100:>9.1f}%{np.nanmedian(g)*100:>8.1f}%"
              f"{np.nanpercentile(g[f],90)*100:>7.1f}%{np.nanmax(g[f])*100:>7.1f}%{f.mean():>7.2f}")
        out.append(dict(space="reparam", proposals=bud, ec3=calls, gap_mean=np.nanmean(g)))
    pd.DataFrame(out).to_csv(f"research/results/diag_reparam_probe_{a.metric}.csv", index=False)


if __name__ == "__main__":
    main()
