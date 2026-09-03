"""
Zero-training-cost diagnostic: decompose the reported PPO cost gap into
(1) constraint-boundary slack, (2) cross-section shape/proportion
allocation error, (3) grade/section-type selection error.

Also tests whether evaluate.py's initial-state construction (reset()
samples geometry for a DIFFERENT random context, then span/load are
overwritten) inflates the measured gap.
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from research.envs.hss_env import HSSBeamEnv
from research.scripts.evaluate import (load_ground_truth, ground_truth_path_for_metric,
                                       ground_truth_optimum, load_policy)

LIM = dict(h=(250., 750.), b=(120., 300.), tf=(8., 35.), tw=(6., 25.))


def eval_design(env, span_mm, load, h, b, tf, tw, fy, stype):
    env.use_storey_load_scaling = False
    env.span, env.load = span_mm, load
    env.h, env.b, env.tf, env.tw, env.fy, env.section_type = h, b, tf, tw, fy, stype
    util, mass, pen, cls_loss, chi, dbg = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    feas = (util <= 1.0 + 1e-3) and dbg["section_class"] <= 3 and pen <= 1e-9
    return dict(util=util, mass=mass, cost=cost, co2=co2, feasible=feas,
                sec_class=dbg["section_class"])


def rescale_to_boundary(env, span_mm, load, d, metric="cost"):
    """Uniformly shrink the agent's section until util hits 1.0 (bisection).
    Pure 'remove unused capacity' move: keeps every proportion ratio and the
    grade/section type identical, so it isolates boundary slack alone."""
    lo, hi = 0.3, 1.0
    best = eval_design(env, span_mm, load, d["h"], d["b"], d["tf"], d["tw"], d["fy"], d["stype"])
    if not best["feasible"]:
        return best, 1.0
    for _ in range(40):
        s = 0.5 * (lo + hi)
        g = {k: float(np.clip(d[k] * s, *LIM[k])) for k in ("h", "b", "tf", "tw")}
        r = eval_design(env, span_mm, load, g["h"], g["b"], g["tf"], g["tw"], d["fy"], d["stype"])
        if r["feasible"]:
            best, hi = r, s
        else:
            lo = s
    return best, hi


def run_episode(env, policy_fn, span_m, load, seed, consistent_init):
    obs, _ = env.reset(seed=seed)
    env.use_storey_load_scaling = False
    env.span = float(span_m) * 1000.0
    env.load = float(load)
    env.storey = 20
    if consistent_init:
        # Re-apply reset()'s own heuristic initialiser, but for the FORCED
        # context instead of the random one reset() happened to draw.
        span_mm = env.span
        h_target = np.clip(span_mm / 1000.0 * 42.0, 250., 650.)
        lf = (env.load - env.LOAD_MIN) / (env.LOAD_MAX - env.LOAD_MIN)
        env.h = float(np.clip(h_target, *LIM["h"]))
        env.b = float(np.clip(env.h / 3.0, *LIM["b"]))
        env.tf = float(np.clip(10.0 + lf * 20.0, *LIM["tf"]))
        env.tw = float(np.clip(7.0 + lf * 14.0, *LIM["tw"]))
    obs = env._get_obs()
    best, best_c = None, np.inf
    for _ in range(env.max_steps):
        a = policy_fn(obs)
        obs, r, term, trunc, info = env.step(a)
        if info["feasible"] and info[env.economy_metric] < best_c:
            best_c = info[env.economy_metric]
            best = dict(h=info["h"], b=info["b"], tf=info["tf"], tw=info["tw"],
                        fy=info["fy"], stype=info["section_type"],
                        util=info["utilization"], cost=info["cost"], mass=info["mass"])
        if term or trunc:
            break
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--gt_dir", default="research/pretrain_data")
    p.add_argument("--metric", default="cost")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    gt = load_ground_truth(ground_truth_path_for_metric(a.metric, a.gt_dir))
    opt = ground_truth_optimum(gt, a.metric)
    policy_fn = load_policy(a.model_path, "ppo")
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=a.metric,
                     economy_reward_mode="log_relative")

    rows = []
    for mode in (False, True):
        for i, r in opt.iterrows():
            span_m, load = r["span_m"], r["load_kNm"]
            d = run_episode(env, policy_fn, span_m, load, seed=i, consistent_init=mode)
            if d is None:
                rows.append(dict(mode=mode, span_m=span_m, load_kNm=load, infeasible=True))
                continue
            resc, s = rescale_to_boundary(env, span_m * 1000.0, load, d, a.metric)
            # best geometry achievable at the AGENT's grade + section type
            sub = gt[(gt.span_m == span_m) & (gt.load_kNm == load) &
                     (gt.grade == int(d["fy"])) & (gt.section_type == d["stype"])]
            c_samegrade = float(sub[a.metric].min()) if len(sub) else np.nan
            c_opt = float(r[a.metric])
            rows.append(dict(
                mode=mode, span_m=span_m, load_kNm=load,
                util=d["util"], grade=d["fy"], stype=d["stype"],
                h=d["h"], b=d["b"], tf=d["tf"], tw=d["tw"],
                c_agent=d["cost"] if a.metric == "cost" else d[a.metric],
                c_rescaled=resc[a.metric], util_rescaled=resc["util"], scale=s,
                c_samegrade=c_samegrade, c_opt=c_opt,
                opt_grade=r["grade"], opt_type=r["section_type"],
                infeasible=False))
    df = pd.DataFrame(rows)
    df["gap_total"] = df.c_agent / df.c_opt - 1
    df["gap_after_rescale"] = df.c_rescaled / df.c_opt - 1
    df["gap_shape_plus_grade"] = df.c_samegrade / df.c_opt - 1
    df["comp_slack"] = df.gap_total - df.gap_after_rescale
    df["comp_shape"] = df.gap_after_rescale - (df.c_samegrade / df.c_opt - 1)
    df["comp_grade"] = df.c_samegrade / df.c_opt - 1

    out = a.out or f"research/results/diag_decomp_{a.label}.csv"
    df.to_csv(out, index=False)
    for mode in (False, True):
        s = df[(df["mode"] == mode) & (~df.infeasible)]
        print(f"\n=== {a.label} | consistent_init={mode} | n={len(s)}")
        print(f"  util                 mean {s.util.mean():.3f}")
        print(f"  gap_total            mean {s.gap_total.mean()*100:6.2f}%  med {s.gap_total.median()*100:6.2f}%")
        print(f"  after rescale->u=1   mean {s.gap_after_rescale.mean()*100:6.2f}%  med {s.gap_after_rescale.median()*100:6.2f}%")
        print(f"  -- component: boundary slack  {s.comp_slack.mean()*100:6.2f} pp")
        print(f"  -- component: shape/prop      {s.comp_shape.mean()*100:6.2f} pp")
        print(f"  -- component: grade/type      {s.comp_grade.mean()*100:6.2f} pp")
        print(f"  grade match {(s.grade==s.opt_grade).mean():.2f}  rescale factor mean {s.scale.mean():.3f}")
    print("\n->", out)


if __name__ == "__main__":
    main()
