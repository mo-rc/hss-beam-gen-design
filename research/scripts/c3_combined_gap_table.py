"""
research/scripts/c3_combined_gap_table.py
================================================================
Comment 3: report mass, cost, and carbon (CO2) optimality gap together, for
every method compared, not cost alone.

MECHANISM (no re-training, no re-evaluation, no policy re-run):
the corrected-costing per-context result files already store each design's
full geometry (h, b, tf, tw, fy, section_type) at every context, for every
arm and repair_mode. This script re-derives mass and CO2 for each STORED
design from that geometry via HSSBeamEnv's own EC3 model (the same
Cost/CO2 calculation the env uses at generation time), then compares each
metric against ITS OWN ground-truth optimum at that context (NOT the
cost-optimal design's incidental mass/CO2 -- see
evaluate.ground_truth_optimum_all_metrics docstring for why that
distinction matters). This mirrors exactly what evaluate.py's
evaluate_policy_vs_ground_truth() already does for un-post-processed
single-seed runs; this script extends the same logic to (a) post-processed
(repaired) designs and (b) multi-seed aggregation.

SOURCES (all corrected-costing; verified against pretrain_data_corrected
before use):
  GA / rule-based / random search (40, 4800):
      research/results/e4_nonrl_baselines_per_context.csv
      (the canonical source per results/c1_number_ledger.csv -- NOT
      e3_corrcost_retrain_per_context.csv, which has a different,
      non-canonical GA run)
  PPO (Comment 2's 5-seed headline, e5_ppo_s{42..46}_anneal_linear):
      research/results/e5_ppo_s{seed}_anneal_linear_curve_1000000_per_context.csv

USAGE
    python research/scripts/c3_combined_gap_table.py
Writes:
    research/results/c3_combined_gap_table.csv        (one row per method x repair_mode)
    research/results/c3_combined_gap_per_context.csv  (per-context mass/cost/co2 gaps, for reuse)
================================================================
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.scripts.evaluate import (ground_truth_optimum_all_metrics,
                                       ground_truth_path_for_metric, load_ground_truth)

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(R, "results")
DEFAULT_GT_DIR = os.path.join(R, "pretrain_data_corrected")
REPAIR_MODES = ["none", "scale+thin"]


def assert_env_matches_ground_truth(env, gt_df, economy_metric, n_check=25, rtol=1e-4):
    """Self-contained costing guard (no shared helper exists in evaluate.py in this
    branch of the repo): re-cost n_check stored ground-truth designs with `env`'s
    own EC3/cost model and abort if they don't reproduce the stored value. Prevents
    the Comment-1 class of bug (env costing model != ground-truth costing model,
    which silently shifts every gap by ~20-40%)."""
    if economy_metric == "mass":
        return 0.0
    rows = gt_df.sample(n=min(n_check, len(gt_df)), random_state=0)
    prev = env.use_storey_load_scaling
    env.use_storey_load_scaling = False
    worst = 0.0
    try:
        for r in rows.itertuples():
            env.span, env.load = float(r.span_m) * 1000.0, float(r.load_kNm)
            env.h, env.b, env.tf, env.tw = float(r.h), float(r.b), float(r.tf), float(r.tw)
            env.fy, env.section_type = float(r.grade), r.section_type
            _, mass, _, _, _, _ = env._ec3_analysis()
            cost, co2, _ = env._calculate_cost_co2(mass)
            val = {"cost": cost, "co2": co2}[economy_metric]
            worst = max(worst, abs(val / getattr(r, economy_metric) - 1.0))
    finally:
        env.use_storey_load_scaling = prev
    if worst > rtol:
        raise ValueError(
            f"Costing mismatch: env does not reproduce the ground truth at "
            f"{gt_df is not None and 'this dir' or ''} (max deviation {100*worst:.2f}% > "
            f"{100*rtol:.3f}% tolerance). Refusing to compute gaps against a ground truth "
            f"built under a different costing model than this env.")
    return worst

BASELINE_FILE = os.path.join(RES, "e4_nonrl_baselines_per_context.csv")
BASELINE_ARMS = ["ga", "rule_based", "random_search_40", "random_search_4800"]
PPO_SEEDS = [42, 43, 44, 45, 46]
PPO_RUN_PREFIX = "e5_ppo_s{seed}_anneal_linear"


def _env():
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric="cost")
    env.use_storey_load_scaling = False
    return env


def add_mass_co2_gaps(df, gt_all, env):
    """Given a per-context frame with stored geometry, add gap_mass / gap_co2
    columns computed from that geometry, against each metric's own ground-truth
    optimum. NaN where infeasible (mirrors evaluate.py's gap convention)."""
    gap_mass, gap_co2, cost_check = [], [], []
    for r in df.itertuples():
        if not bool(r.feasible):
            gap_mass.append(np.nan)
            gap_co2.append(np.nan)
            cost_check.append(np.nan)
            continue
        env.span, env.load = r.span_m * 1000.0, r.load_kNm
        env.h, env.b, env.tf, env.tw, env.fy, env.section_type = r.h, r.b, r.tf, r.tw, r.fy, r.section_type
        _, mass, _, _, _, _ = env._ec3_analysis()
        cost, co2, _ = env._calculate_cost_co2(mass)
        key = (r.span_m, r.load_kNm)
        gt = gt_all.get(key, {})
        gap_mass.append(mass / gt["mass"] - 1.0 if "mass" in gt else np.nan)
        gap_co2.append(co2 / gt["co2"] - 1.0 if "co2" in gt else np.nan)
        cost_check.append(cost)
    df = df.copy()
    df["gap_mass"], df["gap_co2"] = gap_mass, gap_co2
    df["_cost_recomputed"] = cost_check
    return df


def load_baselines(env, gt_all):
    d = pd.read_csv(BASELINE_FILE)
    d = d[d.arm.isin(BASELINE_ARMS) & d.repair_mode.isin(REPAIR_MODES)].copy()
    d["method"] = d["arm"]
    d["seed"] = -1  # deterministic / not seed-varying
    return add_mass_co2_gaps(d, gt_all, env)


def load_ppo(env, gt_all):
    frames = []
    for seed in PPO_SEEDS:
        run = PPO_RUN_PREFIX.format(seed=seed)
        path = os.path.join(RES, f"{run}_curve_1000000_per_context.csv")
        d = pd.read_csv(path)
        d = d[d.repair_mode.isin(REPAIR_MODES)].copy()
        d["method"], d["seed"] = "ppo_5seed_headline", seed
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    return add_mass_co2_gaps(d, gt_all, env)


def aggregate(df):
    """Per method x repair_mode. For the seed-varying PPO arm this is the
    mean-of-per-seed-means (same convention as Comment 2's c2_multiseed
    aggregation), so a single seed's outlier run cannot dominate the table;
    for seed==-1 (deterministic baselines) it collapses to a plain mean."""
    rows = []
    for (method, mode), g in df.groupby(["method", "repair_mode"]):
        feasibility = g["feasible"].astype(bool).mean()
        per_seed = g.groupby("seed").agg(
            gap_mean=("gap", "mean"), gap_median=("gap", "median"),
            gap_mass_mean=("gap_mass", "mean"), gap_mass_median=("gap_mass", "median"),
            gap_co2_mean=("gap_co2", "mean"), gap_co2_median=("gap_co2", "median"),
            util_mean=("utilization", "mean"), feasibility=("feasible", lambda x: x.astype(bool).mean()),
        )
        n_seeds = len(per_seed)
        row = dict(method=method, repair_mode=mode, n_contexts=len(g) // n_seeds, n_seeds=n_seeds,
                   feasibility=feasibility)
        for col in ["gap_mean", "gap_median", "gap_mass_mean", "gap_mass_median",
                    "gap_co2_mean", "gap_co2_median", "util_mean"]:
            row[col] = float(per_seed[col].mean())
            if n_seeds > 1:
                row[col + "_sd_across_seeds"] = float(per_seed[col].std(ddof=1))
        rows.append(row)
    out = pd.DataFrame(rows)
    order = ["ga", "rule_based", "random_search_40", "random_search_4800", "ppo_5seed_headline"]
    out["method"] = pd.Categorical(out["method"], categories=order, ordered=True)
    return out.sort_values(["method", "repair_mode"]).reset_index(drop=True)


def main():
    gt_dir = DEFAULT_GT_DIR
    gt_all = ground_truth_optimum_all_metrics(gt_dir)
    env = _env()
    # Guard: this script's env must reproduce the SAME ground truth the source
    # files were scored against (Comment 1's costing-mismatch class of bug).
    gt_cost = load_ground_truth(ground_truth_path_for_metric("cost", gt_dir))
    assert_env_matches_ground_truth(env, gt_cost, "cost")

    base = load_baselines(env, gt_all)
    ppo = load_ppo(env, gt_all)
    combined = pd.concat([base, ppo], ignore_index=True)

    # Sanity check: recomputed cost must reproduce the stored 'achieved' cost
    # for every feasible row (confirms the geometry -> EC3 re-derivation is
    # exactly consistent with how these files were originally scored).
    feas = combined["feasible"].astype(bool)
    dev = (combined.loc[feas, "_cost_recomputed"] / combined.loc[feas, "achieved"] - 1.0).abs()
    worst = float(dev.max())
    print(f"Sanity check: max |recomputed_cost / stored_achieved_cost - 1| over "
          f"{feas.sum()} feasible rows = {worst:.2e}")
    assert worst < 1e-6, "geometry-based recomputation does not match the stored cost; aborting"
    combined = combined.drop(columns=["_cost_recomputed"])

    pc_path = os.path.join(RES, "c3_combined_gap_per_context.csv")
    combined.to_csv(pc_path, index=False)

    table = aggregate(combined)
    table_path = os.path.join(RES, "c3_combined_gap_table.csv")
    table.to_csv(table_path, index=False)

    pd.set_option("display.width", 220)
    show = table.copy()
    for c in ["feasibility", "gap_mean", "gap_median", "gap_mass_mean", "gap_mass_median",
              "gap_co2_mean", "gap_co2_median", "util_mean"]:
        show[c] = (100 * show[c]).round(2) if c != "util_mean" else show[c].round(3)
    print("\nComment 3 combined table (mass / cost / carbon gap, %, mean & median; util_mean is a ratio):")
    print(show[["method", "repair_mode", "n_seeds", "feasibility", "gap_mass_mean", "gap_mass_median",
                "gap_mean", "gap_median", "gap_co2_mean", "gap_co2_median", "util_mean"]]
          .rename(columns={"gap_mean": "cost_gap_mean", "gap_median": "cost_gap_median"})
          .to_string(index=False))
    print(f"\nwrote {table_path}\nwrote {pc_path}")


if __name__ == "__main__":
    main()
