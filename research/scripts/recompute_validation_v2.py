"""
research/scripts/recompute_validation_v2.py
================================================================
Refreshes the "paper-style validation" table (feasibility rate, cost-ratio
distribution, and RC-paper-style Table-11/12-style comparison) after the
grade cost-factor fix, WITHOUT retraining anything.

WHAT THIS DOES AND DOES NOT RECOMPUTE
---------------------------------------
- The policy's RAW design output (h, b, tf, tw, fy, section_type) for each
  of the 5 seeds x 142 contexts is READ AS-IS from the existing per-context
  CSVs -- the frozen policy is not re-run, no GPU/torch needed here.
- Its COST is recomputed from that same physical design under the
  CORRECTED cost coefficients (mass is unaffected by the fix; only cost
  changes for designs with fy != 355).
- The `scale` and `scale+thin` repair operators are RE-RUN fresh on that
  raw design under the corrected env (cheap, CPU-only, no training --
  repair() is a bounded local search over an analytic rescale + a few
  thinning candidates, a few dozen EC3 evaluations per context at most).
  This matters because repair's own choices are cost-guided, so its output
  can shift slightly under the new coefficients even for the same input.
- The reference "optimal" is read fresh from the regenerated ground-truth
  file (research/pretrain_data_corrected/ec3_optimal_designs_cost.csv,
  AFTER you've run regenerate_ground_truth.py and replaced that file).

This does NOT redo the GA/DE/CMA-ES matched-budget optimizer search --
those genuinely need to re-search the new cost surface from scratch, which
is a heavier re-run of expA2_pipeline_baselines.py, not a cheap recompute.
This script's "number of analyses" comparison therefore reuses the OLD
GA/DE numbers from expA2 as an approximate reference and labels them as
such; treat that part as provisional until expA2 is rerun.

USAGE
------
    # after: python research/scripts/regenerate_ground_truth.py \\
    #          --out_dir research/pretrain_data_corrected --metrics cost
    python research/scripts/recompute_validation_v2.py \\
        --gt_dir research/pretrain_data_corrected \\
        --per_context_glob "research/results/e5_ppo_s*_anneal_linear_curve_1000000_per_context.csv" \\
        --out_csv research/results/validation_v2_corrected_costs.csv
================================================================
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.algo.repair import repair, analyse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", default="research/pretrain_data_corrected")
    ap.add_argument("--metric", default="cost")
    ap.add_argument("--storey", type=int, default=20)
    ap.add_argument("--per_context_glob", required=True,
                     help="glob matching the 5 existing seed per-context CSVs")
    ap.add_argument("--out_csv", default="research/results/validation_v2_corrected_costs.csv")
    a = ap.parse_args()

    # ---- fresh ground truth (must already be regenerated with corrected coefficients)
    gt_path = os.path.join(a.gt_dir, f"ec3_optimal_designs_{a.metric}.csv")
    gt = pd.read_csv(gt_path)
    gt["span_m"] = gt["span_m"].round(6)
    gt["load_kNm"] = gt["load_kNm"].round(6)
    opt = gt.loc[gt.groupby(["span_m", "load_kNm"])[a.metric].idxmin()][["span_m", "load_kNm", a.metric]]
    opt = opt.rename(columns={a.metric: "optimal_new"})

    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=a.metric)
    env.reset(seed=0)

    files = sorted(glob.glob(a.per_context_glob))
    if not files:
        sys.exit(f"no files matched {a.per_context_glob}")
    print(f"found {len(files)} per-context files: {[os.path.basename(f) for f in files]}")

    rows = []
    for f in files:
        d = pd.read_csv(f)
        raw = d[d.repair_mode == "none"].copy()
        raw["span_m"] = raw["span_m"].round(6)
        raw["load_kNm"] = raw["load_kNm"].round(6)
        for _, r in raw.iterrows():
            design = dict(h=r.h, b=r.b, tf=r.tf, tw=r.tw, fy=r.fy, section_type=r.section_type)
            for mode in ("none", "scale", "scale+thin"):
                out = repair(env, r.span_m * 1000.0, r.load_kNm, design,
                             metric=a.metric, storey=a.storey, mode=mode)
                rows.append(dict(
                    seed=os.path.basename(f), span_m=r.span_m, load_kNm=r.load_kNm,
                    mode=mode, achieved_new=out[a.metric], feasible=out["feasible"],
                    n_ec3=out["n_ec3"],
                ))
    out = pd.DataFrame(rows).merge(opt, on=["span_m", "load_kNm"], how="left")
    assert out.optimal_new.notna().all(), "some contexts had no match in the regenerated ground truth"
    out["cost_ratio"] = out.achieved_new / out.optimal_new
    out.to_csv(a.out_csv, index=False)

    print(f"\n=== paper-style validation, corrected cost coefficients (n={out.seed.nunique()} seeds x "
          f"{out.span_m.nunique()*out.load_kNm.nunique() if False else len(opt)} contexts) ===")
    for mode in ("none", "scale", "scale+thin"):
        sub = out[out["mode"] == mode]
        print(f"--- mode={mode} (n={len(sub)}) ---")
        print(f"  feasibility: {100*sub.feasible.mean():.1f}%")
        print(f"  cost ratio: mean={sub.cost_ratio.mean():.3f} median={sub.cost_ratio.median():.3f}")
        for thresh in (1.10, 1.125, 1.25, 1.50):
            print(f"  within {int(thresh*100)}%: {100*(sub.cost_ratio <= thresh).mean():.1f}%")
        # 40 = the fixed policy rollout length already spent to produce the raw design
        # (n_ec3_gen in the source per-context CSVs); repair adds sub.n_ec3 on top.
        print(f"  mean total evals (40 rollout + repair): {40 + sub.n_ec3.mean():.1f}")

    print(f"\nwrote {a.out_csv}")


if __name__ == "__main__":
    main()
