"""
research/scripts/expA5_ood_amortizer.py
================================================================
Exp A''' -- does an IN-DISTRIBUTION-trained amortiser extrapolate?

QUESTION
--------
Exp A'' showed a kNN amortiser trained on ~208 DE-solved contexts (drawn
uniformly from span 6-15m, load 20-140 kNm) reaches ~0.1-0.9% gap
in-distribution, at a fraction of PPO's training AND inference cost. But
kNN is pure local interpolation: it has no mechanism to extrapolate
outside the convex hull of its training contexts. This script re-uses
the EXACT SAME trained models (same label file, no relabelling, no
re-fitting logic beyond what expA3 already does) and asks them to design
for OUT-OF-DISTRIBUTION (span, load) contexts, using an OOD ground-truth
directory built by regenerate_ground_truth.py.

WHY NO RELABELLING
-------------------
The whole point is "what does the artefact we already built do when
asked to generalise". Re-fitting on OOD-adjacent labels would answer a
different, easier question. Training contexts stay span 6-15m /
load 20-140 kNm as in expA3; only the EVALUATION contexts move.

REUSES, UNCHANGED
------------------
research.scripts.expA_strong_baselines.load_contexts   (OOD GT loader)
research.scripts.expA3_amortized_baselines.{coords, features, make_model,
    _u_from, _infer_task, GRADES, TYPES}                (fit + decode + repair)
research.scripts.expA3_amortized_baselines.compare      (paired comparison)
research.algo.repair.repair                             (same operator as PPO)

USAGE
------
    # 1. build OOD ground truth once (regenerate_ground_truth.py), e.g.
    python research/scripts/regenerate_ground_truth.py \\
        --out_dir research/pretrain_data_ood_span --metrics cost \\
        --span_min_m 16 --span_max_m 22 --load_min 20 --load_max 140 \\
        --n_spans 8 --n_loads 8

    # 2. this script (no GPU needed; ~minutes, reuses existing labels)
    python research/scripts/expA5_ood_amortizer.py \\
        --gt_dir research/pretrain_data_ood_span --region span_extrapolation \\
        --labels_csv research/results/expA3/expA3_full_labels.csv \\
        --a2_runs research/results/expA5_ood/span_extrapolation/expA2_span_runs.csv \\
        --rl_csv "research/results/expA5_ood/span_extrapolation/e5_ppo_s*_ood_span_per_context.csv" \\
        --out_dir research/results/expA5_ood/span_extrapolation --tag span
================================================================
"""
import os
import sys
import glob
import json
import math
import argparse
import platform
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
import research.scripts.expA_strong_baselines as A
import research.scripts.expA2_pipeline_baselines as P
import research.scripts.expA3_amortized_baselines as A3

STOREY = 20


def main():
    ap = argparse.ArgumentParser(description="Exp A''': OOD extrapolation test of the Exp A'' amortiser")
    ap.add_argument("--metric", default="cost", choices=["mass", "cost", "co2"])
    ap.add_argument("--gt_dir", required=True, help="OOD ground-truth dir, built by regenerate_ground_truth.py")
    ap.add_argument("--region", required=True, help="label written into the output rows, e.g. span_extrapolation")
    ap.add_argument("--labels_csv", default="research/results/expA3/expA3_full_labels.csv",
                     help="EXISTING in-distribution label file; never regenerated here")
    ap.add_argument("--out_dir", default="research/results/expA5_ood")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--models", default="knn,gp,poly3,mlp")
    ap.add_argument("--N", type=int, default=208, help="label-set size (must be <= rows per label_set in labels_csv)")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--n_label_sets", type=int, default=3)
    ap.add_argument("--size_tol", type=float, default=0.003)
    ap.add_argument("--size_max", type=int, default=10)
    ap.add_argument("--n_contexts", type=int, default=None, help="subsample the OOD grid (debug/smoke only)")
    # optional: fold in already-computed OOD baseline (expA2) / PPO rows for a paired comparison
    ap.add_argument("--a2_runs", default=None, help="expA2_pipeline_baselines runs.csv computed on this SAME --gt_dir")
    ap.add_argument("--rl_csv", nargs="*", default=[], help="evaluate_with_repair.py per_context.csv(s) computed on this SAME --gt_dir")
    ap.add_argument("--rl_label", default="ppo_ood")
    ap.add_argument("--compare_gen_budget", type=int, default=40,
                     help="gen_budget to pull from --a2_runs (method=de, space=reparam) for the DE reference line")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    p = lambda s: os.path.join(a.out_dir, f"expA5_{a.tag}_{s}")
    models = a.models.split(",")

    # ---- 1. OOD evaluation contexts (ground truth built OUTSIDE the training envelope)
    opt = A.load_contexts(a.gt_dir, a.metric, a.n_contexts, 0)
    print(f"[{a.region}] {len(opt)} OOD evaluation contexts loaded from {a.gt_dir}")
    print(f"  span range: {opt.span_m.min():.2f}-{opt.span_m.max():.2f} m | "
          f"load range: {opt.load_kNm.min():.2f}-{opt.load_kNm.max():.2f} kNm")

    # ---- 2. load the EXISTING in-distribution labels (no relabelling)
    lab_all = pd.read_csv(a.labels_csv)
    max_avail = lab_all.groupby("label_set").size().min()
    if a.N > max_avail:
        sys.exit(f"--N {a.N} exceeds available labelled contexts per set ({max_avail}) in {a.labels_csv}")

    # ---- 3. fit on in-distribution labels, predict OOD contexts, decode + repair (all counted)
    A3._init(a.metric, a.size_tol, a.size_max)
    Xe = A3.features(opt.span_m.values, opt.load_kNm.values)
    all_rows = []
    for s in range(a.n_label_sets):
        ls = lab_all[lab_all.label_set == s].sort_values("j")
        d = ls.head(a.N)
        d = d[d.feasible == 1]
        Y = np.array([A3.coords(r) for r in d.to_dict("records")])
        combo = (d.fy.map({g: i for i, g in enumerate(A3.GRADES)}).values * 2
                 + (d.section_type == "welded").astype(int).values)
        X = A3.features(d.span_m.values, d.load_kNm.values)
        from sklearn.neighbors import KNeighborsClassifier
        clf = KNeighborsClassifier(n_neighbors=min(3, len(d)), weights="distance").fit(X, combo)
        proba = clf.predict_proba(Xe)
        top = [[int(clf.classes_[i]) for i in np.argsort(-pr, kind="stable")[:a.topk]] for pr in proba]
        for mname in models:
            mdl = A3.make_model(mname, s).fit(X, Y)
            pred = mdl.predict(Xe)
            for i, r in enumerate(opt.itertuples()):
                task = (s, a.N, mname, i, r.span_m, r.load_kNm, float(getattr(r, a.metric)),
                        tuple(float(x) for x in pred[i]), top[i])
                all_rows.extend(A3._infer_task(task))
    runs = pd.DataFrame(all_rows)
    runs["region"] = a.region
    runs.to_csv(p("runs.csv"), index=False)

    # ---- 4. summarise
    summ = A3.summarize(runs, {a.N: np.nan})  # label_evals not meaningful here (labels are reused, not new)
    summ["region"] = a.region
    summ.to_csv(p("summary.csv"), index=False)
    pd.set_option("display.width", 250)
    t = summ.copy()
    for c in ("gap_mean", "gap_median", "gap_p90"):
        t[c] = (100 * t[c]).round(2)
    print(f"\n=== [{a.region}] amortiser (fit in-distribution) evaluated OOD: gap % vs OOD GT ===")
    print(t[["mode", "model", "topk", "feasibility_rate", "gap_mean", "gap_median", "gap_p90",
             "mean_n_total"]].round(2).to_string(index=False))

    # ---- 5. optional paired comparisons against OOD GA/DE/PPO (all must share this SAME --gt_dir)
    comps = []
    if a.rl_csv:
        rl, _ = P.load_rl(a.rl_csv, opt, a.metric, a.rl_label)
        rl.to_csv(p("rl_rows.csv"), index=False)
        comps.append(A3.compare(runs, rl, f"PPO OOD ({a.rl_label})"))
        rl_summ = P.summarize(rl)
        rl_summ["region"] = a.region
        rl_summ.to_csv(p("rl_summary.csv"), index=False)
        rt = rl_summ.copy()
        for c in ("gap_mean", "gap_median", "gap_p90"):
            rt[c] = (100 * rt[c]).round(2)
        print(f"\n=== [{a.region}] PPO evaluated OOD (same GT) ===")
        print(rt[["mode", "feasibility_rate", "gap_mean", "gap_median", "gap_p90", "mean_n_total"]]
              .round(2).to_string(index=False))
    if a.a2_runs and os.path.exists(a.a2_runs):
        a2 = pd.read_csv(a.a2_runs)
        de = a2[(a2.method == "de") & (a2.space == "reparam") & (a2.gen_budget == a.compare_gen_budget)]
        if len(de):
            comps.append(A3.compare(runs, de, f"DE reparam G={a.compare_gen_budget} + repair (OOD)"))
        ga = a2[(a2.method == "ga") & (a2.space == "raw") & (a2.gen_budget == a.compare_gen_budget)]
        if len(ga):
            comps.append(A3.compare(runs, ga, f"GA raw G={a.compare_gen_budget} + repair (OOD)"))
    if comps:
        c = pd.concat(comps)
        c["region"] = a.region
        c.to_csv(p("compare.csv"), index=False)
        big = c[(c.N == c.N.max()) & (c.topk.isin([1, 3]))].copy()
        for col in ("amortiser_gap", "reference_gap", "diff", "ci95_lo", "ci95_hi"):
            big[col] = (100 * big[col]).round(2)
        print(f"\n=== [{a.region}] paired vs OOD references (negative diff = amortiser better; pp of gap) ===")
        print(big[["reference", "mode", "model", "topk", "amortiser_gap", "reference_gap", "diff",
                   "ci95_lo", "ci95_hi", "amortiser_win_rate", "amortiser_evals", "reference_evals"]]
              .round(2).to_string(index=False))

    json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(), args=vars(a),
                   python=platform.python_version(), platform=platform.platform(),
                   n_ood_contexts=len(opt), label_source=a.labels_csv, region=a.region),
              open(p("meta.json"), "w"), indent=2, default=str)
    print(f"\nwrote files with prefix {p('')}")


if __name__ == "__main__":
    main()
