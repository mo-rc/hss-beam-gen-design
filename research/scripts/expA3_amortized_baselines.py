"""
research/scripts/expA3_amortized_baselines.py
================================================================
Exp A'' -- amortisation WITHOUT reinforcement learning.

QUESTION
--------
Exp A' showed PPO(+scale+thin) reaches ~9% gap at ~113 EC3 evals/design while
optimiser pipelines need ~300-400 evals for the same gap. Is that advantage
due to RL, or would ANY amortiser trained on a comparable number of solved
contexts do as well? This script trains plain supervised regressors on
contexts solved by a strong optimiser (DE, raw space, 4800 evals -- the best
Exp A baseline), then designs new contexts with

    predict (h, b/h, lambda_f, lambda_w, grade, type)  ->  decode + size to the
    utilisation boundary (Exp A reparam decoder, counted)  ->  [optional] the
    SAME repair() operator as PPO / Exp A' (counted)

FAIRNESS
--------
* Same 142 evaluation contexts, same physics, objective, feasibility, GT and
  operator as Exp A / A'.
* Training contexts are drawn continuously and uniformly in the GT grid's
  ranges (span 6..15 m, load 20..140 kNm) -> disjoint from the 142 grid
  contexts. Labels come from the optimiser, NEVER from the GT files.
* Training cost is counted: label_evals = 4800 x (#training contexts). The RL
  policy costs ~1M evals per training run; N=208 contexts ~ 1.0M evals, so
  N=208 is the budget-matched setting (N=50,100 are cheaper).
* Inference cost is counted with the same meter: decode+sizing evaluations
  (<= size_max per candidate, top-k candidate (grade,type) combinations) plus
  operator evaluations. `n_total` is per design.
* Repeated over independent label sets (contexts AND optimiser seeds); the SD
  across label sets is reported. No cherry-picking of models: all models and
  all top-k values are reported.

MODELS (inputs: normalised span, load; outputs: reparam coordinates)
  knn   k=3 distance-weighted | gp  RBF+White GP | poly3  ridge cubic |
  mlp   sklearn MLP 64x64 (seeded).  Grade/type: kNN(3) classifier over the
  12 combinations; top-k candidates by class probability are all decoded,
  sized and the best feasible design kept (evals counted).

USAGE
    pip install scikit-learn        # if missing
    python research/scripts/expA3_amortized_baselines.py --tag full --n_jobs 4
    python research/scripts/expA3_amortized_baselines.py --tag full --aggregate_only
================================================================
"""
import os
import sys
import json
import math
import time
import argparse
import platform
from datetime import datetime, timezone
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

from research.envs.hss_env import HSSBeamEnv
from research.algo.repair import repair
import research.scripts.expA_strong_baselines as A
import research.scripts.expA2_pipeline_baselines as P

SPAN_RNG, LOAD_RNG = (6.0, 15.0), (20.0, 140.0)                  # GT grid ranges
MODES = ["none", "scale", "scale+thin"]
STOREY = 20
LABEL_BUDGET = 4800
_W = {}


def _init(metric, size_tol, size_max):
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=metric,
                     enforce_rolled_manufacturability=True)
    _W["meter"] = P.PipeMeter(env, metric)
    _W["raw"], _W["rep"] = A.RawSpace(), A.ReparamSpace(size_tol, size_max)
    _W["metric"] = metric


def coords(r):
    eps = math.sqrt(235.0 / r["fy"])
    rr = 0.1 * r["tf"] if r["section_type"] == "rolled" else 0.0
    lf = ((r["b"] - r["tw"]) / 2.0 - rr) / (r["tf"] * eps)
    lw = ((r["h"] - 2 * r["tf"]) - 2 * rr) / (r["tw"] * eps)
    return r["h"], r["b"] / r["h"], lf, lw


# ---------------------------------------------------------------- labelling
def _label_task(a):
    set_id, j, span_m, load, budget, hp = a
    m, sp = _W["meter"], _W["raw"]
    m.reset(span_m * 1000.0, load, budget)
    rng = np.random.default_rng(np.random.SeedSequence([777, set_id, j]))
    try:
        A.RUNNERS["de"](sp, m, budget, rng, hp)
    except A.BudgetExhausted:
        pass
    row = dict(label_set=set_id, j=j, span_m=span_m, load_kNm=load, n_evals=m.n,
               feasible=int(m.best_econ is not None), economy=m.best_econ if m.best_econ is not None else np.nan)
    if m.best_econ is not None:
        row.update(zip(["h", "b", "tf", "tw", "fy", "section_type"], m.best_design))
    return row


# ---------------------------------------------------------------- models
def make_model(name, seed):
    if name == "knn":
        return KNeighborsRegressor(n_neighbors=3, weights="distance")
    if name == "poly3":
        return make_pipeline(StandardScaler(), PolynomialFeatures(3), Ridge(alpha=1e-1))
    if name == "gp":
        return GaussianProcessRegressor(C(1.0) * RBF([0.3, 0.3]) + WhiteKernel(1e-3), normalize_y=True,
                                        n_restarts_optimizer=1, random_state=seed)
    if name == "mlp":
        return TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=3000, random_state=seed)),
            transformer=StandardScaler())
    raise ValueError(name)


def features(span_m, load):
    return np.c_[(np.asarray(span_m) - SPAN_RNG[0]) / (SPAN_RNG[1] - SPAN_RNG[0]),
                 (np.asarray(load) - LOAD_RNG[0]) / (LOAD_RNG[1] - LOAD_RNG[0])]


GRADES, TYPES = A.GRADES, A.TYPES


def _u_from(h, bh, lf, lw, gi, ti):
    L, R = A.LIM, A.REP_LIM
    f = lambda v, lo, hi: float(np.clip((v - lo) / (hi - lo), 0.0, 1.0 - 1e-9))
    return np.array([f(h, *L["h"]), f(bh, *R["bh"]), f(lf, *R["lf"]), f(lw, *R["lw"]),
                     (gi + 0.5) / len(GRADES), (ti + 0.5) / 2.0])


# ---------------------------------------------------------------- inference (counted)
def _infer_task(a):
    (set_id, N, model, ctx_i, span_m, load, gt, pred, combos) = a
    m, sp, metric = _W["meter"], _W["rep"], _W["metric"]
    m.reset(span_m * 1000.0, load, math.inf)
    n_ops = 0
    rows = []
    for k, c in enumerate(combos, 1):
        gi, ti = c // 2, c % 2
        sp.propose(_u_from(*pred, gi, ti), m)                     # decode + sizing, counted
        n_dec = m.n - n_ops
        feas0 = m.best_econ is not None
        base = dict(label_set=set_id, N=N, model=model, topk=k, context_idx=ctx_i, span_m=span_m,
                    load_kNm=load, gt_economy=gt)
        rows.append(dict(base, mode="none", feasible=int(feas0),
                         gap=(m.best_econ / gt - 1.0) if feas0 else np.nan, n_dec=n_dec, n_repair=0, n_total=n_dec))
        dsg = m.best_design if feas0 else m.min_viol_design
        design = dict(zip(["h", "b", "tf", "tw", "fy", "section_type"], dsg))
        for mode in ("scale", "scale+thin"):
            n0 = m.n
            out = repair(m.env, span_m * 1000.0, load, design, metric=metric, storey=STOREY, mode=mode)
            nr = m.n - n0
            assert nr == out["n_ec3"]
            n_ops += nr
            ok = bool(out["feasible"])
            rows.append(dict(base, mode=mode, feasible=int(ok), gap=(out[metric] / gt - 1.0) if ok else np.nan,
                             n_dec=n_dec, n_repair=nr, n_total=n_dec + nr))
    return rows


# ---------------------------------------------------------------- summaries
def summarize(runs, labels_cost):
    recs = []
    for (model, N, topk, mode), d in runs.groupby(["model", "N", "topk", "mode"]):
        f = d[d.feasible == 1]
        g = f.gap.values
        sm = f.groupby("label_set").gap.mean()
        recs.append(dict(model=model, N=int(N), topk=int(topk), mode=mode, n_runs=len(d),
                         n_label_sets=d.label_set.nunique(), feasibility_rate=float(d.feasible.mean()),
                         gap_mean=float(np.mean(g)) if len(g) else np.nan,
                         gap_median=float(np.median(g)) if len(g) else np.nan,
                         gap_p90=float(np.percentile(g, 90)) if len(g) else np.nan,
                         gap_worst=float(np.max(g)) if len(g) else np.nan,
                         labelset_sd=float(sm.std(ddof=1)) if len(sm) > 1 else np.nan,
                         frac_gap_le_5pct=float(((d.gap <= 0.05) & (d.feasible == 1)).mean()),
                         mean_n_total=float(d.n_total.mean()), mean_n_repair=float(d.n_repair.mean()),
                         label_evals=float(labels_cost.get(int(N), np.nan))))
    return pd.DataFrame(recs).sort_values(["mode", "N", "model", "topk"]).reset_index(drop=True)


def _ctx_means(d):
    return d[d.feasible == 1].groupby("context_idx").gap.mean()


def compare(runs, ref, ref_name, n_boot=4000, seed=0):
    """Paired over contexts: (amortiser - reference); negative = amortiser better."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model, N, topk, mode), d in runs.groupby(["model", "N", "topk", "mode"]):
        r = ref[ref["mode"] == mode]
        if r.empty:
            continue
        a, b = _ctx_means(d), _ctx_means(r)
        idx = a.index.intersection(b.index)
        diff = (a.loc[idx] - b.loc[idx]).values
        bt = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
        rows.append(dict(reference=ref_name, model=model, N=int(N), topk=int(topk), mode=mode,
                         amortiser_gap=float(a.loc[idx].mean()), reference_gap=float(b.loc[idx].mean()),
                         diff=float(diff.mean()), ci95_lo=float(np.percentile(bt, 2.5)),
                         ci95_hi=float(np.percentile(bt, 97.5)), amortiser_win_rate=float((diff < 0).mean()),
                         amortiser_evals=float(d.n_total.mean()), reference_evals=float(r.n_total.mean())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Exp A'': amortisation without RL")
    ap.add_argument("--metric", default="cost", choices=["mass", "cost", "co2"])
    ap.add_argument("--gt_dir", default="research/pretrain_data_corrected")
    ap.add_argument("--out_dir", default="research/results/expA3")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--n_label_sets", type=int, default=3)
    ap.add_argument("--sizes", default="50,100,208")
    ap.add_argument("--models", default="knn,gp,poly3,mlp")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--label_budget", type=int, default=LABEL_BUDGET)
    ap.add_argument("--n_contexts", type=int, default=None)
    ap.add_argument("--size_tol", type=float, default=0.003)
    ap.add_argument("--size_max", type=int, default=10)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--rl_rows", default="research/results/expA2/expA2_full_rl_rows.csv")
    ap.add_argument("--a2_runs", default="research/results/expA2/expA2_full_runs.csv")
    ap.add_argument("--rl_train_evals", type=float, default=1.0e6)
    a = ap.parse_args()

    sizes = [int(x) for x in a.sizes.split(",")]
    models = a.models.split(",")
    os.makedirs(a.out_dir, exist_ok=True)
    p = lambda s: os.path.join(a.out_dir, f"expA3_{a.tag}_{s}")
    labels_csv, runs_csv = p("labels.csv"), p("runs.csv")
    opt_all = A.load_contexts(a.gt_dir, a.metric, None, 0)
    make_it = lambda fn, tasks: (Pool(a.n_jobs, initializer=_init, initargs=(a.metric, a.size_tol, a.size_max)).imap_unordered(fn, tasks)
                                 if a.n_jobs > 1 else (_init(a.metric, a.size_tol, a.size_max), map(fn, tasks))[1])

    if not a.aggregate_only:
        for f in (labels_csv, runs_csv):
            if os.path.exists(f):
                sys.exit(f"{f} exists; use a new --tag (or --aggregate_only). Refusing to overwrite.")
        chk = A.gt_sanity_check(opt_all, a.metric, a.size_tol, a.size_max)
        print("GT sanity check:", chk)
        if chk["max_rel_dev_gt_reeval"] > 1e-3 or chk["gt_designs_infeasible"] > 0:
            sys.exit("GT re-evaluation mismatch; aborting.")
        t0 = time.time()
        # ---- 1. labels: continuous contexts disjoint from the GT grid, solved by DE (raw), cost counted
        nmax = max(sizes)
        tasks = []
        for s in range(a.n_label_sets):
            rng = np.random.default_rng(1000 + s)
            sp_ = rng.uniform(*SPAN_RNG, nmax); ld_ = rng.uniform(*LOAD_RNG, nmax)
            tasks += [(s, j, float(sp_[j]), float(ld_[j]), a.label_budget, A.hp_for(A.DEFAULT_HP, "de", "raw"))
                      for j in range(nmax)]
        lab = pd.DataFrame(list(make_it(_label_task, tasks)))
        lab.sort_values(["label_set", "j"]).to_csv(labels_csv, index=False)
        print(f"labels: {len(lab)} contexts, feasible {int(lab.feasible.sum())}, {(time.time() - t0) / 60:.1f} min")
        # ---- 2. fit models, predict the 142 evaluation contexts
        Xe = features(opt_all.span_m.values, opt_all.load_kNm.values)
        eval_tasks = []
        for s in range(a.n_label_sets):
            ls = lab[lab.label_set == s].sort_values("j")
            for N in sizes:
                d = ls.head(N)
                d = d[d.feasible == 1]
                Y = np.array([coords(r) for r in d.to_dict("records")])
                combo = (d.fy.map({g: i for i, g in enumerate(GRADES)}).values * 2
                         + (d.section_type == "welded").astype(int).values)
                X = features(d.span_m.values, d.load_kNm.values)
                clf = KNeighborsClassifier(n_neighbors=min(3, len(d)), weights="distance").fit(X, combo)
                proba = clf.predict_proba(Xe)
                top = [[int(clf.classes_[i]) for i in np.argsort(-pr, kind="stable")[:a.topk]] for pr in proba]
                for mname in models:
                    mdl = make_model(mname, s).fit(X, Y)
                    pred = mdl.predict(Xe)
                    for i, r in enumerate(opt_all.itertuples()):
                        if a.n_contexts and i >= a.n_contexts:
                            break
                        eval_tasks.append((s, N, mname, i, r.span_m, r.load_kNm, float(getattr(r, a.metric)),
                                           tuple(float(x) for x in pred[i]), top[i]))
        print(f"{len(eval_tasks)} inference tasks")
        header = True
        for k, rows in enumerate(make_it(_infer_task, eval_tasks), 1):
            pd.DataFrame(rows).to_csv(runs_csv, mode="a", header=header, index=False)
            header = False
            if k % max(1, len(eval_tasks) // 20) == 0:
                el = time.time() - t0
                print(f"  {k}/{len(eval_tasks)} | {el / 60:.1f} min", flush=True)
        json.dump(dict(created_utc=datetime.now(timezone.utc).isoformat(), args=vars(a), git_commit=A._git_commit(),
                       python=platform.python_version(), platform=platform.platform(), gt_check=chk,
                       label_method="DE raw, hp=" + json.dumps(A.hp_for(A.DEFAULT_HP, "de", "raw")),
                       training_contexts="uniform continuous, span 6..15, load 20..140, disjoint from GT grid",
                       versions={k: __import__(k).__version__ for k in ("numpy", "scipy", "pandas", "sklearn")}),
                  open(p("meta.json"), "w"), indent=2, default=str)

    # ---- 3. aggregate
    runs = pd.read_csv(runs_csv)
    lab = pd.read_csv(labels_csv)
    cost = {N: lab[lab.j < N].groupby("label_set").n_evals.sum().mean() for N in sorted(runs.N.unique())}
    summ = summarize(runs, cost)
    summ.to_csv(p("summary.csv"), index=False)
    json.dump(json.loads(summ.to_json(orient="records")), open(p("summary.json"), "w"), indent=2)
    comps = []
    if os.path.exists(a.rl_rows):
        rl = pd.read_csv(a.rl_rows)
        comps.append(compare(runs, rl, "PPO (5 seeds)"))
    if os.path.exists(a.a2_runs):
        a2 = pd.read_csv(a.a2_runs)
        de = a2[(a2.method == "de") & (a2.space == "reparam") & (a2.gen_budget == 40)]
        comps.append(compare(runs, de, "DE reparam G=40 + same operator"))
    if comps:
        pd.concat(comps).to_csv(p("compare.csv"), index=False)

    pd.set_option("display.width", 250)
    pc = lambda s: (100 * s).round(2)
    t = summ.copy()
    for c in ("gap_mean", "gap_median", "gap_p90", "labelset_sd"):
        t[c] = pc(t[c])
    print("\n=== amortised design quality (gap % vs GT; label_evals = training cost in EC3 evals) ===")
    print(t[["mode", "N", "model", "topk", "feasibility_rate", "gap_mean", "gap_median", "gap_p90", "labelset_sd",
             "mean_n_total", "label_evals"]].round(1).to_string(index=False))
    if comps:
        c = pd.concat(comps)
        for col in ("amortiser_gap", "reference_gap", "diff", "ci95_lo", "ci95_hi"):
            c[col] = pc(c[col])
        print("\n=== paired vs references (negative diff = amortiser better; pp of gap) : N=208, topk=1 and topk=3, all models ===")
        big = c[(c.N == c.N.max()) & (c.topk.isin([1, 3]))]
        print(big[["reference", "mode", "model", "topk", "amortiser_gap", "reference_gap", "diff", "ci95_lo", "ci95_hi",
                   "amortiser_win_rate", "amortiser_evals", "reference_evals"]].round(2).to_string(index=False))
    print(f"\nRL training cost ~{a.rl_train_evals:.0f} evals vs labelling cost above. wrote prefix {p('')}")


if __name__ == "__main__":
    main()
