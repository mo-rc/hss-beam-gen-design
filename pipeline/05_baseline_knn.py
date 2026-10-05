"""
pipeline/05_baseline_knn.py -- k-nearest-neighbour amortized-design baseline (CPU only).

WHAT IT IS
----------
A non-RL amortized mapping from the 2-parameter context (span_m, load_kN_per_m) to a complete
design (grade, section_type, h, b, tf, tw). "Training" is just storing solved contexts: the
labels are the best-known optimum per context taken from a POOLED ground-truth directory (the
same references the RL policy is scored against). No gradient training, no EC3 evaluation at
prediction time.

HOW A DESIGN IS PREDICTED (fixed rule, declared before any result was seen)
---------------------------------------------------------------------------
1. Features = (span, load), each min-max scaled with the TRAINING set's range.
2. Take the k nearest training contexts (Euclidean). Weights w = 1 / (d + 1e-3).
3. (grade, section_type) is chosen by weighted vote of the neighbours' pairs. It is NOT averaged:
   averaging a categorical grade index across neighbours was the diagnosed failure mode of the
   earlier kNN (systematic grade under-prediction).
4. (h, b, tf, tw) = weighted geometric mean (log-space) over the neighbours that share the
   winning (grade, section_type).
The predicted design is then scored with the SAME three operator modes as the RL policy
(`none`, `scale`, `scale+thin`, from hssbeamgen.algo.posthoc_operators) and the SAME summary
metrics (imported from pipeline/03_evaluate_agent.py), so rows are directly comparable.
Prediction itself costs 0 EC3 evaluations; `ec3_evals_per_design` counts the operator's analyses
only (1 for `none`).

PROTOCOLS (--protocol)
----------------------
  loo        every context in --train_gt_dir predicted from the OTHER contexts of that grid.
             Secondary number: the 12x12 grid gives each held-out point close neighbours on all
             sides, so this is optimistic for interpolation.
  subsample  for each --sizes n and repetition: train on a random n-context subset of the grid,
             test on the held-out remainder. Primary in-distribution number; gives a sample-size
             curve. Repetitions use seeds --seed0 .. --seed0 + --reps - 1.
  ood        train on ALL contexts of --train_gt_dir, test on every context of --test_gt_dir
             (e.g. ood_span_pooled). No leakage possible: test contexts lie outside the range.

k: --k takes several values (default 1 3 5). The headline k is --headline_k (default 3), fixed in
advance and NOT chosen on the test results; other k are a sensitivity table.

Labelling cost: the labels are the pooled ground truth, i.e. every training context was solved
by search (see the ground-truth meta.json for the per-context evaluation budget). Report
n_train x that budget as the kNN counterpart of RL's one-time training cost.

USAGE
-----
    # smoke test (a handful of contexts, seconds)
    python pipeline/05_baseline_knn.py --protocol loo --economy_metric cost \
        --train_gt_dir data/ground_truth/main_grid_pooled --n_contexts 8 --out /tmp/knn_smoke.csv

    # in-distribution, primary
    python pipeline/05_baseline_knn.py --protocol subsample --sizes 36 72 108 --reps 10 \
        --economy_metric cost --train_gt_dir data/ground_truth/main_grid_pooled \
        --out results/knn_subsample_cost.csv
    # in-distribution, secondary
    python pipeline/05_baseline_knn.py --protocol loo --economy_metric cost \
        --train_gt_dir data/ground_truth/main_grid_pooled --out results/knn_loo_cost.csv
    # out-of-distribution
    python pipeline/05_baseline_knn.py --protocol ood --economy_metric cost \
        --train_gt_dir data/ground_truth/main_grid_pooled \
        --test_gt_dir data/ground_truth/ood_span_pooled --out results/knn_ood_span_cost.csv

Writes: --out (aggregated over repetitions: mean and sd per protocol/size/k/operator) and
<out stem>_per_rep.csv (one row per repetition), plus <out stem>_meta.json. --out_detail DIR
additionally dumps per-context rows (needed to restrict an RL comparison to the same contexts).
"""
import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

WEIGHT_EPS = 1e-3
GEOM = ("h", "b", "tf", "tw")


def _load_eval_module():
    """Import pipeline/03_evaluate_agent.py (a digit-leading name) to reuse its exact
    ground_truth_optimum / summarise / OPERATOR_MODES so metrics are defined in ONE place."""
    spec = importlib.util.spec_from_file_location(
        "evaluate_agent", os.path.join(REPO, "pipeline", "03_evaluate_agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class KNNDesigner:
    """Stores solved contexts; predicts a full design for a new (span_m, load)."""

    def __init__(self, train_opt: pd.DataFrame, k: int):
        if len(train_opt) < 1:
            raise ValueError("empty training set")
        self.k = int(k)
        self.span = train_opt["span_m"].to_numpy(float)
        self.load = train_opt["load_kN_per_m"].to_numpy(float)
        self.lo = np.array([self.span.min(), self.load.min()])
        self.hi = np.array([self.span.max(), self.load.max()])
        self.scale = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.X = (np.c_[self.span, self.load] - self.lo) / self.scale
        self.grade = train_opt["grade"].to_numpy(float)
        self.stype = train_opt["section_type"].astype(str).to_numpy()
        self.logg = np.log(train_opt[list(GEOM)].to_numpy(float))

    def predict(self, span_m: float, load: float) -> dict:
        q = (np.array([span_m, load]) - self.lo) / self.scale
        d = np.linalg.norm(self.X - q, axis=1)
        idx = np.argsort(d, kind="stable")[: min(self.k, len(d))]
        w = 1.0 / (d[idx] + WEIGHT_EPS)
        votes = {}
        for j, wj in zip(idx, w):
            key = (self.grade[j], self.stype[j])
            votes[key] = votes.get(key, 0.0) + wj
        # tie -> the pair of the nearest neighbour (idx is distance-ordered)
        best = max(votes.values())
        winner = next((self.grade[j], self.stype[j]) for j in idx
                      if votes[(self.grade[j], self.stype[j])] == best)
        keep = [i for i, j in enumerate(idx) if (self.grade[j], self.stype[j]) == winner]
        wk = w[keep] / w[keep].sum()
        g = np.exp((self.logg[idx[keep]] * wk[:, None]).sum(axis=0))
        out = dict(zip(GEOM, map(float, g)))
        out["fy"], out["section_type"] = float(winner[0]), winner[1]
        return out


def make_env(economy_metric: str):
    from hssbeamgen.envs.hss_env import HSSBeamEnv
    from hssbeamgen.train_utils import env_kwargs, resolve_config
    cfg = resolve_config(os.path.join(REPO, "configs", "rl_final.yaml"), "ppo")
    cfg["economy_metric"] = economy_metric
    return HSSBeamEnv(**env_kwargs(cfg, reward_mode="feasibility_gated"))


def score(env, ev, designer: KNNDesigner, test_opt: pd.DataFrame, metric: str, storey: int):
    """-> {mode: per-context DataFrame}. Mirrors 03's per-context row schema."""
    from hssbeamgen.algo.posthoc_operators import apply_operator
    rows = {m: [] for m in ev.OPERATOR_MODES}
    for _, r in test_opt.iterrows():
        design = designer.predict(r["span_m"], r["load_kN_per_m"])
        for m in ev.OPERATOR_MODES:
            res = apply_operator(env, r["span_m"] * 1000.0, r["load_kN_per_m"], design,
                                 metric=metric, storey=storey, mode=m)
            gap = (res[metric] - r[metric]) / r[metric] if res["feasible"] else np.nan
            rows[m].append(dict(
                span_m=r["span_m"], load_kN_per_m=r["load_kN_per_m"],
                optimal=r[metric], optimal_grade=r["grade"], optimal_type=r["section_type"],
                achieved=res[metric], gap=gap, feasible=res["feasible"],
                grade=res["fy"], section_type=res["section_type"],
                grade_match=float(res["fy"] == r["grade"]), utilization=res["utilization"],
                adjusted=res["adjusted"], n_ec3=res["n_ec3"]))
    return {m: pd.DataFrame(v) for m, v in rows.items()}


def splits(protocol, train_opt, test_opt, sizes, reps, seed0):
    """Yield (n_train, rep, train_df, test_df | None-for-loo)."""
    n = len(train_opt)
    if protocol == "ood":
        yield n, 0, train_opt, test_opt
    elif protocol == "subsample":
        for size in sizes:
            if not 1 <= size < n:
                raise ValueError(f"--sizes must be in [1, {n - 1}], got {size}")
            for rep in range(reps):
                perm = np.random.default_rng(seed0 + rep).permutation(n)
                yield size, rep, train_opt.iloc[perm[:size]], train_opt.iloc[perm[size:]]
    elif protocol == "loo":
        for i in range(n):  # one split per held-out context
            yield n - 1, i, train_opt.drop(train_opt.index[i]), train_opt.iloc[[i]]
    else:
        raise ValueError(protocol)


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=REPO, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", required=True, choices=("loo", "subsample", "ood"))
    p.add_argument("--economy_metric", default="cost", choices=("mass", "cost", "co2"))
    p.add_argument("--train_gt_dir", required=True, help="POOLED dir providing the labelled contexts")
    p.add_argument("--test_gt_dir", default=None, help="POOLED dir for the test contexts (ood only)")
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    p.add_argument("--headline_k", type=int, default=3)
    p.add_argument("--sizes", type=int, nargs="+", default=[36, 72, 108])
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--storey", type=int, default=20)
    p.add_argument("--n_contexts", type=int, default=None,
                   help="smoke test: use only this many randomly chosen training contexts")
    p.add_argument("--out", required=True)
    p.add_argument("--out_detail", default=None)
    a = p.parse_args()

    for d in (a.train_gt_dir, a.test_gt_dir):
        if d is not None and not d.rstrip("/").endswith("_pooled"):
            sys.exit(f"{d!r}: evaluate against the _pooled ground-truth directories only")
    if a.protocol == "ood" and not a.test_gt_dir:
        sys.exit("--protocol ood needs --test_gt_dir")
    if a.headline_k not in a.k:
        sys.exit("--headline_k must be one of --k")

    ev = _load_eval_module()
    train_opt = ev.ground_truth_optimum(a.train_gt_dir, a.economy_metric)
    test_opt = ev.ground_truth_optimum(a.test_gt_dir, a.economy_metric) if a.test_gt_dir else None
    if a.protocol == "ood" and (test_opt is None or len(test_opt) == 0):
        sys.exit(f"{a.test_gt_dir}: no feasible contexts for {a.economy_metric} (untestable)")
    if a.n_contexts is not None and a.n_contexts < len(train_opt):
        train_opt = train_opt.sample(n=a.n_contexts, random_state=a.seed0).reset_index(drop=True)
        if test_opt is not None:
            test_opt = test_opt.head(a.n_contexts)
    if a.protocol == "subsample" and a.n_contexts is not None:
        a.sizes = [max(1, min(s, len(train_opt) - 1)) for s in a.sizes[:1]]
        a.reps = min(a.reps, 2)

    env = make_env(a.economy_metric)
    t0 = time.time()
    per_rep, detail = [], {}
    for k in a.k:
        loo_parts = {m: [] for m in ev.OPERATOR_MODES}
        for n_train, rep, tr, te in splits(a.protocol, train_opt, test_opt, a.sizes, a.reps, a.seed0):
            res = score(env, ev, KNNDesigner(tr, k), te, a.economy_metric, a.storey)
            if a.protocol == "loo":  # one summary over ALL held-out contexts, not per split
                for m in ev.OPERATOR_MODES:
                    loo_parts[m].append(res[m])
                continue
            for m in ev.OPERATOR_MODES:
                per_rep.append(dict(protocol=a.protocol, k=k, n_train=n_train, rep=rep,
                                    **ev.summarise(res[m], m)))
                detail[(k, n_train, rep, m)] = res[m]
        if a.protocol == "loo":
            for m in ev.OPERATOR_MODES:
                df = pd.concat(loo_parts[m], ignore_index=True)
                per_rep.append(dict(protocol="loo", k=k, n_train=len(train_opt) - 1, rep=0,
                                    **ev.summarise(df, m)))
                detail[(k, len(train_opt) - 1, 0, m)] = df
    wall = time.time() - t0

    per = pd.DataFrame(per_rep)
    keys = ["protocol", "k", "n_train", "operator_mode"]
    vals = [c for c in per.columns if c not in keys + ["rep", "n"]]
    agg = per.groupby(keys)[vals].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    agg.insert(4, "n_reps", per.groupby(keys).size().to_numpy())
    agg["headline"] = agg["k"] == a.headline_k

    stem = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    agg.to_csv(a.out, index=False)
    per.to_csv(stem + "_per_rep.csv", index=False)
    if a.out_detail:
        os.makedirs(a.out_detail, exist_ok=True)
        for (k, n_train, rep, m), df in detail.items():
            df.to_csv(os.path.join(a.out_detail, f"knn_{a.protocol}_k{k}_n{n_train}_rep{rep}_{m}.csv"), index=False)
    gt_meta = {}
    for d in (a.train_gt_dir, a.test_gt_dir):
        if d and os.path.exists(os.path.join(d, "meta.json")):
            gt_meta[d] = json.load(open(os.path.join(d, "meta.json")))
    json.dump(dict(args=vars(a), git_commit=_git("rev-parse", "HEAD"),
                   git_dirty=bool(_git("status", "--porcelain")), wall_time_s=wall,
                   n_train_pool=len(train_opt), generated_utc=datetime.now(timezone.utc).isoformat(),
                   python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   ground_truth_meta=gt_meta), open(stem + "_meta.json", "w"), indent=1, default=str)

    show = agg[agg.operator_mode == "scale+thin"]
    cols = ["protocol", "k", "n_train", "n_reps", "feasibility_mean", "cost_ratio_mean_mean",
            "cost_ratio_mean_std", "grade_match_mean"]
    print(show[cols].to_string(index=False))
    print(f"\nwall {wall:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
