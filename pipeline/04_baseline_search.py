"""
pipeline/04_baseline_search.py -- budget-matched per-context search baselines (CPU only):
GA, differential evolution (DE) and random search, re-solved from scratch for every context.

WHAT IT MEASURES
----------------
For each context in a POOLED ground-truth directory, each method gets a fixed budget of
B EC3 evaluations and returns its best design; the gap to the pooled best-known optimum is then
computed exactly as for the policy (pipeline/03_evaluate_agent.py's summary function is reused).
Two rows per (method, budget, seed):
  none        the search's own best design. Evaluations = B.
  scale+thin  the same post-hoc operators applied to the policy's output, applied to the
              search result (hssbeamgen.algo.posthoc_operators), so the comparison with the
              policy is like for like. Evaluations = B + the operator's analyses (reported in
              ec3_evals_per_design, so gap-versus-evaluations curves are honest).
(`scale` is also written.) The policy's reference points are ~40 / 57 / 100-112 evaluations per
design (none / scale / scale+thin); the kNN needs 1 / ~18 / ~60. Read each method's gap at the
same TOTAL evaluations from ec3_evals_per_design.

BUDGET RULE (fixed in advance)
------------------------------
GA: pop_size = clip(round(sqrt(B)), 6, 60), n_generations = B // pop_size (so actual evaluations
<= B, reported as n_evals_search). DE: same pop rule, exact budget. Random: exactly B samples.
Everything else is the stock setting of hssbeamgen/algo/ga_search.py. No per-budget tuning.
Seeds: --seeds independent repetitions (seed s for every context uses seed s*100000 + context
index), summarised per seed over contexts, like the RL seeds.

The full-budget row (B = 4800) is NOT the ground truth: the reference is the pooled best over 12
fixed (grade, section type) GA searches x 2 restarts x 3 objective searches, so a free-grade
search can land slightly BELOW the reference (negative gap contribution); this is reported as is.

USAGE
-----
    # smoke test: 6 contexts, 2 seeds, 3 budgets (seconds)
    python pipeline/04_baseline_search.py --economy_metric cost \
        --gt_dir data/ground_truth/main_grid_pooled --n_contexts 6 --seeds 2 \
        --budgets 40 112 400 --out /tmp/search_smoke.csv
    # full
    python pipeline/04_baseline_search.py --economy_metric cost \
        --gt_dir data/ground_truth/main_grid_pooled --workers 4 --out results/search_main_cost.csv
Writes --out (mean and sd over seeds per method/budget/operator), <stem>_per_seed.csv and
<stem>_meta.json.
"""
import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

METHODS = ("ga", "de", "random")
DEFAULT_BUDGETS = (20, 40, 60, 112, 200, 400, 1000, 4800)


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_agent", os.path.join(REPO, "pipeline", "03_evaluate_agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ga_pop(budget: int) -> int:
    return int(np.clip(round(np.sqrt(budget)), 6, 60))


def run_search(method, span_mm, load, storey, metric, budget, seed):
    from hssbeamgen.algo.de_search import de_design
    from hssbeamgen.algo.ga_search import ga_design, random_search_design
    if method == "ga":
        pop = ga_pop(budget)
        return ga_design(span_mm, load, storey, metric, pop_size=pop,
                         n_generations=max(1, budget // pop), seed=seed)
    if method == "de":
        return de_design(span_mm, load, storey, metric, n_evaluations=budget, seed=seed)
    if method == "random":
        return random_search_design(span_mm, load, storey, metric, n_evaluations=budget, seed=seed)
    raise ValueError(method)


def _work(job):
    """One (context, method, budget, seed): search once, then score under every operator mode."""
    from hssbeamgen.algo.posthoc_operators import apply_operator
    (ci, ctx, method, budget, seed, metric, storey, modes) = job
    env = _work.env if hasattr(_work, "env") else None
    if env is None:
        from hssbeamgen.envs.hss_env import HSSBeamEnv
        from hssbeamgen.train_utils import env_kwargs, resolve_config
        cfg = resolve_config(os.path.join(REPO, "configs", "rl_final.yaml"), "ppo")
        cfg["economy_metric"] = metric
        env = _work.env = HSSBeamEnv(**env_kwargs(cfg, reward_mode="feasibility_gated"))
    res = run_search(method, ctx["span_m"] * 1000.0, ctx["load_kN_per_m"], storey, metric,
                     budget, seed * 100000 + ci)
    design = {k: res[k] for k in ("h", "b", "tf", "tw", "fy", "section_type")}
    out = []
    for m in modes:
        r = apply_operator(env, ctx["span_m"] * 1000.0, ctx["load_kN_per_m"], design,
                           metric=metric, storey=storey, mode=m)
        # `none` scores the search's own best design; operators add their own analyses on top.
        n_ec3 = res["n_evaluations"] + (0 if m == "none" else r["n_ec3"])
        gap = (r[metric] - ctx[metric]) / ctx[metric] if r["feasible"] else np.nan
        out.append((method, budget, seed, m, dict(
            span_m=ctx["span_m"], load_kN_per_m=ctx["load_kN_per_m"], optimal=ctx[metric],
            optimal_grade=ctx["grade"], optimal_type=ctx["section_type"], achieved=r[metric],
            gap=gap, feasible=r["feasible"], grade=r["fy"], section_type=r["section_type"],
            grade_match=float(r["fy"] == ctx["grade"]), utilization=r["utilization"],
            adjusted=r["adjusted"], n_ec3=n_ec3, n_evals_search=res["n_evaluations"],
            search_wall_s=res["wall_time_s"])))
    return out


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=REPO, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--economy_metric", default="cost", choices=("mass", "cost", "co2"))
    p.add_argument("--gt_dir", required=True, help="POOLED ground-truth directory")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--storey", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--n_contexts", type=int, default=None, help="smoke test: first N contexts only")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if not a.gt_dir.rstrip("/").endswith("_pooled"):
        sys.exit(f"{a.gt_dir!r}: evaluate against the _pooled ground-truth directories only")

    ev = _load_eval_module()
    opt = ev.ground_truth_optimum(a.gt_dir, a.economy_metric)
    if len(opt) == 0:
        sys.exit(f"{a.gt_dir}: no feasible contexts for {a.economy_metric} (untestable)")
    if a.n_contexts is not None:
        opt = opt.head(a.n_contexts)
    modes = tuple(ev.OPERATOR_MODES)
    ctxs = [r.to_dict() for _, r in opt.iterrows()]
    jobs = [(ci, c, m, b, s, a.economy_metric, a.storey, modes)
            for m in a.methods for b in a.budgets for s in range(a.seeds) for ci, c in enumerate(ctxs)]
    print(f"{len(ctxs)} contexts x {len(a.methods)} methods x {len(a.budgets)} budgets x {a.seeds} seeds "
          f"= {len(jobs)} searches", flush=True)

    t0 = time.time()
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_work, jobs, chunksize=max(1, len(jobs) // (a.workers * 20))))
    else:
        results = [_work(j) for j in jobs]
    wall = time.time() - t0

    groups = {}
    for res in results:
        for method, budget, seed, mode, row in res:
            groups.setdefault((method, budget, seed, mode), []).append(row)
    per_seed = []
    for (method, budget, seed, mode), rows in sorted(groups.items()):
        df = pd.DataFrame(rows)
        s = ev.summarise(df, mode)
        s["n_evals_search"] = df.n_evals_search.mean()
        s["search_wall_s_per_context"] = df.search_wall_s.mean()
        per_seed.append(dict(method=method, budget=budget, seed=seed, **s))
    per = pd.DataFrame(per_seed)
    keys = ["method", "budget", "operator_mode"]
    vals = [c for c in per.columns if c not in keys + ["seed", "n"]]
    agg = per.groupby(keys)[vals].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    agg.insert(3, "n_seeds", per.groupby(keys).size().to_numpy())

    stem = os.path.splitext(a.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    agg.to_csv(a.out, index=False)
    per.to_csv(stem + "_per_seed.csv", index=False)
    json.dump(dict(args=vars(a), git_commit=_git("rev-parse", "HEAD"),
                   git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")), wall_time_s=wall,
                   n_contexts=len(ctxs), generated_utc=datetime.now(timezone.utc).isoformat(),
                   python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                   ground_truth_meta=json.load(open(os.path.join(a.gt_dir, "meta.json")))
                   if os.path.exists(os.path.join(a.gt_dir, "meta.json")) else {}),
              open(stem + "_meta.json", "w"), indent=1, default=str)

    show = agg[agg.operator_mode.isin(["none", "scale+thin"])]
    cols = ["method", "budget", "operator_mode", "n_seeds", "ec3_evals_per_design_mean",
            "feasibility_mean", "cost_ratio_mean_mean", "cost_ratio_mean_std"]
    print(show[cols].to_string(index=False))
    print(f"\nwall {wall:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
