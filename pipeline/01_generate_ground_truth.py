"""Step 1 - ground-truth generation (GA search, per objective).

For every context (span, load, grade, section_type) on a grid, runs a GA over the four
continuous geometry genes (h, b, tf, tw) with grade and section type fixed, once per
objective (mass, cost, co2), keeping the best feasible design over `--n_restarts`.
The per-context optimum used for evaluation is later taken as the minimum over
(grade, section_type) - see hssbeamgen/algo/ga_search.py for the GA.

Why one search per objective: the mass-optimal geometry is not the cost- or CO2-optimal
geometry, so each objective gets its own ground truth.

Grids (span m, factored UDL kN/m):
    main       6-15  x  20-140   12 x 12   training range / in-distribution evaluation
    ood_span   16-22 x  20-140    8 x 8    span extrapolation
    ood_load   6-15  x 150-260    8 x 8    load extrapolation
    ood_joint  16-22 x 150-260    8 x 8    joint extrapolation (expected: no feasible design)
Any grid value can be overridden with --span_min_m/--span_max_m/--load_min/--load_max/
--n_spans/--n_loads.

Outputs in --out:
    ec3_optimal_designs_{mass,cost,co2}.csv   one row per feasible (context, grade, type)
    meta.json                                 provenance (git commit, args, versions, hashes,
                                              feasibility counts)

Usage:
    python pipeline/01_generate_ground_truth.py --grid main --out data/ground_truth/main_grid
    # smoke test (seconds):
    python pipeline/01_generate_ground_truth.py --grid main --n_spans 2 --n_loads 2 \
        --pop_size 20 --n_generations 20 --n_restarts 1 --metrics cost --out /tmp/gt_smoke
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool

import numpy as np
import pandas as pd

import hssbeamgen
from hssbeamgen.algo.ga_search import ga_design_fixed_grade

GRADES = [355, 460, 500, 550, 620, 690]
SECTION_TYPES = ["rolled", "welded"]
METRICS = ["mass", "cost", "co2"]
STOREY = 20  # kept for API compatibility with the env; not used by the UDL-based context

GRID_PRESETS = {
    "main":      dict(span_min_m=6.0,  span_max_m=15.0, load_min=20.0,  load_max=140.0, n_spans=12, n_loads=12),
    "ood_span":  dict(span_min_m=16.0, span_max_m=22.0, load_min=20.0,  load_max=140.0, n_spans=8,  n_loads=8),
    "ood_load":  dict(span_min_m=6.0,  span_max_m=15.0, load_min=150.0, load_max=260.0, n_spans=8,  n_loads=8),
    "ood_joint": dict(span_min_m=16.0, span_max_m=22.0, load_min=150.0, load_max=260.0, n_spans=8,  n_loads=8),
}
COLUMNS = ["span_m", "load_kN_per_m", "grade", "section_type", "h", "b", "tf", "tw",
           "util", "mass", "cost", "co2", "governing"]


def _solve(job):
    """One context, one objective -> best feasible GA result over restarts (or None)."""
    i, span_m, load, grade, stype, metric, pop, gens, restarts, seed = job
    best, best_val = None, np.inf
    for r in range(restarts):
        res = ga_design_fixed_grade(
            span_mm=span_m * 1000.0, load_kN_per_m=load, storey=STOREY, grade=grade,
            section_type=stype, economy_metric=metric, pop_size=pop, n_generations=gens,
            seed=seed * 100000 + i * 10 + r)
        if res["feasible"] and res[metric] < best_val:
            best, best_val = res, res[metric]
    if best is None:
        return None
    return dict(span_m=span_m, load_kN_per_m=load, grade=grade, section_type=stype,
                h=best["h"], b=best["b"], tf=best["tf"], tw=best["tw"], util=best["util"],
                mass=best["mass"], cost=best["cost"], co2=best["co2"], governing=f"GA_{metric}")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", required=True, choices=list(GRID_PRESETS))
    p.add_argument("--out", required=True)
    for k, t in [("span_min_m", float), ("span_max_m", float), ("load_min", float), ("load_max", float),
                 ("n_spans", int), ("n_loads", int)]:
        p.add_argument(f"--{k}", type=t, default=None, help="override the grid preset")
    p.add_argument("--pop_size", type=int, default=50)
    p.add_argument("--n_generations", type=int, default=80)
    p.add_argument("--n_restarts", type=int, default=2)
    p.add_argument("--metrics", nargs="+", default=METRICS, choices=METRICS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=1,
                   help="processes; results are identical for any value (seeds depend only on the context index)")
    a = p.parse_args()

    g = dict(GRID_PRESETS[a.grid])
    for k in g:
        if getattr(a, k) is not None:
            g[k] = getattr(a, k)
    spans = np.linspace(g["span_min_m"], g["span_max_m"], g["n_spans"])
    loads = np.linspace(g["load_min"], g["load_max"], g["n_loads"])
    contexts = [(float(s), float(l), gr, t) for s in spans for l in loads for gr in GRADES for t in SECTION_TYPES]
    n_grid = len(spans) * len(loads)
    os.makedirs(a.out, exist_ok=True)

    print(f"grid={a.grid} span {g['span_min_m']}-{g['span_max_m']} m x{g['n_spans']}, "
          f"load {g['load_min']}-{g['load_max']} kN/m x{g['n_loads']}  ->  {n_grid} (span, load) contexts")
    print(f"{len(contexts)} (context, grade, type) searches x {len(a.metrics)} metrics x {a.n_restarts} restarts "
          f"(pop={a.pop_size}, gens={a.n_generations}); est. ~{len(contexts)*len(a.metrics)*a.n_restarts*0.42/60/a.workers:.1f} min")

    t0 = time.time()
    summary = {}
    for metric in a.metrics:
        jobs = [(i, s, l, gr, t, metric, a.pop_size, a.n_generations, a.n_restarts, a.seed)
                for i, (s, l, gr, t) in enumerate(contexts)]
        if a.workers > 1:
            with Pool(a.workers) as pool:
                results = pool.map(_solve, jobs, chunksize=4)
        else:
            results = [_solve(j) for j in jobs]
        df = pd.DataFrame([r for r in results if r is not None], columns=COLUMNS)
        path = os.path.join(a.out, f"ec3_optimal_designs_{metric}.csv")
        df.to_csv(path, index=False)
        n_ctx_feasible = int(df[["span_m", "load_kN_per_m"]].drop_duplicates().shape[0])
        summary[metric] = dict(rows=len(df), searches=len(contexts), contexts_with_feasible_design=n_ctx_feasible,
                               contexts_in_grid=n_grid)
        print(f"[{metric}] {len(df)}/{len(contexts)} feasible (context, grade, type) rows; "
              f"{n_ctx_feasible}/{n_grid} contexts have >=1 feasible design -> {path}")
        if n_ctx_feasible == 0:
            print(f"[{metric}] WARNING: NO feasible design anywhere on grid '{a.grid}' inside the design-space box. "
                  f"An empty (header-only) CSV was written; this grid cannot be used for evaluation.")

    env_file = os.path.join(os.path.dirname(hssbeamgen.__file__), "envs", "hss_env.py")
    ga_file = os.path.join(os.path.dirname(hssbeamgen.__file__), "algo", "ga_search.py")
    meta = dict(
        script="pipeline/01_generate_ground_truth.py", grid=a.grid, grid_definition=g,
        args=vars(a), argv=sys.argv, generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        wall_time_min=round((time.time() - t0) / 60, 2),
        git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain")),
        sha256=dict(hss_env=_sha256(env_file), ga_search=_sha256(ga_file)),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        load_unit="kN/m (factored UDL); M_Ed = w L^2 / 8", summary=summary,
        note=("No feasible design inside the design-space box for any context: grid is untestable."
              if all(v["contexts_with_feasible_design"] == 0 for v in summary.values()) else ""),
    )
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta.json written; total {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    main()
