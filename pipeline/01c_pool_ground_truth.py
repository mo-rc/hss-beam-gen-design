"""Step 1c - pool the ground truth across the three objective searches.

01_generate_ground_truth.py runs an independent GA per objective (mass, cost, co2). Every design
it finds is a valid, feasible point for ALL objectives, so a design found while optimising mass
may be cheaper than the design found while optimising cost (GA search noise). This script builds
the best-known reference: for each (span, load, grade, section_type) and each objective, it keeps
the best feasible design found by ANY of the three searches.

What it does NOT do: no new search, no change to the EC3 physics or to the objective definitions.
It only takes a minimum over designs that already exist. The raw files are left untouched; the
pooled files go to a sibling directory with the same schema, so downstream scripts only need a
different --gt_dir. The `governing` column records provenance ("GA_cost" = the objective's own
search; "pooled:GA_mass" = taken from another objective's search).

Every design that changes source is re-evaluated through the environment before it is accepted
(values must reproduce, design must be feasible); the script aborts otherwise.

Outputs in --out (default: <dir>_pooled):
    ec3_optimal_designs_{mass,cost,co2}.csv   same columns as the raw files
    meta.json                                 verifier-compatible provenance (01b runs on this directory)
    pooling_report.json                       counts and improvement statistics, input file hashes

Usage:
    python pipeline/01c_pool_ground_truth.py --dir data/ground_truth/main_grid
    python pipeline/01b_verify_ground_truth.py --dir data/ground_truth/main_grid_pooled
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import hssbeamgen
from hssbeamgen.envs.hss_env import HSSBeamEnv

METRICS = ["mass", "cost", "co2"]
KEY = ["span_m", "load_kN_per_m", "grade", "section_type"]
COLUMNS = ["span_m", "load_kN_per_m", "grade", "section_type", "h", "b", "tf", "tw",
           "util", "mass", "cost", "co2", "governing"]
STOREY = 20
REL_TOL = 1e-9


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def _reevaluate(env, r):
    env.h, env.b, env.tf, env.tw = float(r.h), float(r.b), float(r.tf), float(r.tw)
    env.fy = float(r.grade)
    env.section_type = str(r.section_type)
    env.span, env.load, env.storey = float(r.span_m) * 1000.0, float(r.load_kN_per_m), STOREY
    util, mass, penalty, class_loss, _chi, _dbg = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    viol = env._constraint_violations(util, class_loss, penalty)
    return util, mass, cost, co2, all(v <= 1e-3 for v in viol.values())


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="raw ground-truth directory from 01_generate_ground_truth.py")
    p.add_argument("--out", default=None, help="output directory (default: <dir>_pooled)")
    a = p.parse_args()
    out = a.out or a.dir.rstrip("/\\") + "_pooled"

    src_meta_path = os.path.join(a.dir, "meta.json")
    if not os.path.exists(src_meta_path):
        sys.exit(f"missing {src_meta_path}; run pipeline/01_generate_ground_truth.py first")
    with open(src_meta_path) as f:
        src_meta = json.load(f)

    raw, in_hash = {}, {}
    for m in METRICS:
        path = os.path.join(a.dir, f"ec3_optimal_designs_{m}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, float_precision="round_trip")
            if list(df.columns) != COLUMNS:
                sys.exit(f"{path}: unexpected columns {list(df.columns)}")
            raw[m], in_hash[os.path.basename(path)] = df, _sha256(path)
    if not raw:
        sys.exit("no ec3_optimal_designs_*.csv found")
    os.makedirs(out, exist_ok=True)

    frames = []
    for m, df in raw.items():
        d = df.copy()
        d["_src"] = m
        frames.append(d)
    allrows = pd.concat(frames, ignore_index=True)

    env_cache = {}
    stats, pooled = {}, {}
    for m in raw:
        # best design per (context, grade, type) across all searches; ties go to the objective's own search
        allrows["_prio"] = (allrows["_src"] != m).astype(int)
        best = (allrows.sort_values(KEY + [m, "_prio"], kind="mergesort")
                       .groupby(KEY, sort=False, as_index=False).head(1).copy())
        foreign = best["_src"] != m
        best.loc[foreign, "governing"] = "pooled:" + best.loc[foreign, "governing"].astype(str)

        # re-evaluate every design that came from another objective's search
        if m not in env_cache:
            env_cache[m] = HSSBeamEnv(reward_mode="lagrangian", economy_metric=m)
        env = env_cache[m]
        worst = 0.0
        for r in best[foreign].itertuples():
            util, mass, cost, co2, feas = _reevaluate(env, r)
            dev = max(_rel(util, r.util), _rel(mass, r.mass), _rel(cost, r.cost), _rel(co2, r.co2))
            worst = max(worst, dev)
            if dev > REL_TOL or not feas:
                sys.exit(f"[{m}] pooled design failed re-evaluation at {tuple(getattr(r, k) for k in KEY)}: "
                         f"deviation {dev:.2e}, feasible={feas}")

        res = best[COLUMNS].sort_values(KEY).reset_index(drop=True)
        pooled[m] = res
        res.to_csv(os.path.join(out, f"ec3_optimal_designs_{m}.csv"), index=False)

        own = raw[m].set_index(KEY)[m]
        new = res.set_index(KEY)[m]
        added = int((~new.index.isin(own.index)).sum())          # keys the own search never found feasible
        common = new.index.intersection(own.index)
        row_imp = (own.loc[common] / new.loc[common] - 1.0)
        ctx = ["span_m", "load_kN_per_m"]
        c_own = raw[m].groupby(ctx)[m].min()
        c_new = res.groupby(ctx)[m].min()
        c_imp = (c_own.reindex(c_new.index) / c_new - 1.0).dropna()
        best_grade = res.loc[res.groupby(ctx)[m].idxmin()].grade.value_counts().sort_index()
        stats[m] = dict(
            rows_own_search=int(len(own)), rows_pooled=int(len(res)), rows_added=added,
            rows_improved=int((row_imp > 1e-9).sum()), row_improvement_max=float(row_imp.max()),
            rows_from_other_search=int(foreign.sum()),
            source_of_foreign_rows={k: int(v) for k, v in best.loc[foreign, "_src"].value_counts().items()},
            contexts_feasible_before=int(len(c_own)), contexts_feasible_after=int(len(c_new)),
            contexts_improved=int((c_imp > 1e-9).sum()),
            context_improvement_mean=float(c_imp.mean()), context_improvement_max=float(c_imp.max()),
            max_reevaluation_deviation=float(worst),
            optimal_grade_counts={int(k): int(v) for k, v in best_grade.items()},
        )
        print(f"[{m}] rows {len(own)} -> {len(res)} (+{added} added, {stats[m]['rows_improved']} improved, "
              f"{int(foreign.sum())} taken from another search); contexts improved "
              f"{stats[m]['contexts_improved']}/{len(c_new)}, mean {stats[m]['context_improvement_mean']:.3%}, "
              f"max {stats[m]['context_improvement_max']:.2%}; grades {stats[m]['optimal_grade_counts']}")

    # verifier-compatible meta.json for the pooled directory
    pkg = os.path.dirname(hssbeamgen.__file__)
    summary = {}
    for m, res in pooled.items():
        n_ctx = int(res[["span_m", "load_kN_per_m"]].drop_duplicates().shape[0])
        summary[m] = dict(rows=int(len(res)), searches=src_meta["summary"][m]["searches"],
                          contexts_with_feasible_design=n_ctx,
                          contexts_in_grid=src_meta["summary"][m]["contexts_in_grid"])
    meta = dict(
        script="pipeline/01c_pool_ground_truth.py", grid=src_meta["grid"], grid_definition=src_meta["grid_definition"],
        source_dir=os.path.abspath(a.dir), source_meta=src_meta, input_sha256=in_hash,
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        sha256=dict(hss_env=_sha256(os.path.join(pkg, "envs", "hss_env.py")),
                    ga_search=_sha256(os.path.join(pkg, "algo", "ga_search.py"))),
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        load_unit=src_meta.get("load_unit"), summary=summary,
        note="Best-known reference: per-objective minimum over the designs found by all objective searches.",
    )
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "pooling_report.json"), "w") as f:
        json.dump(dict(input_sha256=in_hash, per_objective=stats), f, indent=2)
    print(f"wrote {out}/ (3 CSVs, meta.json, pooling_report.json); raw files in {a.dir} untouched")


if __name__ == "__main__":
    main()
