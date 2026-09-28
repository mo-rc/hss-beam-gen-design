"""Step 1b - verify a ground-truth directory produced by 01_generate_ground_truth.py.

Checks (each printed as PASS / WARN / FAIL):
  1. meta.json: present, git_dirty is False, the sha256 of hss_env.py / ga_search.py recorded
     at generation time still matches the files in this checkout, summary counts match the CSVs.
  2. CSV schema: expected columns, no NaN, no duplicate (span, load, grade, section_type),
     values inside the design box, grades / section types valid.
  3. Independent re-evaluation: a random sample of stored designs is re-run through the
     environment's EC3 functions; utilisation, mass, cost, CO2 must reproduce, and the design
     must be feasible under the environment's own constraint check.
  4. Cross-objective consistency: for the same (context, grade, type), the cost-optimal design
     should not cost more than the mass-optimal design (and likewise for CO2), up to GA noise.
  5. Coverage: which (span, load) contexts have no feasible design, and the optimal-grade
     distribution per objective (minimum over grade and section type).

Writes verification_report.json into the ground-truth directory. Exit code 1 if any FAIL.

Usage:
    python pipeline/01b_verify_ground_truth.py --dir data/ground_truth/main_grid
    python pipeline/01b_verify_ground_truth.py --dir data/ground_truth/ood_joint   # empty grid is fine
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

import hssbeamgen
from hssbeamgen.envs.hss_env import HSSBeamEnv

GRADES = [355, 460, 500, 550, 620, 690]
SECTION_TYPES = ["rolled", "welded"]
METRICS = ["mass", "cost", "co2"]
COLUMNS = ["span_m", "load_kN_per_m", "grade", "section_type", "h", "b", "tf", "tw",
           "util", "mass", "cost", "co2", "governing"]
BOUNDS = dict(h=(250.0, 750.0), b=(120.0, 300.0), tf=(8.0, 35.0), tw=(6.0, 25.0))
KEY = ["span_m", "load_kN_per_m", "grade", "section_type"]
STOREY = 20

results = []  # (level, name, detail)


def report(level, name, detail=""):
    results.append((level, name, detail))
    print(f"[{level}] {name}" + (f" - {detail}" if detail else ""))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def reevaluate(env, row):
    env.h, env.b, env.tf, env.tw = float(row.h), float(row.b), float(row.tf), float(row.tw)
    env.fy = float(row.grade)
    env.section_type = str(row.section_type)
    env.span, env.load, env.storey = float(row.span_m) * 1000.0, float(row.load_kN_per_m), STOREY
    util, mass, penalty, class_loss, _chi, _dbg = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    viol = env._constraint_violations(util, class_loss, penalty)
    feasible = all(v <= 1e-3 for v in viol.values())
    return util, mass, cost, co2, feasible


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="ground-truth directory (contains meta.json and the CSVs)")
    p.add_argument("--n_sample", type=int, default=200, help="rows per objective to re-evaluate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rel_tol", type=float, default=1e-6, help="re-evaluation reproduction tolerance")
    p.add_argument("--xobj_tol", type=float, default=0.02,
                   help="allowed relative excess of cost-optimal over mass-optimal cost (GA noise)")
    a = p.parse_args()

    # ---------------------------------------------------------------- 1. meta.json
    meta_path = os.path.join(a.dir, "meta.json")
    meta = None
    if not os.path.exists(meta_path):
        report("FAIL", "meta.json present", meta_path)
    else:
        with open(meta_path) as f:
            meta = json.load(f)
        report("PASS", "meta.json present", f"grid={meta.get('grid')} commit={str(meta.get('git_commit'))[:8]}")
        report("PASS" if meta.get("git_dirty") is False else "FAIL", "generated from a clean git tree",
               f"git_dirty={meta.get('git_dirty')}")
        pkg = os.path.dirname(hssbeamgen.__file__)
        for key, rel in [("hss_env", "envs/hss_env.py"), ("ga_search", "algo/ga_search.py")]:
            now = sha256(os.path.join(pkg, rel))
            then = meta.get("sha256", {}).get(key)
            report("PASS" if now == then else "FAIL", f"{rel} unchanged since generation",
                   "" if now == then else "checksum differs: ground truth was made with a different file version")
        report("INFO", "environment at generation", f"python={meta.get('python')} numpy={meta.get('numpy')} "
                                                       f"pandas={meta.get('pandas')}")

    # ---------------------------------------------------------------- 2. CSV schema
    dfs = {}
    for m in METRICS:
        path = os.path.join(a.dir, f"ec3_optimal_designs_{m}.csv")
        if not os.path.exists(path):
            report("INFO", f"{m} file", "not present (objective not generated)")
            continue
        df = pd.read_csv(path, float_precision="round_trip")  # exact float parsing (default parser is not)
        dfs[m] = df
        if list(df.columns) != COLUMNS:
            report("FAIL", f"[{m}] columns", f"got {list(df.columns)}")
            continue
        report("PASS", f"[{m}] columns", f"{len(df)} rows")
        if len(df) == 0:
            report("INFO", f"[{m}] empty file", "no feasible design on this grid (expected for ood_joint)")
            continue
        report("PASS" if not df.isna().any().any() else "FAIL", f"[{m}] no NaN")
        report("PASS" if not df.duplicated(KEY).any() else "FAIL", f"[{m}] no duplicate keys")
        report("PASS" if df.grade.isin(GRADES).all() else "FAIL", f"[{m}] grades valid")
        report("PASS" if df.section_type.isin(SECTION_TYPES).all() else "FAIL", f"[{m}] section types valid")
        eps = 1e-6
        inside = all(df[k].between(lo - eps, hi + eps).all() for k, (lo, hi) in BOUNDS.items())
        report("PASS" if inside else "FAIL", f"[{m}] geometry inside design box")
        report("PASS" if (df.util <= 1.0 + 1e-3 + 1e-9).all() else "FAIL", f"[{m}] util <= 1.001",
               f"max util {df.util.max():.5f}")
        if meta and m in meta.get("summary", {}):
            ok = meta["summary"][m]["rows"] == len(df)
            report("PASS" if ok else "FAIL", f"[{m}] row count matches meta.json",
                   f"csv={len(df)} meta={meta['summary'][m]['rows']}")

    # ---------------------------------------------------------------- 3. re-evaluation
    rng = np.random.default_rng(a.seed)
    for m, df in dfs.items():
        if len(df) == 0:
            continue
        env = HSSBeamEnv(reward_mode="lagrangian", economy_metric=m)
        idx = rng.choice(len(df), size=min(a.n_sample, len(df)), replace=False)
        bad_repro, bad_feas, worst = 0, 0, 0.0
        for i in idx:
            r = df.iloc[i]
            util, mass, cost, co2, feas = reevaluate(env, r)
            rel = max(abs(util - r.util) / max(abs(r.util), 1e-9), abs(mass - r.mass) / max(abs(r.mass), 1e-9),
                      abs(cost - r.cost) / max(abs(r.cost), 1e-9), abs(co2 - r.co2) / max(abs(r.co2), 1e-9))
            worst = max(worst, rel)
            bad_repro += rel > a.rel_tol
            bad_feas += (not feas)
        report("PASS" if bad_repro == 0 else "FAIL", f"[{m}] re-evaluation reproduces stored values",
               f"{len(idx)} rows, worst relative deviation {worst:.2e}")
        report("PASS" if bad_feas == 0 else "FAIL", f"[{m}] re-evaluated designs are feasible",
               f"{bad_feas}/{len(idx)} infeasible")

    # ---------------------------------------------------------------- 4. cross-objective consistency
    for m in ("cost", "co2"):
        if m in dfs and "mass" in dfs and len(dfs[m]) and len(dfs["mass"]):
            j = dfs[m].merge(dfs["mass"], on=KEY, suffixes=("", "_mass"))
            excess = (j[m] / j[f"{m}_mass"] - 1.0)
            n_bad = int((excess > a.xobj_tol).sum())
            report("PASS" if n_bad == 0 else "WARN",
                   f"[{m}] {m}-optimal <= mass-optimal design's {m} (tol {a.xobj_tol:.0%})",
                   f"{n_bad}/{len(j)} exceed; worst {excess.max():+.2%} (GA search noise if small)")

    # ---------------------------------------------------------------- 5. coverage and grade distribution
    summary = {}
    for m, df in dfs.items():
        if meta and "grid_definition" in meta:
            g = meta["grid_definition"]
            spans = np.linspace(g["span_min_m"], g["span_max_m"], g["n_spans"])
            loads = np.linspace(g["load_min"], g["load_max"], g["n_loads"])
            n_grid = len(spans) * len(loads)
        else:
            n_grid = None
        if len(df) == 0:
            summary[m] = dict(rows=0, contexts_feasible=0, contexts_in_grid=n_grid)
            continue
        best = df.loc[df.groupby(["span_m", "load_kN_per_m"])[m].idxmin()]
        dist = {int(k): int(v) for k, v in best.grade.value_counts().sort_index().items()}
        types = {str(k): int(v) for k, v in best.section_type.value_counts().items()}
        summary[m] = dict(rows=len(df), contexts_feasible=len(best), contexts_in_grid=n_grid,
                          optimal_grade_counts=dist, optimal_section_type_counts=types)
        report("INFO", f"[{m}] feasible contexts", f"{len(best)}/{n_grid}")
        report("INFO", f"[{m}] optimal grade counts", str(dist))
        if n_grid and len(best) < n_grid:
            missing = pd.DataFrame(
                [(s, l) for s in spans for l in loads], columns=["span_m", "load_kN_per_m"]).merge(
                best[["span_m", "load_kN_per_m"]], how="left", indicator=True)
            missing = missing[missing["_merge"] == "left_only"]
            corner = ", ".join(f"({r.span_m:.1f} m, {r.load_kN_per_m:.0f})" for r in missing.itertuples())
            report("INFO", f"[{m}] contexts with no feasible design ({len(missing)})", corner[:300])
            summary[m]["infeasible_contexts"] = [(float(r.span_m), float(r.load_kN_per_m)) for r in missing.itertuples()]

    n_fail = sum(1 for lv, _, _ in results if lv == "FAIL")
    n_warn = sum(1 for lv, _, _ in results if lv == "WARN")
    out = dict(directory=os.path.abspath(a.dir), n_fail=n_fail, n_warn=n_warn, summary=summary,
               checks=[dict(level=lv, name=n, detail=d) for lv, n, d in results])
    with open(os.path.join(a.dir, "verification_report.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{n_fail} FAIL, {n_warn} WARN -> verification_report.json")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()