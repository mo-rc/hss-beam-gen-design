"""
pipeline/14_deflection_sensitivity.py -- how much would a tighter serviceability (deflection) limit move the reference?

No search, no training. The reference designs in the pooled ground truth were found with the limit L/250
hard-coded in hss_env.py (`deflection_util = delta / (L / 250.0)`, utilisation = max(moment, deflection)).
Deflection utilisation is exactly proportional to 250 / n for a limit L/n, so each stored design can be
re-checked against L/n without re-analysing anything else:  a design stays feasible at L/n iff
deflection_util(L/250) * n / 250 <= 1.001  (the tolerance used for feasibility everywhere else).

Per objective and limit it counts, over the reference designs (one per (span, load) context, the same
selection as 03_evaluate_agent.ground_truth_optimum) and over all stored (grade, type) optima, how many would
violate. It also gives a lower bound on the uniform scale factor the `scale` operator would need to repair
a violating reference design (deflection ~ s^-4 for a section scaled by s; other checks and box clipping ignored).

    python pipeline/14_deflection_sensitivity.py --ground_truth_dir data/ground_truth/main_grid_pooled \
        --out results/deflection_sensitivity.csv
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

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OBJECTIVES = ("cost", "mass", "co2")
TOL = 1.001
STOREY = 20  # as in 01 / 01b / 01c; the load of a context is used directly


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def deflection_utils(df, objective):
    """moment and deflection utilisation (limit L/250) of every stored design, from the env's own EC3 analysis."""
    from hssbeamgen.algo.ga_search import _make_probe_env
    env = _make_probe_env(objective)
    mu, du = [], []
    for r in df.itertuples():
        env.h, env.b, env.tf, env.tw = float(r.h), float(r.b), float(r.tf), float(r.tw)
        env.fy, env.section_type = float(r.grade), r.section_type
        env.span, env.load, env.storey = float(r.span_m) * 1000.0, float(r.load_kN_per_m), STOREY
        dbg = env._ec3_analysis()[-1]
        mu.append(dbg["moment_util"])
        du.append(dbg["deflection_util"])
    return np.array(mu), np.array(du)


def reference_index(df, objective):
    """Same selection as 03_evaluate_agent.ground_truth_optimum: best row per (span, load)."""
    return df.groupby(["span_m", "load_kN_per_m"])[objective].idxmin().to_numpy()


def sensitivity(df, objective, limits):
    mu, du = deflection_utils(df, objective)
    ref = reference_index(df, objective)
    rows = []
    for n in limits:
        k = n / 250.0
        v_ref, v_all = du[ref] * k, du * k
        bad = v_ref > TOL
        rows.append(dict(
            objective=objective, limit=f"L/{n}", n_contexts=len(ref),
            n_ref_deflection_governs_at_250=int((du[ref] >= mu[ref] * 0.999).sum()) if n == 250 else np.nan,
            n_ref_violating=int(bad.sum()), frac_ref_violating=float(bad.mean()),
            n_stored_designs=len(df), n_stored_violating=int((v_all > TOL).sum()),
            frac_stored_violating=float((v_all > TOL).mean()),
            scale_lb_median=float(np.median(v_ref[bad] ** 0.25)) if bad.any() else np.nan,
            scale_lb_max=float(np.max(v_ref[bad] ** 0.25)) if bad.any() else np.nan))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ground_truth_dir", default=os.path.join("data", "ground_truth", "main_grid_pooled"))
    ap.add_argument("--limits", type=int, nargs="+", default=[250, 300, 360, 500], help="n in L/n")
    ap.add_argument("--out", default=os.path.join("results", "deflection_sensitivity.csv"))
    a = ap.parse_args()
    parts, hashes = [], {}
    for o in OBJECTIVES:
        path = os.path.join(a.ground_truth_dir, f"ec3_optimal_designs_{o}.csv")
        hashes[os.path.basename(path)] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        parts.append(sensitivity(pd.read_csv(path), o, a.limits))
    out = pd.concat(parts, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    meta = dict(script="pipeline/14_deflection_sensitivity.py", args=vars(a), input_sha256=hashes,
                git_commit=_git("rev-parse", "HEAD"),
                git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
                generated_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                numpy=np.__version__, pandas=pd.__version__)
    with open(os.path.splitext(a.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
