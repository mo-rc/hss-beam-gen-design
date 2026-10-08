"""Step 11 - build every paper table from the committed result files (figures: pipeline/10_make_figures.py).

Reads only `results/` (no training, no EC3 evaluation, no randomness) and writes to `--out`
(default `paper/`): `tables/*.csv` (numeric) and `tables/*.md` (formatted).
Every number in a table therefore traces to a file in `results/`, and the build can be re-run at
any time to check that the tables still match the data.

Conventions (same as the docs/results_*.md logs): gap = (achieved / best-known optimum - 1) in %,
averaged over the feasible contexts of a run; lower is better, 0% = reference optimum. The column
`cost_ratio_mean` in the evaluation CSVs is the ratio for the *evaluated objective* (cost, mass or
CO2). "+-" is the sample sd over seeds (RL, search) or over label seeds (kNN cheap labels), n = 5 or
3, descriptive only. Headline operator mode is `scale+thin`.

Usage:
    python pipeline/11_build_paper_tables.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPS = ["none", "scale", "scale+thin"]
HEAD = "scale+thin"
SEEDS = range(42, 47)
OBJ = ["cost", "mass", "co2"]
PPO_MAIN = {"cost": "2a_feasibility_gated", "mass": "2c_mass", "co2": "2c_co2"}


# ---------------------------------------------------------------- loading helpers
def _pct(ratio):
    return (np.asarray(ratio, dtype=float) - 1.0) * 100.0


def load_eval(res, stem, seeds=SEEDS):
    """Per-seed evaluation CSVs of one RL arm -> one DataFrame (columns of 03 plus `seed`)."""
    frames = []
    for s in seeds:
        d = pd.read_csv(os.path.join(res, f"{stem}_seed{s}_eval.csv"))
        d["seed"] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def ms(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else float("nan"))


def fmt_ms(mean, sd, nd=1):
    return f"{mean:.{nd}f}" if np.isnan(sd) else f"{mean:.{nd}f} ± {sd:.{nd}f}"


def rl_row(res, stem):
    """One RL arm summarised over seeds: gap for the three operator modes plus scale+thin metrics."""
    d = load_eval(res, stem)
    out = {}
    for op in OPS:
        x = d[d.operator_mode == op]
        m, s = ms(_pct(x.cost_ratio_mean))
        out[f"gap_{op}"], out[f"gap_{op}_sd"] = m, s
    x = d[d.operator_mode == HEAD]
    out.update(feasibility=x.feasibility.mean() * 100, within_110=x.within_110pct.mean() * 100,
               within_125=x.within_125pct.mean() * 100, grade_match=x.grade_match.mean() * 100,
               evals=x.ec3_evals_per_design.mean())
    return out


def _summary_table(res, arms):
    rows = []
    for label, stem in arms.items():
        r = rl_row(res, stem)
        rows.append({"arm": label, **r})
    return pd.DataFrame(rows)


def _format_summary(df):
    out = pd.DataFrame({"arm": df.arm})
    for op in OPS:
        out[f"gap {op} (%)"] = [fmt_ms(m, s) for m, s in zip(df[f"gap_{op}"], df[f"gap_{op}_sd"])]
    out["feasibility (%)"] = df.feasibility.map("{:.0f}".format)
    out["within 110% (%)"] = df.within_110.map("{:.1f}".format)
    out["within 125% (%)"] = df.within_125.map("{:.1f}".format)
    out["grade match (%)"] = df.grade_match.map("{:.0f}".format)
    out["EC3 evals/design"] = df.evals.map("{:.0f}".format)
    return out


# ---------------------------------------------------------------- tables
def table_rl_experiments(res):
    """2a reward modes, 2b algorithms, 2c objectives: per-arm summaries (raw numeric frames)."""
    return {
        "t1_reward_mode_2a": _summary_table(res, {"feasibility_gated": "2a_feasibility_gated",
                                                  "lagrangian": "2a_lagrangian", "shaped": "2a_shaped"}),
        "t2_algorithm_2b": _summary_table(res, {"PPO": "2a_feasibility_gated", "SAC": "2b_sac",
                                                "TD3": "2b_td3", "DDPG": "2b_ddpg"}),
        "t3_objective_transfer_2c": _summary_table(res, {"cost (2a)": "2a_feasibility_gated",
                                                         "mass": "2c_mass", "CO2": "2c_co2"}),
    }


def table_pairwise(res):
    """Holm-corrected pairwise comparisons of the three RL experiments (scale+thin)."""
    frames = []
    for exp, fname in [("2a reward mode", "2a_comparison.csv"), ("2b algorithm", "2b_comparison.csv"),
                       ("2c objective", "2c_comparison.csv")]:
        d = pd.read_csv(os.path.join(res, fname))
        d.insert(0, "experiment", exp)
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    return d[["experiment", "arm_a", "arm_b", "mean_diff_pct_points", "ci95_lo_pct_points",
              "ci95_hi_pct_points", "p_raw", "p_holm"]]


def table_ood(res):
    """PPO in distribution and on both OOD sets, per objective (scale+thin)."""
    rows = []
    for o in OBJ:
        for setname, stem in [("main grid", PPO_MAIN[o]), ("ood_span", f"ood_span_{o}"),
                              ("ood_load", f"ood_load_{o}")]:
            d = load_eval(res, stem)
            x = d[d.operator_mode == HEAD]
            m, s = ms(_pct(x.cost_ratio_mean))
            rows.append({"objective": o, "set": setname, "n_contexts": int(x.n.iloc[0]), "gap": m, "gap_sd": s,
                         "feasibility": x.feasibility.mean() * 100, "within_110": x.within_110pct.mean() * 100,
                         "grade_match": x.grade_match.mean() * 100, "evals": x.ec3_evals_per_design.mean()})
    return pd.DataFrame(rows)


def _search(res, name):
    d = pd.read_csv(os.path.join(res, f"search_{name}.csv"))
    d = d[d.operator_mode == HEAD].copy()
    d["gap"] = _pct(d.cost_ratio_mean_mean)
    d["gap_sd"] = d.cost_ratio_mean_std * 100.0
    return d


def table_search_budgets(res):
    """GA / DE / random: gap vs budget (scale+thin) for every objective on the main grid."""
    rows = []
    for o in OBJ:
        for _, r in _search(res, f"main_{o}").iterrows():
            rows.append({"objective": o, "method": r.method, "budget": int(r.budget), "gap": r.gap, "gap_sd": r.gap_sd,
                         "feasibility": r.feasibility_mean * 100, "evals": r.ec3_evals_per_design_mean})
    return pd.DataFrame(rows)


def _knn(res, fname, **flt):
    d = pd.read_csv(os.path.join(res, fname))
    d = d[(d.k == 3) & (d.operator_mode == HEAD)]
    for k, v in flt.items():
        d = d[d[k] == v]
    return d.iloc[0]


def _knn_cheap(res, B, prot, obj, n=None):
    """kNN with search labels: mean and sd of gap over the three label seeds."""
    gaps, feas, evals = [], [], []
    for s in range(3):
        d = pd.read_csv(os.path.join(res, f"knn_cheap_de{B}_s{s}_{prot}_{obj}.csv"))
        d = d[(d.k == 3) & (d.operator_mode == HEAD)]
        if n is not None:
            d = d[d.n_train == n]
        r = d.iloc[0]
        gaps.append(float(_pct(r.cost_ratio_mean_mean)))
        feas.append(r.feasibility_mean * 100)
        evals.append(r.ec3_evals_per_design_mean)
    m, sd = ms(gaps)
    return m, sd, float(np.mean(feas)), float(np.mean(evals))


def table_matched(res, search_budget=40):
    """Main grid, per objective: RL vs kNN (best-case and search labels) vs search at matched evaluations."""
    rows = []
    for o in OBJ:
        r = rl_row(res, PPO_MAIN[o])
        rows.append({"objective": o, "method": "PPO (5 seeds)", "gap": r["gap_scale+thin"], "gap_sd": r["gap_scale+thin_sd"],
                     "feasibility": r["feasibility"], "evals_per_design": r["evals"], "labelling_evals": 0})
        k = _knn(res, f"knn_loo_{o}.csv")
        rows.append({"objective": o, "method": "kNN, pooled labels (LOO)", "gap": float(_pct(k.cost_ratio_mean_mean)),
                     "gap_sd": float("nan"), "feasibility": k.feasibility_mean * 100,
                     "evals_per_design": k.ec3_evals_per_design_mean, "labelling_evals": np.nan})
        for B in (1000, 4800):
            m, sd, f, e = _knn_cheap(res, B, "loo", o)
            rows.append({"objective": o, "method": f"kNN, DE labels B={B} (LOO)", "gap": m, "gap_sd": sd,
                         "feasibility": f, "evals_per_design": e, "labelling_evals": 141 * B})
        s = _search(res, f"main_{o}")
        for meth in ("de", "ga", "random"):
            x = s[(s.method == meth) & (s.budget == search_budget)].iloc[0]
            rows.append({"objective": o, "method": f"{meth.upper() if meth != 'random' else 'random'} search, B={search_budget}",
                         "gap": x.gap, "gap_sd": x.gap_sd, "feasibility": x.feasibility_mean * 100,
                         "evals_per_design": x.ec3_evals_per_design_mean, "labelling_evals": 0})
    return pd.DataFrame(rows)


def table_knn_cheap(res):
    """kNN with search labels vs label budget, in distribution (n = 36 and LOO) and on the OOD sets."""
    rows = []
    for o in OBJ:
        rl_gap = rl_row(res, PPO_MAIN[o])["gap_scale+thin"]
        for label, prot, n, bp in [("LOO", "loo", None, "loo"), ("subsample n=36", "subsample", 36, "subsample"),
                                   ("OOD span", "ood_span", None, "ood_span"), ("OOD load", "ood_load", None, "ood_load")]:
            best = _knn(res, f"knn_{bp}_{o}.csv", **({"n_train": n} if n else {}))
            row = {"objective": o, "protocol": label, "best_case_gap": float(_pct(best.cost_ratio_mean_mean)),
                   "best_case_feas": best.feasibility_mean * 100}
            for B in (400, 1000, 4800):
                m, sd, f, _ = _knn_cheap(res, B, prot, o, n)
                row[f"B{B}_gap"], row[f"B{B}_sd"], row[f"B{B}_feas"] = m, sd, f
            if prot.startswith("ood"):
                row["ppo_gap"] = ms(_pct(load_eval(res, f"{prot}_{o}").query("operator_mode == @HEAD").cost_ratio_mean))[0]
            else:
                row["ppo_gap"] = rl_gap
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output
def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _write(outdir, name, raw, formatted=None, note=""):
    os.makedirs(outdir, exist_ok=True)
    raw.round(6).to_csv(os.path.join(outdir, f"{name}.csv"), index=False)
    f = formatted if formatted is not None else raw
    with open(os.path.join(outdir, f"{name}.md"), "w") as fh:
        fh.write(md_table(f))
        if note:
            fh.write("\n" + note + "\n")


def _fmt_generic(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: "" if np.isnan(v) else f"{v:.1f}")
    return out


def build_tables(res, outdir):
    t = os.path.join(outdir, "tables")
    for name, df in table_rl_experiments(res).items():
        _write(t, name, df, _format_summary(df), "Gap to best-known optimum, mean ± sd over 5 seeds (descriptive).")
    pw = table_pairwise(res)
    f = pw.copy()
    for c in ["mean_diff_pct_points", "ci95_lo_pct_points", "ci95_hi_pct_points"]:
        f[c] = f[c].map("{:+.1f}".format)
    for c in ["p_raw", "p_holm"]:
        f[c] = f[c].map("{:.3f}".format)
    _write(t, "t4_pairwise_rl", pw, f, "scale+thin; exact permutation test, Holm-corrected within each experiment; n = 5 seeds per arm.")

    ood = table_ood(res)
    f = ood.copy()
    f["gap"] = [fmt_ms(m, s) for m, s in zip(ood.gap, ood.gap_sd)]
    f = f.drop(columns="gap_sd")
    for c in ["feasibility", "within_110", "grade_match", "evals"]:
        f[c] = f[c].map("{:.1f}".format)
    _write(t, "t5_ood_ppo", ood, f, "PPO, scale+thin; gaps over feasible contexts only.")

    sb = table_search_budgets(res)
    f = sb.copy()
    f["gap"] = [fmt_ms(m, s) for m, s in zip(sb.gap, sb.gap_sd)]
    f = f.drop(columns="gap_sd")
    f["feasibility"] = f.feasibility.map("{:.0f}".format)
    f["evals"] = f.evals.map("{:.0f}".format)
    _write(t, "t6_search_budgets", sb, f, "scale+thin; evals = total EC3 evaluations per design including the operators.")

    mt = table_matched(res)
    f = mt.copy()
    f["gap"] = [fmt_ms(m, s) for m, s in zip(mt.gap, mt.gap_sd)]
    f = f.drop(columns="gap_sd")
    f["feasibility"] = f.feasibility.map("{:.0f}".format)
    f["evals_per_design"] = f.evals_per_design.map("{:.0f}".format)
    f["labelling_evals"] = f.labelling_evals.map(lambda v: "" if np.isnan(v) else f"{int(v):,}")
    _write(t, "t7_methods_matched", mt, f,
           "Main grid, scale+thin. kNN pooled labels cost about 96,000 evaluations per labelled context; "
           "RL training is 1,000,000 steps; search and RL need no labels.")

    kc = table_knn_cheap(res)
    f = _fmt_generic(kc)
    for B in (400, 1000, 4800):
        f[f"B={B}"] = [fmt_ms(m, s) for m, s in zip(kc[f"B{B}_gap"], kc[f"B{B}_sd"])]
        f = f.drop(columns=[f"B{B}_gap", f"B{B}_sd", f"B{B}_feas"])
    _write(t, "t8_knn_cheap_labels", kc, f, "kNN gap (%) with DE labels of budget B, mean ± sd over 3 label seeds.")
    return {"ood": ood, "search": sb, "matched": mt, "knn_cheap": kc}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--out", default=os.path.join(REPO, "paper"))
    a = ap.parse_args()
    build_tables(a.results_dir, a.out)
    print(f"tables -> {os.path.join(a.out, 'tables')}")


if __name__ == "__main__":
    main()
