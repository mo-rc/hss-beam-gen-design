"""
Comment 1 -- trace every checkable manuscript number to exactly one result file.

For each number quoted in research/manuscript_final_comprehensive.pdf that can
be recomputed from the repository, this script recomputes it from the named
source file and reports:

    PASS      recomputed value rounds to the quoted value
    PASS(1u)  differs by one unit in the last quoted digit (rounding of a
              difference of rounded numbers, e.g. 8.53-5.79 = 2.74 vs "2.75")
    FAIL      does not match -- the manuscript number is wrong or untraceable

Run from the repo root:
    PYTHONPATH=$PWD python research/scripts/c1_verify_manuscript_numbers.py

Writes research/results/c1_number_ledger.csv.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd
from scipy import stats

R = "research/results"
D = "research/perplexity_docs/results"
rows = []


def chk(where, what, quoted, computed, dec, source, note=""):
    quoted = float(quoted); computed = float(computed)
    unit = 10.0 ** (-dec)
    if round(computed, dec) == round(quoted, dec):
        st = "PASS"
    elif abs(computed - quoted) <= unit * 1.0001 + 1e-12:
        st = "PASS(1u)"
    else:
        st = "FAIL"
    rows.append(dict(manuscript_location=where, quantity=what, quoted=quoted,
                     recomputed=round(computed, dec + 2), status=st, source=source, note=note))


def chk_bool(where, what, quoted_claim, computed_truth, source, note=""):
    rows.append(dict(manuscript_location=where, quantity=what, quoted=quoted_claim,
                     recomputed=computed_truth, status="PASS" if bool(computed_truth) else "FAIL",
                     source=source, note=note))


pct = lambda x: 100.0 * x
row = lambda df, **kw: df.query(" and ".join(f"{k} == {v!r}" for k, v in kw.items())).iloc[0]

# ---------------------------------------------------------------- GA baseline
e4n = pd.read_csv(f"{R}/e4_nonrl_baselines_summary.csv")
ga_none = row(e4n, arm="ga", repair_mode="none"); ga_st = row(e4n, arm="ga", repair_mode="scale+thin")
src = "results/e4_nonrl_baselines_summary.csv"
chk("Abstract, S4.3, T3, T5, Concl.", "GA mean gap, scale+thin (corrected GT)", 1.47, pct(ga_st.gap_mean), 2, src,
    "config = GA(4,800 evals)+scale+thin; NOT the unrepaired GA")
chk("S4.3, T5", "GA median gap, scale+thin", 0.54, pct(ga_st.gap_median), 2, src)
chk("S4.3, T2 ('noise floor')", "GA mean gap, unrepaired (corrected GT)", 1.72, pct(ga_none.gap_mean), 2, src,
    "config = GA(4,800 evals), no operator")
ga_old = json.load(open(f"{R}/ga_cost_summary.json"))
chk("T2 (pre-correction)", "GA mean gap, unrepaired (original GT)", 1.31, pct(ga_old["gap_mean"]), 2,
    "results/ga_cost_summary.json", "config = original costing + original ground truth")

# ------------------------------------------------- T1: E0, pre-correction, 5 seeds
e0 = pd.read_csv(f"{D}/e0_repair_gated_merged_summary.csv"); e0 = e0[e0.arm != "ga"]
src = "perplexity_docs/results/e0_repair_gated_merged_summary.csv"
pre = {}
for m, q_mean, q_ci, q_med, q_ev in (("none", 25.51, 6.06, 24.0, 1), ("scale", 13.88, 2.65, 13.1, 18),
                                      ("scale+thin", 5.79, 0.82, 5.0, 73)):
    x = e0[e0.repair_mode == m]; v = pct(x.gap_mean.values)
    pre[m] = v.mean()
    ci = stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    chk("T1", f"{m}: mean gap", q_mean, v.mean(), 2, src, "pre-correction costing; 5 seeds")
    chk("T1", f"{m}: '+-' value", q_ci, ci, 2, src, "IS the 95% t-CI half-width over seeds (T5 uses sample SD instead)")
    chk("T1", f"{m}: median gap", q_med, pct(x.gap_median.mean()), 1, src, "mean of per-seed medians")
    chk("T1", f"{m}: EC3 evals/design (repair part)", q_ev, x.ec3_evals_per_design.mean(), 0, src)

# ------------------------------------------ T2/S3.5: corrected costing, 5-seed transfer
cg = pd.read_csv(f"{D}/corrgt_gated_merged_summary.csv"); cg = cg[cg.arm != "ga"]
src = "perplexity_docs/results/corrgt_gated_merged_summary.csv"
post = {m: pct(cg[cg.repair_mode == m].gap_mean.values).mean() for m in ("none", "scale", "scale+thin")}
chk("T2, S3.5, S7.1, S7.4", "best PPO arm unrepaired, corrected costing", 30.15, post["none"], 2, src, "5-seed transfer")
chk("T2, S3.5, S7.1, S7.4", "best PPO arm scale+thin, corrected costing", 8.53, post["scale+thin"], 2, src, "5-seed transfer")
chk("T2", "change unrepaired (pp)", 4.64, post["none"] - pre["none"], 2, src)
chk("T2", "change scale+thin (pp)", 2.75, post["scale+thin"] - pre["scale+thin"], 2, src,
    "unrounded difference is 2.74")
chk("T2", "GA floor change (pp)", 0.41, pct(ga_none.gap_mean) - pct(ga_old["gap_mean"]), 2, "ga rows")
rows.append(dict(manuscript_location="S3.5 (NEW)", quantity="decomposition of 30.15 -> 8.53",
                 quoted=np.nan, recomputed=f"none {post['none']:.2f} -> scale {post['scale']:.2f} -> scale+thin {post['scale+thin']:.2f}",
                 status="INFO", source=src,
                 note=f"scale (projection) accounts for {post['none']-post['scale']:.2f} pp; thinning for a further {post['scale']-post['scale+thin']:.2f} pp"))

# --------------------------------------------- T2: welded counts & slenderness
from research.envs.manufacturability import effective_section_type


def optima(d):
    gt = pd.read_csv(f"research/{d}/ec3_optimal_designs_cost.csv")
    return gt.loc[gt.groupby(["span_m", "load_kNm"])["cost"].idxmin()]


def lam_w(o):
    r = 0.1 * o.tf; eps = np.sqrt(235.0 / o.grade)
    return ((o.h - 2 * o.tf - 2 * r) / o.tw / eps)


for d, tag in (("pretrain_data", "original"), ("pretrain_data_corrected", "corrected")):
    o = optima(d)
    eff = pd.Series([effective_section_type(r.section_type, r.h, r.b, r.tf, r.tw, r.grade) for r in o.itertuples()])
    n_w = int((eff == "welded").sum())
    rows.append(dict(manuscript_location="T2 / S4.3", quantity=f"{tag} GT: stored label / physically-welded count",
                     quoted=("139/142" if tag == "original" else "4/142 (T2) vs 138/142 (S4.3)"),
                     recomputed=f"label rolled={int((o.section_type=='rolled').sum())}/142; welded-under-rule={n_w}/142; rolled-under-rule={142-n_w}/142",
                     status=("PASS" if (tag == "original" and n_w == 139) or (tag == "corrected" and n_w == 4) else "FAIL"),
                     source=f"research/{d}", note=("139 = designs that would have to be WELDED but were COSTED as rolled" if tag == "original"
                                                    else "S4.3's '138/142 costed as welded' is inverted: 138 are rolled")))
    chk("T2", f"median web slenderness d/(tw*eps), {tag} GT", 122.6 if tag == "original" else 72.9,
        lam_w(o).median(), 1, f"research/{d}")

# ---------------------------------------------------------------- T4
tr = pd.read_csv(f"{D}/e3_transfer_seed43_summary.csv"); rt = pd.read_csv(f"{R}/e3_corrcost_retrain_summary.csv")
rt = rt[rt.arm.str.startswith("corrcost")]
chk("T4", "transfer unrepaired", 23.20, pct(row(tr, repair_mode="none").gap_mean), 2, "e3_transfer_seed43_summary.csv")
chk("T4", "retrained unrepaired", 22.65, pct(row(rt, repair_mode="none").gap_mean), 2, "results/e3_corrcost_retrain_summary.csv")
chk("T4", "transfer scale+thin", 7.53, pct(row(tr, repair_mode="scale+thin").gap_mean), 2, "e3_transfer_seed43_summary.csv")
chk("T4", "retrained scale+thin", 6.63, pct(row(rt, repair_mode="scale+thin").gap_mean), 2, "results/e3_corrcost_retrain_summary.csv")
chk("T4", "transfer mean utilisation (unrepaired)", 0.838, row(tr, repair_mode="none").util_mean, 3, "e3_transfer_seed43_summary.csv")
chk("T4", "retrained mean utilisation (unrepaired)", 0.841, row(rt, repair_mode="none").util_mean, 3, "results/e3_corrcost_retrain_summary.csv")

# ---------------------------------------------------------------- T5
src = "results/e4_*_summary.csv"
ppo = [row(pd.read_csv(f"{R}/e4_ppo_seed{s}_summary.csv"), repair_mode="scale+thin") for s in (42, 43, 44)]
pg = np.array([pct(p.gap_mean) for p in ppo])
chk("T5", "PPO 3-seed mean gap", 10.94, pg.mean(), 2, src)
chk("T5", "PPO 3-seed '+-' (sample SD)", 1.28, pg.std(ddof=1), 2, src, "T5 '+-' is a sample SD; T1 '+-' is a 95% CI")
med_of = np.mean([pct(p.gap_median) for p in ppo])
pooled = pd.concat([pd.read_csv(f"{R}/e4_ppo_seed{s}_per_context.csv").query("repair_mode=='scale+thin'") for s in (42, 43, 44)])
chk("T5", "PPO median gap", 6.41, med_of, 2, src,
    f"UNTRACEABLE: mean of per-seed medians = {med_of:.2f}; pooled-426 median = {pct(pooled.gap.median()):.2f}")
for nm, f, qm, qd in (("DDPG seed 42", "e4_ddpg_seed42_summary.csv", 12.65, 6.64), ("SAC seed 42", "e4_sac_seed42_summary.csv", 22.24, 20.89),
                      ("TD3 seed 42", "e4_td3_seed42_summary.csv", 48.07, 40.84)):
    x = row(pd.read_csv(f"{R}/{f}"), repair_mode="scale+thin")
    chk("T5", f"{nm} mean gap", qm, pct(x.gap_mean), 2, f); chk("T5", f"{nm} median gap", qd, pct(x.gap_median), 2, f)
rs = row(e4n, arm="random_search_4800", repair_mode="scale+thin"); rs0 = row(e4n, arm="random_search_4800", repair_mode="none")
r40 = row(e4n, arm="random_search_40", repair_mode="scale+thin")
chk("T3, T5", "random search 4,800 + scale+thin, mean", 11.60, pct(rs.gap_mean), 2, src)
chk("T5", "random search 4,800 + scale+thin, median", 7.01, pct(rs.gap_median), 2, src)
chk("T3", "random search 40 + scale+thin, mean", 33.22, pct(r40.gap_mean), 2, src)
chk("S5.3 key finding", "'beats random search (23.77%) by 15.2 pp'", 15.2, pct(rs0.gap_mean) - post["scale+thin"], 1,
    src, f"MIXED CONFIG: 23.77 is UNREPAIRED random search vs PPO+scale+thin; like-for-like (both scale+thin) is {pct(rs.gap_mean)-post['scale+thin']:.2f} pp")
chk("S5.3", "unrepaired random search 4,800 mean", 23.77, pct(rs0.gap_mean), 2, src)

# ---------------------------------------------------------------- T6
def e5(n): return row(pd.read_csv(f"{R}/{n}_curve_1000000_summary.csv"), repair_mode="scale+thin")
chk("T6", "no-anneal (E4 seed 43)", 12.36, pg[1], 2, "e4_ppo_seed43_summary.csv")
chk("T6", "anneal + linear economy (E5-B)", 6.86, pct(e5("e5_ppo_s43_anneal_linear").gap_mean), 2, "e5_ppo_s43_anneal_linear_curve_1000000_summary.csv")
chk("T6", "anneal + log_relative (E5-A)", 6.34, pct(e5("e5_ppo_s43_anneal_logrel").gap_mean), 2, "e5_ppo_s43_anneal_logrel_curve_1000000_summary.csv")
chk("T6", "delta E5-B vs none (pp)", 5.50, pg[1] - pct(e5("e5_ppo_s43_anneal_linear").gap_mean), 2, "e5 + e4",
    "the 5.50 pp isolates annealing; the headline 6.34 (E5-A) also changes the economy reward, so its delta is 6.01")

# ---------------------------------------------------------------- T8 + S5.7 claims
kn = pd.read_csv(f"{D}/e3_knn_oneshot_corrected_summary.csv"); kn = kn[kn.label_frac == 1.0]
for k, qm, qd, qf in ((1, 2.00, 0.64, 0.99), (3, 2.56, 0.83, 0.99), (5, 2.83, 1.01, 0.99)):
    x = row(kn, k=k); src = "e3_knn_oneshot_corrected_summary.csv"
    chk("T8", f"k={k} mean", qm, pct(x.gap_mean), 2, src); chk("T8", f"k={k} median", qd, pct(x.gap_median), 2, src)
    chk("T8", f"k={k} feasibility", qf, x.feasibility, 2, src)
k1 = row(kn, k=1)
chk_bool("S5.7 key finding", "'k-NN median 0.64% is BELOW GA median 0.54%'", "claim: 0.64 < 0.54", pct(k1.gap_median) < pct(ga_st.gap_median),
         "e3_knn... vs e4_nonrl...", f"FALSE as written. It is below the UNREPAIRED GA median ({pct(ga_none.gap_median):.2f}%) - and the k-NN designs are analytically sized to util=1 (a `scale`-type step), so compare configs explicitly")

pc = pd.read_csv(f"{D}/e3_knn_oneshot_corrected_per_context.csv"); pc = pc[(pc.k == 1) & (pc.label_frac == 1.0)]
corner = (pc.span_m >= 14.0) & (pc.load_kNm >= 100.0)
rows.append(dict(manuscript_location="S5.7", quantity="k-NN interior (136) / corner (6) mean gap", quoted="0.97% / 37.16%",
                 recomputed=f"corner(span>=14 & load>=100) n={int(corner.sum())}: interior {pct(pc[~corner].gap.mean()):.2f}% ; corner {pct(pc[corner].gap.mean()):.2f}%",
                 status="INFO", source="e3_knn_oneshot_corrected_per_context.csv",
                 note="corner definition not stated in repo; reported for the record"))

# ---------------------------------------------------------------- T7 (vs committed files)
for prop, evals, qmean, qmed, f, n in ((40, 960, 14.5, 8.2, "e3_reparam_probe_costing_ab_n40.csv", 40),
                                        (200, 4800, 5.9, 3.1, "e3_reparam_probe_costing_ab_n20.csv", 20),
                                        (1000, 24000, 3.1, 2.4, "e3_reparam_probe_costing_ab_n20.csv", 20)):
    d = pd.read_csv(f"{D}/{f}"); d = d[(d.space == "reparam") & (d.proposals == prop)]
    c = d[d.costing == "corrected"].iloc[0]; o = d[d.costing == "original"].iloc[0]
    chk("T7", f"{prop} proposals: mean", qmean, pct(c.gap_mean), 1, f, f"corrected costing, {n} contexts")
    chk("T7", f"{prop} proposals: median", qmed, pct(c.gap_median), 1, f,
        f"corrected-costing median = {pct(c.gap_median):.1f}; original-costing median = {pct(o.gap_median):.1f} -> quoted value matches NEITHER" if
        round(pct(c.gap_median), 1) != qmed else "corrected costing")

# ---------------------------------------------------------------- T7 regenerated (single run)
rg = pd.read_csv(f"{R}/c1_reparam_budget_table.csv")
for prop in (40, 200, 1000):
    x = rg[(rg.costing == "corrected") & (rg.space == "reparam") & (rg.proposals == prop)].iloc[0]
    rows.append(dict(manuscript_location="T7 (REPLACEMENT)", quantity=f"{prop} proposals: mean / median / p90 / worst / feas",
                     quoted=np.nan,
                     recomputed=f"{pct(x.gap_mean):.2f}% / {pct(x.gap_median):.2f}% / {pct(x.gap_p90):.2f}% / {pct(x.gap_worst):.1f}% / {x.feasibility:.3f}",
                     status="INFO", source="results/c1_reparam_budget_table.csv",
                     note=f"corrected costing, n={int(x.n_contexts)} contexts, {int(x.ec3_per_context)} EC3 evals/context, one run"))

out = pd.DataFrame(rows)
out.to_csv(f"{R}/c1_number_ledger.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
print(out[["manuscript_location", "quantity", "quoted", "recomputed", "status"]].to_string(index=False))
print("\nSUMMARY:", out.status.value_counts().to_dict())
print("\nFAILURES:")
print(out[out.status == "FAIL"][["manuscript_location", "quantity", "quoted", "recomputed", "note"]].to_string(index=False))
