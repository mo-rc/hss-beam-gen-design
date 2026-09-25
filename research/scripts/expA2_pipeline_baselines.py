"""
research/scripts/expA2_pipeline_baselines.py
================================================================
Exp A' -- fair PIPELINE comparison: baselines get the SAME post-hoc operator
as the RL policy, at the same total EC3-evaluation count, plus training-cost
break-even. STANDALONE diagnostic; imports Exp A's meter/spaces/optimisers and
the repo's own repair operator UNCHANGED; writes only to a NEW folder.

WHY
---
Exp A showed that GA/DE/CMA-ES reach ~0% gap only at >=1600-4800 EC3 evals and
that at ~100 evals the best baseline has ~20% gap. The current PPO pipeline
(40 policy steps + `scale+thin`, ~113 evals) reports ~9%. That comparison is
NOT yet fair: PPO's output is post-processed by `algo/repair.py`, the baselines
in Exp A were not. Here every baseline runs

        optimiser(G evals)  ->  same repair() operator  ->  final design

and every EC3 evaluation of BOTH stages is counted by the same meter
(`n_gen` + `n_repair` = `n_total`). The operator is `research.algo.repair.repair`
called exactly as for PPO (mode "scale" or "scale+thin", storey=20, metric =
--metric); its own reported `n_ec3` is asserted equal to the meter's count.

WHAT IS RUN
-----------
  methods  : ga, de, cmaes (Exp A optimisers, Exp A tuned hyper-parameters)
             + `rand` (uniform random search): the NULL baseline that shows
             how much of any result is the operator alone. (ms_nm / ms_slsqp
             available via --methods, off by default: weak in Exp A.)
  spaces   : raw, reparam (same definitions as Exp A)
  G        : generation budgets 10,25,40,100,400,1600 EC3 evals
  modes    : none (generation only) | scale | scale+thin
  seeds    : 5, the SAME per-run seed derivation as Exp A, so `none` rows at
             G in {25,100,400,1600} reproduce Exp A cells bit-for-bit
             (checked with --expA_runs).
  operator input: best FEASIBLE design found by the optimiser (lowest metric);
             if none was feasible, the least-violation design evaluated --
             the same convention as the PPO rollout ("best feasible step of the
             episode, else the last step").

RL SIDE (no training here; re-scoring existing per-context outputs)
  --rl_csv globs of existing per-context result files (columns span_m,
  load_kNm, optimal, achieved, gap, feasible, repair_mode, n_ec3_gen,
  n_ec3_repair, n_ec3). Gap is RECOMPUTED against the same GT as the baselines
  and the GT match is asserted. Default: the five 1M-step e5 anneal_linear
  PPO seeds (42..46).

OUTPUTS (research/results/expA2/, tag-prefixed, never overwrites)
  expA2_<tag>_runs.csv       one row per baseline run x mode
  expA2_<tag>_rl_rows.csv    RL per-context rows in the same schema
  expA2_<tag>_summary.csv/json   mean/median/p90/feasibility/seed-SD + eval counts
  expA2_<tag>_matched.csv    RL vs every baseline at matched generation budget
                             (paired over contexts, bootstrap 95% CI, win rate)
  expA2_<tag>_equivalence.csv/json  evals a baseline needs to match RL's gap,
                             baseline gap at RL's eval count, training break-even
  expA2_<tag>_meta.json      arguments, versions, hyper-parameters, checks
  expA2_<tag>_frontier.png   (if matplotlib is available)

TRAINING-COST ACCOUNTING
  One PPO training run = --rl_train_evals EC3 evaluations (default 1,000,000:
  one per environment step; episode resets add <=3%). Break-even designs =
  train_evals / (baseline_evals_to_match_RL_gap - RL_evals_per_design); reported
  only when the baseline needs more evaluations per design than RL. Wall-clock
  is NOT part of this accounting (network inference is cheaper than EC3 calls
  in eval count but not free).

USAGE
    python research/scripts/expA2_pipeline_baselines.py --tag full --n_jobs 4
    python research/scripts/expA2_pipeline_baselines.py --tag full --n_jobs 4 --resume
    python research/scripts/expA2_pipeline_baselines.py --tag full --aggregate_only
================================================================
"""
import os
import sys
import re
import glob
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

from research.envs.hss_env import HSSBeamEnv
from research.algo.repair import repair
import research.scripts.expA_strong_baselines as A

METHODS_IDX = ["ga", "de", "cmaes", "ms_nm", "ms_slsqp", "rand"]     # first 5 == Exp A indices
assert METHODS_IDX[:5] == A.METHODS_ALL
DEFAULT_METHODS = ["ga", "de", "cmaes", "rand"]
MODES = ["none", "scale", "scale+thin"]
RL_GEN_NOMINAL = 40      # PPO policy horizon (env max_steps); actual n_gen per row is kept in `n_gen`
STOREY = 20                                                        # as in research/scripts/evaluate.py


class PipeMeter(A.Ec3Meter):
    """Exp A meter + tracking of the least-violation infeasible design."""

    def reset(self, span_mm, load, budget):
        super().reset(span_mm, load, budget)
        self.min_viol, self.min_viol_design = None, None

    def evaluate(self, h, b, tf, tw, fy, st):
        res = super().evaluate(h, b, tf, tw, fy, st)
        if not res["feas"] and (self.min_viol is None or res["viol"] < self.min_viol):
            self.min_viol, self.min_viol_design = res["viol"], (float(h), float(b), float(tf), float(tw), float(fy), st)
        return res


def run_rand(space, m, B, rng, cfg):
    while True:
        space.propose(rng.random(A.DIM), m)


RUNNERS = dict(A.RUNNERS)
RUNNERS["rand"] = run_rand

_W = {}


def _init_worker(metric, size_tol, size_max):
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=metric,
                     enforce_rolled_manufacturability=True)
    _W["meter"] = PipeMeter(env, metric)
    _W["spaces"] = dict(raw=A.RawSpace(), reparam=A.ReparamSpace(size_tol, size_max))
    _W["metric"] = metric


def _task(a):
    ctx_i, span_m, load, gt, method, space, gens, n_seeds, base_seed, hp = a
    m, sp, metric = _W["meter"], _W["spaces"][space], _W["metric"]
    mid, sid = METHODS_IDX.index(method), A.SPACES_ALL.index(space)
    rows = []
    for G in gens:
        for s in range(n_seeds):
            m.reset(span_m * 1000.0, load, G)
            rng = np.random.default_rng(np.random.SeedSequence([base_seed, ctx_i, mid, sid, G, s]))   # == Exp A seeding
            t0 = time.perf_counter()
            try:
                RUNNERS[method](sp, m, G, rng, hp)
            except A.BudgetExhausted:
                pass
            n_gen = m.n
            assert n_gen <= G and n_gen == m.n_direct + m.n_sizing
            base = dict(context_idx=ctx_i, span_m=span_m, load_kNm=load, gt_economy=gt, method=method,
                        space=space, gen_budget=G, seed=s, n_gen=n_gen, n_gen_sizing=m.n_sizing)
            feas0 = m.best_econ is not None
            rows.append(dict(base, mode="none", feasible=int(feas0),
                             best_economy=m.best_econ if feas0 else np.nan,
                             gap=(m.best_econ / gt - 1.0) if feas0 else np.nan,
                             n_repair=0, n_total=n_gen, wall_s=time.perf_counter() - t0))
            dsg = m.best_design if feas0 else m.min_viol_design
            design = dict(zip(["h", "b", "tf", "tw", "fy", "section_type"], dsg))
            m.budget = math.inf                                   # operator stage: uncapped but COUNTED
            for mode in ("scale", "scale+thin"):
                t1 = time.perf_counter()
                n0 = m.n
                out = repair(m.env, span_m * 1000.0, load, design, metric=metric, storey=STOREY, mode=mode)
                n_rep = m.n - n0
                assert n_rep == out["n_ec3"], f"operator eval accounting mismatch {n_rep} vs {out['n_ec3']}"
                ok = bool(out["feasible"])
                rows.append(dict(base, mode=mode, feasible=int(ok),
                                 best_economy=out[metric] if ok else np.nan,
                                 gap=(out[metric] / gt - 1.0) if ok else np.nan,
                                 n_repair=n_rep, n_total=n_gen + n_rep, wall_s=time.perf_counter() - t1))
    return rows


# ----------------------------------------------------------------------------
# RL rows
# ----------------------------------------------------------------------------
def load_rl(globs, gt_opt, metric, label):
    files = sorted({f for g in globs for f in glob.glob(g)})
    if not files:
        sys.exit(f"--rl_csv matched no files: {globs}")
    gt_opt = gt_opt.copy()
    gt_opt["span_m"] = gt_opt["span_m"].round(6)
    gt_opt["load_kNm"] = gt_opt["load_kNm"].round(6)
    gt = gt_opt[["span_m", "load_kNm", metric]].rename(columns={metric: "gt_val"})
    out = []
    ci_map = gt_opt.reset_index().set_index(["span_m", "load_kNm"])["index"]
    for f in files:
        d = pd.read_csv(f)
        d["span_m"] = d["span_m"].round(6)
        d["load_kNm"] = d["load_kNm"].round(6)
        seed = int(re.search(r"_s(\d+)_", os.path.basename(f)).group(1)) if re.search(r"_s(\d+)_", os.path.basename(f)) else len(out)
        m = d.merge(gt, on=["span_m", "load_kNm"], how="inner")
        assert len(m) == len(d) and abs(m.optimal - m.gt_val).max() < 1e-9, f"GT mismatch in {f}"
        feas = m.feasible.astype(bool)
        r = pd.DataFrame(dict(
            context_idx=ci_map.reindex(pd.MultiIndex.from_arrays([m.span_m, m.load_kNm])).values,
            span_m=m.span_m, load_kNm=m.load_kNm, gt_economy=m.gt_val, method=label, space="policy",
            gen_budget=RL_GEN_NOMINAL, seed=seed, mode=m.repair_mode,
            n_gen=m.n_ec3_gen, n_repair=m.n_ec3 - m.n_ec3_gen, n_total=m.n_ec3,
            feasible=feas.astype(int), best_economy=np.where(feas, m.achieved, np.nan)))
        r["gap"] = np.where(feas, m.achieved / m.gt_val - 1.0, np.nan)
        out.append(r[r["mode"].isin(MODES)])
    return pd.concat(out, ignore_index=True), files
# ----------------------------------------------------------------------------
# Summaries / comparisons
# ----------------------------------------------------------------------------
def summarize(runs):
    recs = []
    for (method, space, mode, G), d in runs.groupby(["method", "space", "mode", "gen_budget"]):
        f = d[d.feasible == 1]
        g = f.gap.values
        sm = f.groupby("seed").gap.mean()
        recs.append(dict(method=method, space=space, mode=mode, gen_budget=int(G), n_runs=len(d),
                         n_contexts=d.context_idx.nunique(), n_seeds=d.seed.nunique(),
                         feasibility_rate=float(d.feasible.mean()),
                         gap_mean=float(np.mean(g)) if len(g) else np.nan,
                         gap_median=float(np.median(g)) if len(g) else np.nan,
                         gap_p90=float(np.percentile(g, 90)) if len(g) else np.nan,
                         gap_worst=float(np.max(g)) if len(g) else np.nan,
                         seedmean_sd=float(sm.std(ddof=1)) if len(sm) > 1 else np.nan,
                         frac_gap_le_2pct=float(((d.gap <= 0.02) & (d.feasible == 1)).mean()),
                         frac_gap_le_5pct=float(((d.gap <= 0.05) & (d.feasible == 1)).mean()),
                         mean_n_gen=float(d.n_gen.mean()), mean_n_repair=float(d.n_repair.mean()),
                         mean_n_total=float(d.n_total.mean()), max_n_total=int(d.n_total.max())))
    return pd.DataFrame(recs).sort_values(["mode", "space", "method", "gen_budget"]).reset_index(drop=True)


def _ctx_means(d):
    return d[d.feasible == 1].groupby("context_idx").gap.mean()


def matched(runs, rl, match_G, n_boot=5000, seed=0):
    """RL vs every baseline config at the SAME generation budget, per mode; paired over contexts."""
    rng = np.random.default_rng(seed)
    rows = []
    for mode in MODES:
        r = _ctx_means(rl[rl["mode"] == mode])
        for (method, space), d in runs[(runs["mode"] == mode) & (runs.gen_budget == match_G)].groupby(["method", "space"]):
            b = _ctx_means(d)
            idx = r.index.intersection(b.index)
            diff = (r.loc[idx] - b.loc[idx]).values                      # RL - baseline, negative = RL better
            boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
            rows.append(dict(mode=mode, gen_budget=match_G, baseline=method, space=space, n_contexts=len(idx),
                             baseline_gap_mean=float(b.loc[idx].mean()), rl_gap_mean=float(r.loc[idx].mean()),
                             diff_rl_minus_baseline=float(diff.mean()),
                             ci95_lo=float(np.percentile(boots, 2.5)), ci95_hi=float(np.percentile(boots, 97.5)),
                             rl_win_rate=float((diff < 0).mean()),
                             baseline_mean_total_evals=float(d.n_total.mean()),
                             rl_mean_total_evals=float(rl[rl["mode"] == mode].n_total.mean())))
    return pd.DataFrame(rows)


def _interp_evals_for_gap(tot, gap, target):
    """Evals at which a (monotone-enveloped) gap-vs-evals curve reaches `target` (log-linear)."""
    o = np.argsort(tot)
    tot, gap = np.asarray(tot)[o], np.minimum.accumulate(np.asarray(gap)[o])
    if target < gap[-1]:
        return None                                                  # never reached within tested budgets
    if target >= gap[0]:
        return float(tot[0])                                         # reached at or before smallest budget
    return float(np.exp(np.interp(-target, -gap, np.log(tot))))


def _interp_gap_at_evals(tot, gap, evals):
    o = np.argsort(tot)
    tot, gap = np.asarray(tot)[o], np.minimum.accumulate(np.asarray(gap)[o])
    if evals < tot[0] or evals > tot[-1]:
        return None
    return float(np.interp(math.log(evals), np.log(tot), gap))


def equivalence(summ, rl_summ, train_evals):
    rows = []
    for rr in rl_summ.itertuples():
        for (method, space), d in summ[(summ["mode"] == rr.mode) & (summ.method != rr.method)].groupby(["method", "space"]):
            e_match = _interp_evals_for_gap(d.mean_n_total.values, d.gap_mean.values, rr.gap_mean)
            g_at = _interp_gap_at_evals(d.mean_n_total.values, d.gap_mean.values, rr.mean_n_total)
            saving = None if e_match is None else e_match - rr.mean_n_total
            rows.append(dict(rl=rr.method, mode=rr.mode, rl_gap_mean=rr.gap_mean, rl_evals_per_design=rr.mean_n_total,
                             baseline=method, space=space,
                             baseline_gap_at_rl_evals=g_at,
                             baseline_evals_to_match_rl_gap=e_match,
                             eval_saving_per_design=saving,
                             breakeven_designs=(train_evals / saving) if (saving is not None and saving > 0) else None,
                             note=("baseline never reaches RL gap within tested budgets" if e_match is None else "")))
    return pd.DataFrame(rows)


def plot_frontier(summ, rl_summ, path, mode_pairs=("none", "scale", "scale+thin")):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axs = plt.subplots(1, len(mode_pairs), figsize=(5.2 * len(mode_pairs), 4.2), sharey=True)
    for ax, mode in zip(np.atleast_1d(axs), mode_pairs):
        for (method, space), d in summ[summ["mode"] == mode].groupby(["method", "space"]):
            d = d.sort_values("mean_n_total")
            ax.plot(d.mean_n_total, 100 * d.gap_mean, marker="o", ms=3, lw=1,
                    ls="-" if space == "reparam" else "--", label=f"{method}/{space}")
        for rr in rl_summ[rl_summ["mode"] == mode].itertuples():
            ax.scatter([rr.mean_n_total], [100 * rr.gap_mean], marker="*", s=180, c="k", zorder=5, label=rr.method)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"operator = {mode}")
        ax.set_xlabel("mean EC3 evals / design (generation + operator)"); ax.grid(alpha=.3, which="both")
    np.atleast_1d(axs)[0].set_ylabel("mean gap vs GT (%)")
    np.atleast_1d(axs)[-1].legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Exp A': equal-pipeline baselines + RL re-scoring + break-even")
    ap.add_argument("--metric", default="cost", choices=["mass", "cost", "co2"])
    ap.add_argument("--gt_dir", default="research/pretrain_data_corrected")
    ap.add_argument("--out_dir", default="research/results/expA2")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--spaces", default="raw,reparam")
    ap.add_argument("--gen_budgets", default="10,25,40,100,400,1600")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--n_contexts", type=int, default=None)
    ap.add_argument("--context_seed", type=int, default=0)
    ap.add_argument("--base_seed", type=int, default=0)
    ap.add_argument("--size_tol", type=float, default=0.003)
    ap.add_argument("--size_max", type=int, default=10)
    ap.add_argument("--hp_json", default=None)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--rl_csv", nargs="*",
                    default=[f"research/results/e5_ppo_s{s}_anneal_linear_curve_1000000_per_context.csv" for s in range(42, 47)])
    ap.add_argument("--rl_label", default="ppo_e5_linear_1M")
    ap.add_argument("--rl_train_evals", type=float, default=1.0e6)
    ap.add_argument("--match_G", type=int, default=40, help="generation budget matched to the RL policy's ~40 steps")
    ap.add_argument("--expA_runs", default="research/results/expA/expA_full_runs.csv",
                    help="if present, `none` rows are checked bit-for-bit against Exp A cells")
    a = ap.parse_args()

    methods, spaces = a.methods.split(","), a.spaces.split(",")
    gens = [int(x) for x in a.gen_budgets.split(",")]
    assert all(x in METHODS_IDX for x in methods) and all(x in A.SPACES_ALL for x in spaces)
    os.makedirs(a.out_dir, exist_ok=True)
    p = lambda s: os.path.join(a.out_dir, f"expA2_{a.tag}_{s}")
    runs_csv = p("runs.csv")
    hp_table = dict(A.DEFAULT_HP)
    if a.hp_json:
        hp_table.update(json.load(open(a.hp_json)))
    hp_table.setdefault("rand|raw", {}); hp_table.setdefault("rand|reparam", {})

    opt_all = A.load_contexts(a.gt_dir, a.metric, None, 0)          # all contexts (RL merge needs full index)

    if not a.aggregate_only:
        if os.path.exists(runs_csv) and not a.resume:
            sys.exit(f"{runs_csv} exists. Use --resume or a new --tag. (Refusing to overwrite.)")
        opt = A.load_contexts(a.gt_dir, a.metric, a.n_contexts, a.context_seed)
        # keep the SAME context_idx numbering as Exp A / the full GT list
        key = opt_all.reset_index().set_index(["span_m", "load_kNm"])["index"]
        chk = A.gt_sanity_check(opt, a.metric, a.size_tol, a.size_max)
        print("GT sanity check:", chk)
        if chk["max_rel_dev_gt_reeval"] > 1e-3 or chk["gt_designs_infeasible"] > 0:
            sys.exit("GT re-evaluation mismatch; aborting.")
        done = set()
        if os.path.exists(runs_csv):
            prev = pd.read_csv(runs_csv).drop_duplicates(["context_idx", "method", "space", "mode", "gen_budget", "seed"])
            cnt = prev.groupby(["context_idx", "method", "space"]).size()
            need = len(gens) * a.n_seeds * len(MODES)
            done = {k for k, v in cnt.items() if v >= need}
            keep = prev.set_index(["context_idx", "method", "space"]).index.isin(list(done))
            if not keep.all() or len(prev) != sum(1 for _ in open(runs_csv)) - 1:
                print(f"resume: dropping {int((~keep).sum())} rows of incomplete tasks / duplicates")
                prev[keep].to_csv(runs_csv, index=False)
        tasks = []
        for r in opt.itertuples():
            ci = int(key.loc[(r.span_m, r.load_kNm)])
            for mth in methods:
                for sp in spaces:
                    if (ci, mth, sp) not in done:
                        tasks.append((ci, r.span_m, r.load_kNm, float(getattr(r, a.metric)), mth, sp, gens,
                                      a.n_seeds, a.base_seed, A.hp_for(hp_table, mth, sp)))
        meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), args=vars(a), git_commit=A._git_commit(),
                    python=platform.python_version(), platform=platform.platform(), gt_check=chk,
                    versions={k: __import__(k).__version__ for k in ("numpy", "scipy", "pandas", "cma")},
                    hp_table=hp_table, operator=dict(fn="research.algo.repair.repair", modes=["scale", "scale+thin"],
                                                     storey=STOREY, input="best feasible design else least-violation design"))
        json.dump(meta, open(p("meta.json"), "w"), indent=2, default=str)
        print(f"{len(tasks)} tasks ({len(done)} done) | methods={methods} spaces={spaces} G={gens} seeds={a.n_seeds} jobs={a.n_jobs}")
        init = (a.metric, a.size_tol, a.size_max)
        if a.n_jobs > 1:
            it = Pool(a.n_jobs, initializer=_init_worker, initargs=init).imap_unordered(_task, tasks)
        else:
            _init_worker(*init)
            it = map(_task, tasks)
        t0, header = time.time(), not os.path.exists(runs_csv)
        for k, rows in enumerate(it, 1):
            pd.DataFrame(rows).to_csv(runs_csv, mode="a", header=header, index=False)
            header = False
            if k % max(1, len(tasks) // 20) == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"  {k}/{len(tasks)} | {el / 60:.1f} min | ETA {(el / k) * (len(tasks) - k) / 60:.1f} min", flush=True)

    runs = pd.read_csv(runs_csv).drop_duplicates(["context_idx", "method", "space", "mode", "gen_budget", "seed"])
    rl, rl_files = load_rl(a.rl_csv, opt_all, a.metric, a.rl_label)
    rl.to_csv(p("rl_rows.csv"), index=False)
    rl = rl[rl.context_idx.isin(runs.context_idx.unique())]         # same contexts as the baseline runs
    summ = summarize(runs)
    rl_summ = summarize(rl)
    allsum = pd.concat([summ, rl_summ], ignore_index=True)
    allsum.to_csv(p("summary.csv"), index=False)
    json.dump(json.loads(allsum.to_json(orient="records")), open(p("summary.json"), "w"), indent=2)

    checks = {}
    if os.path.exists(a.expA_runs):                                   # bit-for-bit consistency with Exp A
        ea = pd.read_csv(a.expA_runs)
        m = runs[runs["mode"] == "none"].merge(
            ea, left_on=["context_idx", "method", "space", "gen_budget", "seed"],
            right_on=["context_idx", "method", "space", "budget", "seed"], suffixes=("", "_A"))
        if len(m):
            checks["expA_none_rows_compared"] = int(len(m))
            checks["expA_none_rows_max_abs_gap_diff"] = float(np.nanmax(np.abs(m.gap - m.gap_A))) if m.gap.notna().any() else 0.0
            checks["expA_none_rows_feasibility_equal"] = bool((m.feasible == m.feasible_A).all())
    mt = matched(runs, rl, a.match_G)
    mt.to_csv(p("matched.csv"), index=False)
    eq = equivalence(summ, rl_summ, a.rl_train_evals)
    eq.to_csv(p("equivalence.csv"), index=False)
    json.dump(dict(checks=checks, equivalence=json.loads(eq.to_json(orient="records"))), open(p("equivalence.json"), "w"), indent=2)
    plotted = plot_frontier(summ, rl_summ, p("frontier.png"))

    pd.set_option("display.width", 250)
    pct = lambda s: (100 * s).round(2)
    print("\n=== RL rows (existing per-context outputs, same GT) ===")
    print(rl_summ.assign(gap_mean=pct(rl_summ.gap_mean), gap_median=pct(rl_summ.gap_median), gap_p90=pct(rl_summ.gap_p90))
          [["method", "mode", "feasibility_rate", "gap_mean", "gap_median", "gap_p90", "seedmean_sd", "mean_n_total"]].to_string(index=False))
    print(f"\n=== Matched generation budget G={a.match_G}: RL minus baseline (pp of gap, negative = RL better) ===")
    t = mt.copy()
    for c in ("baseline_gap_mean", "rl_gap_mean", "diff_rl_minus_baseline", "ci95_lo", "ci95_hi"):
        t[c] = pct(t[c])
    print(t[["mode", "baseline", "space", "baseline_gap_mean", "rl_gap_mean", "diff_rl_minus_baseline", "ci95_lo", "ci95_hi",
             "rl_win_rate", "baseline_mean_total_evals", "rl_mean_total_evals"]].round(2).to_string(index=False))
    print("\n=== Evals a baseline needs to match RL's mean gap; training break-even ===")
    e = eq.copy(); e["rl_gap_mean"] = pct(e.rl_gap_mean); e["baseline_gap_at_rl_evals"] = pct(e.baseline_gap_at_rl_evals)
    print(e[["mode", "baseline", "space", "rl_gap_mean", "rl_evals_per_design", "baseline_gap_at_rl_evals",
             "baseline_evals_to_match_rl_gap", "breakeven_designs"]].round(1).to_string(index=False))
    if checks:
        print("\nconsistency with Exp A (`none` rows):", checks)
    print(f"\nRL files: {len(rl_files)}  | plot: {plotted}\nwrote files with prefix {p('')}")


if __name__ == "__main__":
    main()
