"""
Costing-paired RAW-vs-REPARAM probe (zero training) -- Comment 1 fix.

WHY THIS FILE EXISTS
--------------------
The E3 manifest cites `research/scripts/diag_reparam_probe_costing_ab.py`
as the producer of `e3_reparam_probe_costing_ab_n{20,40}.csv`, but that
driver was never committed. Meanwhile `diag_reparam_probe.py` builds its
environment with the class default (`enforce_rolled_manufacturability=True`,
i.e. CORRECTED costing) and defaults `--gt_dir` to the ORIGINAL ground
truth, so running it as-is scores a corrected-cost search against an
original-cost reference (verified: 13.5% mean at 200 proposals, versus
3.6% original/original and 5.9% corrected/corrected).

WHAT THIS DRIVER GUARANTEES
---------------------------
* Each costing is ALWAYS paired with its own ground truth:
      original  -> enforce_rolled_manufacturability=False + research/pretrain_data
      corrected -> enforce_rolled_manufacturability=True  + research/pretrain_data_corrected
  (an assertion checks both ground-truth files cover the same contexts).
* ONE invocation evaluates EVERY budget on the SAME context set, so every
  row of the resulting table comes from one evaluation run.
* Mean, median, p90, worst and feasibility are written together, from the
  same per-context array, in the same file (the legacy CSV stored only the
  mean, which is how medians ended up being transcribed from elsewhere).
* Per-context gaps are saved, so any subset statistic is derivable.

RNG MODES
---------
`--rng per_context` (default): the random stream for context i is
    default_rng([seed, i]) with i = position in (span, load)-sorted order, so a
    context's result does not depend on which other contexts are evaluated.
    Budgets share the stream start, so a larger budget's proposals are a
    superset of a smaller one's (monotone by construction).
`--rng sequential`: bit-for-bit reproduction of the legacy scripts (one
    stream shared across contexts, contexts = opt.sample(n, random_state=seed)).
    Used to prove this driver reproduces the E3 A/B numbers.

Usage (from repo root, PYTHONPATH=$PWD):
    python research/scripts/diag_reparam_probe_costing_ab.py \
        --n_contexts 142 --budgets 10 20 40 200 1000 \
        --raw_budgets 40 400 4000 4800 \
        --out research/results/c1_reparam_budget_table.csv
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.scripts import diag_reparam_probe as P

COSTINGS = {
    "original":  dict(enforce=False, gt_dir="research/pretrain_data"),
    "corrected": dict(enforce=True,  gt_dir="research/pretrain_data_corrected"),
}


def load_reference(gt_dir, metric):
    gt = pd.read_csv(os.path.join(gt_dir, f"ec3_optimal_designs_{metric}.csv"))
    opt = gt.loc[gt.groupby(["span_m", "load_kNm"])[metric].idxmin()]
    return opt.sort_values(["span_m", "load_kNm"]).reset_index(drop=True)


def stats(g):
    g = np.asarray(g, dtype=float)
    f = np.isfinite(g)
    return dict(gap_mean=float(np.nanmean(g)), gap_median=float(np.nanmedian(g)),
                gap_p90=float(np.nanpercentile(g[f], 90)), gap_worst=float(np.nanmax(g[f])),
                feasibility=float(f.mean()))


def run_costing(name, cfg, a):
    # Both ground truths are sorted by (span, load) and asserted (in main) to
    # cover the same contexts, so a position-based sample picks the same
    # contexts under both costings.
    opt = load_reference(cfg["gt_dir"], a.metric)
    if a.rng == "sequential" and a.n_contexts < len(opt):
        opt = opt.sample(a.n_contexts, random_state=a.seed).reset_index(drop=True)
    elif a.rng == "per_context" and a.n_contexts < len(opt):
        opt = opt.sample(a.n_contexts, random_state=a.seed).sort_values(
            ["span_m", "load_kNm"]).reset_index(drop=True)
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=a.metric,
                     enforce_rolled_manufacturability=cfg["enforce"])
    rows, per_ctx = [], []
    plan = [("raw", b) for b in a.raw_budgets] + [("reparam", b) for b in a.budgets]
    for space, bud in plan:
        seq_rng = np.random.default_rng(a.seed)
        gaps, calls = [], None
        for i, r in opt.iterrows():
            rng = seq_rng if a.rng == "sequential" else np.random.default_rng([a.seed, i])
            if space == "raw":
                c = P.search_raw(env, r["span_m"], r["load_kNm"], bud, rng, a.metric)
                calls = bud
            else:
                c, calls = P.search_reparam(env, r["span_m"], r["load_kNm"], bud, rng, a.metric)
            gap = np.nan if not np.isfinite(c) else c / r[a.metric] - 1
            gaps.append(gap)
            per_ctx.append(dict(costing=name, space=space, proposals=bud, span_m=r["span_m"],
                                load_kNm=r["load_kNm"], reference=r[a.metric],
                                achieved=(np.nan if not np.isfinite(c) else c), gap=gap))
        rows.append(dict(costing=name, gt_dir=cfg["gt_dir"], n_contexts=len(opt), space=space,
                         proposals=bud, ec3_per_context=calls, **stats(gaps)))
        s = rows[-1]
        print(f"[{name:9s}] {space:7s} {bud:>5d} props  {calls:>6d} EC3/ctx  "
              f"mean {s['gap_mean']*100:6.2f}%  median {s['gap_median']*100:6.2f}%  "
              f"p90 {s['gap_p90']*100:6.2f}%  worst {s['gap_worst']*100:6.1f}%  feas {s['feasibility']:.3f}",
              flush=True)
    return rows, per_ctx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="cost")
    p.add_argument("--n_contexts", type=int, default=142)
    p.add_argument("--budgets", type=int, nargs="+", default=[10, 20, 40, 200, 1000],
                   help="REPARAM decoder proposals")
    p.add_argument("--raw_budgets", type=int, nargs="+", default=[40, 400, 4000, 4800],
                   help="RAW uniform proposals (= EC3 evals)")
    p.add_argument("--costings", nargs="+", default=["original", "corrected"], choices=list(COSTINGS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rng", choices=["per_context", "sequential"], default="per_context")
    p.add_argument("--out", default="research/results/c1_reparam_budget_table.csv")
    a = p.parse_args()

    # Both ground-truth files must cover exactly the same contexts.
    key_sets = [set(map(tuple, load_reference(COSTINGS[c]["gt_dir"], a.metric)[["span_m", "load_kNm"]]
                       .round(6).values.tolist())) for c in a.costings]
    assert all(k == key_sets[0] for k in key_sets), "ground-truth context sets differ between costings"
    n_avail = len(key_sets[0])
    print(f"contexts available: {n_avail}   evaluating: {min(a.n_contexts, n_avail)}   "
          f"rng={a.rng}   seed={a.seed}\n", flush=True)

    t0 = time.time()
    all_rows, all_ctx = [], []
    for c in a.costings:
        r, pc = run_costing(c, COSTINGS[c], a)
        all_rows += r; all_ctx += pc
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pd.DataFrame(all_rows).to_csv(a.out, index=False)
    pd.DataFrame(all_ctx).to_csv(a.out.replace(".csv", "_per_context.csv"), index=False)
    print(f"\nwrote {a.out} and *_per_context.csv   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
