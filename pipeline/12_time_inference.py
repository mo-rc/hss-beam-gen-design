"""Step 12 - wall-clock time per design, policy vs per-context search, on ONE machine, single process.

The evaluation-count comparison (Fig. 1) says nothing about seconds: a policy step costs a neural
forward pass plus one EC3 analysis, a search evaluation costs one EC3 analysis. This script times
both on the same machine, with the same environment and the same operator stage, so the timing
claim in the paper is measured rather than assumed.

For a random subset of ground-truth contexts it times, per design, after a warm-up:
  - PPO (or any checkpoint): the 40-step deterministic rollout ("none"), and rollout + the
    `scale+thin` operator;
  - GA / DE / random search at each budget: the search ("none"), and search + `scale+thin`.
Everything runs in one process (torch is limited to one thread unless --threads is given), so the
numbers are per-core latencies, not throughput. Results depend on the machine; report the CPU.

Usage:
    python pipeline/12_time_inference.py --model runs/2a_reward_mode/feasibility_gated/seed42/final_model \\
        --ground_truth_dir data/ground_truth/main_grid_pooled --economy_metric cost \\
        --n_contexts 20 --repeats 3 --out results/timing_cost.csv
Writes --out (one row per method / budget) and <stem>_meta.json (git commit, machine, versions).
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
DEFAULT_BUDGETS = (40, 112, 400, 1000)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "pipeline", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _stats(ms_values):
    v = np.asarray(ms_values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else float("nan")


def time_designs(designer, operator_fn, contexts, repeats):
    """Time `designer(ctx) -> (design, n_evals)` and `operator_fn(ctx, design) -> n_extra_evals`.

    Returns per-design lists (milliseconds) for the design stage alone and for design + operator, and the mean
    number of EC3 evaluations. A warm-up call on the first context is not timed."""
    designer(contexts[0])
    t_design, t_total, evals_design, evals_total = [], [], [], []
    for _ in range(repeats):
        for ctx in contexts:
            t0 = time.perf_counter()
            design, n = designer(ctx)
            t1 = time.perf_counter()
            extra = operator_fn(ctx, design)
            t2 = time.perf_counter()
            t_design.append((t1 - t0) * 1e3)
            t_total.append((t2 - t0) * 1e3)
            evals_design.append(n)
            evals_total.append(n + extra)
    return t_design, t_total, float(np.mean(evals_design)), float(np.mean(evals_total))


def _row(method, budget, td, tt, e_d, e_t):
    m_d, s_d = _stats(td)
    m_t, s_t = _stats(tt)
    return dict(method=method, budget=budget, ms_design_mean=m_d, ms_design_sd=s_d, ms_total_mean=m_t, ms_total_sd=s_t,
                evals_design=e_d, evals_total=e_t, ms_per_eval_design=m_d / e_d, ms_per_eval_total=m_t / e_t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a saved SB3 model, WITHOUT .zip (meta.json next to it)")
    ap.add_argument("--ground_truth_dir", required=True, help="a POOLED ground-truth dir")
    ap.add_argument("--economy_metric", default="cost", choices=("mass", "cost", "co2"))
    ap.add_argument("--n_contexts", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--methods", nargs="+", default=["de", "ga", "random"], choices=("de", "ga", "random"))
    ap.add_argument("--storey", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1, help="torch threads (default 1: per-core latency)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.ground_truth_dir.rstrip("/").endswith("_pooled"):
        sys.exit(f"{a.ground_truth_dir!r}: use a _pooled ground-truth directory")

    import torch
    torch.set_num_threads(a.threads)
    from hssbeamgen.algo.posthoc_operators import apply_operator
    from hssbeamgen.envs.hss_env import HSSBeamEnv
    from hssbeamgen.train_utils import env_kwargs

    ev = _load("evaluate_agent", "03_evaluate_agent.py")
    sb = _load("baseline_search", "04_baseline_search.py")
    run_meta = json.load(open(os.path.join(os.path.dirname(a.model), "meta.json")))
    algo = run_meta["args"]["algo"]
    policy_fn = ev.load_policy(a.model, algo)
    kwargs = env_kwargs(run_meta["config"], reward_mode="feasibility_gated")
    kwargs["economy_metric"] = a.economy_metric
    env = HSSBeamEnv(**kwargs)

    opt = ev.ground_truth_optimum(a.ground_truth_dir, a.economy_metric)
    opt = opt.sample(n=min(a.n_contexts, len(opt)), random_state=a.seed).reset_index(drop=True)
    contexts = [dict(i=i, span_m=r.span_m, load=r.load_kN_per_m) for i, r in opt.iterrows()]

    def op(ctx, design):
        r = apply_operator(env, ctx["span_m"] * 1000.0, ctx["load"], design, metric=a.economy_metric,
                           storey=a.storey, mode="scale+thin")
        return r["n_ec3"]

    def policy_designer(ctx):
        design, _feas, n = ev.rollout_design(env, policy_fn, ctx["span_m"], ctx["load"], a.storey, a.seed + ctx["i"])
        return design, n

    rows = [_row(f"policy ({algo})", np.nan, *time_designs(policy_designer, op, contexts, a.repeats))]
    for method in a.methods:
        for B in a.budgets:
            def search_designer(ctx, method=method, B=B):
                res = sb.run_search(method, ctx["span_m"] * 1000.0, ctx["load"], a.storey, a.economy_metric, B, ctx["i"])
                return {k: res[k] for k in ("h", "b", "tf", "tw", "fy", "section_type")}, res["n_evaluations"]
            rows.append(_row(method, B, *time_designs(search_designer, op, contexts, a.repeats)))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.round(4).to_csv(a.out, index=False)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, text=True).strip())
    except Exception:
        commit, dirty = "unknown", None
    meta = dict(script="pipeline/12_time_inference.py", args=vars(a), n_contexts_used=len(contexts), git_commit=commit,
                git_dirty=dirty, generated_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                machine=platform.platform(), processor=platform.processor(), torch=torch.__version__, torch_threads=a.threads,
                cpu_count=os.cpu_count())
    json.dump(meta, open(os.path.splitext(a.out)[0] + "_meta.json", "w"), indent=1)
    print(df.round(2).to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
