"""Step 3 - evaluate one trained checkpoint against the pooled ground truth.

Mirrors the old evaluate.py + evaluate_with_repair.py: for each (span, load) context in the
ground-truth grid, roll out ONE deterministic policy episode (up to 40 refinement steps),
keep the best FEASIBLE step (or the terminal step if none was feasible -- an honestly
infeasible result, not silently dropped), then apply every post-hoc operator to that SAME
rolled-out design so "none"/"scale"/"scale+thin" differ only in the operator, nothing else.

Grade and section type are NOT forced: action[4]/action[5] let the policy choose them every
step (see hssbeamgen/envs/hss_env.py:_update_design), so a forced-context episode still lets
the policy pick whichever grade/type it thinks is best for that context, exactly as trained.

The reference optimum per context is the pooled ground truth (see pipeline/01c): the best
design found by ANY of the three per-objective GA searches, re-verified in the environment.
Use --ground_truth_dir .../main_grid_pooled (or an OOD *_pooled dir), never the raw dir.

Every environment-shaping parameter (ltb_factor, sls_factor, enforce_rolled_manufacturability,
max_steps, ...) is read from the run's own meta.json config -- the SAME env_kwargs() used at
training time -- not from HSSBeamEnv's constructor defaults, so a checkpoint from a physics
ablation is scored under its own physics rather than whatever the current frozen config says.
meta.json is therefore required next to --model, and a run whose status isn't "complete" is
refused unless --allow_incomplete is passed. --economy_metric is the one deliberate override:
it lets a checkpoint be scored on a metric it did NOT train on (its incidental gap); reward_mode
is hardcoded because it (and economy_reward_mode) only affect the reward signal -- see
hss_env.py's _compute_reward/_economy_reward -- which this deterministic rollout never reads.
VecNormalize is not needed at inference either: training wraps only the reward
(norm_obs=False), so the bare saved policy already takes raw, unnormalised observations.

Usage:
    python pipeline/03_evaluate_agent.py --model runs/2a_reward_mode/feasibility_gated/seed42/final_model \
        --ground_truth_dir data/ground_truth/main_grid_pooled --out results/2a_feasibility_gated_seed42_eval.csv

Smoke test (~10 s, uses a handful of contexts):
    python pipeline/03_evaluate_agent.py --model /tmp/train_smoke/final_model \
        --ground_truth_dir data/ground_truth/main_grid_pooled --n_contexts 5 \
        --out /tmp/eval_smoke.csv
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

OPERATOR_MODES = ("none", "scale", "scale+thin")
DESIGN_KEYS = ("h", "b", "tf", "tw", "fy", "section_type")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_policy(model_path: str, algo: str):
    from stable_baselines3 import PPO, SAC, TD3, DDPG
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3, "ddpg": DDPG}[algo]
    model = cls.load(model_path, device="cpu")

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy_fn


def ground_truth_optimum(gt_dir: str, metric: str) -> pd.DataFrame:
    """One row per (span_m, load_kN_per_m): the best (grade, section_type, geometry) among
    every combination in the pooled CSV, i.e. the reference the agent is compared against."""
    df = pd.read_csv(os.path.join(gt_dir, f"ec3_optimal_designs_{metric}.csv"))
    idx = df.groupby(["span_m", "load_kN_per_m"])[metric].idxmin()
    return df.loc[idx].reset_index(drop=True)


def rollout_design(env, policy_fn, span_m: float, load: float, storey: int, seed: int):
    """One forced-context episode, run for env.max_steps (the SAME horizon the checkpoint was
    trained under). Returns (design_dict, was_feasible, n_ec3_analyses)."""
    env.reset(seed=seed)
    env.use_storey_load_scaling = False
    env.span, env.load, env.storey = float(span_m) * 1000.0, float(load), int(storey)
    obs = env._get_obs()

    best, best_econ, last = None, np.inf, None
    n_ec3 = 0
    for _ in range(env.max_steps):
        action = policy_fn(obs)
        obs, _r, terminated, truncated, info = env.step(action)
        n_ec3 += 1
        last = info
        if info["feasible"] and info[env.economy_metric] < best_econ:
            best_econ, best = info[env.economy_metric], info
        if terminated or truncated:
            break
    src = best if best is not None else last
    return {k: src[k] for k in DESIGN_KEYS}, (best is not None), n_ec3


def evaluate(model_path: str, run_meta: dict, ground_truth_dir: str, economy_metric: str,
            n_contexts: int | None, seed: int, storey: int, max_steps: int | None):
    from hssbeamgen.algo.posthoc_operators import apply_operator
    from hssbeamgen.envs.hss_env import HSSBeamEnv
    from hssbeamgen.train_utils import env_kwargs

    algo = run_meta["args"]["algo"]
    policy_fn = load_policy(model_path, algo)
    opt = ground_truth_optimum(ground_truth_dir, economy_metric)
    if n_contexts is not None and n_contexts < len(opt):
        opt = opt.sample(n=n_contexts, random_state=seed).reset_index(drop=True)

    # Build the SAME environment the checkpoint was trained under: every physics/costing
    # parameter (ltb_factor, sls_factor, enforce_rolled_manufacturability, ...) comes from the
    # run's own saved config, via the identical env_kwargs() used at training time -- not from
    # HSSBeamEnv's constructor defaults, which happen to match rl_final.yaml today but would
    # silently diverge the moment anyone runs a physics ablation (the env's own docstring
    # anticipates this: these fields are "exposed for ablation studies"). economy_metric is
    # the one deliberate override: --economy_metric lets you score a checkpoint on a metric it
    # was NOT trained on (e.g. a cost-trained model's incidental CO2 gap), same as evaluate.py's
    # secondary-metric gaps did; reward_mode/economy_reward_mode only affect the reward signal
    # (see hss_env.py _compute_reward/_economy_reward), which this deterministic rollout never
    # reads, so hardcoding a reward_mode below is harmless -- max_steps must still match training.
    kwargs = env_kwargs(run_meta["config"], reward_mode="feasibility_gated")
    kwargs["economy_metric"] = economy_metric
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    env = HSSBeamEnv(**kwargs)
    rows = {m: [] for m in OPERATOR_MODES}
    t0 = time.time()
    for i, r in opt.iterrows():
        design, was_feasible, n_gen = rollout_design(
            env, policy_fn, r["span_m"], r["load_kN_per_m"], storey, seed + i)
        for m in OPERATOR_MODES:
            res = apply_operator(env, r["span_m"] * 1000.0, r["load_kN_per_m"], design,
                                 metric=economy_metric, storey=storey, mode=m)
            gap = (res[economy_metric] - r[economy_metric]) / r[economy_metric] if res["feasible"] else np.nan
            rows[m].append(dict(
                span_m=r["span_m"], load_kN_per_m=r["load_kN_per_m"],
                optimal=r[economy_metric], optimal_grade=r["grade"], optimal_type=r["section_type"],
                achieved=res[economy_metric], gap=gap, feasible=res["feasible"],
                rollout_reached_feasible=was_feasible,
                grade=res["fy"], section_type=res["section_type"],
                grade_match=float(res["fy"] == r["grade"]), utilization=res["utilization"],
                adjusted=res["adjusted"], n_ec3=n_gen + res["n_ec3"],
            ))
    wall_time = time.time() - t0
    return {m: pd.DataFrame(rows[m]) for m in OPERATOR_MODES}, wall_time, len(opt)


def summarise(df: pd.DataFrame, mode: str) -> dict:
    g = df["gap"].to_numpy(dtype=float)
    finite = np.isfinite(g)
    gv = g[finite]
    ratio = 1.0 + gv  # "cost ratio to optimum", matching results_validation_draft's table
    return dict(
        operator_mode=mode, n=int(len(df)), feasibility=float(df["feasible"].mean()),
        cost_ratio_mean=float(np.mean(ratio)) if len(gv) else float("nan"),
        cost_ratio_median=float(np.median(ratio)) if len(gv) else float("nan"),
        within_110pct=float(np.mean(ratio <= 1.10)) if len(gv) else float("nan"),
        within_125pct=float(np.mean(ratio <= 1.25)) if len(gv) else float("nan"),
        grade_match=float(df["grade_match"].mean()),
        ec3_evals_per_design=float(df["n_ec3"].mean()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="path to a saved SB3 model, WITHOUT .zip")
    p.add_argument("--ground_truth_dir", required=True,
                   help="a POOLED ground-truth dir, e.g. data/ground_truth/main_grid_pooled")
    p.add_argument("--economy_metric", default="cost", choices=("mass", "cost", "co2"),
                   help="metric to SCORE against; need not match what the checkpoint trained "
                       "on (e.g. a cost-trained model's incidental CO2 gap)")
    p.add_argument("--n_contexts", type=int, default=None, help="subsample for a quick check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--storey", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=None,
                   help="override the episode horizon; defaults to what the run actually "
                       "trained with (meta.json config), NOT the environment's constructor default")
    p.add_argument("--allow_incomplete", action="store_true",
                   help="evaluate a run whose meta.json status isn't 'complete' (e.g. an "
                       "interrupted run's latest checkpoint) instead of refusing")
    p.add_argument("--out", required=True, help="summary CSV path (one row per operator mode)")
    p.add_argument("--out_detail", default=None,
                   help="optional: dir to also dump per-context, per-mode detail CSVs")
    args = p.parse_args()

    # meta.json is REQUIRED, not just a convenience for --algo: env_kwargs() below rebuilds the
    # environment from the run's own saved config (ltb_factor, sls_factor,
    # enforce_rolled_manufacturability, max_steps, ...), not from HSSBeamEnv's constructor
    # defaults, so a physics ablation run is scored under its own physics, not silently under
    # whatever rl_final.yaml happens to say today.
    run_meta_path = os.path.join(os.path.dirname(args.model), "meta.json")
    if not os.path.exists(run_meta_path):
        raise SystemExit(f"no meta.json next to {args.model}; this script reads the run's own "
                         f"training config from it (physics parameters, algo, max_steps) and "
                         f"cannot evaluate a checkpoint without it.")
    run_meta = json.load(open(run_meta_path))
    if run_meta.get("status") != "complete" and not args.allow_incomplete:
        raise SystemExit(f"{run_meta_path} has status={run_meta.get('status')!r}, not "
                         f"'complete' -- this looks like an interrupted or in-progress run. "
                         f"Pass --allow_incomplete to evaluate its latest checkpoint anyway.")

    print(f"algo={run_meta['args']['algo']} model={args.model} "
         f"ground_truth={args.ground_truth_dir} metric={args.economy_metric}")
    per_mode, wall_time, n_contexts_used = evaluate(
        args.model, run_meta, args.ground_truth_dir, args.economy_metric,
        args.n_contexts, args.seed, args.storey, args.max_steps)

    summary = pd.DataFrame([summarise(per_mode[m], m) for m in OPERATOR_MODES])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))
    print(f"{n_contexts_used} contexts, {wall_time:.1f}s -> {args.out}")

    if args.out_detail:
        os.makedirs(args.out_detail, exist_ok=True)
        for m in OPERATOR_MODES:
            per_mode[m].to_csv(os.path.join(args.out_detail, f"detail_{m}.csv"), index=False)

    env_file = "hssbeamgen/envs/hss_env.py"
    meta = dict(
        script="pipeline/03_evaluate_agent.py", args=vars(args), n_contexts_used=n_contexts_used,
        wall_time_s=round(wall_time, 1), generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git_commit=_git("rev-parse", "HEAD"), git_dirty=bool(_git("status", "--porcelain", "--untracked-files=no")),
        sha256=dict(hss_env=_sha256(env_file)) if os.path.exists(env_file) else {},
        python=platform.python_version(), trained_run_meta=run_meta,
    )
    with open(args.out.rsplit(".", 1)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
