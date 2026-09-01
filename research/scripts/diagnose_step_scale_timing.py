"""
research/scripts/diagnose_step_scale_timing.py
================================================================
Zero-additional-training-cost diagnostic for the "schedule-coupled
action-budget" hypothesis: that chronic under-utilisation (mean util
~0.6-0.85 vs. a true optimum of ~1.0, observed across every reward mode
and hyperparameter configuration tried so far) is caused, at least in
part, by `_update_design`'s `step_scale` anneal being paced against raw
episode-step count. A design that only nears the feasibility boundary
late in its 40-step episode arrives there with a shrunk step_scale
(as low as 0.30x the early-episode step size) and may not have enough
remaining action-budget to close the last part of the gap -- a ceiling
built into the environment's dynamics, not the reward function, which
would explain why five different reward/exploration-schedule fixes
(see the project handoff doc) all landed at a similar util plateau.

THIS SUPERSEDES AN EARLIER, CONFOUNDED VERSION of this diagnostic that
extended episodes by increasing `max_steps` directly. That also re-paced
`step_scale` (which depended on curr_step/max_steps), so steps beyond
the original 40 used LARGER, never-trained-under step sizes -- any
"continued improvement" observed could have been an artifact of bigger
available moves, not genuine evidence the trained policy would keep
converging given more time at its trained pace. This script uses
`research/envs/hss_env.py`'s new `step_scale_horizon` parameter
(decoupled from `max_steps`) to fix that: episodes are extended via
`max_steps`, while `step_scale_horizon` is pinned at the value the
policy actually trained under (40, for every existing checkpoint), so
step_scale freezes at its terminal value (0.30) for any step beyond 40
instead of being recomputed against a larger denominator.

NO TRAINING. NO CHANGE TO REWARD, OBJECTIVE, OR EC3 PHYSICS. Reuses
existing checkpoints only.

WHAT THIS PRODUCES
--------------------
1. A per-step trace CSV (every step of every evaluated context): step
   index, step_scale, utilisation, feasibility, economy value.
2. A per-context summary CSV: best-feasible economy/util reached within
   the native (trained) horizon vs. across the full extended rollout,
   plus the step index at which the design first entered feasibility /
   the training target band.
3. Printed aggregate statistics directly relevant to confirming or
   rejecting the hypothesis (see module docstring in the handoff doc
   for the exact decision rule):
     - Do contexts continue to improve, in aggregate, past step 40 when
       step_scale is correctly frozen (not re-paced)?
     - Is there a negative correlation between "step at which the design
       first neared the boundary" and "final achieved utilisation"?

USAGE
------
    python research/scripts/diagnose_step_scale_timing.py \\
        --model_path research/models/gated_cost_seed42/final_model \\
        --algo ppo --economy_metric cost --reward_mode_for_env feasibility_gated \\
        --ground_truth_dir research/pretrain_data \\
        --native_max_steps 40 --extended_max_steps 100 \\
        --run_name gated_cost_seed42_stepscale_diag \\
        --out_csv research/results/gated_cost_seed42_stepscale_trace.csv \\
        --out_summary_csv research/results/gated_cost_seed42_stepscale_summary.csv
================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.scripts.evaluate import (
    load_ground_truth,
    ground_truth_path_for_metric,
    ground_truth_optimum,
    load_policy,
)


def run_traced_episode(env: HSSBeamEnv, policy_fn, span_m: float, load_kNm: float,
                        storey: int, native_max_steps: int, extended_max_steps: int, seed: int):
    """
    Single deterministic rollout at a forced (span, load) context, run to
    `extended_max_steps` (env.max_steps), with the design-update pace
    frozen at `native_max_steps` (env.step_scale_horizon) -- i.e. steps
    native_max_steps+1..extended_max_steps move the design at EXACTLY the
    step size (0.30x) the policy last experienced at the end of its
    trained 40-step episodes, not a larger, never-trained-under size.

    Returns a list of per-step dict records (one per step actually taken;
    the env's own `in_target_band`-based early termination can still end
    the episode before extended_max_steps, exactly as it would during
    training/normal evaluation -- this diagnostic does not disable that).
    """
    obs, _ = env.reset(seed=seed)
    env.use_storey_load_scaling = False
    env.span = float(span_m) * 1000.0
    env.load = float(load_kNm)
    env.storey = int(storey)
    obs = env._get_obs()

    records = []
    for t in range(extended_max_steps):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        records.append(dict(
            step=info["curr_step"],
            step_scale=info["step_scale"],
            utilization=info["utilization"],
            economy=info[env.economy_metric],
            feasible=info["feasible"],
            in_target_band=info["in_target_band"],
            beyond_native_horizon=info["curr_step"] > native_max_steps,
        ))
        if terminated or truncated:
            break
    return records


def summarize_context(records: list, native_max_steps: int, economy_metric_name: str, gt_optimum: float) -> dict:
    native_recs = [r for r in records if r["step"] <= native_max_steps]
    all_recs = records  # native + any extension steps actually taken

    def best_feasible(recs):
        feas = [r for r in recs if r["feasible"]]
        if not feas:
            return None, None
        best = min(feas, key=lambda r: r["economy"])
        return best["economy"], best["utilization"]

    native_best_econ, native_best_util = best_feasible(native_recs)
    ext_best_econ, ext_best_util = best_feasible(all_recs)

    first_feasible_step = next((r["step"] for r in native_recs if r["feasible"]), None)
    first_band_step = next((r["step"] for r in native_recs if r["in_target_band"]), None)

    def gap(econ):
        return None if econ is None else (econ - gt_optimum) / gt_optimum

    return dict(
        n_native_steps=len(native_recs),
        n_total_steps=len(all_recs),
        episode_extended_past_native=len(all_recs) > len(native_recs),
        first_feasible_step=first_feasible_step,
        first_in_target_band_step=first_band_step,
        native_best_util=native_best_util,
        native_best_economy=native_best_econ,
        native_gap=gap(native_best_econ),
        extended_best_util=ext_best_util,
        extended_best_economy=ext_best_econ,
        extended_gap=gap(ext_best_econ),
        # positive => extension (frozen-pace) found a strictly better feasible
        # design than was ever seen within the native horizon.
        improved_by_extension=(
            None if (native_best_econ is None or ext_best_econ is None)
            else bool(ext_best_econ < native_best_econ - 1e-9)
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--algo", choices=["ppo", "ddpg", "td3"], default="ppo")
    p.add_argument("--economy_metric", choices=["mass", "cost", "co2"], default="cost")
    p.add_argument("--reward_mode_for_env", type=str, default="feasibility_gated",
                    help="Only affects the (unused, for this diagnostic) reward computation "
                         "inside env.step(); does not affect util/economy/feasibility, which "
                         "are reward-mode-independent.")
    p.add_argument("--ground_truth_dir", type=str, default="research/pretrain_data")
    p.add_argument("--native_max_steps", type=int, default=40,
                    help="The max_steps/step_scale_horizon value the checkpoint was TRAINED "
                         "with. 40 for every existing checkpoint in this project.")
    p.add_argument("--extended_max_steps", type=int, default=100,
                    help="How far to let the episode run (termination budget only).")
    p.add_argument("--n_contexts", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--out_csv", type=str, required=True, help="Per-step trace CSV.")
    p.add_argument("--out_summary_csv", type=str, required=True, help="Per-context summary CSV.")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary_csv), exist_ok=True)

    df = load_ground_truth(ground_truth_path_for_metric(args.economy_metric, args.ground_truth_dir))
    opt = ground_truth_optimum(df, args.economy_metric)
    if args.n_contexts is not None and args.n_contexts < len(opt):
        opt = opt.sample(n=args.n_contexts, random_state=args.seed).reset_index(drop=True)

    policy_fn = load_policy(args.model_path, args.algo)
    env = HSSBeamEnv(
        reward_mode=args.reward_mode_for_env, economy_metric=args.economy_metric,
        max_steps=args.extended_max_steps, step_scale_horizon=args.native_max_steps,
    )

    trace_rows, summary_rows = [], []
    for i, r in opt.iterrows():
        records = run_traced_episode(
            env, policy_fn, r["span_m"], r["load_kNm"], storey=20,
            native_max_steps=args.native_max_steps, extended_max_steps=args.extended_max_steps,
            seed=args.seed + i,
        )
        for rec in records:
            trace_rows.append(dict(span_m=r["span_m"], load_kNm=r["load_kNm"], context_idx=i, **rec))

        gt_optimal = r[args.economy_metric]
        csum = summarize_context(records, args.native_max_steps, args.economy_metric, gt_optimal)
        summary_rows.append(dict(span_m=r["span_m"], load_kNm=r["load_kNm"], context_idx=i,
                                  optimal_economy=gt_optimal, **csum))

    trace_df = pd.DataFrame(trace_rows)
    summary_df = pd.DataFrame(summary_rows)
    trace_df.to_csv(args.out_csv, index=False)
    summary_df.to_csv(args.out_summary_csv, index=False)

    # ---- Aggregate diagnostic statistics -------------------------------
    n = len(summary_df)
    n_extended = summary_df["episode_extended_past_native"].sum()
    n_improved = summary_df["improved_by_extension"].fillna(False).sum()
    native_util_mean = summary_df["native_best_util"].dropna().mean()
    extended_util_mean = summary_df["extended_best_util"].dropna().mean()
    native_gap_mean = summary_df["native_gap"].dropna().mean()
    extended_gap_mean = summary_df["extended_gap"].dropna().mean()

    # Timing-coupling correlation: step of first target-band entry (within
    # the native horizon) vs. the native-horizon best achieved utilisation.
    timing_df = summary_df.dropna(subset=["first_in_target_band_step", "native_best_util"])
    corr = (timing_df["first_in_target_band_step"].corr(timing_df["native_best_util"])
            if len(timing_df) >= 3 else float("nan"))

    print(f"\n=== {args.run_name}: step-scale/horizon timing diagnostic ===")
    print(f"  n_contexts                              : {n}")
    print(f"  n_contexts whose episode ran past step {args.native_max_steps:<3d} : {n_extended} ({n_extended/n:.1%})")
    print(f"  n_contexts IMPROVED by frozen-pace extension : {n_improved} ({n_improved/n:.1%})")
    print(f"  mean best-feasible util, native (<= {args.native_max_steps} steps) : {native_util_mean:.4f}")
    print(f"  mean best-feasible util, extended (<= {args.extended_max_steps} steps): {extended_util_mean:.4f}")
    print(f"  mean gap %, native   : {native_gap_mean*100:.2f}%")
    print(f"  mean gap %, extended : {extended_gap_mean*100:.2f}%")
    print(f"  corr(first_in_target_band_step, native_best_util), n={len(timing_df)}: {corr:.4f}")
    print()
    print("  Decision rule (see handoff doc):")
    print("    CONFIRMS hypothesis  -> meaningful fraction improve under frozen-pace extension")
    print("                            AND/OR negative correlation above (late arrival -> worse util).")
    print("    REJECTS hypothesis   -> extension barely changes util/gap, no timing correlation;")
    print("                            look at exploration-noise/risk-averse-retreat mechanism next.")
    print(f"\nPer-step trace    -> {args.out_csv}")
    print(f"Per-context summary -> {args.out_summary_csv}")


if __name__ == "__main__":
    main()
