"""
================================================================
diagnostic_stability_trace.py
----------------------------------------------------------------
HKU HSS Beam RL — Per-Step Stability / Drift Trace

WHY THIS EXISTS:
    diagnostic_grade_comparison.py's trajectory summary showed that
    on several hard scenarios (e.g. 13m/140kN/welded, 13m/100kN/welded
    on exp54), the policy reaches the feasible band [0.90, 1.05] early
    (step 2-4) but the episode runs the full 40 steps without ever
    locking in 3 consecutive successful steps — i.e. it touches the
    target and then drifts away, rather than never getting close.

    That's a genuinely different failure mode from "can't learn
    context-conditioning" (which the earlier diagnostic bugs falsely
    suggested). This script logs EVERY step of a rollout — action
    vector, resulting geometry, utilisation, section class, and an
    approximate success-zone flag — so we can see exactly what's
    happening: does it oscillate in and out of the band repeatedly?
    Overshoot once and never come back? Get nudged out by a single
    large late action?

WHAT'S LOGGED PER STEP (all sourced from `info`, never from live env
attributes — see diagnostic_grade_comparison.py's fix #3 for why that
matters: SB3 VecEnvs auto-reset the env internally the instant a step
returns done=True, silently corrupting any subsequent direct attribute
read):
    step, action (6-dim, raw policy output before env scaling),
    h, b, tf, tw, fy, section_type, utilization, section_class,
    reward, approx_success_zone, approx_consec_success

CAVEAT on approx_success_zone:
    The env's real success_counter also checks a geometry penalty
    term (b>h ratio penalty, folded into feasibility_penalty) that
    isn't exposed as a standalone field in `info`. This script
    approximates success zone using util in [0.90,1.05] AND
    section_class in {1,2,3} only, which covers the two dominant
    conditions but may occasionally disagree with the env's own
    internal success_counter at the margin.

USAGE:
    python diagnostic_stability_trace.py \
        --model ./models/hss_exp54/best_model.zip \
        --env   ./env/high_rise_generative_env_claude_final.py \
        --scenario 13,140,welded \
        --scenario 13,100,welded \
        --csv-out ./pretrain_data/stability_trace.csv

DEPENDENCIES:  pip install stable-baselines3 gymnasium numpy
================================================================
"""

import argparse
import importlib.util
import csv
import os
import sys
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[ERROR] stable-baselines3 is required for this script.")
    sys.exit(1)


SEP  = "=" * 78
SEP2 = "-" * 78

# Default scenarios: the two exp54 cases that touched the target band
# early (steps 3-4) then drifted for the remaining ~36 steps without
# re-locking in. Override with --scenario span,load,section_type
# (repeatable) to trace others.
DEFAULT_SCENARIOS = [
    (13.0, 140.0, "welded"),
    (13.0, 100.0, "welded"),
]


def load_env_class(env_path):
    spec   = importlib.util.spec_from_file_location("env_module", env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HighRiseGenerativeEnv


def make_bare_env(EnvClass, span_m, load_kNm, section_type, ltb_factor=0.25):
    env = EnvClass(use_storey_load_scaling=False, ltb_restraint_factor=ltb_factor)
    env.np_random    = np.random.default_rng(42)
    env.span         = span_m * 1000.0
    env.load         = load_kNm
    env.storey       = 20
    env.section_type = section_type
    for attr in ["current_util", "current_mass", "current_cost", "current_co2",
                 "prev_util", "prev_mass", "moment_ratio"]:
        setattr(env, attr, 0.0)
    env.current_chi_lt  = 1.0
    env.current_class   = 1
    env.current_Mrd     = 1.0
    env.current_area    = 1.0
    env.memory          = []
    env.curr_step       = 0
    env.success_counter = 0
    return env


def trace_scenario(model, EnvClass, span_m, load_kNm, section_type,
                    vecnorm_path, ltb_factor):
    vec_env = DummyVecEnv([lambda: make_bare_env(
        EnvClass, span_m, load_kNm, section_type, ltb_factor)])
    vec_norm = None
    if vecnorm_path and os.path.exists(vecnorm_path):
        vec_norm = VecNormalize.load(vecnorm_path, vec_env)
        vec_norm.training    = False
        vec_norm.norm_reward = False
    active = vec_norm if vec_norm else vec_env

    active.reset()

    raw = active.venv.envs[0] if hasattr(active, "venv") else active.envs[0]
    raw.span = span_m * 1000.0;  raw.load = load_kNm
    raw.section_type = section_type
    raw.curr_step = 0;  raw.success_counter = 0

    # Same reseed fix as diagnostic_grade_comparison.py: start from a
    # neutral, span-appropriate geometry rather than whatever
    # active.reset() randomly drew.
    span_m_now = raw.span / 1000.0
    h_target   = float(np.clip(span_m_now * 42.0, 250.0, 750.0))
    raw.h  = h_target
    raw.b  = float(np.clip(h_target / 3.0, 120.0, 300.0))
    raw.tf = 20.0
    raw.tw = 12.0
    raw.fy = 500.0

    raw_obs = raw._get_obs()[np.newaxis, :]
    obs = vec_norm.normalize_obs(raw_obs) if vec_norm else raw_obs

    max_steps = getattr(raw, "max_steps", 40)
    done = False; step = 0; consec = 0
    trajectory = []

    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, infos = active.step(action)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        ec3 = info.get("ec3", {}) if isinstance(info, dict) else {}

        util = info.get("utilization")
        sec_class = ec3.get("section_class")
        in_band = (util is not None and 0.90 <= util <= 1.05
                   and sec_class is not None and 0 < sec_class < 4)
        consec = consec + 1 if in_band else 0

        act = np.asarray(action).reshape(-1)
        trajectory.append({
            "step": step,
            "a_h": act[0], "a_b": act[1], "a_tf": act[2],
            "a_tw": act[3], "a_grade": act[4], "a_sectype": act[5],
            "action_l2": float(np.linalg.norm(act)),
            "h": info.get("h"), "b": info.get("b"),
            "tf": info.get("tf"), "tw": info.get("tw"),
            "fy": info.get("fy"), "section_type": info.get("section_type"),
            "util": util, "section_class": sec_class,
            "reward": float(reward[0]) if hasattr(reward, "__len__") else float(reward),
            "in_band": in_band, "consec": consec,
        })
        step += 1

    return trajectory


def print_trace(span_m, load_kNm, section_type, trajectory):
    Med = load_kNm * span_m ** 2 / 8.0
    print(SEP)
    print(f"  {span_m:.0f}m / {load_kNm:.0f} kN/m / {section_type}   Med={Med:.0f} kNm")
    print(SEP)
    print(f"  {'step':>4} {'|action|':>8} {'a_h':>6} {'a_b':>6} {'a_tf':>6} "
          f"{'a_tw':>6} {'a_grd':>6} {'h':>6} {'b':>5} {'tf':>5} {'tw':>5} {'fy':>5} "
          f"{'util':>6} {'cls':>4} {'in_band':>7} {'consec':>6}")
    print(f"  {SEP2}")
    for t in trajectory:
        band_str = "YES" if t["in_band"] else "  ."
        util_str = f"{t['util']:.3f}" if t["util"] is not None else "  n/a"
        print(f"  {t['step']:>4} {t['action_l2']:>8.3f} {t['a_h']:>6.2f} "
              f"{t['a_b']:>6.2f} {t['a_tf']:>6.2f} {t['a_tw']:>6.2f} "
              f"{t['a_grade']:>6.2f} "
              f"{t['h']:>6.0f} {t['b']:>5.0f} {t['tf']:>5.1f} {t['tw']:>5.1f} "
              f"S{t['fy']:>4.0f} {util_str:>6} {t['section_class']:>4} "
              f"{band_str:>7} {t['consec']:>6}")

    # ── Summary diagnostics ──────────────────────────────────────────
    n = len(trajectory)
    half = n // 2
    first_half_mag = np.mean([t["action_l2"] for t in trajectory[:half]]) if half else 0.0
    second_half_mag = np.mean([t["action_l2"] for t in trajectory[half:]]) if n - half else 0.0
    band_entries = sum(
        1 for i in range(1, n)
        if trajectory[i]["in_band"] and not trajectory[i-1]["in_band"]
    )
    if trajectory and trajectory[0]["in_band"]:
        band_entries += 1
    max_consec = max((t["consec"] for t in trajectory), default=0)
    ever_in_band = any(t["in_band"] for t in trajectory)

    print(f"\n  SUMMARY")
    print(f"  {SEP2}")
    print(f"  Mean |action| — first half : {first_half_mag:.3f}")
    print(f"  Mean |action| — second half: {second_half_mag:.3f}")
    print(f"  Times entered feasible band: {band_entries}")
    print(f"  Longest consecutive streak in band: {max_consec}  (needs 3 to terminate)")
    if not ever_in_band:
        print(f"  Diagnosis: NEVER reached the feasible band — genuine convergence")
        print(f"             failure, not a stability/oscillation issue.")
    elif max_consec >= 3:
        print(f"  Diagnosis: reached and HELD the band (should have terminated —")
        print(f"             check the geometry-penalty term if it didn't).")
    elif band_entries >= 2:
        print(f"  Diagnosis: OSCILLATING — enters and exits the band repeatedly")
        print(f"             without ever holding 3 consecutive steps. Action")
        print(f"             magnitude late in the episode "
              f"({'similar to' if abs(first_half_mag-second_half_mag)<0.1 else 'different from'}"
              f" early) is the thing to look at above.")
    else:
        print(f"  Diagnosis: touched the band ONCE, briefly, then drifted away")
        print(f"             and never returned — single overcorrection, not")
        print(f"             oscillation. Check the action at the step right")
        print(f"             after it first entered the band.")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnorm", default=None)
    parser.add_argument("--env", default="./env/high_rise_generative_env_claude_final.py")
    parser.add_argument("--ltb-factor", type=float, default=0.25)
    parser.add_argument("--scenario", action="append", default=None,
                         help="span,load,section_type — repeatable. "
                              "Defaults to the two exp54 drift cases.")
    parser.add_argument("--csv-out", default=None,
                         help="Optional path to dump the full combined trajectory to CSV.")
    args = parser.parse_args()

    if not os.path.exists(args.env):
        print(f"[ERROR] Env not found: {args.env}"); sys.exit(1)
    EnvClass = load_env_class(args.env)

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}"); sys.exit(1)
    model = PPO.load(args.model)

    vecnorm_path = args.vecnorm
    if vecnorm_path is None:
        candidate = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        if os.path.exists(candidate):
            vecnorm_path = candidate

    print(f"  Env    : {args.env}")
    print(f"  Model  : {args.model}")
    print(f"  VecNorm: {vecnorm_path if vecnorm_path else 'not found'}\n")

    scenarios = DEFAULT_SCENARIOS
    if args.scenario:
        scenarios = []
        for s in args.scenario:
            span_s, load_s, sec_s = s.split(",")
            scenarios.append((float(span_s), float(load_s), sec_s.strip()))

    all_rows = []
    for span_m, load_kNm, section_type in scenarios:
        trajectory = trace_scenario(
            model, EnvClass, span_m, load_kNm, section_type,
            vecnorm_path, args.ltb_factor)
        print_trace(span_m, load_kNm, section_type, trajectory)
        for t in trajectory:
            row = dict(t)
            row["span_m"] = span_m
            row["load_kNm"] = load_kNm
            row["scenario_section_type"] = section_type
            all_rows.append(row)

    if args.csv_out and all_rows:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Full trajectory CSV written to: {args.csv_out}")

    print(SEP)
    print("  Trace complete.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()