"""
================================================================
inspect_obs_normalization.py
----------------------------------------------------------------
HKU HSS Beam RL — VecNormalize distortion probe

WHY THIS EXISTS:
    Three consecutive fixes (exp58 grade hysteresis, exp59 geometry
    damping, exp60 grade tier-restriction) each suppressed oscillation
    on the specific action channel they targeted, only for the SAME
    scenario (13m/140kN/welded) to show oscillation relocate to a
    different, unprotected channel each time. This is checkable as a
    single unifying cause WITHOUT retraining anything: if VecNormalize
    is distorting/saturating the observation at these extreme,
    near-design-envelope states (h/b near ceiling, high Med), the
    policy may be unable to reliably distinguish nearby states on ANY
    channel -- which would explain why restricting one channel just
    pushes the same underlying indecision onto another.

WHAT THIS SCRIPT DOES:
    Loads the trained VecNormalize stats (no model, no training) and
    prints RAW vs NORMALISED observations for hand-picked state pairs
    -- by default, the two alternating micro-states actually seen in
    exp60's 13m/140kN/welded trace (steps 36 vs 37, util 0.839 vs
    0.745 despite very similar geometry). If the normalised vectors
    for these two genuinely-different-reward states come out nearly
    identical, or if any dimension is pinned at the clip_obs boundary,
    that's direct evidence for the VecNormalize hypothesis. If they
    remain clearly distinguishable, that hypothesis is NOT supported
    by this evidence and the oscillation is more likely a policy-
    capacity issue in this specific region.

USAGE:
    python inspect_obs_normalization.py \
        --vecnorm ./models/hss_exp60/vecnormalize.pkl \
        --env ./env/high_rise_generative_env_exp60.py

DEPENDENCIES: stable-baselines3, gymnasium, numpy
================================================================
"""

import argparse
import importlib.util
import os
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def load_env_class(env_path):
    spec = importlib.util.spec_from_file_location("env_module", env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HighRiseGenerativeEnv


def make_state_obs(EnvClass, span_m, load_kNm, section_type,
                    h, b, tf, tw, fy, ltb_factor=0.25):
    """Build a bare env, force it into an exact geometry/grade state,
    and return its raw observation vector (unnormalised)."""
    env = EnvClass(use_storey_load_scaling=False, ltb_restraint_factor=ltb_factor)
    env.np_random = np.random.default_rng(0)
    env.span, env.load, env.section_type = span_m * 1000.0, load_kNm, section_type
    env.h, env.b, env.tf, env.tw, env.fy = h, b, tf, tw, fy
    env.storey = 20
    env.curr_step = 10
    # Populate current_* / prev_* via one internal analysis pass, matching
    # what the real env would have accumulated by this point in an episode.
    util, mass, penalty, class_loss, chi_lt, ec3 = env._ec3_analysis()
    cost, co2, _ = env._calculate_cost_co2(mass)
    env.current_util, env.current_mass = util, mass
    env.current_cost, env.current_co2 = cost, co2
    env.current_chi_lt = chi_lt
    env.current_class = ec3.get("section_class", 4)
    env.current_Mrd = ec3.get("Mrd", 1.0)
    env.moment_ratio = ec3["Med"] / max(ec3["Mrd"], 1e-6)
    return env._get_obs(), util


# Obs feature names, in order, matching _get_obs() in the 27-dim env lineage.
# Update this list if your env's obs layout differs (25-dim exp54-era files
# won't have the last two).
FEATURE_NAMES = [
    "span", "load", "storey", "h", "b", "tf", "tw", "h/b_ratio", "fy",
    "section_flag", "util", "moment_ratio", "mass", "cost", "co2", "chi_lt",
    "flange_slend", "web_slend", "class1", "class2", "class3", "class4",
    "util_delta", "mass_delta", "Med_norm", "h_target_norm", "geometry_gap",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vecnorm", required=True)
    parser.add_argument("--env", default="../env/high_rise_generative_env.py")
    parser.add_argument("--ltb-factor", type=float, default=0.25)
    args = parser.parse_args()

    if not os.path.exists(args.env):
        print(f"[ERROR] Env not found: {args.env}"); return
    if not os.path.exists(args.vecnorm):
        print(f"[ERROR] VecNormalize stats not found: {args.vecnorm}"); return

    EnvClass = load_env_class(args.env)

    # Default: the two alternating micro-states actually observed in
    # exp60's 13m/140kN/welded trace, steps 36 and 37.
    state_A = dict(span_m=13.0, load_kNm=140.0, section_type="welded",
                   h=739.0, b=285.0, tf=35.0, tw=25.0, fy=460.0)  # util~0.839 (better)
    state_B = dict(span_m=13.0, load_kNm=140.0, section_type="welded",
                   h=744.0, b=287.0, tf=35.0, tw=25.0, fy=460.0)  # util~0.745 (worse)

    dummy_env = EnvClass(use_storey_load_scaling=False, ltb_restraint_factor=args.ltb_factor)
    venv = DummyVecEnv([lambda: dummy_env])
    vec_norm = VecNormalize.load(args.vecnorm, venv)
    vec_norm.training = False
    vec_norm.norm_reward = False

    raw_A, util_A = make_state_obs(EnvClass, ltb_factor=args.ltb_factor, **state_A)
    raw_B, util_B = make_state_obs(EnvClass, ltb_factor=args.ltb_factor, **state_B)

    norm_A = vec_norm.normalize_obs(raw_A[np.newaxis, :])[0]
    norm_B = vec_norm.normalize_obs(raw_B[np.newaxis, :])[0]

    clip_obs = getattr(vec_norm, "clip_obs", None)

    print(f"State A: h={state_A['h']}, b={state_A['b']}, tf={state_A['tf']}, "
          f"tw={state_A['tw']}, fy=S{state_A['fy']:.0f}  ->  util={util_A:.3f}")
    print(f"State B: h={state_B['h']}, b={state_B['b']}, tf={state_B['tf']}, "
          f"tw={state_B['tw']}, fy=S{state_B['fy']:.0f}  ->  util={util_B:.3f}")
    print(f"VecNormalize clip_obs = {clip_obs}\n")

    n = len(raw_A)
    names = FEATURE_NAMES if len(FEATURE_NAMES) == n else [f"obs[{i}]" for i in range(n)]

    print(f"{'feature':<16} {'raw_A':>8} {'raw_B':>8} {'raw_diff':>9} | "
          f"{'norm_A':>8} {'norm_B':>8} {'norm_diff':>10}  flag")
    print("-" * 90)
    for i, name in enumerate(names):
        raw_diff = raw_A[i] - raw_B[i]
        norm_diff = norm_A[i] - norm_B[i]
        flag = ""
        if clip_obs is not None and (abs(norm_A[i]) >= clip_obs * 0.98
                                      or abs(norm_B[i]) >= clip_obs * 0.98):
            flag = "<-- near/at clip_obs boundary"
        elif abs(raw_diff) > 1e-6 and abs(norm_diff) < 1e-3:
            flag = "<-- raw states differ but normalised obs nearly IDENTICAL"
        print(f"{name:<16} {raw_A[i]:>8.4f} {raw_B[i]:>8.4f} {raw_diff:>9.4f} | "
              f"{norm_A[i]:>8.4f} {norm_B[i]:>8.4f} {norm_diff:>10.4f}  {flag}")

    print("\nInterpretation:")
    print("  - Any row flagged 'near/at clip_obs boundary' supports the")
    print("    VecNormalize-saturation hypothesis directly.")
    print("  - Any row flagged 'nearly IDENTICAL' despite differing raw")
    print("    values means the policy literally cannot distinguish these")
    print("    two states on that feature after normalisation -- also")
    print("    supports the hypothesis.")
    print("  - If neither flag fires anywhere, the normalised observation")
    print("    faithfully preserves the distinction between these two")
    print("    states, and VecNormalize is NOT the explanation for this")
    print("    oscillation -- the cause is more likely in the policy/value")
    print("    network's learned behaviour in this input region.")


if __name__ == "__main__":
    main()
