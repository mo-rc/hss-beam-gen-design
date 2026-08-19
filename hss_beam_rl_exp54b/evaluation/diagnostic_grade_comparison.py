"""
================================================================
diagnostic_grade_comparison.py  —  Compact Edition
----------------------------------------------------------------
HKU HSS Beam RL — Grade Comparison Diagnostic

OUTPUT PER SCENARIO (~15 lines):
  - Policy output geometry + grade
  - Summary table: util, mass, cost, CO2, chi_lt, Mrd, class
    for each grade at the frozen policy geometry
  - Minimum-mass sweep: lightest feasible section per grade
  - Mass ordering check: confirms S690 < S620 < ... ordering

REMOVED (were noise, not signal):
  - Verbose per-grade EC3 block (15 lines × 5 grades)
  - LTB sensitivity table (established in exp50, stable)
  - Cost model breakdown (one-time analysis, done)

USAGE
-----
    python diagnostic_grade_comparison.py \
        --model ./models/hss_exp54/best_model.zip \
        --env   ./env/high_rise_generative_env_claude_final.py

DEPENDENCIES:  pip install stable-baselines3 gymnasium numpy
================================================================
"""

import argparse
import importlib.util
import sys
import os
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[WARNING] stable-baselines3 not found. EC3-only mode.\n")


# ================================================================
# CONFIGURATION — edit these as needed
# ================================================================

SCENARIOS = [
    (13.0, 140.0, "welded"),
    (15.0, 140.0, "rolled"),
    (15.0, 120.0, "rolled"),
    (13.0, 100.0, "welded"),
    (11.0, 120.0, "rolled"),
]

GRADES              = [460, 500, 550, 620, 690]
LTB_RESTRAINT_FACTOR = 0.25
SWEEP_STEPS          = 5   # 5^4 = 625 geometries per grade

# If True, prints raw vs VecNormalize-normalised obs[:6] (span, load, storey,
# h, b, tf) at t=0 for each scenario. Tests the hypothesis that VecNormalize's
# clip_obs=10.0 (applied on top of already hand-normalised [0,1] features) is
# saturating/collapsing distinguishability between these out-of-distribution
# heavy-demand scenarios BEFORE the policy ever sees them.
DEBUG_OBS = False

SEP  = "=" * 68
SEP2 = "-" * 68


# ================================================================
# CORE EC3 HELPERS
# ================================================================

def load_env_class(env_path):
    spec   = importlib.util.spec_from_file_location("env_module", env_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HighRiseGenerativeEnv


def make_bare_env(EnvClass, span_m, load_kNm, section_type):
    env = EnvClass(use_storey_load_scaling=False,
                   ltb_restraint_factor=LTB_RESTRAINT_FACTOR)
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


def ec3_for_geometry(env, h, b, tf, tw, fy, section_type):
    env.h = h; env.b = b; env.tf = tf; env.tw = tw
    env.fy = float(fy); env.section_type = section_type
    util, mass, penalty, class_loss, chi_lt, ec3 = env._ec3_analysis()
    cost, co2, lca = env._calculate_cost_co2(mass)
    return util, mass, cost, co2, chi_lt, ec3, lca, penalty


def geometry_sweep(env, section_type, fy):
    """Find minimum-mass section with util ∈ [0.88, 1.05]."""
    best_mass, best_geom, best_util = np.inf, None, None
    for h  in np.linspace(250, 750, SWEEP_STEPS):
        for b  in np.linspace(120, 300, SWEEP_STEPS):
            for tf in np.linspace(8,   35, SWEEP_STEPS):
                for tw in np.linspace(6,   25, SWEEP_STEPS):
                    util, mass, *_, ec3, _, penalty = \
                        ec3_for_geometry(env, h, b, tf, tw, fy, section_type)
                    if (0.88 <= util <= 1.05
                            and ec3["section_class"] <= 3
                            and penalty == 0
                            and mass < best_mass):
                        best_mass, best_geom, best_util = mass, (h, b, tf, tw), util
    return best_geom, best_mass, best_util


# ================================================================
# POLICY ROLLOUT
# ================================================================

def rollout_policy(model, EnvClass, span_m, load_kNm, section_type,
                   vecnorm_path=None):
    """Deterministic rollout on a fixed scenario with correct VecNormalize."""
    vec_env  = DummyVecEnv([lambda: make_bare_env(EnvClass, span_m,
                                                   load_kNm, section_type)])
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

    # ── FIX [diagnostic bug 1] ─────────────────────────────────────────
    # active.reset() above calls the REAL HighRiseGenerativeEnv.reset(),
    # which draws its own random span/load/geometry from scratch. Since
    # make_bare_env() seeds np_random = default_rng(42) fresh every call,
    # that random draw was IDENTICAL across all 5 SCENARIOS (a phantom
    # ~9.9m/123kN/m context), and only span/load were overwritten back
    # to the real scenario above — h/b/tf/tw/fy were silently left at
    # that shared, scenario-irrelevant draw. Every "different" scenario
    # rollout was therefore starting the policy from bit-identical
    # geometry, which is very likely the dominant cause of the observed
    # "same collapsed geometry regardless of span/load" signature.
    #
    # Reseed geometry deterministically (no randomness) from the ACTUAL
    # scenario's span, matching reset()'s own span-proportional formula
    # but without the h_noise/grade-curriculum randomization, so every
    # scenario starts from its own comparable, context-appropriate point.
    span_m_now = raw.span / 1000.0
    h_target   = float(np.clip(span_m_now * 42.0, 250.0, 750.0))
    raw.h  = h_target
    raw.b  = float(np.clip(h_target / 3.0, 120.0, 300.0))
    raw.tf = 20.0
    raw.tw = 12.0
    raw.fy = 500.0   # neutral mid-grade start (was: leftover phantom-draw grade)
    # ────────────────────────────────────────────────────────────────

    # Recompute obs from forced scenario and normalise through wrapper
    raw_obs = raw._get_obs()[np.newaxis, :]
    obs = vec_norm.normalize_obs(raw_obs) if vec_norm else raw_obs

    if DEBUG_OBS:
        print(f"    [debug] raw_obs[:6]  = {np.round(raw_obs[0][:6], 3).tolist()}")
        print(f"    [debug] norm_obs[:6] = {np.round(obs[0][:6], 3).tolist()}")

    max_steps = getattr(raw, "max_steps", 40)
    done = False;  step = 0

    # ── FIX [diagnostic bug 2 + 3] ───────────────────────────────────────
    # BUG 2 (trajectory visibility): the original version returned only
    # raw.h/b/tf/tw/fy AFTER the loop exits, conflating "never got close"
    # with "got close then drifted." Fixed by tracking every step.
    #
    # BUG 3 (auto-reset corruption — the big one): SB3 VecEnvs auto-reset
    # the underlying env internally, inside step(), the instant a step
    # returns done=True (whether from genuine termination OR from hitting
    # max_steps truncation) — BEFORE returning control to this code. That
    # means reading raw.h/b/tf/tw/fy via live attribute access after ANY
    # step call that set done=True (i.e. always, since that's the only way
    # this loop exits) reads a FRESH RANDOM DRAW from the internal reset,
    # not the policy's actual terminal design. Verified directly: info['h']
    # showed 546.0 (correct) while raw.h read immediately after the same
    # step() call showed 488.4 (already reset). Since no randomness is
    # consumed anywhere in step()/_update_design()/_ec3_analysis()/reward/
    # termination — only reset() draws — and every scenario gets a FRESH
    # np_random(42) via make_bare_env(), this phantom auto-reset draw is
    # BIT-IDENTICAL across every scenario regardless of span/load/steps
    # taken. That is almost certainly what produced the "policy collapses
    # to one fixed geometry regardless of context" signature in every
    # diagnostic run from exp53 onward — it was never reading real policy
    # output for the terminal state.
    #
    # Fix: source ALL geometry from the `info` dict (populated correctly
    # during the actual step computation, before any internal reset),
    # never from live env attributes, for every trajectory entry AND the
    # final returned values.
    trajectory = []
    while not done and step < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = active.step(action)
        info = infos[0] if isinstance(infos, (list, tuple)) else infos
        ec3 = info.get("ec3", {}) if isinstance(info, dict) else {}
        trajectory.append({
            "step": step,
            "h": info.get("h"), "b": info.get("b"),
            "tf": info.get("tf"), "tw": info.get("tw"),
            "fy": info.get("fy"), "section_type": info.get("section_type"),
            "util": info.get("utilization", None) if isinstance(info, dict) else None,
            "section_class": ec3.get("section_class", None),
        })
        step += 1

    terminated_early = done and step < max_steps
    feasible_steps = [
        t for t in trajectory
        if t["util"] is not None and 0.90 <= t["util"] <= 1.05
        and t["section_class"] is not None and 0 < t["section_class"] < 4
    ]
    if feasible_steps:
        best = min(feasible_steps, key=lambda t: abs(t["util"] - 0.95))
        print(f"    [trajectory] terminated_early={terminated_early} steps={step} "
              f"best_feasible_step={best['step']} best_util={best['util']:.3f} "
              f"geometry@best: h={best['h']:.0f} b={best['b']:.0f} "
              f"tf={best['tf']:.1f} tw={best['tw']:.1f} fy=S{best['fy']:.0f}")
    else:
        utils_seen = [t["util"] for t in trajectory if t["util"] is not None]
        min_util_seen = min(utils_seen) if utils_seen else None
        print(f"    [trajectory] terminated_early={terminated_early} steps={step} "
              f"NO feasible step reached — best util seen was "
              f"{min_util_seen:.3f} (target 0.90-1.05)" if min_util_seen is not None
              else f"    [trajectory] no util data captured")

    # Return the LAST recorded step's info-sourced geometry — the actual
    # terminal design the policy produced — NOT live raw.* attributes,
    # which may already reflect an internal auto-reset by this point.
    last = trajectory[-1] if trajectory else None
    if last is not None and last["h"] is not None:
        return last["h"], last["b"], last["tf"], last["tw"], last["section_type"], last["fy"]
    # Fallback (should not normally happen): no trajectory captured at all.
    return raw.h, raw.b, raw.tf, raw.tw, raw.section_type, raw.fy


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None)
    parser.add_argument("--vecnorm", default=None)
    parser.add_argument("--env",
        default="../env/high_rise_generative_env.py")
    args = parser.parse_args()

    if not os.path.exists(args.env):
        print(f"[ERROR] Env not found: {args.env}"); sys.exit(1)

    EnvClass = load_env_class(args.env)
    print(f"\n  Env   : {args.env}")

    model = None
    if args.model and SB3_AVAILABLE and os.path.exists(args.model):
        model = PPO.load(args.model)
        print(f"  Model : {args.model}")

    vecnorm_path = args.vecnorm
    if vecnorm_path is None and args.model:
        candidate = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        if os.path.exists(candidate):
            vecnorm_path = candidate
    status = vecnorm_path if vecnorm_path else "not found"
    print(f"  VecNorm: {status}\n")

    for span_m, load_kNm, section_type in SCENARIOS:

        Med_kNm = load_kNm * span_m ** 2 / 8.0
        print(SEP)
        print(f"  {span_m:.0f}m / {load_kNm:.0f} kN/m / {section_type}"
              f"   Med={Med_kNm:.0f} kNm")
        print(SEP)

        bare_env = make_bare_env(EnvClass, span_m, load_kNm, section_type)

        # ── Policy geometry ──────────────────────────────────────
        if model is not None:
            try:
                h_p, b_p, tf_p, tw_p, sec_p, fy_p = rollout_policy(
                    model, EnvClass, span_m, load_kNm,
                    section_type, vecnorm_path)
                print(f"  Policy → h={h_p:.0f} b={b_p:.0f} tf={tf_p:.1f} "
                      f"tw={tw_p:.1f} fy=S{fy_p:.0f} {sec_p}")
            except Exception as e:
                import traceback; traceback.print_exc()
                h_p, b_p, tf_p, tw_p, sec_p, fy_p = \
                    min(span_m*42, 750), 200.0, 20.0, 12.0, section_type, 500.0
                print(f"  Policy rollout failed ({e}) — using fallback geometry")
        else:
            h_p  = min(span_m * 42.0, 750.0)
            b_p  = min(h_p / 3.0, 300.0)
            tf_p, tw_p, sec_p, fy_p = 20.0, 12.0, section_type, 500.0
            print(f"  Fallback → h={h_p:.0f} b={b_p:.0f} tf={tf_p} "
                  f"tw={tw_p} fy=S{fy_p:.0f} {sec_p}")

        # ── Summary table at frozen geometry ─────────────────────
        print(f"\n  {'Grade':<7} {'util':>6} {'mass':>7} {'cost£':>7} "
              f"{'CO2':>7} {'χ_lt':>6} {'Mrd':>7} {'λ_lt':>6} "
              f"{'cls':>4}  {'govern':>6}")
        print(f"  {'-'*7} {'-'*6} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*6} {'-'*7} {'-'*6} "
              f"{'-'*4}  {'-'*6}")

        summary_rows = []
        for fy in GRADES:
            util, mass, cost, co2, chi_lt, ec3, lca, penalty = \
                ec3_for_geometry(bare_env, h_p, b_p, tf_p, tw_p, fy, sec_p)
            Mrd       = ec3["Mrd"]
            lambda_lt = ec3["lambda_lt"]
            cls       = ec3["section_class"]
            governing = "DEFL" if ec3["deflection_util"] > ec3["moment_util"] \
                        else "MOM"
            print(f"  S{fy:<6} {util:>6.3f} {mass:>7.0f} {cost:>7.0f} "
                  f"{co2:>7.0f} {chi_lt:>6.3f} {Mrd:>7.1f} {lambda_lt:>6.3f} "
                  f"{cls:>4}  {governing:>6}")
            summary_rows.append((fy, util, mass, cost, co2,
                                  chi_lt, Mrd, lambda_lt, cls))

        feasible = [r for r in summary_rows if 0.88 <= r[1] <= 1.05]
        if not feasible:
            print("  !! No grade feasible at this geometry — section too small")

        # ── Minimum-mass sweep ───────────────────────────────────
        print(f"\n  Sweep (min-mass, util∈[0.88,1.05]):")
        print(f"  {'Grade':<7} {'util':>6} {'mass':>7} "
              f"{'h':>5} {'b':>5} {'tf':>5} {'tw':>5}")
        print(f"  {'-'*7} {'-'*6} {'-'*7} "
              f"{'-'*5} {'-'*5} {'-'*5} {'-'*5}")

        sweep_results = []
        for fy in GRADES:
            geom, best_mass, best_util = geometry_sweep(bare_env, sec_p, fy)
            if geom is not None:
                h, b, tf, tw = geom
                sweep_results.append((fy, best_util, best_mass))
                print(f"  S{fy:<6} {best_util:>6.3f} {best_mass:>7.0f} "
                      f"{h:>5.0f} {b:>5.0f} {tf:>5.1f} {tw:>5.1f}")
            else:
                sweep_results.append((fy, None, None))
                print(f"  S{fy:<6} {'---':>6} {'no feasible':>7}")

        # Mass ordering check
        valid = [(r[0], r[2]) for r in sweep_results if r[2] is not None]
        ordering = []
        for i in range(len(valid) - 1):
            fy_lo, m_lo = valid[i]
            fy_hi, m_hi = valid[i + 1]
            tag = "OK" if m_hi <= m_lo else "INVERTED"
            ordering.append(f"S{fy_lo}→S{fy_hi}:{tag}")
        print(f"\n  Mass order: {' | '.join(ordering) if ordering else 'n/a'}")
        print()

    print(SEP)
    print("  Diagnostic complete.")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
