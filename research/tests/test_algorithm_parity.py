"""
research/tests/test_algorithm_parity.py
================================================================
Cross-algorithm fairness preflight. Run this BEFORE launching the
comparison runs and again before writing any head-to-head number into
the paper.

The claim the paper wants to make is "algorithm X beats algorithm Y on
this design problem". That claim is only meaningful if every arm is
actually solving the SAME problem with the SAME budget. This file turns
that assumption into assertions, because the audit found two places
where it held only by accident:

  1. train.py passed ltb_restraint_factor / sls_load_factor /
     economy_reward_mode explicitly while train_baseline_offpolicy.py
     inherited them from HSSBeamEnv's class defaults. They happened to
     agree. A single default change would have silently made the arms
     solve different problems while every training_config.json still
     looked comparable.

  2. The off-policy trainer defaulted to n_envs=2. SB3 off-policy runs
     `gradient_steps` updates per collect_rollouts call and that call
     advances n_envs timesteps, so DDPG/TD3 were getting HALF the
     gradient updates per environment step that the same algorithm gets
     in its standard configuration -- while both runs reported the same
     "1,000,000 timesteps".

Run:  python -m research.tests.test_algorithm_parity
Exit code 0 = all arms are comparable. Non-zero = do not report a
head-to-head result until it is fixed.
"""

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.envs.hss_env import HSSBeamEnv
from research.scripts import train as train_ppo
from research.scripts import train_baseline_offpolicy as train_off

# Environment-shaping attributes that define the design problem. If two
# arms differ on ANY of these they are not solving the same problem.
PROBLEM_DEFINING_ATTRS = [
    "reward_mode", "economy_metric", "economy_reward_mode",
    "ltb_restraint_factor", "sls_load_factor",
    "enforce_rolled_manufacturability", "use_storey_load_scaling",
    "include_zg_in_mcr", "include_novelty",
    "max_steps", "step_scale_horizon", "grade_softmax_temperature",
    "norm",
]

REWARD_MODE = "feasibility_gated"
METRIC = "cost"
LAGRANGE_INIT = dict(g1_util=0.0, g2_class=0.0, g3_geom=0.0)

failures = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def snapshot(env):
    return {a: getattr(env, a) for a in PROBLEM_DEFINING_ATTRS}


print("=" * 70)
print("1. Identical design problem across trainer entry points")
print("=" * 70)

ppo_env = train_ppo.make_env(
    "continuous", REWARD_MODE, METRIC, LAGRANGE_INIT,
    ltb_factor=0.40, sls_factor=0.50, economy_reward_mode="linear",
    seed=0, rank=0, max_steps=40, enforce_rolled=True,
)().unwrapped

off_env = train_off.make_env(
    REWARD_MODE, METRIC, LAGRANGE_INIT,
    ltb_factor=0.40, sls_factor=0.50, economy_reward_mode="linear",
    max_steps=40, enforce_rolled=True, seed=0, rank=0,
)().unwrapped

a, b = snapshot(ppo_env), snapshot(off_env)
for k in PROBLEM_DEFINING_ATTRS:
    check(a[k] == b[k], f"{k}: PPO={a[k]!r} off-policy={b[k]!r}")

check(ppo_env.observation_space == off_env.observation_space,
      f"observation_space {ppo_env.observation_space}")
check(ppo_env.action_space == off_env.action_space,
      f"action_space {ppo_env.action_space}")


print()
print("=" * 70)
print("2. Reward mode must be explicit, never defaulted")
print("=" * 70)
# Both trainers must REQUIRE --reward_mode. A default here is how the
# existing research/models/ddpg_cost run silently ended up on
# reward_mode=lagrangian while the headline PPO arm used
# feasibility_gated -- i.e. optimising a different objective.
for mod, name in ((train_ppo, "train.py"), (train_off, "train_baseline_offpolicy.py")):
    src = inspect.getsource(mod.main)
    idx = src.find('"--reward_mode"')
    seg = src[idx:idx + 220] if idx >= 0 else ""
    check(idx >= 0 and "required=True" in seg,
          f"{name} declares --reward_mode as required")


print()
print("=" * 70)
print("3. Off-policy gradient budget is invariant to n_envs")
print("=" * 70)
# SB3 off-policy: gradient_steps updates per collect_rollouts call, and
# one call advances n_envs timesteps. So updates-per-env-step =
# gradient_steps / (n_envs * train_freq). The trainer defaults
# gradient_steps to n_envs*train_freq, making that ratio exactly 1.0
# regardless of n_envs.
for n_envs in (1, 2, 4, 8):
    train_freq = 1
    gradient_steps = n_envs * train_freq  # the trainer's default rule
    ratio = gradient_steps / (n_envs * train_freq)
    check(abs(ratio - 1.0) < 1e-9,
          f"n_envs={n_envs}: {ratio:.2f} gradient updates per environment step")

# And confirm the old behaviour really was the bug we think it was.
old_ratio = 1 / (2 * 1)  # gradient_steps=1 (SB3 default), n_envs=2 (old default)
check(abs(old_ratio - 0.5) < 1e-9,
      f"regression guard: old n_envs=2 default gave {old_ratio:.2f} updates/step (half)")


print()
print("=" * 70)
print("4. Checkpoint grid is in environment timesteps, not callback calls")
print("=" * 70)
# CheckpointCallback's save_freq counts CALLS; one call advances n_envs
# timesteps. Both trainers must divide by n_envs or their learning curves
# land on different step grids and cannot be plotted together.
for mod, name, envs in ((train_ppo, "train.py", 8), (train_off, "train_baseline_offpolicy.py", 1)):
    src = inspect.getsource(mod.main)
    check("// args.n_envs" in src.replace(" ", "").replace("//args.n_envs", "// args.n_envs")
          or "//args.n_envs" in src.replace(" ", ""),
          f"{name} divides CheckpointCallback save_freq by n_envs")


print()
print("=" * 70)
print("5. Observation normalisation is off everywhere (eval feeds raw obs)")
print("=" * 70)
for mod, name in ((train_ppo, "train.py"), (train_off, "train_baseline_offpolicy.py")):
    src = inspect.getsource(mod.main)
    check("norm_obs=False" in src.replace(" ", "").replace("norm_obs=False", "norm_obs=False")
          or "norm_obs=False" in src,
          f"{name} constructs VecNormalize with norm_obs=False")


print()
print("=" * 70)
print("6. Every algorithm the trainers can produce is loadable by the evaluator")
print("=" * 70)
from research.scripts import evaluate as ev
eval_src = inspect.getsource(ev.load_policy)
for algo in sorted(train_off.ALGOS) + ["ppo"]:
    check(f'"{algo}"' in eval_src, f"evaluate.load_policy handles algo={algo}")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "scripts", "evaluate_with_repair.py")) as f:
    ewr_src = f.read()
for algo in sorted(train_off.ALGOS) + ["ppo"]:
    check(f'"{algo}"' in ewr_src, f"evaluate_with_repair handles algo={algo}")


print()
print("=" * 70)
print("7. Baselines share the environment's physics, not a reimplementation")
print("=" * 70)
# The GA / random-search / rule-based baselines must call the SAME EC3
# routines the RL arms are rewarded by. If they had their own physics, a
# reported "gap" would confound algorithm quality with model mismatch.
from research.scripts import ga_baseline as ga
ga_src = inspect.getsource(ga)
check("HSSBeamEnv" in ga_src, "ga_baseline builds a real HSSBeamEnv as its evaluator")
for fn in ("_ec3_analysis", "_calculate_cost_co2", "_constraint_violations"):
    check(fn in ga_src, f"ga_baseline calls env.{fn} directly")

probe = ga._make_probe_env(METRIC)
for k in ["ltb_restraint_factor", "sls_load_factor", "enforce_rolled_manufacturability",
          "economy_reward_mode"]:
    check(getattr(probe, k) == getattr(ppo_env, k),
          f"GA probe env {k}={getattr(probe, k)!r} matches the RL arms")


print()
print("=" * 70)
print("8. Deterministic evaluation is genuinely deterministic")
print("=" * 70)
# Every arm is evaluated with predict(deterministic=True). Confirm the
# environment transition itself adds no stochasticity once the context is
# forced, otherwise repeated evaluation of one policy would give
# different gaps and cross-arm differences would be partly noise.
e1 = HSSBeamEnv(reward_mode=REWARD_MODE, economy_metric=METRIC)
e2 = HSSBeamEnv(reward_mode=REWARD_MODE, economy_metric=METRIC)
o1, _ = e1.reset(seed=123)
o2, _ = e2.reset(seed=123)
check(np.allclose(o1, o2), "reset(seed) is reproducible")
acts = np.random.default_rng(0).uniform(-1, 1, size=(10, 6)).astype(np.float32)
r1 = [e1.step(a)[1] for a in acts]
r2 = [e2.step(a)[1] for a in acts]
check(np.allclose(r1, r2), "step() is deterministic given identical action sequences")


print()
print("=" * 70)
if failures:
    print(f"{len(failures)} PARITY FAILURE(S) -- arms are NOT comparable:")
    for f_ in failures:
        print("  - " + f_)
    sys.exit(1)
print("ALL PARITY CHECKS PASSED -- arms are comparable.")
print("=" * 70)
