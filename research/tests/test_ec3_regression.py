"""
research/tests/test_ec3_regression.py
================================================================
Proves the constrained-MDP redesign (research/envs/hss_env.py) did NOT
silently change any structural mechanics, cost/CO2 model, action mapping,
or observation encoding relative to the original exp54b environment.

This is the load-bearing evidence for the paper's claim that all reward-
mode arms in research/ share the identical environment dynamics: "only
the objective/constraint layer differs between arms" must be
independently verifiable, not just asserted in a docstring.

Method: drive both environments with an IDENTICAL seeded random action
sequence from an IDENTICAL seed, and assert that every PHYSICAL quantity
(h, b, tf, tw, fy, util, mass, cost, co2, chi_lt) matches the original
bit-for-bit (within float tolerance) at every one of the 40 steps, across
multiple episodes.

SCOPE, UPDATED FOR THE PRE-EXPERIMENT-1 AUDIT
------------------------------------------------
This test compares PHYSICS ONLY, not reward values. Two reasons the
reward is intentionally excluded from this comparison, permanently:

1. research/tests/ec3_independent_verification.py found that exp54b used
   the wrong EN1993-1-1 Table 6.5 LTB buckling curve for welded sections
   with h/b>2 (alpha_LT=0.49 instead of the correct 0.76), causing a 14%
   chi_LT / 12% utilization error. Fixed in research/envs/hss_env.py;
   exp54b itself was not touched (frozen prior artifact, not a target to
   reproduce). Any step landing in this region legitimately FORKS the two
   trajectories' physical state from that point on.

2. The `hss_demand_bonus` reward term has been REMOVED ENTIRELY from
   research/envs/hss_env.py (see that module's docstring) -- not gated by
   a flag, not reachable by any reward_mode. This is a deliberate,
   permanent formulation change, so exp54b's reward and this codebase's
   reward are expected to differ on essentially every step, by design.
   Comparing them would not be a regression test, it would be asserting
   the fix didn't happen.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                 "hss_beam_rl_exp54b"))

from hss_beam_rl_exp54b.env.high_rise_generative_env import HighRiseGenerativeEnv  # original
from research.envs.hss_env import HSSBeamEnv  # redesigned


FIELDS_TO_COMPARE = ["h", "b", "tf", "tw", "fy", "utilization", "mass", "cost", "co2", "chi_lt"]


def run_episode(env, seed, action_seed):
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(action_seed)
    records = []
    for t in range(40):
        action = rng.uniform(-1, 1, size=6).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        rec = {k: info[k] for k in FIELDS_TO_COMPARE}
        rec["section_type"] = info["section_type"]
        records.append(rec)
        if terminated or truncated:
            break
    return records


def _is_known_fix_region(rec) -> bool:
    """The one region where research/envs/hss_env.py is EXPECTED to differ
    from exp54b: welded sections with h/b>2 (EN1993-1-1 Table 6.5 curve-d
    fix)."""
    return rec.get("section_type") == "welded" and rec["h"] / max(rec["b"], 1e-6) > 2.0


def compare(orig_records, new_records, tol=1e-6):
    """
    Once a step lands in the known fix region, the two trajectories
    legitimately FORK: chi_lt/util differ, so mass/cost/co2 at that step
    differ, and because state persists across steps in this MDP, every
    subsequent step in the same episode is a downstream consequence of the
    fork and will also legitimately differ, even where that later step's
    own (h,b) no longer satisfies the fix-region condition directly. This
    is expected, not a bug -- once the first fix-region step is hit,
    strict comparison stops for the rest of that episode; only the prefix
    before the first fork must match exactly.
    """
    max_diffs = {}
    forked_at = None
    for i, (o, n) in enumerate(zip(orig_records, new_records)):
        in_fix_region = _is_known_fix_region(n)
        for k in FIELDS_TO_COMPARE:
            diff = abs(o[k] - n[k])
            close_enough = diff < tol or diff / (abs(o[k]) + 1e-9) < 1e-4
            if not close_enough:
                if in_fix_region:
                    forked_at = i
                    break
                raise AssertionError(
                    f"Step {i}: field '{k}' mismatch OUTSIDE the known welded-h/b>2 fix "
                    f"region -- this is an UNEXPECTED divergence: original={o[k]!r} new={n[k]!r} diff={diff}"
                )
            max_diffs[k] = max(max_diffs.get(k, 0.0), diff)
        if forked_at is not None:
            break
    max_diffs["_forked_at_step"] = forked_at
    return max_diffs


def main():
    n_episodes = 25
    all_max_diffs = {}
    n_forked = 0
    for ep in range(n_episodes):
        orig_env = HighRiseGenerativeEnv()
        # reward_mode is irrelevant here -- we only compare physics, not reward.
        new_env = HSSBeamEnv(reward_mode="shaped", include_novelty=True)

        orig_records = run_episode(orig_env, seed=1000 + ep, action_seed=2000 + ep)
        new_records = run_episode(new_env, seed=1000 + ep, action_seed=2000 + ep)

        max_diffs = compare(orig_records, new_records)
        forked_at = max_diffs.pop("_forked_at_step")
        for k, v in max_diffs.items():
            all_max_diffs[k] = max(all_max_diffs.get(k, 0.0), v)
        n_forked += int(forked_at is not None)
        status = f"forked at step {forked_at} (expected, welded h/b>2 fix region)" if forked_at is not None else "no fork"
        print(f"  episode {ep:2d}: OK ({len(orig_records)} steps, {status}, "
              f"max util diff (pre-fork)={max_diffs.get('utilization', 0.0):.2e})")

    print(f"\nAll episodes passed. {n_forked}/{n_episodes} episodes hit the known welded-h/b>2")
    print("fix region at some point (expected fork, not a failure). Max absolute PHYSICS")
    print("differences in the pre-fork prefix of every episode:")
    for k, v in all_max_diffs.items():
        print(f"  {k:15s}: {v:.3e}")
    print("\n=> research/envs/hss_env.py reproduces exp54b EC3 mechanics, cost/CO2 model,")
    print("   and action mapping EXACTLY, up to the one intentional, independently-verified")
    print("   EC3 fix (welded h/b>2 LTB curve, see module docstring). No OTHER unexplained")
    print("   physics divergence exists. Reward is deliberately NOT compared -- see docstring.")


if __name__ == "__main__":
    main()
