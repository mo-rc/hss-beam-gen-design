"""
research/envs/hss_reparam_env.py
================================================================
H1 experiment env: one-shot, feasible-by-construction reparameterized
action space, built on top of HSSBeamEnv.

WHY THIS EXISTS (see research/README.md Sec 6-8 and manuscript_final_
comprehensive.md Sec 6-7): the existing diagnosis already shows, with
zero training, that:
  - blind uniform sampling in a REPARAMETERIZED space (h, b/h, lambda_f,
    lambda_w, grade, type) -- feasible-by-construction w.r.t. section
    class, then bisected to utilisation == 1.0 -- reaches 5.3% mean gap
    at 4,800 evaluations, beating the fully-trained 5-seed incremental
    PPO policy (9.05% at 112 evals) in quality (research/scripts/
    diag_reparam_probe.py, research/results/c1_reparam_budget_table.csv)
  - the incumbent PPO policy's failure mode is a Gaussian mean that
    cannot sit on the discontinuous Class-3/Class-4 cliff and so
    retreats several sigma away from it (util converges to ~0.84, not
    1.0)
This was flagged as the highest-priority, not-yet-executed future-work
item ("Reformulated action space... Hypothesis: this closes the gap to
GA"). This file makes that hypothesis testable BY TRAINING: it wraps the
identical decode-and-bisect-to-boundary procedure used by the (already
validated, zero-training) probe as a single-step Gym env, so PPO learns
to predict the reparameterized action directly instead of the raw
(h,b,tf,tw) increments.

WHAT IS AND ISN'T CHANGED (per the constraint that EC3 physics, ground
truth, objective, and evaluation rules must not be touched to make RL
look better):
  - `_ec3_analysis`, `_calculate_cost_co2`, `_economy`, the ground-truth
    CSVs, and the evaluation harness (`evaluate.py`) are all REUSED
    UNCHANGED (inherited from HSSBeamEnv / imported from the existing,
    already-published probe module). Nothing about what counts as
    feasible, optimal, or how gap is computed is touched.
  - What changes is ONLY the action encoding/decoding (representation),
    exactly the axis the user authorised changing, and exactly the axis
    the existing diagnosis pins the bottleneck to.
  - `decode()` and `size_to_boundary()` are imported verbatim from
    `research/scripts/diag_reparam_probe.py` (not reimplemented) so
    there is a single source of truth for the reparameterization; this
    module cannot silently drift from the already-reported probe
    numbers.

Episode structure: single step per episode (this IS the one-shot
parameterization -- there is no multi-step refinement to diagnose here).
reset() reuses HSSBeamEnv.reset() for context sampling (span/load/storey
curriculum) unchanged.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.envs.hss_env import HSSBeamEnv
from research.scripts.diag_reparam_probe import decode, size_to_boundary

_LAM_F = (5.0, 14.0)
_LAM_W = (60.0, 124.0)
_B_OVER_H = (0.20, 0.75)


def _denorm(a: float, lo: float, hi: float) -> float:
    a = float(np.clip(a, -1.0, 1.0))
    return lo + (a + 1.0) * 0.5 * (hi - lo)


class HSSReparamEnv(HSSBeamEnv):
    """One-shot action space: (h, b/h, lambda_f, lambda_w, grade, type).

    Action (6,) in [-1, 1], same dimensionality as HSSBeamEnv so existing
    SB3 training scripts need only swap the env class:
        a[0] -> h            in [250, 750]  mm
        a[1] -> b/h          in [0.20, 0.75]
        a[2] -> lambda_f     in [5, 14]      (EC3 Table 5.2 flange slenderness; Class-3 cap)
        a[3] -> lambda_w     in [60, 124]    (EC3 Table 5.2 web slenderness; Class-3 cap)
        a[4] -> grade        same softmax-snap mechanism as HSSBeamEnv (mechanism-consistent
                              with the `raw` arm, per README Sec 8's deferred-comparison note)
        a[5] -> section_type same threshold mechanism as HSSBeamEnv

    tf, tw are DERIVED (not free actions) so section class <= 3 holds by
    construction; the decoded geometry is then uniformly scaled
    (utilisation-preserving-ratios) until utilisation == 1.0 by bisection,
    so the capacity constraint also holds by construction. Both
    mechanisms are byte-identical to `diag_reparam_probe.py`.
    """

    def __init__(self, *args, n_bisect: int = 18, infeasible_penalty: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bisect = n_bisect
        self.infeasible_penalty = infeasible_penalty
        # one-shot: episode is always exactly 1 step regardless of max_steps/
        # step_scale_horizon (those govern the incremental arm's schedule,
        # not this env).

    def step(self, action: np.ndarray):
        self.curr_step += 1
        action = np.asarray(action, dtype=np.float64)

        h = _denorm(action[0], *self.design_limits["h"])
        b_over_h = _denorm(action[1], *_B_OVER_H)
        lam_f = _denorm(action[2], *_LAM_F)
        lam_w = _denorm(action[3], *_LAM_W)

        # Identical grade softmax-snap mechanism to HSSBeamEnv._update_design,
        # so the grade-selection channel is mechanism-consistent with the
        # `raw` arm (only geometry encoding differs).
        grade_logits = np.array(
            [-((action[4] - c) ** 2) / self.grade_softmax_temperature for c in self._grade_centres],
            dtype=np.float64,
        )
        grade_logits -= grade_logits.max()
        grade_probs = np.exp(grade_logits)
        grade_probs /= grade_probs.sum()
        fy = float(self.grades[int(np.argmax(grade_probs))])
        section_type = "rolled" if action[5] < 0 else "welded"

        geom = decode(h, b_over_h, lam_f, lam_w, fy, section_type)
        res, g, s = size_to_boundary(self, self.span, self.load, geom, fy, section_type,
                                      n_bisect=self.n_bisect)

        if res is not None:
            self.h, self.b, self.tf, self.tw = [float(v) for v in g]
            self.fy = fy
            self.section_type = section_type
            util, mass, cost, co2 = res["util"], res["mass"], res["cost"], res["co2"]
            feasible = True
            economy = self._economy(mass, cost, co2)
            reward = -economy
        else:
            # size_to_boundary could not find a feasible scale within the
            # box bounds (e.g. context demand exceeds what this grade/type
            # can deliver even at max dimensions) -- honestly infeasible,
            # not silently clamped to something feasible.
            self.h, self.b, self.tf, self.tw = (
                self.design_limits["h"][1], self.design_limits["b"][1],
                self.design_limits["tf"][1], self.design_limits["tw"][1],
            )
            self.fy, self.section_type = fy, section_type
            util, mass, cost, co2 = 2.0, 4000.0, 8000.0 * 2, 10000.0 * 2
            feasible = False
            reward = -float(self.infeasible_penalty)

        self.current_util, self.current_mass = util, mass
        self.current_cost, self.current_co2 = cost, co2

        terminated = True
        truncated = False
        info = {
            "h": self.h, "b": self.b, "tf": self.tf, "tw": self.tw,
            "fy": self.fy, "section_type": self.section_type,
            "utilization": util, "mass": mass, "cost": cost, "co2": co2,
            "reward": reward,
            "span": self.span, "load": self.load, "storey": self.storey,
            "feasible": feasible,
            "in_target_band": feasible,
            "curr_step": self.curr_step,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
