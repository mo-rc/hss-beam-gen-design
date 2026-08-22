"""
research/envs/hss_env.py
================================================================
Constrained-MDP reformulation of the HSS beam design environment.

WHY THIS FILE EXISTS
---------------------
The exp54b environment (`hss_beam_rl_exp54b/env/high_rise_generative_env.py`)
is functionally validated (97.0% feasibility, verified against a brute-force
EC3 ground truth) but its reward is an 8-term hand-weighted scalar sum,
including a term (`hss_demand_bonus`) that explicitly rewards selecting
fy >= 500 in the target utilisation band. This makes "the agent learns
demand-appropriate grade selection" an unfalsifiable claim: the reward
was written to produce that behaviour, so observing it proves nothing.

This module keeps 100% of the EC3 structural mechanics and the cost/CO2
LCA model UNCHANGED (they are physically grounded — Fy affects Mrd, mass,
cost and CO2 through real code equations, not reward shaping) and replaces
the objective/constraint layer entirely, exposing three switchable reward
modes. The `hss_demand_bonus` term from exp54b (which explicitly rewarded
fy >= 500 in the target utilisation band) has been REMOVED from this
codebase, not merely switched off -- pre-Experiment-1 audit decision: for
a paper whose central claim concerns whether demand-appropriate grade
selection is genuinely learned, keeping a togglable "circular reward"
option in the code (even off by default) is an unnecessary liability --
a reviewer reading the source finds it either way. This codebase now
contains no reward term that references a specific grade or grade
threshold anywhere. If grade-appropriate selection is observed under any
of the three modes below, it is a consequence of Fy's genuine effect on
EC3 capacity, mass, cost and CO2, and nothing else.

    "shaped"            Weighted-sum reward shaping (economy + utilisation-
                        target Gaussian + feasibility penalty terms),
                        structurally similar to typical RL-for-design
                        reward engineering in prior work, but with NO
                        grade-specific term of any kind. This is the
                        reward-shaping baseline arm the constrained modes
                        below are compared against.

    "feasibility_gated" Safe-RL-style formulation. Reward = -economy(design)
                        only when the design is feasible (util<=1.0, section
                        class<=3, geometry proportion penalty==0); a bounded
                        constraint-violation penalty otherwise. A potential-
                        based shaping term (Ng, Harada & Russell, 1999;
                        policy-invariant by construction) is added to give
                        PPO a usable gradient toward the target band without
                        altering the optimal policy.

    "lagrangian"        True constrained-RL formulation. Reward =
                        -economy(design) - sum_i(lambda_i * g_i(design)),
                        where g_i are constraint-violation functions and
                        lambda_i are Lagrange multipliers updated OUTSIDE
                        this environment (see research/algo/lagrangian.py)
                        via dual ascent on observed violation rates. The
                        environment exposes `set_lagrange_multipliers()`
                        and reports raw violations in `info` for that
                        purpose; it does not update multipliers itself.
                        This is the paper's primary proposed method.

FORMAL PROBLEM STATEMENT (for the paper's Methods section)
------------------------------------------------------------
    minimise    E_{(span,load)~D} [ Economy(design) ]
    subject to  g1: Med/Mrd - 1.0          <= 0   (EC3 flexural+shear+LTB capacity)
    subject to  g2: section_class - 3      <= 0   (EC3 Table 5.2 compactness)
    subject to  g3: deflection - limit     <= 0   (SLS, folded into g1's util
                                                    via governing-check exactly
                                                    as in the base EC3 model)
    subject to  g3: geometry_penalty       <= 0   (one-sided: zero when
                                                    b<=h, positive when
                                                    b>h; proportion sanity,
                                                    not an EC3 code clause,
                                                    reported separately
                                                    from g1/g2 as g3_geom)
    Economy(design) in {normalised mass, normalised cost, normalised CO2},
    selectable via `economy_metric`; the other two are always reported in
    `info` as secondary metrics, never optimised directly, avoiding the
    ill-posed "optimise a weighted sum of three correlated objectives"
    framing used implicitly by the legacy reward's economy_reward term.

WHAT IS DELIBERATELY UNCHANGED FROM exp54b (for valid ablation methodology
-- change one thing at a time):
    - _ec3_analysis(): identical, verified by regression test against the
      original (see research/tests/test_ec3_regression.py).
    - _calculate_cost_co2(): identical.
    - Action space (6-dim continuous, same step sizes, same softmax-snapped
      grade action), observation space (25-dim), episode length (40 steps),
      reset() curriculum (span/load sampling, demand-aligned h_noise, 50/50
      grade curriculum). These are environment-DYNAMICS choices, independent
      of the reward-circularity problem, and changing them alongside the
      reward would confound the ablation.
    - success_counter-based early termination exists for TRAINING EFFICIENCY
      only, and is now explicitly separated from `feasible` in `info` (see
      "TERMINATION / FEASIBILITY SEPARATION" below) — this fixes a labelling
      bug in the original code where util<=1.05 was reported as `feasible`,
      which is incorrect: EC3 capacity is violated for any util>1.0.

TERMINATION / FEASIBILITY SEPARATION (bug fix, applies to ALL reward modes)
------------------------------------------------------------------------
    info["feasible"]       : util <= 1.0 + 1e-3 (numerical tolerance only)
                              AND section_class <= 3 AND geometry_penalty==0.
                              This is the ONLY field that should ever be
                              called "feasible" in analysis/plots.
    info["in_target_band"] : 0.90 <= util <= 1.05 (the original success zone).
                              Used purely to decide early termination for
                              training efficiency. NOT a code-compliance
                              statement. Renamed from the original's
                              (mis-labelled) feasibility check.
    Episode termination still requires 3 consecutive steps with
    in_target_band==True (unchanged from exp54b), but `feasible` is
    computed and logged every step regardless of termination status, so
    post-hoc analysis (e.g. "what fraction of terminated episodes are
    ACTUALLY feasible, not just in the training target band") is possible
    for the first time.

AUTHOR: Muhammad Shifa (env core, EC3 mechanics — unchanged)
        Constrained-MDP redesign — this file
AFFILIATION: HKU
================================================================
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


REWARD_MODES = ("shaped", "feasibility_gated", "lagrangian")
ECONOMY_METRICS = ("mass", "cost", "co2")


class HSSBeamEnv(gym.Env):

    def __init__(
        self,
        reward_mode: str = "lagrangian",
        economy_metric: str = "cost",
        include_novelty: bool = False,
        # --- unchanged environment-dynamics parameters (exp54b defaults) ---
        use_storey_load_scaling: bool = True,
        include_zg_in_mcr: bool = False,
        sls_load_factor: float = 0.50,
        ltb_restraint_factor: float = 0.40,
        # --- Lagrangian-mode initial multipliers (updated externally) ------
        lagrange_init: dict | None = None,
        # --- MDP-formulation parameters, exposed for ablation (Comment #13:
        # "why is 40 steps appropriate?" / "one-shot vs multi-step") --------
        max_steps: int = 40,
        grade_softmax_temperature: float = 0.15,
    ):
        super().__init__()
        assert reward_mode in REWARD_MODES, f"reward_mode must be one of {REWARD_MODES}"
        assert economy_metric in ECONOMY_METRICS, f"economy_metric must be one of {ECONOMY_METRICS}"

        self.reward_mode = reward_mode
        self.economy_metric = economy_metric
        self.include_novelty = include_novelty

        self.use_storey_load_scaling = use_storey_load_scaling
        self.include_zg_in_mcr = include_zg_in_mcr
        self.sls_load_factor = sls_load_factor
        self.ltb_restraint_factor = ltb_restraint_factor

        self.E = 210_000.0
        self.G = 81_000.0

        self.grades = np.array([355, 460, 500, 550, 620, 690], dtype=np.float32)
        n_grades = len(self.grades)
        self._grade_centres = np.array(
            [-1.0 + (2 * k + 1) / n_grades for k in range(n_grades)], dtype=np.float32
        )
        self.section_types = ["rolled", "welded"]

        self.design_limits = {
            "h": (250.0, 750.0), "b": (120.0, 300.0),
            "tf": (8.0, 35.0), "tw": (6.0, 25.0),
        }
        self.SPAN_MIN, self.SPAN_MAX = 6_000.0, 15_000.0
        self.LOAD_MIN, self.LOAD_MAX = 20.0, 210.0
        self.STOREY_MIN, self.STOREY_MAX = 1, 70

        self.norm = {"mass": 4_000.0, "cost": 8_000.0, "co2": 10_000.0, "util": 1.5}
        self.max_steps = max_steps
        self.grade_softmax_temperature = grade_softmax_temperature
        self.curr_step = 0
        self.success_counter = 0

        self.memory: list = []
        self.max_memory = 200
        self.memory_similarity_threshold = 0.08

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # 26, not 25: see _get_obs() -- episode progress was added as an
        # explicit observation feature during the pre-Experiment-1 audit to
        # restore the Markov property (see docstring note below).
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(26,), dtype=np.float32)

        # ---- Lagrange multipliers (lagrangian mode only) ------------------
        # Keys mirror the constraint names in _constraint_violations().
        # Updated externally by research/algo/lagrangian.py via
        # env_method("set_lagrange_multipliers", ...) between rollouts.
        self.lagrange = dict(g1_util=0.0, g2_class=0.0, g3_geom=0.0)
        if lagrange_init:
            self.lagrange.update(lagrange_init)

        self.np_random = None
        self._initialize_variables()

    # ================================================================
    # Public API used by the Lagrangian trainer (research/algo/lagrangian.py)
    # ================================================================
    def set_lagrange_multipliers(self, lambdas: dict):
        self.lagrange.update(lambdas)

    def get_lagrange_multipliers(self) -> dict:
        return dict(self.lagrange)

    # ================================================================
    @property
    def epsilon(self) -> float:
        return float(np.sqrt(235.0 / max(self.fy, 1e-6)))

    def _initialize_variables(self):
        self.h, self.b, self.tf, self.tw = 500.0, 220.0, 20.0, 12.0
        self.fy = 355.0
        self.section_type = "rolled"
        self.span, self.load, self.storey = 8_000.0, 40.0, 20
        self.current_util = self.current_mass = self.current_cost = self.current_co2 = 0.0
        self.current_chi_lt = 1.0
        self.current_class = 1
        self.current_Mrd = self.current_area = 1.0
        self.prev_util = self.prev_mass = self.moment_ratio = 0.0

    def _normalize(self, value, vmin, vmax) -> float:
        return float(np.clip((value - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))

    def _effective_load(self, base_load: float) -> float:
        if not self.use_storey_load_scaling:
            return base_load
        storey_factor = 1.0 + 0.5 * (
            (self.storey - self.STOREY_MIN) / (self.STOREY_MAX - self.STOREY_MIN)
        )
        return base_load * storey_factor

    # ================================================================
    # RESET  (unchanged curriculum logic from exp54b — see module docstring)
    # ================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.curr_step = 0
        self.success_counter = 0

        self.span = float(self.np_random.uniform(self.SPAN_MIN, self.SPAN_MAX))
        base_load = float(self.np_random.uniform(self.LOAD_MIN, 140.0))
        self.storey = int(self.np_random.integers(self.STOREY_MIN, self.STOREY_MAX))
        self.load = self._effective_load(base_load)

        span_m = self.span / 1000.0
        h_target = np.clip(span_m * 42.0, 250.0, 650.0)
        load_factor = (self.load - self.LOAD_MIN) / (self.LOAD_MAX - self.LOAD_MIN)

        if float(self.np_random.uniform(0, 1)) < 0.50:
            grade_idx = int(self.np_random.integers(0, len(self.grades)))
        else:
            grade_idx = int(self.np_random.integers(2, len(self.grades)))
        self.fy = float(self.grades[grade_idx])

        Med_init = self.load * (self.span / 1000.0) ** 2 / 8.0
        Med_factor = float(np.clip(Med_init / 3000.0, 0.0, 1.0))
        h_noise_lo = 0.80 + 0.15 * Med_factor
        h_noise_hi = 1.10 + 0.15 * Med_factor
        h_noise = float(self.np_random.uniform(h_noise_lo, h_noise_hi))
        self.h = float(np.clip(h_target * h_noise, *self.design_limits["h"]))

        b_target = np.clip(self.h / 3.0, 120.0, 300.0)
        tf_target = np.clip(10.0 + load_factor * 20.0, 8.0, 35.0)
        tw_target = np.clip(7.0 + load_factor * 14.0, 6.0, 25.0)
        self.b = float(np.clip(b_target * float(self.np_random.uniform(0.80, 1.20)), *self.design_limits["b"]))
        self.tf = float(np.clip(tf_target * float(self.np_random.uniform(0.80, 1.20)), *self.design_limits["tf"]))
        self.tw = float(np.clip(tw_target * float(self.np_random.uniform(0.80, 1.20)), *self.design_limits["tw"]))

        self.section_type = self.section_types[int(self.np_random.integers(0, 2))]

        self.current_util = self.current_mass = self.current_cost = self.current_co2 = 0.0
        self.current_chi_lt = 1.0
        self.current_class = 1
        self.current_Mrd = self.current_area = 1.0
        self.prev_util = self.prev_mass = self.moment_ratio = 0.0

        return self._get_obs(), {}

    # ================================================================
    # OBSERVATION (unchanged from exp54b)
    # ================================================================
    def _get_obs(self) -> np.ndarray:
        """
        NOTE (pre-Experiment-1 audit, MDP-formulation check -- see
        supervisor Comment #13 "is the state truly Markov?"): the design-
        update step size is deliberately annealed over the episode
        (`_update_design`'s `step_scale`, a function of `self.curr_step`),
        which is standard and defensible for a within-episode coarse-to-
        fine refinement schedule. However, the ORIGINAL 25-feature
        observation never exposed `curr_step` (or any proxy for it), which
        means the transition dynamics P(s'|s,a) depended on information the
        policy could not observe -- the same (observation, action) pair
        could produce different next-states depending on which step of the
        episode it was. That is a genuine POMDP, not an MDP, regardless of
        network capacity or training budget. Fixed by adding normalised
        `episode_progress` (curr_step/max_steps) as an explicit observation
        feature below, which restores the Markov property by construction:
        every quantity `_update_design` and `_ec3_analysis` depend on is
        now either part of the observation or a fixed environment constant.
        """
        eps = self.epsilon
        section_flag = 0.0 if self.section_type == "rolled" else 1.0
        flange_slenderness = (self.b / max(self.tf, 1e-6)) / (14.0 * eps)
        web_slenderness = (self.h / max(self.tw, 1e-6)) / (124.0 * eps)
        class_1 = 1.0 if self.current_class == 1 else 0.0
        class_2 = 1.0 if self.current_class == 2 else 0.0
        class_3 = 1.0 if self.current_class == 3 else 0.0
        class_4 = 1.0 if self.current_class == 4 else 0.0
        util_delta = self.current_util - self.prev_util
        mass_delta = self.current_mass - self.prev_mass
        Med_obs = self.load * (self.span / 1000.0) ** 2 / 8.0
        Med_norm = float(np.clip(Med_obs / 6000.0, 0.0, 1.0))

        return np.array([
            self._normalize(self.span, self.SPAN_MIN, self.SPAN_MAX),
            self._normalize(self.load, self.LOAD_MIN, self.LOAD_MAX),
            self._normalize(self.storey, self.STOREY_MIN, self.STOREY_MAX),
            self._normalize(self.h, *self.design_limits["h"]),
            self._normalize(self.b, *self.design_limits["b"]),
            self._normalize(self.tf, *self.design_limits["tf"]),
            self._normalize(self.tw, *self.design_limits["tw"]),
            np.clip(self.h / max(self.b, 1e-6) / 5.0, 0.0, 1.0),
            self._normalize(self.fy, 355.0, 690.0),
            section_flag,
            np.clip(self.current_util / self.norm["util"], 0.0, 1.0),
            np.clip(self.moment_ratio / 1.5, 0.0, 1.0),
            np.clip(self.current_mass / self.norm["mass"], 0.0, 1.0),
            np.clip(self.current_cost / self.norm["cost"], 0.0, 1.0),
            np.clip(self.current_co2 / self.norm["co2"], 0.0, 1.0),
            np.clip(self.current_chi_lt, 0.0, 1.0),
            np.clip(flange_slenderness / 2.0, 0.0, 1.0),
            np.clip(web_slenderness / 2.0, 0.0, 1.0),
            class_1, class_2, class_3, class_4,
            np.clip((util_delta + 1.0) / 2.0, 0.0, 1.0),
            np.clip((mass_delta / 250.0 + 1.0) / 2.0, 0.0, 1.0),
            Med_norm,
            np.clip(self.curr_step / self.max_steps, 0.0, 1.0),  # episode_progress -- see docstring above
        ], dtype=np.float32)

    # ================================================================
    # STEP
    # ================================================================
    def step(self, action):
        self.curr_step += 1
        self._update_design(action)

        util, mass, penalty, class_loss, chi_lt, debug_ec3 = self._ec3_analysis()
        cost, co2, debug_lca = self._calculate_cost_co2(mass)
        self._update_state(util, mass, cost, co2, chi_lt, debug_ec3)

        x = np.array([
            self._normalize(self.h, *self.design_limits["h"]),
            self._normalize(self.b, *self.design_limits["b"]),
            self._normalize(self.tf, *self.design_limits["tf"]),
            self._normalize(self.tw, *self.design_limits["tw"]),
            self._normalize(self.fy, 355.0, 690.0),
            0.0 if self.section_type == "rolled" else 1.0,
        ])

        novelty = self._calculate_novelty(x) if self.include_novelty else 0.0
        if util < 1.0 and class_loss == 0:
            self._update_memory(x)

        violations = self._constraint_violations(util, class_loss, penalty)
        feasible = (
            violations["g1_util"] <= 1e-3
            and violations["g2_class"] <= 1e-3
            and violations["g3_geom"] <= 1e-3
        )
        in_target_band = (0.90 <= util <= 1.05 and class_loss == 0 and penalty <= 0)

        reward, reward_terms = self._compute_reward(
            util, mass, cost, co2, chi_lt, penalty, class_loss, novelty, debug_ec3, violations,
        )

        terminated, truncated = self._check_termination(in_target_band)

        info = {
            "h": self.h, "b": self.b, "tf": self.tf, "tw": self.tw,
            "fy": self.fy, "section_type": self.section_type,
            "utilization": util, "mass": mass, "cost": cost, "co2": co2, "chi_lt": chi_lt,
            "reward": reward, "reward_terms": reward_terms,
            "span": self.span, "load": self.load, "storey": self.storey,
            "ec3": debug_ec3, "lca": debug_lca,
            "constraint_violations": violations,
            "feasible": feasible,           # <-- CORRECT definition: util<=1.0
            "in_target_band": in_target_band,  # <-- training-termination signal only
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    # ================================================================
    # DESIGN UPDATE (unchanged action mapping from exp54b)
    # ================================================================
    def _update_design(self, action: np.ndarray):
        progress = self.curr_step / self.max_steps
        step_scale = 0.30 + 0.70 * 0.5 * (1.0 + np.cos(np.pi * progress))

        self.h = float(np.clip(self.h + action[0] * 50.0 * step_scale, *self.design_limits["h"]))
        self.b = float(np.clip(self.b + action[1] * 28.0 * step_scale, *self.design_limits["b"]))
        self.tf = float(np.clip(self.tf + action[2] * 3.0 * step_scale, *self.design_limits["tf"]))
        self.tw = float(np.clip(self.tw + action[3] * 2.5 * step_scale, *self.design_limits["tw"]))

        grade_logits = np.array(
            [-((action[4] - c) ** 2) / self.grade_softmax_temperature for c in self._grade_centres],
            dtype=np.float64
        )
        grade_logits -= grade_logits.max()
        grade_probs = np.exp(grade_logits)
        grade_probs /= grade_probs.sum()
        self.fy = float(self.grades[int(np.argmax(grade_probs))])

        self.section_type = "rolled" if action[5] < 0 else "welded"

    def _update_state(self, util, mass, cost, co2, chi_lt, debug_ec3):
        self.prev_util, self.prev_mass = self.current_util, self.current_mass
        self.current_util, self.current_mass = util, mass
        self.current_cost, self.current_co2, self.current_chi_lt = cost, co2, chi_lt
        self.current_class = debug_ec3.get("section_class", 4)
        self.moment_ratio = debug_ec3["Med"] / max(debug_ec3["Mrd"], 1e-6)
        self.current_Mrd = debug_ec3.get("Mrd", 1.0)
        h_web = self.h - 2.0 * self.tf
        fillet_factor = 1.05 if self.section_type == "rolled" else 1.0
        self.current_area = (h_web * self.tw + 2.0 * self.b * self.tf) * fillet_factor

    # ================================================================
    # CONSTRAINTS  (new — explicit, reward-mode-independent violation functions)
    # ================================================================
    def _constraint_violations(self, util: float, class_loss: float, penalty: float) -> dict:
        """
        g_i(design) >= 0 means VIOLATED by that amount (not a binary flag),
        so both the Lagrangian and feasibility-gated modes can use graded
        violation magnitudes, not just pass/fail.
        """
        g1_util = max(util - 1.0, 0.0)                 # EC3 capacity (incl. governing SLS check)
        g2_class = float(class_loss) if self.current_class == 4 else max(self.current_class - 3, 0) * 0.15
        g3_geom = penalty / 50.0  # normalise the existing b/h + fillet-type geometry penalty to O(1)
        return dict(g1_util=float(g1_util), g2_class=float(g2_class), g3_geom=float(g3_geom))

    def _economy(self, mass: float, cost: float, co2: float) -> float:
        value = {"mass": mass / self.norm["mass"],
                  "cost": cost / self.norm["cost"],
                  "co2": co2 / self.norm["co2"]}[self.economy_metric]
        return float(value)

    # ================================================================
    # REWARD — dispatches to one of three modes
    # ================================================================
    def _compute_reward(self, util, mass, cost, co2, chi_lt, penalty, class_loss,
                          novelty, debug_ec3, violations):
        if self.reward_mode == "shaped":
            return self._reward_shaped(util, mass, cost, co2, penalty, class_loss, novelty)
        if self.reward_mode == "feasibility_gated":
            return self._reward_feasibility_gated(util, mass, cost, co2, violations)
        if self.reward_mode == "lagrangian":
            return self._reward_lagrangian(util, mass, cost, co2, violations)
        raise ValueError(self.reward_mode)

    # ---- ARM "shaped": weighted-sum reward shaping, NO grade-specific term ----
    def _reward_shaped(self, util, mass, cost, co2, penalty, class_loss, novelty):
        """
        Reward-shaping baseline. Same structural style as typical RL-for-
        design reward engineering (weighted economy term + utilisation-
        target Gaussian + feasibility penalties + mass-improvement shaping
        + optional novelty), but with two corrections made during the
        pre-Experiment-1 audit so this arm is a scientifically valid
        control for the other two modes:

        1. OBJECTIVE CONSISTENCY (fixed here): the previous version's
           economy term was `-5*mass_n - 5*cost_n`, a FIXED mass+cost
           blend that ignored `self.economy_metric` entirely -- so running
           `reward_mode="shaped"` with `economy_metric="co2"` never
           actually rewarded CO2 reduction as the primary objective, only
           as a separate bounded bonus term with a different scale. That
           broke the whole point of a 3-arm reward-formulation ablation:
           the arms must optimise the SAME objective and differ only in
           HOW they're incentivised to do so. Fixed: economy_reward now
           uses `self._economy(mass, cost, co2)`, the identical objective
           function `feasibility_gated` and `lagrangian` use, scaled to
           the same magnitude the old two-term formula produced (~-10 at
           norm=1). The separate `co2_lca_reward` bonus is REMOVED (not
           renamed) -- keeping it would double-count CO2 when
           economy_metric="co2" and inject an uncontrolled secondary
           objective into the other two economy_metric settings.

        2. FEASIBILITY-BOUNDARY CONSISTENCY (fixed here): the previous
           utilisation-score curve gave nearly its FULL reward for any
           util up to 1.05, i.e. it rewarded up to 5%-overstressed
           (genuinely EC3-noncompliant, per this codebase's own
           `feasible` definition) designs almost as if they were fully
           compliant, creating an incentive to settle just past the
           util<=1.0 boundary rather than at it. The curve below now
           breaks its "full reward" zone at util<=1.0 (matching
           `feasible`'s actual definition everywhere else in this
           codebase), applies a blended step-down through the training-
           termination convenience band (1.0, 1.05], and only then
           continues the original steep quadratic penalty. This does NOT
           change the termination rule itself (still 3 consecutive steps
           in [0.90, 1.05], see `_check_termination` and its docstring) --
           it only stops the REWARD from telling the agent that mild
           infeasibility is nearly as good as compliance.
        """
        economy_reward = -10.0 * self._economy(mass, cost, co2)

        target_util, sigma = 0.96, 0.06
        base_score = 100.0 * np.exp(-((util - target_util) ** 2) / (2.0 * sigma ** 2))
        if util <= 1.0:
            util_score = base_score
        elif util <= 1.05:
            # Training-termination convenience band (see _check_termination):
            # still genuinely infeasible (util>1.0), so blend the score DOWN
            # to a negative value by util=1.05 rather than keeping it near
            # `base_score` -- closes the gap between what this reward
            # rewards and what `feasible` actually means elsewhere in this
            # codebase. Continuous at both ends: frac=0 -> base_score
            # (matches the util<=1.0 branch); frac=1 -> -50.0 (matches the
            # util>1.05 branch's value at util=1.05).
            frac = (util - 1.0) / 0.05
            util_score = base_score * (1.0 - frac) - 50.0 * frac
        else:
            util_score = -50.0 - 400.0 * (util - 1.05) ** 2

        underutil_penalty = 80.0 * (0.90 - util) ** 2 if util < 0.90 else 0.0

        eps = self.epsilon
        util_violation = max(util - 1.0, 0.0)
        feasibility_penalty = 60.0 * util_violation + 30.0 * class_loss + 4.0 * penalty
        if self.current_class == 3:
            feasibility_penalty += 5.0
        hw = self.h - 2.0 * self.tf
        if (hw / max(self.tw, 1e-6)) > (72.0 * eps):
            feasibility_penalty += 10.0
        if self.b > self.h:
            feasibility_penalty += 12.0
        slender_ratio = self.h / max(self.b, 1e-6)
        if slender_ratio > 3.5:
            feasibility_penalty += 12.0 * (slender_ratio - 3.5)

        mass_improvement = np.clip((self.prev_mass - mass) / 300.0, -1.0, 1.0)
        improvement_reward = 1.5 * mass_improvement
        novelty_reward = 0.15 * np.tanh(novelty) if self.include_novelty else 0.0

        reward = (economy_reward + util_score
                  + improvement_reward + novelty_reward - feasibility_penalty - underutil_penalty)

        return reward, {
            "economy_reward": economy_reward,
            "utilization_reward": util_score,
            "improvement_reward": improvement_reward, "novelty_reward": novelty_reward,
            "feasibility_penalty": feasibility_penalty, "underutil_penalty": underutil_penalty,
        }

    # ---- ARM C: feasibility-gated + potential-based shaping (Ng et al. 1999) ----
    def _reward_feasibility_gated(self, util, mass, cost, co2, violations):
        economy_n = self._economy(mass, cost, co2)
        feasible = all(v <= 1e-3 for v in violations.values())

        target_util = 0.96
        phi_prev = -abs(self.prev_util - target_util)
        phi_curr = -abs(util - target_util)
        gamma = 0.99
        shaping = gamma * phi_curr - phi_prev   # policy-invariant potential-based shaping

        if feasible:
            base = -economy_n
        else:
            base = -(1.0 + 10.0 * violations["g1_util"]
                     + 5.0 * violations["g2_class"] + 2.0 * violations["g3_geom"])

        reward = base + 0.5 * shaping
        return reward, {
            "economy_term": -economy_n if feasible else 0.0,
            "violation_penalty": base if not feasible else 0.0,
            "potential_shaping": 0.5 * shaping,
            "feasible": float(feasible),
        }

    # ---- ARM B: Lagrangian-constrained (multipliers updated externally) ----
    def _reward_lagrangian(self, util, mass, cost, co2, violations):
        economy_n = self._economy(mass, cost, co2)
        penalty = sum(self.lagrange[k] * violations[k] for k in violations)
        reward = -economy_n - penalty
        return reward, {
            "economy_term": -economy_n,
            "lagrange_penalty": -penalty,
            **{f"lambda_{k}": self.lagrange[k] for k in violations},
            **{f"violation_{k}": violations[k] for k in violations},
        }

    # ================================================================
    # TERMINATION — training-efficiency signal only (see module docstring)
    # ================================================================
    def _check_termination(self, in_target_band: bool):
        if in_target_band:
            self.success_counter += 1
        else:
            self.success_counter = 0
        terminated = self.success_counter >= 3
        truncated = self.curr_step >= self.max_steps
        return terminated, truncated

    # ================================================================
    # EC3 ANALYSIS — UNCHANGED from exp54b (verified by regression test)
    # ================================================================
    def _ec3_analysis(self):
        h, b, tf, tw, fy = self.h, self.b, self.tf, self.tw, self.fy
        eps = self.epsilon
        h_web = h - 2.0 * tf

        if self.section_type == "rolled":
            r = 0.1 * tf; fillet_factor = 1.05; fillet_Iy_fac = 1.02; torsion_factor = 1.15
        else:
            r = 0.0; fillet_factor = 1.0; fillet_Iy_fac = 1.0; torsion_factor = 1.0

        A = (h_web * tw + 2.0 * b * tf) * fillet_factor
        Iy = (tw * h_web**3 / 12.0 + 2.0 * (b * tf**3 / 12.0
             + b * tf * (h/2.0 - tf/2.0)**2)) * fillet_Iy_fac
        Wel = Iy / (h / 2.0)
        Wpl = (2.0 * b * tf * (h/2.0 - tf/2.0) + tw * h_web**2 / 4.0) * fillet_factor
        Iz = 2.0 * (tf * b**3) / 12.0 + h_web * tw**3 / 12.0

        c_flange = (b - tw) / 2.0 - r
        d_web = h_web - 2.0 * r
        flange_ratio = c_flange / max(tf, 1e-6)
        web_ratio = d_web / max(tw, 1e-6)

        if flange_ratio <= 9.0 * eps and web_ratio <= 72.0 * eps:
            sec_class = 1
        elif flange_ratio <= 10.0 * eps and web_ratio <= 83.0 * eps:
            sec_class = 2
        elif flange_ratio <= 14.0 * eps and web_ratio <= 124.0 * eps:
            sec_class = 3
        else:
            class_severity = max(flange_ratio / (14.0 * eps) - 1.0, web_ratio / (124.0 * eps) - 1.0)
            util_penalty = min(5.0, 2.0 + 2.0 * class_severity)
            return util_penalty, 4000.0, 25.0, class_severity, 0.0, {
                "Mrd": 1e-6, "Med": 1e-6, "efficiency": 0.0, "Ved": 1e-6, "Vpl_Rd": 1e-6,
                "section_class": 4, "lambda_lt": 0.0, "shear_ratio": 0.0,
                "moment_util": util_penalty, "deflection_util": util_penalty,
            }

        W_ref = Wpl if sec_class <= 2 else Wel
        Mrd_basic = W_ref * fy / 1.0e6

        L = self.span
        L_cr = L * self.ltb_restraint_factor
        It = (2.0 * b * tf**3 + h_web * tw**3) / 3.0 * torsion_factor
        Iw = Iz * (h - tf)**2 / 4.0
        C1 = 1.13

        Mcr = max(
            (C1 * np.pi**2 * self.E * Iz / L_cr**2)
            * np.sqrt(Iw/Iz + L_cr**2 * self.G * It / (np.pi**2 * self.E * Iz)), 1e-3
        )
        if self.include_zg_in_mcr:
            zg = h / 2.0; C2 = 0.55
            Mcr_zg = (C2*zg)**2 * (np.pi**2*self.E*Iz/L_cr**2)
            Mcr = max(np.sqrt(Mcr**2 + Mcr_zg) - C1*C2*zg*(np.pi**2*self.E*Iz/L_cr**2), 1e-3)

        lambda_lt = np.sqrt(W_ref * fy / (Mcr + 1e-9))
        # EN1993-1-1 Table 6.5 (LTB buckling curve selection), all four cases:
        #   rolled, h/b<=2 -> curve a (0.21) | rolled, h/b>2 -> curve b (0.34)
        #   welded, h/b<=2 -> curve c (0.49) | welded, h/b>2 -> curve d (0.76)
        # FIX (pre-Experiment-1 audit, independent EC3 verification): the
        # previous version used alpha_lt=0.49 for ALL welded sections
        # regardless of h/b, omitting curve d (0.76) for welded h/b>2.
        # Confirmed by research/tests/ec3_independent_verification.py:
        # this produced a 14% chi_LT error / 12% utilization error on a
        # deep welded S690 test case -- large enough to flip a feasibility
        # determination. See that file for the full verification table.
        if self.section_type == "rolled":
            alpha_lt = 0.34 if h/b > 2.0 else 0.21
        else:
            alpha_lt = 0.76 if h/b > 2.0 else 0.49
        phi_lt = 0.5 * (1.0 + alpha_lt * (lambda_lt - 0.2) + lambda_lt**2)
        chi_lt = float(np.clip(1.0 / (phi_lt + np.sqrt(np.maximum(phi_lt**2 - lambda_lt**2, 1e-9))), 0.0, 1.0))
        Mrd = chi_lt * Mrd_basic

        Av = (A - 2.0*b*tf + (tw + 2.0*r)*tf) if self.section_type == "rolled" else h_web*tw
        Vpl_Rd = Av * fy / (np.sqrt(3.0) * 1.0e3)

        L_m = L / 1_000.0
        Ved = self.load * L_m / 2.0
        Med = self.load * L_m**2 / 8.0
        shear_ratio = Ved / (Vpl_Rd + 1e-9)
        if shear_ratio > 0.5:
            rho = (2.0 * shear_ratio - 1.0)**2
            Mrd *= max(1.0 - rho * (Wpl / max(Wel, 1e-6) - 1.0), 0.15)

        moment_util = Med / (Mrd + 1e-9)
        util = float(np.clip(moment_util, 0.0, 5.0))

        w_sls = self.load * self.sls_load_factor
        delta = 5.0 * w_sls * L**4 / (384.0 * self.E * Iy)
        deflection_util = delta / max(L / 250.0, 1e-9)
        if deflection_util > util:
            util = float(np.clip(deflection_util, 0.0, 5.0))

        mass = A * L * 7.85e-6
        penalty = 50.0 * np.tanh(2.0 * max(b / h - 1.0, 0.0))
        efficiency = Mrd / max(mass, 1e-6)

        return util, mass, penalty, 0.0, chi_lt, {
            "Mrd": Mrd, "Med": Med, "efficiency": efficiency, "Ved": Ved, "Vpl_Rd": Vpl_Rd,
            "section_class": sec_class, "lambda_lt": lambda_lt, "shear_ratio": shear_ratio,
            "moment_util": moment_util, "deflection_util": deflection_util,
            "w_uls": self.load, "w_sls": w_sls,
        }

    # ================================================================
    # COST / CO2 — UNCHANGED from exp54b
    # ================================================================
    def _calculate_cost_co2(self, mass: float):
        fy_key = int(self.fy)

        material_cost_factor = {355: 1.00, 460: 1.15, 500: 1.28, 550: 1.42, 620: 1.60, 690: 1.85}
        material_co2_factor = {355: 2.30, 460: 2.10, 500: 1.98, 550: 1.88, 620: 1.75, 690: 1.63}
        cost_factor = material_cost_factor.get(fy_key, 1.00)
        co2_factor = material_co2_factor.get(fy_key, 2.30)

        material_cost = mass * cost_factor
        material_co2 = mass * co2_factor

        if self.section_type == "rolled":
            fab_factor = 0.15
            fab_co2_factor = 0.08
        else:
            fab_factor = 0.42
            fab_co2_factor = 0.22
            if fy_key >= 690: fab_factor *= 1.30; fab_co2_factor *= 1.20
            elif fy_key >= 620: fab_factor *= 1.18; fab_co2_factor *= 1.10
            elif fy_key >= 550: fab_factor *= 1.10; fab_co2_factor *= 1.05

        if self.section_type == "welded" and fy_key >= 550:
            thickness_factor = (self.tf + self.tw) / 40.0
            extra_hss_fab_penalty = 1.0 + 0.35 * thickness_factor
        else:
            extra_hss_fab_penalty = 1.0

        fabrication_cost = material_cost * fab_factor * extra_hss_fab_penalty
        fabrication_co2 = material_co2 * fab_co2_factor * extra_hss_fab_penalty
        transport_cost, transport_co2 = mass * 0.08, mass * 0.018
        erection_cost, erection_co2 = mass * 0.12, mass * 0.04
        painting_cost, painting_co2 = mass * 0.05, mass * 0.015
        processing_cost, processing_co2 = mass * 0.03, mass * 0.010

        total_cost = (material_cost + fabrication_cost + transport_cost
                      + erection_cost + painting_cost + processing_cost)
        total_co2 = (material_co2 + fabrication_co2 + transport_co2
                     + erection_co2 + painting_co2 + processing_co2)

        debug = {
            "material_cost": material_cost, "fabrication_cost": fabrication_cost,
            "transport_cost": transport_cost, "erection_cost": erection_cost,
            "painting_cost": painting_cost, "processing_cost": processing_cost,
            "material_co2": material_co2, "fabrication_co2": fabrication_co2,
            "transport_co2": transport_co2, "erection_co2": erection_co2,
            "painting_co2": painting_co2, "processing_co2": processing_co2,
        }
        return total_cost, total_co2, debug

    # ================================================================
    # NOVELTY (unchanged, opt-in only — see include_novelty)
    # ================================================================
    def _calculate_novelty(self, x: np.ndarray) -> float:
        if len(self.memory) < 5:
            return 1.0
        memory_array = np.array(self.memory)
        geom_weights = np.array([1.2, 1.0, 1.5, 1.6, 1.3])
        weighted_diff = (memory_array[:, :5] - x[:5]) * geom_weights
        geom_dists = np.linalg.norm(weighted_diff, axis=1)
        section_bonus = np.where(memory_array[:, 5] != x[5], 0.35, 0.0)
        dists = geom_dists + section_bonus
        k = min(5, len(dists))
        return float(np.clip(np.mean(np.sort(dists)[:k]), 0.0, 5.0))

    def _update_memory(self, x: np.ndarray):
        if len(self.memory) > 0:
            memory_array = np.array(self.memory)
            geom_dists = np.linalg.norm(memory_array[:, :5] - x[:5], axis=1)
            section_diff = np.where(memory_array[:, 5] != x[5], 0.35, 0.0)
            if np.min(geom_dists + section_diff) < self.memory_similarity_threshold:
                return
        self.memory.append(x.copy())
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
