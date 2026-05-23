"""
================================================================
high_rise_generative_env.py
----------------------------------------------------------------
Research-Grade EC3 Reinforcement Learning Environment
HKU — Generative Design of HSS Optimal Beam Using RL

CHANGELOG vs original:
  [CRITICAL FIX 1]  Removed invalid 0.92 factor on Class 3 Mrd.
                    EC3 already uses Wel for Class 3; no further
                    reduction is permitted (§6.2.5).

  [CRITICAL FIX 2]  Cast fy to int before dict lookup in
                    _calculate_cost_co2() to avoid silent key-miss
                    when fy is a numpy float (e.g. 460.0 ≠ key 460).

  [FIX 3]           Web classification now uses the correct clear
                    depth d = h − 2(tf + r) for rolled sections
                    (EC3 Table 5.2) instead of h_web = h − 2tf.

  [FIX 4]           Web shear-buckling feasibility limit corrected
                    to 72ε (without transverse stiffeners) per
                    EC3 §6.2.6. The previous 150ε had no code basis.

  [FIX 5]           Deflection now applies a serviceability load
                    factor ψ = 0.50 so the SLS check is not driven
                    by the ULS design load (EC3 §7.2).

  [FIX 6]           Early episode termination on class_loss removed.
                    Class 4 sections now receive a large penalty but
                    do NOT terminate the episode, preventing the
                    degenerate ultra-short episodes that harm PPO
                    during early training. Hard termination is kept
                    only for extreme overload (util > 1.20).

  [FIX 7]           storey is now functionally used to modulate the
                    tributary load (higher storeys → broader tributary
                    width ≈ longer effective span share). If this
                    physical coupling is unwanted it can be switched
                    off via use_storey_load_scaling = False.

  [FIX 8]           Underutilization penalty made smooth (no
                    discontinuous if-gate at util = 0.80). Replaced
                    with: 15 · max(0.92 − util, 0)^1.5.

  [FIX 9]           Compactness penalty re-scaled so it is
                    meaningful relative to economy/utilization terms.
                    Now uses normalized plate thicknesses.

  [FIX 10]          success_counter threshold reduced from 3 to 2
                    consecutive steps, making early convergence more
                    reachable without sacrificing quality.

  [FIX 11]          epsilon refactored into a @property to eliminate
                    redundant computation in _get_obs and _ec3_analysis.

  [FIX 12]          mass_delta normalization divisor reduced from
                    1000 → 250 for better per-step resolution.

  [FIX 13]          Grade rounding replaced with symmetric bin-edge
                    approach so all six grades receive equal
                    probability mass from the continuous action.

  [FIX 14]          Step-scale minimum raised from 0.20 → 0.30 and
                    switched to a cosine schedule so late-episode
                    corrections remain feasible.

  [FIX 15]          np_random.choice usage corrected to be compatible
                    with gymnasium's numpy.random.Generator API.

  [FIX 16]          Mcr load-height effect documented; zg term added
                    as optional (conservative omission noted in paper).

  [FIX 17]          fillet contribution to Iy now approximated
                    separately with a reduced fillet_Iy_factor (0.02)
                    rather than the same 1.05 area factor.

  [DEBUG FIX A]     _effective_load() (storey scaling) removed from
                    _ec3_analysis(). self.load is the ULS design load
                    already set by the user / reset(); applying a storey
                    factor inside analysis was double-counting and caused
                    Med to be 21% too high, making every section appear
                    overloaded. _effective_load() is now called only
                    inside reset() to set self.load for that episode.

  [DEBUG FIX B]     ltb_restraint_factor added (default 0.25). Composite
                    floor beams with metal decking are laterally restrained
                    at approximately quarter-points (~every 3 m on a 12 m
                    span), giving Lcr ≈ 0.25 × L. Using the full span
                    gave lambda_lt > 2.0 and chi_lt ≈ 0.16 for a 600 mm
                    welded S460 beam at 12 m (util = 2.72 in debug).
                    With Lcr = 0.25 × L the same section gives
                    chi_lt = 0.698 and util = 0.963 — correctly optimal.

  [DEBUG FIX C]     Grade bin mapping corrected from searchsorted (left)
                    to searchsorted(..., side='right') - 1 so the mapping
                    is left-inclusive. Previously action[4]=0.0 snapped
                    to grade index 3 (S550) instead of staying at S460,
                    meaning a "no-op" action unexpectedly changed grade.

AUTHOR: [Your Name]
AFFILIATION: HKU
================================================================
"""


import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HighRiseGenerativeEnv(gym.Env):
    """
    Generative RL environment for EC3-compliant HSS beam design.
    Optimises section geometry (h, b, tf, tw), steel grade (fy),
    and fabrication type (rolled / welded) under bending, shear,
    LTB, and deflection constraints.

    Observation  : 24-dimensional normalised state vector
    Action       : 6-dimensional continuous Box ([-1, 1]^6)
    Reward       : multi-objective (economy, utilisation, LTB,
                   novelty, feasibility penalties)
    """

    # ============================================================
    # INITIALIZATION
    # ============================================================
    def __init__(
        self,
        use_storey_load_scaling: bool = True,
        include_zg_in_mcr: bool = False,
        sls_load_factor: float = 0.50,
        ltb_restraint_factor: float = 0.40,
    ):
        """
        Parameters
        ----------
        use_storey_load_scaling : bool
            If True, storey count scales the effective tributary load
            when self.load is sampled at reset(). The ULS load stored
            in self.load is already the final design value; this flag
            only affects how reset() populates self.load.
        include_zg_in_mcr : bool
            If True, adds the destabilising load-height term to Mcr
            (top-flange loading). Conservative to omit (default).
        sls_load_factor : float
            ψ factor applied to design load for SLS deflection check.
            Typical value 0.50 per EC3 NA.
        ltb_restraint_factor : float
            Effective LTB buckling length as a fraction of span:
            Lcr = ltb_restraint_factor × span. Default 0.25 represents
            a composite floor beam with metal decking providing lateral
            restraint at approximately quarter-points (~every 3 m on a
            12 m span). Use 0.50 for restraint at mid-span only, or
            1.0 for a fully unrestrained beam.
        """
        super().__init__()

        self.use_storey_load_scaling = use_storey_load_scaling
        self.include_zg_in_mcr      = include_zg_in_mcr
        self.sls_load_factor         = sls_load_factor
        self.ltb_restraint_factor    = ltb_restraint_factor

        # =====================================================
        # MATERIAL CONSTANTS  [N/mm², MPa]
        # =====================================================
        self.E = 210_000.0   # Young's modulus
        self.G = 81_000.0    # Shear modulus

        # =====================================================
        # EC3 STEEL GRADES  [MPa]
        # =====================================================
        self.grades = np.array(
            [355, 460, 500, 550, 620, 690],
            dtype=np.float32,
        )

        # [DEBUG FIX C] Grade centres for nearest-centre action mapping.
        # Each centre is the midpoint of its equal-width bin in [-1, 1].
        # Centres: [-0.833, -0.500, -0.167, +0.167, +0.500, +0.833]
        n_grades = len(self.grades)
        self._grade_centres = np.array(
            [-1.0 + (2 * k + 1) / n_grades for k in range(n_grades)],
            dtype=np.float32,
        )

        # =====================================================
        # SECTION TYPES
        # =====================================================
        self.section_types = ["rolled", "welded"]

        # =====================================================
        # GENERATIVE DESIGN LIMITS  [mm]
        # =====================================================
        self.design_limits = {
            "h":  (250.0, 750.0),
            "b":  (120.0, 300.0),
            "tf": (8.0,   35.0),
            "tw": (6.0,   25.0),
        }

        # =====================================================
        # GLOBAL DEMANDS
        # =====================================================
        self.SPAN_MIN   = 6_000.0    # mm
        self.SPAN_MAX   = 15_000.0   # mm
        self.LOAD_MIN   = 20.0       # kN/m
        self.LOAD_MAX   = 100.0      # kN/m  [VAL FIX I] 140→100
        # Validation: load>100kN/m was 78% infeasible within
        # current design limits (h≤750mm). Capping at 100 keeps
        # the demand space achievable for the agent to learn from.
        self.STOREY_MIN = 1
        self.STOREY_MAX = 70

        # =====================================================
        # NORMALISATION CONSTANTS
        # =====================================================
        self.norm = {
            "mass": 4_000.0,
            "cost": 8_000.0,
            "co2":  10_000.0,
            "util": 1.5,
        }

        # =====================================================
        # RL TARGETS
        # =====================================================
        self.target_util = 0.95

        # =====================================================
        # PPO CONTROL
        # =====================================================
        self.max_steps = 40   # [EXP7 FIX 1] 20→40: 100% of underutil episodes
        # truncated at step 20 — agent had no time to recover from bad inits
        self.curr_step = 0
        self.success_counter = 0

        # =====================================================
        # NOVELTY MEMORY
        # =====================================================
        self.memory: list = []
        self.max_memory = 200
        self.memory_similarity_threshold = 0.08

        # =====================================================
        # ACTION SPACE  — [Δh, Δb, Δtf, Δtw, fy_bin, section]
        # =====================================================
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # =====================================================
        # OBSERVATION SPACE  (24-dim, all in [0, 1])
        # =====================================================
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(24,), dtype=np.float32
        )

        # =====================================================
        # INTERNAL STATE
        # =====================================================
        self.np_random = None
        self._initialize_variables()

    # ============================================================
    # EPSILON PROPERTY  [FIX 11]
    # Eliminates redundant computation across _get_obs / _ec3_analysis
    # ============================================================
    @property
    def epsilon(self) -> float:
        """EC3 material factor ε = √(235 / fy)."""
        return float(np.sqrt(235.0 / max(self.fy, 1e-6)))

    # ============================================================
    # VARIABLE INITIALIZATION
    # ============================================================
    def _initialize_variables(self):

        # Design variables
        self.h  = 500.0
        self.b  = 220.0
        self.tf = 20.0
        self.tw = 12.0
        self.fy = 355.0
        self.section_type = "rolled"

        # Demand variables
        self.span   = 8_000.0
        self.load   = 40.0
        self.storey = 20

        # Response variables
        self.current_util    = 0.0
        self.current_mass    = 0.0
        self.current_cost    = 0.0
        self.current_co2     = 0.0
        self.current_chi_lt  = 1.0
        self.current_class   = 1

        # History
        self.prev_util = 0.0
        self.prev_mass = 0.0

        # Debug
        self.moment_ratio = 0.0

    # ============================================================
    # NORMALISATION HELPER
    # ============================================================
    def _normalize(self, value: float, vmin: float, vmax: float) -> float:
        return float(np.clip((value - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))

    # ============================================================
   
    # ============================================================
    def _effective_load(self, base_load: float) -> float:
        if not self.use_storey_load_scaling:
            return base_load
        storey_factor = 1.0 + 0.5 * (
            (self.storey - self.STOREY_MIN)
            / (self.STOREY_MAX - self.STOREY_MIN)
        )
        return base_load * storey_factor

    # ============================================================
    # RESET
    # ============================================================
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.curr_step       = 0
        self.success_counter = 0

        # Random demands
        self.span = float(
            self.np_random.uniform(self.SPAN_MIN, self.SPAN_MAX)
        )
        base_load = float(
            self.np_random.uniform(self.LOAD_MIN, self.LOAD_MAX)
        )
        self.storey = int(
            self.np_random.integers(self.STOREY_MIN, self.STOREY_MAX)
        )
        # [DEBUG FIX A] Storey scaling applied here at episode start only.
        # self.load is the final ULS design load for the entire episode.
        self.load = self._effective_load(base_load)

        
        span_m = self.span / 1000.0
        h_target = np.clip(span_m * 42.0, 250.0, 650.0)  # span/18 in mm
        h_noise  = float(self.np_random.uniform(0.85, 1.15))
        self.h   = float(np.clip(h_target * h_noise, *self.design_limits["h"]))

        # b proportional to h, tf/tw proportional to load intensity
        load_factor = (self.load - self.LOAD_MIN) / (self.LOAD_MAX - self.LOAD_MIN)
        b_target = np.clip(self.h / 3.0, 120.0, 300.0)
        self.b   = float(np.clip(
            b_target * float(self.np_random.uniform(0.80, 1.20)),
            *self.design_limits["b"]
        ))
        tf_target = np.clip(10.0 + load_factor * 20.0, 8.0, 35.0)
        self.tf  = float(np.clip(
            tf_target * float(self.np_random.uniform(0.80, 1.20)),
            *self.design_limits["tf"]
        ))
        tw_target = np.clip(7.0 + load_factor * 14.0, 6.0, 25.0)
        self.tw  = float(np.clip(
            tw_target * float(self.np_random.uniform(0.80, 1.20)),
            *self.design_limits["tw"]
        ))

        # [FIX 15] Correct gymnasium Generator API usage
        # [EXP7 FIX 2b] Grade curriculum: 40% of episodes force a random
        # grade from the full set to prevent S500/S550 collapse.
        # 60% of episodes let the agent choose via reset sampling.
        if float(self.np_random.uniform(0, 1)) < 0.40:
            grade_idx = int(self.np_random.integers(0, len(self.grades)))
        else:
            # Bias toward mid-to-high grades to encourage HSS exploration
            grade_idx = int(self.np_random.integers(2, len(self.grades)))  # S500–S690
        self.fy = float(self.grades[grade_idx])

        sec_idx           = int(self.np_random.integers(0, 2))
        self.section_type = self.section_types[sec_idx]

        # Reset state
        self.current_util   = 0.0
        self.current_mass   = 0.0
        self.current_cost   = 0.0
        self.current_co2    = 0.0
        self.current_chi_lt = 1.0
        self.current_class  = 1
        self.prev_util      = 0.0
        self.prev_mass      = 0.0
        self.moment_ratio   = 0.0

        return self._get_obs(), {}

    # ============================================================
    # OBSERVATION
    # ============================================================
    def _get_obs(self) -> np.ndarray:

        eps = self.epsilon  # [FIX 11] — use property

        section_flag = 0.0 if self.section_type == "rolled" else 1.0

        # Slenderness ratios (normalised against Class 3 limits)
        flange_slenderness = (
            (self.b / max(self.tf, 1e-6)) / (14.0 * eps)
        )
        web_slenderness = (
            (self.h / max(self.tw, 1e-6)) / (124.0 * eps)
        )

        class_1 = 1.0 if self.current_class == 1 else 0.0
        class_2 = 1.0 if self.current_class == 2 else 0.0
        class_3 = 1.0 if self.current_class == 3 else 0.0
        class_4 = 1.0 if self.current_class == 4 else 0.0

        util_delta = self.current_util - self.prev_util
        mass_delta = self.current_mass - self.prev_mass

        obs = np.array([
            # --- Demands ---
            self._normalize(self.span,   self.SPAN_MIN,   self.SPAN_MAX),
            self._normalize(self.load,   self.LOAD_MIN,   self.LOAD_MAX),
            self._normalize(self.storey, self.STOREY_MIN, self.STOREY_MAX),

            # --- Geometry ---
            self._normalize(self.h,  *self.design_limits["h"]),
            self._normalize(self.b,  *self.design_limits["b"]),
            self._normalize(self.tf, *self.design_limits["tf"]),
            self._normalize(self.tw, *self.design_limits["tw"]),
            np.clip(self.h / max(self.b, 1e-6) / 5.0, 0.0, 1.0),

            # --- Material ---
            self._normalize(self.fy, 355.0, 690.0),
            section_flag,

            # --- Response ---
            np.clip(self.current_util   / self.norm["util"], 0.0, 1.0),
            np.clip(self.moment_ratio   / 1.5,               0.0, 1.0),
            np.clip(self.current_mass   / self.norm["mass"], 0.0, 1.0),
            np.clip(self.current_cost   / self.norm["cost"], 0.0, 1.0),
            np.clip(self.current_co2    / self.norm["co2"],  0.0, 1.0),
            np.clip(self.current_chi_lt,                     0.0, 1.0),

            # --- Slenderness ---
            np.clip(flange_slenderness / 2.0, 0.0, 1.0),
            np.clip(web_slenderness    / 2.0, 0.0, 1.0),

            # --- Section class (one-hot) ---
            class_1, class_2, class_3, class_4,

            # --- History ---
            np.clip((util_delta + 1.0) / 2.0,            0.0, 1.0),
            # [FIX 12] divisor reduced 1000 → 250 for better resolution
            np.clip((mass_delta / 250.0 + 1.0) / 2.0,    0.0, 1.0),

        ], dtype=np.float32)

        return obs

    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):

        self.curr_step += 1

        self._update_design(action)

        util, mass, penalty, class_loss, chi_lt, debug_ec3 = (
            self._ec3_analysis()
        )

        cost, co2, debug_lca = self._calculate_cost_co2(mass)

        self._update_state(util, mass, cost, co2, chi_lt, debug_ec3)

        x = np.array([
            self._normalize(self.h,  *self.design_limits["h"]),
            self._normalize(self.b,  *self.design_limits["b"]),
            self._normalize(self.tf, *self.design_limits["tf"]),
            self._normalize(self.tw, *self.design_limits["tw"]),
            self._normalize(self.fy, 355.0, 690.0),
            0.0 if self.section_type == "rolled" else 1.0,
        ])

        novelty = self._calculate_novelty(x)

        reward, reward_terms = self._compute_reward(
            util, mass, cost, co2, chi_lt,
            penalty, class_loss, novelty,
        )

        if util < 1.0 and class_loss == 0:
            self._update_memory(x)

        terminated, truncated = self._check_termination(
            util, class_loss, penalty
        )

        info = {
            "h": self.h, "b": self.b, "tf": self.tf, "tw": self.tw,
            "fy": self.fy, "section_type": self.section_type,
            "utilization": util, "mass": mass,
            "cost": cost, "co2": co2, "chi_lt": chi_lt,
            "reward": reward, "reward_terms": reward_terms,
            "span": self.span, "load": self.load,
            "storey": self.storey,
            "ec3": debug_ec3, "lca": debug_lca,
        }

        return (
            self._get_obs(),
            float(reward),
            # float(np.clip(reward, -100.0, 100.0)),  # [EXP5 FIX 7] clip [-100,100]
            terminated,
            truncated,
            info,
        )

    # ============================================================
    # DESIGN UPDATE
    # ============================================================
    def _update_design(self, action: np.ndarray):

        progress = self.curr_step / self.max_steps

        # [FIX 14] Cosine schedule; minimum raised to 0.30
        step_scale = 0.30 + 0.70 * 0.5 * (
            1.0 + np.cos(np.pi * progress)
        )

        h_step  = 50.0
        b_step  = 28.0
        tf_step = 3.0
        tw_step = 2.5

        self.h  = float(np.clip(
            self.h  + action[0] * h_step  * step_scale,
            *self.design_limits["h"],
        ))
        self.b  = float(np.clip(
            self.b  + action[1] * b_step  * step_scale,
            *self.design_limits["b"],
        ))
        self.tf = float(np.clip(
            self.tf + action[2] * tf_step * step_scale,
            *self.design_limits["tf"],
        ))
        self.tw = float(np.clip(
            self.tw + action[3] * tw_step * step_scale,
            *self.design_limits["tw"],
        ))

        grade_idx = int(np.argmin(np.abs(self._grade_centres - action[4])))
        self.fy   = float(self.grades[grade_idx])

        self.section_type = "rolled" if action[5] < 0 else "welded"

    # ============================================================
    # UPDATE STATE
    # ============================================================
    def _update_state(
        self, util, mass, cost, co2, chi_lt, debug_ec3
    ):
        self.prev_util = self.current_util
        self.prev_mass = self.current_mass

        self.current_util   = util
        self.current_mass   = mass
        self.current_cost   = cost
        self.current_co2    = co2
        self.current_chi_lt = chi_lt
        self.current_class  = debug_ec3.get("section_class", 4)
        self.moment_ratio   = (
            debug_ec3["Med"] / max(debug_ec3["Mrd"], 1e-6)
        )

    # ============================================================
    # REWARD
    # ============================================================
    def _compute_reward(
        self,
        util, mass, cost, co2,
        chi_lt, penalty, class_loss, novelty,
    ):
        mass_n = mass / self.norm["mass"]
        cost_n = cost / self.norm["cost"]
        co2_n  = co2  / self.norm["co2"]

        economy_reward = (
            -7.5 * mass_n
            -1.8 * cost_n
            -0.9 * co2_n
        )

        # --- Utilisation  [EXP5 FIX 1 + EXP8 FIX 2 + EXP11 FIX 3] ---
        # [EXP11 FIX 3] Adjusted to match new gradual underutil structure.
        # Ramp 0→60 for stronger push toward 0.90. Peak +50 at 0.95 target.
        # ==========================================================
        # UTILIZATION REWARD
        # ==========================================================

        if util <= 1.0:

            # broad learning signal
            # stronger pressure toward high utilization
            ramp_reward = 85.0 * (util ** 1.35)
            # ramp_reward = 70.0 * util

            # target-zone bonus
            target_bonus = 0.0
            
            if 0.95 <= util <= 1.02:
                target_bonus = 125.0

            elif 0.88 <= util < 0.95:
                target_bonus = 60.0

            elif 0.82 <= util < 0.88:
                target_bonus = 25.0
            # mild Gaussian shaping near optimum
            peak_bonus = 40.0 * np.exp(
                -((util - 0.97) ** 2) / (2 * 0.04 ** 2)
            )

            utilization_reward = (
                ramp_reward
                + target_bonus
                + peak_bonus
            )

        else:
            utilization_reward = -250.0 * (util - 1.0) ** 1.5
        
       
        # ==========================================================
        # UNDERUTILIZATION PENALTY
        # Penalize oversized / inefficient beams
        # ==========================================================

        if util < 0.80:
            underutil_penalty = 25.0 * (0.80 - util) ** 1.2
        else:
            underutil_penalty = 0.0

        # --- Feasibility penalties ---
        eps = self.epsilon  # [FIX 11]

        util_violation    = max(util - 1.0, 0.0)
        feasibility_penalty  = 10.0 * util_violation
        feasibility_penalty += 5.0  * class_loss    # Class 4 penalty
        feasibility_penalty += 4.0  * penalty
        # [EXP7 FIX 4] Soft Class 3 signal — Class 3 uses Wel not Wpl,
        # reducing moment capacity by ~15%. Gentle penalty encourages
        # the agent toward compact (Class 1/2) sections without hard termination.
        if hasattr(self, 'current_class') and self.current_class == 3:
            feasibility_penalty += 1.0

        # [FIX 4] Web shear-buckling limit: 72ε (no transverse stiffeners)
        hw = self.h - 2.0 * self.tf
        if (hw / max(self.tw, 1e-6)) > (72.0 * eps):
            feasibility_penalty += 10.0

        if self.b > self.h:
            feasibility_penalty += 12.0

        slender_ratio = self.h / max(self.b, 1e-6)
        if slender_ratio > 3.5:
            feasibility_penalty += 12.0 * (slender_ratio - 3.5)

        # --- LTB stability  [VAL FIX E + EXP10 FIX 5] ---
        # [EXP10 FIX 5] Increased from 1.5 to 2.0 to encourage good LTB as
        # secondary objective when primary (util) is satisfied.
        stability_reward = 2.0 * chi_lt

        # --- Mass improvement ---
        mass_improvement = np.clip(
            (self.prev_mass - mass) / 300.0, -1.0, 1.0
        )
        improvement_reward = 1.5 * mass_improvement  # [VAL FIX F] 0.5→1.5

        compactness_penalty = 0.0
        

        # --- Novelty ---
        novelty_reward = 0.15 * np.tanh(novelty)

        # --- Total ---
        reward = (
            economy_reward
            + utilization_reward
            + stability_reward
            + improvement_reward
            + novelty_reward
            - feasibility_penalty
            - underutil_penalty
            - compactness_penalty
        )

        reward_terms = {
            "economy_reward":      economy_reward,
            "utilization_reward":  utilization_reward,
            "stability_reward":    stability_reward,
            "improvement_reward":  improvement_reward,
            "novelty_reward":      novelty_reward,
            "feasibility_penalty": feasibility_penalty,
            "underutil_penalty":   underutil_penalty,
            "compactness_penalty": compactness_penalty,
        }

        return reward, reward_terms

    # ============================================================
    # TERMINATION  [FIX 6 + FIX 10]
    # ============================================================
    def _check_termination(self, util, class_loss, penalty):

        terminated = False
        truncated  = False

        success_zone = (
            0.90 <= util <= 1.05
            and class_loss == 0
            and penalty <= 0
        )

        if success_zone:
            self.success_counter += 1
        else:
            self.success_counter = 0

        # [FIX 10] Threshold reduced 3 → 2
        if self.success_counter >= 2:
            #Disable early success termination
            terminated = False
            
        if self.curr_step >= self.max_steps:
            truncated = True
        
        return terminated, truncated
    # ============================================================
    def _calculate_cost_co2(self, mass: float):

        # [FIX 2] Cast fy to int to guarantee dict key match
        fy_key = int(self.fy)

        material_cost_factor = {
                    355: 1.00,
                    460: 1.15,
                    500: 1.28,
                    550: 1.42,
                    620: 1.60,
                    690: 1.85,
                }
        # CO2 factors (kg CO2-eq / kg steel):
        # Higher-strength steel has lower embodied CO2 per kg
        # (fewer kg needed per unit resistance).
        material_co2_factor = {
            355: 2.30, 460: 2.10, 500: 1.98,
            550: 1.88, 620: 1.75, 690: 1.63,
        }

        cost_factor = material_cost_factor.get(fy_key, 1.00)
        co2_factor  = material_co2_factor.get(fy_key, 2.30)

        material_cost = mass * cost_factor
        material_co2  = mass * co2_factor

        # Fabrication
        # ==========================================================
        # FABRICATION COST / CO2
        # ==========================================================

        # Base fabrication factors
        if self.section_type == "rolled":
            fab_factor     = 0.15
            fab_co2_factor = 0.08
        else:
            fab_factor     = 0.42
            fab_co2_factor = 0.22

            # additional HSS welding complexity
            if fy_key >= 690:
                fab_factor *= 1.30
                fab_co2_factor *= 1.20

            elif fy_key >= 620:
                fab_factor *= 1.18
                fab_co2_factor *= 1.10

            elif fy_key >= 550:
                fab_factor *= 1.10
                fab_co2_factor *= 1.05
        # HSS premium (heat treatment, QA/QC, procurement)
        if   fy_key >= 690: hss_premium = 1.85
        elif fy_key >= 620: hss_premium = 1.60
        elif fy_key >= 550: hss_premium = 1.42
        elif fy_key >= 500: hss_premium = 1.28
        elif fy_key >= 460: hss_premium = 1.15
        else:               hss_premium = 1.00

        # ==========================================================
        # ADDITIONAL FABRICATION COMPLEXITY
        # Thick HSS welded sections are expensive to fabricate:
        # - preheating
        # - controlled welding
        # - inspection
        # - distortion control
        # ==========================================================

        thickness_factor = (self.tf + self.tw) / 40.0

        extra_hss_fab_penalty = 1.0

        if self.section_type == "welded" and fy_key >= 550:
            extra_hss_fab_penalty += 0.35 * thickness_factor

        fabrication_cost = (
            material_cost
            * fab_factor
            * hss_premium
            * extra_hss_fab_penalty
        )

        fabrication_co2 = (
            material_co2
            * fab_co2_factor
            * hss_premium
            * extra_hss_fab_penalty
        )
        # Other life-cycle stages
        # Transport: 150 km assumed; 0.12 kg CO2 / tonne·km → 0.018 kg/kg
        transport_cost = mass * 0.08
        transport_co2  = mass * 0.018

        erection_cost  = mass * 0.12
        erection_co2   = mass * 0.04

        painting_cost  = mass * 0.05
        painting_co2   = mass * 0.015

        processing_cost = mass * 0.03
        processing_co2  = mass * 0.010

        total_cost = (
            material_cost + fabrication_cost
            + transport_cost + erection_cost
            + painting_cost  + processing_cost
        )
        total_co2 = (
            material_co2 + fabrication_co2
            + transport_co2 + erection_co2
            + painting_co2  + processing_co2
        )

        debug = {
            "material_cost":    material_cost,
            "fabrication_cost": fabrication_cost,
            "transport_cost":   transport_cost,
            "erection_cost":    erection_cost,
            "painting_cost":    painting_cost,
            "processing_cost":  processing_cost,
            "material_co2":     material_co2,
            "fabrication_co2":  fabrication_co2,
            "transport_co2":    transport_co2,
            "erection_co2":     erection_co2,
            "painting_co2":     painting_co2,
            "processing_co2":   processing_co2,
            "hss_premium":      hss_premium,
        }

        return total_cost, total_co2, debug

    # ============================================================
    # EC3 ANALYSIS
    # ============================================================
    def _ec3_analysis(self):
        """
        Performs EC3 cross-section and member resistance checks:
          1. Section classification (Table 5.2)
          2. Bending resistance (§6.2.5)
          3. Lateral-torsional buckling (§6.3.2)
          4. Shear resistance (§6.2.6) with shear-bending interaction
          5. Deflection (§7.2) at SLS

        Returns
        -------
        util       : float  — governing utilisation ratio (clipped 0–5)
        mass       : float  — beam mass in kg
        penalty    : float  — geometric penalty (b > h etc.)
        class_loss : float  — 1.0 if Class 4, else 0.0
        chi_lt     : float  — LTB reduction factor
        debug      : dict   — intermediate results for logging
        """

        h  = self.h
        b  = self.b
        tf = self.tf
        tw = self.tw
        fy = self.fy

        eps = self.epsilon  # [FIX 11]

        h_web = h - 2.0 * tf   # clear web height (Hw)

        # ---------------------------------------------------------
        # SECTION TYPE PARAMETERS
        # ---------------------------------------------------------
        if self.section_type == "rolled":
            r               = 0.1 * tf      # root fillet radius
            fillet_factor   = 1.05          # area correction
            # [FIX 17] Separate Iy fillet correction (smaller)
            fillet_Iy_fac   = 1.02
            torsion_factor  = 1.15
        else:
            r               = 0.0
            fillet_factor   = 1.0
            fillet_Iy_fac   = 1.0
            torsion_factor  = 1.0

        # ---------------------------------------------------------
        # GROSS CROSS-SECTION AREA  [mm²]
        # ---------------------------------------------------------
        A = (h_web * tw + 2.0 * b * tf) * fillet_factor

        # ---------------------------------------------------------
        # SECOND MOMENT OF AREA ABOUT MAJOR AXIS  [mm⁴]
        # ---------------------------------------------------------
        Iy = (
            tw * h_web ** 3 / 12.0
            + 2.0 * (
                b * tf ** 3 / 12.0
                + b * tf * (h / 2.0 - tf / 2.0) ** 2
            )
        ) * fillet_Iy_fac   # [FIX 17]

        # ---------------------------------------------------------
        # ELASTIC + PLASTIC SECTION MODULI  [mm³]
        # ---------------------------------------------------------
        Wel = Iy / (h / 2.0)

        Wpl = (
            2.0 * b * tf * (h / 2.0 - tf / 2.0)
            + tw * h_web ** 2 / 4.0
        ) * fillet_factor

        # ---------------------------------------------------------
        # SECOND MOMENT ABOUT MINOR AXIS  [mm⁴]
        # ---------------------------------------------------------
        Iz = 2.0 * (tf * b ** 3) / 12.0 + h_web * tw ** 3 / 12.0

        # ---------------------------------------------------------
        # SECTION CLASSIFICATION  (EC3 Table 5.2)
        # [FIX 3] Use correct clear projection c and depth d
        # ---------------------------------------------------------
        # Flange outstand (half-width of compression flange):
        #   c = (b − tw)/2 − r   (EC3 Table 5.2, sheet 2)
        c_flange = (b - tw) / 2.0 - r

        # Web clear depth between fillets (EC3 Table 5.2, sheet 1):
        #   d = hw − 2r   for rolled,  d = hw   for welded
        d_web = h_web - 2.0 * r  # equals h_web when r=0 (welded)

        flange_ratio = c_flange / max(tf, 1e-6)
        web_ratio    = d_web    / max(tw, 1e-6)

        class_loss = 0.0

        if   flange_ratio <= 9.0  * eps and web_ratio <= 72.0  * eps:
            sec_class = 1
        elif flange_ratio <= 10.0 * eps and web_ratio <= 83.0  * eps:
            sec_class = 2
        elif flange_ratio <= 14.0 * eps and web_ratio <= 124.0 * eps:
            sec_class = 3
        else:
            sec_class  = 4
            class_loss = 1.0
            # [FIX 6] Return high but finite values; do NOT terminate
            return 4.0, 4_000.0, 15.0, class_loss, 0.0, {
                "Mrd": 1e-6, "Med": 1e-6,
                "Ved": 1e-6, "Vpl_Rd": 1e-6,
                "section_class": 4,
                "lambda_lt": 0.0, "shear_ratio": 0.0,
                "moment_util": 4.0, "deflection_util": 4.0,
            }

        # ---------------------------------------------------------
        # BENDING RESISTANCE  (§6.2.5)
        # ---------------------------------------------------------
        gamma_M0 = 1.0
        W_ref    = Wpl if sec_class <= 2 else Wel  # Class 3 uses Wel

        # [CRITICAL FIX 1] No 0.92 factor — EC3 Class 3 uses Wel,
        # no additional multiplier is applied.
        Mrd_basic = W_ref * fy / (gamma_M0 * 1.0e6)   # kNm

        # ---------------------------------------------------------
        # LATERAL-TORSIONAL BUCKLING  (§6.3.2)
        # ---------------------------------------------------------
        L    = self.span   # mm  — full span (for Med / Ved / deflection)

        # [DEBUG FIX B] Effective LTB buckling length
        # Lcr < L when decking or secondary beams provide restraint.
        L_cr = L * self.ltb_restraint_factor

        # Saint-Venant torsion constant
        It = (2.0 * b * tf ** 3 + h_web * tw ** 3) / 3.0
        It *= torsion_factor

        # Warping constant
        Iw = Iz * (h - tf) ** 2 / 4.0

        # Equivalent uniform moment factor (UDL, SS beam)
        C1 = 1.13

        # Elastic critical moment — uses Lcr (restrained buckling length)
        Mcr_base = (
            (C1 * np.pi ** 2 * self.E * Iz) / L_cr ** 2
        ) * np.sqrt(
            Iw / Iz
            + L_cr ** 2 * self.G * It / (np.pi ** 2 * self.E * Iz)
        )

        if self.include_zg_in_mcr:
            # Destabilising load height: zg = +h/2 (load at top flange)
            # Simplified Wagner term addition per NCCI SN003
            zg = h / 2.0
            C2 = 0.55   # for UDL
            Mcr_zg_term = (
                (C2 * zg) ** 2
                * (np.pi ** 2 * self.E * Iz) / L_cr ** 2
            )
            Mcr = np.sqrt(Mcr_base ** 2 + Mcr_zg_term) - (
                (C1 * C2 * zg * np.pi ** 2 * self.E * Iz) / L_cr ** 2
            )
            Mcr = max(Mcr, 1e-3)
        else:
            Mcr = max(Mcr_base, 1e-3)

        # Non-dimensional slenderness
        lambda_lt = np.sqrt(W_ref * fy / (Mcr + 1e-9))

        # Imperfection factor (EC3 Table 6.3 + 6.5)
        if self.section_type == "rolled":
            alpha_lt = 0.34 if h / b > 2.0 else 0.21   # curve b or a
        else:
            alpha_lt = 0.49                              # curve c

        phi_lt = 0.5 * (
            1.0 + alpha_lt * (lambda_lt - 0.2) + lambda_lt ** 2
        )

        chi_lt = 1.0 / (
            phi_lt + np.sqrt(
                np.maximum(phi_lt ** 2 - lambda_lt ** 2, 1e-9)
            )
        )
        chi_lt = float(np.clip(chi_lt, 0.0, 1.0))

        Mrd = chi_lt * Mrd_basic   # kNm (LTB-reduced)

        # ---------------------------------------------------------
        # SHEAR RESISTANCE  (§6.2.6)
        # ---------------------------------------------------------
        if self.section_type == "rolled":
            Av = A - 2.0 * b * tf + (tw + 2.0 * r) * tf   # EC3 (6.18)
        else:
            Av = h_web * tw   # welded: web only

        Vpl_Rd = Av * fy / (np.sqrt(3.0) * gamma_M0 * 1.0e3)  # kN

        # [DEBUG FIX A] Use self.load directly — already the ULS design
        # load set at reset(). Do NOT call _effective_load() here.
        w_uls = self.load          # kN/m
        L_m   = L / 1_000.0       # mm → m

        Ved = w_uls * L_m / 2.0        # kN
        Med = w_uls * L_m ** 2 / 8.0   # kNm

        shear_ratio = Ved / (Vpl_Rd + 1e-9)

        # Shear–bending interaction (§6.2.8)
        if shear_ratio > 0.5:
            rho    = (2.0 * shear_ratio - 1.0) ** 2
            Mrd   *= max(1.0 - rho * (Wpl / max(Wel, 1e-6) - 1.0), 0.15)

        # Bending utilisation
        moment_util = Med / (Mrd + 1e-9)
        util        = float(np.clip(moment_util, 0.0, 5.0))

        # ---------------------------------------------------------
        # DEFLECTION  (§7.2)  [FIX 5]
        # SLS load = ψ × ULS load
        # ---------------------------------------------------------
        w_sls        = w_uls * self.sls_load_factor   # kN/m
        delta        = 5.0 * w_sls * L ** 4 / (384.0 * self.E * Iy)  # mm
        delta_limit  = L / 250.0                      # mm

        deflection_util = delta / max(delta_limit, 1e-9)

        if deflection_util > util:
            util = float(np.clip(deflection_util, 0.0, 5.0))

        # ---------------------------------------------------------
        # BEAM MASS  [kg]
        # ---------------------------------------------------------
        mass = A * L * 7.85e-6   # mm² × mm × kg/mm³

        # ---------------------------------------------------------
        # GEOMETRIC PENALTY (b > h is physically unreasonable)
        # ---------------------------------------------------------
        penalty = 50.0 * np.tanh(2.0 * max(b / h - 1.0, 0.0))

        return (
            util,
            mass,
            penalty,
            class_loss,
            chi_lt,
            {
                "Mrd":             Mrd,
                "Med":             Med,
                "Ved":             Ved,
                "Vpl_Rd":          Vpl_Rd,
                "section_class":   sec_class,
                "lambda_lt":       lambda_lt,
                "shear_ratio":     shear_ratio,
                "moment_util":     moment_util,
                "deflection_util": deflection_util,
                "w_uls":           w_uls,
                "w_sls":           w_sls,
            },
        )

    # ============================================================
    # NOVELTY
    # ============================================================
    def _calculate_novelty(self, x: np.ndarray) -> float:

        if len(self.memory) < 5:
            return 1.0

        memory_array = np.array(self.memory)

        geom_weights = np.array([1.2, 1.0, 1.5, 1.6, 1.3])

        weighted_diff = (memory_array[:, :5] - x[:5]) * geom_weights
        geom_dists    = np.linalg.norm(weighted_diff, axis=1)

        section_bonus = np.where(
            memory_array[:, 5] != x[5], 0.35, 0.0
        )
        dists = geom_dists + section_bonus

        k      = min(5, len(dists))
        novelty = float(np.mean(np.sort(dists)[:k]))

        return float(np.clip(novelty, 0.0, 5.0))

    # ============================================================
    # MEMORY UPDATE
    # ============================================================
    def _update_memory(self, x: np.ndarray):

        if len(self.memory) > 0:
            memory_array = np.array(self.memory)
            geom_dists   = np.linalg.norm(
                memory_array[:, :5] - x[:5], axis=1
            )
            section_diff = np.where(
                memory_array[:, 5] != x[5], 0.35, 0.0
            )
            if np.min(geom_dists + section_diff) < (
                self.memory_similarity_threshold
            ):
                return

        self.memory.append(x.copy())

        if len(self.memory) > self.max_memory:
            self.memory.pop(0)