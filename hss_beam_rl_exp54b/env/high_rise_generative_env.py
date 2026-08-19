"""
================================================================
high_rise_generative_env.py
----------------------------------------------------------------
Research-Grade EC3 Reinforcement Learning Environment
HKU — Generative Design of High-Strength Steel (HSS) Beams via PPO

FINAL / REFERENCE MODEL (exp54, trained 1M timesteps → "exp54b")
----------------------------------------------------------------
This is the environment used to produce the project's final reported
results. It is architecturally identical to the exp54 checkpoint
(96.5% feasibility at 500k timesteps); trained for 1M timesteps
instead of 500k ("exp54b"), it reached 97.0% feasibility, 2/64
infeasible grid cells, and the best HSS-grade diversity of any
checkpoint produced during this project (S620+S690 = 40.2% of
feasible episodes).

WHY THIS VERSION, AND NOT ANY OF THE LATER exp57-62 VARIANTS:
    A parallel line of experiments (exp57-62) attempted to fix a
    specific residual issue — the trained policy occasionally touches
    a feasible design in the most extreme demand corner (span 11-15m,
    load >=100 kN/m) without holding it for 3 consecutive steps,
    preventing early termination there — via reward shaping (exp57),
    action-level hysteresis on grade selection (exp58, exp60),
    action-level damping on geometry (exp59), enlarged observations
    with EC3-fitted flange/web-thickness targets (exp61), and a
    relaxed termination rule (exp62). Each fix suppressed the
    targeted symptom but the underlying oscillation relocated to a
    different action channel each time (grade -> geometry -> grade),
    and two of the six attempts (exp57, exp62) caused outright
    regressions in overall feasibility when validated broadly rather
    than only against the two hardest scenarios.

    The decisive control experiment: training this exact, unmodified
    exp54 environment for the same 1M-timestep budget as exp61
    ("exp54b") matched or exceeded every one of exp61's five stacked
    changes on every headline metric (feasibility, grid coverage,
    grade diversity) while being far simpler to describe and defend.
    exp54b still exhibits the same residual oscillation in the
    extreme demand corner as every other checkpoint tested, including
    the ones built specifically to fix it — direct evidence the
    oscillation is a PPO advantage-estimate noise property of that
    underrepresented region of the demand space, not a defect in this
    environment's design that further environment engineering was
    going to resolve. This is disclosed as a known limitation rather
    than treated as unsolved; it does not affect the feasibility or
    quality of the best design found in that region (both diagnostic
    tooling and grid validation confirm genuinely feasible designs
    are reached there), only the training-time efficiency of
    terminating early once one is found.

    See docs/EXPERIMENT_LOG.md for the full exp46-exp62 history.

----------------------------------------------------------------

CHANGELOG (exp46 → exp47):
----------------------------------------------------------------

  SCIENTIFIC OBJECTIVE (exp47):
    Enforce a physically correct, demand-driven grade transition:
    S355/S460 for light loading → S500/S550 for medium → S620/S690
    for heavy demand (long spans, high UDL). The agent must learn
    that HSS is economically and structurally justified only when
    conventional steel would require excessive mass or depth.

  [EXP47 FIX 1]  grade_efficiency_reward replaced with
                 demand_normalised_grade_reward.

    PROBLEM (exp46):
      grade_efficiency_reward = 25 × (fy/690) × clip(1 − mass/3500).
      S690 mean mass = 2636 kg → mass_factor = 0.247.
      S550 mean mass = 1825 kg → mass_factor = 0.479.
      S690 earned HALF the grade reward of S550, explicitly
      penalising the agent for the correct engineering choice
      at high demand where S690 beams ARE heavy in absolute terms.

    FIX:
      Reward strength-utilisation efficiency = fy × util / (A_cm²).
      A_cm² = cross-sectional area in cm², representing material used
      per unit length. Higher fy → smaller required area → reward is
      proportional to structural efficiency, not absolute mass.
      This is demand-neutral: a small S690 section and a large S355
      section at the same utilisation both earn proportional reward.
      Peak calibrated so S690 at util=0.96 earns +28 vs S355 +16.

  [EXP47 FIX 2]  hss_mass_saving_bonus: explicit counterfactual reward.

    PROBLEM:
      The agent had no signal comparing S690 cost/mass against what
      a lower grade would require for the same demand. At 13m/140kN·m,
      S355 would need ~6200 kg to hit util=0.95; S690 needs ~3300 kg.
      Without seeing this 2900 kg saving, the agent couldn't learn
      that the S690 cost premium is structurally justified.

    FIX:
      Estimate the mass a notional S355 section would need for the
      current Med using the same geometry (area scaled by fy/355).
      reward = k × clip((mass_S355_equiv − mass_actual) / 1000, 0, 3).
      Fires only when util ∈ [0.88, 1.08] and fy ≥ 500.
      k scales with fy/690 so S690 earns maximum bonus.
      At low demand (short spans), mass_S355_equiv ≈ mass_actual
      (small sections for both grades), so the bonus is near-zero —
      no spurious incentive to upgrade grade when demand is light.
      At high demand the saving can be 1500–3000 kg → bonus 6–18 pts.

  [EXP47 FIX 3]  Economy reward decoupled: cost vs CO₂ treated
                 physically correctly.

    PROBLEM:
      exp46: economy_reward = -5·mass_n - 5·cost_n - 4·co2_n.
      S690 CO₂ factor = 1.63 kg/kg vs S355 = 2.30 kg/kg.
      But because economy_reward penalised CO₂ proportionally to mass,
      a 1600 kg S690 beam (CO₂ = 2608 kg) was penalised more than a
      2100 kg S460 beam (CO₂ = 4410 kg), which is physically incorrect.
      EC3 and LCA literature consistently show HSS has lower lifecycle
      CO₂ per unit of load-carrying capacity.

    FIX:
      Split economy into cost penalty (always) and CO₂ bonus (mild,
      demand-conditioned). CO₂ per unit resistance = co2 / Mrd.
      Lower CO₂/Mrd → better LCA efficiency → small positive signal.
      Calibrated so the CO₂ bonus (max ~4 pts/step) never overrides
      the utilisation signal, but accumulates over 40 steps to ~160 pts,
      meaningful at the episode level in the paper's LCA analysis.
      Cost penalty unchanged: HSS cost premium is a real-world constraint
      that the agent must balance against mass saving.

  [EXP47 FIX 4]  underutil_penalty threshold raised 0.90 → 0.88 and
                 shape changed to quadratic (was 1.5-power).

    PROBLEM:
      At 6m/20kN·m the grid showed util = 0.022, underutil penalty ≈ 35 pts.
      Despite this, the agent held the same geometry because it was
      optimised for 10–13m spans. The penalty was large but not shape-
      corrected: a 1.5-power curve has a very flat gradient near util=0.
      The agent got the same large penalty whether util was 0.02 or 0.75,
      giving no gradient to improve from badly under-utilised states.

    FIX:
      Smooth quadratic from util=0 to util=0.88, max penalty 45 at util=0.
      Quadratic gives strong gradient at all under-utilised states,
      guiding the agent to resize the section rather than hold still.
      Threshold lowered to 0.88 to give the agent a slightly wider
      feasible band while still strongly penalising severe underuse.

  [EXP47 FIX 5]  success_counter threshold: 3 → 2 (restored from exp46
                 comment-out). With the new grade rewards taking ~5 steps
                 to stabilise, requiring 3 consecutive successes at
                 util ∈ [0.90, 1.05] was rarely achievable in 40 steps
                 with the new reward landscape. 2 consecutive successes
                 is sufficient for paper-quality convergence evidence.

  [EXP47 NOTE]   All EC3 structural mechanics, section classification,
                 LTB, shear, deflection, and cost/CO₂ calculations are
                 unchanged from exp46. Only _compute_reward() is modified.

  PROBLEM STATEMENT (exp51-52):
    Feasibility rate collapsed from 97.5% (exp48) to 47.5%/49.5%.
    Diagnostic showed policy outputting one fixed geometry
    (h=404 in exp51, h=284 in exp52) for every span and load.
    Both experiments introduced multiple simultaneous changes
    (stratified sampling + wide noise + larger action steps +
    uniform grade curriculum) that interacted to destabilise
    training.

  STRATEGY (exp53):
    Roll back to the stable exp48 training configuration.
    Keep only the definitively correct fixes (LOAD_MAX=210,
    Med_norm obs, incremental grade, success_counter=3).
    Add one new, targeted improvement: grade-aligned h_noise.

  [EXP53 FIX 1]  Reverted: stratified sampling removed.
                 Replaced with uniform span/load sampling,
                 identical to exp48. Stratified sampling caused
                 bimodal policy collapse by creating episodes
                 with very different reward scales that PPO's
                 value function could not reconcile.
                 The 60% HSS grade curriculum bias already
                 oversamples high-grade episodes sufficiently.

  [EXP53 FIX 2]  Reverted: h_noise back to (0.85, 1.15).
                 Wide noise (0.65, 1.35) caused too many episodes
                 to start far from feasibility. At 13m/140kN·m
                 with h_noise=0.65, starting h=354mm requires
                 ~20 steps of full-scale upward movement to reach
                 the required ~720mm — barely achievable in 40
                 steps, giving the policy no room to explore grade
                 or flange geometry. Reverting to (0.85, 1.15)
                 keeps starting h within ~15% of the target, which
                 exp48 demonstrated is sufficient for convergence.

  [EXP53 FIX 3]  Reverted: action step sizes back to exp48 values.
                 exp52 enlarged h_step 50→80, b_step 28→40,
                 tf_step 3→5, tw_step 2.5→4. Larger steps combined
                 with wider noise created oscillation rather than
                 convergence. Reverting to 50/28/3/2.5.

  [EXP53 FIX 4]  Reverted: h_target formula back to span_m * 42.
                 exp52 changed this to span_m*35 + 150*load_factor.
                 Both formulas produce similar values (within 5%)
                 but the original is simpler and was validated in
                 exp48. No reason to change a formula that worked.

  [EXP53 FIX 5]  Reverted: 60% HSS grade curriculum bias restored.
                 exp52 changed to fully uniform grade sampling.
                 The curriculum bias (40% uniform, 60% S500-S690)
                 is necessary to provide sufficient gradient signal
                 for HSS grades during training. Without it the
                 agent defaults to S355/S460 in early training and
                 never learns HSS-appropriate geometry.

  [EXP53 FIX 6]  Reverted: underutil penalty back to uncapped form.
                 exp52 used min(80*(0.90-util)^2, 80).
                 The cap was added to prevent 1345 pts/step at
                 util=5.0, but with h_noise=(0.85,1.15) the policy
                 rarely produces util=5.0 at episode start. The
                 uncapped quadratic provides a cleaner gradient.
                 Restored: 80*(0.90-util)^2 without cap.

  [EXP53 NEW 1]  Grade-aligned h_noise: when fy >= 620 at reset,
                 h_noise uses (0.95, 1.20) instead of (0.85, 1.15).
                 S620/S690 are needed at high demand which requires
                 large sections. Starting S620/S690 episodes with
                 h ≥ 0.95 × h_target ensures the policy begins
                 close to feasibility for the demand level that
                 justifies those grades. Does not require stratified
                 sampling — works within the uniform distribution.

  KEPT FROM EXP51/52 (correct fixes, not reverted):
    LOAD_MAX = 210       — fixes real observation clipping bug
    obs shape (25,)      — Med_norm as obs[24]
    Incremental grade    — replaces destructive argmin overwrite
    success_counter >= 3 — sufficient for convergence evidence

  EC3, LTB, cost, CO₂, reward terms: unchanged from exp50.


CHANGELOG (exp53 → exp54):
----------------------------------------------------------------

  PROBLEM STATEMENT (exp53):
    Feasibility rate: 44.0%. Policy collapsed to one fixed geometry
    (h=488, b=159, tf=13, tw=10, fy=690) for every scenario.
    Grid shows 6m-11m at loads below 80 kN/m all infeasible
    (util=0.014-0.24). The policy learned "S690 + fixed large
    geometry" for mid-range spans and accepted heavy underutil
    penalties elsewhere.

  ROOT CAUSE IDENTIFIED:
    The incremental grade action (exp51) coupled grade to geometry.
    With 60% HSS curriculum the policy initialises at S620/S690
    frequently. Grade-aligned h_noise (exp53) then starts those
    episodes with large h. Incremental grade means grade persists
    across steps -- the policy converged to "stay at S690 + large h"
    because that geometry works for 11-15m spans (30% of training).
    Incremental grade created a grade-geometry coupling that
    produced a stable but demand-insensitive local optimum.
    Exp48 (97.5% feasibility) used destructive argmin grade action
    -- grade and geometry were independent each step, preventing
    this coupling.

  STRATEGY (exp54):
    Replace incremental grade with smooth softmax-snapped action.
    Replace grade-aligned h_noise with demand-aligned h_noise.
    Reduce HSS curriculum bias from 60% to 50%.
    Keep all other exp53 settings unchanged.

  [EXP54 FIX 1]  Softmax-snapped grade action replaces incremental.

    PROBLEM (incremental, exp51-53):
      Grade persists between steps. With HSS curriculum + grade-
      aligned noise, policy converged to "start S690, hold it."
      Multi-step transition cost deterred grade exploration, creating
      a stable local optimum at S690 regardless of demand.

    FIX:
      Softmax over grade centres with temperature 0.15.
      action[4] in [-1, 1] drives a smooth probability distribution
      over grades. At action=-1 selects S355 with near-certainty;
      at action=+1 selects S690; at action=0 selects S500.
      Adjacent grades have similar probability for similar action
      values -- no wild jumps. Grade re-selected each step,
      decoupling it from geometry. Restores exp48 behaviour
      with smoother PPO gradient signal than plain argmin.

  [EXP54 FIX 2]  Demand-aligned h_noise replaces grade-aligned.

    PROBLEM (grade-aligned, exp53):
      Coupling h_noise to fy (fy>=620 -> larger h) reinforced the
      grade-geometry local optimum: high grade -> large h -> works
      at high demand -> policy never learned to vary geometry with
      demand independently of grade.

    FIX:
      h_noise scales with Med = w*L^2/8 at reset span/load.
      Med_factor = clip(Med/3000, 0, 1):
        0 at Med=0 kNm, 1 at Med>=3000 kNm (13m/140 -> 0.986)
      h_noise_lo = 0.80 + 0.15*Med_factor  (0.80 light -> 0.95 heavy)
      h_noise_hi = 1.10 + 0.15*Med_factor  (1.10 light -> 1.25 heavy)
      Grade and h_noise fully decoupled at initialisation.
      Light demand starts smaller; heavy demand starts larger.

  [EXP54 FIX 3]  HSS curriculum bias reduced: 60% -> 50%.

    With softmax grade action that re-selects freely each step, the
    60% HSS bias gave S690 ~21.7% initialisation probability vs
    S355 ~6.7%, contributing to the S690 dominance in exp53.
    50%/50% gives more balanced grade coverage at initialisation.

  KEPT UNCHANGED FROM EXP53:
    LOAD_MAX = 210, obs shape (25,), Med_norm as obs[24]
    Uniform span/load sampling (U[6,15m] x U[20,140])
    h_target = span_m * 42, action steps 50/28/3/2.5
    success_counter >= 3, underutil 80*(0.90-util)^2
    All EC3, LTB, cost, CO2, reward terms unchanged

AUTHOR: Muhammad Shifa
AFFILIATION: HKU
================================================================
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class HighRiseGenerativeEnv(gym.Env):

    def __init__(
        self,
        use_storey_load_scaling: bool = True,
        include_zg_in_mcr: bool = False,
        sls_load_factor: float = 0.50,
        ltb_restraint_factor: float = 0.40,
    ):
        super().__init__()

        self.use_storey_load_scaling = use_storey_load_scaling
        self.include_zg_in_mcr      = include_zg_in_mcr
        self.sls_load_factor        = sls_load_factor
        self.ltb_restraint_factor   = ltb_restraint_factor

        self.E = 210_000.0
        self.G = 81_000.0

        self.grades = np.array([355, 460, 500, 550, 620, 690], dtype=np.float32)
        self.grade_index_map = {
            355: 0, 460: 1, 500: 2,
            550: 3, 620: 4, 690: 5,
        }

        n_grades = len(self.grades)
        self._grade_centres = np.array(
            [-1.0 + (2 * k + 1) / n_grades for k in range(n_grades)],
            dtype=np.float32,
        )

        self.section_types = ["rolled", "welded"]

        self.design_limits = {
            "h":  (250.0, 750.0),
            "b":  (120.0, 300.0),
            "tf": (8.0,   35.0),
            "tw": (6.0,   25.0),
        }

        self.SPAN_MIN   = 6_000.0
        self.SPAN_MAX   = 15_000.0
        self.LOAD_MIN   = 20.0
        self.LOAD_MAX   = 210.0   # [EXP51] corrected: storey scaling → max 210
        self.STOREY_MIN = 1
        self.STOREY_MAX = 70

        self.norm = {
            "mass": 4_000.0,
            "cost": 8_000.0,
            "co2":  10_000.0,
            "util": 1.5,
        }

        self.target_util     = 0.95
        self.max_steps       = 40
        self.curr_step       = 0
        self.success_counter = 0

        self.memory: list = []
        self.max_memory = 200
        self.memory_similarity_threshold = 0.08

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(25,), dtype=np.float32  # [EXP51] 24→25
        )

        self.np_random = None
        self._initialize_variables()

    @property
    def epsilon(self) -> float:
        return float(np.sqrt(235.0 / max(self.fy, 1e-6)))

    def _initialize_variables(self):
        self.h  = 500.0
        self.b  = 220.0
        self.tf = 20.0
        self.tw = 12.0
        self.fy = 355.0
        self.section_type = "rolled"

        self.span   = 8_000.0
        self.load   = 40.0
        self.storey = 20

        self.current_util   = 0.0
        self.current_mass   = 0.0
        self.current_cost   = 0.0
        self.current_co2    = 0.0
        self.current_chi_lt = 1.0
        self.current_class  = 1
        self.current_Mrd    = 1.0
        self.current_area   = 1.0

        self.prev_util = 0.0
        self.prev_mass = 0.0
        self.moment_ratio = 0.0

    def _normalize(self, value: float, vmin: float, vmax: float) -> float:
        return float(np.clip((value - vmin) / (vmax - vmin + 1e-9), 0.0, 1.0))

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

        # ── UNIFORM SPAN × LOAD SAMPLING [EXP53 FIX 1] ──────────────────
        # Stratified sampling (exp51-52) caused bimodal policy collapse.
        # Reverted to uniform sampling, identical to exp48 which achieved
        # 97.5% feasibility. The 60% HSS grade curriculum bias below
        # provides sufficient high-grade exposure without stratification.
        self.span   = float(self.np_random.uniform(self.SPAN_MIN, self.SPAN_MAX))
        base_load   = float(self.np_random.uniform(self.LOAD_MIN, 140.0))
        self.storey = int(self.np_random.integers(self.STOREY_MIN, self.STOREY_MAX))
        self.load   = self._effective_load(base_load)

        # ── SPAN-PROPORTIONAL INITIALISATION [EXP53 FIX 4] ──────────────
        # h_target = span_m * 42 (reverted from exp52's span*35+150*lf).
        # Gives a section depth proportional to span, validated in exp48.
        span_m   = self.span / 1000.0
        h_target = np.clip(span_m * 42.0, 250.0, 650.0)

        load_factor = (self.load - self.LOAD_MIN) / (self.LOAD_MAX - self.LOAD_MIN)

        # ── GRADE CURRICULUM [EXP54 FIX 3] ───────────────────────────────
        # 50% uniform over all grades, 50% biased to S500-S690.
        # Reduced from 60% bias (exp53) which gave S690 ~21.7% init
        # probability vs S355 ~6.7%, contributing to S690 dominance.
        # 50%/50% gives more balanced coverage across all grades.
        if float(self.np_random.uniform(0, 1)) < 0.50:
            grade_idx = int(self.np_random.integers(0, len(self.grades)))
        else:
            grade_idx = int(self.np_random.integers(2, len(self.grades)))
        self.fy = float(self.grades[grade_idx])

        # ── DEMAND-ALIGNED H_NOISE [EXP54 FIX 2] ────────────────────────
        # h_noise scales with Med = w*L^2/8 at this episode's span/load.
        # Decouples h_noise from grade (exp53 grade-aligned caused the
        # policy to learn "high grade -> large h -> hold both forever").
        # Med_factor: 0 at light demand, 1 at Med >= 3000 kNm.
        # Low Med -> starts smaller (appropriate for light demand).
        # High Med -> starts larger (closer to feasibility for heavy demand).
        Med_init   = self.load * (self.span / 1000.0) ** 2 / 8.0
        Med_factor = float(np.clip(Med_init / 3000.0, 0.0, 1.0))
        h_noise_lo = 0.80 + 0.15 * Med_factor   # 0.80 light → 0.95 heavy
        h_noise_hi = 1.10 + 0.15 * Med_factor   # 1.10 light → 1.25 heavy
        h_noise    = float(self.np_random.uniform(h_noise_lo, h_noise_hi))

        self.h = float(np.clip(h_target * h_noise, *self.design_limits["h"]))

        b_target  = np.clip(self.h / 3.0, 120.0, 300.0)
        tf_target = np.clip(10.0 + load_factor * 20.0, 8.0, 35.0)
        tw_target = np.clip(7.0  + load_factor * 14.0, 6.0, 25.0)

        self.b  = float(np.clip(
            b_target  * float(self.np_random.uniform(0.80, 1.20)),
            *self.design_limits["b"]
        ))
        self.tf = float(np.clip(
            tf_target * float(self.np_random.uniform(0.80, 1.20)),
            *self.design_limits["tf"]
        ))
        self.tw = float(np.clip(
            tw_target * float(self.np_random.uniform(0.80, 1.20)),
            *self.design_limits["tw"]
        ))

        sec_idx           = int(self.np_random.integers(0, 2))
        self.section_type = self.section_types[sec_idx]

        self.current_util   = 0.0
        self.current_mass   = 0.0
        self.current_cost   = 0.0
        self.current_co2    = 0.0
        self.current_chi_lt = 1.0
        self.current_class  = 1
        self.current_Mrd    = 1.0
        self.current_area   = 1.0
        self.prev_util      = 0.0
        self.prev_mass      = 0.0
        self.moment_ratio   = 0.0

        return self._get_obs(), {}

    # ============================================================
    # OBSERVATION
    # ============================================================
    def _get_obs(self) -> np.ndarray:
        eps          = self.epsilon
        section_flag = 0.0 if self.section_type == "rolled" else 1.0

        flange_slenderness = (self.b / max(self.tf, 1e-6)) / (14.0 * eps)
        web_slenderness    = (self.h / max(self.tw, 1e-6)) / (124.0 * eps)

        class_1 = 1.0 if self.current_class == 1 else 0.0
        class_2 = 1.0 if self.current_class == 2 else 0.0
        class_3 = 1.0 if self.current_class == 3 else 0.0
        class_4 = 1.0 if self.current_class == 4 else 0.0

        util_delta = self.current_util - self.prev_util
        mass_delta = self.current_mass - self.prev_mass

        # [EXP51] Normalised Med — direct structural demand signal.
        # Med = w*L²/8. Ceiling 6000 kNm = 210 × 15² / 8 = 5906 kNm.
        Med_obs  = self.load * (self.span / 1000.0) ** 2 / 8.0
        Med_norm = float(np.clip(Med_obs / 6000.0, 0.0, 1.0))

        return np.array([
            self._normalize(self.span,   self.SPAN_MIN,   self.SPAN_MAX),
            self._normalize(self.load,   self.LOAD_MIN,   self.LOAD_MAX),
            self._normalize(self.storey, self.STOREY_MIN, self.STOREY_MAX),
            self._normalize(self.h,  *self.design_limits["h"]),
            self._normalize(self.b,  *self.design_limits["b"]),
            self._normalize(self.tf, *self.design_limits["tf"]),
            self._normalize(self.tw, *self.design_limits["tw"]),
            np.clip(self.h / max(self.b, 1e-6) / 5.0, 0.0, 1.0),
            self._normalize(self.fy, 355.0, 690.0),
            section_flag,
            np.clip(self.current_util   / self.norm["util"], 0.0, 1.0),
            np.clip(self.moment_ratio   / 1.5,               0.0, 1.0),
            np.clip(self.current_mass   / self.norm["mass"], 0.0, 1.0),
            np.clip(self.current_cost   / self.norm["cost"], 0.0, 1.0),
            np.clip(self.current_co2    / self.norm["co2"],  0.0, 1.0),
            np.clip(self.current_chi_lt,                     0.0, 1.0),
            np.clip(flange_slenderness / 2.0, 0.0, 1.0),
            np.clip(web_slenderness    / 2.0, 0.0, 1.0),
            class_1, class_2, class_3, class_4,
            np.clip((util_delta + 1.0) / 2.0,          0.0, 1.0),
            np.clip((mass_delta / 250.0 + 1.0) / 2.0,  0.0, 1.0),
            Med_norm,   # obs[24] — normalised Med = wL²/8, ceiling 6000 kNm
        ], dtype=np.float32)

    # ============================================================
    # STEP
    # ============================================================
    def step(self, action):
        self.curr_step += 1
        self._update_design(action)

        util, mass, penalty, class_loss, chi_lt, debug_ec3 = self._ec3_analysis()
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
            util, mass, cost, co2, chi_lt, penalty, class_loss, novelty,
            debug_ec3,
        )

        if util < 1.0 and class_loss == 0:
            self._update_memory(x)

        terminated, truncated = self._check_termination(util, class_loss, penalty)

        info = {
            "h": self.h, "b": self.b, "tf": self.tf, "tw": self.tw,
            "fy": self.fy, "section_type": self.section_type,
            "utilization": util, "mass": mass,
            "cost": cost, "co2": co2, "chi_lt": chi_lt,
            "reward": reward, "reward_terms": reward_terms,
            "span": self.span, "load": self.load, "storey": self.storey,
            "ec3": debug_ec3, "lca": debug_lca,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    # ============================================================
    # DESIGN UPDATE
    # ============================================================
    def _update_design(self, action: np.ndarray):
        progress   = self.curr_step / self.max_steps
        step_scale = 0.30 + 0.70 * 0.5 * (1.0 + np.cos(np.pi * progress))

        # [EXP53 FIX 3] Action step sizes reverted to exp48 values.
        # exp52 enlarged these (80/40/5/4) causing oscillation. 50/28/3/2.5
        # was validated in exp48 at 97.5% feasibility rate.
        self.h  = float(np.clip(self.h  + action[0] * 50.0 * step_scale, *self.design_limits["h"]))
        self.b  = float(np.clip(self.b  + action[1] * 28.0 * step_scale, *self.design_limits["b"]))
        self.tf = float(np.clip(self.tf + action[2] *  3.0 * step_scale, *self.design_limits["tf"]))
        self.tw = float(np.clip(self.tw + action[3] *  2.5 * step_scale, *self.design_limits["tw"]))

        # [EXP54 FIX 1] Softmax-snapped grade action.
        # Replaces incremental (exp51-53) which caused grade-geometry
        # coupling and policy collapse to S690 regardless of demand.
        # Softmax with temperature 0.15 over grade centres:
        #   action=-1.0 → S355 (near-certain)
        #   action= 0.0 → S500 (centre grade)
        #   action=+1.0 → S690 (near-certain)
        # Adjacent grades share similar probability for nearby action
        # values, giving PPO a smooth gradient. Grade re-selected
        # each step independently of geometry — matches exp48 behaviour.
        grade_logits = np.array([
            -((action[4] - c) ** 2) / 0.15
            for c in self._grade_centres
        ], dtype=np.float64)
        grade_logits -= grade_logits.max()
        grade_probs   = np.exp(grade_logits)
        grade_probs  /= grade_probs.sum()
        grade_idx     = int(np.argmax(grade_probs))
        self.fy       = float(self.grades[grade_idx])

        self.section_type = "rolled" if action[5] < 0 else "welded"

    def _update_state(self, util, mass, cost, co2, chi_lt, debug_ec3):
        self.prev_util      = self.current_util
        self.prev_mass      = self.current_mass
        self.current_util   = util
        self.current_mass   = mass
        self.current_cost   = cost
        self.current_co2    = co2
        self.current_chi_lt = chi_lt
        self.current_class  = debug_ec3.get("section_class", 4)
        self.moment_ratio   = debug_ec3["Med"] / max(debug_ec3["Mrd"], 1e-6)
        self.current_Mrd    = debug_ec3.get("Mrd", 1.0)
        h_web = self.h - 2.0 * self.tf
        fillet_factor = 1.05 if self.section_type == "rolled" else 1.0
        self.current_area   = (h_web * self.tw + 2.0 * self.b * self.tf) * fillet_factor

    # ============================================================
    # REWARD
    # ============================================================
    def _compute_reward(
        self,
        util, mass, cost, co2,
        chi_lt, penalty, class_loss, novelty,
        debug_ec3: dict,
    ):
        mass_n = mass / self.norm["mass"]
        cost_n = cost / self.norm["cost"]

        # ── ECONOMY ──────────────────────────────────────────────────────
        economy_reward = -5.0 * mass_n - 5.0 * cost_n

        # ── CO₂ LIFECYCLE SIGNAL ─────────────────────────────────────────
        if 0.85 <= util <= 1.10 and class_loss == 0:
            Mrd_kNm     = max(self.current_Mrd, 1.0)
            co2_per_Mrd = co2 / Mrd_kNm
            co2_lca_reward = 4.0 * np.clip(
                (8.0 - co2_per_Mrd) / 8.0, 0.0, 1.0
            )
        else:
            co2_lca_reward = 0.0

        # ── UTILISATION SCORE ────────────────────────────────────────────
        target_util = 0.96
        sigma       = 0.06

        base_score = 100.0 * np.exp(
            -((util - target_util) ** 2) / (2.0 * sigma ** 2)
        )

        if util <= 1.05:
            util_score = base_score
        else:
            util_score = (
                100.0 * np.exp(
                    -((1.05 - target_util) ** 2) / (2.0 * sigma ** 2)
                )
                - 400.0 * (util - 1.05) ** 2
            )

        # ── UNDERUTILISATION PENALTY [EXP53 FIX 6] ───────────────────────
        # Uncapped quadratic. exp52 used min(80*(0.90-util)^2, 80).
        # With h_noise=(0.85,1.15) the policy rarely starts at util=5.0
        # so the cap is unnecessary. Uncapped form gives cleaner gradient.
        if util < 0.90:
            underutil_penalty = 80.0 * (0.90 - util) ** 2
        else:
            underutil_penalty = 0.0

        # ── DEMAND-DRIVEN HSS INCENTIVE ───────────────────────────────────
        if 0.88 <= util <= 1.05 and class_loss == 0 and self.fy >= 500:
            Med_kNm       = max(debug_ec3.get("Med", 0.0), 0.0)
            demand_factor = np.clip(Med_kNm / 1500.0, 0.0, 1.0)
            k_grade       = np.clip((self.fy - 460.0) / (690.0 - 460.0), 0.0, 1.0)
            util_factor   = np.exp(-((util - 0.96) ** 2) / (2.0 * 0.08 ** 2))
            hss_demand_bonus = 10.0 * demand_factor * k_grade * util_factor
        else:
            hss_demand_bonus = 0.0

        # ── FEASIBILITY PENALTIES ─────────────────────────────────────────
        eps            = self.epsilon
        util_violation = max(util - 1.0, 0.0)

        feasibility_penalty  = 60.0 * util_violation
        feasibility_penalty += 30.0 * class_loss
        feasibility_penalty += 4.0  * penalty
        if hasattr(self, 'current_class') and self.current_class == 3:
            feasibility_penalty += 5.0

        hw = self.h - 2.0 * self.tf
        if (hw / max(self.tw, 1e-6)) > (72.0 * eps):
            feasibility_penalty += 10.0
        if self.b > self.h:
            feasibility_penalty += 12.0
        slender_ratio = self.h / max(self.b, 1e-6)
        if slender_ratio > 3.5:
            feasibility_penalty += 12.0 * (slender_ratio - 3.5)

        # ── MASS IMPROVEMENT ──────────────────────────────────────────────
        mass_improvement   = np.clip((self.prev_mass - mass) / 300.0, -1.0, 1.0)
        improvement_reward = 1.5 * mass_improvement

        # ── NOVELTY ───────────────────────────────────────────────────────
        novelty_reward = 0.15 * np.tanh(novelty)

        # ── TOTAL ─────────────────────────────────────────────────────────
        reward = (
            economy_reward
            + co2_lca_reward
            + util_score
            + hss_demand_bonus
            + improvement_reward
            + novelty_reward
            - feasibility_penalty
            - underutil_penalty
        )

        reward_terms = {
            "economy_reward":    economy_reward,
            "co2_lca_reward":    co2_lca_reward,
            "utilization_reward": util_score,
            "hss_demand_bonus":  hss_demand_bonus,
            "improvement_reward": improvement_reward,
            "novelty_reward":    novelty_reward,
            "feasibility_penalty": feasibility_penalty,
            "underutil_penalty": underutil_penalty,
        }

        return reward, reward_terms

    # ============================================================
    # TERMINATION
    # ============================================================
    def _check_termination(self, util, class_loss, penalty):
        terminated = False
        truncated  = False

        success_zone = (0.90 <= util <= 1.05 and class_loss == 0 and penalty <= 0)

        if success_zone:
            self.success_counter += 1
        else:
            self.success_counter = 0

        if self.success_counter >= 3:
            terminated = True

        if self.curr_step >= self.max_steps:
            truncated = True

        return terminated, truncated

    # ============================================================
    # COST + CO2
    # ============================================================
    def _calculate_cost_co2(self, mass: float):
        fy_key = int(self.fy)

        material_cost_factor = {
            355: 1.00, 460: 1.15, 500: 1.28,
            550: 1.42, 620: 1.60, 690: 1.85,
        }
        material_co2_factor = {
            355: 2.30, 460: 2.10, 500: 1.98,
            550: 1.88, 620: 1.75, 690: 1.63,
        }

        cost_factor = material_cost_factor.get(fy_key, 1.00)
        co2_factor  = material_co2_factor.get(fy_key, 2.30)

        material_cost = mass * cost_factor
        material_co2  = mass * co2_factor

        if self.section_type == "rolled":
            fab_factor     = 0.15
            fab_co2_factor = 0.08
        else:
            fab_factor     = 0.42
            fab_co2_factor = 0.22
            if   fy_key >= 690: fab_factor *= 1.30; fab_co2_factor *= 1.20
            elif fy_key >= 620: fab_factor *= 1.18; fab_co2_factor *= 1.10
            elif fy_key >= 550: fab_factor *= 1.10; fab_co2_factor *= 1.05

        if self.section_type == "welded" and fy_key >= 550:
            thickness_factor      = (self.tf + self.tw) / 40.0
            extra_hss_fab_penalty = 1.0 + 0.35 * thickness_factor
        else:
            extra_hss_fab_penalty = 1.0

        fabrication_cost = material_cost * fab_factor * extra_hss_fab_penalty
        fabrication_co2  = material_co2  * fab_co2_factor * extra_hss_fab_penalty

        transport_cost  = mass * 0.08;  transport_co2  = mass * 0.018
        erection_cost   = mass * 0.12;  erection_co2   = mass * 0.04
        painting_cost   = mass * 0.05;  painting_co2   = mass * 0.015
        processing_cost = mass * 0.03;  processing_co2 = mass * 0.010

        total_cost = (material_cost + fabrication_cost + transport_cost
                      + erection_cost + painting_cost + processing_cost)
        total_co2  = (material_co2  + fabrication_co2  + transport_co2
                      + erection_co2  + painting_co2  + processing_co2)

        debug = {
            "material_cost": material_cost, "fabrication_cost": fabrication_cost,
            "transport_cost": transport_cost, "erection_cost": erection_cost,
            "painting_cost": painting_cost, "processing_cost": processing_cost,
            "material_co2": material_co2, "fabrication_co2": fabrication_co2,
            "transport_co2": transport_co2, "erection_co2": erection_co2,
            "painting_co2": painting_co2, "processing_co2": processing_co2,
        }
        return total_cost, total_co2, debug

    # ============================================================
    # EC3 ANALYSIS
    # ============================================================
    def _ec3_analysis(self):
        h, b, tf, tw, fy = self.h, self.b, self.tf, self.tw, self.fy
        eps   = self.epsilon
        h_web = h - 2.0 * tf

        if self.section_type == "rolled":
            r = 0.1 * tf; fillet_factor = 1.05; fillet_Iy_fac = 1.02; torsion_factor = 1.15
        else:
            r = 0.0;      fillet_factor = 1.0;  fillet_Iy_fac = 1.0;  torsion_factor = 1.0

        A   = (h_web * tw + 2.0 * b * tf) * fillet_factor
        Iy  = (tw * h_web**3 / 12.0 + 2.0 * (b * tf**3 / 12.0
               + b * tf * (h/2.0 - tf/2.0)**2)) * fillet_Iy_fac
        Wel = Iy / (h / 2.0)
        Wpl = (2.0 * b * tf * (h/2.0 - tf/2.0) + tw * h_web**2 / 4.0) * fillet_factor
        Iz  = 2.0 * (tf * b**3) / 12.0 + h_web * tw**3 / 12.0

        c_flange = (b - tw) / 2.0 - r
        d_web    = h_web - 2.0 * r

        flange_ratio = c_flange / max(tf, 1e-6)
        web_ratio    = d_web    / max(tw, 1e-6)

        if flange_ratio <= 9.0 * eps and web_ratio <= 72.0 * eps:
            sec_class = 1
        elif flange_ratio <= 10.0 * eps and web_ratio <= 83.0 * eps:
            sec_class = 2
        elif flange_ratio <= 14.0 * eps and web_ratio <= 124.0 * eps:
            sec_class = 3
        else:
            class_severity = max(
                flange_ratio / (14.0 * eps) - 1.0,
                web_ratio    / (124.0 * eps) - 1.0
            )
            util_penalty = min(5.0, 2.0 + 2.0 * class_severity)

            return util_penalty, 4000.0, 25.0, class_severity, 0.0, {
                "Mrd": 1e-6, "Med": 1e-6, "efficiency": 0.0,
                "Ved": 1e-6, "Vpl_Rd": 1e-6, "section_class": 4,
                "lambda_lt": 0.0, "shear_ratio": 0.0,
                "moment_util": util_penalty, "deflection_util": util_penalty,
            }

        W_ref     = Wpl if sec_class <= 2 else Wel
        Mrd_basic = W_ref * fy / 1.0e6

        L    = self.span
        L_cr = L * self.ltb_restraint_factor

        It  = (2.0 * b * tf**3 + h_web * tw**3) / 3.0 * torsion_factor
        Iw  = Iz * (h - tf)**2 / 4.0
        C1  = 1.13

        Mcr = max(
            (C1 * np.pi**2 * self.E * Iz / L_cr**2)
            * np.sqrt(Iw/Iz + L_cr**2 * self.G * It / (np.pi**2 * self.E * Iz)),
            1e-3
        )

        if self.include_zg_in_mcr:
            zg = h / 2.0; C2 = 0.55
            Mcr_zg = (C2*zg)**2 * (np.pi**2*self.E*Iz/L_cr**2)
            Mcr = max(np.sqrt(Mcr**2 + Mcr_zg)
                      - C1*C2*zg*(np.pi**2*self.E*Iz/L_cr**2), 1e-3)

        lambda_lt = np.sqrt(W_ref * fy / (Mcr + 1e-9))
        alpha_lt  = (0.34 if h/b > 2.0 else 0.21) if self.section_type == "rolled" else 0.49
        phi_lt    = 0.5 * (1.0 + alpha_lt * (lambda_lt - 0.2) + lambda_lt**2)
        chi_lt    = float(np.clip(
            1.0 / (phi_lt + np.sqrt(np.maximum(phi_lt**2 - lambda_lt**2, 1e-9))),
            0.0, 1.0
        ))
        Mrd = chi_lt * Mrd_basic

        Av     = (A - 2.0*b*tf + (tw + 2.0*r)*tf) if self.section_type == "rolled" else h_web*tw
        Vpl_Rd = Av * fy / (np.sqrt(3.0) * 1.0e3)

        L_m  = L / 1_000.0
        Ved  = self.load * L_m / 2.0
        Med  = self.load * L_m**2 / 8.0

        shear_ratio = Ved / (Vpl_Rd + 1e-9)
        if shear_ratio > 0.5:
            rho  = (2.0 * shear_ratio - 1.0)**2
            Mrd *= max(1.0 - rho * (Wpl / max(Wel, 1e-6) - 1.0), 0.15)

        moment_util = Med / (Mrd + 1e-9)
        util        = float(np.clip(moment_util, 0.0, 5.0))

        w_sls           = self.load * self.sls_load_factor
        delta           = 5.0 * w_sls * L**4 / (384.0 * self.E * Iy)
        deflection_util = delta / max(L / 250.0, 1e-9)
        if deflection_util > util:
            util = float(np.clip(deflection_util, 0.0, 5.0))

        mass       = A * L * 7.85e-6
        penalty    = 50.0 * np.tanh(2.0 * max(b / h - 1.0, 0.0))
        efficiency = Mrd / max(mass, 1e-6)

        return util, mass, penalty, 0.0, chi_lt, {
            "Mrd": Mrd, "Med": Med, "efficiency": efficiency,
            "Ved": Ved, "Vpl_Rd": Vpl_Rd, "section_class": sec_class,
            "lambda_lt": lambda_lt, "shear_ratio": shear_ratio,
            "moment_util": moment_util, "deflection_util": deflection_util,
            "w_uls": self.load, "w_sls": w_sls,
        }

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
        section_bonus = np.where(memory_array[:, 5] != x[5], 0.35, 0.0)
        dists   = geom_dists + section_bonus
        k       = min(5, len(dists))
        novelty = float(np.mean(np.sort(dists)[:k]))
        return float(np.clip(novelty, 0.0, 5.0))

    def _update_memory(self, x: np.ndarray):
        if len(self.memory) > 0:
            memory_array = np.array(self.memory)
            geom_dists   = np.linalg.norm(memory_array[:, :5] - x[:5], axis=1)
            section_diff = np.where(memory_array[:, 5] != x[5], 0.35, 0.0)
            if np.min(geom_dists + section_diff) < self.memory_similarity_threshold:
                return
        self.memory.append(x.copy())
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
