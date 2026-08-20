"""
research/envs/hss_catalog_env.py
================================================================
Redesign 2 (catalog-realism arm). Subclasses HSSBeamEnv and overrides
ONLY the action space and the design-update mechanics: instead of nudging
four continuous geometry variables, the agent navigates a discrete
catalog of rolled I-section geometries (research/envs/rolled_catalog.py)
and a discrete grade list, both by relative step (not absolute jump), to
preserve the same iterative-refinement MDP structure (up to 40 steps per
episode) as the continuous arm — this keeps the two arms comparable head-
to-head rather than confounding "discrete vs continuous" with "one-shot
vs iterative".

EVERYTHING ELSE IS INHERITED, UNCHANGED: EC3 mechanics (_ec3_analysis),
cost/CO2 model (_calculate_cost_co2), all four reward modes, constraint
definitions, feasible/in_target_band labelling, termination logic. This
means the catalog arm and continuous arm can be trained with the SAME
research/scripts/train.py reward-mode machinery and compared fairly:
only the action space and what counts as a "reachable" geometry differ.

MODELLING SCOPE: this arm covers ROLLED sections only (section_type is
fixed to "rolled" throughout an episode). Welded plate girders are
inherently custom-fabricated to continuous dimensions, so restricting
them to a discrete catalog would misrepresent real practice; the
continuous arm remains the correct model for welded design. When
comparing the two arms in the paper, restrict the continuous arm's own
evaluation set to its rolled-section episodes for a fair like-for-like
comparison (a filter on `info["section_type"]`, no retraining needed).

ACTION SPACE
-------------
    MultiDiscrete([7, 5])
    action[0]: catalog index step   in {-10,-5,-2,-1,0,+1,+2} (index 0..6)
    action[1]: grade index step     in {-2,-1,0,+1,+2}         (index 0..4)
Both are RELATIVE moves (like the continuous arm's nudges), applied to the
agent's current catalog_index / grade_index, clipped to valid range.
================================================================
"""

import os
import numpy as np
from gymnasium import spaces

from research.envs.hss_env import HSSBeamEnv
from research.envs.rolled_catalog import generate_catalog

CATALOG_INDEX_STEPS = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
GRADE_INDEX_STEPS = [-2, -1, 0, 1, 2]


class HSSBeamCatalogEnv(HSSBeamEnv):

    def __init__(self, catalog_csv: str | None = None, **kwargs):
        kwargs["reward_mode"] = kwargs.get("reward_mode", "lagrangian")
        super().__init__(**kwargs)

        if catalog_csv and os.path.exists(catalog_csv):
            import pandas as pd
            self.catalog = pd.read_csv(catalog_csv)
        else:
            self.catalog = generate_catalog()
        self.n_catalog = len(self.catalog)
        self._h_arr = self.catalog["h_mm"].to_numpy()
        self._b_arr = self.catalog["b_mm"].to_numpy()
        self._tf_arr = self.catalog["tf_mm"].to_numpy()
        self._tw_arr = self.catalog["tw_mm"].to_numpy()

        # Override the inherited continuous Box action space.
        self.action_space = spaces.MultiDiscrete([len(CATALOG_INDEX_STEPS), len(GRADE_INDEX_STEPS)])

        self.catalog_index = self.n_catalog // 2
        self.grade_index = 0

    def _apply_catalog_index(self, idx: int):
        idx = int(np.clip(idx, 0, self.n_catalog - 1))
        self.catalog_index = idx
        self.h = float(self._h_arr[idx])
        self.b = float(self._b_arr[idx])
        self.tf = float(self._tf_arr[idx])
        self.tw = float(self._tw_arr[idx])

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Snap the parent class's continuous curriculum-sampled starting
        # geometry to the nearest catalog entry, so the curriculum (demand-
        # appropriate starting depth etc.) still applies here.
        dists = ((self._h_arr - self.h) ** 2 + (self._b_arr - self.b) ** 2
                 + 25 * (self._tf_arr - self.tf) ** 2 + 25 * (self._tw_arr - self.tw) ** 2)
        self._apply_catalog_index(int(np.argmin(dists)))
        self.section_type = "rolled"
        self.grade_index = int(np.argmin(np.abs(self.grades - self.fy)))
        self.fy = float(self.grades[self.grade_index])
        return self._get_obs(), info

    def _update_design(self, action):
        # action is a length-2 int array from MultiDiscrete: [catalog_step_idx, grade_step_idx]
        catalog_step = CATALOG_INDEX_STEPS[int(action[0])]
        grade_step = GRADE_INDEX_STEPS[int(action[1])]

        self._apply_catalog_index(self.catalog_index + catalog_step)
        self.grade_index = int(np.clip(self.grade_index + grade_step, 0, len(self.grades) - 1))
        self.fy = float(self.grades[self.grade_index])
        self.section_type = "rolled"
