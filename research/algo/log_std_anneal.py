"""
research/algo/log_std_anneal.py
================================================================
An annealed CEILING on the PPO policy's action log_std, applied only
over the back portion of training, in ISOLATION from the entropy
coefficient (which this codebase holds constant throughout via
`--ent_coef`, exactly as in the best validated result to date --
feasibility_gated, 38.69% +/- 6.72% cost gap, 100% feasibility, 5
seeds -- so this callback changes nothing about that baseline's
exploration mechanism except this one, explicitly-scoped intervention).

WHY THIS EXISTS (see project research-diagnosis history for the full
chain of evidence; summarised here so this file is self-contained)
--------------------------------------------------------------------
1. A corrected episode-truncation diagnostic (`diagnose_step_scale_timing.py`,
   zero training cost, existing checkpoints only) showed that trained
   `feasibility_gated` policies converge to a stable utilisation FIXED
   POINT well below the true optimum (mean util ~0.68-0.82 vs. true
   optimum ~1.0) almost immediately -- by step ~5-10 of a 40-step
   episode -- and then stay essentially flat for the rest of the
   episode, AND for 60 additional steps run at the exact step-size the
   policy trained under (i.e. genuinely given more time at its trained
   pace, not more time at a re-paced, never-trained-under step size).
   Step-to-step utilisation change decays ~70x (0.64 -> 0.009) while the
   step-size-anneal schedule itself only shrinks ~3.3x (1.0x -> 0.30x)
   over the same window, meaning the convergence is a property of the
   POLICY, not of a shrinking action-magnitude budget running out.
   -> This rules OUT "the episode ends before the policy is done
      improving" as the mechanism, and points at "the policy has
      converged to the wrong fixed point" instead.

2. Inspecting the SAME checkpoints' learned action log_std directly
   (all 15 arm x seed combinations from Experiment 1: shaped,
   feasibility_gated, lagrangian) shows every one has grown far beyond
   the actual action range: this environment's action_space is
   Box(-1, 1) (width 2), but trained std values range ~2.3-5.8 --
   1.15x to 2.9x the ENTIRE range width. Within the cleanest arm
   (feasibility_gated, least confounded by the separately-documented
   section-type-selection instability of the `shaped` arm), a
   checkpoint's mean action std correlates strongly (r=0.94 with
   distance-from-optimum, r=0.97 with cost gap; n=5 seeds) with how far
   its converged policy sits from the boundary.
   -> Consistent with a risk-averse-retreat mechanism: SB3 samples
      unbounded Gaussian noise around the mean action and clips to
      [-1, 1] at execution time. With std this large relative to the
      range, a large fraction of TRAINING-time sampled actions saturate
      at the clip boundary regardless of what the mean predicts. A
      policy trained under noise that large, next to a hard
      constraint-violation penalty, has an incentive to keep its MEAN
      action safely away from the boundary, since even a "correct" mean
      action would frequently get knocked into a costly violation by
      sampling noise.

3. A PRIOR, DIFFERENT attempt at controlling this (documented in the
   project history; that code is not present in this repository) applied
   an aggressive entropy-coefficient anneal-to-zero AND a CONSTANT
   (present from timestep 0, never annealed) hard log_std ceiling of
   0.70 -- simultaneously -- and made results much worse (100-104% gap
   vs. the 38-70% baseline). That is not evidence against controlling
   std; it is evidence against stacking two independent, non-annealed,
   from-the-very-start exploration-killing interventions, which starves
   a PPO policy of the exploration it needs early in training before it
   has learned anything useful.

This callback is deliberately scoped to avoid repeating that mistake:
  (a) it is the ONLY exploration-affecting intervention active when used
      -- ent_coef stays at whatever constant value is passed to PPO,
      identical to the best-known baseline;
  (b) it leaves std COMPLETELY uncontrolled for the first
      `anneal_start_frac` of training, so early-training exploration
      (when the policy has the least idea what a good design looks
      like) is bit-for-bit identical to the unmodified baseline;
  (c) it anneals its ceiling down GRADUALLY over the remaining training
      budget, rather than snapping to a fixed value, so there is no
      discontinuity for the optimizer to react to;
  (d) `ceiling_start` should be set generously above the largest std the
      unclamped baseline is ever observed to reach by
      `anneal_start_frac` of training (empirically ~2.3-5.8 by the END
      of a full 1M-step run; a generous default is provided), so the
      callback is a guaranteed no-op throughout the uncontrolled phase
      and there is no discontinuity at the phase boundary either.

Only ever CLAMPS THE MAXIMUM of log_std (`.clamp_(max=...)`); never
raises it, never touches the mean-action head, never touches the reward,
objective, or physics in any way.
================================================================
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class LogStdAnnealCallback(BaseCallback):
    """
    Anneal a ceiling on `model.policy.log_std` from `ceiling_start` down to
    `ceiling_end` (both given as plain std values, not log-std) linearly in
    log-space, starting only after `anneal_start_frac` of `total_timesteps`
    has elapsed. Before that point, this callback is a no-op.

    `total_timesteps` is the TRUE, GRAND total training budget for the run
    (e.g. 1_000_000), passed in explicitly at construction -- deliberately
    NOT read from `self.model._total_timesteps`. SB3 resets that internal
    attribute at the start of every `.learn()` call to whatever budget is
    passed to THAT call, which is only equal to the grand total for a
    single, uninterrupted run. If training is resumed/chunked across
    multiple `.learn()` invocations (e.g. because the process was
    restarted, or run in smaller pieces), using the model's own
    `_total_timesteps` would silently reset the anneal-progress fraction
    at every resume boundary -- exactly the class of bug already
    documented in this project's history for the (unrelated)
    entropy-coefficient annealing callback, where recomputing the schedule
    from scratch on resume caused a discontinuous jump. Taking the grand
    total as an explicit, externally-supplied constant instead makes this
    callback's progress fraction (`self.num_timesteps / total_timesteps`)
    correct and continuous regardless of how many chunks the run is split
    into, since `self.num_timesteps` (SB3's own persistent global step
    counter) is preserved correctly across `PPO.load()` + resumed
    `.learn(reset_num_timesteps=False)` calls.
    """

    def __init__(self, total_timesteps: int, ceiling_start: float = 8.0, ceiling_end: float = 1.0,
                 anneal_start_frac: float = 0.5, verbose: int = 0):
        super().__init__(verbose)
        assert total_timesteps > 0
        assert 0.0 <= anneal_start_frac < 1.0, "anneal_start_frac must be in [0, 1)"
        assert 0.0 < ceiling_end < ceiling_start, "ceiling_end must be a smaller, positive std than ceiling_start"
        self.total_timesteps = total_timesteps
        self.ceiling_start = ceiling_start
        self.ceiling_end = ceiling_end
        self.log_ceiling_start = float(np.log(ceiling_start))
        self.log_ceiling_end = float(np.log(ceiling_end))
        self.anneal_start_frac = anneal_start_frac
        self._last_logged_ceiling = None

    def _on_step(self) -> bool:
        frac = self.num_timesteps / self.total_timesteps
        if frac <= self.anneal_start_frac:
            return True  # uncontrolled phase -- identical to unmodified baseline

        anneal_progress = min((frac - self.anneal_start_frac) / (1.0 - self.anneal_start_frac), 1.0)
        log_ceiling = self.log_ceiling_start + anneal_progress * (self.log_ceiling_end - self.log_ceiling_start)

        with torch.no_grad():
            self.model.policy.log_std.data.clamp_(max=log_ceiling)

        if self._last_logged_ceiling is None or abs(log_ceiling - self._last_logged_ceiling) > 1e-4:
            self.logger.record("log_std_anneal/ceiling_std", float(np.exp(log_ceiling)))
            self.logger.record("log_std_anneal/actual_mean_std",
                                float(np.exp(self.model.policy.log_std.detach().cpu().numpy()).mean()))
            self._last_logged_ceiling = log_ceiling
        return True
