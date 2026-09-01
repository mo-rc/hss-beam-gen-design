"""
research/algo/log_std_anneal_adaptive.py
================================================================
FALLBACK for if extending seed42's training doesn't resolve its
regression (see chat / MERGE_NOTES.md for the full diagnosis chain).
Only build/use this if the plain extension test shows seed42 is stuck,
not just slow -- don't reach for this pre-emptively.

PROBLEM WITH THE FIXED-FRACTION SCHEDULE (log_std_anneal.py):
`anneal_start_frac` is identical across seeds, but seeds do NOT reach
the same peak exploration std before that point -- observed range
across 5 seeds of the actual merged run: 2.29-3.11 (train/std peak),
with seed42 the highest (2.95-3.11 depending on which log is read) and
seed45 the lowest (2.29). A seed that free-explores to a higher peak
needs more post-peak "settling" updates to re-converge its mean action
precisely -- but under a fixed wall-clock schedule, every seed gets the
SAME absolute number of settling steps regardless of how much settling
it actually needs. Seed42's lower final reward and unfinished-looking
reward trajectory (still rising at the final checkpoint, unlike the
other 4 seeds which visibly plateau) is consistent with this specific
mechanism, though not proven by it alone -- see the seed42 extension
test this fallback is conditioned on.

FIX: trigger the anneal off each seed's OWN observed std trajectory
(plateau detection) instead of a fixed fraction of total_timesteps, so
every seed gets an anneal window sized relative to its own exploration
history rather than an identical wall-clock window.

MECHANISM
----------
- Track std at every rollout (already logged by SB3's own PPO logger
  as `train/std`; this callback reads `model.policy.log_std` directly,
  same source, no extra cost).
- Maintain a short rolling window (`plateau_window` rollouts). Once the
  relative change in std across that window drops below
  `plateau_rel_tol` for `plateau_patience` consecutive checks, treat
  that as "the free-exploration growth phase has topped out" and begin
  the anneal from there.
- Safety rails, both defensible in a methods section without extra
  hand-tuning per seed:
    * `min_start_frac`: never start the anneal before this fraction of
      training, even if a plateau is detected early (guards against
      spuriously flat std very early in training, before the policy
      has learned anything -- same rationale as the original
      callback's fixed `anneal_start_frac` floor, just as a lower
      bound instead of the exact trigger).
    * `max_start_frac`: if no plateau is ever detected (e.g. std keeps
      climbing throughout), fall back to starting the anneal here
      regardless, so the callback always guarantees some settling
      window and degrades to fixed-fraction behaviour rather than
      silently never triggering.
- Once triggered, the anneal itself is identical in shape to the
  original callback: linear-in-log-space decay from `ceiling_start`
  (which should be set at/just above the observed peak std at trigger
  time -- this callback reads that automatically rather than requiring
  a hand-set generous constant) down to `ceiling_end`, over the
  remaining `anneal_duration_frac` of total_timesteps from the trigger
  point.

This callback is a strict generalisation of the fixed-fraction one --
setting `plateau_patience` to infinity (or `plateau_window` larger
than the run) reduces it to triggering at `max_start_frac`, i.e. the
same fixed-fraction behaviour, so it is a safe drop-in replacement to
validate against the existing 5-seed result before trusting it on new
arms.
================================================================
"""
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class AdaptiveLogStdAnnealCallback(BaseCallback):
    def __init__(self, total_timesteps: int, ceiling_end: float = 1.0,
                 min_start_frac: float = 0.3, max_start_frac: float = 0.75,
                 anneal_duration_frac: float = 0.4,
                 plateau_window: int = 5, plateau_rel_tol: float = 0.03,
                 plateau_patience: int = 3, ceiling_start_margin: float = 1.15,
                 verbose: int = 0):
        super().__init__(verbose)
        assert total_timesteps > 0
        assert 0.0 <= min_start_frac < max_start_frac < 1.0
        assert ceiling_end > 0.0
        assert ceiling_start_margin > 1.0, "margin must be >1.0x the observed peak std at trigger time"
        self.total_timesteps = total_timesteps
        self.ceiling_end = ceiling_end
        self.log_ceiling_end = float(np.log(ceiling_end))
        self.min_start_frac = min_start_frac
        self.max_start_frac = max_start_frac
        self.anneal_duration_frac = anneal_duration_frac
        self.plateau_window = plateau_window
        self.plateau_rel_tol = plateau_rel_tol
        self.plateau_patience = plateau_patience
        self.ceiling_start_margin = ceiling_start_margin

        self._std_history = []          # (num_timesteps, mean_std) per rollout
        self._plateau_count = 0
        self._triggered = False
        self._trigger_timestep = None
        self._log_ceiling_start = None  # set once, at trigger time
        self._last_logged_ceiling = None

    def _current_mean_std(self) -> float:
        with torch.no_grad():
            return float(torch.exp(self.model.policy.log_std.detach()).mean().cpu())

    def _check_plateau(self) -> bool:
        if len(self._std_history) < self.plateau_window + 1:
            return False
        recent = [v for _, v in self._std_history[-(self.plateau_window + 1):]]
        rel_changes = [abs(recent[i] - recent[i - 1]) / max(recent[i - 1], 1e-6)
                        for i in range(1, len(recent))]
        return max(rel_changes) < self.plateau_rel_tol

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        frac = self.num_timesteps / self.total_timesteps
        mean_std = self._current_mean_std()

        if not self._triggered:
            self._std_history.append((self.num_timesteps, mean_std))
            plateaued = frac >= self.min_start_frac and self._check_plateau()
            forced = frac >= self.max_start_frac
            if plateaued:
                self._plateau_count += 1
            else:
                self._plateau_count = 0

            if self._plateau_count >= self.plateau_patience or forced:
                self._triggered = True
                self._trigger_timestep = self.num_timesteps
                peak_std = max(v for _, v in self._std_history)
                self._log_ceiling_start = float(np.log(peak_std * self.ceiling_start_margin))
                trigger_reason = "plateau_detected" if plateaued else "max_start_frac_fallback"
                if self.verbose > 0:
                    print(f"[AdaptiveLogStdAnneal] TRIGGERED at t={self.num_timesteps} "
                          f"(frac={frac:.3f}, reason={trigger_reason}, "
                          f"peak_std_seen={peak_std:.4f}, ceiling_start={np.exp(self._log_ceiling_start):.4f})")
                self.logger.record("log_std_anneal_adaptive/trigger_timestep", self._trigger_timestep)
                self.logger.record("log_std_anneal_adaptive/peak_std_at_trigger", peak_std)
            return  # uncontrolled phase this rollout regardless (trigger takes effect next rollout)

        # annealing phase, relative to the seed-specific trigger point
        window_steps = self.anneal_duration_frac * self.total_timesteps
        anneal_progress = min((self.num_timesteps - self._trigger_timestep) / max(window_steps, 1.0), 1.0)
        log_ceiling = self._log_ceiling_start + anneal_progress * (self.log_ceiling_end - self._log_ceiling_start)

        with torch.no_grad():
            self.model.policy.log_std.data.clamp_(max=log_ceiling)

        if self._last_logged_ceiling is None or abs(log_ceiling - self._last_logged_ceiling) > 1e-4:
            self.logger.record("log_std_anneal_adaptive/ceiling_std", float(np.exp(log_ceiling)))
            self.logger.record("log_std_anneal_adaptive/actual_mean_std", mean_std)
            self._last_logged_ceiling = log_ceiling

    def get_diagnostics(self):
        import pandas as pd
        df = pd.DataFrame(self._std_history, columns=["timestep", "mean_std"])
        df["triggered_at"] = self._trigger_timestep
        return df
