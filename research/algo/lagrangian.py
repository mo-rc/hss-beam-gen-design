"""
research/algo/lagrangian.py
================================================================
PPO-Lagrangian via dual ascent (Chow et al. 2017; Ray, Achiam & Amodei
2019 "Benchmarking Safe RL" formulation), implemented as an SB3 callback
rather than a modified PPO loss.

WHY THIS DESIGN (callback, not a custom PPO class)
----------------------------------------------------
The environment (research/envs/hss_env.py, reward_mode="lagrangian")
already computes reward = -economy - sum_i(lambda_i * g_i(s)) internally,
using whatever lambda_i values are currently set via
`env.set_lagrange_multipliers()`. This means standard, unmodified SB3 PPO
can be used as the optimizer -- it just sees a scalar reward like any
other RL problem. The only additional machinery needed is: after each
rollout, look at how much each constraint was actually violated on
average, and adjust lambda_i up (more penalty) if constraints are being
violated more than the target budget, or down (less penalty, allowing
more economy-seeking) if the budget is being comfortably met.

This keeps the environment PPO-agnostic (the same reward_mode="lagrangian"
environment could drive any on-policy or off-policy algorithm without
modification) and keeps the Lagrangian mechanism fully auditable as a
~40-line callback rather than buried inside a modified policy-gradient
loss, which matters for a reviewer trying to verify correctness.

DUAL ASCENT UPDATE RULE
-------------------------
    lambda_i <- max(0, lambda_i + eta_i * (mean_violation_i - budget_i))

`budget_i` is normally 0 (constraint must never be violated, in
expectation), but is exposed as a parameter so a small non-zero budget
(e.g. accept 2% mean violation of the utilization constraint) can be used
if pure zero-budget proves too conservative in practice (a legitimate,
literature-supported relaxation -- report whichever budget is used and
why in the paper's Methods section, don't just silently tune it away).

USAGE
------
    from stable_baselines3 import PPO
    from research.algo.lagrangian import LagrangianCallback

    lagrangian_cb = LagrangianCallback(
        constraint_names=["g1_util", "g2_class", "g3_geom"],
        etas={"g1_util": 5.0, "g2_class": 5.0, "g3_geom": 2.0},
        budgets={"g1_util": 0.0, "g2_class": 0.0, "g3_geom": 0.0},
        lambda_max=200.0,
        log_every=1,
    )
    model = PPO("MlpPolicy", vec_env, ...)
    model.learn(total_timesteps=..., callback=lagrangian_cb)
================================================================
"""

from __future__ import annotations
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LagrangianCallback(BaseCallback):

    def __init__(
        self,
        constraint_names: list[str],
        etas: dict[str, float],
        budgets: dict[str, float] | None = None,
        lambda_max: float = 200.0,
        update_freq: int = 2048,
        log_every: int = 1,
        verbose: int = 1,
    ):
        """
        update_freq: number of ENVIRONMENT TIMESTEPS between dual-ascent
            updates, enforced explicitly via a timestep counter rather than
            relying on `_on_rollout_end`. This matters because SB3 fires
            `_on_rollout_end` once per (n_steps * n_envs) for on-policy
            algorithms (PPO) but essentially once per training step for
            off-policy algorithms (DDPG/TD3) -- using the raw event for
            off-policy algorithms would update the multiplier thousands of
            times more often than intended, saturating it almost instantly
            on a handful of unlucky steps rather than a stable running
            estimate of the violation rate. Using an explicit timestep
            interval makes the update cadence identical (and therefore
            comparable) across every algorithm in the comparison suite.
        """
        super().__init__(verbose)
        self.constraint_names = constraint_names
        self.etas = etas
        self.budgets = budgets or {k: 0.0 for k in constraint_names}
        self.lambda_max = lambda_max
        self.update_freq = update_freq
        self.log_every = log_every
        self._violation_buffer = {k: [] for k in constraint_names}
        self._update_count = 0
        self._last_update_at = 0
        self.lambda_history = []  # list of dicts, one per dual-ascent update — for the paper's Fig.

    def _on_step(self) -> bool:
        # Collect per-step constraint violations from every parallel env's info dict.
        for info in self.locals.get("infos", []):
            violations = info.get("constraint_violations")
            if violations is not None:
                for k in self.constraint_names:
                    self._violation_buffer[k].append(violations[k])

        if self.num_timesteps - self._last_update_at >= self.update_freq:
            self._dual_ascent_update()
            self._last_update_at = self.num_timesteps
        return True

    def _on_rollout_end(self) -> None:
        """Kept as a no-op override (was the update trigger in an earlier
        version of this callback). Update cadence is now enforced explicitly
        in `_on_step` via `update_freq`, which behaves identically regardless
        of which SB3 algorithm (on- or off-policy) is training, unlike
        `_on_rollout_end`'s native firing frequency which differs between
        the two. Left in place so any external code relying on the
        callback's rollout-boundary hook (e.g. for logging) still works."""
        pass

    def _dual_ascent_update(self) -> None:
        mean_violations = {}
        for k in self.constraint_names:
            buf = self._violation_buffer[k]
            mean_violations[k] = float(np.mean(buf)) if buf else 0.0
            self._violation_buffer[k] = []

        current_lambdas = self.training_env.env_method("get_lagrange_multipliers")[0]
        new_lambdas = {}
        for k in self.constraint_names:
            grad = mean_violations[k] - self.budgets[k]
            new_lambdas[k] = float(np.clip(current_lambdas[k] + self.etas[k] * grad, 0.0, self.lambda_max))

        self.training_env.env_method("set_lagrange_multipliers", new_lambdas)

        self._update_count += 1
        record = {"update": self._update_count, "timesteps": self.num_timesteps,
                  **{f"violation_{k}": mean_violations[k] for k in self.constraint_names},
                  **{f"lambda_{k}": new_lambdas[k] for k in self.constraint_names}}
        self.lambda_history.append(record)

        if self.verbose and self._update_count % self.log_every == 0:
            v_str = " ".join(f"{k}={mean_violations[k]:.4f}" for k in self.constraint_names)
            l_str = " ".join(f"{k}={new_lambdas[k]:.2f}" for k in self.constraint_names)
            print(f"[Lagrangian] update {self._update_count:4d} (t={self.num_timesteps:>8d})  "
                  f"mean_violations: {v_str}  ->  lambdas: {l_str}")

        # Log to SB3's own logger too, so it shows up in TensorBoard alongside
        # everything else (loss, entropy, ep_rew_mean, ...) without extra plumbing.
        for k in self.constraint_names:
            self.logger.record(f"lagrangian/mean_violation_{k}", mean_violations[k])
            self.logger.record(f"lagrangian/lambda_{k}", new_lambdas[k])

    def get_history_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.lambda_history)
