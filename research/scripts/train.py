"""
research/scripts/train.py
================================================================
Unified training entry point for all reward-mode arms (shaped,
feasibility_gated, lagrangian). Hyperparameters default to a standard
PPO configuration for this environment so results are comparable across
arms -- ONLY reward_mode (and, for lagrangian, the dual-ascent settings)
differs between runs, per the paper's ablation methodology.

USAGE
------
  # Arm "shaped" (weighted-sum reward, NO grade-specific term):
  python research/scripts/train.py --reward_mode shaped --run_name shaped_cost --seed 42

  # Feasibility-gated (safe-RL style):
  python research/scripts/train.py --reward_mode feasibility_gated --run_name gated_cost --seed 42

  # Lagrangian-constrained (primary proposed method):
  python research/scripts/train.py --reward_mode lagrangian --economy_metric cost \\
      --run_name lagrangian_cost --seed 42

  # Catalog action-space variant (discrete rolled-section catalog):
  python research/scripts/train.py --reward_mode lagrangian --env_type catalog \\
      --run_name lagrangian_cost_catalog --seed 42

  # Multi-seed replication (run this 5x with --seed 42/43/44/45/46 per arm,
  # or use research/scripts/run_multiseed.py to automate it)
================================================================
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from research.envs.hss_env import HSSBeamEnv, REWARD_MODES, ECONOMY_METRICS
from research.envs.hss_catalog_env import HSSBeamCatalogEnv
from research.algo.lagrangian import LagrangianCallback
from research.algo.log_std_anneal import LogStdAnnealCallback


def make_env(env_type, reward_mode, economy_metric, lagrange_init, ltb_factor, sls_factor,
             economy_reward_mode, seed, rank):
    """
    env_type: "continuous" (default, HSSBeamEnv -- 6-dim Box action, softmax-
        snapped grade) or "catalog" (HSSBeamCatalogEnv -- MultiDiscrete action
        over research/envs/rolled_catalog.py's procedural rolled-section
        catalog, rolled-only).
    """
    def _init():
        cls = HSSBeamCatalogEnv if env_type == "catalog" else HSSBeamEnv
        env = cls(
            reward_mode=reward_mode, economy_metric=economy_metric,
            lagrange_init=lagrange_init,
            ltb_restraint_factor=ltb_factor, sls_load_factor=sls_factor,
            economy_reward_mode=economy_reward_mode,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_type", choices=["continuous", "catalog"], default="continuous",
                    help="'continuous': HSSBeamEnv (6-dim Box action). "
                         "'catalog': HSSBeamCatalogEnv (MultiDiscrete action over a "
                         "procedural rolled-section catalog, rolled sections only).")
    p.add_argument("--reward_mode", choices=REWARD_MODES, required=True)
    p.add_argument("--economy_metric", choices=ECONOMY_METRICS, default="cost")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.15)
    p.add_argument("--ent_coef", type=float, default=0.03)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--ltb_factor", type=float, default=0.40)
    p.add_argument("--sls_factor", type=float, default=0.50)
    # --- Lagrangian-only settings ---
    p.add_argument("--eta_util", type=float, default=5.0)
    p.add_argument("--eta_class", type=float, default=5.0)
    p.add_argument("--eta_geom", type=float, default=2.0)
    p.add_argument("--lambda_max", type=float, default=200.0)
    p.add_argument("--budget_util", type=float, default=0.0,
                    help="Allowed mean utilisation-constraint violation (0 = strict).")
    # --- Annealed log_std ceiling (opt-in; default OFF reproduces the
    # unmodified 38.69%+/-6.72% feasibility_gated baseline exactly) ---
    # See research/algo/log_std_anneal.py for the full rationale. Applied
    # in isolation from ent_coef, which stays constant as normal.
    p.add_argument("--log_std_anneal", action="store_true",
                    help="Enable the annealed log_std ceiling (see research/algo/log_std_anneal.py).")
    p.add_argument("--log_std_ceiling_start", type=float, default=8.0,
                    help="Std ceiling (not log-std) during/at the start of the anneal phase; "
                         "should stay a no-op through the uncontrolled phase (default is generously "
                         "above the ~2.3-5.8 std observed at the end of unclamped 1M-step runs).")
    p.add_argument("--log_std_ceiling_end", type=float, default=1.0,
                    help="Std ceiling at the end of training.")
    p.add_argument("--log_std_anneal_start_frac", type=float, default=0.5,
                    help="Fraction of total_timesteps before the ceiling starts tightening; "
                         "before this point std is completely uncontrolled (bit-for-bit baseline).")
    p.add_argument("--economy_reward_mode", choices=["linear", "log_relative"], default="linear",
                    help="See research/envs/hss_env.py's _economy_reward() docstring. Default 'linear' "
                         "reproduces existing behaviour exactly.")
    p.add_argument("--out_dir", type=str, default="./research/models")
    args = p.parse_args()

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    lagrange_init = dict(g1_util=0.0, g2_class=0.0, g3_geom=0.0)

    env_fns = [make_env(args.env_type, args.reward_mode, args.economy_metric, lagrange_init,
                         args.ltb_factor, args.sls_factor, args.economy_reward_mode, args.seed, i)
               for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)  # n_envs==1 fix: VecNormalize requires VecEnv semantics, not a raw env
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=50.0, gamma=args.gamma)

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm, seed=args.seed,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log=os.path.join(args.out_dir, "..", "runs"),
        verbose=1,
    )

    callbacks = [CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1),
                                     save_path=run_dir, name_prefix="checkpoint",
                                     save_vecnormalize=True)]
    lagrangian_cb = None
    if args.reward_mode == "lagrangian":
        lagrangian_cb = LagrangianCallback(
            constraint_names=["g1_util", "g2_class", "g3_geom"],
            etas={"g1_util": args.eta_util, "g2_class": args.eta_class, "g3_geom": args.eta_geom},
            budgets={"g1_util": args.budget_util, "g2_class": 0.0, "g3_geom": 0.0},
            lambda_max=args.lambda_max, update_freq=args.n_steps * args.n_envs, log_every=1, verbose=1,
        )
        callbacks.append(lagrangian_cb)

    if args.log_std_anneal:
        callbacks.append(LogStdAnnealCallback(
            total_timesteps=args.timesteps,
            ceiling_start=args.log_std_ceiling_start, ceiling_end=args.log_std_ceiling_end,
            anneal_start_frac=args.log_std_anneal_start_frac, verbose=1,
        ))

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                tb_log_name=args.run_name)

    model.save(os.path.join(run_dir, "final_model"))
    vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    if lagrangian_cb is not None:
        lagrangian_cb.get_history_dataframe().to_csv(
            os.path.join(run_dir, "lagrange_history.csv"), index=False)

    import json
    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nDone. Model, VecNormalize stats, config" +
          (", and Lagrange multiplier history" if lagrangian_cb else "") +
          f" saved to {run_dir}")


if __name__ == "__main__":
    main()