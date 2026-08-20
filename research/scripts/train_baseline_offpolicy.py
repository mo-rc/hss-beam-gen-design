"""
research/scripts/train_baseline_offpolicy.py
================================================================
DDPG and TD3 baselines on the IDENTICAL environment/reward used for the
PPO arms (research/envs/hss_env.py). This directly answers the question
"is PPO actually the right algorithmic choice, or would any continuous-
control method do?" -- previously an unsupported assertion in paper.md,
now a real comparison.

Both DDPG and TD3 are included because DDPG alone is a weak baseline by
2020s standards (well-documented overestimation-bias issues) -- TD3 is
DDPG's direct, twin-critic-delayed-policy fix, and including both lets
the paper report "PPO vs. the DDPG family" rather than "PPO vs. one
specific, somewhat dated, algorithm", which is a stronger comparison.

Note: DDPG/TD3 use a replay buffer and off-policy updates, so `n_envs`
should stay LOW (1-4) -- unlike PPO these algorithms do not benefit from
massive parallel rollout collection in the same way, and high n_envs with
small buffers can destabilise training. Defaults reflect this.

USAGE
------
  python research/scripts/train_baseline_offpolicy.py --algo ddpg \\
      --reward_mode lagrangian --run_name baseline_ddpg --seed 42
  python research/scripts/train_baseline_offpolicy.py --algo td3 \\
      --reward_mode lagrangian --run_name baseline_td3 --seed 42
================================================================
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np

from research.envs.hss_env import HSSBeamEnv, REWARD_MODES, ECONOMY_METRICS
from research.algo.lagrangian import LagrangianCallback


def make_env(reward_mode, economy_metric, seed, rank):
    def _init():
        env = HSSBeamEnv(reward_mode=reward_mode, economy_metric=economy_metric)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ddpg", "td3"], required=True)
    p.add_argument("--reward_mode", choices=REWARD_MODES, default="lagrangian")
    p.add_argument("--economy_metric", choices=ECONOMY_METRICS, default="cost")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--buffer_size", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--action_noise_sigma", type=float, default=0.2)
    p.add_argument("--eta_util", type=float, default=5.0)
    p.add_argument("--eta_class", type=float, default=5.0)
    p.add_argument("--eta_geom", type=float, default=2.0)
    p.add_argument("--lambda_max", type=float, default=200.0)
    p.add_argument("--lagrangian_update_freq", type=int, default=2048,
                    help="Environment timesteps between dual-ascent updates (kept equal "
                         "to the PPO arms' n_steps*n_envs by default, for a fair cadence "
                         "comparison across algorithms).")
    p.add_argument("--out_dir", type=str, default="./research/models")
    args = p.parse_args()

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    env_fns = [make_env(args.reward_mode, args.economy_metric, args.seed, i) for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=50.0)

    n_actions = vec_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.action_noise_sigma * np.ones(n_actions))

    algo_cls = DDPG if args.algo == "ddpg" else TD3
    model = algo_cls(
        "MlpPolicy", vec_env,
        learning_rate=args.lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
        learning_starts=args.learning_starts, action_noise=action_noise,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log=os.path.join(args.out_dir, "..", "runs"),
        seed=args.seed, verbose=1,
    )

    callbacks = [CheckpointCallback(save_freq=200_000, save_path=run_dir, name_prefix="checkpoint")]
    lagrangian_cb = None
    if args.reward_mode == "lagrangian":
        lagrangian_cb = LagrangianCallback(
            constraint_names=["g1_util", "g2_class", "g3_geom"],
            etas={"g1_util": args.eta_util, "g2_class": args.eta_class, "g3_geom": args.eta_geom},
            budgets={"g1_util": 0.0, "g2_class": 0.0, "g3_geom": 0.0},
            lambda_max=args.lambda_max, update_freq=args.lagrangian_update_freq, verbose=1,
        )
        callbacks.append(lagrangian_cb)

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks), tb_log_name=args.run_name)

    model.save(os.path.join(run_dir, "final_model"))
    vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    if lagrangian_cb is not None:
        lagrangian_cb.get_history_dataframe().to_csv(os.path.join(run_dir, "lagrange_history.csv"), index=False)
    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Done. Saved to {run_dir}")


if __name__ == "__main__":
    main()
