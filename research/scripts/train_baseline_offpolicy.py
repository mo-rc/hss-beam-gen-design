"""
research/scripts/train_baseline_offpolicy.py
================================================================
Off-policy continuous-control baselines (DDPG, TD3, SAC) on the
IDENTICAL environment/reward used for the PPO arms
(research/envs/hss_env.py). This answers "is PPO actually the right
algorithmic choice, or would any continuous-control method do?" --
previously an unsupported assertion, now a real comparison.

WHY THESE THREE
---------------
DDPG alone is a weak baseline by 2020s standards (well-documented
overestimation-bias issues). TD3 is DDPG's direct twin-critic /
delayed-policy fix. SAC is the current strongest off-policy continuous-
control method and is the arm most likely to actually win here: its
entropy temperature is auto-tuned, so unlike PPO (whose exploration is
governed by a fixed ent_coef plus an optional hand-set log_std ceiling)
it adapts its own action variance to the reward landscape. The E3 audit
found the PPO policy pinned at its annealed log_std ceiling for 100% of
the final 30% of training, i.e. asking for more action variance than the
schedule allowed -- SAC is the principled algorithmic answer to exactly
that symptom, which is why it belongs in this comparison rather than
being an arbitrary third algorithm.

ENVIRONMENT PARITY (audit fix)
-------------------------------
Every environment-shaping parameter that research/scripts/train.py
exposes is exposed here too and passed EXPLICITLY. These previously
matched only by coincidence of default values: this file constructed
`HSSBeamEnv(reward_mode=..., economy_metric=...)` and inherited
`ltb_restraint_factor`, `sls_load_factor`, `economy_reward_mode`,
`max_steps` and `enforce_rolled_manufacturability` from the class
defaults, while train.py passed several of them explicitly. Any future
change to a class default would then have silently made the off-policy
arms solve a DIFFERENT design problem than the PPO arms while every
config file still looked comparable. research/tests/test_algorithm_parity.py
asserts the two entry points now build identical environments.

`--reward_mode` is REQUIRED (it previously defaulted to "lagrangian"
while the headline PPO arm uses "feasibility_gated" -- a silent
objective mismatch; the existing research/models/ddpg_cost run has
exactly that mismatch and is NOT comparable to the corrected-cost PPO
results).

BUDGET PARITY (audit fix)
--------------------------
SB3's off-policy default is train_freq=1, gradient_steps=1, meaning ONE
gradient step per call of collect_rollouts -- which advances the clock by
`n_envs` environment timesteps. So n_envs silently divides the number of
gradient updates for a fixed timestep budget. Measured on this env:

    DDPG n_envs=1 -> 900 updates / 1000 env steps
    DDPG n_envs=2 -> 450 updates / 1000 env steps   (the old default)
    DDPG n_envs=4 -> 225 updates / 1000 env steps

The old `--n_envs 2` default therefore handed DDPG/TD3 HALF the learning
per environment interaction that the same algorithm gets in its standard
single-env configuration, for the same reported "1M timesteps". n_envs
now defaults to 1, and `gradient_steps` defaults to `n_envs` so the
update-per-env-step ratio is invariant if n_envs is raised for speed.

Note also that a plain NormalActionNoise emits one (6,) sample that
BROADCASTS across all envs, so n_envs>1 gives every env identical
exploration noise. n_envs=1 avoids that too.

USAGE
------
  python research/scripts/train_baseline_offpolicy.py --algo ddpg \\
      --reward_mode feasibility_gated --run_name baseline_ddpg_seed42 --seed 42
  python research/scripts/train_baseline_offpolicy.py --algo td3 \\
      --reward_mode feasibility_gated --run_name baseline_td3_seed42 --seed 42
  python research/scripts/train_baseline_offpolicy.py --algo sac \\
      --reward_mode feasibility_gated --run_name baseline_sac_seed42 --seed 42
================================================================
"""

import argparse
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import DDPG, TD3, SAC
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
import numpy as np

from research.envs.hss_env import HSSBeamEnv, REWARD_MODES, ECONOMY_METRICS
from research.algo.lagrangian import LagrangianCallback

ALGOS = {"ddpg": DDPG, "td3": TD3, "sac": SAC}


def make_env(reward_mode, economy_metric, lagrange_init, ltb_factor, sls_factor,
             economy_reward_mode, max_steps, enforce_rolled, seed, rank):
    """Mirrors research/scripts/train.py:make_env exactly. Every
    environment-shaping argument is passed explicitly so the off-policy
    arms cannot silently drift onto a different design problem."""
    def _init():
        env = HSSBeamEnv(
            reward_mode=reward_mode, economy_metric=economy_metric,
            lagrange_init=lagrange_init,
            ltb_restraint_factor=ltb_factor, sls_load_factor=sls_factor,
            economy_reward_mode=economy_reward_mode,
            max_steps=max_steps,
            enforce_rolled_manufacturability=enforce_rolled,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=sorted(ALGOS), required=True)
    # REQUIRED (audit fix): previously defaulted to "lagrangian" while the
    # headline PPO arm trains under "feasibility_gated".
    p.add_argument("--reward_mode", choices=REWARD_MODES, required=True)
    p.add_argument("--economy_metric", choices=ECONOMY_METRICS, default="cost")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    # n_envs=1 (audit fix): SB3 off-policy does gradient_steps updates per
    # collect_rollouts call, which advances n_envs timesteps -- n_envs>1
    # silently divides the update count for a fixed timestep budget, and
    # broadcasts identical action noise across envs.
    p.add_argument("--n_envs", type=int, default=1)
    p.add_argument("--train_freq", type=int, default=1,
                   help="collect_rollouts length in STEPS (SB3 default 1). Kept "
                        "step-based, never episode-based, so the update cadence "
                        "is independent of this env's variable episode length.")
    p.add_argument("--gradient_steps", type=int, default=None,
                   help="Gradient updates per collect_rollouts call. Defaults to "
                        "n_envs * train_freq, which keeps the updates-per-"
                        "environment-step ratio invariant to --n_envs.")
    p.add_argument("--lr", type=float, default=None,
                   help="Defaults to the algorithm's own tuned value "
                        "(1e-3 DDPG/TD3, 3e-4 SAC).")
    p.add_argument("--buffer_size", type=int, default=1_000_000,
                   help="SB3 default. The old 200k value discarded 80%% of a 1M-step "
                        "run's experience; the observation is 26-dim so a full-size "
                        "buffer costs well under 1 GB.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Kept equal to the PPO arms' gamma.")
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--action_noise_sigma", type=float, default=0.2,
                   help="DDPG/TD3 only. SAC explores via its entropy term and "
                        "takes no external action noise.")
    p.add_argument("--net_arch", type=int, nargs="+", default=[256, 256, 128],
                   help="Kept identical to the PPO arms.")
    # --- environment-shaping args, mirrored from train.py (audit fix) ---
    p.add_argument("--ltb_factor", type=float, default=0.40)
    p.add_argument("--sls_factor", type=float, default=0.50)
    p.add_argument("--max_steps", type=int, default=40)
    p.add_argument("--economy_reward_mode", choices=["linear", "log_relative"], default="linear")
    p.add_argument("--no_rolled_manufacturability", action="store_true",
                   help="Disable E1 manufacturability-aware costing (reproduces "
                        "pre-correction numbers). Leave OFF for all current work.")
    # --- Lagrangian-only settings ---
    p.add_argument("--eta_util", type=float, default=5.0)
    p.add_argument("--eta_class", type=float, default=5.0)
    p.add_argument("--eta_geom", type=float, default=2.0)
    p.add_argument("--lambda_max", type=float, default=200.0)
    p.add_argument("--budget_util", type=float, default=0.0)
    p.add_argument("--lagrangian_update_freq", type=int, default=8192,
                   help="Environment timesteps between dual-ascent updates. "
                        "Default matches the PPO arms' n_steps*n_envs (1024*8).")
    p.add_argument("--checkpoint_every", type=int, default=200_000,
                   help="Environment timesteps between checkpoints. Matches the "
                        "PPO arms so learning curves share a step grid.")
    p.add_argument("--out_dir", type=str, default="./research/models")
    args = p.parse_args()

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    lagrange_init = dict(g1_util=0.0, g2_class=0.0, g3_geom=0.0)
    enforce_rolled = not args.no_rolled_manufacturability

    env_fns = [make_env(args.reward_mode, args.economy_metric, lagrange_init,
                        args.ltb_factor, args.sls_factor, args.economy_reward_mode,
                        args.max_steps, enforce_rolled, args.seed, i)
               for i in range(args.n_envs)]
    vec_env = DummyVecEnv(env_fns)
    # Identical normalisation to the PPO arms. norm_obs stays False: the
    # evaluation harness loads the bare policy without VecNormalize, so
    # observation normalisation would silently break evaluation.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                           clip_reward=50.0, gamma=args.gamma)

    n_actions = vec_env.action_space.shape[-1]
    action_noise = None
    if args.algo in ("ddpg", "td3"):
        base_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                       sigma=args.action_noise_sigma * np.ones(n_actions))
        # A bare NormalActionNoise emits one (6,) sample that broadcasts across
        # all envs, giving every env identical exploration. Wrap it when n_envs>1.
        action_noise = base_noise if args.n_envs == 1 else VectorizedActionNoise(base_noise, args.n_envs)

    gradient_steps = args.gradient_steps if args.gradient_steps is not None \
        else args.n_envs * args.train_freq
    lr = args.lr if args.lr is not None else (3e-4 if args.algo == "sac" else 1e-3)

    kwargs = dict(
        learning_rate=lr, buffer_size=args.buffer_size, batch_size=args.batch_size,
        learning_starts=args.learning_starts, gamma=args.gamma,
        train_freq=args.train_freq, gradient_steps=gradient_steps,
        policy_kwargs=dict(net_arch=list(args.net_arch)),
        tensorboard_log=os.path.join(args.out_dir, "..", "runs"),
        seed=args.seed, verbose=1,
    )
    if action_noise is not None:
        kwargs["action_noise"] = action_noise

    model = ALGOS[args.algo]("MlpPolicy", vec_env, **kwargs)

    # save_freq is counted in CALLS, and one call advances n_envs timesteps --
    # divide so the checkpoint grid is in environment timesteps, exactly as
    # research/scripts/train.py does.
    callbacks = [CheckpointCallback(
        save_freq=max(args.checkpoint_every // args.n_envs, 1),
        save_path=run_dir, name_prefix="checkpoint", save_vecnormalize=True)]

    lagrangian_cb = None
    if args.reward_mode == "lagrangian":
        lagrangian_cb = LagrangianCallback(
            constraint_names=["g1_util", "g2_class", "g3_geom"],
            etas={"g1_util": args.eta_util, "g2_class": args.eta_class, "g3_geom": args.eta_geom},
            budgets={"g1_util": args.budget_util, "g2_class": 0.0, "g3_geom": 0.0},
            lambda_max=args.lambda_max, update_freq=args.lagrangian_update_freq, verbose=1,
        )
        callbacks.append(lagrangian_cb)

    model.learn(total_timesteps=args.timesteps, callback=CallbackList(callbacks),
                tb_log_name=args.run_name)

    model.save(os.path.join(run_dir, "final_model"))
    vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    if lagrangian_cb is not None:
        lagrangian_cb.get_history_dataframe().to_csv(
            os.path.join(run_dir, "lagrange_history.csv"), index=False)

    cfg = vars(args).copy()
    cfg["resolved_lr"] = lr
    cfg["resolved_gradient_steps"] = gradient_steps
    cfg["enforce_rolled_manufacturability"] = enforce_rolled
    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Done. Saved to {run_dir}")


if __name__ == "__main__":
    main()
