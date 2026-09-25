"""
research/scripts/train_reparam_ppo.py
================================================================
Real PPO (stable-baselines3), NOT the numpy-ES pilot, trained on the
one-shot reparameterized action space (HSSReparamEnv). This is the
actual H1 experiment -- run this on real compute (GPU or multi-core
CPU with torch installed; this dev sandbox could not reliably install
torch: see research/results/H1_PILOT_SUMMARY.md). Same PPO
hyperparameters as train.py's incumbent arm so the comparison isolates
the action-space change (per the user's fairness requirement: same
design space/constraints/contexts/objective/eval procedure, clearly
reported budget) -- only --timesteps is recommended lower (see below)
because each episode is 1 step here instead of up to 40, so the same
number of GRADIENT UPDATES needs far fewer env timesteps; report BOTH
env-timesteps and EC3-evaluation-count so the budget comparison to GA
and to the incumbent PPO arm stays apples-to-apples on inference cost,
not just on env steps.

USAGE
------
  python research/scripts/train_reparam_ppo.py --economy_metric cost \\
      --run_name reparam_ppo_cost --seed 42 --timesteps 300000

  # 5-seed replication, same as the incumbent arm:
  for s in 42 43 44 45 46; do
    python research/scripts/train_reparam_ppo.py --economy_metric cost \\
        --run_name reparam_ppo_cost_seed$s --seed $s --timesteps 300000
  done

Then evaluate with the SAME evaluate.py harness the incumbent arm uses
(HSSReparamEnv is registered the same way; evaluate.py's model.predict
loop works unchanged since action_space/observation_space match
HSSBeamEnv) -- do NOT write a second bespoke evaluator for this, to
avoid any chance of the eval procedure silently drifting between arms.
================================================================
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from research.envs.hss_reparam_env import HSSReparamEnv
from research.envs.hss_env import ECONOMY_METRICS


def make_env(economy_metric, seed, rank):
    def _init():
        env = HSSReparamEnv(reward_mode="feasibility_gated", economy_metric=economy_metric)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--economy_metric", choices=ECONOMY_METRICS, default="cost")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    # One-shot episodes -> far more gradient signal per env-timestep than
    # the incumbent's up-to-40-step episodes. Start at ~300k and check the
    # eval-gap-vs-timesteps curve (mirror e4_curve_* in results/) before
    # assuming 1M steps is needed; it may plateau far earlier.
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--n_epochs", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)  # irrelevant for 1-step episodes but kept for PPO API parity
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_range", type=float, default=0.15)
    p.add_argument("--ent_coef", type=float, default=0.03)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--out_dir", type=str, default="./research/models")
    args = p.parse_args()

    np.random.seed(args.seed)
    env_fns = [make_env(args.economy_metric, args.seed, i) for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    model = PPO(
        "MlpPolicy", vec_env, verbose=1, seed=args.seed,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        tensorboard_log=os.path.join(args.out_dir, "tb", args.run_name),
    )

    ckpt = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                               save_path=os.path.join(args.out_dir, args.run_name),
                               name_prefix="ckpt")
    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=False)

    out_path = os.path.join(args.out_dir, f"{args.run_name}_final.zip")
    model.save(out_path)
    print(f"saved final model -> {out_path}")
    print(f"total env-timesteps={args.timesteps}  (== EC3-decode-and-bisect calls, "
          f"since each episode is 1 step); report this alongside eval-time EC3 "
          f"evals/design for a fair budget comparison to GA and the incumbent PPO arm.")


if __name__ == "__main__":
    main()
