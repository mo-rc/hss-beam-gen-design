"""
research/scripts/resume_training.py
================================================================
Chunked/resumable training driver. Functionally identical to
`train.py` for a single-shot run, but splits `total_timesteps` into
`chunk_timesteps`-sized pieces across repeated invocations of this
script, so a run can be completed across multiple separate process
launches (e.g. because the execution environment enforces a wall-clock
limit shorter than a full run) without corrupting anything that must
stay continuous across the resume boundary:

  - PPO's own step counter / optimizer state / network weights: SB3
    handles this correctly via `PPO.load(...)` + `.learn(...,
    reset_num_timesteps=False)`.
  - VecNormalize's running reward-normalization statistics: saved and
    reloaded explicitly every chunk (`checkpoint_vecnormalize` on save,
    `VecNormalize.load` on resume) -- NOT handled by plain
    `PPO.load()`, and NOT saved by `train.py`'s CheckpointCallback until
    this file's `save_vecnormalize=True` was added.
  - The annealed log_std ceiling (`research/algo/log_std_anneal.py`):
    deliberately takes the GRAND total timesteps as an explicit,
    externally-supplied constant (not read from
    `model._total_timesteps`, which SB3 resets at every `.learn()`
    call to that call's own, chunk-sized budget) specifically so its
    progress fraction stays correct and continuous across resumes. See
    that file's docstring for why this matters -- it is the same bug
    class already documented in this project's history for the
    (different, not-present-in-this-repo) entropy-annealing callback.

NOT YET SUPPORTED: `--reward_mode lagrangian` resumption. The
Lagrangian dual-ascent state (`LagrangianCallback`'s multiplier values)
is not currently checkpointed/restored across chunks, so resuming a
lagrangian run with this script would silently restart its multipliers
from their initial values at every chunk boundary. This script refuses
to run with `--reward_mode lagrangian` until that is added. It is not
needed for the log_std-anneal experiment (feasibility_gated), which is
the only thing this script has been used for so far.

USAGE (run repeatedly with the same args until it prints "run complete"):
------
  python research/scripts/resume_training.py \\
      --reward_mode feasibility_gated --economy_metric cost \\
      --run_name gated_cost_seed42_logstdanneal --seed 42 \\
      --total_timesteps 1000000 --chunk_timesteps 250000 --n_envs 8 \\
      --log_std_anneal --log_std_ceiling_start 8.0 --log_std_ceiling_end 1.0 \\
      --log_std_anneal_start_frac 0.5 --out_dir research/models
================================================================
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from research.envs.hss_env import HSSBeamEnv, REWARD_MODES, ECONOMY_METRICS
from research.envs.hss_catalog_env import HSSBeamCatalogEnv
from research.algo.log_std_anneal import LogStdAnnealCallback


def make_env(env_type, reward_mode, economy_metric, ltb_factor, sls_factor, economy_reward_mode, seed, rank):
    def _init():
        cls = HSSBeamCatalogEnv if env_type == "catalog" else HSSBeamEnv
        env = cls(reward_mode=reward_mode, economy_metric=economy_metric,
                   ltb_restraint_factor=ltb_factor, sls_load_factor=sls_factor,
                   economy_reward_mode=economy_reward_mode)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env_type", choices=["continuous", "catalog"], default="continuous")
    p.add_argument("--reward_mode", choices=REWARD_MODES, required=True)
    p.add_argument("--economy_metric", choices=ECONOMY_METRICS, default="cost")
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--total_timesteps", type=int, required=True, help="GRAND total for the whole run.")
    p.add_argument("--chunk_timesteps", type=int, required=True, help="Steps to run THIS invocation.")
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
    p.add_argument("--log_std_anneal", action="store_true")
    p.add_argument("--log_std_ceiling_start", type=float, default=8.0)
    p.add_argument("--log_std_ceiling_end", type=float, default=1.0)
    p.add_argument("--log_std_anneal_start_frac", type=float, default=0.5)
    p.add_argument("--economy_reward_mode", choices=["linear", "log_relative"], default="linear")
    p.add_argument("--out_dir", type=str, default="./research/models")
    args = p.parse_args()

    if args.reward_mode == "lagrangian":
        raise NotImplementedError(
            "Lagrangian dual-ascent state is not checkpointed across resumes yet -- "
            "see this file's module docstring. Use train.py for a lagrangian run that "
            "fits in a single invocation, or extend this script first."
        )

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    latest_model_path = os.path.join(run_dir, "resume_latest.zip")
    latest_vecnorm_path = os.path.join(run_dir, "resume_latest_vecnormalize.pkl")
    state_path = os.path.join(run_dir, "resume_state.json")

    env_fns = [make_env(args.env_type, args.reward_mode, args.economy_metric,
                         args.ltb_factor, args.sls_factor, args.economy_reward_mode, args.seed, i)
               for i in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)  # n_envs==1 fix: VecNormalize requires VecEnv semantics, not a raw env

    resuming = os.path.exists(latest_model_path)
    if resuming:
        vec_env = VecNormalize.load(latest_vecnorm_path, vec_env)
        model = PPO.load(latest_model_path, env=vec_env)
        print(f"Resumed from {latest_model_path} at num_timesteps={model.num_timesteps}")
    else:
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
        print("Starting fresh run.")

    already_done = model.num_timesteps
    remaining = args.total_timesteps - already_done
    if remaining <= 0:
        print(f"\nrun complete: num_timesteps={already_done} >= total_timesteps={args.total_timesteps}")
        model.save(os.path.join(run_dir, "final_model"))
        vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
        with open(os.path.join(run_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        return

    this_chunk = min(args.chunk_timesteps, remaining)

    callbacks = [CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1),
                                     save_path=run_dir, name_prefix="checkpoint",
                                     save_vecnormalize=True)]
    if args.log_std_anneal:
        callbacks.append(LogStdAnnealCallback(
            total_timesteps=args.total_timesteps,  # grand total -- see module docstring
            ceiling_start=args.log_std_ceiling_start, ceiling_end=args.log_std_ceiling_end,
            anneal_start_frac=args.log_std_anneal_start_frac, verbose=1,
        ))

    model.learn(total_timesteps=this_chunk, reset_num_timesteps=False,
                callback=CallbackList(callbacks), tb_log_name=args.run_name)

    model.save(latest_model_path.replace(".zip", ""))
    vec_env.save(latest_vecnorm_path)
    with open(state_path, "w") as f:
        json.dump(dict(num_timesteps=model.num_timesteps, total_timesteps=args.total_timesteps), f, indent=2)

    print(f"\nChunk done: num_timesteps={model.num_timesteps} / {args.total_timesteps} "
          f"({model.num_timesteps/args.total_timesteps:.1%})")
    if model.num_timesteps >= args.total_timesteps:
        model.save(os.path.join(run_dir, "final_model"))
        vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
        with open(os.path.join(run_dir, "training_config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"run complete. Final model saved to {run_dir}/final_model")
    else:
        print("Re-run the same command to continue.")


if __name__ == "__main__":
    main()
