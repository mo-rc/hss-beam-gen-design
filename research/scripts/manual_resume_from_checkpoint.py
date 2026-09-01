"""
research/scripts/manual_resume_from_checkpoint.py
================================================================
For resuming a run that was started with train.py's checkpoints
(checkpoint_{steps}_steps.zip + a matching vecnormalize file), NOT
resume_training.py's own resume_latest.zip/resume_state.json
bookkeeping. Works directly with SB3's PPO.load(..., env=...) and
reset_num_timesteps=False, which only needs the checkpoint's own
saved num_timesteps -- it does not need any external state file.

CORRECTNESS NOTES (read before running):
- SB3 stores `num_timesteps` inside the saved model .zip itself. When
  you call PPO.load(checkpoint_path), the loaded model already knows
  it's at e.g. 1,000,000 steps. `model.learn(total_timesteps=1300000,
  reset_num_timesteps=False)` then runs ONLY the remaining 300,000
  steps -- it does not restart from 0. This is a property of SB3's
  API, not something specific to resume_training.py.
- You MUST reconstruct the environment identically to how it was
  built during the original training call (same reward_mode,
  economy_metric, economy_reward_mode, n_envs) or the loaded policy
  will be evaluated against a mismatched observation/reward
  distribution. Check the --economy_reward_mode default below against
  what you actually used.
- The VecNormalize running statistics (obs mean/std, reward scale)
  matter as much as the policy weights -- loading the checkpoint
  without the matching VecNormalize file will silently corrupt
  training (the policy will see wrongly-scaled observations). This
  script requires the vecnormalize path explicitly, on purpose --
  there's no safe default.
- If you used log_std_anneal, this script reconstructs
  LogStdAnnealCallback with total_timesteps=--new_total_timesteps
  (the NEW grand total, not the remaining amount -- this callback's
  own `total_timesteps` argument IS the grand total, since it's used
  only to compute a progress fraction, not passed to SB3's .learn()).
  Per the schedule math discussed in chat: resuming at
  num_timesteps=1,000,000 with new_total_timesteps=1,300,000 gives
  frac=0.769 at the moment of resume, which is already past
  anneal_start_frac=0.5 -- so the ceiling is recomputed immediately at
  that point in the NEW schedule (safely, since it clamps a maximum
  and starts above the current actual std) and continues decaying to
  --log_std_ceiling_end by the new total.
- CORRECTED (this was wrong in an earlier version of this script, and
  caused a real overshoot -- see chat): SB3's model.learn(total_timesteps=X,
  reset_num_timesteps=False) treats X as an INCREMENT added on top of
  the loaded model's existing num_timesteps, NOT an absolute target.
  This script now computes `remaining_timesteps = new_total_timesteps -
  model.num_timesteps` and passes THAT to .learn(), so
  --new_total_timesteps keeps its intuitive "grand total" meaning at
  the CLI level while still calling SB3's API correctly underneath.
================================================================
"""
import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.envs.hss_env import HSSBeamEnv
from research.algo.lagrangian import LagrangianCallback
from research.algo.log_std_anneal import LogStdAnnealCallback


def make_env_fn(reward_mode, economy_metric, economy_reward_mode, seed, rank):
    def _init():
        env = HSSBeamEnv(reward_mode=reward_mode, economy_metric=economy_metric,
                          economy_reward_mode=economy_reward_mode)
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", required=True,
                    help="Path to the SB3 checkpoint .zip, e.g. "
                         "research/models/gated_cost_merged_seed42/checkpoint_1000000_steps.zip "
                         "(or final_model.zip if that's what you have at 1M).")
    p.add_argument("--vecnormalize_path", required=True,
                    help="Path to the matching VecNormalize .pkl saved at the same step count.")
    p.add_argument("--new_total_timesteps", type=int, required=True,
                    help="New GRAND TOTAL (absolute), e.g. 1300000 -- not the remaining amount.")
    p.add_argument("--run_name", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--reward_mode", default="feasibility_gated")
    p.add_argument("--economy_metric", default="cost")
    p.add_argument("--economy_reward_mode", default="log_relative")
    p.add_argument("--log_std_anneal", action="store_true", default=True)
    p.add_argument("--log_std_ceiling_start", type=float, default=8.0)
    p.add_argument("--log_std_ceiling_end", type=float, default=1.0)
    p.add_argument("--log_std_anneal_start_frac", type=float, default=0.5)
    p.add_argument("--out_dir", default="research/models")
    args = p.parse_args()

    run_dir = os.path.join(args.out_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    env_fns = [make_env_fn(args.reward_mode, args.economy_metric, args.economy_reward_mode,
                            args.seed, i) for i in range(args.n_envs)]
    raw_vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    vec_env = VecNormalize.load(args.vecnormalize_path, raw_vec_env)
    vec_env.training = True  # keep updating running stats during the extra steps

    model = PPO.load(args.checkpoint_path, env=vec_env)
    print(f"Loaded checkpoint at num_timesteps={model.num_timesteps}")
    assert model.num_timesteps < args.new_total_timesteps, (
        "new_total_timesteps must be greater than the checkpoint's own step count -- "
        f"checkpoint is already at {model.num_timesteps}."
    )

    callbacks = []
    if args.reward_mode == "lagrangian":
        callbacks.append(LagrangianCallback())  # only if your training used this; omit otherwise
    if args.log_std_anneal:
        callbacks.append(LogStdAnnealCallback(
            ceiling_start=args.log_std_ceiling_start,
            ceiling_end=args.log_std_ceiling_end,
            anneal_start_frac=args.log_std_anneal_start_frac,
            total_timesteps=args.new_total_timesteps,  # NEW grand total, per the note above
            verbose=1,
        ))

    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=run_dir,
        name_prefix="checkpoint",
    )
    callbacks.append(checkpoint_cb)

    # IMPORTANT: SB3 treats the `total_timesteps` argument to .learn() as an
    # INCREMENT added on top of the loaded model's existing num_timesteps
    # when reset_num_timesteps=False -- it is NOT an absolute target. Passing
    # args.new_total_timesteps directly here would train to
    # (args.new_total_timesteps + model.num_timesteps), badly overshooting
    # the intended grand total. Compute the remaining delta explicitly so
    # --new_total_timesteps keeps its intuitive "grand total" meaning at the
    # CLI level.
    remaining_timesteps = args.new_total_timesteps - model.num_timesteps
    print(f"Requesting {remaining_timesteps} additional steps "
          f"(model.learn(total_timesteps={remaining_timesteps}, reset_num_timesteps=False) "
          f"-> SB3 will stop at {model.num_timesteps + remaining_timesteps} = args.new_total_timesteps)")

    model.learn(
        total_timesteps=remaining_timesteps,  # delta, NOT args.new_total_timesteps -- see note above
        reset_num_timesteps=False,            # resumes, does not restart
        callback=CallbackList(callbacks),
        tb_log_name=args.run_name,
    )

    model.save(os.path.join(run_dir, "final_model"))
    vec_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    print(f"Done. Final model at {model.num_timesteps} steps saved to {run_dir}")


if __name__ == "__main__":
    main()