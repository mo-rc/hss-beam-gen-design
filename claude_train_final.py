"""
================================================================
train.py
----------------------------------------------------------------
PPO Training Script — Generative Design of HSS Optimal Beam
HKU Research Project

USAGE:
    python train.py                        # default config
    python train.py --timesteps 2000000    # longer run
    python train.py --run-name my_exp      # custom name
    python train.py --no-storey-scaling    # disable storey load

MONITOR (live):
    tensorboard --logdir ./runs

OUTPUT:
    runs/<run_name>/                       # TensorBoard logs
    models/<run_name>/best_model.zip       # best checkpoint
    models/<run_name>/final_model.zip      # end-of-training
    models/<run_name>/training_config.json # reproducibility
================================================================
"""

import os
import json
import argparse
import time
import datetime
import numpy as np

# ── Stable-Baselines3 ─────────────────────────────────────────
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor

# ── TensorBoard ───────────────────────────────────────────────
from torch.utils.tensorboard import SummaryWriter

# ── Environment ───────────────────────────────────────────────
from env.high_rise_generative_env_claude_final import HighRiseGenerativeEnv


# ================================================================
# ARGUMENT PARSER
# ================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="PPO training — HSS Beam Generative Design"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_000_000,
        help="Total training timesteps (default: 1_000_000)"
    )
    parser.add_argument(
        "--n-envs", type=int, default=8,
        help="Number of parallel environments (default: 8)"
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Run name for logs/models. Auto-generated if not set."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--ltb-factor", type=float, default=0.25,
        help="LTB restraint factor Lcr/L (default: 0.25)"
    )
    parser.add_argument(
        "--sls-factor", type=float, default=0.50,
        help="SLS load factor psi (default: 0.50)"
    )
    parser.add_argument(
        "--no-storey-scaling", action="store_true",
        help="Disable storey-based load scaling in reset()"
    )
    # PPO hyperparameters  [EXP9 FIXES + EXP11 TUNING]
    # [EXP11] Conservative retuning for stability and exploration
    parser.add_argument("--lr",          type=float, default=3e-4)      # increased from 5e-4
    parser.add_argument("--n-steps",     type=int,   default=1024)      # reduced from 2048 for more updates
    parser.add_argument("--batch-size",  type=int,   default=256)       # smaller batches for stability
    parser.add_argument("--n-epochs",    type=int,   default=8)        # more epochs per update
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--gae-lambda",  type=float, default=0.95)
    parser.add_argument("--clip-range",  type=float, default=0.15)      # slightly larger clip
    parser.add_argument("--ent-coef",    type=float, default=0.03)      # higher entropy for exploration
    parser.add_argument("--vf-coef",     type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser.parse_args()


# ================================================================
# ENVIRONMENT FACTORY
# ================================================================
def make_env(
    ltb_factor: float,
    sls_factor: float,
    use_storey: bool,
    seed: int,
    rank: int,
):
    """Returns a callable that creates a monitored env instance."""
    def _init():
        env = HighRiseGenerativeEnv(
            use_storey_load_scaling=use_storey,
            include_zg_in_mcr=False,
            sls_load_factor=sls_factor,
            ltb_restraint_factor=ltb_factor,
        )
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ================================================================
# CUSTOM TENSORBOARD CALLBACK
# Logs EC3-specific metrics (utilisation, mass, chi_lt, CO2…)
# that SB3's default logger doesn't capture.
# ================================================================
class EC3TensorBoardCallback(BaseCallback):
    """
    Writes per-episode engineering metrics to TensorBoard.

    Tracked scalars (under 'ec3/' prefix):
        utilization          — governing util ratio (target 0.90–1.05)
        mass_kg              — beam mass in kg
        cost                 — total lifecycle cost
        co2_kg               — total CO2 equivalent
        chi_lt               — LTB reduction factor
        section_class        — EC3 section class (1–4)
        reward_economy       — economy sub-reward
        reward_utilization   — utilisation sub-reward
        reward_stability     — LTB stability sub-reward
        penalty_feasibility  — feasibility penalty
        penalty_underutil    — underutilisation penalty
        success_rate         — fraction of episodes with util in [0.90,1.05]
        mean_episode_length  — mean steps per episode
    """

    def __init__(self, writer: SummaryWriter, verbose: int = 0):
        super().__init__(verbose)
        self.writer = writer
        self.episode_count = 0

        # Buffers — filled from info dict at episode end
        self._ep_utils:    list = []
        self._ep_masses:   list = []
        self._ep_costs:    list = []
        self._ep_co2s:     list = []
        self._ep_chi_lts:  list = []
        self._ep_classes:  list = []
        self._ep_lengths:  list = []
        self._ep_rewards:  list = []

        # Reward sub-term buffers
        self._ep_r_economy:   list = []
        self._ep_r_util:      list = []
        self._ep_r_stab:      list = []
        self._ep_p_feas:      list = []
        self._ep_p_under:     list = []

    def _on_step(self) -> bool:
        # SB3 provides `infos` and `dones` for every parallel env
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.episode_count += 1

                util     = info.get("utilization", 0.0)
                mass     = info.get("mass", 0.0)
                cost     = info.get("cost", 0.0)
                co2      = info.get("co2", 0.0)
                chi_lt   = info.get("chi_lt", 0.0)
                ec3      = info.get("ec3", {})
                sec_cls  = ec3.get("section_class", 0) if ec3 else 0
                r_terms  = info.get("reward_terms", {})

                ep_len   = info.get("episode", {}).get("l", 0)
                ep_rew   = info.get("episode", {}).get("r", 0.0)

                self._ep_utils.append(util)
                self._ep_masses.append(mass)
                self._ep_costs.append(cost)
                self._ep_co2s.append(co2)
                self._ep_chi_lts.append(chi_lt)
                self._ep_classes.append(sec_cls)
                self._ep_lengths.append(ep_len)
                self._ep_rewards.append(ep_rew)

                self._ep_r_economy.append(r_terms.get("economy_reward",      0.0))
                self._ep_r_util.append(   r_terms.get("utilization_reward",  0.0))
                self._ep_r_stab.append(   r_terms.get("stability_reward",    0.0))
                self._ep_p_feas.append(   r_terms.get("feasibility_penalty", 0.0))
                self._ep_p_under.append(  r_terms.get("underutil_penalty",   0.0))

        # Flush every 100 episodes (balances resolution vs overhead)
        if self.episode_count > 0 and self.episode_count % 100 == 0:
            self._flush()

        return True

    def _flush(self):
        step = self.num_timesteps

        def mean(lst): return float(np.mean(lst)) if lst else 0.0
        def std(lst):  return float(np.std(lst))  if lst else 0.0

        utils = self._ep_utils

        # Success rate: util in [0.90, 1.05] AND class < 4
        success = [
            1 for u, c in zip(self._ep_utils, self._ep_classes)
            if 0.90 <= u <= 1.05 and c < 4
        ]
        success_rate = len(success) / max(len(utils), 1)

        # Engineering metrics
        self.writer.add_scalar("ec3/utilization_mean",   mean(utils),              step)
        self.writer.add_scalar("ec3/utilization_std",    std(utils),               step)
        self.writer.add_scalar("ec3/mass_kg_mean",       mean(self._ep_masses),    step)
        self.writer.add_scalar("ec3/cost_mean",          mean(self._ep_costs),     step)
        self.writer.add_scalar("ec3/co2_kg_mean",        mean(self._ep_co2s),      step)
        self.writer.add_scalar("ec3/chi_lt_mean",        mean(self._ep_chi_lts),   step)
        self.writer.add_scalar("ec3/section_class_mean", mean(self._ep_classes),   step)
        self.writer.add_scalar("ec3/success_rate",       success_rate,             step)
        self.writer.add_scalar("ec3/mean_episode_length",mean(self._ep_lengths),   step)
        self.writer.add_scalar("ec3/episode_reward_mean",mean(self._ep_rewards),   step)

        # Reward sub-terms
        self.writer.add_scalar("reward_terms/economy",        mean(self._ep_r_economy), step)
        self.writer.add_scalar("reward_terms/utilization",    mean(self._ep_r_util),    step)
        self.writer.add_scalar("reward_terms/stability",      mean(self._ep_r_stab),    step)
        self.writer.add_scalar("reward_terms/feas_penalty",   mean(self._ep_p_feas),    step)
        self.writer.add_scalar("reward_terms/underutil_penalty", mean(self._ep_p_under),step)

        # Clear buffers
        self._ep_utils.clear();   self._ep_masses.clear()
        self._ep_costs.clear();   self._ep_co2s.clear()
        self._ep_chi_lts.clear(); self._ep_classes.clear()
        self._ep_lengths.clear(); self._ep_rewards.clear()
        self._ep_r_economy.clear(); self._ep_r_util.clear()
        self._ep_r_stab.clear();  self._ep_p_feas.clear()
        self._ep_p_under.clear()

    def _on_training_end(self):
        # Final flush for any remaining episodes
        if self._ep_utils:
            self._flush()
        self.writer.flush()


# ================================================================
# MAIN TRAINING FUNCTION
# ================================================================
def train(args):

    # ── Run name & directories ────────────────────────────────
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = args.run_name or f"ppo_hss_{timestamp}"

    log_dir   = os.path.join("runs",   run_name)
    model_dir = os.path.join("models", run_name)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 64)
    print("  PPO Training — HSS Beam Generative Design")
    print("=" * 64)
    print(f"  Run name          : {run_name}")
    print(f"  TensorBoard logs  : {log_dir}")
    print(f"  Model checkpoints : {model_dir}")
    print(f"  Total timesteps   : {args.timesteps:,}")
    print(f"  Parallel envs     : {args.n_envs}")
    print(f"  Seed              : {args.seed}")
    print(f"  ltb_factor        : {args.ltb_factor}")
    print(f"  sls_factor        : {args.sls_factor}")
    print(f"  storey scaling    : {not args.no_storey_scaling}")
    print("=" * 64)

    # ── Save config for reproducibility ──────────────────────
    config = vars(args)
    config["run_name"]  = run_name
    config["timestamp"] = timestamp
    with open(os.path.join(model_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Vectorised training environments ─────────────────────
    use_storey = not args.no_storey_scaling

    train_env = SubprocVecEnv([
        make_env(args.ltb_factor, args.sls_factor, use_storey, args.seed, i)
        for i in range(args.n_envs)
    ])
    train_env = VecMonitor(train_env)
    # [EXP9 FIX 1] Increased clip_reward from 10.0 to 50.0
    # With norm_reward=True, clipping to ±10 was too aggressive and flattened
    # the reward signal. Increased to 50 to allow more dynamic range while
    # still preventing rare outlier episodes from destabilizing training.
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=50.0,        # was 10.0 — too aggressive
        gamma=args.gamma,
    )

    # ── Evaluation environment (single, fixed seed) ───────────
    eval_env = SubprocVecEnv([
        make_env(args.ltb_factor, args.sls_factor, use_storey, args.seed + 999, 0)
    ])
    eval_env = VecMonitor(eval_env)
    # Eval env shares normalisation stats from train_env (training=False)
    # [EXP9 FIX 2] Increased clip_obs to 50.0 for consistency with train_env
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,   # do NOT normalise eval rewards — we want true values
        clip_obs=10.0,       
        training=False,
    )

    # ── TensorBoard writer ────────────────────────────────────
    writer = SummaryWriter(log_dir=log_dir)

    # Log hyperparameters as a TensorBoard hparam entry
    writer.add_hparams(
        hparam_dict={
            "lr":             args.lr,
            "n_steps":        args.n_steps,
            "batch_size":     args.batch_size,
            "n_epochs":       args.n_epochs,
            "gamma":          args.gamma,
            "gae_lambda":     args.gae_lambda,
            "clip_range":     args.clip_range,
            "ent_coef":       args.ent_coef,
            "ltb_factor":     args.ltb_factor,
            "sls_factor":     args.sls_factor,
            "n_envs":         args.n_envs,
        },
        metric_dict={"hparam/placeholder": 0.0},
    )

    # ── Callbacks ─────────────────────────────────────────────

    # 1. EC3 custom metrics → TensorBoard
    ec3_callback = EC3TensorBoardCallback(writer=writer, verbose=0)

    # 2. Periodic evaluation (every 25k steps, keep best model) [EXP9 FIX 3]
    # Increased frequency from 50k to 25k for better model selection.
    # More frequent evals help catch good models before they diverge.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(25_000 // args.n_envs, 1),  # was 50_000
        n_eval_episodes=20,
        deterministic=True,
        render=False,
        verbose=1,
    )

    # 3. Checkpoint every 200k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(200_000 // args.n_envs, 1),
        save_path=model_dir,
        name_prefix="checkpoint",
        verbose=1,
    )

    callback = CallbackList([ec3_callback, eval_callback, checkpoint_callback])

    # ── PPO Model ─────────────────────────────────────────────
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],  # [EXP9 FIX 4] Enlarged from [256, 256, 128]
        ),
        tensorboard_log=log_dir,
        seed=args.seed,
        verbose=1,
    )

    print(f"\n  Policy network : [256, 256, 128] (actor + critic)")  
    print(f"  Observation    : {train_env.observation_space.shape}")
    print(f"  Action         : {train_env.action_space.shape}")
    print(f"\n  Starting training...")
    print(f"  Monitor: tensorboard --logdir {log_dir}\n")

    # ── Train ─────────────────────────────────────────────────
    t0 = time.time()

    # Share obs normalisation stats with eval_env so evaluation is consistent
    eval_env.obs_rms = train_env.obs_rms

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=run_name,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    elapsed = time.time() - t0

    # ── Save final model ──────────────────────────────────────
    final_path = os.path.join(model_dir, "final_model")
    model.save(final_path)
    # Save normalisation stats — REQUIRED to load model correctly later
    vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    train_env.save(vecnorm_path)
    print(f"  VecNormalize stats : {vecnorm_path}")

    print(f"\n{'='*64}")
    print(f"  Training complete in {elapsed/60:.1f} min")
    print(f"  Final model saved  : {final_path}.zip")
    print(f"  Best model saved   : {model_dir}/best_model.zip")
    print(f"  TensorBoard logs   : {log_dir}")
    print(f"  Run:  tensorboard --logdir runs/")
    print(f"{'='*64}\n")

    writer.close()
    train_env.close()
    eval_env.close()

    return model, run_name


# ================================================================
# ENTRY POINT
# ================================================================
if __name__ == "__main__":
    args = parse_args()
    train(args)