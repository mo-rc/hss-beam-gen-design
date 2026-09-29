"""Shared training code for pipeline/02_train_ppo.py (PPO, SAC, TD3, DDPG).

Everything that must be IDENTICAL across algorithms lives here, once:

* `resolve_config`  - frozen YAML defaults + explicit overrides -> one flat, validated dict.
* `env_kwargs`      - the environment-shaping arguments. One function for every algorithm,
                      so the arms cannot silently solve different design problems
                      (this replaces the old test_algorithm_parity.py; see tests/).
* `make_env`, `build_model` - environment and model construction.
* `ResumeCallback` + `save_resume_state` / `load_resume_state` - atomic, complete resume state.

Only stable_baselines3 / torch are imported lazily where needed, so config resolution and
`env_kwargs` can be imported and tested without them.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any

import numpy as np
import yaml

ALGOS = ("ppo", "sac", "td3", "ddpg")
OFFPOLICY = ("sac", "td3", "ddpg")
SECTIONS_ALWAYS = ("common", "env", "lagrangian")
LAGRANGE_KEYS = ("g1_util", "g2_class", "g3_geom")

RESUME_MODEL = "resume_latest.zip"
RESUME_VECNORM = "resume_latest_vecnormalize.pkl"
RESUME_BUFFER = "resume_latest_replay_buffer.pkl"
RESUME_STATE = "resume_state.json"


# ----------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(flat: dict, algo: str, overrides: dict[str, Any] | None = None,
                    sets: list[str] | None = None) -> dict:
    """Apply CLI overrides/--set to an already-flat config dict, in place, then return it.

    Unknown keys raise, so a typo can never silently fall back to a default. Shared between
    a fresh run (base = the frozen YAML) and a resumed run (base = the ORIGINAL run's saved
    config, from meta.json) -- resuming must never silently re-pick up a config-file default
    such as n_envs, which is baked into the saved model/VecNormalize shapes.
    """
    pairs = {k: v for k, v in (overrides or {}).items() if v is not None}
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set expects KEY=VALUE, got {s!r}")
        k, v = s.split("=", 1)
        pairs[k.strip()] = yaml.safe_load(v)
    for k, v in pairs.items():
        if k not in flat:
            raise ValueError(f"unknown config key {k!r} for algo {algo!r}; valid keys: {sorted(flat)}")
        if k == "log_std_anneal" and v and algo in OFFPOLICY:
            raise ValueError("log_std_anneal is PPO-only (it clamps PPO's state-independent log_std).")
        flat[k] = v
    if not flat["enforce_rolled_manufacturability"]:
        raise ValueError("enforce_rolled_manufacturability=False reproduces the pre-E1 exploit; "
                         "it is not allowed for any current work.")
    return flat


def resolve_config(config_path: str, algo: str, overrides: dict[str, Any] | None = None,
                   sets: list[str] | None = None) -> dict:
    """Flatten the frozen YAML for one algorithm and apply overrides. Use only for a NEW run;
    to resume, use base_config_from_meta() + apply_overrides() so the actual settings the
    original run trained with (in particular n_envs) are reused, not the file's defaults."""
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}; choose from {ALGOS}")
    raw = load_yaml(config_path)
    algo_section = "offpolicy" if algo in OFFPOLICY else "ppo"
    flat: dict[str, Any] = {}
    for sec in (*SECTIONS_ALWAYS, algo_section):
        for k, v in raw[sec].items():
            if k in flat:
                raise ValueError(f"duplicate config key {k!r} in section {sec!r}")
            flat[k] = v
    if algo in OFFPOLICY:  # anneal is a PPO-only device; keep the key so overrides can be rejected
        flat["log_std_anneal"] = False
    flat["algo"] = algo
    flat = apply_overrides(flat, algo, overrides, sets)
    if algo in OFFPOLICY:
        if flat["gradient_steps"] is None:
            flat["gradient_steps"] = flat["n_envs"] * flat["train_freq"]
        if flat["lr"] is None:
            flat["lr"] = 3e-4 if algo == "sac" else 1e-3
    flat["net_arch"] = list(flat["net_arch"])
    return flat


def base_config_from_meta(prev_meta: dict) -> dict:
    """The exact flat config a previous run trained with, as saved in its meta.json.

    gradient_steps/lr are already resolved (non-null) in a saved config, so resuming never
    re-derives them from a possibly different --n_envs passed on the resume command.
    """
    flat = dict(prev_meta["config"])
    flat["net_arch"] = list(flat["net_arch"])
    return flat


# ----------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------
def env_kwargs(cfg: dict, reward_mode: str) -> dict:
    """EVERY environment-shaping argument, passed explicitly, for every algorithm."""
    return dict(
        reward_mode=reward_mode,
        economy_metric=cfg["economy_metric"],
        lagrange_init=dict.fromkeys(LAGRANGE_KEYS, 0.0),
        ltb_restraint_factor=cfg["ltb_factor"],
        sls_load_factor=cfg["sls_factor"],
        economy_reward_mode=cfg["economy_reward_mode"],
        max_steps=cfg["max_steps"],
        enforce_rolled_manufacturability=cfg["enforce_rolled_manufacturability"],
    )


def make_env(cfg: dict, reward_mode: str, seed: int, rank: int):
    def _init():
        from stable_baselines3.common.monitor import Monitor
        from hssbeamgen.envs.hss_env import HSSBeamEnv
        env = Monitor(HSSBeamEnv(**env_kwargs(cfg, reward_mode)))
        env.reset(seed=seed + rank)
        return env
    return _init


def build_vec_env(cfg: dict, reward_mode: str, seed: int):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    fns = [make_env(cfg, reward_mode, seed, i) for i in range(cfg["n_envs"])]
    return SubprocVecEnv(fns) if cfg["n_envs"] > 1 else DummyVecEnv(fns)


def wrap_normalize(vec_env, cfg: dict):
    """norm_obs stays False: the evaluation harness loads the bare policy without VecNormalize."""
    from stable_baselines3.common.vec_env import VecNormalize
    return VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                        clip_reward=cfg["reward_norm_clip"], gamma=cfg["gamma"])


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
def algo_class(algo: str):
    from stable_baselines3 import DDPG, PPO, SAC, TD3
    return dict(ppo=PPO, sac=SAC, td3=TD3, ddpg=DDPG)[algo]


def build_model(cfg: dict, vec_env, seed: int, tb_dir: str | None):
    algo = cfg["algo"]
    cls = algo_class(algo)
    common = dict(policy_kwargs=dict(net_arch=list(cfg["net_arch"])), seed=seed, verbose=1,
                  device=cfg["device"], tensorboard_log=tb_dir)
    if algo == "ppo":
        return cls("MlpPolicy", vec_env, learning_rate=cfg["lr"], n_steps=cfg["n_steps"],
                   batch_size=cfg["batch_size"], n_epochs=cfg["n_epochs"], gamma=cfg["gamma"],
                   gae_lambda=cfg["gae_lambda"], clip_range=cfg["clip_range"], ent_coef=cfg["ent_coef"],
                   vf_coef=cfg["vf_coef"], max_grad_norm=cfg["max_grad_norm"], **common)
    kwargs = dict(learning_rate=cfg["lr"], buffer_size=cfg["buffer_size"], batch_size=cfg["batch_size"],
                  learning_starts=cfg["learning_starts"], gamma=cfg["gamma"], train_freq=cfg["train_freq"],
                  gradient_steps=cfg["gradient_steps"], **common)
    if algo in ("ddpg", "td3"):
        from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
        n_act = vec_env.action_space.shape[-1]
        base = NormalActionNoise(mean=np.zeros(n_act), sigma=cfg["action_noise_sigma"] * np.ones(n_act))
        kwargs["action_noise"] = base if cfg["n_envs"] == 1 else VectorizedActionNoise(base, cfg["n_envs"])
    return cls("MlpPolicy", vec_env, **kwargs)


def build_callbacks(cfg: dict, reward_mode: str, run_dir: str):
    """Returns (callbacks, lagrangian_cb_or_None). Includes the resume-state writer."""
    from stable_baselines3.common.callbacks import CheckpointCallback
    from hssbeamgen.algo.lagrangian import LagrangianCallback
    from hssbeamgen.algo.log_std_anneal import LogStdAnnealCallback

    n_envs = cfg["n_envs"]
    cbs = [CheckpointCallback(save_freq=max(cfg["checkpoint_every"] // n_envs, 1), save_path=run_dir,
                              name_prefix="checkpoint", save_vecnormalize=True)]
    lag = None
    if reward_mode == "lagrangian":
        lag = LagrangianCallback(
            constraint_names=list(LAGRANGE_KEYS),
            etas={"g1_util": cfg["eta_util"], "g2_class": cfg["eta_class"], "g3_geom": cfg["eta_geom"]},
            budgets={"g1_util": cfg["budget_util"], "g2_class": 0.0, "g3_geom": 0.0},
            lambda_max=cfg["lambda_max"], update_freq=cfg["update_freq"], log_every=1, verbose=1)
        cbs.append(lag)
    if cfg["algo"] == "ppo" and cfg["log_std_anneal"]:
        cbs.append(LogStdAnnealCallback(   # GRAND total, so the schedule stays correct across resumes
            total_timesteps=cfg["timesteps"], ceiling_start=cfg["log_std_ceiling_start"],
            ceiling_end=cfg["log_std_ceiling_end"], anneal_start_frac=cfg["log_std_anneal_start_frac"],
            verbose=1))
    cbs.append(ResumeCallback(run_dir, cfg, lag))
    return cbs, lag


# ----------------------------------------------------------------------------------------
# Resume
# ----------------------------------------------------------------------------------------
def _atomic_write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def save_resume_state(model, vec_env, run_dir: str, cfg: dict, lag_cb) -> None:
    """Write model, VecNormalize stats, replay buffer (off-policy) and Lagrangian state.

    Files are written under temporary names and moved into place; the JSON state file is
    written LAST and records num_timesteps, so a crash mid-save is detected on resume.
    """
    tmp_model = os.path.join(run_dir, "resume_tmp.zip")
    model.save(tmp_model[:-4])
    os.replace(tmp_model, os.path.join(run_dir, RESUME_MODEL))
    tmp_vn = os.path.join(run_dir, "resume_tmp_vecnormalize.pkl")
    vec_env.save(tmp_vn)
    os.replace(tmp_vn, os.path.join(run_dir, RESUME_VECNORM))
    if cfg["algo"] in OFFPOLICY:
        tmp_buf = os.path.join(run_dir, "resume_tmp_replay_buffer.pkl")
        model.save_replay_buffer(tmp_buf)
        os.replace(tmp_buf, os.path.join(run_dir, RESUME_BUFFER))
    state = dict(num_timesteps=int(model.num_timesteps), algo=cfg["algo"], timesteps=cfg["timesteps"])
    if lag_cb is not None:
        state["lagrange_multipliers"] = vec_env.env_method("get_lagrange_multipliers")[0]
        state["lagrangian_callback"] = lag_cb.state_dict()
    _atomic_write_json(os.path.join(run_dir, RESUME_STATE), state)


def load_resume_state(run_dir: str, cfg: dict, vec_env_raw):
    """Returns (model, vec_env, state) restored from run_dir, or None if there is nothing to resume."""
    from stable_baselines3.common.vec_env import VecNormalize
    state_path = os.path.join(run_dir, RESUME_STATE)
    if not os.path.exists(state_path):
        return None
    with open(state_path) as f:
        state = json.load(f)
    if state["algo"] != cfg["algo"]:
        raise SystemExit(f"resume state is for algo={state['algo']}, not {cfg['algo']}")
    vec_env = VecNormalize.load(os.path.join(run_dir, RESUME_VECNORM), vec_env_raw)
    vec_env.training = True
    model = algo_class(cfg["algo"]).load(os.path.join(run_dir, RESUME_MODEL), env=vec_env, device=cfg["device"])
    if model.num_timesteps != state["num_timesteps"]:
        raise SystemExit(
            f"resume files are inconsistent (model at {model.num_timesteps} steps, state file says "
            f"{state['num_timesteps']}): the previous run was interrupted while saving. "
            f"Delete resume_* in {run_dir} and use the newest checkpoint_*_steps.zip instead.")
    if cfg["algo"] in OFFPOLICY:
        model.load_replay_buffer(os.path.join(run_dir, RESUME_BUFFER))
    if "lagrange_multipliers" in state:
        vec_env.env_method("set_lagrange_multipliers", state["lagrange_multipliers"])
    return model, vec_env, state


def restore_lagrangian_callback(lag_cb, state: dict) -> None:
    if lag_cb is not None and "lagrangian_callback" in state:
        lag_cb.load_state_dict(state["lagrangian_callback"])


def _resume_callback_cls():
    from stable_baselines3.common.callbacks import BaseCallback

    class _ResumeCallback(BaseCallback):
        """Saves the full resume state every `checkpoint_every` environment timesteps."""

        def __init__(self, run_dir, cfg, lag_cb):
            super().__init__(verbose=0)
            self.run_dir, self.cfg, self.lag_cb = run_dir, cfg, lag_cb
            self._next = None

        def _on_training_start(self):
            every = self.cfg["checkpoint_every"]
            self._next = (self.model.num_timesteps // every + 1) * every

        def _on_step(self) -> bool:
            if self.num_timesteps >= self._next:
                save_resume_state(self.model, self.training_env, self.run_dir, self.cfg, self.lag_cb)
                self._next += self.cfg["checkpoint_every"]
            return True

    return _ResumeCallback


def ResumeCallback(run_dir, cfg, lag_cb):  # noqa: N802 - factory so SB3 is imported lazily
    return _resume_callback_cls()(run_dir, cfg, lag_cb)


def cfg_for_json(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
