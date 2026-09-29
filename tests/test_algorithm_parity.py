"""Every RL arm must train on the SAME environment definition.

Replaces the old research/tests/test_algorithm_parity.py, which caught PPO and the
off-policy arms silently drifting apart. Here it's structural: env_kwargs() is the single
function every algorithm's environment is built from (see hssbeamgen/train_utils.py), so this
test just has to confirm that function is algorithm-independent and that resolve_config()
doesn't leak an algorithm-specific value into it.
"""
import pytest

from hssbeamgen.train_utils import ALGOS, OFFPOLICY, env_kwargs, resolve_config

ENV_SHAPING_KEYS = ("reward_mode", "economy_metric", "lagrange_init", "ltb_restraint_factor",
                    "sls_load_factor", "economy_reward_mode", "max_steps",
                    "enforce_rolled_manufacturability")


@pytest.mark.parametrize("reward_mode", ["feasibility_gated", "lagrangian", "shaped"])
def test_env_kwargs_identical_across_algorithms(reward_mode):
    kwargs_by_algo = {a: env_kwargs(resolve_config("configs/rl_final.yaml", a), reward_mode)
                      for a in ALGOS}
    reference = kwargs_by_algo["ppo"]
    assert set(reference) == set(ENV_SHAPING_KEYS)
    for algo, kw in kwargs_by_algo.items():
        assert kw == reference, f"{algo} env_kwargs differ from ppo: {kw} vs {reference}"


def test_offpolicy_gets_single_env_by_default():
    for algo in OFFPOLICY:
        cfg = resolve_config("configs/rl_final.yaml", algo)
        assert cfg["n_envs"] == 1, (
            f"{algo}: n_envs must default to 1 (parallel envs silently divide "
            f"gradient-updates-per-env-step for off-policy algorithms; see configs/rl_final.yaml)")


def test_log_std_anneal_is_ppo_only():
    assert resolve_config("configs/rl_final.yaml", "ppo")["log_std_anneal"] is True
    for algo in OFFPOLICY:
        assert resolve_config("configs/rl_final.yaml", algo)["log_std_anneal"] is False
    with pytest.raises(ValueError):
        resolve_config("configs/rl_final.yaml", "sac", overrides={"log_std_anneal": True})


def test_gradient_steps_defaults_preserve_updates_per_env_step():
    """n_envs * gradient_steps should be the invariant regardless of how many envs are used."""
    for algo in OFFPOLICY:
        cfg = resolve_config("configs/rl_final.yaml", algo, overrides={"n_envs": 4})
        assert cfg["gradient_steps"] == 4 * cfg["train_freq"]


def test_unknown_config_key_rejected():
    with pytest.raises(ValueError):
        resolve_config("configs/rl_final.yaml", "ppo", overrides={"totally_made_up_key": 1})


def test_manufacturability_cannot_be_disabled():
    with pytest.raises(ValueError):
        resolve_config("configs/rl_final.yaml", "ppo",
                       overrides={"enforce_rolled_manufacturability": False})
