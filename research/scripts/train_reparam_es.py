"""
research/scripts/train_reparam_es.py
================================================================
Trains an AMORTIZED policy for HSSReparamEnv (H1 pilot experiment).

WHY NOT stable-baselines3 PPO (as train.py uses for the raw arm): this
sandbox has 1 CPU core, no GPU, and ~6 GB free disk -- not enough to
reliably install a working torch build (attempted; the CUDA-bundled
PyPI wheel alone exceeded available disk and left a broken install).
So this uses OpenAI-ES (Salimans et al. 2017) -- a black-box,
gradient-free RL-adjacent method -- implemented in plain NumPy, to make
the "does an AMORTIZED policy in the reparam space close the gap"
hypothesis testable at all in this environment. This is explicitly a
PILOT/proof-of-concept, not a replacement for a real PPO run: see the
final report for what a full run (this repo's own train.py, pointed at
HSSReparamEnv, on real compute) would additionally need to verify.

Policy: obs(26) -> tanh(32) -> tanh(6) small MLP, flat-vector params,
antithetic-sampled Gaussian perturbations, rank-based fitness shaping,
Adam-style moment update. Fitness = mean one-shot reward (== -economy
for feasible designs, == -infeasible_penalty otherwise) over a batch of
contexts freshly sampled each generation from the SAME reset()
distribution HSSBeamEnv's raw arm was trained under (train.py), so the
comparison stays apples-to-apples on training distribution.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from research.envs.hss_reparam_env import HSSReparamEnv

OBS_DIM = 26
ACT_DIM = 6


def init_params(hidden, rng):
    scale1 = np.sqrt(2.0 / OBS_DIM)
    scale2 = np.sqrt(2.0 / hidden)
    W1 = rng.standard_normal((OBS_DIM, hidden)).astype(np.float32) * scale1
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.standard_normal((hidden, ACT_DIM)).astype(np.float32) * scale2 * 0.1
    b2 = np.zeros(ACT_DIM, dtype=np.float32)
    return flatten([W1, b1, W2, b2])


def shapes(hidden):
    return [(OBS_DIM, hidden), (hidden,), (hidden, ACT_DIM), (ACT_DIM,)]


def flatten(arrs):
    return np.concatenate([a.ravel() for a in arrs]).astype(np.float32)


def unflatten(theta, hidden):
    sh = shapes(hidden)
    sizes = [int(np.prod(s)) for s in sh]
    parts, i = [], 0
    for s, n in zip(sh, sizes):
        parts.append(theta[i:i + n].reshape(s))
        i += n
    return parts


def forward(theta, hidden, obs_batch):
    W1, b1, W2, b2 = unflatten(theta, hidden)
    h = np.tanh(obs_batch @ W1 + b1)
    a = np.tanh(h @ W2 + b2)
    return a


def rollout_batch(env, theta, hidden, rng, batch_size, seed_base):
    obs_list = np.zeros((batch_size, OBS_DIM), dtype=np.float32)
    for i in range(batch_size):
        obs, _ = env.reset(seed=int(seed_base) + i)
        obs_list[i] = obs
    actions = forward(theta, hidden, obs_list)
    rewards = np.zeros(batch_size, dtype=np.float64)
    for i in range(batch_size):
        env.reset(seed=int(seed_base) + i)  # restore the same context deterministically
        _, r, _, _, info = env.step(actions[i])
        rewards[i] = r
    return rewards


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--economy_metric", default="cost")
    p.add_argument("--reward_mode", default="feasibility_gated")  # unused by reparam env's reward, kept for env ctor signature parity
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--pop", type=int, default=64)  # antithetic pairs -> 2*pop evals/gen
    p.add_argument("--batch", type=int, default=16)  # contexts per fitness evaluation
    p.add_argument("--generations", type=int, default=600)
    p.add_argument("--sigma", type=float, default=0.08)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="research/models/reparam_es_cost.npy")
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--init_from", default=None, help="warm-start theta from a previously saved .npy")
    p.add_argument("--ctx_seed_start", type=int, default=10_000)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    env = HSSReparamEnv(reward_mode=a.reward_mode, economy_metric=a.economy_metric)

    if a.init_from:
        payload = np.load(a.init_from, allow_pickle=True).item()
        theta = payload["theta"].astype(np.float32)
        assert payload["hidden"] == a.hidden, "hidden size mismatch with warm-start file"
        print(f"warm-started from {a.init_from}")
    else:
        theta = init_params(a.hidden, rng)
    n_params = theta.shape[0]
    print(f"n_params={n_params}  pop={a.pop} (x2 antithetic)  batch={a.batch}  "
          f"generations={a.generations}  -> {2*a.pop*a.batch*a.generations} EC3 rollouts total")

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    # Fixed held-out validation batch (disjoint seed range from the training
    # context stream, which uses [ctx_seed_start, ctx_seed_start + pop*batch*
    # generations)) so progress is judged on THE SAME contexts every
    # generation -- training-batch fitness is not comparable across
    # generations since each generation samples a different, variable-
    # difficulty context batch (this was the bug in the first pilot run:
    # "best_fit" tracked raw training-batch fitness and was not meaningful).
    val_seed_base = 5_000_000
    val_size = 48

    def validate(th):
        return rollout_batch(env, th, a.hidden, rng, val_size, val_seed_base).mean()

    t0 = time.time()
    best_val = validate(theta)
    best_theta = theta.copy()
    print(f"init val_fit {best_val:+.4f}")
    ctx_seed = a.ctx_seed_start
    for gen in range(1, a.generations + 1):
        noise = rng.standard_normal((a.pop, n_params)).astype(np.float32)
        fits = np.zeros(2 * a.pop, dtype=np.float64)
        seed_base = ctx_seed  # SAME context batch for every member this generation (paired comparison, lower variance)
        for k in range(a.pop):
            theta_pos = theta + a.sigma * noise[k]
            theta_neg = theta - a.sigma * noise[k]
            fits[2 * k] = rollout_batch(env, theta_pos, a.hidden, rng, a.batch, seed_base).mean()
            fits[2 * k + 1] = rollout_batch(env, theta_neg, a.hidden, rng, a.batch, seed_base).mean()
        ctx_seed += a.batch  # advance context stream so successive generations see fresh contexts

        # rank-based fitness shaping (robust to reward scale/outliers)
        order = np.argsort(fits)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(fits))
        shaped = ranks / (len(fits) - 1) - 0.5  # centered in [-0.5, 0.5]

        grad = np.zeros(n_params, dtype=np.float64)
        for k in range(a.pop):
            grad += (shaped[2 * k] - shaped[2 * k + 1]) * noise[k]
        grad /= (a.pop * a.sigma)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        mhat = m / (1 - beta1 ** gen)
        vhat = v / (1 - beta2 ** gen)
        theta = (theta + a.lr * mhat / (np.sqrt(vhat) + eps_adam)).astype(np.float32)

        gen_mean_fit = fits.mean()

        if gen % a.log_every == 0 or gen == 1 or gen == a.generations:
            val_fit = validate(theta)
            improved = val_fit > best_val
            if improved:
                best_val = val_fit
                best_theta = theta.copy()
            elapsed = time.time() - t0
            print(f"gen {gen:4d}  train_batch_fit {gen_mean_fit:+.4f}  val_fit {val_fit:+.4f}  "
                  f"best_val {best_val:+.4f}{' *' if improved else ''}  elapsed {elapsed:6.1f}s")
            # checkpoint every time we log, so a killed/timed-out run still
            # leaves a usable (best-so-far-on-held-out-val) policy on disk.
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
            np.save(a.out, dict(theta=best_theta, hidden=a.hidden, economy_metric=a.economy_metric,
                                 best_val=float(best_val)), allow_pickle=True)

    print(f"saved best-on-val policy -> {a.out}  (best_val={best_val:.4f}, {time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
