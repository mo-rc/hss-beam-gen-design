# H1 pilot: one-shot reparameterized action space, trained (not just sampled)

Context: research/README.md and manuscript_final_comprehensive.md Sec 6-7
already diagnose the bottleneck as MDP/action-space formulation (not
algorithm, not reward, not training budget), and already show, with ZERO
training, that blind uniform sampling in a reparameterized one-shot space
(h, b/h, lambda_f, lambda_w, grade, type -- feasible-by-construction,
bisected to util=1.0) reaches 5.3% mean gap at 4,800 evals, beating the
fully-trained 5-seed incremental PPO arm (9.05% at 112 evals, `scale+thin`).
This was flagged as the highest-priority untested follow-up: does an
AMORTIZED (trained) policy in that same space do even better -- ideally
closing the gap to GA (1.47-1.72%) at low inference cost?

## What was run
- New env: research/envs/hss_reparam_env.py -- one-shot HSSReparamEnv,
  same action dimensionality (6) as HSSBeamEnv, decode+bisect-to-boundary
  imported VERBATIM from the already-published diag_reparam_probe.py
  (no reimplementation, no drift). EC3 physics/costing/ground truth/eval
  harness untouched (inherited from HSSBeamEnv, same evaluate.py-style
  gap computation).
- Trainer: research/scripts/train_reparam_es.py -- pure-NumPy OpenAI-ES
  (Salimans et al. 2017), NOT PPO. stable-baselines3/torch could not be
  reliably installed in this sandbox (1 CPU core, ~6 GB free disk; the
  CUDA-bundled PyPI torch wheel alone exceeded available space and left
  a broken install after repeated attempts).
- Budget actually run: ~600-700 generations, pop=32 antithetic pairs,
  batch=10-16 contexts/member -> ~230k-380k one-shot rollouts total,
  ~8 minutes wall-clock on 1 CPU core.

## Results (142 held-out contexts, corrected costing, cost metric)

| Policy | Mean gap | Median gap | Feasibility | Inference cost/design |
|---|---|---|---|---|
| Random-init (untrained) reparam policy | 38.5% | 35.6% | 0.97 | 1 fwd pass + ~24 EC3 evals |
| **ES-trained reparam policy (this pilot)** | **10.7-11.3%** | **9.3-10.0%** | 0.92-0.94 | 1 fwd pass + ~24 EC3 evals |
| Incumbent PPO (raw, unrepaired), 5-seed | 38.8-64.8% | 32.4-60.2% | 1.00 | 40 EC3 evals |
| Incumbent PPO + `scale+thin`, 5-seed | 9.05% (mean of means) | ~5-7% | 1.00 | 112 EC3 evals |
| Blind sampling in reparam space, 4,800 evals (no training) | 5.3% | 4.2% | 1.00 | 4,800 EC3 evals |
| GA baseline (unrepaired / `scale+thin`) | 1.72% / 1.47% | 0.73% / 0.54% | 1.00 | 4,801 / 4,868 EC3 evals |

## Interpretation (hypothesis: SUPPORTED, DIRECTIONALLY, NOT YET CONFIRMED)

- Training in the reparameterized space clearly works: it cuts the
  gap from 38.5% (untrained) to ~11% -- a genuine, non-trivial
  improvement from optimization, not just from feasible-by-construction
  geometry.
- Even under-converged (my ES plateaued after ~600-700 generations,
  val-set fitness still noisy generation-to-generation) and with NO
  repair operator applied, the trained one-shot reparam policy is
  already competitive with the fully-trained, repair-operator-assisted
  incumbent PPO (9.05%), at roughly the same order of inference cost
  (~24 EC3 evals vs. 112), and is 3.5-6x better than the incumbent's
  raw/unrepaired result.
- It has NOT yet closed the gap to the zero-training blind-sampling
  result (5.3%) or to GA, unlike the paper's stated hypothesis that a
  trained policy should beat blind sampling at the same budget. Two
  candidate explanations, not yet distinguished:
  (a) my ES optimizer is simply too weak/under-tuned (pop=32 is small
      for a high-variance, wide-context-range fitness landscape; no
      GAE-style variance reduction, no learning-rate schedule) -- a
      real PPO run with proper advantage estimation would likely do
      materially better on the SAME action space, since PPO already
      out-performs plain black-box search on the raw arm;
  (b) OR there's a genuine remaining formulation issue in the one-shot
      space itself (e.g. the fixed-point decode's grade/type coupling,
      or a single small MLP struggling to represent the sharp
      grade-selection boundary across a 2D context range) that a
      stronger optimizer would also struggle with.
- This pilot cannot distinguish (a) from (b) -- that is exactly what
  the follow-up (real PPO/SAC run via stable-baselines3 on
  HSSReparamEnv, on adequate compute) is for.

## Immediate next step (not run here: needs real compute)
`research/envs/hss_reparam_env.py` is a drop-in Gym env with the SAME
action_space/observation_space shape as HSSBeamEnv, so `research/scripts/
train.py` needs only an `--env_class HSSReparamEnv` (or equivalent
one-line swap) to train real PPO on it, at the SAME 1M-step / 5-seed
budget already used for the incumbent arm, for a genuinely fair,
apples-to-apples comparison.
