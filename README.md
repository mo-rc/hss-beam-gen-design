# hssbeamgen

EC3-compliant generative design of high-strength-steel (S355–S690) I-beams:
amortized design policies (reinforcement learning) compared against conventional
optimization (GA/DE/CMA-ES) and simpler learned mappings (kNN).

> **Status: work in progress.** This branch is being rebuilt step by step from an
> audited research codebase. Results, figures and the reproduction guide will be added
> once the final evidence is frozen.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

(Package installation and usage instructions will be added with the first code release.)

## Progress log

Interim results are logged in `docs/` as each pipeline stage completes, separate from the
eventual manuscript (which is rewritten from scratch once all evidence is frozen):

- [`docs/results_2a_reward_mode.md`](docs/results_2a_reward_mode.md) -- reward-mode ablation
  (PPO, feasibility_gated / lagrangian / shaped), complete.

## License

MIT. See `LICENSE`.
