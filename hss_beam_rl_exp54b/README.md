# Generative Design of High-Strength Steel (HSS) Beams via PPO

A reinforcement-learning framework for automated, EC3-compliant structural
design of steel beams, extending demand-driven material selection (S355
through S690) alongside geometry optimization — trained with Proximal
Policy Optimization (PPO) rather than the DDPG approach used in prior work
on RL for structural design (see `docs/EXPERIMENT_LOG.md` for related-work
context).

## What this does

Given a span, distributed load, and number of storeys, the trained agent
iteratively proposes a steel section (depth `h`, width `b`, flange
thickness `tf`, web thickness `tw`), steel grade (`S355`–`S690`), and
section type (`rolled`/`welded`), refining the design over up to 40 steps
to reach EC3-compliant utilization near 1.0 while minimizing mass, cost,
and lifecycle CO₂. The core scientific claim: the agent learns to reserve
expensive, low-CO₂-per-capacity HSS grades for genuinely demanding
spans/loads, and sticks with cheap conventional steel (S355/S460)
otherwise — mirroring real structural-engineering judgment.

## Final / reference result

97.0% feasibility across 200 random (span, load) episodes, 62/64 feasible
cells on a structured 8×8 span×load grid, with demand-driven grade
diversity across all six grades (S620+S690 = 40.2% of feasible episodes,
concentrated at the longest spans / heaviest loads as intended). See
`docs/EXPERIMENT_LOG.md` for the full derivation and the ablation showing
this result over five candidate architectural variants.

## Project structure

```
hss_beam_rl/
├── env/
│   └── high_rise_generative_env.py    # the EC3 RL environment (final/reference version)
├── training/
│   └── train.py                       # PPO training via Stable-Baselines3
├── evaluation/
│   ├── validate.py                    # 200-episode + 64-point grid validation
│   ├── diagnostic_grade_comparison.py # frozen-geometry grade sweep + policy rollout on hard scenarios
│   └── diagnostic_stability_trace.py  # full per-step trajectory logger (action magnitudes, geometry, grade)
├── analysis/
│   ├── generate_ec3_pretrain_dataset.py  # brute-force EC3-optimal geometry dataset (ground truth)
│   ├── inspect_pretrain_dataset.py       # sanity-checks the generated dataset
│   ├── fit_h_target.py                   # fits physics-informed h_target(span, load) regression
│   └── inspect_obs_normalization.py      # probes VecNormalize for observation distortion
├── pretrain_data/
│   └── ec3_optimal_designs.csv        # brute-force EC3-optimal ground truth (1200 span/load/grade contexts)
├── models/                            # trained checkpoints go here (not included — see models/README.md)
├── docs/
│   └── EXPERIMENT_LOG.md              # full exp46→exp62 history, findings, and final ablation
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train

```bash
cd training
python train.py --run-name hss_final --timesteps 1000000
```

Key flags: `--timesteps`, `--n-envs`, `--seed`, `--ltb-factor`,
`--sls-factor`. PPO hyperparameters (`--lr`, `--n-steps`, `--batch-size`,
`--ent-coef`, etc.) are also exposed; defaults match the reference run.
Outputs land in `training/models/<run-name>/` (`best_model.zip`,
`vecnormalize.pkl`, `training_config.json`) and
`training/runs/<run-name>/` (TensorBoard logs).

### 2. Validate

```bash
cd evaluation
python validate.py --model ../training/models/hss_final/best_model.zip --grid
```

Runs 200 random episodes plus the full 64-point span×load grid, reporting
feasibility rate, utilization/mass/cost/CO₂ statistics, grade distribution,
and per-grade efficiency. Writes CSVs and a TensorBoard log alongside the
model.

### 3. Diagnose specific scenarios

```bash
python diagnostic_grade_comparison.py \
    --model ../training/models/hss_final/best_model.zip \
    --env ../env/high_rise_generative_env.py

python diagnostic_stability_trace.py \
    --model ../training/models/hss_final/best_model.zip \
    --env ../env/high_rise_generative_env.py \
    --scenario 13,140,welded --scenario 15,120,rolled \
    --csv-out ../pretrain_data/stability_trace.csv
```

The first compares the policy's design against a full grade sweep and a
brute-force min-mass geometry search at five representative hard scenarios.
The second logs every single step of a rollout (action magnitudes,
resulting geometry, grade, utilization) — use this to inspect whether a
scenario converges cleanly, oscillates, or never reaches feasibility.

### 4. Ground-truth EC3 analysis (optional, no training required)

```bash
cd analysis
python generate_ec3_pretrain_dataset.py --resolution 9 --n-spans 12 --n-loads 12
python inspect_pretrain_dataset.py
python fit_h_target.py
python inspect_obs_normalization.py --vecnorm ../training/models/hss_final/vecnormalize.pkl
```

These tools build an independent, brute-force EC3-optimal dataset (no RL
involved) used throughout this project to validate policy behavior against
ground truth, fit physics-informed observation features, and rule out
`VecNormalize` distortion as a cause of residual instability. See
`docs/EXPERIMENT_LOG.md` for how each was used.

## Known limitation

In the most extreme demand corner of the design space (span 11–15m, load
≥100 kN/m — roughly the top-right 15% of the sampled range), the trained
policy sometimes finds a genuinely feasible design but does not hold it
for the three consecutive steps required for early termination, instead
running the full 40-step episode. This does not affect the reported
feasibility numbers above (both grid validation and targeted diagnostics
confirm the best design found in this region is feasible), only
training-time efficiency in that narrow region. Six dedicated experiments
(documented in `docs/EXPERIMENT_LOG.md`) established this is very likely
PPO advantage-estimate noise specific to this underrepresented region of
the sampled demand space, not a fixable defect in the environment's reward
or action design — the same behavior appears in the unmodified reference
environment regardless of which of several tested action-level or
reward-level interventions is applied.
