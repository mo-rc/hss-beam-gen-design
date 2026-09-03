# E3 — Corrected-Cost PPO Retrain: Recovery Audit and Runbook

**Status:** not started (no artefacts exist). **Date:** 2026-09-03.
**Executes:** E1 §10 open item 1 — *"Retrain the best arm under corrected costing
(single seed, 1.3M steps)"*.
**Repo audited:** `github.com/mo-rc/hss-beam-gen-design` @ `a11e01d`
("experiment 2 results"), `research/` only. **This is still `origin/main` HEAD.**

---

## 1. Recovery audit — what the previous Computer session left behind

### 1.1 Verdict

**Nothing of the corrected-cost retraining run was saved. Zero steps were
completed. There is no checkpoint, no TensorBoard event file, no
`VecNormalize` statistics, no `training_config.json`, and no `resume_state.json`
for any corrected-cost run. Training cannot be resumed — it must be started
from scratch.**

Worse, and more important: **the E1 environment patch itself was never
persisted either.** The previous sessions' cost correction lived only in an
ephemeral sandbox working copy. It has now been reconstructed and verified
(§2), and that reconstruction is the reason this run is reproducible at all.

### 1.2 Evidence

| Check | Result |
|---|---|
| `research/models/` dirs matching `corr*`, `retrain*`, `manu*` | **none** |
| `research/runs/` TB dirs matching `corr*`, `retrain*`, `manu*` | **none** |
| Any `training_config.json` with `enforce_rolled_manufacturability` | **none** |
| Any `resume_state.json` other than `gated_cost_merged_seed42` (pre-correction, 1,302,528 steps) | **none** |
| `grep -rl manufacturab research/` on `a11e01d` | **no matches** |
| `grep -rn enforce_rolled research/` on `a11e01d` | **no matches** |
| `research/pretrain_data_corrected/` (the corrected ground truth) | **absent** |
| `research/envs/manufacturability.py` | **absent from repo** (present in Project Files `code/`) |
| `research/algo/repair.py`, `research/scripts/evaluate_with_repair.py` | **absent from repo** (present in Project Files `code/`) |

Both E1 §11 and E2 §7 state plainly: *"The sandbox has no push credentials for
`github.com/mo-rc/hss-beam-gen-design`; these changes must be committed
locally."* That never happened, and the `a11e01d` tree confirms it.

### 1.3 What *is* recoverable, and from where

| Artefact | Source of truth |
|---|---|
| `manufacturability.py` (`ROLLED_LIMITS`, `effective_section_type`) | Project Files `code/manufacturability.py` — intact |
| `repair.py` (incl. `scale+thin_rolled`) | Project Files `code/repair.py` — intact |
| `evaluate_with_repair.py` (incl. E2 `n_ec3` accounting fix) | Project Files `code/evaluate_with_repair.py` — intact |
| `diag_gap_decomposition.py`, `diag_reparam_probe.py` | Project Files `code/` — intact |
| **`hss_env.py` cost-correction patch** | **Was lost. Reconstructed here from the E1 §3/§11 spec and verified in §2.** |
| `research/pretrain_data_corrected/` | Lost. Must be regenerated (21 min per metric, §4 step 2). |
| Pre-correction 1.3M checkpoints (`gated_cost_merged_seed42..46`) | In repo, unaffected — these remain the transfer-result baseline |

### 1.4 Why resuming is not an option even in principle

There is nothing to resume from. And the pre-correction
`gated_cost_merged_seed42/resume_latest.zip` (1,302,528 steps) **must not** be
used as a warm start: E1 §6.5 is explicit that the corrected environment
changes the reward and the observation (economy enters both), so warm-starting
would produce a fine-tuning result, not the clean retrained result the
experiment is designed to isolate. **Train from random initialisation.**

---

## 2. The reconstructed patch, and proof it is faithful

`research/envs/hss_env.py` gains, exactly per E1 §3/§11:

- constructor flags `enforce_rolled_manufacturability: bool = True` and
  `rolled_limits: dict | None = None`;
- `_calculate_cost_co2` computes
  `eff_type = effective_section_type(section_type, h, b, tf, tw, fy)` and uses
  `eff_type` — not the free `section_type` label — for the fabrication factor,
  the grade multipliers and the ≥S550 thickness penalty;
- `debug` gains `effective_section_type` and `reclassified`.

30 insertions, 2 deletions. No EC3 capacity equation, class check or
constraint definition is touched.

**Regression test (the E1 §3 acceptance criterion), re-run 2026-09-03:**

| test | E1 reported | reproduced here |
|---|---|---|
| Flag **off**, re-cost 400 sampled stored GT rows vs stored `cost` | max rel. err 6.16e-16 | **max rel. err 5.43e-16** |
| Flag **on**, share of all GT rows reclassified | 827/837 rolled-labelled = 98.8% (→ ~50.9% of all 1623 rows) | **197/400 = 49.2% of all rows** |
| Flag **on**, naive re-cost inflation at unchanged geometry | +19.3% mean over the 142 optima | **+13.5% mean over all rows** (different population; optima are the slender tail) |

The first line is the one that matters: with the flag off the patched
environment reproduces every historical number to floating-point identity.
A short end-to-end PPO smoke run (8,192 steps, 4 envs) completes cleanly with
the patch active.

---

## 3. Exact specification of the run

Fixed by E1 §10 item 1 (single seed, 1.3M steps) and by the best arm on disk,
`gated_cost_merged` (`feasibility_gated` + `cost` + `log_relative` +
`log_std_anneal`; `gated_cost_merged_seed42/training_config.json`).
**Every hyperparameter below is identical to the pre-correction 1.3M run — the
corrected costing is the only change. That is what makes it controlled.**

| field | value | note |
|---|---|---|
| run name | `corrcost_gated_merged_seed42` | new namespace; does not collide |
| seed | `42` | canonical first seed — see §6 |
| env | `HSSBeamEnv`, `env_type=continuous` | 6-dim `Box(-1,1)`: Δh, Δb, Δtf, Δtw, grade (softmax-snap, temp 0.15), section type (sign); 26-dim obs; 40 steps; `step_scale = 0.30 + 0.70·½(1+cos(π·progress))` |
| **costing** | `enforce_rolled_manufacturability=True` | **the only change; default True, so no CLI flag needed** |
| reward mode | `feasibility_gated` | |
| economy metric | `cost` | |
| economy reward mode | `log_relative` | |
| log_std anneal | on, ceiling 8.0 → 1.0, `start_frac` 0.5 | |
| total timesteps | `1_300_000` | single run, not 1M + resume |
| n_envs | 8 (`SubprocVecEnv`) | |
| lr / n_steps / batch / epochs | 3e-4 / 1024 / 256 / 8 | |
| γ / λ / clip / ent_coef / vf_coef / max_grad_norm | 0.99 / 0.95 / 0.15 / 0.03 / 0.5 / 0.5 | |
| net arch | `[256, 256, 128]` | `train.py` default |
| VecNormalize | `norm_obs=False, norm_reward=True, clip_reward=50` | so eval with raw obs stays correct |
| ltb_factor / sls_factor | 0.40 / 0.50 | |
| checkpoints | every 200k steps + `final_model.zip` + `vecnormalize.pkl` | `train.py` default |
| expected wall time | **~15–25 min** on 2 vCPU (measured 1,841 fps at 4 envs) | 1.3M steps is cheap; do not budget for hours |

Unused in this arm (`feasibility_gated` ignores them): `eta_util`,
`eta_class`, `eta_geom`, `lambda_max`, `budget_util`.

---

## 4. Runbook — local or Colab

### Step 0 — restore the corrected-cost code into the repo

`origin/main` does **not** contain it. Copy from Project Files `code/` into the
repo tree, then commit (this time actually commit — that is how it was lost):

```bash
git clone https://github.com/mo-rc/hss-beam-gen-design.git
cd hss-beam-gen-design

# from the Project Files `code/` folder of this project:
cp code/manufacturability.py        research/envs/manufacturability.py
cp code/repair.py                   research/algo/repair.py
cp code/evaluate_with_repair.py     research/scripts/evaluate_with_repair.py
cp code/diag_gap_decomposition.py   research/scripts/diag_gap_decomposition.py
cp code/diag_reparam_probe.py       research/scripts/diag_reparam_probe.py

# the reconstructed cost-correction patch — EITHER drop in the whole file:
cp code/hss_env.py                  research/envs/hss_env.py
# OR apply it as a patch to your own copy:
#   git apply code/hss_env_manufacturability.patch

git add -A && git commit -m "E1/E2: manufacturability-aware costing, repair operator, corrected budget accounting"
```

Dependencies (E1 §1.3 note: `stable-baselines3[extra]` fails on some
sandboxes — use the plain package):

```bash
pip install "stable-baselines3==2.3.2" gymnasium numpy pandas tensorboard
```

### Step 0b — verify the patch before spending compute (30 s, do not skip)

```bash
python - <<'PY'
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from research.envs.hss_env import HSSBeamEnv
df = pd.read_csv('research/pretrain_data/ec3_optimal_designs_cost.csv')
idx = np.random.default_rng(0).choice(len(df), 400, replace=False)
def recost(r, enforce):
    e = HSSBeamEnv(reward_mode='feasibility_gated', economy_metric='cost',
                   enforce_rolled_manufacturability=enforce)
    e.h, e.b, e.tf, e.tw = float(r.h), float(r.b), float(r.tf), float(r.tw)
    e.fy = float(r['grade']); e.section_type = str(r.section_type)
    return e._calculate_cost_co2(float(r['mass']))
err = max(abs(recost(df.iloc[i], False)[0] - float(df.iloc[i]['cost']))
          / float(df.iloc[i]['cost']) for i in idx)
recl = sum(recost(df.iloc[i], True)[2]['reclassified'] for i in idx)
print("flag OFF max rel err: %.3e  (must be < 1e-12)" % err)
print("flag ON reclassified: %d/400  (expect ~197)" % recl)
PY
```

### Step 1 — train (the experiment)

```bash
python research/scripts/train.py \
    --env_type continuous \
    --reward_mode feasibility_gated \
    --economy_metric cost \
    --economy_reward_mode log_relative \
    --log_std_anneal \
    --log_std_ceiling_start 8.0 \
    --log_std_ceiling_end 1.0 \
    --log_std_anneal_start_frac 0.5 \
    --run_name corrcost_gated_merged_seed42 \
    --seed 42 \
    --timesteps 1300000 \
    --n_envs 8 \
    --lr 3e-4 \
    --n_steps 1024 \
    --batch_size 256 \
    --n_epochs 8 \
    --gamma 0.99 \
    --gae_lambda 0.95 \
    --clip_range 0.15 \
    --ent_coef 0.03 \
    --vf_coef 0.5 \
    --max_grad_norm 0.5 \
    --ltb_factor 0.40 \
    --sls_factor 0.50 \
    --out_dir research/models
```

Writes `research/models/corrcost_gated_merged_seed42/{final_model.zip,
vecnormalize.pkl, checkpoint_*_steps.zip, training_config.json}` and
`research/runs/corrcost_gated_merged_seed42_1/events.out.tfevents.*`.

**No CLI flag turns the correction on** — it is the `hss_env.py` default. Confirm
it is active by checking the smoke test in Step 0b passed, and that the run's
learning curve differs from the stored pre-correction run.

### Step 2 — corrected ground truth (needed only for evaluation; ~21 min per metric)

Skip if you still have `research/pretrain_data_corrected/` from E1. Otherwise:

```bash
python research/scripts/regenerate_ground_truth.py \
    --out_dir research/pretrain_data_corrected --metrics cost \
    --pop_size 50 --n_generations 80 --n_restarts 2
python research/scripts/regenerate_ground_truth.py \
    --out_dir research/pretrain_data_corrected --metrics co2 \
    --pop_size 50 --n_generations 80 --n_restarts 2
cp research/pretrain_data/ec3_optimal_designs_mass.csv research/pretrain_data_corrected/
```

Acceptance check: **1623/1728 feasible rows for each metric** (E1 §4), cost
optima median `d/t_w ≈ 72.9 ε`, only ~4/142 costed as welded.

### Step 3 — evaluate the retrained policy against the corrected reference

```bash
python research/scripts/evaluate_with_repair.py \
    --ground_truth_dir research/pretrain_data_corrected \
    --modes none scale scale+thin scale+thin_rolled \
    --models research/models/corrcost_gated_merged_seed42/final_model \
    --include_ga \
    --out_prefix research/results/e3_corrcost_retrain
```

Run **serially** on an idle machine (E2 §2: wall-clock is contention-sensitive;
report EC3 evaluation counts, not seconds).

### Colab

```python
!git clone https://github.com/mo-rc/hss-beam-gen-design.git
%cd hss-beam-gen-design
!pip install -q "stable-baselines3==2.3.2" gymnasium tensorboard
# upload the 6 files from this project's Project Files `code/` folder, then:
!cp manufacturability.py research/envs/ && cp repair.py research/algo/
!cp evaluate_with_repair.py diag_*.py research/scripts/
!cp hss_env.py research/envs/hss_env.py
# run Step 0b, then Step 1 verbatim. Colab CPU is fine; 1.3M steps ~20-40 min.
# n_envs 8 with SubprocVecEnv works on Colab; drop to 4 if you hit worker limits.
```

---

## 5. What this experiment answers, and the numbers to compare against

The question is E1 §6.5's: *is the widened gap "the reference got harder", or
"the policy is mis-specified"?* Read the result against these three, all on the
**corrected** GT, n = 142 contexts, ~**1.72%** optimiser noise floor:

| reference point | unrepaired | + `scale+thin` |
|---|---|---|
| GA control | — | 1.47–1.72% |
| **pre-correction 1.3M checkpoints, transferred** (E1 §6) | **30.15% ±4.50** | **8.53% ±1.47** |
| this run, retrained under corrected costing | *to be measured* | *to be measured* |

Interpretation, agreed in advance:

- Gap recovers materially toward ~25% unrepaired → most of the widening was
  transfer shock; the policy adapts to correct costing.
- Gap stays near 30% → the widening is real mis-specification, and E1 §6.5's
  caveat becomes a finding rather than a caveat.
- Either way this is the **first defensible post-correction PPO number** and is
  what the paper must quote instead of a transfer number.

Single seed → **no confidence interval.** Report it as a point estimate against
the 5-seed pre-correction band and say so explicitly.

---

## 6. One decision left open

**Which seed.** Seed 42 is the canonical first seed and is used above. Note it
was the *worst* of the five pre-correction 1.3M runs (33.9% vs 21.8/22.4/25.1/24.3
for seeds 43–46), and the diagnosis note calls seed **43** "the best
configuration on disk". Seed 42 keeps the numbering convention; seed 43 gives
the fairest like-for-like against the best pre-correction run. Change `--seed`
and the `--run_name` suffix together if you prefer 43. **No other field should
change.**

---

## 7. Files added or changed by this recovery

| path | change |
|---|---|
| `code/hss_env.py` | **new** — full patched environment (reconstructed, verified §2) |
| `code/hss_env_manufacturability.patch` | **new** — the same change as a 30-line diff against `a11e01d` |
| `experiments/E3_corrected-cost-retrain_RUNBOOK.md` | **new** — this document |

No methodology, reward, action space, hyperparameter or costing decision was
changed. Nothing was trained.

**Sources for the manufacturability envelope** (unchanged from E1, restated so
this runbook stands alone): hot-rolled geometry from the
[British Steel universal beams datasheet](https://www.britishsteel.co.uk/wp-content/uploads/2026/02/british-steel-universal-beams-datasheet-190724.pdf);
grade availability from
[EN 10025-6:2019](https://standards.iteh.ai/catalog/standards/cen/77ea2e36-0020-4c39-afef-337cb89046ca/en-10025-6-2019),
a flat-product standard that does not cover hot-rolled sections.
