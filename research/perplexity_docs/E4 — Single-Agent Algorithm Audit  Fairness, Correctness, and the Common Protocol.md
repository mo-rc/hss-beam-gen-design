# E4 — Single-Agent Algorithm Audit: Fairness, Correctness, and the Common Protocol

**Scope:** `research/` only. No training was run. All changes below are code/protocol changes verified by static inspection, measurement, and smoke tests.
**Repo state audited:** `mo-rc/hss-beam-gen-design` @ `8a06e5e`.
**Date:** 2026-09-04.

---

## 0. Executive summary

Four things were wrong and are now fixed:

1. **DDPG/TD3 were being handicapped by exactly 2×.** The off-policy trainer defaulted to `--n_envs 2`. SB3 performs `gradient_steps` updates per `collect_rollouts` call, and one call advances `n_envs` timesteps — so at `n_envs=2` the agent got **0.5 gradient updates per environment step instead of 1.0**, while the run still reported "1,000,000 timesteps". Measured directly on this environment (below).
2. **The off-policy arms could silently have been solving a different design problem.** `train.py` passed `ltb_restraint_factor`, `sls_load_factor`, `economy_reward_mode` explicitly; `train_baseline_offpolicy.py` inherited them from class defaults. They agreed *by coincidence*. One default change would have broken comparability with no visible symptom.
3. **The one existing off-policy run is not comparable to the PPO results and must be discarded.** `research/models/ddpg_cost` was trained with `reward_mode="lagrangian"` while the headline PPO arm (`corrcost_gated_merged_seed43`) uses `feasibility_gated` — a *different objective*, not a different algorithm. It also ran at `n_envs=2` (halved updates), checkpointed on a 400k grid instead of 200k, and was never evaluated at all (no `results/*ddpg*` files exist).
4. **SAC was missing.** The comparison contained DDPG and its own direct fix (TD3), but not the strongest off-policy continuous-control method — and the one whose mechanism directly targets E3's diagnosed failure mode.

And one thing is **not** a code bug, which is the most important finding in this document:

> The binding constraint on RL performance here is the **MDP formulation**, not the algorithm choice. Under corrected costing, the GA reaches a **1.47%** mean gap at 4,868 EC3 evaluations, while the retrained PPO agent reaches **6.63%** — and a *blind* sampler in a reparameterised one-shot action space reaches **5.9%** at the same 4,800-evaluation budget, i.e. **random sampling in a better-posed action space already beats a fully trained agent in the current one.** Swapping PPO for SAC is a legitimate and worthwhile test, but it is unlikely on its own to close a 4.5-point gap. Section 7 sets out the decision this forces.

---

## 1. What I verified as already correct

These were checked and need no change. Recording them so they are not re-litigated.

| Component | Verdict |
|---|---|
| `HSSBeamEnv` obs/action spaces | `Box(0,1,(26,))` / `Box(-1,1,(6,))`, identical for every continuous arm |
| EC3 mechanics shared by RL and baselines | `ga_baseline.py` builds a **real `HSSBeamEnv`** and calls `env._ec3_analysis`, `_calculate_cost_co2`, `_constraint_violations` directly — there is no second physics implementation to diverge |
| GA probe env parameters | `ltb=0.40`, `sls=0.50`, `economy_reward_mode=linear`, `enforce_rolled_manufacturability=True` — all match the RL arms |
| Ground truth | Objective-specific (`ec3_optimal_designs_{mass,cost,co2}.csv`), routed through one function, so cost gaps are never read off a mass-optimal geometry |
| Deterministic evaluation | `predict(deterministic=True)` is correct for PPO (mean action) and for DDPG/TD3/SAC (actor output, noise bypassed); `reset(seed)` and `step()` verified bit-reproducible |
| Best-feasible-in-episode reporting | Correct and clearly justified — the 3-step target-band termination rule does not guarantee the last step is the best design |
| EC3 evaluation accounting | `n_ec3` counts generator steps + repair steps; infeasible designs are excluded from gap statistics but still counted in `feasibility`, so an arm cannot buy a better gap by abandoning hard contexts |
| PPO checkpoint cadence | `save_freq = 200_000 // n_envs` — already correct |
| Lagrangian dual ascent | Update cadence keyed to environment timesteps, so it is invariant to vectorisation |

---

## 2. Measured evidence for the gradient-budget bug

Run on this environment with `learning_starts=200`, 2,000 timesteps:

| Configuration | Gradient updates per 1,000 environment steps |
|---|---|
| DDPG `n_envs=1` | **900** (1.0 per env step, post-warmup) |
| DDPG `n_envs=2` *(the old default)* | **450** (0.5 per env step) |
| DDPG `n_envs=4` | 225 |
| PPO `n_envs=8` (minibatch updates) | ≈31 |

Two separate facts fall out, and they must not be confused:

- **The `n_envs=2` default was a bug.** It gave DDPG/TD3 half the learning per environment interaction that the *same algorithm* gets in its standard configuration. Fixed.
- **PPO doing ~31 updates per 1,000 steps while off-policy does ~1,000 is not a bug.** It is the definitional difference between on-policy and off-policy learning. We do **not** equalise it. We equalise **environment steps**, because one environment step is one EC3 analysis, which is the same currency the GA and random-search budgets are accounted in (established in E2). This asymmetry is disclosed in the paper rather than engineered away.

---

## 3. Changes made

### 3.1 `research/scripts/train_baseline_offpolicy.py` — rewritten

| Change | Why |
|---|---|
| `--n_envs` default `2 → 1` | Restores 1.0 gradient update per environment step (§2) |
| `--gradient_steps` defaults to `n_envs * train_freq` | Makes the update-per-env-step ratio invariant if `n_envs` is raised for speed, so the bug cannot reappear |
| Added **SAC** | §3.5 |
| `--reward_mode` is now **required** | It defaulted to `lagrangian` while the headline PPO arm uses `feasibility_gated`. That default silently produced the non-comparable `ddpg_cost` run |
| Added `--ltb_factor`, `--sls_factor`, `--max_steps`, `--economy_reward_mode`, `--no_rolled_manufacturability`, all passed explicitly | Removes the coincidental-default coupling; the two trainers now have an identical environment-shaping surface |
| `save_freq = checkpoint_every // n_envs`, `save_vecnormalize=True` | Puts off-policy checkpoints on the same 200k environment-step grid as PPO, so learning curves are plottable together |
| `buffer_size` `200_000 → 1_000_000` | The old value discarded 80% of a 1M-step run's experience for no stated reason; the observation is 26-dim so a full buffer costs well under 1 GB |
| `net_arch` default `[256,256,128]`, `gamma` exposed and defaulted to `0.99` | Explicitly matched to the PPO arms |
| `NormalActionNoise` wrapped in `VectorizedActionNoise` when `n_envs>1` | A bare `NormalActionNoise` emits one `(6,)` sample that broadcasts across all envs, giving every env *identical* exploration noise |
| `lagrangian_update_freq` default `2048 → 8192` | Matches the PPO arms' `n_steps * n_envs` (1024×8), so dual ascent runs at the same environment-step cadence in both paths |
| `training_config.json` now records `resolved_lr`, `resolved_gradient_steps`, `enforce_rolled_manufacturability` | Provenance: the existing `ddpg_cost` config cannot tell us which costing model it used |

### 3.2 `research/scripts/train.py`

Added `--max_steps` and `--no_rolled_manufacturability` and passed both explicitly into `make_env`; recorded `enforce_rolled_manufacturability` in `training_config.json`. **Both defaults reproduce every existing run bit-for-bit** — this is a surface-parity change, not a behaviour change.

### 3.3 `research/scripts/run_multiseed.py`

`cmd_train` hardcoded `train.py`, so **there was no multi-seed path for the off-policy baselines at all** — which is precisely why `ddpg_cost` exists as a lone, unevaluated seed. Added `--algo` routing to `train_baseline_offpolicy.py`, a separate `--offpolicy_n_envs` (default 1) so PPO's `--n_envs 8` cannot leak into the off-policy path, and a hard error if `--env_type catalog` is combined with a continuous-only algorithm.

### 3.4 Evaluation harnesses

- `evaluate.py`, `evaluate_with_repair.py`, `generalization_test.py`, `run_multiseed.py`: `sac` added to every algorithm map and `--algo` constrained to `{ppo,ddpg,td3,sac}` everywhere (`evaluate_with_repair.py` previously accepted any string and would `KeyError` late).
- `evaluate_with_repair.py`: added a **hard abort** if a run's saved `vecnormalize.pkl` has `norm_obs=True`. The harness deliberately evaluates the bare policy without VecNormalize; that is only sound because every trainer sets `norm_obs=False`. It was an undocumented invariant, and violating it would have silently produced meaningless gap numbers rather than an error.

### 3.5 Why SAC was added (a judgement call — flagging it explicitly)

This is the one change that adds an arm rather than fixing a defect, so the reasoning should be on the record and you should overrule it if you disagree:

- DDPG alone is a weak 2020s baseline; TD3 is DDPG's own twin-critic/delayed-policy fix. Shipping only those two makes "no RL method beat the GA" a much less interesting negative result.
- SAC's exploration temperature is **auto-tuned**. E3 found the PPO policy pinned at its annealed `log_std` ceiling for **100% of the final 30% of training** — the policy was asking for more action variance than the hand-set schedule allowed. SAC is the principled algorithmic response to exactly that symptom, not an arbitrary third algorithm.
- It is additive: it changes no existing arm and invalidates no existing result.

### 3.6 New: `research/tests/test_algorithm_parity.py`

A preflight that converts "the arms are comparable" from an assumption into assertions. **All checks currently pass.** It verifies: identical problem-defining env attributes and spaces across both trainers; `--reward_mode` required in both; gradient-budget invariance to `n_envs` (with a regression guard on the old 0.5 ratio); checkpoint grids in environment timesteps; `norm_obs=False` everywhere; every trainable algorithm loadable by both evaluators; GA sharing the RL arms' physics and env parameters; and `reset`/`step` determinism.

Run it before launching the comparison and again before any head-to-head number goes in the paper.

---

## 4. The common training budget

**1,000,000 environment timesteps per seed, for every RL arm.**

- **Why environment steps:** one environment step is one EC3 analysis. This is the same currency the GA (4,868 evaluations) and random search (4,800) are budgeted in, per E2. It is the only currency in which RL and non-RL arms can be compared at all.
- **Why 1M:** matches the existing PPO default and the E1/E2 five-seed reference. Note the E3 headline run reached 1,302,528 steps via a resume; for the head-to-head it must be **evaluated at its 1,000,000-step checkpoint**, with the 1.3M result reported separately as an over-budget observation.
- **Evaluate from `checkpoint_1000000_steps.zip`, never from `final_model.zip`.** Verified in final checks: SB3's PPO only stops on a whole-rollout boundary, so `learn(total_timesteps=1_000_000)` with `n_steps=1024, n_envs=8` (rollout = 8,192) actually runs **1,007,616** steps, while the off-policy arms stop at exactly 1,000,000. The overshoot is only +0.76% and would not change a conclusion, but it makes "exactly 1,000,000 steps for every arm" false as written. The 200k checkpoint grid lands exactly on 1,000,000 for all four algorithms (confirmed against the existing `n_envs=8` runs in `research/models/`), so evaluating the checkpoint rather than the final model makes the budget exactly equal at zero cost and with no code change.
- **Seeds:** 42, 43, 44 minimum; 42–46 preferred. Fewer than 3 seeds cannot support a winner claim — E3 already established that single-seed deltas here have bootstrap CIs spanning zero.
- **Disclosed, not equalised:** PPO ≈31 minibatch updates per 1,000 env steps vs ≈1,000 for off-policy (§2).

**Measured throughput** (this sandbox's CPU; your local machine will differ, but the *ratios* should hold):

| Arm | steps/s | 1M steps |
|---|---|---|
| PPO (`n_envs=8`) | ~2,290 | **~0.12 h/seed** |
| DDPG (`n_envs=1`) | ~125 | **~2.2 h/seed** |
| TD3 (`n_envs=1`) | ~125 | **~2.2 h/seed** |
| SAC (`n_envs=1`) | ~82 | **~3.4 h/seed** |

3 seeds × 4 algorithms ≈ **24 CPU-hours**; 5 seeds ≈ **40 CPU-hours**. Run PPO first (it is nearly free), then SAC, then TD3, then DDPG — that order gets the most informative arms done earliest if you have to stop.

---

## 5. Exact commands

Run everything from the repo root with `PYTHONPATH` set to the repo root. Prefix with `TMPDIR=...` only if your temp volume is small.

### 5.1 Preflight (run once, before any training)

```bash
python -m research.tests.test_algorithm_parity
```
Exit code must be 0.

### 5.2 Training — one command per algorithm, all seeds

```bash
# PPO  (~0.12 h/seed)
python research/scripts/run_multiseed.py train \
  --algo ppo --reward_mode feasibility_gated --economy_metric cost \
  --env_type continuous --run_prefix e4_ppo --seeds 42 43 44 \
  --timesteps 1000000 --n_envs 8 --economy_reward_mode linear

# SAC  (~3.4 h/seed)
python research/scripts/run_multiseed.py train \
  --algo sac --reward_mode feasibility_gated --economy_metric cost \
  --run_prefix e4_sac --seeds 42 43 44 \
  --timesteps 1000000 --offpolicy_n_envs 1 --economy_reward_mode linear

# TD3  (~2.2 h/seed)
python research/scripts/run_multiseed.py train \
  --algo td3 --reward_mode feasibility_gated --economy_metric cost \
  --run_prefix e4_td3 --seeds 42 43 44 \
  --timesteps 1000000 --offpolicy_n_envs 1 --economy_reward_mode linear

# DDPG (~2.2 h/seed)
python research/scripts/run_multiseed.py train \
  --algo ddpg --reward_mode feasibility_gated --economy_metric cost \
  --run_prefix e4_ddpg --seeds 42 43 44 \
  --timesteps 1000000 --offpolicy_n_envs 1 --economy_reward_mode linear
```

Every arm therefore gets: same env, same `reward_mode=feasibility_gated`, same `economy_metric=cost`, same corrected costing, same `max_steps=40`, same `[256,256,128]` network, same `gamma=0.99`, same 1M environment steps, same seeds, same 200k checkpoint grid.

If you prefer to invoke the trainers directly rather than through the multi-seed driver, the equivalent single-seed form is:

```bash
python research/scripts/train_baseline_offpolicy.py --algo sac \
  --reward_mode feasibility_gated --economy_metric cost \
  --run_name e4_sac_seed42 --seed 42 --timesteps 1000000 --n_envs 1
```

### 5.3 Evaluation — identical for every arm

```bash
for RUN in e4_ppo e4_sac e4_td3 e4_ddpg; do
  case $RUN in e4_ppo) A=ppo;; e4_sac) A=sac;; e4_td3) A=td3;; e4_ddpg) A=ddpg;; esac
  for S in 42 43 44; do
    python research/scripts/evaluate_with_repair.py \
      --models research/models/${RUN}_seed${S}/checkpoint_1000000_steps.zip \
      --algo $A \
      --economy_metric cost \
      --reward_mode_for_env feasibility_gated \
      --ground_truth_dir research/pretrain_data_corrected \
      --modes none scale+thin \
      --n_contexts 142 --seed 0 \
      --out_prefix research/results/${RUN}_seed${S}
  done
done
```

`--seed 0` is fixed across every arm, so all arms are scored on the **same 142 contexts with the same per-context initial states**. This is the single most important line in the protocol — do not vary it per arm.

Verified in final checks: the corrected ground truth contains **exactly 142 unique (span, load) contexts**, so `--n_contexts 142` selects the complete set and there is no context-sampling variance between arms at all. Verified separately: no trainer reads `pretrain_data*` or any ground-truth file, so there is no train/evaluation leakage.

### 5.4 Non-RL baselines (reference numbers, already established in E1/E2)

```bash
# GA, rule-based, and BOTH random-search controls in one run.
# --random_evals takes multiple values, so 40 (budget-matched to one
# 40-step RL rollout) and 4800 (deliberate over-budget reference) are
# produced as separate arms by a single command.
python research/scripts/evaluate_with_repair.py \
  --include_ga --ga_pop 60 --ga_gen 80 \
  --include_rule_based \
  --random_evals 40 4800 \
  --economy_metric cost --reward_mode_for_env feasibility_gated \
  --ground_truth_dir research/pretrain_data_corrected \
  --modes none scale+thin --n_contexts 142 --seed 0 \
  --out_prefix research/results/e4_nonrl_baselines
```

> **Corrected in final verification:** an earlier draft of this section used `--ga`, which the parser rejects. The flags are `--include_ga` and `--include_rule_based`. Verified by dry-parsing against the current repo.

The 4,800-evaluation GA and random-search figures are a deliberate over-budget upper reference, not a like-for-like comparison, and must be labelled as such in the paper. `--random_evals 40` is the honest budget-matched control.

The catalog arm (`--env_type catalog`) is **PPO-only** and stays out of the head-to-head: its action space is `MultiDiscrete`, so DDPG/TD3/SAC cannot train on it. It is an action-space ablation, reported separately.

---

## 6. How the winner is decided

**Primary metric: mean relative cost gap to the objective-specific EC3 ground-truth optimum, under the `scale+thin` repair mode, over the 142 shared contexts, averaged across seeds.** Lower is better.

Reported alongside it, for every arm:

| Metric | Role |
|---|---|
| `gap_mean` (`scale+thin`) | **Primary.** The headline number |
| `gap_mean` (`none`) | Raw generator quality, before any repair |
| `gap_median`, `gap_p90` | Distribution shape — catches an arm that wins on average by being lucky on easy contexts |
| `feasibility` | **Gate.** An arm with feasibility < 1.000 cannot be declared winner regardless of gap; infeasible designs are excluded from gap stats, so a low-feasibility arm's gap is not trustworthy |
| `within_5pct` | Fraction of contexts at near-optimal cost — practical usefulness |
| `ec3_evals_per_design` | The budget actually consumed at inference. An arm that wins while spending more EC3 calls has not won on equal terms |
| `util_mean` | Sanity check: should sit near the 0.90–1.05 target band |
| `grade_match` | Whether the agent picks the ground truth's steel grade — the paper's HSS-specific claim |

**Statistical rule.** Report mean ± SD across seeds. For any head-to-head claim, run a **paired bootstrap over the 142 shared contexts** (arms are paired by context, so this is far more powerful than an unpaired test) and report the CI. **A difference whose 95% CI includes zero is reported as "no detectable difference", not as a win.** E3 already demonstrated why this matters here: a 0.90-point improvement had a CI of [−2.92, +0.73].

**The claim we are entitled to make** is then whichever of these the data supports:
- an RL arm beats the GA on `gap_mean` with a CI excluding zero → the paper's central claim stands;
- an RL arm beats the GA *at matched inference budget* (RL's ~109 EC3 evals/design vs GA's 4,868) but not in absolute gap → a genuine and publishable efficiency claim, but it must be stated as an efficiency claim, not a quality claim;
- no RL arm beats the GA on either → we do **not** force it. Section 7.

---

## 7. The decision this audit forces

The code is now fair, and I am reasonably confident these runs will be scientifically clean. I am **not** confident they will produce an RL winner, and it would be dishonest to hand you a protocol without saying why.

The E3 evidence points at the problem formulation, not the optimiser:

- GA under corrected costing: **1.47%** mean gap (4,868 evals).
- Retrained PPO: **6.63%** (`scale+thin`, ~109 evals/design).
- Blind sampling in a **reparameterised one-shot action space**: **5.9%** at 4,800 evals, **3.1% mean / 2.4% median** over 1,000 proposals.

That third line is the uncomfortable one. Random sampling in a better-posed action space already beats a fully trained agent in the current one. That is a signature of the *MDP formulation* being the bottleneck — specifically the 40-step incremental-refinement action space, and the Class-4 short-circuit that returns a discontinuous `mass=4000` cliff which no policy gradient can usefully traverse.

So there are three paths, and I have deliberately **not** taken any of them without your sign-off, since each is a methodology change rather than a bug fix:

1. **Run the protocol as written first.** ~24 CPU-hours for 3 seeds. It is the honest next step regardless: it either produces an RL winner, or it produces the strongest possible evidence that the algorithm is not the problem — which is exactly what justifies path 2 in the paper. **This is my recommendation.**
2. **Reformulate the action space** (one-shot / reparameterised generation instead of 40-step incremental refinement) and re-run. The E3 probe says this is where the headroom is. It is a new experiment and changes the paper's framing from "which RL algorithm" to "which RL formulation".
3. **Soften the Class-4 cliff** into a continuous penalty. Smaller change than 2, plausibly unblocks gradient flow, but still a reward-function change and therefore a new experiment.

Two smaller items I also left alone for the same reason, both worth revisiting only if PPO underperforms SAC/TD3 in a way that points at them:
- PPO's `ent_coef=0.03` is very high for continuous control (typical: 0.0), and it pushes action std *up* — plausibly interacting with the observed `log_std` ceiling pinning.
- PPO's `clip_range=0.15` with a measured `clip_fraction` of 0.39 means roughly two in five samples are clipped, which is high.

**Discard before you start:** `research/models/ddpg_cost` (wrong reward mode, halved gradient budget, unverifiable costing provenance, never evaluated) and `research/models/smoketest_td3`. Do not report either.

---

## References

All figures in §2 and §4 were measured directly in this audit on the repo at `8a06e5e`. The E1/E2/E3 reference numbers cited in §0 and §7 are from this project's own prior experiment records: `experiments/E1_rolled-vs-welded-costing-correction.md`, `experiments/E2_ga-timing-and-equal-budget-claims.md`, and `experiments/E3_corrected-cost-retrain_RESULTS.md`.
