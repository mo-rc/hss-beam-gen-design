# E5 — `log_std` Anneal Ablation: the E3/E4 Discrepancy Resolved

**Repo state:** `mo-rc/hss-beam-gen-design` @ `3101e63`.
**Date:** 2026-09-05.
**Runs:** `e5_ppo_s43_anneal_logrel`, `e5_ppo_s43_anneal_linear` — both seed 43, 1M steps, `feasibility_gated`, `economy_metric=cost`, `n_envs=8`, corrected costing. Only `--log_std_anneal` and `--economy_reward_mode` vary.
**Also included:** the E4 TD3/SAC checkpoint-grid re-evaluation (§4).

---

## 0. Executive summary

The ablation worked, and the answer is unambiguous.

1. **`log_std_anneal` explains the entire E3/E4 discrepancy. `economy_reward_mode` explains none of it.** Arm B (anneal on, `linear` economy — one flag different from E4) went from **12.36% → 6.86%**, a paired **−5.50 pp, CI [−8.98, −2.44]**. Arm A (anneal on, `log_relative`) reached **6.34%**. A vs B is **−0.52 pp, CI [−1.56, +0.29] — no detectable difference**, and inside the ±0.8 pp noise floor. One flag, five and a half points.
2. **Both arms beat the E3 target.** E3's seed-43 reference at 1M was 7.91%; A lands at 6.34% and B at 6.86%. A even beats E3's over-budget 1.3M result (6.63%). The reproduction is not merely successful, it is better than the thing it was reproducing.
3. **This upgrades the paper's central claim.** With PPO correctly configured, it no longer merely ties 4,800-evaluation random search — it **beats it outright by 5.30 pp (CI [−7.82, −3.21])** while spending 111 EC3 evaluations against 4,871. That is a quality *and* efficiency win over the random-search control, not just an efficiency win.
4. **The E4 head-to-head must be reissued.** Its PPO arm was misconfigured and understated by ~5.5 pp. The GA's margin drops from 9.47 pp to **4.87 pp**, and PPO's win over DDPG grows from a fragile 2 pp to a solid **6.30 pp (CI [−9.63, −3.50])**. See §3.
5. **The E3 "exploration ceiling" interpretation is now settled, and my earlier reading in `E4_head-to-head_RESULTS.md` §4 was wrong.** See §2.3 — I state the correction explicitly rather than quietly editing it.
6. **TD3 and SAC both degrade with training.** TD3's best checkpoint is 24.67% at 200k against 48.07% at 1M; SAC's is 16.27% at 400k against 22.24%. Grade-match decays monotonically for both. This is strong corroboration of the reward/objective misalignment (§4).

---

## 1. The ablation result

`scale+thin` mean cost gap (%), 142 shared contexts, `--seed 0`, corrected ground truth. All seed 43.

| Steps | **A: anneal + log_relative** | **B: anneal + linear** | E3 retrained (reference) | E3 pre-correction | E4 (no anneal + linear) |
|---|---|---|---|---|---|
| 200,000 | 35.92 | 35.53 | 26.82 | 48.64 | — |
| 400,000 | 11.54 | 17.92 | 9.16 | 20.49 | — |
| 600,000 | 9.40 | 9.70 | 10.99 | 10.44 | — |
| 800,000 | 8.42 | 6.42 | 8.89 | 11.67 | — |
| **1,000,000** | **6.34** | **6.86** | **7.91** | 7.84 | **12.36** |

Unrepaired (`none`) on the same grid — A: 158.52, 56.90, 56.54, 39.09, **29.79**; B: 163.04, 117.30, 56.22, 37.32, **40.66**; E3 retrained: 148.48, 63.62, 51.93, 33.31, **27.80**; E4 seed 43 at 1M: **62.20**. The annealed arms roughly halve the unrepaired gap relative to E4, so the improvement is in the generator itself and is not an artefact of the repair operator.

At 1,000,000 steps:

| Arm | gap_mean | feasibility | util_mean | grade_match | EC3 evals/design |
|---|---|---|---|---|---|
| A — anneal + log_relative | **6.34** | 1.000 | 1.001 | **0.838** | 111.0 |
| B — anneal + linear | 6.86 | 1.000 | 1.001 | 0.796 | 114.0 |
| E4 seed 43 — no anneal + linear | 12.36 | 1.000 | 1.001 | 0.810 | 113.5 |

### Paired bootstrap, 20,000 resamples

| A | B | Δ (pp) | 95% CI | Verdict |
|---|---|---|---|---|
| E5-A | E4 seed 43 | **−6.01** | [−9.78, −2.76] | E5-A better, significant |
| E5-B | E4 seed 43 | **−5.50** | [−8.98, −2.44] | E5-B better, significant |
| **E5-A** | **E5-B** | **−0.52** | **[−1.56, +0.29]** | **No detectable difference** |
| E5-A | GA | +4.87 | [+3.89, +5.99] | GA better, significant |
| E5-B | GA | +5.38 | [+4.15, +6.84] | GA better, significant |
| **E5-A** | **Random 4,800** | **−5.30** | **[−7.82, −3.21]** | **E5-A better, significant** |
| E5-B | Random 4,800 | −4.78 | [−6.68, −3.06] | E5-B better, significant |
| E5-A | Random 40 | −27.57 | [−32.33, −23.24] | E5-A better, significant |
| E5-A | DDPG seed 42 | −6.30 | [−9.63, −3.50] | E5-A better, significant |
| E5-A | SAC seed 42 | −15.90 | [−18.57, −13.30] | E5-A better, significant |
| E5-A | Rule-based | −14.13 | [−15.78, −12.52] | E5-A better, significant |

**The B-vs-E4 row is the whole experiment.** B differs from E4 seed 43 in exactly one flag. −5.50 pp with a CI comfortably clear of zero. **`log_std_anneal` is the cause; `economy_reward_mode` is not.**

---

## 2. Mechanism, from the TensorBoard logs

| Quantity | E4 seed 43 (no anneal) | E5-B (anneal, linear) | E5-A (anneal, log_rel) |
|---|---|---|---|
| `log_std_anneal/*` tags present | **absent** | present | present |
| `train/std` first → last | 1.011 → **4.347** | 1.013 → **1.028** | 1.013 → **1.026** |
| `train/std` max | 4.347 | 2.667 | 2.872 |
| `log_std_anneal/actual_mean_std` first → last | — | 1.877 → **1.0001** | 2.048 → **1.0001** |
| `train/entropy_loss` last | **−17.228** | −8.613 | −8.609 |
| `train/clip_fraction` last | 0.226 | **0.390** | **0.392** |
| `train/approx_kl` last | 0.012 | 0.018 | 0.022 |
| `train/explained_variance` last | 0.957 | 0.997 | 0.999 |

The tag presence confirms the callback fired, so the flag took effect. The story is clean: without the ceiling, action standard deviation runs to **4.35 on a `Box(-1,1)` action space** — the sampled action is clipped at a bound in nearly every draw and the policy's stochastic behaviour degenerates toward bang-bang. With the ceiling, std is held at ~1.03 and terminal entropy loss stays at −8.6 rather than collapsing to −17.2.

### 2.1 The pinning recurs, and it is benign

`actual_mean_std` terminates at **1.0001**, i.e. exactly at the annealed ceiling (`log_std_ceiling_end=1.0`), in both E5 arms — the same pinning E3 recorded for 100% of its final 30%. Terminal `clip_fraction` also returns to 0.39, matching E3 exactly, against E4's 0.23.

So the pinning is reproducible and it is **a feature of the healthy configuration, not a symptom of a sick one.** The two best runs in this project's history both pin; the run that did not pin is 5.5 pp worse.

### 2.2 What the ceiling is actually doing

It is a **guard against a misspecified entropy bonus.** `ent_coef=0.03` is very high for continuous control, and it applies constant upward pressure on action variance. The anneal ceiling caps that pressure. Remove the cap and the entropy term wins, variance triples, and the policy stops being able to place its mean precisely near the constrained cost optimum — which is exactly the vertex-seeking behaviour this problem needs.

This reframes the mechanism: the E3 signal was never "the policy wants more variance and is being starved." It was "**the entropy bonus wants more variance and is being correctly restrained.**"

### 2.3 Correction to `E4_head-to-head_RESULTS.md` §4

That section read the E4 regression as evidence that "the ceiling was not the constraint" and that E3's pinning was "a symptom of the entropy bonus fighting the objective, not evidence of harmful under-exploration." The second half of that sentence stands. **The first half was wrong.** The ceiling *was* load-bearing — it is the only thing holding `ent_coef=0.03` in check, and removing it costs 5.5 pp. I inferred "not the constraint" from a single confounded pair; the controlled ablation reverses it. §4 of that document has been annotated accordingly.

### 2.4 The obvious follow-up this creates

If the anneal schedule is compensating for a bad `ent_coef`, then **fixing `ent_coef` directly should match or beat it, with one less moving part.** That is a strictly better method for the paper — a hand-tuned two-parameter variance schedule is a much harder thing to defend to a reviewer than a sensible entropy coefficient. See §5 item 1. This is now the highest-value cheap experiment available.

Note that `ep_rew_mean` is **not comparable between A and B** (A terminates at +8.31, B at −43.77) because `log_relative` and `linear` are different reward scales. Only the gap metric is comparable across them. A's reward advantage is a change of units, not of quality — and indeed its gap advantage over B is not statistically detectable.

---

## 3. Consequences for the E4 head-to-head

**The E4 PPO arm was misconfigured.** All three E4 PPO seeds ran without the anneal ceiling, so the reported 10.94% understates PPO by roughly 5.5 pp. Every PPO row and every PPO comparison in `E4_head-to-head_RESULTS.md` §1–§3 is affected. Corrected picture, using E5-A as the seed-43 estimate:

| Claim | E4 as published | Corrected (E5 config) |
|---|---|---|
| PPO mean gap | 10.94% | ~6.3–6.9% (seed 43; seeds 42/44 pending) |
| PPO vs GA | +9.47 pp, GA wins | **+4.87 pp**, GA still wins |
| PPO vs random 4,800 | −0.67 pp, tie | **−5.30 pp, PPO wins** |
| PPO vs random 40 | −22.93 pp | −27.57 pp |
| PPO vs DDPG | −2.07 pp, fragile (CI to −0.028) | **−6.30 pp, solid** |

Two things this fixes and one it does not:

- **Fixed — the algorithm ranking is now robust.** The PPO–DDPG fragility flagged in that document's §9.2 disappears: 6.30 pp with a CI clear of zero. **DDPG no longer needs seeds 43/44**, which reverses the recommendation made there.
- **Fixed — the headline claim strengthens.** "Matches 4,800-evaluation random search at 43× lower inference cost" becomes "**beats** 4,800-evaluation random search by 5.3 pp at 43× lower inference cost." Materially better, and it is the honest reading of the corrected data.
- **Not fixed — the GA still wins.** 4.87 pp with CI [+3.89, +5.99]. The negative result on absolute design quality survives, and the audit's §7 conclusion that the MDP formulation is the binding constraint is unaffected: 6.34% is still worse than blind sampling in the reparameterised one-shot space (5.9% at 4,800 proposals, 3.1% at 1,000) and far worse than the LOO k-NN predictor (1.87%).

**Do not publish the E4 PPO numbers.** Re-run seeds 42 and 44 with the anneal flag (§5 item 2) and reissue §1–§3 of that document from the corrected arm.

---

## 4. TD3 and SAC checkpoint curves — both degrade with training

`scale+thin` mean gap (%), seed 42:

| Steps | TD3 | TD3 grade_match | SAC | SAC grade_match |
|---|---|---|---|---|
| 200,000 | **24.67** | 0.606 | 20.23 | 0.627 |
| 400,000 | 38.21 | 0.268 | **16.27** | 0.563 |
| 600,000 | 27.20 | 0.190 | 18.53 | 0.500 |
| 800,000 | 44.21 | 0.113 | 23.93 | 0.317 |
| 1,000,000 *(reported)* | 48.07 | 0.211 | 22.24 | 0.310 |

Feasibility is 1.000 and `util_mean` ~1.001 at every point, so nothing here is a feasibility artefact.

**TD3's reported score is roughly twice its best.** Its best checkpoint (24.67% at 200k) is 23.4 pp better than the 1M number the protocol reports. SAC's best (16.27% at 400k) is 5.97 pp better than its reported 22.24%.

**`grade_match` decays monotonically for both** — TD3 0.606 → 0.113 before a partial rebound, SAC 0.627 → 0.310. The agents are progressively *unlearning* correct steel-grade selection while their training reward improves. For SAC we already know reward rose throughout (terminal `ep_rew_mean` −24.06, best of any arm). This is the reward/objective misalignment from `E4_head-to-head_RESULTS.md` §5 shown as a time series, and it is the cleanest evidence in the project that the reward function's optimum is not the design optimum.

**Reporting rule — important.** These best-checkpoint figures are selected post-hoc on the same 142 contexts used for scoring, so they are **not** valid scores and must never be quoted as an arm's result. Report the 1M checkpoint per protocol; present the curve as a diagnostic figure showing training instability and reward misalignment. If a best-checkpoint number is wanted as a legitimate score, it needs early stopping on a held-out context split, which does not currently exist.

---

## 5. What to do next

**1. `ent_coef` sweep, no anneal. One seed each, ~0.12 h per run. Highest value per CPU-hour in the project right now.**
§2.2 says the anneal ceiling is compensating for `ent_coef=0.03`. If a sane coefficient matches it, the method loses a hand-tuned schedule and gains a defensible hyperparameter.

```bash
for EC in 0.0 0.005 0.01; do
  python research/scripts/train.py --env_type continuous \
    --reward_mode feasibility_gated --economy_metric cost \
    --run_name e6_ppo_s43_entcoef${EC} --seed 43 --timesteps 1000000 --n_envs 8 \
    --economy_reward_mode linear --ent_coef $EC
done
```
Evaluate exactly as in §6. Target: match or beat 6.86% (arm B) without the anneal flag. Watch `train/std` — it should settle near 1.0 on its own rather than being clamped there.

**2. PPO seeds 42 and 44 with the anneal flag. ~0.24 h total. Required before anything is published.**
The head-to-head needs a legitimate 3-seed PPO arm. Use arm B's config (`linear`), since A's advantage over B is not detectable and `linear` keeps the economy shaping identical to every other E4 arm — that is the cleaner comparison and one fewer difference to explain.

```bash
for S in 42 44; do
  python research/scripts/train.py --env_type continuous \
    --reward_mode feasibility_gated --economy_metric cost \
    --run_name e5_ppo_s${S}_anneal_linear --seed $S --timesteps 1000000 --n_envs 8 \
    --log_std_anneal --economy_reward_mode linear
done
```

**3. Reissue `E4_head-to-head_RESULTS.md` §1–§3** from the corrected PPO arm once item 2 lands. Keep the E4 no-anneal arm in the paper as a deliberate ablation — it is now a useful result (it quantifies the cost of an unbounded entropy bonus at 5.5 pp), not an embarrassment.

**4. Then the action-space reformulation, unchanged in priority.** Nothing here alters the §7 conclusion of the audit. A correctly configured PPO at 6.34% is still beaten by a zero-training k-NN lookup at 1.87% and by blind sampling in the reparameterised space at 3.1%. The formulation remains the bottleneck; we have simply stopped handicapping the optimiser while making that argument, which makes the argument stronger.

**5. Drop from the plan:** DDPG seeds 43/44 (§3 — the ranking is no longer fragile), and any further `economy_reward_mode` investigation (settled, no detectable effect).

---

## 6. Evaluation commands used

```bash
for RUN in e5_ppo_s43_anneal_logrel e5_ppo_s43_anneal_linear; do
  for S in 200000 400000 600000 800000 1000000; do
    python research/scripts/evaluate_with_repair.py \
      --models research/models/${RUN}/checkpoint_${S}_steps.zip \
      --algo ppo --economy_metric cost \
      --reward_mode_for_env feasibility_gated \
      --ground_truth_dir research/pretrain_data_corrected \
      --modes none scale+thin --n_contexts 142 --seed 0 \
      --out_prefix research/results/${RUN}_curve_${S}
  done
done
```

Note on harness semantics, verified by inspection: `evaluate_with_repair.py` constructs the evaluation environment as `HSSBeamEnv(reward_mode=..., economy_metric=...)` and takes class defaults for everything else, so `economy_reward_mode` is always `linear` at evaluation time regardless of the arm's training configuration. This does **not** bias arm A, because the best-in-episode design is selected by `info[env.economy_metric]` — realised cost, never reward. The evaluation environment is therefore identical across E3, E4 and both E5 arms.

---

## References

All figures computed in this analysis from `research/results/{e5_*,e4_curve_*,e4_*,e5_*}_{summary,per_context}.csv`, `research/models/{e4,e5}_*/training_config.json`, and the TensorBoard event files in `research/runs/{e4,e5}_*`, at repo commit `3101e63`. Bootstrap CIs use 20,000 paired resamples over the 142 shared contexts with `numpy` seed 0. Reference values are from this project's records: `experiments/E4_head-to-head_RESULTS.md`, `experiments/E4_single-agent-algorithm-audit.md`, `experiments/E3_corrected-cost-retrain_RESULTS.md`, `results/E3_RESULTS_MANIFEST.md`, and `diagnosis/2026-09-02_bottleneck-diagnosis-and-reformulation.md`.
