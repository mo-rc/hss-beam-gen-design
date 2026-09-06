# E4 — Head-to-Head Results: PPO vs SAC vs TD3 vs DDPG vs GA

**Repo state:** `mo-rc/hss-beam-gen-design` @ `85022e4` ("e4 audited code experiments models, runs and results added").
**Date analysed:** 2026-09-05.
**Protocol:** as specified in `experiments/E4_single-agent-algorithm-audit.md` §4–§6.
**Coverage delivered:** PPO seeds 42/43/44; SAC, TD3, DDPG seed 42 only; non-RL baselines complete.

---

## 0. Executive summary

The runs are clean and the protocol held. The result is a **negative result on quality and a positive result on efficiency**, which is outcome (2) of the three the audit anticipated in §6.

1. **The GA wins outright on quality.** GA 1.47% vs PPO 10.94% mean cost gap — a paired delta of **+9.47 pp, 95% CI [+7.15, +12.23]**. Not close, not seed noise. The paper's central "RL beats the GA" claim is dead.
2. **PPO wins decisively on inference efficiency.** At a matched 40-evaluation generation budget, PPO is **22.93 pp better than random search (CI [−27.89, −18.22])**, and it *matches* 4,800-evaluation random search (**−0.67 pp, CI [−2.51, +1.28]**, no detectable difference) while spending **112 EC3 evaluations against 4,871 — a 43× reduction**. That is a real, defensible, publishable claim.
3. **Algorithm ranking: PPO > DDPG > SAC > TD3**, and the PPO–DDPG gap is statistically real (−2.07 pp, CI [−3.58, −0.68]). SAC did **not** rescue the exploration problem — it was 11.67 pp *worse* than PPO. The audit's §3.5 judgement call to add SAC was worth making, but the hypothesis it tested is refuted.
4. **Two findings that are more valuable than the head-to-head itself:** the E3 `log_std` ceiling confound is now effectively resolved in the *opposite* direction to the natural reading (§4), and SAC exposes a **reward/objective misalignment** — the arm with the *best* training reward has the *second-worst* cost gap (§5).

One blocking caveat: **the E4 PPO arm is not the E3 PPO arm.** Two config fields differ, so E4's 10.94% cannot be presented in the same table as E3's 6.63% without a labelled explanation (§4).

---

## 1. Primary results table

Mean relative cost gap to the objective-specific EC3 optimum, `scale+thin` repair, 142 shared contexts, `--seed 0`. Gaps as percentages. Lower is better.

| Arm | gap_mean | gap_median | gap_p90 | within_5% | feasibility | EC3 evals/design | grade_match |
|---|---|---|---|---|---|---|---|
| **GA** (over-budget ref) | **1.47** | 0.54 | 4.56 | 0.908 | 1.000 | 4,868 | 0.880 |
| **PPO** (3 seeds, mean) | **10.94** | 6.41 | 14.08 | 0.469 avg | 1.000 | 112.4 | 0.810 |
| — PPO seed 42 | 10.58 | 5.34 | 12.91 | 0.458 | 1.000 | 112.4 | 0.810 |
| — PPO seed 43 | 12.36 | 4.40 | 21.92 | 0.549 | 1.000 | 113.5 | 0.810 |
| — PPO seed 44 | 9.89 | 7.25 | 19.60 | 0.401 | 1.000 | 110.3 | 0.838 |
| Random search, 4,800 (over-budget ref) | 11.60 | 7.01 | 22.61 | 0.362 | 0.993 | 4,871 | 0.725 |
| **DDPG** seed 42 | 12.65 | 6.64 | 19.68 | 0.437 | 1.000 | 113.6 | 0.817 |
| Rule-based | 19.17 | 20.16 | 31.23 | 0.060 | 0.944 | 73.3 | 0.817 |
| **SAC** seed 42 | 22.24 | 20.89 | 48.28 | 0.162 | 1.000 | 83.7 | 0.310 |
| Random search, 40 (**budget-matched**) | 33.22 | 26.87 | 71.01 | 0.072 | 0.979 | 41 | 0.380 |
| **TD3** seed 42 | 48.07 | 40.84 | 92.01 | 0.028 | 1.000 | 92.7 | 0.211 |

PPO across-seed spread: **10.94 ± 1.28 pp SD** (10.58 / 12.36 / 9.89). Every arm passes the feasibility gate at `scale+thin` (1.000), so no arm is disqualified on feasibility; SAC (0.979) and TD3 (0.986) fall below 1.000 only *unrepaired*.

Unrepaired (`none`) gaps, for the record: PPO 64.80 / 62.20 / 38.79, DDPG 99.35, SAC 55.81, TD3 91.93, GA 1.72, random-4800 23.77. Raw generator quality is poor for every RL arm; the repair operator is carrying a large share of the result, which is itself a finding worth stating plainly in the paper.

---

## 2. Paired bootstrap, 20,000 resamples over the 142 shared contexts

Delta = A − B in percentage points; negative means A is better. A CI containing zero is reported as no detectable difference, per the audit's statistical rule.

| A | B | Δ (pp) | 95% CI | Verdict |
|---|---|---|---|---|
| PPO (3-seed) | GA | **+9.47** | [+7.15, +12.23] | GA better, significant |
| PPO (3-seed) | Random 4,800 | −0.67 | [−2.51, +1.28] | **No detectable difference** |
| PPO (3-seed) | Random 40 | **−22.93** | [−27.89, −18.22] | PPO better, significant |
| PPO (3-seed) | Rule-based | −10.99 | [−13.27, −8.49] | PPO better, significant |
| PPO seed 42 | DDPG seed 42 | −2.07 | [−3.58, −0.68] | PPO better, significant |
| PPO seed 42 | SAC seed 42 | −11.67 | [−14.47, −8.86] | PPO better, significant |
| PPO seed 42 | TD3 seed 42 | −37.49 | [−42.54, −32.57] | PPO better, significant |
| DDPG seed 42 | Random 4,800 | +1.05 | [−1.12, +3.57] | No detectable difference |
| DDPG seed 42 | SAC seed 42 | −9.60 | [−12.68, −6.35] | DDPG better, significant |
| GA | Random 4,800 | −10.16 | [−12.84, −7.89] | GA better, significant |

The `n` for pairs involving random search and rule-based is 141/139/134 rather than 142, because contexts infeasible for either arm drop out of the pairing.

**Note the DDPG line.** With the `n_envs` gradient-budget bug fixed (audit §2), DDPG is now statistically indistinguishable from 4,800-evaluation random search and within 2 pp of PPO. Before the fix it was running at half its gradient budget and was never evaluated at all. The fix mattered.

---

## 3. The claim the data entitles us to

Of the three options in audit §6, the data selects the second, and only the second:

> An RL policy matches the quality of a 4,800-evaluation random search at 43× lower inference cost, and beats a budget-matched random search by 23 percentage points — but a genetic algorithm given the same 4,800-evaluation budget remains 9.5 points better in absolute quality.

This must be framed as an **amortised-inference efficiency claim**, not a design-quality claim. The honest one-line framing: *training a policy buys you a 43× cheaper query at inference, at a cost of ~9.5 pp of optimality versus a direct search.* For a designer running one beam, the GA wins. For a designer running thousands of beams interactively, the policy is the only viable option of the two. That is a legitimate engineering contribution and it does not require overclaiming.

`sec_per_design` supports the same story: PPO 0.053 s vs GA 1.105 s, a 21× wall-clock reduction.

---

## 4. The E3/E4 PPO discrepancy — must be resolved before publication

E4 PPO seed 43 reaches **12.36%**. The E3 corrected-cost retrain, **same seed 43**, at its 1,000,000-step checkpoint, reached **7.91%** (and 6.63% at 1.30M). That is a 4.45 pp regression on the same seed, same reward mode, same corrected costing, same 1M budget. It is not seed noise — the E4 seed spread is only ±1.28 pp SD.

Diffing `training_config.json` between `corrcost_gated_merged_seed43` and `e4_ppo_seed43` shows **exactly two differences**, and both are consequential:

| Field | E3 headline (`corrcost_gated_merged_seed43`) | E4 (`e4_ppo_seed{42,43,44}`) |
|---|---|---|
| `log_std_anneal` | `true` (ceiling 8.0 → 1.0, from 50% of training) | **`false`** |
| `economy_reward_mode` | `log_relative` | **`linear`** |

Everything else is identical (lr 3e-4, n_steps 1024, n_envs 8, batch 256, n_epochs 8, gamma 0.99, gae_lambda 0.95, clip_range 0.15, ent_coef 0.03, eta/lambda terms, ltb 0.40, sls 0.50, max_steps 40, rolled manufacturability on).

**Consequence, measured in the TensorBoard logs.** With no `log_std` ceiling and `ent_coef=0.03` still pushing variance up, PPO's `train/std` rose monotonically from 1.01 to **4.46 (seed 42), 4.35 (seed 43), 2.98 (seed 44)** — on a `Box(-1,1)` action space. A standard deviation of 4.46 on a unit-bounded action space means the sampled action is clipped at a bound in the overwhelming majority of draws: the policy's stochastic behaviour is close to bang-bang random. Terminal `entropy_loss` fell to −17.36 against E3's −8.56 starting point. Meanwhile `explained_variance` reached 0.981–0.987, so the critic was again accurate — the failure is policy-side, exactly as in E3.

**Why this is good news scientifically.** E3 recorded the `log_std` ceiling pinning as an unresolved confound: the policy sat pinned at its ceiling for 100% of the final 30% of training, and it was not possible to tell whether it was under-exploring or correctly asking for variance the schedule denied. E4 accidentally ran the anneal-off ablation, and the answer is **the ceiling was not the constraint**. Removing it let variance triple and made the arm **4.45 pp worse**. The E3 pinning was a symptom of the entropy bonus fighting the objective, not evidence of harmful under-exploration.

**Why it is still blocking.** Two fields changed simultaneously, so the 4.45 pp cannot be attributed to `log_std_anneal` alone — `log_relative` vs `linear` economy shaping is a genuine alternative explanation. One 1-seed ablation resolves it (§7, item 1). Until then:

- Do **not** put E3's 6.63% and E4's 10.94% in the same table without labelling the config difference.
- The E4 arms are internally consistent with each other on all four algorithms, so **the head-to-head in §1–§2 is valid as it stands**. Only cross-experiment comparison is affected.

---

## 5. SAC: entropy collapse and a reward/objective misalignment

SAC is the most informative failure in the set, and it is not a bug.

From `research/runs/e4_sac_seed42_2`: auto-tuned `ent_coef` collapsed from 0.967 to **0.0003**, `ep_len_mean` fell from 40.0 to **7.07**, and `ep_rew_mean` reached **−16.93 (best of any arm in the study; terminal −24.06)** against PPO's best of −49.92. Per-context, SAC uses a mean of **7.28 generator steps (min 3)** versus PPO's 39.54.

So SAC found the highest-reward policy available and it is a **bad designer**: 22.24% cost gap, `grade_match` **0.310** vs PPO's 0.810, mean section class 2.24 vs 1.38. It learned to satisfy the 3-step target-utilisation band as fast as possible and terminate — collecting the feasibility-gated reward while never refining toward cost optimality.

This is a **reward-function finding, not an algorithm finding**: under `feasibility_gated` with `linear` economy shaping, the reward's optimum is not the cost optimum, and a sufficiently strong optimiser exploits the difference. PPO's poor sample efficiency was partially *protecting* it from this. This belongs in the paper as a designed-in observation about reward specification in generative structural design, and it independently corroborates the audit's §7 thesis that the MDP formulation — not the algorithm — is the binding constraint.

TD3 shows a related but distinct pathology: `ep_rew_mean` peaked at **−23.19** mid-training and then **degraded to −87.60** by 1M, with `actor_loss` diverging from 0.03 to 2.68 and `critic_loss` from 0.001 to 0.296. `ep_len_mean` settled at 19.02 and `grade_match` at 0.211. TD3 **collapsed late in training**, so evaluating it only at 1M understates it, possibly badly (§7, item 2).

---

## 6. Protocol compliance and provenance checks

| Check | Status |
|---|---|
| All arms reached the 1M budget | **Pass.** PPO 1,007,616 (expected +0.76% rollout overshoot); DDPG 999,885; TD3 999,897; SAC 999,998 |
| 200k checkpoint grid + vecnormalize, all 6 runs | **Pass.** 200k/400k/600k/800k/1M `.zip` + matching `_vecnormalize_*.pkl` present in every run dir |
| `reward_mode=feasibility_gated`, `economy_metric=cost` everywhere | **Pass** (all 6 `training_config.json`) |
| Off-policy `n_envs=1`, `resolved_gradient_steps=1` | **Pass** — the audit's 2× handicap is gone |
| `buffer_size=1_000_000`, `net_arch=[256,256,128]`, `gamma=0.99`, `lagrangian_update_freq=8192` | **Pass**, all three off-policy arms |
| `enforce_rolled_manufacturability=true`, `ltb=0.40`, `sls=0.50`, `max_steps=40` | **Pass**, all 6 arms |
| Same 142 contexts, `--seed 0`, all arms | **Pass.** All per-context files carry 142 unique `(span_m, load_kNm)` pairs |
| Feasibility gate at `scale+thin` | **Pass**, 1.000 for all four RL arms |
| Discarded runs kept out | **Pass.** `ddpg_cost` and `smoketest_td3` are still on disk but produce no `e4_*` results |
| Aborted partial runs | Two exist and are correctly superseded: `e4_ppo_seed43_1` (stopped at 303,104) and `e4_sac_seed42_1` (stopped at 10,574). The `_2` directories are the real full runs. **Delete or clearly mark these two** so no one plots them |
| Evaluated from `checkpoint_1000000_steps.zip`, not `final_model.zip` | **Pass, and now empirically immaterial.** Confirmed by commit `88f5f84`, and §9 re-ran the whole grid from `final_model.zip`: the 3-seed PPO headline moves by 0.04 pp |
| Catalog action-space ablation | **Not run.** PPO-only, reported separately, still outstanding |

---

## 7. What to do next, in priority order

**1. The `log_std_anneal` / `economy_reward_mode` ablation. One seed, ~0.12 h. Do this first.**
It is the cheapest run in the whole project and it is the only thing standing between E3 and E4 being one coherent story. Run seed 43 with the E3 settings restored, changing one field at a time:

```bash
# A: restore both E3 settings -> should reproduce ~7.9% if the configs explain the gap
python research/scripts/train.py --env_type continuous \
  --reward_mode feasibility_gated --economy_metric cost \
  --run_name e5_ppo_s43_anneal_logrel --seed 43 --timesteps 1000000 --n_envs 8 \
  --log_std_anneal --economy_reward_mode log_relative

# B: anneal only, linear economy -> isolates log_std_anneal
python research/scripts/train.py --env_type continuous \
  --reward_mode feasibility_gated --economy_metric cost \
  --run_name e5_ppo_s43_anneal_linear --seed 43 --timesteps 1000000 --n_envs 8 \
  --log_std_anneal --economy_reward_mode linear
```
Evaluate both with the §5.3 harness. If A lands near 7.9% and B does not, the E3 result was driven by `log_relative` shaping; if B also recovers, it was the variance ceiling. Either answer is a paper sentence.

**2. Re-evaluate TD3 and SAC across the checkpoint grid. Zero training, evaluation only.**
TD3's reward peaked mid-training and collapsed by 1M, so its 48.07% is probably not its best policy. Evaluating 200k–1M for both arms costs nothing and either rescues them or produces a clean training-instability curve. Reporting TD3 at 48% when its 600k checkpoint may be at 20% would be an unforced error.

```bash
for S in 200000 400000 600000 800000 1000000; do
  for A in td3 sac; do
    python research/scripts/evaluate_with_repair.py \
      --models research/models/e4_${A}_seed42/checkpoint_${S}_steps.zip --algo $A \
      --economy_metric cost --reward_mode_for_env feasibility_gated \
      --ground_truth_dir research/pretrain_data_corrected \
      --modes none scale+thin --n_contexts 142 --seed 0 \
      --out_prefix research/results/e4_curve_${A}_seed42_${S}
  done
done
```

**3. Decide whether SAC/TD3/DDPG get seeds 43 and 44 — partially revised by §9.2; DDPG now needs its seeds, SAC and TD3 do not.**
The audit's rule is that fewer than 3 seeds cannot support a winner claim. But none of these three arms is a candidate winner: DDPG is 2 pp behind PPO, SAC is 11.7 pp behind, TD3 is 37.5 pp behind, and PPO's own seed spread is only ±1.28 pp. Two more seeds each is ~15.6 CPU-hours to add error bars to three arms that lost. **Report them as single-seed with the limitation stated explicitly**, and spend the compute on item 1 and on the reformulation instead. The only exception: if item 2 shows TD3's mid-training checkpoint is competitive with PPO, TD3 then *is* a candidate and needs its seeds.

**4. ~~Confirm the evaluation checkpoint provenance.~~ Done — see §9.** The full grid was re-evaluated from `final_model.zip` and the headline moves 0.04 pp. Closed.

**5. Then take the audit's §7 path 2 — reformulate the action space.** The head-to-head has now done its job: it is the strongest possible evidence that the algorithm is not the bottleneck. Four algorithms spanning on-policy, deterministic off-policy, twin-critic, and maximum-entropy all land between 10.9% and 48%, while blind sampling in a reparameterised one-shot space reached 5.9% at the same budget and 3.1% at 1,000 proposals, and the LOO k-NN one-shot predictor reached 1.87% mean. **A zero-training nearest-neighbour lookup beats every trained agent in this study by 9 pp.** That is the headline of the next experiment, and it points squarely at one-shot generation over 40-step incremental refinement.

Two supporting items for that reformulation, both now better motivated than they were in the audit:
- **Drop or sharply reduce `ent_coef=0.03`.** §4 shows it drove `train/std` to 4.46 on a unit action space once the ceiling was removed. The audit flagged this as speculative; it is now measured.
- **Soften the Class-4 cliff** (audit §7 path 3). SAC's mean section class of 2.24 and TD3's 2.69, against PPO's 1.38, show the stronger optimisers are being pushed into worse section classes — consistent with the discontinuous `mass=4000` penalty distorting the landscape rather than fencing it off.

---

## 8. What not to do

- **Do not re-run the head-to-head at more seeds hoping for an RL winner.** The GA margin is 9.47 pp with a CI of [+7.15, +12.23]. Seeds will not close that.
- **Do not report E4's 10.94% as an improvement or regression against E3's 6.63%** until item 1 lands. Different configs.
- **Do not present the 4,800-evaluation GA and random-search figures as like-for-like.** They are deliberate over-budget references, per audit §5.4. `random_search_40` is the honest budget-matched control, and PPO beats it by 22.93 pp — that comparison is the one that carries the efficiency claim.
- **Do not quietly drop TD3** because it looks bad. Its collapse is a legitimate reported result and item 2 characterises it properly.

---

## 9. Final-model re-evaluation (added 2026-09-05, repo `7f6f18d`)

The entire evaluation grid was re-run from `final_model.zip` instead of `checkpoint_1000000_steps.zip`, with every other flag identical (`--modes none scale+thin --n_contexts 142 --seed 0`, corrected GT). This closes §6's one open provenance item and, as a bonus, delivers something the study did not previously have: **a measured noise floor.**

### 9.1 Result — the protocol choice does not matter

`scale+thin` mean gap, percentages:

| Arm | 1M checkpoint | final_model | Δ | Paired CI on Δ |
|---|---|---|---|---|
| PPO seed 42 | 10.58 | 11.08 | +0.51 | [−0.35, +1.32] |
| PPO seed 43 | 12.36 | 11.57 | −0.79 | [−2.66, +0.60] |
| PPO seed 44 | 9.89 | 10.04 | +0.15 | [−0.03, +0.36] |
| **PPO 3-seed mean** | **10.94** | **10.90** | **−0.04** | **[−0.72, +0.51]** |
| DDPG seed 42 | 12.65 | 12.65 | **+0.000** | [0.000, 0.000] |
| SAC seed 42 | 22.24 | 22.19 | −0.05 | [−0.17, +0.02] |
| TD3 seed 42 | 48.07 | 47.90 | −0.16 | [−1.84, +1.44] |

Not one arm shows a detectable difference. The audit's concern in §4 — that PPO's +0.76% rollout overshoot (1,007,616 vs 1,000,000 steps) made "exactly 1M steps for every arm" false as written — was correct as a matter of bookkeeping and is now shown to be **worth 0.04 pp in practice**. Every headline conclusion is unchanged:

| Comparison (final_model) | Δ (pp) | 95% CI | Verdict |
|---|---|---|---|
| PPO 3-seed vs GA | +9.42 | [+7.21, +12.12] | GA better, significant |
| PPO 3-seed vs Random 4,800 | −0.71 | [−2.46, +1.11] | No detectable difference |
| PPO 3-seed vs Random 40 | −22.97 | [−27.95, −18.24] | PPO better, significant |
| PPO 3-seed vs Rule-based | −11.03 | [−13.15, −8.69] | PPO better, significant |

Compare against §2: GA margin 9.47 → 9.42, budget-matched margin 22.93 → 22.97. **Report either evaluation; do not report both as independent measurements.** Recommendation: keep `checkpoint_1000000_steps.zip` as the headline, because it is the exactly-equal-budget choice, and cite §9.1 in a footnote as the robustness check. That is a stronger position than either number alone.

### 9.2 What this buys us: a noise floor of ±0.8 pp

The two evaluations are not a null re-run — the weights genuinely differ. Comparing `policy.pth` tensors directly:

| Arm | Tensors differing | Max abs weight diff | Δ gap (scale+thin) |
|---|---|---|---|
| PPO seed 42/43/44 | 17 / 17 (all) | 2.5e−2 / 3.5e−2 / 2.9e−2 | +0.51 / −0.79 / +0.15 |
| DDPG seed 42 | 24 / 32 | 1.8e−3 | +0.000 |
| SAC seed 42 | 42 / 42 (all) | 9.5e−4 | −0.05 |
| TD3 seed 42 | 48 / 48 (all) | 3.0e−3 | −0.16 |

PPO's `final_model` is 7,616 environment steps (≈0.93 of one 8,192-step rollout) past the checkpoint, hence the ~1e−2 weight movement. The off-policy arms drift only ~1e−3, because their `final_model` sits within a handful of single-step gradient updates of the 200k-grid checkpoint.

So **±0.8 pp is an empirical bound on how much the reported gap moves under ~1e−2 of late-training weight drift** on a single seed. Three consequences, and they matter for how the paper is written:

1. **The E3/E4 discrepancy in §4 is real.** 4.45 pp on the same seed is 5.6× this noise floor. It is a config effect, not evaluation jitter, which makes item 1 of §7 worth running.
2. **The PPO–DDPG ranking is fragile and must be stated as such.** On the checkpoint evaluation it was −2.07 pp, CI [−3.58, −0.68]; on the final-model evaluation it is **−1.56 pp, CI [−3.47, −0.028]** — still nominally significant, but the upper bound is 0.028 pp from zero. Two evaluations of the same pair of runs straddling the significance boundary is exactly the situation where a single-seed claim should not be made. **Do not claim PPO > DDPG.** Say they are within ~2 pp and that separating them needs DDPG seeds 43/44. This is the one place §7 item 3's "skip the extra seeds" advice should be revised: if the paper wants an algorithm ordering rather than just "PPO is representative," DDPG needs its two seeds (~4.4 CPU-hours). SAC (−11.7 pp) and TD3 (−37.5 pp) remain far outside the noise floor and still do not need seeds.
3. **PPO's across-seed SD is larger than its within-seed drift** (±1.28 pp checkpoint, ±0.78 pp final-model, vs ±0.8 pp drift). Seed variance and late-training drift are the same order of magnitude here, so any future claim under ~2 pp in this environment needs multiple seeds regardless of how tight its paired CI looks.

### 9.3 One oddity worth a sentence: DDPG is exactly invariant

DDPG's weights moved by up to 1.8e−3 across 24 of 32 tensors, and **all 22 reported metrics were identical to full printed precision** — gap mean/median/SD/p90/p95/worst, within-1/5/10%, utilisation, class, grade match, and EC3 counts, on both repair modes, across all 142 contexts. Not approximately equal; equal.

That is not a bug (the harness is deterministic, as this pair demonstrates). It says DDPG's actor is mapping a whole neighbourhood of weights onto **the same 142 designs** — the policy has converged onto a locally flat, effectively discrete decision surface. Combined with DDPG's mean of 39.54 generator steps (it uses its full 40-step episode) and its low mean section class of 1.35, the reading is that DDPG has settled into a rigid, near-deterministic design rule rather than a context-sensitive policy. Worth one line in the paper as a convergence/degeneracy observation, and worth checking whether its per-context outputs are actually distinct across contexts before describing it as "learned".

### 9.4 Housekeeping confirmed

Commit `c341ff8` removed the duplicate files and the aborted `e4_ppo_seed43_1` run directory flagged in §6. `e4_sac_seed42_1` (the 10,574-step abort) should get the same treatment. Note that the `for S in 42 43 44` loop in the final-model script necessarily produced results only for seed 42 on SAC/TD3/DDPG, since seeds 43/44 of those arms do not exist — the missing outputs are expected, not failures.

---

## References

All figures in §1, §2, §5, §6 and §9 were computed in this analysis directly from `research/results/e4_*_{summary,per_context}.csv`, `research/models/e4_*/training_config.json`, the TensorBoard event files in `research/runs/e4_*`, and direct `policy.pth` tensor comparison inside the model archives, at repo commits `85022e4` (§1–§8) and `7f6f18d` (§9). Bootstrap CIs use 20,000 paired resamples over the 142 shared contexts with `numpy` seed 0. Prior-experiment reference numbers are from this project's own records: `experiments/E3_corrected-cost-retrain_RESULTS.md`, `results/E3_RESULTS_MANIFEST.md`, `diagnosis/2026-09-02_bottleneck-diagnosis-and-reformulation.md`, and the protocol in `experiments/E4_single-agent-algorithm-audit.md`.
