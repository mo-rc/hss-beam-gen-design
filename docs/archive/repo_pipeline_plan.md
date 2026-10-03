# Clean Repository, Execution Pipeline, and Figures Plan

*Planning document only — nothing below has been executed or committed. This defines the structure, file mapping, run order, and figure set for you to review before anything moves.*

---

## 1. Strategy: new branch, old history preserved

Recommendation: create a new branch `paper-release` from `main` (not `research-next`), and build the clean structure there. Keep `research-next` exactly as-is, untouched, as the working/audit trail — do not delete it. This gives you:
- A clean, publication-ready structure at the link you'd put in the paper.
- The full messy history still available in git log for reproducibility claims ("full experimental history available on request/in branch X") without it cluttering what a reviewer or reader sees by default.
- No risk of losing anything the audit flagged as "diagnostic but not headline" (E3, E4, Exp1) — these get archived, not deleted.

If you'd rather this be a brand-new separate repository instead of a branch, everything below still applies — only the `git checkout -b` step at the start changes to a fresh `git init`.

---

## 2. Clean directory structure

```
hss-beam-rl-design/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── configs/
│   └── ppo_final.yaml          (frozen RL hyperparameters, each cited to the experiment that set it)
├── .github/workflows/tests.yml (optional CI)
├── src/
│   ├── envs/
│   │   ├── hss_env.py
│   │   ├── manufacturability.py
│   │   ├── rolled_catalog.py
│   │   └── rolled_catalog.csv
│   └── algo/
│       ├── repair.py
│       ├── lagrangian.py
│       └── log_std_anneal.py
├── pipeline/
│   ├── 01_generate_ground_truth.py
│   ├── 02_train_ppo.py
│   ├── 03_evaluate_policy.py
│   ├── 04_baseline_ga_de.py
│   ├── 05_baseline_knn.py
│   ├── 06_ood_generalization.py
│   ├── 07_handcalc_validation.py
│   ├── 08_make_figures.py
│   └── 09_build_paper_tables.py
├── data/
│   └── ground_truth/
│       ├── main_grid/
│       ├── ood_span/
│       ├── ood_load/
│       └── ood_joint/          (documented as empty — see README note)
├── models/
│   ├── ppo_seed{42..46}/                  (cost headline = winning arm of Step 2a)
│   ├── ppo_{mass,co2}_seed{42..46}/       (Step 2c transfer check)
│   ├── ablation_reward_{mode}_seed{..}/   (Step 2a)
│   └── ablation_algo_{algo}_seed{..}/     (Step 2b)
├── results/
│   ├── evaluation/
│   ├── baselines/
│   ├── ood/
│   └── tables/
├── figures/
├── tests/
└── docs/
    ├── methodology.md
    └── archive/
        └── research_audit.md    (this project's full messy history + audit, kept for transparency)
```

---

## 3. File disposition — what moves, renames, merges, or gets archived

| Current file/dir | Disposition | New name / location | Why |
|---|---|---|---|
| `envs/hss_env.py`, `manufacturability.py`, `rolled_catalog.py(.csv)` | Keep, move | `src/envs/` | Core physics, unchanged, no rename needed beyond relocation |
| `algo/repair.py`, `lagrangian.py`, `log_std_anneal.py` | Keep, move | `src/algo/` | Core algorithm code, stable |
| `algo/log_std_anneal_adaptive.py` | **Archive** | `docs/archive/` | Superseded — the non-adaptive version is what's actually used by the authoritative checkpoints |
| `scripts/regenerate_ground_truth.py` | Rename | `pipeline/01_generate_ground_truth.py` | Add `meta.json` provenance output (audit item, see §5) |
| `scripts/train.py`, `train_baseline_offpolicy.py`, `resume_training.py`, `manual_resume_from_checkpoint.py`, `run_multiseed.py` | **Merge → one file** | `pipeline/02_train_ppo.py` | Currently 5 overlapping entrypoints for the same task; one script with `--algo {ppo,sac,td3,ddpg}` and `--seed` flags is clearer for a reader |
| `scripts/evaluate_with_repair.py` | Rename | `pipeline/03_evaluate_policy.py` | This is the main evaluation entrypoint; `evaluate.py` (no repair) is folded in as a `--repair none` mode instead of a separate file |
| `scripts/evaluate.py` | **Merge into 03**, then archive | — | Redundant with `evaluate_with_repair.py --modes none` |
| `scripts/expA_strong_baselines.py` + `expA2_pipeline_baselines.py` | **Merge → one file** | `pipeline/04_baseline_ga_de.py` | These two together are "the GA/DE baseline pipeline"; keeping them as two separate 700+/450-line files with overlapping helper functions is exactly the kind of thing a reviewer will ask about |
| `scripts/ga_baseline.py` | Keep as internal module | `src/algo/ga_search.py` | Called by `04_baseline_ga_de.py`, not a standalone entrypoint |
| `scripts/expA3_amortized_baselines.py` | Rename | `pipeline/05_baseline_knn.py` | — |
| `scripts/expA5_ood_amortizer.py` + `scripts/generalization_test.py` | **Merge → one file** | `pipeline/06_ood_generalization.py` | Same reasoning as 04 |
| `scripts/expA7_handcalc_comparison.py` | Rename, **fix the bug** | `pipeline/07_handcalc_validation.py` | Carry forward the `section_type`/geometry fix already applied |
| `scripts/aggregate_seeds.py`, `c1_verify_manuscript_numbers.py`, `c2_aggregate_generalization.py`, `c2_aggregate_multiseed.py`, `c3_combined_gap_table.py` | **Merge → one file** | `pipeline/09_build_paper_tables.py` | These five each aggregate a slightly different subset; one script producing all final paper tables from `results/` is much easier for a reader to follow and re-run |
| `scripts/diag_reparam_probe.py`, `diag_reparam_probe_costing_ab.py`, `diag_gap_decomposition.py`, `diagnose_step_scale_timing.py`, `grade_policy_analysis.py` | **Archive** | `docs/archive/` | One-off diagnostics that produced the evidence behind specific claims in the methodology write-up (action-space reparameterization, gap decomposition) — valuable as provenance, not part of the reproducible pipeline a reader needs to re-run |
| `scripts/extend_seed42.sh`, `extend_seed42_pure_settling.sh`, `tb_parse.py` | **Delete** | — | Ad hoc, single-use utility scripts tied to specific past debugging sessions, no ongoing value |
| `pretrain_data/`, `pretrain_data_corrected/`, `pretrain_data_corrected_v2/` | **Superseded by regeneration** | `data/ground_truth/main_grid/` | One clean regeneration under fully-current `hss_env.py` (see §4, step 1) replaces all three |
| `pretrain_data_ood_{span,load,joint}/` | **Superseded by regeneration** | `data/ground_truth/ood_{span,load,joint}/` | Same — regenerate once cleanly |
| `models/gated_cost_catalog_seed42`, `corrcost_gated_merged_seed43`, `smoketest_*`, `test_run` | **Archive (do not retrain/re-include)** | `docs/archive/legacy_checkpoints_manifest.md` (a manifest listing what each was and why it's archived, not the checkpoint binaries themselves, to keep the clean repo small) | Narrow, single-purpose diagnostics (discrete-catalog probe, the E3 retrain-recovery test, smoke tests) that answered their question already and don't need repeating |
| `models/gated_cost_seed*`, `shaped_cost_seed*`, `lagrangian_cost_seed*`, `gated_cost_merged_seed*` | **Archive originals; re-run as Step 2a** | manifest entry + fresh runs at `models/ablation_reward_{mode}_seed{seed}/` | Pre-fix reward-mode ablation, per your note — re-run properly under corrected ground truth instead of assuming the old ranking still holds |
| `models/ddpg_cost`, `e4_sac_seed42`, `e4_td3_seed42`, `e4_ddpg_seed42` | **Archive originals; re-run as Step 2b** | manifest entry + fresh runs at `models/ablation_algo_{algo}_seed{seed}/` | Pre-material-cost-fix algorithm comparison, per your note — re-run under the winning reward mode from 2a and corrected ground truth |
| `models/e4_ppo_seed{42-44}`, `e5_ppo_s43_anneal_logrel` | **Archive as diagnostic**, keep manifest entry | same as above | `log_std_anneal`/`economy_reward_mode` ablations that established the current best PPO configuration — the configuration itself carries forward into Step 2, the specific old checkpoints don't need re-running |
| `models/e5_ppo_s{42-46}_anneal_linear` | **Superseded by Step 2a's winning arm** | `models/ppo_seed{42-46}/` | No separate PPO-only retrain needed — see restructured Step 2 |
| `perplexity_docs/*.md` | **Archive as a single consolidated history doc** | `docs/archive/full_experimental_history.md` (concatenated, chronologically ordered) | Valuable narrative provenance; a reader shouldn't need to open 9 separate files to understand how the project got here |
| `manuscript_final_comprehensive.md`, `manuscript_expA_section_draft.md`, `results_validation_draft.md` | **Superseded by the new manuscript** | `docs/archive/` once the new manuscript (§7, separate future task) exists | Do not delete until the new manuscript is written and approved |
| `tests/*` | Keep, move | `tests/` (unchanged) | Already well-organized; run once as a sanity gate before step 1 |
| `results/*`, `runs/*` | **Regenerated fresh** | `results/`, discard old `runs/` TensorBoard logs from superseded checkpoints | Old result CSVs tied to superseded checkpoints/GT get replaced by the pipeline's fresh output; `runs/` (TensorBoard event files) are large and not needed for the paper |

---

## 4. Sequential execution pipeline

Every step lists what it depends on, what it produces, and roughly how expensive it is, so you can plan hardware/time. **Nothing here has been run.**

### Step 0 — Sanity gate (CPU, minutes)
```bash
pytest tests/
```
Confirms EC3 physics, repair operator, and optimizer independence tests pass on the current code before spending any compute downstream. Depends on nothing.

### Step 1 — Ground truth generation (CPU, ~1-2 hours total across all four grids)
```bash
python pipeline/01_generate_ground_truth.py --grid main   --out data/ground_truth/main_grid
python pipeline/01_generate_ground_truth.py --grid ood_span --out data/ground_truth/ood_span
python pipeline/01_generate_ground_truth.py --grid ood_load --out data/ground_truth/ood_load
python pipeline/01_generate_ground_truth.py --grid ood_joint --out data/ground_truth/ood_joint
```
Generates mass, cost, and CO₂ ground truth together (unlike the current `pretrain_data_corrected_v2`, which only has cost), and writes a `meta.json` per directory (git commit hash, script args, generation date) — closing the provenance gap the audit flagged. `ood_joint` is expected to come back with zero feasible contexts; the script should say so explicitly in its own console output and in `meta.json` rather than silently producing an empty file. Depends on nothing else (Step 0 optional gate).

### Step 2 — Training, in three tiers (GPU, the expensive step overall — see cost breakdown below)

Your two points both land here, and restructuring this as three tiers avoids retraining anything twice: the reward-mode ablation's winning arm *becomes* the final headline batch, rather than being trained once for the ablation and then retrained again as "the real one."

**Step 2a — Reward-mode ablation (re-run of Exp1, post-fix), 5 seeds.** This is the one that answers "does `feasibility_gated` still win once the costing/manufacturability defects Exp1 never saw are fixed?" — Exp1's original conclusion was drawn entirely on pre-E1, pre-cost-fix ground truth (audit §4), so it's not safe to just assume the old answer still holds.
```bash
for mode in feasibility_gated lagrangian shaped; do
  for seed in 42 43 44 45 46; do
    python pipeline/02_train_ppo.py --algo ppo --seed $seed \
      --reward_mode $mode --log_std_anneal --economy_reward_mode log_relative \
      --economy_metric cost --timesteps 1000000 \
      --out models/ablation_reward_${mode}_seed${seed}
  done
done
```
15 runs (3 reward modes × 5 seeds). **The 5 seeds trained under whichever reward mode wins this ablation are the final headline batch** — copy/symlink them to `models/ppo_seed{42-46}/` rather than training a fourth time.

**Step 2b — Algorithm comparison (re-run of E4, post-fix), under the winning reward mode from 2a, 5 seeds.** This directly answers your second point: PPO is only one option, and E4 already built the infrastructure for SAC/TD3/DDPG — it just needs re-running on corrected ground truth, same as everything else the audit flagged.
```bash
for algo in sac td3 ddpg; do
  for seed in 42 43 44 45 46; do
    python pipeline/02_train_ppo.py --algo $algo --seed $seed \
      --reward_mode <winner_from_2a> --economy_metric cost --timesteps 1000000 \
      --out models/ablation_algo_${algo}_seed${seed}
  done
done
```
15 runs.

**Step 2c — Mass/CO₂ transfer check (Option C: winning configuration only, not a full re-ablation), 5 seeds each.** Rather than repeating 2a+2b for mass and CO₂ (Option B — 3× the compute above, mostly re-answering a question already answered), take the single winning reward-mode + algorithm combination from 2a/2b and check it transfers to the other two objectives:
```bash
for metric in mass co2; do
  for seed in 42 43 44 45 46; do
    python pipeline/02_train_ppo.py --algo <winning_algo> --seed $seed \
      --reward_mode <winner_from_2a> --economy_metric $metric --timesteps 1000000 \
      --out models/ppo_${metric}_seed${seed}
  done
done
```
10 runs. This supports "the approach transfers to mass and CO₂ objectives using the same configuration established for cost" in the paper — a checked claim, not an assumed one. It also means Steps 4/5 (GA/DE and kNN baselines) need to run once per objective (`--economy_metric {cost,mass,co2}`) rather than cost-only, to have something to compare each of 2c's two new checkpoint batches against.

**Cost tally, all of Step 2: 15 + 15 + 10 = 40 training runs at 1M steps each.** This is the single biggest compute commitment in the whole pipeline — worth being explicit about before it starts. If it turns out to be more than your hardware/time budget can absorb, the cheapest lever is seed count on 2b/2c specifically (comparison/transfer arms, not the headline), not on 2a's winning arm.

Every tier depends only on the current `src/envs/hss_env.py` (training samples contexts on the fly; does not read Step 1's ground-truth files at all — worth stating plainly in the methodology doc, since it surprises readers used to supervised-learning pipelines).

**Config freeze (new — closes the "do I have the right RL configuration" question).** The EC3 physics is already covered by an independent test suite (`tests/`), but the RL side has no equivalent: hyperparameters (`ltb_factor`, `sls_factor`, `eta_util/class/geom`, `lambda_max`, `log_std_anneal_*`, `economy_reward_mode`, `ent_coef`, `n_steps`, `batch_size`, etc.) are currently just whatever value happens to be in each `training_config.json`, inherited run to run with no single document explaining why. Alongside Step 2, add `configs/ppo_final.yaml` listing every one of these values with a one-line citation to whichever past experiment established it (E4 settled the algorithm choice, E5 settled `log_std_anneal`/`economy_reward_mode`). This turns "is the config right" from something to trust into something checkable line by line.

### Step 3 — Evaluate the trained policy, all objectives (CPU/GPU inference, fast — minutes per seed)
```bash
for metric in cost mass co2; do
  python pipeline/03_evaluate_policy.py \
    --models models/ppo_${metric}_seed{42,43,44,45,46} \
    --ground_truth data/ground_truth/main_grid --economy_metric $metric \
    --modes none scale scale+thin \
    --out results/evaluation/ppo_${metric}_main_grid
done
```
(For `cost`, `models/ppo_seed{42-46}` from Step 2a's winning arm.) Depends on Steps 1 and 2.

### Step 4 — GA/DE matched-budget baseline, all objectives (CPU, moderate — this is the heaviest CPU-only step; budget an afternoon per objective)
```bash
for metric in cost mass co2; do
  python pipeline/04_baseline_ga_de.py \
    --ground_truth data/ground_truth/main_grid --economy_metric $metric \
    --rl_results results/evaluation/ppo_${metric}_main_grid \
    --methods ga,de --gen_budgets 10,25,40,100,400,1600,4800 \
    --out results/baselines/ga_de_${metric}_main_grid
done
```
Depends on Step 1 (baseline search itself) and Step 3 (to merge in the matched-budget comparison table).

### Step 5 — kNN amortized baseline, all objectives (CPU, fast)
```bash
for metric in cost mass co2; do
  python pipeline/05_baseline_knn.py \
    --ground_truth data/ground_truth/main_grid --economy_metric $metric \
    --rl_results results/evaluation/ppo_${metric}_main_grid \
    --out results/baselines/knn_${metric}_main_grid
done
```
Depends on Step 1 and Step 3.

### Step 6 — OOD generalization (CPU for baselines, GPU inference for PPO, fast)
```bash
python pipeline/06_ood_generalization.py \
  --models models/ppo_seed{42,43,44,45,46} \
  --ground_truth_dirs data/ground_truth/ood_span data/ground_truth/ood_load \
  --out results/ood
```
Fixes the seed-parsing defect flagged in the audit by writing 5 separate per-seed files internally rather than one merged file. `ood_joint` is skipped with an explicit note, not silently omitted. Depends on Steps 1 and 2.

### Step 7 — Hand-calculation validation (GPU inference, seconds)
```bash
python pipeline/07_handcalc_validation.py \
  --models models/ppo_seed{42,43,44,45,46} \
  --out results/evaluation/handcalc
```
Depends on Step 2 only.

### Step 8 — Figures (CPU, fast once data exists)
```bash
python pipeline/08_make_figures.py --out figures/
```
Depends on Steps 3-7 all being complete. See §6 for the figure list.

### Step 9 — Final paper tables (CPU, fast)
```bash
python pipeline/09_build_paper_tables.py --out results/tables/
```
Depends on everything above.

---

## 5. Provenance closure (folded into Step 1, stated separately for visibility)

Every ground-truth directory and every trained checkpoint gets a `meta.json` recording: git commit hash of the code that produced it, exact CLI args, and generation/training date. This is the single biggest process gap the audit found (no existing GT directory in the repo has this today) and is cheap to add now, before the final data exists, rather than reconstructing it from `git log` again in the future.

---

## 6. Proposed figures for the manuscript

| # | Figure | Data source | What it shows |
|---|---|---|---|
| 1 | PPO training curves (reward, feasibility rate, episode length) across 5 seeds | TensorBoard logs from Step 2 | Standard RL convergence evidence; shows the 5 seeds converge consistently |
| 2 | Cost-gap vs. evaluation-budget frontier, GA/DE (raw and reparam space) vs. PPO's single operating point | Step 3 + Step 4 | The core "low-budget regime" result — where PPO sits relative to the classical frontier, and the crossover point where classical search overtakes it |
| 3 | Feasibility and cost-ratio distribution, by repair mode (none/scale/scale+thin) | Step 3 | Mirrors the reference paper's own validation figure; shows repair's effect directly |
| 4 | Paired comparison bar chart with bootstrap CIs: PPO vs. GA vs. DE vs. kNN at matched budget | Steps 3-5 | The statistical headline result in one figure |
| 5 | OOD degradation: feasibility and gap, in-distribution vs. span-extrapolation vs. load-extrapolation, PPO vs. kNN vs. GA | Step 6 | Shows the asymmetry (kNN's feasibility weakness vs. PPO's quality weakness) found during the audit's evidence base |
| 6 | Hand-calculation comparison bar chart (hand-calc reference vs. environment's true optimum vs. PPO), the exact scenario from the supervisor's document | Step 7 | Concrete, real-world-grounded validation point, easy for a non-specialist reviewer to sanity-check |
| 7 | Grade-selection distribution (optimal grade histogram, main grid) before/after the cost-coefficient correction | Step 1 (both old and new GT, already available) | Demonstrates the correction had a real, sensible, non-trivial effect — supports the calibration methodology section |
| 8 | (Optional) Training-cost amortization breakeven curve — cumulative deployment cost, PPO (training + per-design) vs. GA/DE (per-design only), crossing at the breakeven design count | Step 4's equivalence table | Directly supports the "when is amortization worth it" framing from your reframed research question |
| 9 | Reward-mode ablation, post-fix: feasibility_gated vs. lagrangian vs. shaped, cost-gap distribution, 5 seeds each | Step 2a | Answers whether the project's reward-design choice still holds once the ground truth is corrected, rather than resting on the pre-fix Exp1 result |
| 10 | Algorithm comparison, post-fix: PPO vs. SAC vs. TD3 vs. DDPG under the same (winning) reward mode | Step 2b + Step 3 | Directly answers "why PPO and not another algorithm," with a fair, current comparison instead of the pre-cost-fix E4 result |
| 11 | Objective transfer check: cost-gap-equivalent for mass and CO₂, same configuration as the cost headline | Step 2c + Steps 3-5 | Supports the "transfers across objectives" claim from your mass/CO₂ question, checked rather than assumed |

Figures 2-6 are the ones I'd consider essential; 1, 7, 9, 10, and 11 are supporting/methodology evidence (9, 10, and 11 specifically justify the algorithm, reward-mode, and objective-scope choices to a reviewer); 8 is a nice-to-have if there's time.

---

## 6b. Additional items worth adding for a strong first-attempt submission

A few things beyond what either of us has raised directly, flagged now while the repository is still being planned rather than after a reviewer asks:

- **Multiple-comparison correction.** The paper reports many pairwise comparisons (PPO vs. GA/DE/kNN, at several budgets, across three objectives). Running that many bootstrap CIs without correction is exactly the kind of thing a statistically literate reviewer flags in round one. `pipeline/09_build_paper_tables.py` should apply a Holm-Bonferroni (or similar) correction where multiple comparisons share a claim, and the methodology doc should say explicitly which comparisons were treated as a family.
- **Dependency pinning / Gym→Gymnasium migration.** The `expA7` run you shared showed a live deprecation warning ("Gym has been unmaintained since 2022 and does not support NumPy 2.0"). For a repository meant to stay reproducible for as long as a published paper exists, this should be resolved before release — either migrate to Gymnasium (the maintained fork) or pin exact working versions of `gym`/`numpy` in `requirements.txt` and state that pin explicitly in the README, so a reader in two years' time doesn't hit a silent breakage.
- **Data/code archival with a DOI.** A GitHub link alone is generally not treated as a persistent, citable reference by most journals' data-availability requirements. Before submission, archive the final clean branch (code + final ground truth + final results, not the multi-hundred-MB legacy checkpoints) via Zenodo or a similar service to get a DOI, and cite that DOI in the paper's data-availability statement, with the GitHub link as the "actively maintained" pointer.
- **A short "Reproducibility" section in `docs/methodology.md`**, listing exact software versions, hardware used for training, and total compute (wall-clock and/or GPU-hours) across all 40 Step-2 runs — increasingly expected by top venues and cheap to record if done as training happens rather than reconstructed afterward.
- **Optional CI workflow** (`.github/workflows/tests.yml` running `pytest tests/` on push) — signals rigor to anyone browsing the repo, costs almost nothing to add.

---

## 7. README.md outline (for a global, non-specialist-friendly audience)

1. **One-paragraph summary** — what the repo does, in plain language, no jargon before it's defined.
2. **The research question** — the reframed one from your last direction message, not "RL beats GA."
3. **Quick start** — environment setup, one command to reproduce the headline figure/table.
4. **Repository structure** — a copy of the tree in §2, one line each.
5. **Full pipeline reproduction** — the 9 numbered steps from §4, with expected runtime and hardware requirements stated plainly (so someone without a GPU knows what they can and can't reproduce).
6. **Results summary** — the headline numbers and 2-3 key figures, inline.
7. **Data and ground truth provenance** — how ground truth was generated, what EC3 checks are implemented, where the manufacturability and cost-calibration corrections are documented.
8. **Limitations** — the honest ones from the validation draft (OOD asymmetry, joint-extrapolation infeasibility, geometry-idealization gap, kNN's role).
9. **Citation** (once published) and **license**.
10. **Full experimental history** — a link to `docs/archive/`, one sentence acknowledging the repo's iterative history for anyone who wants to dig further.

---

## 8. What I need from you before starting

1. ~~Confirm branch vs. new repository~~ — settled: new clean branch, built up one piece at a time (§9 gives the build-up order).
2. Confirm you're OK with the file merges in §3 (e.g. `train.py` + 4 other training entrypoints → one `02_train_ppo.py`) — these are the changes most likely to need your input on exact CLI flag names if you have existing habits/scripts depending on them.
3. ~~Seed-count decision~~ — settled: 5 seeds everywhere, flat, 40 training runs total across Steps 2a/2b/2c.
4. Green light to start the actual training (Step 2a/2b/2c) — this is real GPU time (40 runs at 1M steps) and should not start silently.
5. Anything in the archive list (§3) you specifically want kept in the clean repo rather than archived.
6. OK with the additions in §6b (multiple-comparison correction, Gym→Gymnasium/dependency pinning, Zenodo DOI archival, reproducibility section, optional CI)? These weren't asked for directly but I think they materially reduce the risk of a first-round reviewer objection — flag any you'd rather skip.

## 9. Build-up order for the new branch

Since you want the branch built incrementally rather than all at once: create the branch and add the skeleton (`README.md` placeholder, directory structure, `src/` unchanged code) first, verified against Step 0's test suite — then add each pipeline script only immediately before the step that needs it, in the exact order of §4, so the branch is always in a runnable state and never has a script committed whose inputs don't exist yet. I'll follow this order when we start moving files.
