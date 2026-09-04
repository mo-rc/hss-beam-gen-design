# E3 result files — verification manifest

Everything below lives in this project's `results/` directory and is intended
to be copied into `research/perplexity_docs/results/` in the GitHub repo.
Verify a copy by checking row count, column count and MD5 against this table.

Row counts exclude the header line. MD5 is over the full file bytes including
the header, computed at the state committed alongside this manifest.

| File | Data rows | Cols | Bytes | MD5 (first 12) |
|---|---|---|---|---|
| `e3_transfer_seed43_summary.csv` | 4 | 22 | 1,710 | `cc5c3149bb57` |
| `e3_transfer_seed43_per_context.csv` | 568 | 29 | 198,780 | `d4693e7fb28d` |
| `e3_curve_retrained_seed43_summary.csv` | 14 | 24 | 5,472 | `e10af516db72` |
| `e3_curve_precorrection_seed43_summary.csv` | 14 | 24 | 5,372 | `b9dd6883d8f2` |
| `e3_reparam_probe_costing_ab_n40.csv` | 12 | 11 | 1,742 | `c56a0721ff28` |
| `e3_reparam_probe_costing_ab_n20.csv` | 6 | 11 | 955 | `a44299819102` |
| `e3_knn_oneshot_corrected_summary.csv` | 9 | 16 | 2,068 | `40fc256fec37` |
| `e3_knn_oneshot_corrected_per_context.csv` | 1,278 | 19 | 268,644 | `b2aacca65dd0` |
| `e3_knn_oneshot_originalcosting_summary.csv` | 3 | 16 | 822 | `8e8f4e5a83b9` |
| `e3_knn_oneshot_originalcosting_per_context.csv` | 426 | 19 | 87,707 | `10329ecbd633` |

Already present in the repo at `research/results/` and unchanged by this
update: `e3_corrcost_retrain_summary.csv` (8 rows × 22),
`e3_corrcost_retrain_per_context.csv` (1,136 rows × 29).

---

## 1. `e3_transfer_seed43_summary.csv` — paired same-seed transfer control

The pre-correction `gated_cost_merged_seed43` **final_model**, unchanged, scored
against the corrected ground truth. This is the correct baseline for the
retrained arm: same seed, same architecture, same steps, only the cost model
and the reference differ.

4 rows = 1 arm × 4 repair modes (`none`, `scale`, `scale+thin`,
`scale+thin_rolled`). `n = 142`, `feasibility = 1.0` in every row.

Columns (22, in order):

```
arm, repair_mode, n, feasibility,
gap_mean, gap_median, gap_sd, gap_p90, gap_p95, gap_worst,
within_1pct, within_5pct, within_10pct,
util_mean, class_mean, grade_match, repaired_frac,
ec3_gen_per_design, ec3_repair_per_design, ec3_evals_per_design,
sec_per_design, sec_per_design_sd
```

Gaps are stored as **fractions**, not percentages (`0.0753` = 7.53%).
`arm` is `gated_cost_merged_seed43` in all 4 rows.

Key values: `gap_mean` = 0.232022 (`none`), 0.113915 (`scale`),
0.075269 (`scale+thin`), 0.094179 (`scale+thin_rolled`).

## 2. `e3_transfer_seed43_per_context.csv` — per-context detail for the above

568 rows = 142 contexts × 4 repair modes. This is the file the paired bootstrap
CIs in the results doc are computed from; pairing is on
`(span_m, load_kNm, repair_mode)` against `e3_corrcost_retrain_per_context.csv`.

Columns (29, in order):

```
arm, repair_mode, span_m, load_kNm,
optimal, optimal_grade, optimal_type,
achieved, gap, feasible, feasible_before,
utilization, section_class, h, b, tf, tw, fy, section_type,
grade_match, repaired, variant, scale,
n_ec3_gen, n_ec3_repair, n_ec3, t_gen, t_repair, sec_per_design
```

`optimal` is the corrected-GT reference cost; `achieved` is the policy's cost
under corrected costing; `gap = achieved/optimal - 1`.

## 3. `e3_curve_retrained_seed43_summary.csv` — retrained learning curve

Every checkpoint of `corrcost_gated_merged_seed43` evaluated against the
corrected ground truth. 14 rows = 7 checkpoints × 2 repair modes
(`none`, `scale+thin`), sorted by `steps` then `repair_mode`.

**Two columns were added to this file in this update** so it is
self-describing; they are inserted at positions 3 and 4, giving 24 columns
rather than the 22 of the summary files above:

```
arm, repair_mode, checkpoint, steps, n, feasibility, ... (then as §1)
```

- `checkpoint` — the model filename, e.g. `checkpoint_800000_steps` or `final_model`
- `steps` — 200000, 400000, 600000, 800000, 1000000, 1200000, 1302528

`final_model` is at 1,302,528 steps, which is the true terminal step count
from the TensorBoard log, not a round 1.3M.

`scale+thin` `gap_mean` by step: 0.268179, 0.091644, 0.109879, 0.088934,
0.079100, 0.067488, 0.066314.

## 4. `e3_curve_precorrection_seed43_summary.csv` — pre-correction learning curve

Every checkpoint of the pre-correction `gated_cost_merged_seed43`, scored
against the same corrected ground truth. Same 24-column layout as §3.

**This file changed in two ways in this update.** It previously had 12 rows
covering 6 checkpoints; the 1,200,000-step checkpoint exists on disk but had
not been evaluated, leaving the pre-correction arm without a midpoint over its
final 300k steps while the retrained arm had one. It has now been evaluated and
added, so the file is **14 rows** and the two curves are on identical step
grids. The `checkpoint` / `steps` columns were added as in §3.

`scale+thin` `gap_mean` by step: 0.486399, 0.204854, 0.104416, 0.116728,
0.078404, 0.083586, 0.075269.

One cross-file consistency note. The `final_model` row here and the
corresponding row of `e3_transfer_seed43_summary.csv` (§1) are the same model
on the same ground truth, and they agree exactly for `scale+thin`
(0.075269 both) but differ in the 5th decimal for `none`
(0.232009 vs 0.232022; `util_mean` 0.837802 vs 0.837751). The two files came
from separate invocations with different `--modes` lists, so the environment's
RNG stream differs and the unrepaired rollouts are not bit-identical. The
discrepancy is 0.0013 pp, far below anything the results doc claims, and the
repaired arm is unaffected because the repair projects both rollouts onto the
same `util = 1.0` boundary. Quote 23.20% from either file; do not present the
two as independent measurements.

Note the new 1.2M point (0.083586) is **above** both its neighbours. The
pre-correction run oscillates across its final 300k steps
(7.84% → 8.36% → 7.53%) whereas the retrained run falls monotonically
(7.91% → 6.75% → 6.63%). The results-doc claim that the retrained run was
still improving at termination while the pre-correction run had plateaued is
strengthened, not weakened, by this addition.

## 5. `e3_reparam_probe_costing_ab_n40.csv` / `_n20.csv` — reparameterisation probe A/B

Matched re-run of the zero-training RAW-vs-REPARAM probe under both costing
models. Produced by `code/diag_reparam_probe_costing_ab.py`.

`_n40`: 12 rows = 2 costings × (3 RAW budgets + 3 REPARAM budgets), 40 contexts.
`_n20`: 6 rows = 2 costings × (1 RAW + 2 REPARAM budgets), 20 contexts.

Columns (11, in order):

```
costing, gt_dir, n_contexts, space, proposals, ec3_per_context,
gap_mean, gap_median, gap_p90, gap_worst, feasibility
```

`costing` is `original` or `corrected`. The `original` rows reproduce the
probe tables in `diagnosis/2026-09-02_bottleneck-diagnosis-and-reformulation.md`
§4 exactly, to all reported digits, which is what licenses reading the
`corrected` rows as a costing effect.

## 6. `e3_knn_oneshot_*_summary.csv` / `_per_context.csv` — LOO k-NN predictor

Leave-one-out k-nearest-neighbour one-shot predictor from the bottleneck report
§5, re-run against corrected costing and corrected ground truth. Produced by
`code/diag_knn_oneshot.py`.

`_corrected_summary`: 9 rows = k ∈ {1,3,5} × label fraction ∈ {1.0, 0.5, 0.25}.
`_originalcosting_summary`: 3 rows = k ∈ {1,3,5} at full labels, original
costing and original GT — the reproduction check. Its k=3 row is
1.87% mean / 0.86% median / 5.14% p90 / 20.2% worst / 0.87 within-5%, matching
the bottleneck report's table exactly.

Summary columns (16, in order):

```
predictor, k, label_frac, n_labels, n, feasibility,
gap_mean, gap_median, gap_sd, gap_p90, gap_p95, gap_worst,
within_1pct, within_5pct, within_10pct, ec3_evals_per_design
```

Per-context columns (19, in order):

```
k, label_frac, n_labels, span_m, load_kNm, feasible,
reference, achieved, gap, util, section_class,
h, b, tf, tw, fy, section_type, grade_match, n_ec3
```

Per-context row counts are 142 contexts × 9 configurations = 1,278
(corrected) and 142 × 3 = 426 (original costing). Infeasible contexts carry
`feasible=0` and `gap=NaN`; summary `gap_*` statistics are computed with
`nan`-aware reducers over the feasible subset, and `feasibility` reports the
fraction that were feasible.

---

## Reproduction

From the repo root, with `research/pretrain_data_corrected/` present:

```bash
# §5 — probe A/B
python research/scripts/diag_reparam_probe_costing_ab.py \
  --metric cost --n_contexts 40 --budgets 10 20 40 \
  --raw_budgets 40 400 4000 --seed 0 \
  --out research/perplexity_docs/results/e3_reparam_probe_costing_ab_n40.csv

python research/scripts/diag_reparam_probe_costing_ab.py \
  --metric cost --n_contexts 20 --budgets 200 1000 \
  --raw_budgets 4800 --seed 0 \
  --out research/perplexity_docs/results/e3_reparam_probe_costing_ab_n20.csv

# §6 — LOO k-NN, corrected
python research/scripts/diag_knn_oneshot.py \
  --metric cost --gt_dir research/pretrain_data_corrected \
  --k 1 3 5 --label_fracs 1.0 0.5 0.25 --seed 0 \
  --out_prefix research/perplexity_docs/results/e3_knn_oneshot_corrected

# §4 — the added pre-correction 1.2M point
python research/scripts/evaluate_with_repair.py \
  --models research/models/gated_cost_merged_seed43/checkpoint_1200000_steps.zip \
  --economy_metric cost --reward_mode_for_env feasibility_gated \
  --ground_truth_dir research/pretrain_data_corrected \
  --modes none scale+thin --n_contexts 142 --seed 0 \
  --out_prefix /tmp/precorr_1p2M
```

The `diag_knn_oneshot.py` reproduction check under original costing requires
constructing the environment with `enforce_rolled_manufacturability=False`;
`diag_reparam_probe_costing_ab.py` shows the one-line monkey-patch used.
