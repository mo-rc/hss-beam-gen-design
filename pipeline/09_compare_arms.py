"""Step 9 (partial) - statistical comparison of the 2a reward-mode arms.

Compares feasibility_gated / lagrangian / shaped (5 seeds each) on scale+thin cost_ratio_mean,
reading results/2a_{reward_mode}_seed{seed}_eval.csv as written by pipeline/03_evaluate_agent.py.

Method: seeds are PAIRED by seed number across reward modes (seed 42's feasibility_gated,
lagrangian and shaped runs all start from the same PPO weight initialisation and the same
env-context sampling sequence -- see hssbeamgen/train_utils.py:build_model/build_vec_env,
both seeded from the single --seed value), so a paired test has more power than an unpaired
one at n=5. For each pair of arms:
  - exact paired sign-flip permutation test on the mean paired difference (2^5 = 32
    permutations enumerated exactly; no normality assumption, appropriate at n=5)
  - percentile bootstrap 95% CI on the mean paired difference (resampling the 5 pairs)
This mirrors the bootstrap-CI style already used for the GA/DE comparison in
results_validation_draft.md Sec.3, applied here to the 2a arms.
Holm-Bonferroni correction is applied across the 3 pairwise comparisons (plan's settled
decision: "Holm-type multiple-comparison correction").

This script needs only numpy/pandas/scipy -- no torch/stable-baselines3 -- so it's meant to
be run locally, not on the training machine.

Usage:
    python pipeline/09_compare_arms.py --results_dir results --arms feasibility_gated lagrangian shaped \
        --out results/2a_comparison.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

SEEDS = (42, 43, 44, 45, 46)
OPERATOR_MODE = "scale+thin"
N_BOOTSTRAP = 20000


def load_seed_values(results_dir: str, arm: str, metric: str = "cost_ratio_mean") -> np.ndarray:
    vals = []
    for seed in SEEDS:
        path = os.path.join(results_dir, f"2a_{arm}_seed{seed}_eval.csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- need all {len(SEEDS)} seeds for arm {arm!r}")
        df = pd.read_csv(path)
        row = df[df["operator_mode"] == OPERATOR_MODE]
        if len(row) != 1:
            raise SystemExit(f"{path}: expected exactly one {OPERATOR_MODE!r} row, found {len(row)}")
        vals.append(float(row[metric].iloc[0]))
    return np.array(vals)


def exact_sign_flip_test(diff: np.ndarray) -> float:
    """Two-sided exact permutation p-value for mean(diff) != 0, enumerating all 2^n sign patterns."""
    n = len(diff)
    observed = abs(diff.mean())
    n_extreme = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs((np.array(signs) * diff).mean()) >= observed - 1e-12:
            n_extreme += 1
    return n_extreme / (2 ** n)


def bootstrap_ci(diff: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_means = rng.choice(diff, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the ORIGINAL order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * pvals[idx], 1.0)
        running_max = max(running_max, val)  # step-down monotonicity
        adjusted[idx] = running_max
    return adjusted.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--arms", nargs="+", default=["feasibility_gated", "lagrangian", "shaped"])
    p.add_argument("--metric", default="cost_ratio_mean")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    values = {arm: load_seed_values(args.results_dir, arm, args.metric) for arm in args.arms}

    print(f"Per-seed {args.metric} ({OPERATOR_MODE}), by arm:")
    summary_rows = []
    for arm, v in values.items():
        print(f"  {arm:20s} " + " ".join(f"{x:.4f}" for x in v) + f"   mean={v.mean():.4f}")
        summary_rows.append(dict(arm=arm, **{f"seed{s}": x for s, x in zip(SEEDS, v)}, mean=v.mean()))
    print()

    pairs = list(itertools.combinations(args.arms, 2))
    raw_pvals, rows = [], []
    for a, b in pairs:
        diff = values[a] - values[b]  # positive => a worse (higher cost ratio) than b
        pval = exact_sign_flip_test(diff)
        ci_lo, ci_hi = bootstrap_ci(diff)
        raw_pvals.append(pval)
        rows.append(dict(arm_a=a, arm_b=b, mean_diff=diff.mean(),
                         mean_diff_pct_points=diff.mean() * 100,
                         ci95_lo_pct_points=ci_lo * 100, ci95_hi_pct_points=ci_hi * 100,
                         p_raw=pval))
    adjusted = holm_correct(raw_pvals)
    for row, p_adj in zip(rows, adjusted):
        row["p_holm"] = p_adj
        row["significant_at_0.05"] = p_adj < 0.05

    comparison = pd.DataFrame(rows)
    print("Pairwise comparisons (Holm-corrected across {} tests):".format(len(pairs)))
    print(comparison.to_string(index=False))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    comparison.to_csv(args.out, index=False)
    summary_path = args.out.rsplit(".", 1)[0] + "_per_seed.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n-> {args.out}\n-> {summary_path}")


if __name__ == "__main__":
    main()
