"""Step 9 (partial) - statistical comparison of the 2a reward-mode arms.

Compares feasibility_gated / lagrangian / shaped (5 independent seeds each) on the chosen
operator mode's cost_ratio_mean, reading results/2a_{reward_mode}_seed{seed}_eval.csv as
written by pipeline/03_evaluate_agent.py.

Method. The three arms are treated as INDEPENDENT samples of 5 seeds each. (A paired-by-seed
test was considered and rejected: the shared seed value only fixes the initial weights and the
context-sampling sequence; the reward differs, so trajectories diverge, and the observed
across-arm correlation of per-seed gaps is ~0, i.e. pairing buys no power. A paired exact
sign-flip test at n=5 also has a smallest attainable two-sided p of 2/32 = 0.0625, which Holm
over 3 comparisons turns into 0.19 -- it could never reach significance.)
For each pair of arms:
  - exact two-sample permutation test on the difference of means, enumerating all
    C(10,5) = 252 splits (no distributional assumption). Smallest attainable two-sided p:
    2/252 = 0.0079 (0.024 after Holm over 3 tests), so significance is at least possible.
  - percentile bootstrap 95% CI on the difference of means (each arm resampled separately).
    With n=5 per arm this interval is narrow relative to the true uncertainty; read it as
    descriptive, and treat the permutation p-value as the test.
Holm-Bonferroni correction is applied across the pairwise comparisons.

Needs only numpy/pandas -- no torch/stable-baselines3 -- so it runs locally.

Usage:
    python pipeline/09_compare_arms.py --results_dir results --out results/2a_comparison.csv
    python pipeline/09_compare_arms.py --operator_mode none --out results/2a_comparison_none.csv
    # 2b: PPO (= the 2a feasibility_gated runs) vs SAC / TD3 / DDPG, 4 arms -> 6 pairs.
    # Smallest attainable Holm p over 6 tests is 6 * 2/252 = 0.048, so significance is only just
    # possible: only a complete separation of two arms' 5 seeds can reach it.
    python pipeline/09_compare_arms.py --arms ppo=2a_feasibility_gated sac=2b_sac td3=2b_td3 \
        ddpg=2b_ddpg --out results/2b_comparison.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44, 45, 46)
N_BOOTSTRAP = 20000


def load_seed_values(results_dir: str, stem: str, metric: str = "cost_ratio_mean",
                     operator_mode: str = "scale+thin") -> np.ndarray:
    """stem is the file stem before '_seed{N}_eval.csv', e.g. '2a_shaped' or '2b_sac'."""
    vals = []
    for seed in SEEDS:
        path = os.path.join(results_dir, f"{stem}_seed{seed}_eval.csv")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- need all {len(SEEDS)} seeds for arm stem {stem!r}")
        df = pd.read_csv(path)
        row = df[df["operator_mode"] == operator_mode]
        if len(row) != 1:
            raise SystemExit(f"{path}: expected exactly one {operator_mode!r} row, found {len(row)}")
        vals.append(float(row[metric].iloc[0]))
    return np.array(vals)


def exact_permutation_test(x: np.ndarray, y: np.ndarray) -> float:
    """Two-sided exact p-value for mean(x) != mean(y): all C(nx+ny, nx) label splits."""
    v = np.concatenate([x, y])
    nx = len(x)
    observed = abs(x.mean() - y.mean())
    n_extreme = n_total = 0
    for idx in itertools.combinations(range(len(v)), nx):
        mask = np.zeros(len(v), dtype=bool)
        mask[list(idx)] = True
        n_total += 1
        if abs(v[mask].mean() - v[~mask].mean()) >= observed - 1e-12:
            n_extreme += 1
    return n_extreme / n_total


def bootstrap_ci(x: np.ndarray, y: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bx = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    by = rng.choice(y, size=(n_boot, len(y)), replace=True).mean(axis=1)
    d = bx - by
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


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
    p.add_argument("--prefix", default="2a",
                   help="file-stem prefix for bare arm names: {prefix}_{arm}_seed{N}_eval.csv")
    p.add_argument("--arms", nargs="+", default=["feasibility_gated", "lagrangian", "shaped"],
                   help="arm names. A bare name 'x' reads {prefix}_x_seed*_eval.csv; 'label=stem' "
                        "reads {stem}_seed*_eval.csv under that label (used by 2b to reuse the "
                        "2a PPO runs: ppo=2a_feasibility_gated sac=2b_sac td3=2b_td3 ddpg=2b_ddpg)")
    p.add_argument("--metric", default="cost_ratio_mean")
    p.add_argument("--operator_mode", default="scale+thin",
                   choices=["none", "scale", "scale+thin"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    arm_stems = {}
    for spec in args.arms:
        label, _, stem = spec.partition("=")
        arm_stems[label] = stem or f"{args.prefix}_{label}"
    args.arms = list(arm_stems)
    values = {arm: load_seed_values(args.results_dir, stem, args.metric, args.operator_mode)
              for arm, stem in arm_stems.items()}

    print(f"Per-seed {args.metric} ({args.operator_mode}), by arm:")
    summary_rows = []
    for arm, v in values.items():
        print(f"  {arm:20s} " + " ".join(f"{x:.4f}" for x in v) + f"   mean={v.mean():.4f}")
        summary_rows.append(dict(arm=arm, **{f"seed{s}": x for s, x in zip(SEEDS, v)}, mean=v.mean()))
    print()

    pairs = list(itertools.combinations(args.arms, 2))
    raw_pvals, rows = [], []
    for a, b in pairs:
        diff = values[a].mean() - values[b].mean()  # positive => a worse (higher cost ratio)
        pval = exact_permutation_test(values[a], values[b])
        ci_lo, ci_hi = bootstrap_ci(values[a], values[b])
        raw_pvals.append(pval)
        rows.append(dict(arm_a=a, arm_b=b, mean_a=values[a].mean(), mean_b=values[b].mean(), mean_diff=diff,
                         mean_diff_pct_points=diff * 100,
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
