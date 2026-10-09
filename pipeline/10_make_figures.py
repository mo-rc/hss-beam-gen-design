"""
pipeline/10_make_figures.py -- paper figures from the SAVED results only (no training, no search).

Reads results/*.csv (per-run evaluation outputs, search / kNN summaries) and writes
figures/fig*.png and fig*.pdf plus figures/fig*_data.csv (the exact numbers plotted).
Every plotted number is taken from a results file; nothing is recomputed or tuned here.
Headline metric everywhere: scale+thin cost-ratio minus 1, in % (0 = matches the best-known
reference, lower is better), as in the docs/results_*.md logs.

    python pipeline/10_make_figures.py            # all figures
    python pipeline/10_make_figures.py --only 1 4 # selected figures
    python pipeline/10_make_figures.py --results_dir results --out_dir figures

Figures
  fig1  gap vs total EC3 evaluations per design (GA / DE / random search vs PPO and kNN), per objective
  fig2  in-distribution vs OOD: gap and feasibility for PPO and kNN, per objective
  fig3  ablations: PPO reward modes (2a) and algorithms (2b), per-seed gaps
  fig4  kNN gap vs labelling cost (realistic search labels vs pooled labels), PPO training cost marked
  fig5  operator ablation: every method under none / scale / scale+thin (gap and feasibility)
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OBJ = ("cost", "mass", "co2")
OBJ_LABEL = {"cost": "Cost", "mass": "Mass", "co2": "CO$_2$"}
SEEDS = range(42, 47)
MODE = "scale+thin"
PPO_STEM = {"cost": "2a_feasibility_gated", "mass": "2c_mass", "co2": "2c_co2"}
LABEL_COST_POOLED_PER_CONTEXT = 96_000  # 12 (grade,type) searches x 2 restarts x 50 x 80
RL_TRAIN_STEPS = 1_000_000
COL = {"ga": "#1b9e77", "de": "#d95f02", "random": "#7570b3", "ppo": "#e7298a", "knn": "#1f78b4",
       "pooled": "#444444"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120,
                     "savefig.bbox": "tight"})


def gap(x):
    return (x - 1.0) * 100.0


def load_ppo(rd, stem, mode=MODE):
    """5-seed table for one RL arm: one row per seed."""
    rows = []
    for s in SEEDS:
        d = pd.read_csv(os.path.join(rd, f"{stem}_seed{s}_eval.csv"))
        r = d[d.operator_mode == mode].iloc[0]
        rows.append(dict(seed=s, gap=gap(r.cost_ratio_mean), feas=r.feasibility * 100,
                         evals=r.ec3_evals_per_design))
    return pd.DataFrame(rows)


def knn_row(path, k=3, mode=MODE, n=None):
    d = pd.read_csv(path)
    d = d[(d.k == k) & (d.operator_mode == mode)]
    if n is not None:
        d = d[d.n_train == n]
    r = d.iloc[0]
    return dict(gap=gap(r.cost_ratio_mean_mean), feas=r.feasibility_mean * 100,
                evals=r.ec3_evals_per_design_mean)


def panel_letters(axes):
    """(a), (b), ... in the top-left corner of each panel, in reading order."""
    for i, ax in enumerate(np.ravel(axes)):
        ax.text(0.0, 1.02, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="bottom", ha="left")


def save(fig, out_dir, name, data):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    pd.DataFrame(data).to_csv(os.path.join(out_dir, f"{name}_data.csv"), index=False)
    plt.close(fig)
    print("wrote", os.path.join(out_dir, name + ".png"))


# --------------------------------------------------------------------------------------- fig 1
def fig1(rd, od):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    data = []
    for ax, o in zip(axes, OBJ):
        s = pd.read_csv(os.path.join(rd, f"search_main_{o}.csv"))
        s = s[s.operator_mode == MODE]
        for m, lab in (("random", "Random"), ("ga", "GA"), ("de", "DE")):
            t = s[s.method == m].sort_values("budget")
            x, y, sd = t.ec3_evals_per_design_mean.to_numpy(), gap(t.cost_ratio_mean_mean.to_numpy()), t.cost_ratio_mean_std.to_numpy() * 100
            ax.plot(x, y, "-o", ms=3, lw=1.2, color=COL[m], label=lab)
            ax.fill_between(x, y - sd, y + sd, color=COL[m], alpha=0.15, lw=0)
            data += [dict(obj=o, method=m, budget=int(b), evals=xi, gap=yi, sd=si) for b, xi, yi, si in zip(t.budget, x, y, sd)]
        p = load_ppo(rd, PPO_STEM[o])
        ax.errorbar(p.evals.mean(), p.gap.mean(), yerr=p.gap.std(ddof=1), fmt="*", ms=11, color=COL["ppo"],
                    mec="k", mew=0.5, capsize=2, zorder=5, label="PPO (5 seeds)")
        data.append(dict(obj=o, method="ppo", evals=p.evals.mean(), gap=p.gap.mean(), sd=p.gap.std(ddof=1)))
        k = knn_row(os.path.join(rd, f"knn_loo_{o}.csv"))
        ax.plot(k["evals"], k["gap"], "D", ms=6, color=COL["knn"], mec="k", mew=0.5, zorder=5,
                label="kNN (leave-one-out)")
        data.append(dict(obj=o, method="knn_loo", evals=k["evals"], gap=k["gap"], sd=np.nan))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=2)
        ax.set_yticks([0, 1, 2, 5, 10, 20, 50, 100])
        ax.set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100"])
        ax.set_title(OBJ_LABEL[o])
        ax.set_xlabel("EC3 evaluations per design")
    axes[0].set_ylabel("Gap to best-known optimum (%)")
    panel_letters(axes)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.text(0.5, -0.17, "The kNN needs solved labelled contexts beforehand (cost in Fig. 4); PPO and the searches need none.",
             ha="center", fontsize=8, style="italic")
    fig.suptitle("Main grid, scale+thin: per-context search (B = 20 ... 4800, evaluations include the operators) "
                 "vs amortized methods", fontsize=9.5, y=1.02)
    save(fig, od, "fig1_gap_vs_evaluations", data)


# --------------------------------------------------------------------------------------- fig 2
def fig2(rd, od):
    sets = (("in-dist.", None), ("OOD span", "ood_span"), ("OOD load", "ood_load"))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    data = []
    w = 0.36
    for j, o in enumerate(OBJ):
        for i, (lab, key) in enumerate(sets):
            stem = PPO_STEM[o] if key is None else f"{key}_{o}"
            p = load_ppo(rd, stem)
            kn = knn_row(os.path.join(rd, f"knn_loo_{o}.csv" if key is None else f"knn_{key}_{o}.csv"))
            ax = axes[0, j]
            ax.bar(i - w / 2, p.gap.mean(), w, yerr=p.gap.std(ddof=1), capsize=2, color=COL["ppo"],
                   label="PPO (5 seeds, mean ± sd)" if (i == 0 and j == 0) else None)
            ax.bar(i + w / 2, kn["gap"], w, color=COL["knn"],
                   label="kNN (pooled labels, k = 3)" if (i == 0 and j == 0) else None)
            ax = axes[1, j]
            ax.errorbar(i - w / 2, p.feas.mean(), yerr=p.feas.std(ddof=1), fmt="o", ms=6, capsize=2, color=COL["ppo"])
            ax.plot(i + w / 2, kn["feas"], "D", ms=6, color=COL["knn"])
            axes[0, 0].set_ylabel("Gap (%)")
            axes[1, 0].set_ylabel("Feasible (%)")
            data += [dict(obj=o, set=lab, method="ppo", gap=p.gap.mean(), gap_sd=p.gap.std(ddof=1), feas=p.feas.mean()),
                     dict(obj=o, set=lab, method="knn", gap=kn["gap"], gap_sd=np.nan, feas=kn["feas"])]
        axes[0, j].set_title(OBJ_LABEL[o])
        axes[1, j].set_ylim(60, 103)
        axes[1, j].set_xticks(range(3))
        axes[1, j].set_xticklabels([s[0] for s in sets])
    panel_letters(axes)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("In-distribution vs out-of-distribution (scale+thin; OOD = contexts with a feasible reference only)",
                 fontsize=9.5)
    save(fig, od, "fig2_in_dist_vs_ood", data)


# --------------------------------------------------------------------------------------- fig 3
def fig3(rd, od):
    panels = (("2a: PPO reward mode\nno pairwise difference survives Holm (n = 5)",
               [("gated", "2a_feasibility_gated"), ("lagrangian", "2a_lagrangian"), ("shaped", "2a_shaped")]),
              ("2b: algorithm (feasibility_gated)\nonly PPO vs TD3 significant after Holm (p = 0.048)", [("PPO", "2a_feasibility_gated"), ("SAC", "2b_sac"),
                                                     ("TD3", "2b_td3"), ("DDPG", "2b_ddpg")]))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True, gridspec_kw={"width_ratios": [3, 4]})
    data = []
    rng = np.random.default_rng(0)
    for ax, (title, arms) in zip(axes, panels):
        for i, (lab, stem) in enumerate(arms):
            p = load_ppo(rd, stem)
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(p)), p.gap, s=26, color=COL["ppo"] if i == 0 else "#555555",
                       edgecolor="k", linewidth=0.4, zorder=3)
            ax.hlines(p.gap.mean(), i - 0.25, i + 0.25, color="k", lw=1.6, zorder=4)
            data += [dict(panel=title.split("\n")[0], arm=lab, seed=int(s), gap=g) for s, g in zip(p.seed, p.gap)]
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels([a[0] for a in arms])
        ax.set_title(title, fontsize=9)
        ax.set_xlim(-0.6, len(arms) - 0.4)
    axes[0].set_ylabel("Gap to best-known optimum (%), cost")
    panel_letters(axes)
    fig.suptitle("Per-seed gaps (dots) and mean (horizontal line), scale+thin, main grid", fontsize=9.5, y=1.06)
    save(fig, od, "fig3_ablations", data)


# --------------------------------------------------------------------------------------- fig 4
def fig4(rd, od):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    data = []
    sizes = (36, 72, 108)
    budgets = (400, 1000, 4800)
    mk = {400: "o", 1000: "s", 4800: "^"}
    for ax, o in zip(axes, OBJ):
        for B in budgets:
            xs, ys, sds = [], [], []
            for n in sizes:
                vals = [knn_row(os.path.join(rd, f"knn_cheap_de{B}_s{S}_subsample_{o}.csv"), n=n)["gap"] for S in (0, 1, 2)]
                xs.append(n * B); ys.append(np.mean(vals)); sds.append(np.std(vals, ddof=1))
            ax.errorbar(xs, ys, yerr=sds, fmt="-" + mk[B], ms=5, lw=1.0, capsize=2, color=COL["knn"],
                        mfc=COL["knn"], alpha={400: 0.45, 1000: 0.7, 4800: 1.0}[B], label=f"DE labels, B = {B}")
            data += [dict(obj=o, labels=f"de{B}", n=n, cost=x, gap=y, sd=s) for n, x, y, s in zip(sizes, xs, ys, sds)]
        xs = [n * LABEL_COST_POOLED_PER_CONTEXT for n in sizes]
        ys = [knn_row(os.path.join(rd, f"knn_subsample_{o}.csv"), n=n)["gap"] for n in sizes]
        ax.plot(xs, ys, "--s", ms=5, color=COL["pooled"], mfc="white", lw=1.0, label="pooled labels (best case)")
        data += [dict(obj=o, labels="pooled", n=n, cost=x, gap=y, sd=np.nan) for n, x, y in zip(sizes, xs, ys)]
        p = load_ppo(rd, PPO_STEM[o])
        ax.axhline(p.gap.mean(), color=COL["ppo"], lw=1.2)
        ax.axvline(RL_TRAIN_STEPS, color=COL["ppo"], lw=1.2, ls=":")
        ax.plot(RL_TRAIN_STEPS, p.gap.mean(), "*", ms=12, color=COL["ppo"], mec="k", mew=0.5, zorder=5,
                label="PPO (1M training steps)")
        data.append(dict(obj=o, labels="ppo", n=np.nan, cost=RL_TRAIN_STEPS, gap=p.gap.mean(), sd=p.gap.std(ddof=1)))
        ax.set_xscale("log")
        ax.set_title(OBJ_LABEL[o])
        ax.set_xlabel("labelling / training cost (EC3 evaluations)")
    axes[0].set_ylabel("Gap to best-known optimum (%)")
    panel_letters(axes)
    for ax in axes:
        ax.set_ylim(-0.3, ax.get_ylim()[1] * 1.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("kNN in distribution (subsample, n = 36 / 72 / 108 labelled contexts): gap vs labelling cost; "
                 "mean ± sd over 3 label seeds", fontsize=9.5, y=1.02)
    save(fig, od, "fig4_knn_label_cost", data)


# --------------------------------------------------------------------------------------- fig 5
def fig5(rd, od):
    """Every method under each post-hoc operator: how much of the quality is the operator's."""
    ops = ("none", "scale", "scale+thin")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), sharex=True)
    data = []
    series = [("PPO", COL["ppo"], "*"), ("kNN (pooled labels)", COL["knn"], "D"), ("DE, B = 40", COL["de"], "o"),
              ("GA, B = 40", COL["ga"], "s"), ("Random, B = 40", COL["random"], "^")]
    for j, o in enumerate(OBJ):
        rows = {name: [] for name, _, _ in series}
        for op in ops:
            rows["PPO"].append(load_ppo(rd, PPO_STEM[o], mode=op))
            rows["kNN (pooled labels)"].append(knn_row(os.path.join(rd, f"knn_loo_{o}.csv"), mode=op))
        s_ = pd.read_csv(os.path.join(rd, f"search_main_{o}.csv"))
        for m, name in (("de", "DE, B = 40"), ("ga", "GA, B = 40"), ("random", "Random, B = 40")):
            for op in ops:
                r = s_[(s_.method == m) & (s_.budget == 40) & (s_.operator_mode == op)].iloc[0]
                rows[name].append(dict(gap=gap(r.cost_ratio_mean_mean), feas=r.feasibility_mean * 100))
        for name, col, mk in series:
            gaps, feas = [], []
            for v in rows[name]:
                if isinstance(v, pd.DataFrame):
                    gaps.append(v.gap.mean()); feas.append(v.feas.mean())
                else:
                    gaps.append(v["gap"]); feas.append(v["feas"])
            lw = 2.0 if name == "PPO" else 1.1
            axes[0, j].plot(range(3), gaps, "-" + mk, color=col, lw=lw, ms=9 if mk == "*" else 5, mec="k", mew=0.4,
                            label=name, zorder=5 if name == "PPO" else 3)
            axes[1, j].plot(range(3), feas, "-" + mk, color=col, lw=lw, ms=9 if mk == "*" else 5, mec="k", mew=0.4,
                            zorder=5 if name == "PPO" else 3)
            data += [dict(obj=o, method=name, operator=op, gap=g, feas=f) for op, g, f in zip(ops, gaps, feas)]
        axes[0, j].set_yscale("symlog", linthresh=2)
        axes[0, j].set_yticks([0, 1, 2, 5, 10, 20, 50, 100, 150])
        axes[0, j].set_yticklabels(["0", "1", "2", "5", "10", "20", "50", "100", "150"])
        axes[0, j].set_title(OBJ_LABEL[o])
        axes[1, j].set_ylim(0, 105)
        axes[1, j].set_xticks(range(3))
        axes[1, j].set_xticklabels(["none", "scale", "scale+thin"])
        axes[1, j].set_xlabel("post-hoc operator")
    axes[0, 0].set_ylabel("Gap, feasible contexts (%)")
    axes[1, 0].set_ylabel("Feasible (%)")
    fig.subplots_adjust(hspace=0.4)
    panel_letters(axes)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Effect of the post-hoc operator on every method (main grid; searches at B = 40)", fontsize=9.5)
    save(fig, od, "fig5_operator_ablation", data)


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--only", type=int, nargs="+", default=sorted(FIGS))
    a = ap.parse_args()
    for i in a.only:
        FIGS[i](a.results_dir, a.out_dir)


if __name__ == "__main__":
    main()
