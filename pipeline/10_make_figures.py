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

Figures (captions: figures/captions.md; figure titles are deliberately left to the captions)
  fig0  study overview: context -> methods -> design -> EC3 check + operators -> gap to reference
  fig1  gap vs total EC3 evaluations per design (GA / DE / random search vs PPO and kNN), per objective
  fig2  in-distribution vs OOD: gap and feasibility for PPO and kNN, per objective
  fig3  ablations: PPO reward modes (2a) and algorithms (2b), per-seed gaps
  fig4  kNN gap vs labelling cost (realistic search labels vs pooled labels), PPO training cost marked
  fig5  operator ablation: every method under none / scale / scale+thin (gap and feasibility)
  fig6  reference landscape: optimal grade over the span x load plane, training box, OOD grids
  fig7  training curves (supplementary): reward, episode length, Lagrangian violations and multipliers per seed;
        needs results/training_curves/ from pipeline/13_export_training_curves.py and is skipped if absent
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
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

plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "axes.linewidth": 0.7, "figure.dpi": 120, "savefig.bbox": "tight",
                     "pdf.fonttype": 42, "ps.fonttype": 42})  # embedded TrueType fonts in the PDFs
W2 = 7.2  # double-column width (inches)
GRADES = (355, 460, 500, 550, 620, 690)


def panel(ax, letter):
    ax.text(-0.02, 1.08, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom",
            ha="right")


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


def save(fig, out_dir, name, data):
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))
    pd.DataFrame(data).to_csv(os.path.join(out_dir, f"{name}_data.csv"), index=False)
    plt.close(fig)
    print("wrote", os.path.join(out_dir, name + ".png"))


# --------------------------------------------------------------------------------------- fig 1
def fig1(rd, od):
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7), sharey=True, constrained_layout=True)
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
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    for ax, ch in zip(axes, "abc"):
        panel(ax, ch)
    save(fig, od, "fig1_gap_vs_evaluations", data)


# --------------------------------------------------------------------------------------- fig 2
def fig2(rd, od):
    sets = (("in-dist.", None), ("OOD span", "ood_span"), ("OOD load", "ood_load"))
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.2), sharex=True, constrained_layout=True)
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
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    for ax, ch in zip(axes.ravel(), "abcdef"):
        panel(ax, ch)
    save(fig, od, "fig2_in_dist_vs_ood", data)


# --------------------------------------------------------------------------------------- fig 3
def fig3(rd, od):
    panels = (("2a: PPO reward mode\nno pairwise difference survives Holm (n = 5)",
               [("gated", "2a_feasibility_gated"), ("lagrangian", "2a_lagrangian"), ("shaped", "2a_shaped")]),
              ("2b: algorithm (feasibility_gated)\nonly PPO vs TD3 significant after Holm (p = 0.048)", [("PPO", "2a_feasibility_gated"), ("SAC", "2b_sac"),
                                                     ("TD3", "2b_td3"), ("DDPG", "2b_ddpg")]))
    fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.85, 2.8), sharey=True, gridspec_kw={"width_ratios": [3, 4]},
                             constrained_layout=True)
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
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(-0.6, len(arms) - 0.4)
    axes[0].set_ylabel("Gap to best-known optimum (%), cost")
    for ax, ch in zip(axes, "ab"):
        panel(ax, ch)
    save(fig, od, "fig3_ablations", data)


# --------------------------------------------------------------------------------------- fig 4
def fig4(rd, od):
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.8), sharey=True, constrained_layout=True)
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
        ax.set_xlabel("labelling / training cost (EC3 evals)")
    axes[0].set_ylabel("Gap to best-known optimum (%)")
    for ax in axes:
        ax.set_ylim(-0.3, ax.get_ylim()[1] * 1.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    for ax, ch in zip(axes, "abc"):
        panel(ax, ch)
    save(fig, od, "fig4_knn_label_cost", data)


# --------------------------------------------------------------------------------------- fig 0
def fig0(rd, od):
    """Study overview (schematic; no data)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(W2, 3.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#333333", fs=7, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else None, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#333333"))

    box(1, 14, 15, 12, "Context\nspan $L$ (m)\nfactored UDL\n$w$ (kN/m)\n$M_{Ed}=wL^2/8$", "#f0f0f0", bold=False)
    methods = [("PPO policy\ntrained 1M steps, no labels\n~40 evals per design", "#fbd3e6", 28),
               ("kNN (k = 3)\nsolved contexts as labels\n1 prediction", "#cfe3f4", 15),
               ("GA / DE / random search\nre-solved per context\nbudget $B$ evals", "#fde0c8", 2)]
    for text, fc, y in methods:
        box(25, y, 24, 9.5, text, fc)
        arrow(16.8, 20, 24.6, y + 4.7)
    box(54.5, 14, 18.5, 12, "Design\ngrade (S355-S690)\nsection type\n$h,\\ b,\\ t_f,\\ t_w$", "#f0f0f0", fs=6.6)
    for _, _, y in methods:
        arrow(49.4, y + 4.7, 54.1, 20)
    box(79, 24, 20, 12, "EC3 check\n+ post-hoc operators\nnone / scale /\nscale+thin", "#e5f5e0")
    arrow(73.4, 20, 78.6, 28)
    box(79, 4, 20, 14, "Gap to best-known\nreference (pooled GA\nsearches), feasibility,\nEC3 evals per design", "#f0f0f0")
    arrow(89, 23.4, 89, 18.6)
    ax.text(1, 44.3, "Evaluation sets: main grid (142 feasible contexts), OOD span (26), OOD load (47), "
                     "OOD joint (0, untestable)", fontsize=6.8, va="center")
    ax.text(1, 41.3, "Objectives: cost, mass, CO$_2$ (each against its own reference)", fontsize=6.8, va="center")
    save(fig, od, "fig0_overview", [dict(note="schematic, no plotted data")])


# --------------------------------------------------------------------------------------- fig 5
def fig5(rd, od):
    """Every method under each post-hoc operator: how much of the quality is the operator's."""
    ops = ("none", "scale", "scale+thin")
    fig, axes = plt.subplots(2, 3, figsize=(W2, 4.3), sharex=True, constrained_layout=True)
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
            axes[0, j].plot(range(3), gaps, "-" + mk, color=col, lw=lw, ms=8 if mk == "*" else 4.5, mec="k", mew=0.4,
                            label=name, zorder=5 if name == "PPO" else 3)
            axes[1, j].plot(range(3), feas, "-" + mk, color=col, lw=lw, ms=8 if mk == "*" else 4.5, mec="k", mew=0.4,
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
    for ax, ch in zip(axes.ravel(), "abcdef"):
        panel(ax, ch)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=5, frameon=False)
    save(fig, od, "fig5_operator_ablation", data)


# --------------------------------------------------------------------------------------- fig 6
def _grid(meta):
    g = meta["grid_definition"]
    return (np.linspace(g["span_min_m"], g["span_max_m"], g["n_spans"]),
            np.linspace(g["load_min"], g["load_max"], g["n_loads"]))


def fig6(rd, od):
    import json
    root = os.path.join(os.path.dirname(os.path.abspath(rd)), "data", "ground_truth")
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.9), sharey=True, constrained_layout=True)
    pal = dict(zip(GRADES, plt.cm.viridis(np.linspace(0.05, 0.95, len(GRADES)))))
    data = []
    for ax, o in zip(axes, OBJ):
        for g in ("main_grid", "ood_span", "ood_load", "ood_joint"):
            meta = json.load(open(os.path.join(root, g, "meta.json")))
            spans, loads = _grid(meta)
            f = os.path.join(root, g + "_pooled", f"ec3_optimal_designs_{o}.csv")
            d = pd.read_csv(f)
            best = d.loc[d.groupby(["span_m", "load_kN_per_m"])[o].idxmin()] if len(d) else d
            feas = {(round(r.span_m, 6), round(r.load_kN_per_m, 6)): r for r in best.itertuples()}
            for sp in spans:
                for ld in loads:
                    r = feas.get((round(sp, 6), round(ld, 6)))
                    if r is None:
                        ax.plot(sp, ld, "x", color="#999999", ms=2.6, mew=0.6, zorder=2)
                        data.append(dict(obj=o, grid=g, span=sp, load=ld, feasible=False, grade=np.nan, welded=np.nan))
                    else:
                        ax.scatter(sp, ld, s=9, marker="s", color=pal[int(round(r.grade))],
                                   edgecolor="k" if r.section_type == "welded" else "none", linewidth=0.7, zorder=3)
                        data.append(dict(obj=o, grid=g, span=sp, load=ld, feasible=True, grade=int(round(r.grade)),
                                         welded=r.section_type == "welded"))
        ax.add_patch(plt.Rectangle((6, 20), 9, 120, fill=False, ls="--", lw=0.9, ec="k", zorder=4))
        ax.text(10.5, 4, "training range", ha="center", fontsize=6.3)
        ax.text(19, 4, "OOD span", ha="center", fontsize=6.3)
        ax.text(10.5, 268, "OOD load", ha="center", fontsize=6.3)
        ax.text(19, 268, "OOD joint:\n0/64 feasible", ha="center", fontsize=6.3)
        ax.set_title(OBJ_LABEL[o])
        ax.set_xlabel("span (m)")
        ax.set_xlim(4.5, 23.5)
        ax.set_ylim(-8, 300)
    axes[0].set_ylabel("factored UDL (kN/m)")
    handles = [plt.Line2D([], [], marker="s", ls="", color=pal[g], ms=5, label=f"S{g}") for g in GRADES]
    handles += [plt.Line2D([], [], marker="s", ls="", mfc="white", mec="k", ms=5, label="welded section"),
                plt.Line2D([], [], marker="x", ls="", color="#999999", ms=4, label="no feasible design")]
    fig.legend(handles=handles, loc="outside lower center", ncol=8, frameon=False)
    for ax, ch in zip(axes, "abc"):
        panel(ax, ch)
    save(fig, od, "fig6_reference_landscape", data)


# --------------------------------------------------------------------------------------- fig 7
def fig7(rd, od):
    cdir = os.path.join(rd, "training_curves")
    if not os.path.exists(os.path.join(cdir, "index.csv")):
        print("fig7 skipped: no results/training_curves/index.csv (run pipeline/13_export_training_curves.py)")
        return
    idx = pd.read_csv(os.path.join(cdir, "index.csv"))
    runs = {r.run: pd.read_csv(os.path.join(cdir, r.run + ".csv")) for r in idx.itertuples()}
    algo_col = {"ppo": COL["ppo"], "sac": "#66a61e", "td3": "#e6ab02", "ddpg": "#a6761d"}
    kept = []

    def curves(exp_prefix, arm, tag):
        out = []
        for r in idx[(idx.experiment.str.startswith(exp_prefix)) & (idx.arm == arm)].itertuples():
            d = runs[r.run]
            d = d[d.tag == tag]
            if len(d):
                out.append((r.run, d.step.to_numpy(), d.value.to_numpy()))
        return out

    def draw(ax, pan, items, color, label):
        for k, (run, x, y) in enumerate(items):
            ax.plot(x, y, color=color, lw=0.8, alpha=0.7, label=label if k == 0 else None)
            kept.append(pd.DataFrame({"panel": pan, "run": run, "step": x, "value": y}))

    def mean_curve(items, grid):
        ys = [np.interp(grid, x, y) for _, x, y in items if len(x)]
        return np.mean(ys, axis=0) if ys else None

    obj_col = {"cost": COL["ppo"], "mass": "#377eb8", "co2": "#4daf4a"}
    algos = (("ppo", "2a", "feasibility_gated"), ("sac", "2b", "sac"), ("td3", "2b", "td3"), ("ddpg", "2b", "ddpg"))
    fig, axes = plt.subplots(2, 4, figsize=(W2, 4.0), constrained_layout=True)
    ax = axes[0, 0]
    for o, exp_, arm in (("cost", "2a", "feasibility_gated"), ("mass", "2c", "mass"), ("co2", "2c", "co2")):
        draw(ax, "a", curves(exp_, arm, "rollout/ep_rew_mean"), obj_col[o], OBJ_LABEL[o])
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("training reward (PPO)")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[0, 1]
    for algo, exp_, arm in algos:
        draw(ax, "b", curves(exp_, arm, "rollout/ep_rew_mean"), algo_col[algo], algo.upper())
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("training reward")
    ax.legend(frameon=False, loc="lower right")
    ax = axes[0, 2]
    for algo, exp_, arm in algos:
        draw(ax, "c", curves(exp_, arm, "rollout/ep_len_mean"), algo_col[algo], None)
    ax.set_ylabel("episode length (steps)")
    ax.set_ylim(bottom=0)
    ax = axes[0, 3]
    draw(ax, "d", curves("2a", "shaped", "rollout/ep_rew_mean"), COL["ppo"], None)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("training reward (shaped)")
    ax = axes[1, 0]
    draw(ax, "e", curves("2a", "lagrangian", "rollout/ep_rew_mean"), COL["ppo"], None)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("training reward (Lagr.)")
    grid = np.linspace(10_000, 1_000_000, 100)
    for ax, pan, pre, ylab in ((axes[1, 1], "f", "lagrangian/mean_violation_", "mean constraint violation"),
                               (axes[1, 2], "g", "lagrangian/lambda_", "Lagrange multiplier")):
        for tag, lab, c in (("g1_util", "utilisation", "#1b9e77"), ("g2_class", "section class", "#d95f02"),
                            ("g3_geom", "geometry", "#7570b3")):
            m = mean_curve(curves("2a", "lagrangian", pre + tag), grid)
            if m is not None:
                ax.plot(grid, np.maximum(m, 1e-8), color=c, lw=1.2, label=lab)
                kept.append(pd.DataFrame({"panel": pan, "run": "mean over seeds", "step": grid, "value": m,
                                          "tag": tag}))
        ax.set_yscale("log")
        ax.set_ylabel(ylab)
    h, l = axes[1, 1].get_legend_handles_labels()
    axes[1, 3].axis("off")
    axes[1, 3].legend(h, l, frameon=False, loc="center left", title="constraint")
    for a_, ch in zip(axes.ravel()[:7], "abcdefg"):
        panel(a_, ch)
        a_.set_xlabel("training steps")
        a_.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1e6:g}M" if v else "0"))
    save(fig, od, "fig7_training_curves", pd.concat(kept, ignore_index=True))


FIGS = {0: fig0, 1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7}


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
