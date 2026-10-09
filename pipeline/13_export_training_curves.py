"""
pipeline/13_export_training_curves.py -- export TensorBoard training logs to small CSV files.

Walks runs/<experiment>/<arm>/seed<N>/ (written by pipeline/02_train_agent.py: meta.json + tensorboard/),
keeps only runs whose meta.json says status "complete", reads every scalar tag from all event files of the
run, bins the points by training step and writes one long-format CSV per run:

    results/training_curves/<experiment>__<arm>__seed<N>.csv     columns: tag, step, value
    results/training_curves/index.csv                            one row per exported run

If a run was resumed (several event files) the value logged last for a step wins; index.csv records
n_event_files, resumed_from_steps and the run's git_commit / git_dirty from its meta.json. No training,
no evaluation and no tuning happens here; the figure (fig7 in 10_make_figures.py) only reads these files.

    pip install tensorboard          # event-file reader only, torch is not needed
    python pipeline/13_export_training_curves.py --runs_root runs --out_dir results/training_curves
"""
import argparse
import glob
import json
import os

import pandas as pd


def find_runs(runs_root):
    """Run directories = parents of a 'tensorboard' directory, as (experiment, arm, seed_dir_name, path)."""
    runs = []
    for tb in sorted(glob.glob(os.path.join(runs_root, "**", "tensorboard"), recursive=True)):
        run_dir = os.path.dirname(tb)
        rel = os.path.relpath(run_dir, runs_root).replace("\\", "/").split("/")
        if len(rel) < 3:
            continue  # need experiment/arm/seed
        runs.append(("__".join(rel[:-2]), rel[-2], rel[-1], run_dir))
    return runs


def read_scalars(tb_dir):
    """All scalar events of all event files in tb_dir -> DataFrame(tag, step, value); latest wall_time wins."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows = []
    for f in sorted(glob.glob(os.path.join(tb_dir, "**", "events.out.tfevents*"), recursive=True)):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for e in ea.Scalars(tag):
                rows.append((tag, e.step, e.value, e.wall_time))
    if not rows:
        return pd.DataFrame(columns=["tag", "step", "value"])
    d = pd.DataFrame(rows, columns=["tag", "step", "value", "wall_time"])
    d = d.sort_values("wall_time").drop_duplicates(["tag", "step"], keep="last")
    return d[["tag", "step", "value"]].sort_values(["tag", "step"]).reset_index(drop=True)


def bin_steps(d, width):
    """Mean step and mean value per (tag, step // width) bin; width <= 0 keeps every point."""
    if width <= 0 or d.empty:
        return d
    d = d.assign(bin=d.step // width)
    out = d.groupby(["tag", "bin"], as_index=False).agg(step=("step", "mean"), value=("value", "mean"))
    out["step"] = out.step.round().astype(int)
    return out[["tag", "step", "value"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_root", default="runs")
    ap.add_argument("--out_dir", default=os.path.join("results", "training_curves"))
    ap.add_argument("--bin", type=int, default=10_000, help="bin width in training steps (0 = keep every point)")
    ap.add_argument("--min_steps", type=int, default=900_000, help="a complete run must have logged at least this many steps")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    index, skipped = [], []
    for exp, arm, seed, run_dir in find_runs(a.runs_root):
        name = f"{exp}__{arm}__{seed}"
        meta_path = os.path.join(run_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        status = meta.get("status")
        if status != "complete":
            skipped.append((name, f"status={status!r}" if status else "no meta.json"))
            continue
        d = read_scalars(os.path.join(run_dir, "tensorboard"))
        last = int(d.step.max()) if len(d) else 0
        if last < a.min_steps:
            skipped.append((name, f"only {last:,} logged steps"))
            continue
        binned = bin_steps(d, a.bin)
        binned.to_csv(os.path.join(a.out_dir, name + ".csv"), index=False)
        n_files = len(glob.glob(os.path.join(run_dir, "tensorboard", "**", "events.out.tfevents*"), recursive=True))
        index.append(dict(run=name, experiment=exp, arm=arm, seed=int(seed.replace("seed", "")),
                          last_step=last, n_tags=d.tag.nunique(), n_points=len(binned),
                          n_event_files=n_files, resumed_from_steps=meta.get("resumed_from_steps"),
                          git_commit=meta.get("git_commit"), git_dirty=meta.get("git_dirty")))
        print(f"wrote {name}.csv  ({d.tag.nunique()} tags, last step {last:,})")

    if index:
        pd.DataFrame(index).sort_values("run").to_csv(os.path.join(a.out_dir, "index.csv"), index=False)
    print(f"\n{len(index)} runs exported to {a.out_dir}")
    for name, why in skipped:
        print(f"SKIPPED {name}: {why}")


if __name__ == "__main__":
    main()
