"""hssbeamgen/algo/de_search.py and pipeline/04_baseline_search.py."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("gymnasium")
from hssbeamgen.algo.de_search import de_design  # noqa: E402
from hssbeamgen.algo.ga_search import ga_design, random_search_design  # noqa: E402

spec = importlib.util.spec_from_file_location("bs", os.path.join(REPO, "pipeline", "04_baseline_search.py"))
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def test_de_honours_exact_budget_and_is_deterministic():
    r1 = de_design(8000, 60, 20, "cost", n_evaluations=100, seed=3)
    r2 = de_design(8000, 60, 20, "cost", n_evaluations=100, seed=3)
    r3 = de_design(8000, 60, 20, "cost", n_evaluations=100, seed=4)
    assert r1["n_evaluations"] == 100
    assert (r1["h"], r1["cost"]) == (r2["h"], r2["cost"])
    assert (r1["h"], r1["cost"]) != (r3["h"], r3["cost"])
    with pytest.raises(ValueError):
        de_design(8000, 60, 20, "cost", n_evaluations=3)


def test_more_budget_does_not_hurt_de_on_average():
    lo = np.mean([de_design(9000, 70, 20, "cost", n_evaluations=60, seed=s)["fitness"] for s in range(6)])
    hi = np.mean([de_design(9000, 70, 20, "cost", n_evaluations=1500, seed=s)["fitness"] for s in range(6)])
    assert hi >= lo


def test_ga_budget_rule_never_exceeds_budget():
    for b in (20, 40, 112, 400, 4800):
        pop = bs.ga_pop(b)
        assert 6 <= pop <= 60
        r = ga_design(8000, 60, 20, "cost", pop_size=pop, n_generations=max(1, b // pop), seed=0)
        assert r["n_evaluations"] <= b and r["n_evaluations"] > b - pop


def test_run_search_dispatch_budgets():
    for m in bs.METHODS:
        r = bs.run_search(m, 8000, 60, 20, "cost", 112, 0)
        assert r["n_evaluations"] <= 112 and "cost" in r and "h" in r
    with pytest.raises(ValueError):
        bs.run_search("nope", 8000, 60, 20, "cost", 112, 0)


def test_search_cost_agrees_with_operator_none_row_and_counts_evals():
    """The search's own cost for its best design must equal the `none` row scored through the
    rl_final environment (same physics), and operators must add evaluations on top."""
    ev = bs._load_eval_module()
    opt = ev.ground_truth_optimum(os.path.join(REPO, "data", "ground_truth", "main_grid_pooled"), "cost")
    ctx = opt.iloc[40].to_dict()
    rows = bs._work((40, ctx, "ga", 400, 0, "cost", 20, tuple(ev.OPERATOR_MODES)))
    by = {m: row for (_, _, _, m, row) in rows}
    res = bs.run_search("ga", ctx["span_m"] * 1000.0, ctx["load_kN_per_m"], 20, "cost", 400, 0 * 100000 + 40)
    assert by["none"]["achieved"] == pytest.approx(res["cost"], rel=1e-9)
    assert by["none"]["feasible"] == res["feasible"]
    assert by["none"]["n_ec3"] == res["n_evaluations"]
    for m in ("scale", "scale+thin"):
        assert by[m]["n_ec3"] > by["none"]["n_ec3"]
        if by["none"]["feasible"]:
            assert by[m]["achieved"] <= by["none"]["achieved"] + 1e-9   # operators never worsen a feasible design


def test_script_refuses_non_pooled_dir_and_empty_reference():
    cmd = [sys.executable, os.path.join(REPO, "pipeline", "04_baseline_search.py"), "--out", "/tmp/x.csv"]
    r = subprocess.run(cmd + ["--gt_dir", "data/ground_truth/main_grid"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "_pooled" in (r.stdout + r.stderr)
    r = subprocess.run(cmd + ["--gt_dir", "data/ground_truth/ood_joint_pooled"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "untestable" in (r.stdout + r.stderr)


def test_meta_git_dirty_ignores_the_runs_own_untracked_outputs():
    """Outputs written under the repo are untracked; they must not make git_dirty True (as in 02/03)."""
    import json
    out = os.path.join(REPO, "results", "_pytest_search_meta_check.csv")
    try:
        r = subprocess.run([sys.executable, os.path.join(REPO, "pipeline", "04_baseline_search.py"),
                            "--gt_dir", "data/ground_truth/main_grid_pooled", "--n_contexts", "2", "--seeds", "1",
                            "--budgets", "20", "--methods", "ga", "--out", out],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        tracked_dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                            capture_output=True, text=True).stdout.strip())
        assert json.load(open(out.replace(".csv", "_meta.json")))["git_dirty"] == tracked_dirty
    finally:
        for s in (".csv", "_per_seed.csv", "_meta.json"):
            p = out.replace(".csv", s)
            if os.path.exists(p):
                os.remove(p)
