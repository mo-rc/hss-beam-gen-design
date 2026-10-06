"""pipeline/05_baseline_knn.py: prediction rule, split protocols, and a physics round-trip."""
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(REPO, "data", "ground_truth", "main_grid_pooled")


def _load():
    spec = importlib.util.spec_from_file_location("knn", os.path.join(REPO, "pipeline", "05_baseline_knn.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


knn = _load()


def _train(rows):
    return pd.DataFrame(rows, columns=["span_m", "load_kN_per_m", "grade", "section_type", "h", "b", "tf", "tw"])


def test_k1_at_a_training_context_returns_its_stored_design():
    tr = _train([[6, 20, 355, "rolled", 250, 125, 8, 6], [10, 60, 460, "welded", 400, 200, 12, 8],
                 [15, 140, 690, "welded", 700, 300, 20, 10]])
    d = knn.KNNDesigner(tr, k=1).predict(10, 60)
    assert d["fy"] == 460 and d["section_type"] == "welded"
    assert [d[g] for g in knn.GEOM] == pytest.approx([400, 200, 12, 8])


def test_grade_is_voted_not_averaged_and_geometry_uses_winners_only():
    # two close S355 neighbours outvote one slightly closer-in-scale S690; mean grade (~467) never appears
    tr = _train([[8, 50, 355, "rolled", 300, 150, 10, 6], [8.2, 52, 355, "rolled", 320, 160, 10, 6],
                 [8.1, 51, 690, "welded", 900, 400, 30, 20], [15, 140, 690, "welded", 700, 300, 20, 10]])
    d = knn.KNNDesigner(tr, k=3).predict(8.1, 51)
    assert d["fy"] in (355.0, 690.0)
    if d["fy"] == 355.0:   # geometry is a mean of the two S355 rows only
        assert 300 <= d["h"] <= 320
    # a clear majority must win regardless of scale
    tr2 = _train([[8, 50, 355, "rolled", 300, 150, 10, 6], [8.3, 50, 355, "rolled", 310, 150, 10, 6],
                  [8.2, 50, 355, "rolled", 305, 150, 10, 6], [8.1, 50, 690, "welded", 900, 400, 30, 20],
                  [15, 140, 690, "welded", 700, 300, 20, 10]])
    d2 = knn.KNNDesigner(tr2, k=4).predict(8.15, 50)
    assert d2["fy"] == 355.0 and 300 <= d2["h"] <= 310


def test_tie_goes_to_nearest_neighbour():
    tr = _train([[8, 50, 355, "rolled", 300, 150, 10, 6], [9, 50, 460, "rolled", 400, 200, 12, 8],
                 [15, 140, 690, "welded", 700, 300, 20, 10]])
    d = knn.KNNDesigner(tr, k=2).predict(8.9, 50)   # equal-weight-ish pair; nearest is the S460 row
    assert d["fy"] == 460.0


def test_splits_subsample_disjoint_reproducible_and_loo_excludes_self():
    opt = _train([[i, 10 * i, 355, "rolled", 300, 150, 10, 6] for i in range(1, 21)])
    a = list(knn.splits("subsample", opt, None, [5, 10], 3, 7))
    b = list(knn.splits("subsample", opt, None, [5, 10], 3, 7))
    assert len(a) == 6
    for (n, rep, tr, te), (_, _, tr2, te2) in zip(a, b):
        assert len(tr) == n and len(tr) + len(te) == 20
        assert not set(tr.span_m) & set(te.span_m)
        assert list(tr.span_m) == list(tr2.span_m)
    loo = list(knn.splits("loo", opt, None, [], 1, 0))
    assert len(loo) == 20
    for n, i, tr, te in loo:
        assert len(tr) == 19 and len(te) == 1 and te.span_m.iloc[0] not in set(tr.span_m)
    ood = list(knn.splits("ood", opt, opt.head(3), [], 1, 0))
    assert len(ood) == 1 and len(ood[0][2]) == 20 and len(ood[0][3]) == 3
    with pytest.raises(ValueError):
        list(knn.splits("subsample", opt, None, [20], 1, 0))


def test_script_refuses_non_pooled_ground_truth_dir():
    r = subprocess.run([sys.executable, os.path.join(REPO, "pipeline", "05_baseline_knn.py"), "--protocol", "loo",
                        "--train_gt_dir", "data/ground_truth/main_grid", "--out", "/tmp/x.csv"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "_pooled" in (r.stdout + r.stderr)


def test_roundtrip_against_real_physics_and_ground_truth():
    """With train == test and k=1 the kNN returns the stored optimum, so under mode `none` it must
    reproduce the reference exactly (feasible, gap ~ 0). Catches any column/unit/mapping bug."""
    pytest.importorskip("gymnasium")
    ev = knn._load_eval_module()
    opt = ev.ground_truth_optimum(GT, "cost").sample(n=4, random_state=0).reset_index(drop=True)
    env = knn.make_env("cost")
    res = knn.score(env, ev, knn.KNNDesigner(opt, k=1), opt, "cost", 20)
    none = res["none"]
    assert none.feasible.all()
    assert np.nanmax(np.abs(none.gap.to_numpy(float))) < 1e-6
    assert (none.grade_match == 1.0).all()
    for m in ("scale", "scale+thin"):          # operators may only improve or hold an exact optimum
        assert res[m].feasible.all()
        assert np.nanmax(res[m].gap.to_numpy(float)) < 1e-3


def test_meta_git_dirty_ignores_the_runs_own_untracked_outputs():
    """Outputs written under the repo are untracked; they must not make git_dirty True (as in 02/03)."""
    import json
    out = os.path.join(REPO, "results", "_pytest_knn_meta_check.csv")
    try:
        r = subprocess.run([sys.executable, os.path.join(REPO, "pipeline", "05_baseline_knn.py"), "--protocol", "loo",
                            "--train_gt_dir", GT, "--n_contexts", "4", "--k", "1", "--headline_k", "1", "--out", out],
                           cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        tracked_dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                            capture_output=True, text=True).stdout.strip())
        assert json.load(open(out.replace(".csv", "_meta.json")))["git_dirty"] == tracked_dirty
    finally:
        for s in (".csv", "_per_rep.csv", "_meta.json"):
            p = out.replace(".csv", s)
            if os.path.exists(p):
                os.remove(p)
