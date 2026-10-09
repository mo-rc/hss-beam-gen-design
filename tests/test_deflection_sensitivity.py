"""pipeline/14_deflection_sensitivity.py on the tracked pooled ground truth (no search, a few seconds)."""
import importlib.util
import os

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(REPO, "data", "ground_truth", "main_grid_pooled")
spec = importlib.util.spec_from_file_location("sens", os.path.join(REPO, "pipeline", "14_deflection_sensitivity.py"))
sens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sens)

# reference designs (of 142 contexts) that violate L/300, L/360, L/500
GOLDEN = {"cost": (47, 75, 138), "mass": (129, 142, 142), "co2": (129, 142, 142)}


@pytest.fixture(scope="module")
def tables():
    return {o: sens.sensitivity(pd.read_csv(os.path.join(GT, f"ec3_optimal_designs_{o}.csv")), o, [250, 300, 360, 500])
            for o in sens.OBJECTIVES}


def test_golden_counts_and_baseline_limit(tables):
    for o, (a, b, c) in GOLDEN.items():
        t = tables[o].set_index("limit")
        assert t.loc["L/250", "n_ref_violating"] == 0 and t.loc["L/250", "n_stored_violating"] == 0  # stored designs are feasible at L/250
        assert (t.loc["L/300", "n_ref_violating"], t.loc["L/360", "n_ref_violating"], t.loc["L/500", "n_ref_violating"]) == (a, b, c)
        assert t.loc["L/250", "n_contexts"] == 142
    assert [int(tables[o].set_index("limit").loc["L/250", "n_ref_deflection_governs_at_250"]) for o in ("cost", "mass", "co2")] == [0, 35, 7]


def test_counts_grow_with_tighter_limit_and_scale_bound_is_modest(tables):
    for t in tables.values():
        assert t.n_ref_violating.is_monotonic_increasing and t.n_stored_violating.is_monotonic_increasing
        assert (t.scale_lb_max.dropna() >= 1.0).all() and t.scale_lb_max.max() < 2.2  # inside the scale operator's growth ladder


def test_reference_selection_matches_the_evaluation_script():
    ev = importlib.util.spec_from_file_location("ev", os.path.join(REPO, "pipeline", "03_evaluate_agent.py"))
    try:
        m = importlib.util.module_from_spec(ev)
        ev.loader.exec_module(m)
    except Exception as e:  # e.g. torch / stable-baselines3 missing at import time
        pytest.skip(f"03_evaluate_agent not importable here: {e}")
    for o in sens.OBJECTIVES:
        df = pd.read_csv(os.path.join(GT, f"ec3_optimal_designs_{o}.csv"))
        a = df.loc[sens.reference_index(df, o)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, m.ground_truth_optimum(GT, o))
