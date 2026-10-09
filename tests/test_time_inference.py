"""pipeline/12_time_inference.py: timing helpers count evaluations and report both stages."""
import importlib.util
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("timing", os.path.join(REPO, "pipeline", "12_time_inference.py"))
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def test_time_designs_counts_evals_and_orders_stages():
    ctxs = [dict(i=0), dict(i=1), dict(i=2)]
    td, tt, e_d, e_t = timing.time_designs(lambda c: ({"x": c["i"]}, 40), lambda c, d: 70, ctxs, repeats=2)
    assert len(td) == len(tt) == 6 and e_d == 40 and e_t == 110
    assert all(t_total >= t_design for t_design, t_total in zip(td, tt))
    row = timing._row("m", 40, td, tt, e_d, e_t)
    assert row["ms_per_eval_total"] == row["ms_total_mean"] / 110


def test_real_search_designer_runs_one_context():
    """DE and GA through the same run_search the baselines use, on a real context, with the real operator."""
    sb = timing._load("baseline_search", "04_baseline_search.py")
    res = sb.run_search("de", 9000.0, 60.0, 20, "cost", 40, 0)
    assert res["n_evaluations"] <= 40 and {"h", "b", "tf", "tw", "fy", "section_type"} <= set(res)
