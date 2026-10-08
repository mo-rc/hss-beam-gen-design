"""pipeline/10_make_figures.py: runs on the saved results and plots the numbers the logs report."""
import importlib.util
import os

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytest.importorskip("matplotlib")
spec = importlib.util.spec_from_file_location("figs", os.path.join(REPO, "pipeline", "10_make_figures.py"))
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)
RD = os.path.join(REPO, "results")


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("figs"))
    for i in sorted(figs.FIGS):
        figs.FIGS[i](RD, d)
    return d


def test_all_figures_written(out):
    for n in ("fig1_gap_vs_evaluations", "fig2_in_dist_vs_ood", "fig3_ablations", "fig4_knn_label_cost"):
        for ext in ("png", "pdf", "_data.csv"):
            path = os.path.join(out, n + ("." + ext if not ext.startswith("_") else ext))
            assert os.path.getsize(path) > 0


def test_plotted_numbers_match_the_results_logs(out):
    d1 = pd.read_csv(os.path.join(out, "fig1_gap_vs_evaluations_data.csv"))
    ppo = d1[(d1.method == "ppo")].set_index("obj").gap
    assert ppo["cost"] == pytest.approx(8.9, abs=0.1) and ppo["mass"] == pytest.approx(7.2, abs=0.1)
    assert ppo["co2"] == pytest.approx(4.6, abs=0.1)
    knn = d1[d1.method == "knn_loo"].set_index("obj").gap
    assert knn["cost"] == pytest.approx(1.1, abs=0.1) and knn["co2"] == pytest.approx(0.5, abs=0.1)
    de = d1[(d1.method == "de") & (d1.budget == 4800)].set_index("obj").gap
    assert de["cost"] == pytest.approx(-0.5, abs=0.1)
    d3 = pd.read_csv(os.path.join(out, "fig3_ablations_data.csv"))
    assert d3[d3.arm == "DDPG"].gap.max() == pytest.approx(88.5, abs=0.1)
    assert len(d3) == 3 * 5 + 4 * 5
    d4 = pd.read_csv(os.path.join(out, "fig4_knn_label_cost_data.csv"))
    r = d4[(d4.obj == "cost") & (d4.labels == "de1000") & (d4.n == 36)].iloc[0]
    assert r.cost == 36_000 and r.gap == pytest.approx(3.8, abs=0.1)
