"""pipeline/11_build_paper_tables.py: tables reproduce the numbers logged in docs/ from results/."""
import importlib.util
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")


def _load():
    spec = importlib.util.spec_from_file_location("tables", os.path.join(REPO, "pipeline", "11_build_paper_tables.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t = _load()


def test_rl_summary_matches_logged_numbers():
    d = t.table_rl_experiments(RES)
    ppo = d["t2_algorithm_2b"].set_index("arm")
    assert round(ppo.loc["PPO", "gap_scale+thin"], 1) == 8.9
    assert round(ppo.loc["SAC", "gap_scale+thin"], 1) == 24.9
    assert round(ppo.loc["TD3", "gap_scale+thin"], 1) == 36.1
    assert round(ppo.loc["DDPG", "gap_scale+thin"], 1) == 32.1
    tr = d["t3_objective_transfer_2c"].set_index("arm")
    assert round(tr.loc["mass", "gap_scale+thin"], 1) == 7.2 and round(tr.loc["CO2", "gap_scale+thin"], 1) == 4.6


def test_matched_table_search_and_knn_rows():
    m = t.table_matched(RES).set_index(["objective", "method"])
    assert round(m.loc[("cost", "DE search, B=40"), "gap"], 1) == 39.2
    assert round(m.loc[("co2", "kNN, DE labels B=4800 (LOO)"), "gap"], 1) == 0.2
    assert m.loc[("mass", "PPO (5 seeds)"), "feasibility"] == pytest.approx(100.0)
    assert m.loc[("cost", "kNN, pooled labels (LOO)"), "labelling_evals"] == 141 * 96_000
    assert m.loc[("cost", "PPO (5 seeds)"), "training_steps"] == 1_000_000
    assert not m.labelling_evals.isna().any()


def test_ood_and_pairwise_tables():
    o = t.table_ood(RES).set_index(["objective", "set"])
    assert round(o.loc[("cost", "ood_span"), "gap"], 1) == 9.1 and o.loc[("cost", "ood_span"), "n_contexts"] == 26
    pw = t.table_pairwise(RES)
    row = pw[(pw.arm_a == "ppo") & (pw.arm_b == "td3")].iloc[0]
    assert round(row.p_holm, 3) == 0.048
    assert set(pw.experiment) == {"2a reward mode", "2b algorithm"}   # no cross-objective test


def test_build_writes_tables(tmp_path):
    t.build_tables(RES, str(tmp_path))
    for name in ("t1_reward_mode_2a", "t7_methods_matched", "t8_knn_cheap_labels"):
        assert (tmp_path / "tables" / f"{name}.md").stat().st_size > 0


def test_ppo_vs_search_significance_claims():
    d = t.table_ppo_vs_search(RES)
    st = d[d.operator == "scale+thin"]
    assert len(st) == 9 and (st.p_holm < 0.05).all() and st.separated.all()      # all 9 comparisons
    nn = d[d.operator == "none"]
    assert (nn.p_holm < 0.05).sum() == 7                                         # not mass vs DE / random
    weak = nn[nn.p_holm >= 0.05]
    assert set(weak.objective) == {"mass"} and set(weak.search) == {"DE (B=40)", "random (B=40)"}
    assert d.p_holm.min() >= 3 * 2 / 252 - 1e-9                                  # smallest attainable p


def test_operator_ablation_shows_feasibility_and_operator_effect():
    a = t.table_operator_ablation(RES).set_index(["objective", "method", "operator"])
    assert a.loc[("cost", "PPO", "none"), "feasibility"] == pytest.approx(100.0)
    assert a.loc[("cost", "PPO", "none"), "gap"] > 40 > a.loc[("cost", "PPO", "scale+thin"), "gap"]
    assert a.loc[("cost", "kNN (pooled labels, LOO)", "none"), "feasibility"] < 50   # kNN needs the operator for feasibility
