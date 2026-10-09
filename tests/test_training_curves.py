"""pipeline/13_export_training_curves.py and fig7: run on small synthetic TensorBoard event files."""
import importlib.util
import json
import os

import pandas as pd
import pytest

pytest.importorskip("tensorboard")
pytest.importorskip("matplotlib")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(REPO, "pipeline", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


exp = _load("exp13", "13_export_training_curves.py")
figs = _load("figs10", "10_make_figures.py")


def _write_events(tb_dir, tags, steps, wall0=1.0, offset=0.0):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    os.makedirs(tb_dir, exist_ok=True)
    w = EventFileWriter(tb_dir)
    for i, st in enumerate(steps):
        for tag in tags:
            w.add_event(Event(wall_time=wall0 + i, step=int(st),
                              summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(st) / 1e4 + offset)])))
    w.close()


def _run(root, experiment, arm, seed, status="complete", tags=("rollout/ep_rew_mean", "rollout/ep_len_mean"),
         last=1_000_000):
    d = os.path.join(root, experiment, arm, f"seed{seed}")
    os.makedirs(d)
    if status:
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"status": status, "resumed_from_steps": 0, "git_commit": "abc", "git_dirty": False}, f)
    _write_events(os.path.join(d, "tensorboard", "PPO_1"), tags, range(10_000, last + 1, 10_000))
    return d


def test_export_keeps_complete_runs_only_and_bins(tmp_path):
    runs = str(tmp_path / "runs")
    _run(runs, "2a_reward_mode", "feasibility_gated", 42)
    _run(runs, "2a_reward_mode", "shaped", 42, status="running")
    _run(runs, "2a_reward_mode", "lagrangian", 42, status=None)
    _run(runs, "2b_algorithm", "sac", 42, last=500_000)  # complete status but stopped early
    out = str(tmp_path / "curves")
    import sys
    sys.argv = ["x", "--runs_root", runs, "--out_dir", out, "--bin", "50000"]
    exp.main()
    idx = pd.read_csv(os.path.join(out, "index.csv"))
    assert list(idx.run) == ["2a_reward_mode__feasibility_gated__seed42"]
    assert idx.n_event_files[0] == 1 and idx.resumed_from_steps[0] == 0 and idx.git_commit[0] == "abc" and not idx.git_dirty[0]
    d = pd.read_csv(os.path.join(out, idx.run[0] + ".csv"))
    assert set(d.tag) == {"rollout/ep_rew_mean", "rollout/ep_len_mean"}
    assert d.groupby("tag").size().max() <= 21  # 100 points binned into 50k-step bins
    assert d.step.max() <= 1_000_000


def test_resumed_run_latest_event_wins(tmp_path):
    tb = str(tmp_path / "tb")
    _write_events(os.path.join(tb, "PPO_1"), ["t"], [10, 20, 30], wall0=1.0, offset=0.0)
    _write_events(os.path.join(tb, "PPO_2"), ["t"], [20, 30, 40], wall0=100.0, offset=1000.0)  # resumed, later
    d = exp.read_scalars(tb).set_index("step").value
    assert d[10] == pytest.approx(0.001) and d[20] == pytest.approx(1000.002) and 40 in d.index


def test_fig7_written_from_exported_curves(tmp_path):
    runs, rd, od = str(tmp_path / "runs"), str(tmp_path / "results"), str(tmp_path / "figs")
    lag = ("rollout/ep_rew_mean", "rollout/ep_len_mean", "lagrangian/lambda_g1_util", "lagrangian/lambda_g2_class",
           "lagrangian/lambda_g3_geom", "lagrangian/mean_violation_g1_util", "lagrangian/mean_violation_g2_class",
           "lagrangian/mean_violation_g3_geom")
    for s in (42, 43):
        _run(runs, "2a_reward_mode", "feasibility_gated", s)
        _run(runs, "2a_reward_mode", "shaped", s)
        _run(runs, "2a_reward_mode", "lagrangian", s, tags=lag)
        for a in ("sac", "td3", "ddpg"):
            _run(runs, "2b_algorithm", a, s)
        for o in ("mass", "co2"):
            _run(runs, "2c_objective", o, s)
    import sys
    sys.argv = ["x", "--runs_root", runs, "--out_dir", os.path.join(rd, "training_curves")]
    exp.main()
    figs.fig7(rd, od)
    data = pd.read_csv(os.path.join(od, "fig7_training_curves_data.csv"))
    assert {"2c_objective__mass__seed42", "2c_objective__co2__seed43"} <= set(data[data.panel == "a"].run)
    for ext in (".png", ".pdf", "_data.csv"):
        assert os.path.getsize(os.path.join(od, "fig7_training_curves" + ext)) > 0


def test_fig7_skips_without_curves(tmp_path, capsys):
    figs.fig7(str(tmp_path), str(tmp_path / "f"))
    assert "skipped" in capsys.readouterr().out and not os.path.exists(str(tmp_path / "f"))
