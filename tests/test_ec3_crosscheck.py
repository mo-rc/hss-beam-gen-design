"""pipeline/14_ec3_crosscheck.py: reference-design audit, published IPE 500 benchmark, worksheet, FE section table."""
import importlib.util
import math
import os

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = os.path.join(REPO, "data", "ground_truth", "main_grid_pooled")
spec = importlib.util.spec_from_file_location("xc", os.path.join(REPO, "pipeline", "14_ec3_crosscheck.py"))
xc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xc)


def test_reference_audit_golden_counts():
    a = xc.audit(GT).set_index("objective")
    assert (a.n_reference == 142).all()
    assert a.loc["cost", "n_class1"] == 31 and a.loc["cost", "n_class2"] == 111 and a.loc["cost", "n_class3"] == 0
    assert a.loc["mass", "n_class3"] == 140 and a.loc["co2", "n_class3"] == 139
    assert [a.loc[o, "n_web_gt_72eps"] for o in ("cost", "mass", "co2")] == [85, 141, 141]
    assert [a.loc[o, "n_shear_gt_0p5"] for o in ("cost", "mass", "co2")] == [1, 0, 0]
    assert (a.max_tf_mm <= 35.0).all()  # below the 40 mm limit of EN 1993-1-1 Table 3.1: no yield-strength reduction
    # the omitted shear-buckling check is satisfied by every reference design
    assert (a.n_vbw_fail_rigid == 0).all() and (a.n_vbw_fail_nonrigid == 0).all()
    assert a.max_vbw_util_nonrigid.max() == pytest.approx(0.889, abs=0.001)


def test_shear_buckling_formula_by_hand():
    # h = 520, tf = 10 -> h_w = 500; tw = 10; fy = 355: eps = 0.8136, lambda_w = 500 / (86.4 * 10 * 0.8136) = 0.7113 < 0.83 -> chi_w = eta = 1
    u, lam = xc.shear_buckling_utilisation(520.0, 10.0, 10.0, 355.0, 500.0, eta=1.0)
    assert lam == pytest.approx(0.7113, abs=1e-3)
    assert u == pytest.approx(500.0 / (355.0 * 500.0 * 10.0 / math.sqrt(3) / 1e3), rel=1e-9)
    # lambda_w = 1.45 (web of the slenderest reference designs): rigid end post chi = 1.37 / (0.7 + 1.45)
    u2, lam2 = xc.shear_buckling_utilisation(520.0, 10.0, 500.0 / (1.45 * 86.4 * math.sqrt(235 / 355)), 355.0, 100.0)
    assert lam2 == pytest.approx(1.45, abs=1e-6)
    chi = 1.37 / (0.7 + 1.45)
    tw2 = 500.0 / (1.45 * 86.4 * math.sqrt(235 / 355))
    assert u2 == pytest.approx(100.0 / (chi * 355.0 * 500.0 * tw2 / math.sqrt(3) / 1e3), rel=1e-9)


def test_published_ipe500_benchmark():
    b = xc.benchmark_ipe500().set_index("quantity")
    r = b.rel_diff_pct.abs()
    assert r["Mcr,0 closed form, published section constants (kNm)"] < 0.1     # formula + E, G reproduce the published 895 kNm
    assert r["lambda_LT with published Mcr"] < 0.1 and r["phi_LT, curve b"] < 0.2 and r["chi_LT, curve b"] < 0.1
    assert r["Mcr,0 uniform moment (kNm)"] < 1.0 and r["Wpl,y (cm3)"] < 1.5      # the model's idealised section, IPE 500
    assert r["It (cm4)"] > 5 and b.loc["It (cm4)", "computed"] < b.loc["It (cm4)", "published"]  # torsion constant underestimated: conservative
    assert r.iloc[-1] < 1e-6                                                        # model chi code = published formula at the same lambda


def test_worksheet_hand_values_equal_the_model():
    for o in xc.OBJECTIVES:
        ref = xc.reference_designs(GT, o)
        d = ref.iloc[len(ref) // 2]
        rows, _ = xc.hand_calc(d.h, d.b, d.tf, d.tw, d.grade, d.span_m * 1000.0, d.load_kN_per_m, d.section_type)
        env = xc._env(o)
        xc._set(env, d.h, d.b, d.tf, d.tw, d.grade, d.section_type, d.span_m * 1000.0, d.load_kN_per_m)
        dbg = env._ec3_analysis()[-1]
        by = {r[0]: r[2] for r in rows}
        for step, key in xc.MODEL_KEYS.items():
            assert by[step] == pytest.approx(dbg[key], rel=1e-6), (o, step)
        assert by["M_b,Rd"] == pytest.approx(dbg["Mrd"], rel=1e-6)


def test_worksheet_is_rendered_with_empty_tool_column():
    md = xc.worksheet(GT)
    assert md.count("## ") == 3 and "| Tool |" in md and "| epsilon" in md


def test_fe_section_table_if_present():
    path = os.path.join(REPO, "results", "ec3_sections_fe.csv")
    if not os.path.exists(path):
        pytest.skip("run 14_ec3_crosscheck.py --sections")
    s = pd.read_csv(path).set_index("section")
    assert s.loc["IPE300", "A_cm2"] == pytest.approx(53.8, rel=0.003)        # catalogue value
    assert s.loc["IPE500", "Iy_cm4"] == pytest.approx(48200, rel=0.003)
    assert s.ratio_Wpl.between(1.02, 1.07).all() and (s.ratio_It > 1.1).all()
