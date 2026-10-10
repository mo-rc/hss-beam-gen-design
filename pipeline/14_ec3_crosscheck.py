"""
pipeline/14_ec3_crosscheck.py -- checks of the EC3 model that do not rely on the model's own code.

No training, no search; seconds. Four parts (all written to results/ unless stated):

  audit      Reference designs (best known per (span, load) context) against checks the model omits or
             simplifies: section class, shear ratio, web slenderness, plate thickness, and the shear-buckling
             resistance of EN 1993-1-5 5.3 (web with end stiffeners only, eta = 1.0, gamma_M1 = 1.0), which
             EN 1993-1-1 6.2.6(6) requires when h_w / t_w > 72 eps / eta and which the model does not check.
             If a reference design satisfies an additional constraint it stays best known for the stricter problem.
             -> ec3_reference_audit.csv

  benchmark  Published example: IPE 500, S235, L = 3.75 m, fork supports, general method 6.3.2.2, curve b
             (ECCS TC8 No. 119, example 5; reproduced in SOFiSTiK benchmark DCE-EN24). Elastic critical moment
             for uniform moment, lambda_LT and chi_LT, published vs this model's section properties.
             -> ec3_benchmark_ipe500.csv

  sections   (needs `pip install sectionproperties`) Finite-element section properties of standard European
             rolled sections (true root radius) against the model's sharp-corner formulas, i.e. what the
             model's rolled-section factors (area 1.05, I_y 1.02, W_pl 1.05, I_t 1.15) stand for.
             -> ec3_sections_fe.csv

  designs    (needs per-design detail files from `03_evaluate_agent.py --out_detail`, which now include h, b, tf, tw)
             The same shear-buckling check on the designs a method actually produced, per file.
             -> ec3_methods_audit.csv

  worksheet  Step-by-step hand calculation of three reference designs with the model's value next to each step
             and an empty column for a value from an independent tool (commercial software, spreadsheet).
             -> docs/ec3_worksheet.md

    python pipeline/14_ec3_crosscheck.py                 # audit, benchmark, worksheet
    python pipeline/14_ec3_crosscheck.py --sections      # also the FE section check
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
OBJECTIVES = ("cost", "mass", "co2")
E, G = 210_000.0, 81_000.0
STOREY = 20

# Published values, IPE 500, S235, L = 3750 mm, fork supports (SOFiSTiK DCE-EN24 / ECCS No. 119 ex. 5, Table 1-2):
# section: h 500 (web height 468), b 200, tf 16, tw 10.2, Wpl,y 2194 cm3, Iz 2142 cm4, It 88.57 cm4, Iw 1236e3 cm6.
IPE500 = dict(h=500.0, b=200.0, tf=16.0, tw=10.2, fy=235.0, L=3750.0,
              pub_Wpl_cm3=2194.0, pub_Iz_cm4=2142.0, pub_It_cm4=88.57, pub_Iw_cm6=1236e3,
              pub_Mcr0_kNm=895.0, pub_lambda0=0.759,
              pub_C1=1.194, pub_Mcr_kNm=1068.0, pub_lambda_LT=0.695, pub_phi_LT=0.825, pub_chi_LT=0.787, pub_alpha_LT=0.34)

# Standard European rolled sections (h, b, tw, tf, r) in mm.
ROLLED = {"IPE200": (200, 100, 5.6, 8.5, 12), "IPE300": (300, 150, 7.1, 10.7, 15), "IPE500": (500, 200, 10.2, 16.0, 21),
          "HEA300": (290, 300, 8.5, 14.0, 27), "HEB300": (300, 300, 11.0, 19.0, 27),
          "HEA500": (490, 300, 12.0, 23.0, 27), "HEB500": (500, 300, 14.5, 28.0, 27)}


def _git(*a):
    try:
        return subprocess.check_output(["git", *a], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _env(objective="cost"):
    from hssbeamgen.algo.ga_search import _make_probe_env
    return _make_probe_env(objective)


def _set(env, h, b, tf, tw, fy, section_type, span_mm, load):
    env.h, env.b, env.tf, env.tw = float(h), float(b), float(tf), float(tw)
    env.fy, env.section_type = float(fy), section_type
    env.span, env.load, env.storey = float(span_mm), float(load), STOREY


def reference_designs(gt_dir, objective):
    """Best row per (span, load): the selection of 03_evaluate_agent.ground_truth_optimum."""
    df = pd.read_csv(os.path.join(gt_dir, f"ec3_optimal_designs_{objective}.csv"))
    return df.loc[df.groupby(["span_m", "load_kN_per_m"])[objective].idxmin()].reset_index(drop=True)


def shear_buckling_utilisation(h, tf, tw, fy, Ved_kN, eta=1.0, rigid_end_post=True):
    """EN 1993-1-5 5.2-5.3: web with transverse stiffeners at the supports only, lambda_w = h_w / (86.4 t eps);
    chi_w from Table 5.1; V_bw,Rd = chi_w f_y h_w t_w / (sqrt(3) gamma_M1), gamma_M1 = 1.0; flange contribution ignored."""
    eps = math.sqrt(235.0 / fy)
    hw = h - 2.0 * tf
    lam = hw / (86.4 * tw * eps)
    if lam < 0.83 / eta:
        chi = eta
    elif lam < 1.08:
        chi = 0.83 / lam
    else:
        chi = 1.37 / (0.7 + lam) if rigid_end_post else 0.83 / lam
    return Ved_kN / (chi * fy * hw * tw / math.sqrt(3.0) / 1e3), lam


def audit(gt_dir):
    rows = []
    for o in OBJECTIVES:
        env = _env(o)
        ref = reference_designs(gt_dir, o)
        cls, shear, web, vb_r, vb_n = [], [], [], [], []
        for r in ref.itertuples():
            _set(env, r.h, r.b, r.tf, r.tw, r.grade, r.section_type, r.span_m * 1000.0, r.load_kN_per_m)
            dbg = env._ec3_analysis()[-1]
            eps = math.sqrt(235.0 / env.fy)
            cls.append(dbg["section_class"])
            shear.append(dbg["shear_ratio"])
            web.append((r.h - 2 * r.tf) / r.tw / eps)
            vb_r.append(shear_buckling_utilisation(r.h, r.tf, r.tw, env.fy, dbg["Ved"], 1.0, True)[0])
            vb_n.append(shear_buckling_utilisation(r.h, r.tf, r.tw, env.fy, dbg["Ved"], 1.0, False)[0])
        cls, shear, web, vb_r, vb_n = map(np.array, (cls, shear, web, vb_r, vb_n))
        rows.append(dict(
            objective=o, n_reference=len(ref),
            n_rolled=int((ref.section_type == "rolled").sum()), n_welded=int((ref.section_type == "welded").sum()),
            n_class1=int((cls == 1).sum()), n_class2=int((cls == 2).sum()), n_class3=int((cls == 3).sum()),
            max_shear_ratio=float(shear.max()), n_shear_gt_0p5=int((shear > 0.5).sum()),
            n_web_gt_72eps=int((web > 72).sum()), max_web_slenderness_eps=float(web.max()),
            max_tf_mm=float(ref.tf.max()), max_tw_mm=float(ref.tw.max()),
            max_vbw_util_rigid=float(vb_r.max()), n_vbw_fail_rigid=int((vb_r > 1.0).sum()),
            max_vbw_util_nonrigid=float(vb_n.max()), n_vbw_fail_nonrigid=int((vb_n > 1.0).sum())))
    return pd.DataFrame(rows)


def _chi_curve_b(lam, alpha=0.34):
    phi = 0.5 * (1 + alpha * (lam - 0.2) + lam ** 2)
    return min(1.0, 1 / (phi + math.sqrt(phi ** 2 - lam ** 2)))


def audit_designs(detail_csv):
    """Shear-buckling utilisation (EN 1993-1-5 5.3, end stiffeners only, eta = 1.0) of the designs in one detail file
    of 03_evaluate_agent.py (columns span_m, load_kN_per_m, h, b, tf, tw, grade, feasible). Feasible designs only,
    plus the count of all designs."""
    d = pd.read_csv(detail_csv)
    f = d[d.feasible.astype(bool)]
    ur, un, web = [], [], []
    for r in f.itertuples():
        ved = r.load_kN_per_m * r.span_m / 2.0
        ur.append(shear_buckling_utilisation(r.h, r.tf, r.tw, r.grade, ved, 1.0, True)[0])
        un.append(shear_buckling_utilisation(r.h, r.tf, r.tw, r.grade, ved, 1.0, False)[0])
        web.append((r.h - 2 * r.tf) / r.tw / math.sqrt(235.0 / r.grade))
    ur, un, web = np.array(ur), np.array(un), np.array(web)
    return dict(file=os.path.join(os.path.basename(os.path.dirname(detail_csv)), os.path.basename(detail_csv)).replace(os.sep, "/"), n_designs=len(d), n_feasible=len(f),
                n_web_gt_72eps=int((web > 72).sum()), max_web_slenderness_eps=float(web.max()) if len(f) else float("nan"),
                max_vbw_util_rigid=float(ur.max()) if len(f) else float("nan"), n_vbw_fail_rigid=int((ur > 1.0).sum()),
                max_vbw_util_nonrigid=float(un.max()) if len(f) else float("nan"), n_vbw_fail_nonrigid=int((un > 1.0).sum()))


def benchmark_ipe500():
    """Published IPE 500 example vs the model's formulas on the model's own (idealised) section properties."""
    p = IPE500
    env = _env("cost")
    env.ltb_restraint_factor = 1.0          # fork supports at both ends, L_cr = L
    env.include_zg_in_mcr = False
    _set(env, p["h"], p["b"], p["tf"], p["tw"], p["fy"], "rolled", p["L"], 50.0)
    util_m, _m, _p, _c, chi_model, dbg = env._ec3_analysis()
    # the model uses C1 = 1.13 (UDL); remove it to compare with the published uniform-moment value C1 = 1
    lam_model = dbg["lambda_lt"]
    hw = p["h"] - 2 * p["tf"]
    Wpl_model = (2 * p["b"] * p["tf"] * (p["h"] / 2 - p["tf"] / 2) + p["tw"] * hw ** 2 / 4) * 1.05
    Mcr_model = Wpl_model * p["fy"] / lam_model ** 2 / 1e6          # kNm, includes C1 = 1.13
    Mcr0_model = Mcr_model / 1.13
    lam0_model = math.sqrt(Wpl_model * p["fy"] / (Mcr0_model * 1e6))
    Iz = 2 * p["tf"] * p["b"] ** 3 / 12 + hw * p["tw"] ** 3 / 12
    It = (2 * p["b"] * p["tf"] ** 3 + hw * p["tw"] ** 3) / 3 * 1.15
    Iw = Iz * (p["h"] - p["tf"]) ** 2 / 4
    # published chain: Mcr = C1 Mcr0, lambda, phi, chi (general method, curve b)
    pub_lam = math.sqrt(p["pub_Wpl_cm3"] * 1e3 * p["fy"] / (p["pub_Mcr_kNm"] * 1e6))
    pub_phi = 0.5 * (1 + p["pub_alpha_LT"] * (pub_lam - 0.2) + pub_lam ** 2)
    pub_chi = 1 / (pub_phi + math.sqrt(pub_phi ** 2 - pub_lam ** 2))
    # Mcr0 from the closed form with the published section constants
    pubIz, pubIt, pubIw = p["pub_Iz_cm4"] * 1e4, p["pub_It_cm4"] * 1e4, p["pub_Iw_cm6"] * 1e6
    Mcr0_closed_pub = (math.pi ** 2 * E * pubIz / p["L"] ** 2) * math.sqrt(pubIw / pubIz + p["L"] ** 2 * G * pubIt / (math.pi ** 2 * E * pubIz)) / 1e6
    rows = [
        ("Wpl,y (cm3)", p["pub_Wpl_cm3"], Wpl_model / 1e3),
        ("Iz (cm4)", p["pub_Iz_cm4"], Iz / 1e4),
        ("It (cm4)", p["pub_It_cm4"], It / 1e4),
        ("Iw (cm6)", p["pub_Iw_cm6"], Iw / 1e6),
        ("Mcr,0 uniform moment (kNm)", p["pub_Mcr0_kNm"], Mcr0_model),
        ("lambda_0 uniform moment", p["pub_lambda0"], lam0_model),
        ("Mcr,0 closed form, published section constants (kNm)", p["pub_Mcr0_kNm"], Mcr0_closed_pub),
        ("lambda_LT with published Mcr", p["pub_lambda_LT"], pub_lam),
        ("phi_LT, curve b", p["pub_phi_LT"], pub_phi),
        ("chi_LT, curve b", p["pub_chi_LT"], pub_chi),
        ("chi_LT of the model code at the model's own lambda_LT (C1 = 1.13), vs published formula at that lambda",
         _chi_curve_b(lam_model), chi_model),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "published", "computed"])
    out["rel_diff_pct"] = 100.0 * (out.computed - out.published) / out.published
    out["source_of_computed"] = (["model section formulas (rolled factors)"] * 6 + ["closed form, published constants"]
                                 + ["published chain formulas"] * 3 + ["model code (hss_env) vs published formula"])
    return out


def sections_fe():
    import warnings
    warnings.filterwarnings("ignore")
    from sectionproperties.analysis import Section
    from sectionproperties.pre.library import i_section
    rows = []
    for name, (h, b, tw, tf, r) in ROLLED.items():
        geo = i_section(d=h, b=b, t_f=tf, t_w=tw, r=r, n_r=16)
        geo.create_mesh(mesh_sizes=[max(tf, tw) ** 2 / 4])
        s = Section(geo)
        s.calculate_geometric_properties()
        s.calculate_plastic_properties()
        s.calculate_warping_properties()
        hw = h - 2 * tf
        A0 = hw * tw + 2 * b * tf
        Iy0 = tw * hw ** 3 / 12 + 2 * (b * tf ** 3 / 12 + b * tf * (h / 2 - tf / 2) ** 2)
        Wpl0 = 2 * b * tf * (h / 2 - tf / 2) + tw * hw ** 2 / 4
        It0 = (2 * b * tf ** 3 + hw * tw ** 3) / 3
        Iz0 = 2 * tf * b ** 3 / 12 + hw * tw ** 3 / 12
        rows.append(dict(section=name, A_cm2=s.get_area() / 100, Iy_cm4=s.get_ic()[0] / 1e4, Wpl_cm3=s.get_sp()[0] / 1e3,
                         ratio_A=s.get_area() / A0, ratio_Iy=s.get_ic()[0] / Iy0, ratio_Wpl=s.get_sp()[0] / Wpl0,
                         ratio_It=s.get_j() / It0, ratio_Iz=s.get_ic()[1] / Iz0,
                         model_A=1.05, model_Iy=1.02, model_Wpl=1.05, model_It=1.15))
    return pd.DataFrame(rows).round(6)  # rounded so that regeneration on another machine does not rewrite the tracked file


def hand_calc(h, b, tf, tw, fy, span_mm, load, section_type, ltb_restraint_factor=0.40, sls_load_factor=0.50):
    """Every step of the member check as written in EN 1993-1-1 (clauses as in the docstring of this file),
    returned as ordered (step, formula, value, unit) rows. Written out explicitly so that each line can be
    reproduced with a calculator or a spreadsheet."""
    rolled = section_type == "rolled"
    R = []

    def add(step, formula, value, unit=""):
        R.append((step, formula, value, unit))
        return value
    eps = add("epsilon", "sqrt(235/fy)", math.sqrt(235.0 / fy))
    hw = add("web height h_w", "h - 2 tf", h - 2 * tf, "mm")
    r = add("root radius used", "0.1 tf (rolled), 0 (welded)", 0.1 * tf if rolled else 0.0, "mm")
    cf = add("flange outstand c_f", "(b - tw)/2 - r", (b - tw) / 2 - r, "mm")
    cw = add("web depth c_w", "h_w - 2 r", hw - 2 * r, "mm")
    rf = add("c_f / (t_f eps)", "Table 5.2 sheet 2: class 1 <= 9, 2 <= 10, 3 <= 14", cf / tf / eps)
    rw = add("c_w / (t_w eps)", "Table 5.2 sheet 1 (bending): class 1 <= 72, 2 <= 83, 3 <= 124", cw / tw / eps)
    if rf <= 9 and rw <= 72:
        cls = 1
    elif rf <= 10 and rw <= 83:
        cls = 2
    elif rf <= 14 and rw <= 124:
        cls = 3
    else:
        cls = 4
    add("section class", "worse of flange and web", cls)
    fa, fi, tfac = (1.05, 1.02, 1.15) if rolled else (1.0, 1.0, 1.0)
    A = add("A", "(h_w tw + 2 b tf) x %.2f" % fa, (hw * tw + 2 * b * tf) * fa, "mm2")
    Iy = add("I_y", "[tw h_w^3/12 + 2(b tf^3/12 + b tf (h/2 - tf/2)^2)] x %.2f" % fi,
             (tw * hw ** 3 / 12 + 2 * (b * tf ** 3 / 12 + b * tf * (h / 2 - tf / 2) ** 2)) * fi, "mm4")
    Wel = add("W_el,y", "I_y / (h/2)", Iy / (h / 2), "mm3")
    Wpl = add("W_pl,y", "[2 b tf (h/2 - tf/2) + tw h_w^2/4] x %.2f" % fa, (2 * b * tf * (h / 2 - tf / 2) + tw * hw ** 2 / 4) * fa, "mm3")
    Iz = add("I_z", "2 tf b^3/12 + h_w tw^3/12", 2 * tf * b ** 3 / 12 + hw * tw ** 3 / 12, "mm4")
    W = Wpl if cls <= 2 else Wel
    Mc = add("M_c,Rd", "W fy / gamma_M0 (gamma_M0 = 1.0), W = W_pl (class 1-2) or W_el (class 3)", W * fy / 1e6, "kNm")
    It = add("I_t", "(2 b tf^3 + h_w tw^3)/3 x %.2f" % tfac, (2 * b * tf ** 3 + hw * tw ** 3) / 3 * tfac, "mm4")
    Iw = add("I_w", "I_z (h - tf)^2 / 4", Iz * (h - tf) ** 2 / 4, "mm6")
    Lcr = add("L_cr", "%.2f L (assumed restraint spacing, not an EC3 value)" % ltb_restraint_factor, span_mm * ltb_restraint_factor, "mm")
    C1 = add("C1", "1.13 (UDL, simply supported, k = 1)", 1.13)
    Mcr = add("M_cr", "C1 pi^2 E Iz / L_cr^2 sqrt(Iw/Iz + L_cr^2 G It / (pi^2 E Iz))",
              C1 * math.pi ** 2 * E * Iz / Lcr ** 2 * math.sqrt(Iw / Iz + Lcr ** 2 * G * It / (math.pi ** 2 * E * Iz)) / 1e6, "kNm")
    lam = add("lambda_LT", "sqrt(W fy / M_cr)", math.sqrt(W * fy / 1e6 / Mcr))
    if rolled:
        alpha = 0.34 if h / b > 2 else 0.21
    else:
        alpha = 0.76 if h / b > 2 else 0.49
    add("alpha_LT", "Table 6.4 (general case): rolled h/b<=2 a 0.21, >2 b 0.34; welded h/b<=2 c 0.49, >2 d 0.76", alpha)
    phi = add("phi_LT", "0.5 [1 + alpha (lambda - 0.2) + lambda^2]  (6.56)", 0.5 * (1 + alpha * (lam - 0.2) + lam ** 2))
    chi = add("chi_LT", "min(1, 1 / (phi + sqrt(phi^2 - lambda^2)))  (6.56)", min(1.0, 1.0 / (phi + math.sqrt(phi ** 2 - lam ** 2))))
    Mb = add("M_b,Rd", "chi_LT M_c,Rd", chi * Mc, "kNm")
    Av = add("A_v", "A - 2 b tf + (tw + 2r) tf (rolled); h_w tw (welded)  (6.2.6(3))",
             A - 2 * b * tf + (tw + 2 * r) * tf if rolled else hw * tw, "mm2")
    Vpl = add("V_pl,Rd", "A_v fy / (sqrt(3) gamma_M0)", Av * fy / math.sqrt(3) / 1e3, "kN")
    L_m = span_mm / 1000.0
    Ved = add("V_Ed", "w L / 2", load * L_m / 2, "kN")
    Med = add("M_Ed", "w L^2 / 8", load * L_m ** 2 / 8, "kNm")
    add("V_Ed / V_pl,Rd", "(interaction reduction applies only if > 0.5)", Ved / Vpl)
    u = add("M_Ed / M_b,Rd", "ULS utilisation", Med / Mb)
    delta = add("delta", "5 w_sls L^4 / (384 E I_y), w_sls = %.2f w" % sls_load_factor,
                5 * load * sls_load_factor * span_mm ** 4 / (384 * E * Iy), "mm")
    ud = add("delta / (L/250)", "SLS utilisation", delta / (span_mm / 250.0))
    # not in the model: shear buckling, EN 1993-1-5 5.3 (end stiffeners only, eta = 1.0)
    ub, lw = shear_buckling_utilisation(h, tf, tw, fy, Ved, 1.0, True)
    add("lambda_w (not in model)", "h_w / (86.4 tw eps)  (EN 1993-1-5 5.3(5))", lw)
    add("V_Ed / V_bw,Rd (not in model)", "chi_w from Table 5.1, rigid end post, gamma_M1 = 1.0", ub)
    return R, dict(cls=cls, Mc=Mc, Mcr=Mcr, lam=lam, chi=chi, Mb=Mb, Vpl=Vpl, util_moment=u, util_defl=ud, Ved=Ved, Med=Med)


MODEL_KEYS = {"section class": "section_class", "lambda_LT": "lambda_lt", "V_pl,Rd": "Vpl_Rd", "V_Ed": "Ved", "M_Ed": "Med",
              "V_Ed / V_pl,Rd": "shear_ratio", "M_Ed / M_b,Rd": "moment_util", "delta / (L/250)": "deflection_util"}


def central_design(gt_dir, objective):
    """Reference design of the context closest to (median span, median load)."""
    ref = reference_designs(gt_dir, objective)
    med_s, med_l = ref.span_m.median(), ref.load_kN_per_m.median()
    return ref.iloc[int(((ref.span_m - med_s) ** 2 / med_s ** 2 + (ref.load_kN_per_m - med_l) ** 2 / med_l ** 2).idxmin())]


def write_workbook(path, gt_dir):
    """Excel workbook with live formulas for the three worksheet designs. Inputs (blue) can be edited; the `Python` column
    holds the values of `hand_calc` (fixed numbers) and `Difference` compares the workbook's formulas with them; the last
    column is empty for a reviewer's own calculation. Needs openpyxl; recalculate with LibreOffice or open in Excel."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Font, PatternFill
    A = "Arial"
    blue, bold = Font(name=A, color="0000FF", size=10), Font(name=A, bold=True, size=10)
    norm = Font(name=A, size=10)
    head = PatternFill("solid", fgColor="D9E1F2")
    wb = Workbook()
    rd = wb.active
    rd.title = "Read me"
    for i, t in enumerate([
            "EC3 hand-calculation workbook (generated by pipeline/14_ec3_crosscheck.py --xlsx)",
            "One sheet per objective: the best-known reference design of the central context (median span, median load).",
            "Blue cells are inputs and can be changed; every other number in the column 'Excel' is a live formula.",
            "Column 'Python' holds the values computed by hand_calc() in the script (fixed numbers); 'Difference' = Excel / Python - 1.",
            "Column 'Reviewer' is empty: enter your own calculation (calculator, other software) for any step.",
            "Model assumptions that are not EC3 values are marked 'assumption' next to their cell (L_cr, C1, serviceability load, rolled factors).",
            "Steps marked 'not in model' (shear buckling, EN 1993-1-5 5.3) are not part of the model's check."], 1):
        rd.cell(i, 1, t).font = bold if i == 1 else norm
    rd.column_dimensions["A"].width = 120
    keys = {"section_type": None}
    for o in OBJECTIVES:
        d = central_design(gt_dir, o)
        rows, _ = hand_calc(d.h, d.b, d.tf, d.tw, d.grade, d.span_m * 1000.0, d.load_kN_per_m, d.section_type)
        py = {r[0]: r[2] for r in rows}
        env = _env(o)
        _set(env, d.h, d.b, d.tf, d.tw, d.grade, d.section_type, d.span_m * 1000.0, d.load_kN_per_m)
        dbg = env._ec3_analysis()[-1]
        ws = wb.create_sheet(o)
        ws["A1"] = f"{o}: {d.section_type}, S{int(d.grade)}"
        ws["A1"].font = Font(name=A, bold=True, size=12)
        inputs = [("h", d.h, "mm", ""), ("b", d.b, "mm", ""), ("tf", d.tf, "mm", ""), ("tw", d.tw, "mm", ""),
                  ("fy", float(d.grade), "N/mm2", ""), ("section type", d.section_type, "", "'rolled' or 'welded'"),
                  ("span", float(d.span_m), "m", ""), ("w (factored UDL)", float(d.load_kN_per_m), "kN/m", ""),
                  ("E", 210000.0, "N/mm2", ""), ("G", 81000.0, "N/mm2", ""), ("gamma_M0 = gamma_M1", 1.0, "", ""),
                  ("L_cr / span", 0.40, "", "assumption (restraint spacing), not an EC3 value"),
                  ("C1", 1.13, "", "assumption: UDL, simply supported, k = 1"),
                  ("serviceability load / w", 0.50, "", "assumption of the model, not an EC3 value"),
                  ("deflection limit L /", 250.0, "", "model value"),
                  ("rolled factor A, W_pl", 1.05, "", "assumption (generic fillet factor)"),
                  ("rolled factor I_y", 1.02, "", "assumption"), ("rolled factor I_t", 1.15, "", "assumption"),
                  ("root radius / tf (class check)", 0.10, "", "assumption")]
        ws["A3"], ws["B3"], ws["C3"], ws["D3"] = "Input", "Value", "Unit", "Note"
        for c in "ABCD":
            ws[f"{c}3"].font, ws[f"{c}3"].fill = bold, head
        ref = {}
        for i, (n, v, u, note) in enumerate(inputs, 4):
            ws.cell(i, 1, n).font = norm
            c = ws.cell(i, 2, v)
            c.font = blue
            ws.cell(i, 3, u).font = norm
            ws.cell(i, 4, note).font = norm
            ref[n] = f"$B${i}"
        R = lambda n: ref[n]
        rolled = f'({R("section type")}="rolled")'
        r0 = 4 + len(inputs) + 2
        for c, t in zip("ABCDEFGH", ["Step", "Formula", "Unit", "Excel", "Python", "Difference", "Model (env)", "Reviewer"]):
            ws[f"{c}{r0}"] = t
            ws[f"{c}{r0}"].font, ws[f"{c}{r0}"].fill = bold, head
        steps = []   # (python key, label, formula text, unit, excel formula using names in `s` dict, model key)
        s_ = {}

        def step(key, formula_txt, unit, expr, model_key=None, label=None):
            row = r0 + 1 + len(steps)
            s_[key] = f"$D${row}"
            steps.append((row, key, label or key, formula_txt, unit, expr, model_key))
        S = lambda k: s_[k]
        step("epsilon", "sqrt(235/fy)", "", f'=SQRT(235/{R("fy")})')
        step("web height h_w", "h - 2 tf", "mm", f'={R("h")}-2*{R("tf")}')
        step("root radius used", "0.1 tf (rolled), 0 (welded)", "mm", f'=IF({rolled},{R("root radius / tf (class check)")}*{R("tf")},0)')
        step("flange outstand c_f", "(b - tw)/2 - r", "mm", f'=({R("b")}-{R("tw")})/2-{S("root radius used")}')
        step("web depth c_w", "h_w - 2 r", "mm", f'={S("web height h_w")}-2*{S("root radius used")}')
        step("c_f / (t_f eps)", "Table 5.2: class 1 <= 9, 2 <= 10, 3 <= 14", "", f'={S("flange outstand c_f")}/({R("tf")}*{S("epsilon")})')
        step("c_w / (t_w eps)", "Table 5.2 (bending): class 1 <= 72, 2 <= 83, 3 <= 124", "", f'={S("web depth c_w")}/({R("tw")}*{S("epsilon")})')
        step("section class", "worse of flange and web", "",
             f'=IF(AND({S("c_f / (t_f eps)")}<=9,{S("c_w / (t_w eps)")}<=72),1,IF(AND({S("c_f / (t_f eps)")}<=10,{S("c_w / (t_w eps)")}<=83),2,'
             f'IF(AND({S("c_f / (t_f eps)")}<=14,{S("c_w / (t_w eps)")}<=124),3,4)))', "section_class")
        fA = f'IF({rolled},{R("rolled factor A, W_pl")},1)'
        fI = f'IF({rolled},{R("rolled factor I_y")},1)'
        fT = f'IF({rolled},{R("rolled factor I_t")},1)'
        hw_, h_, b_, tf_, tw_ = S("web height h_w"), R("h"), R("b"), R("tf"), R("tw")
        step("A", "(h_w tw + 2 b tf) x factor", "mm2", f'=({hw_}*{tw_}+2*{b_}*{tf_})*{fA}')
        step("I_y", "[tw h_w^3/12 + 2(b tf^3/12 + b tf (h/2 - tf/2)^2)] x factor", "mm4",
             f'=({tw_}*{hw_}^3/12+2*({b_}*{tf_}^3/12+{b_}*{tf_}*({h_}/2-{tf_}/2)^2))*{fI}')
        step("W_el,y", "I_y / (h/2)", "mm3", f'={S("I_y")}/({h_}/2)')
        step("W_pl,y", "[2 b tf (h/2 - tf/2) + tw h_w^2/4] x factor", "mm3", f'=(2*{b_}*{tf_}*({h_}/2-{tf_}/2)+{tw_}*{hw_}^2/4)*{fA}')
        step("I_z", "2 tf b^3/12 + h_w tw^3/12", "mm4", f'=2*{tf_}*{b_}^3/12+{hw_}*{tw_}^3/12')
        step("M_c,Rd", "W fy / gamma_M0, W = W_pl (class 1-2) or W_el (class 3)", "kNm",
             f'=IF({S("section class")}<=2,{S("W_pl,y")},{S("W_el,y")})*{R("fy")}/{R("gamma_M0 = gamma_M1")}/1000000')
        step("I_t", "(2 b tf^3 + h_w tw^3)/3 x factor", "mm4", f'=(2*{b_}*{tf_}^3+{hw_}*{tw_}^3)/3*{fT}')
        step("I_w", "I_z (h - tf)^2 / 4", "mm6", f'={S("I_z")}*({h_}-{tf_})^2/4')
        step("L_cr", "(L_cr/span) x span", "mm", f'={R("L_cr / span")}*{R("span")}*1000')
        step("C1", "C1", "", f'={R("C1")}')
        step("M_cr", "C1 pi^2 E Iz / L_cr^2 sqrt(Iw/Iz + L_cr^2 G It / (pi^2 E Iz))", "kNm",
             f'={S("C1")}*PI()^2*{R("E")}*{S("I_z")}/{S("L_cr")}^2*SQRT({S("I_w")}/{S("I_z")}+{S("L_cr")}^2*{R("G")}*{S("I_t")}/(PI()^2*{R("E")}*{S("I_z")}))/1000000')
        step("lambda_LT", "sqrt(W fy / M_cr)", "",
             f'=SQRT(IF({S("section class")}<=2,{S("W_pl,y")},{S("W_el,y")})*{R("fy")}/1000000/{S("M_cr")})', "lambda_lt")
        step("alpha_LT", "Table 6.4: rolled h/b<=2 0.21, >2 0.34; welded h/b<=2 0.49, >2 0.76", "",
             f'=IF({rolled},IF({h_}/{b_}>2,0.34,0.21),IF({h_}/{b_}>2,0.76,0.49))')
        step("phi_LT", "0.5 [1 + alpha (lambda - 0.2) + lambda^2]", "", f'=0.5*(1+{S("alpha_LT")}*({S("lambda_LT")}-0.2)+{S("lambda_LT")}^2)')
        step("chi_LT", "min(1, 1/(phi + sqrt(phi^2 - lambda^2)))", "", f'=MIN(1,1/({S("phi_LT")}+SQRT({S("phi_LT")}^2-{S("lambda_LT")}^2)))')
        step("M_b,Rd", "chi_LT M_c,Rd", "kNm", f'={S("chi_LT")}*{S("M_c,Rd")}', "Mrd")
        step("A_v", "A - 2 b tf + (tw + 2r) tf (rolled); h_w tw (welded)", "mm2",
             f'=IF({rolled},{S("A")}-2*{b_}*{tf_}+({tw_}+2*{S("root radius used")})*{tf_},{hw_}*{tw_})')
        step("V_pl,Rd", "A_v fy / (sqrt(3) gamma_M0)", "kN", f'={S("A_v")}*{R("fy")}/SQRT(3)/{R("gamma_M0 = gamma_M1")}/1000', "Vpl_Rd")
        step("V_Ed", "w L / 2", "kN", f'={R("w (factored UDL)")}*{R("span")}/2', "Ved")
        step("M_Ed", "w L^2 / 8", "kNm", f'={R("w (factored UDL)")}*{R("span")}^2/8', "Med")
        step("V_Ed / V_pl,Rd", "interaction reduction only if > 0.5", "", f'={S("V_Ed")}/{S("V_pl,Rd")}', "shear_ratio")
        step("M_Ed / M_b,Rd", "ULS utilisation", "", f'={S("M_Ed")}/{S("M_b,Rd")}', "moment_util")
        step("delta", "5 w_sls L^4 / (384 E I_y)", "mm",
             f'=5*{R("w (factored UDL)")}*{R("serviceability load / w")}*({R("span")}*1000)^4/(384*{R("E")}*{S("I_y")})')
        step("delta / (L/250)", "SLS utilisation", "", f'={S("delta")}/({R("span")}*1000/{R("deflection limit L /")})', "deflection_util")
        step("lambda_w (not in model)", "h_w / (86.4 tw eps)  (EN 1993-1-5 5.3(5))", "", f'={hw_}/(86.4*{tw_}*{S("epsilon")})')
        step("V_Ed / V_bw,Rd (not in model)", "chi_w (Table 5.1, rigid end post) f_y h_w tw / (sqrt(3) gamma_M1)", "",
             f'={S("V_Ed")}/(IF({S("lambda_w (not in model)")}<0.83,1,IF({S("lambda_w (not in model)")}<1.08,0.83/{S("lambda_w (not in model)")},'
             f'1.37/(0.7+{S("lambda_w (not in model)")})))*{R("fy")}*{hw_}*{tw_}/SQRT(3)/1000)')
        for row, key, label, ftxt, unit, expr, mk in steps:
            ws.cell(row, 1, label).font = norm
            ws.cell(row, 2, ftxt).font = norm
            ws.cell(row, 3, unit).font = norm
            ws.cell(row, 4, expr).font = norm
            ws.cell(row, 5, float(py[key])).font = norm
            ws.cell(row, 6, f"=IF(E{row}=0,0,D{row}/E{row}-1)").font = norm
            ws.cell(row, 6).number_format = "0.0000%"
            if mk:
                ws.cell(row, 7, float(dbg[mk])).font = norm
            ws.cell(row, 4).number_format = "General"
        ws.cell(r0 + len(steps) + 2, 1, "Python and Model (env) columns are fixed numbers from hand_calc() and hss_env._ec3_analysis().").font = norm
        for c, w in zip("ABCDEFGH", [30, 62, 8, 14, 14, 12, 14, 14]):
            ws.column_dimensions[c].width = w
        ws.freeze_panes = f"A{r0 + 1}"
    wb.save(path)


def worksheet(gt_dir):
    """Markdown worksheet: the central context (median span, median load) of each objective's reference set."""
    lines = ["# EC3 hand-calculation worksheet", "",
             "Generated by `pipeline/14_ec3_crosscheck.py`. Each design is a best-known reference design of the central context",
             "(median span, median load) of one objective. `Hand` is every step written out from the clauses; `Model` is the value",
             "reported by `hss_env._ec3_analysis()` where it reports one. **`Tool` is empty on purpose: fill it from an independent tool**",
             "(commercial software or a spreadsheet) using the inputs below. Section properties of the model are idealised",
             "(sharp-corner formulas, generic factors for rolled sections), so compare section properties with the same formulas",
             "or use a user-defined section; resistances and utilisations should then agree to rounding, apart from M_cr,",
             "where the tool's C1 and the assumed L_cr = 0.40 L must be set to the same values.", ""]
    for o in OBJECTIVES:
        d = central_design(gt_dir, o)
        rows, res = hand_calc(d.h, d.b, d.tf, d.tw, d.grade, d.span_m * 1000.0, d.load_kN_per_m, d.section_type)
        env = _env(o)
        _set(env, d.h, d.b, d.tf, d.tw, d.grade, d.section_type, d.span_m * 1000.0, d.load_kN_per_m)
        dbg = env._ec3_analysis()[-1]
        lines += [f"## {o}: {d.section_type}, S{int(d.grade)}", "",
                  f"Inputs: h = {d.h:g}, b = {d.b:g}, tf = {d.tf:g}, tw = {d.tw:g} mm; fy = {d.grade:g} N/mm2; span = {d.span_m:g} m; "
                  f"factored UDL w = {d.load_kN_per_m:g} kN/m; E = 210000, G = 81000 N/mm2.", "",
                  "| Step | Formula | Hand | Model | Tool |", "|---|---|---|---|---|"]
        for step, formula, value, unit in rows:
            mk = MODEL_KEYS.get(step)
            m = f"{dbg[mk]:.4g}" if mk else ""
            lines.append(f"| {step} {('(' + unit + ')') if unit else ''} | {formula} | {value:.4g} | {m} | |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ground_truth_dir", default=os.path.join(REPO, "data", "ground_truth", "main_grid_pooled"))
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results"))
    ap.add_argument("--worksheet", default=os.path.join(REPO, "docs", "ec3_worksheet.md"))
    ap.add_argument("--sections", action="store_true", help="also run the FE section check (needs sectionproperties)")
    ap.add_argument("--xlsx", default=None, help="also write the Excel workbook with live formulas to this path")
    ap.add_argument("--detail_csv", nargs="*", default=[], help="detail_<operator>.csv files from 03_evaluate_agent.py --out_detail")
    a = ap.parse_args()
    dirty_at_start = bool(_git("status", "--porcelain", "--untracked-files=no"))  # before any tracked output is rewritten
    os.makedirs(a.results_dir, exist_ok=True)
    outs = {"ec3_reference_audit.csv": audit(a.ground_truth_dir), "ec3_benchmark_ipe500.csv": benchmark_ipe500()}
    if a.sections:
        outs["ec3_sections_fe.csv"] = sections_fe()
    if a.detail_csv:
        outs["ec3_methods_audit.csv"] = pd.DataFrame([audit_designs(f) for f in a.detail_csv])
    for name, df in outs.items():
        df.to_csv(os.path.join(a.results_dir, name), index=False)
        print(f"== {name}\n{df.round(3).to_string(index=False)}\n")
    if a.xlsx:
        write_workbook(a.xlsx, a.ground_truth_dir)
    with open(a.worksheet, "w", encoding="utf-8") as f:
        f.write(worksheet(a.ground_truth_dir))
    rel = lambda v: os.path.relpath(v, REPO) if isinstance(v, str) and os.path.isabs(v) else v
    meta = dict(script="pipeline/14_ec3_crosscheck.py",
                args={k: ([rel(x) for x in v] if isinstance(v, list) else rel(v)) for k, v in vars(a).items()}, files=sorted(outs),
                git_commit=_git("rev-parse", "HEAD"), git_dirty=dirty_at_start,
                numpy=np.__version__, pandas=pd.__version__)
    with open(os.path.join(a.results_dir, "ec3_crosscheck_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
