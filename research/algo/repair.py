"""
research/algo/repair.py
================================================================
DETERMINISTIC POST-HOC FEASIBILITY / OPTIMALITY REPAIR OPERATORS
(experiment "E0" from the 2026-09-02 bottleneck diagnosis)

WHY THIS EXISTS
---------------
Measured structure of the 142 EC3 ground-truth cost optima:

    * utilisation == 1.000 in 100% of contexts        -> capacity ACTIVE
    * section Class 3 in 124/142 (87%)                -> classification ACTIVE
    * web slenderness d/(tw*eps) within 2% of the
      Class-3 limit of 124 in 57% of contexts         -> slenderness ACTIVE
    * h=750 / b=300 / tw=6 box bounds active in
      24% / 29% / 44% of contexts

i.e. the optimum is a VERTEX of the feasible set, and crossing the Class-3
limit makes `_ec3_analysis()` return util>=2.0 with mass=4000 -- a
discontinuous cliff. A Gaussian stochastic policy must keep its mean
several sigma inside such a boundary, so the trained agents land in the
feasible INTERIOR: measured mean utilisation 0.83-0.88, with flanges and
webs 1.30-1.45x thicker than optimal.

That failure mode is entirely correctable AFTER the fact, because moving
from the interior onto the active boundary is a deterministic scalar
problem -- no learning, no search over the 4-D geometry.

THE OPERATORS
-------------
`uniform_scale` (mode="scale") -- THE E0 OPERATOR
    Multiply (h, b, tf, tw) by a single scalar s and bisect on s until
    utilisation == 1.0 from the feasible side. Two properties make this
    safe rather than merely convenient:
      1. Uniform scaling leaves c/tf and d/tw EXACTLY invariant, so it can
         never change the section class and can never push a Class-3
         section over the cliff into Class 4.
      2. Bisection only ever accepts the feasible bracket end, so the
         returned design is feasible by construction.
    It also repairs INFEASIBLE inputs, by growing s instead of shrinking
    it -- relevant for the shaped/lagrangian/catalog arms.

`scale_and_thin` (mode="scale+thin") -- reported separately, NOT E0
    First push the plate slenderness ratios out toward an EC3 Table 5.2
    class limit (thinning tf and tw, which is the direction the stochastic
    policy structurally refuses to go), then re-apply uniform scaling to
    return to utilisation 1.0. Several class targets are tried because
    Class 3 uses Wel while Classes 1-2 use Wpl, so the cheapest admissible
    class is context-dependent and cannot be assumed.
    This is a local IMPROVEMENT operator, not just a projection. It is
    kept separate from mode="scale" so the E0 claim stays clean.

NON-DEGRADATION GUARANTEE
-------------------------
Every operator returns the ORIGINAL design unless the repaired one is both
feasible and strictly cheaper on the active economy metric. A reported
"repaired" number can therefore never be better than the operator's true
merit, and can never be worse than the unrepaired baseline. `n_ec3` counts
every EC3 analysis the operator consumed, so the cost of the repair is
reported alongside its benefit instead of being hidden.
================================================================
"""

from __future__ import annotations

# ROLLED_LIMITS is imported lazily-safe: manufacturability has no repo deps.

import os

import numpy as np

from research.envs.manufacturability import ROLLED_LIMITS

# Design-variable box from HSSBeamEnv (h, b, tf, tw in mm).
DESIGN_LIMITS = {"h": (250.0, 750.0), "b": (120.0, 300.0),
                 "tf": (8.0, 35.0), "tw": (6.0, 25.0)}

# EC3 Table 5.2 slenderness limits, as implemented in hss_env._ec3_analysis:
#   flange_ratio = c/tf <= {9, 10, 14} * eps for class {1, 2, 3}
#   web_ratio    = d/tw <= {72, 83, 124} * eps for class {1, 2, 3}
CLASS_LIMITS = {1: (9.0, 72.0), 2: (10.0, 83.0), 3: (14.0, 124.0)}

# Sit just inside a limit rather than exactly on it, so floating-point noise
# in the fixed-point geometry solve cannot tip a design into the next class.
_LIMIT_SAFETY = 0.995


def _clip_design(h, b, tf, tw):
    return (float(np.clip(h, *DESIGN_LIMITS["h"])),
            float(np.clip(b, *DESIGN_LIMITS["b"])),
            float(np.clip(tf, *DESIGN_LIMITS["tf"])),
            float(np.clip(tw, *DESIGN_LIMITS["tw"])))


def analyse(env, span_mm, load, h, b, tf, tw, fy, section_type, storey=20):
    """One EC3 analysis of an explicit design at an explicit context.

    Writes the design straight onto the env and calls the env's own
    `_ec3_analysis` / `_calculate_cost_co2`, so the repaired design is
    scored by exactly the same verified physics code that produced the
    numbers being repaired -- no reimplementation, no second source of
    truth. Storey load scaling is disabled because `load` is already the
    post-scaling value the ground-truth CSV represents.
    """
    env.use_storey_load_scaling = False
    env.span, env.load, env.storey = float(span_mm), float(load), int(storey)
    env.h, env.b, env.tf, env.tw = float(h), float(b), float(tf), float(tw)
    env.fy, env.section_type = float(fy), section_type
    util, mass, penalty, _class_loss, chi_lt, dbg = env._ec3_analysis()
    cost, co2, dbg_lca = env._calculate_cost_co2(mass)
    sec_class = int(dbg.get("section_class", 4))
    feasible = (util <= 1.0 + 1e-3) and sec_class <= 3 and penalty <= 1e-9
    return bool(feasible), {
        "h": float(h), "b": float(b), "tf": float(tf), "tw": float(tw),
        "fy": float(fy), "section_type": section_type,
        "utilization": float(util), "mass": float(mass),
        "cost": float(cost), "co2": float(co2), "chi_lt": float(chi_lt),
        "section_class": sec_class, "penalty": float(penalty),
        "feasible": bool(feasible), "ec3": dbg, "lca": dbg_lca,
    }


def _slenderness(h, b, tf, tw, fy, section_type):
    """Current (flange, web) slenderness ratios in units of eps, matching
    hss_env's own definition including the rolled root radius r = 0.1*tf."""
    eps = float(np.sqrt(235.0 / fy))
    r = 0.1 * tf if section_type == "rolled" else 0.0
    c_flange = (b - tw) / 2.0 - r
    d_web = (h - 2.0 * tf) - 2.0 * r
    return c_flange / max(tf, 1e-6) / eps, d_web / max(tw, 1e-6) / eps


def _thin_to_limits(h, b, tf, tw, fy, section_type, lam_f, lam_w):
    """Set tf and tw so slenderness reaches explicit (flange, web) limits."""
    eps = float(np.sqrt(235.0 / fy))
    tf_c, tw_c = tf, tw
    for _ in range(24):
        r = 0.1 * tf_c if section_type == "rolled" else 0.0
        tf_new = ((b - tw_c) / 2.0 - r) / (lam_f * eps)
        tf_new = float(np.clip(min(tf_new, tf), *DESIGN_LIMITS["tf"]))
        d_web = (h - 2.0 * tf_new) - 2.0 * (0.1 * tf_new if section_type == "rolled" else 0.0)
        tw_new = float(np.clip(min(d_web / (lam_w * eps), tw), *DESIGN_LIMITS["tw"]))
        if abs(tf_new - tf_c) < 1e-7 and abs(tw_new - tw_c) < 1e-7:
            tf_c, tw_c = tf_new, tw_new
            break
        tf_c, tw_c = tf_new, tw_new
    return tf_c, tw_c


def _thin_to_class(h, b, tf, tw, fy, section_type, target_class):
    """Set tf and tw so the section sits just inside `target_class`.

    Fixed-point iteration, because for rolled sections the root radius
    r = 0.1*tf makes the flange outstand c depend on tf, and the web depth
    d depend on tf as well. Thicknesses are only ever REDUCED (np.minimum):
    this operator's job is to remove the excess thickness the stochastic
    policy adds as a safety margin, never to add more.
    """
    eps = float(np.sqrt(235.0 / fy))
    lam_f = CLASS_LIMITS[target_class][0] * _LIMIT_SAFETY
    lam_w = CLASS_LIMITS[target_class][1] * _LIMIT_SAFETY
    tf_c, tw_c = tf, tw
    for _ in range(24):
        r = 0.1 * tf_c if section_type == "rolled" else 0.0
        tf_new = ((b - tw_c) / 2.0 - r) / (lam_f * eps)
        tf_new = float(np.clip(min(tf_new, tf), *DESIGN_LIMITS["tf"]))
        d_web = (h - 2.0 * tf_new) - 2.0 * (0.1 * tf_new if section_type == "rolled" else 0.0)
        tw_new = float(np.clip(min(d_web / (lam_w * eps), tw), *DESIGN_LIMITS["tw"]))
        if abs(tf_new - tf_c) < 1e-7 and abs(tw_new - tw_c) < 1e-7:
            tf_c, tw_c = tf_new, tw_new
            break
        tf_c, tw_c = tf_new, tw_new
    return tf_c, tw_c


def uniform_scale(env, span_mm, load, design, metric="cost", storey=20,
                  n_bisect=16, grow_steps=(1.08, 1.18, 1.30, 1.45, 1.65, 1.9, 2.2)):
    """Scale (h, b, tf, tw) by one scalar until utilisation == 1.0.

    Returns (best_info_or_None, n_ec3_evaluations, scale_used).

    Utilisation is monotonically decreasing in the scale factor, so a
    bracket [s_lo (infeasible), s_hi (feasible)] can be bisected. Only
    `s_hi` is ever accepted, which is what makes the output feasible by
    construction. Box clipping can break exact scale-invariance of the
    slenderness ratios near the bounds, so the final design is re-analysed
    and re-checked rather than assumed feasible.
    """
    h0, b0, tf0, tw0 = design["h"], design["b"], design["tf"], design["tw"]
    fy, st = design["fy"], design["section_type"]
    n = 0

    def at(s):
        nonlocal n
        n += 1
        return analyse(env, span_mm, load, *_clip_design(h0 * s, b0 * s, tf0 * s, tw0 * s),
                       fy, st, storey)

    ok, info = at(1.0)
    if ok:
        s_hi, best = 1.0, info
    else:
        # Input was infeasible -> GROW until feasible (repair, not refinement).
        s_hi, best = None, None
        for s in grow_steps:
            ok, info = at(s)
            if ok:
                s_hi, best = s, info
                break
        if s_hi is None:
            return None, n, None
        # A grown design is already minimal-ish within this bracket; still
        # bisect between the last infeasible step and s_hi to tighten it.

    s_lo = 0.25 if s_hi == 1.0 else s_hi / 1.15
    for _ in range(n_bisect):
        s = 0.5 * (s_lo + s_hi)
        ok, info = at(s)
        if ok:
            s_hi, best = s, info
        else:
            s_lo = s
    return best, n, s_hi


def repair(env, span_mm, load, design, metric="cost", storey=20, mode="scale",
           n_bisect=16, class_targets=(3, 2, 1)):
    """Apply a repair operator to one design at one context.

    mode = "none"        : no-op passthrough (baseline column).
    mode = "scale"       : E0 uniform scaling to utilisation 1.0.
    mode = "scale+thin"  : thin plates toward a class limit, then rescale;
                           the cheapest feasible candidate across
                           `class_targets` (plus plain scaling) wins.
    mode = "scale+thin_rolled" : as above, but thinning is capped at the
                           ROLLED-MANUFACTURABILITY envelope instead of
                           the EC3 class limits, so the repaired design
                           remains a section a mill could actually roll.

    Returns a dict with the chosen design's info plus:
      n_ec3      -- EC3 analyses consumed by the operator
      repaired   -- True if the operator's output was accepted
      scale      -- accepted uniform scale factor (None if not applicable)
      variant    -- which candidate won ("original" / "scale" / "thin_cN")
    """
    base_ok, base = analyse(env, span_mm, load, design["h"], design["b"],
                            design["tf"], design["tw"], design["fy"],
                            design["section_type"], storey)
    n_total = 1
    best = dict(base, n_ec3=n_total, repaired=False, scale=None,
                variant="original", feasible_before=base_ok)
    # Only an already-feasible baseline can be "beaten"; an infeasible one
    # must be replaced by anything feasible at all.
    best_cost = base[metric] if base_ok else np.inf

    if mode == "none":
        return best

    cands = [("scale", design)]
    if mode == "scale+thin_rolled":
        # Thin only as far as the ROLLED-MANUFACTURABILITY envelope allows.
        # Thinning to the EC3 Class-3 web limit (124 eps) leaves the rolled
        # envelope (75 eps) and so re-costs the design as a welded plate
        # girder at fab factor 0.42 -- which is why the unconstrained
        # scale+thin operator loses most of its benefit once fabrication
        # cost is computed correctly.
        lam_f = ROLLED_LIMITS["flange_slenderness_max"] * _LIMIT_SAFETY
        lam_w = ROLLED_LIMITS["web_slenderness_max"] * _LIMIT_SAFETY
        tf_t, tw_t = _thin_to_limits(design["h"], design["b"], design["tf"],
                                     design["tw"], design["fy"],
                                     design["section_type"], lam_f, lam_w)
        if tf_t < design["tf"] - 1e-6 or tw_t < design["tw"] - 1e-6:
            cands.append(("thin_rolled", dict(design, tf=tf_t, tw=tw_t)))
    if mode == "scale+thin":
        for tc in class_targets:
            tf_t, tw_t = _thin_to_class(design["h"], design["b"], design["tf"],
                                        design["tw"], design["fy"],
                                        design["section_type"], tc)
            if tf_t < design["tf"] - 1e-6 or tw_t < design["tw"] - 1e-6:
                cands.append((f"thin_c{tc}", dict(design, tf=tf_t, tw=tw_t)))

    for name, cand in cands:
        info, n, s = uniform_scale(env, span_mm, load, cand, metric, storey, n_bisect)
        n_total += n
        if info is not None and info["feasible"] and info[metric] < best_cost - 1e-9:
            best_cost = info[metric]
            best = dict(info, n_ec3=0, repaired=True, scale=s,
                        variant=name, feasible_before=base_ok)
    best["n_ec3"] = n_total
    return best


# ======================================================================
# Catalog-aware repair (discrete arm)
# ======================================================================
#
# Uniform scaling is INVALID for the catalog arm: multiplying (h,b,tf,tw)
# by a real scalar leaves the discrete catalog, so the "repaired" design
# is no longer a section anyone can order. The discrete analogue of
# "shrink until utilisation reaches 1.0" is "step down the size table
# until the section stops working", which is what the two operators below
# do. Every design they return is copied verbatim from a catalog row, so
# membership is preserved by construction (and asserted in
# `verify_membership`).
#
# Efficiency note: at a fixed grade the environment's cost/mass/CO2 model
# is a strictly monotone function of cross-sectional area (verified), and
# it needs no EC3 analysis to evaluate. So candidates can be priced
# analytically, sorted, and then EC3-checked in increasing-cost order.
# The FIRST feasible candidate in that order is therefore provably the
# cheapest feasible one -- an exact result, with early exit rather than a
# full 62-member sweep.

_STEEL_DENSITY_FACTOR = 1.05 * 7850e-9  # kg/mm^3, incl. the env's 5% allowance


def load_catalog(catalog_csv: str | None = None):
    """Return the rolled catalog as a DataFrame with a catalog_index."""
    import pandas as pd
    if catalog_csv and os.path.exists(catalog_csv):
        cat = pd.read_csv(catalog_csv)
    else:
        from research.envs.rolled_catalog import generate_catalog
        cat = generate_catalog()
    if "catalog_index" not in cat.columns:
        cat = cat.reset_index(drop=True)
        cat["catalog_index"] = cat.index
    if "label" not in cat.columns:
        cat["label"] = [f"SEC{int(r.h_mm)}x{int(r.b_mm)}x{int(r.tf_mm)}"
                        for r in cat.itertuples()]
    return cat


def _priced_candidates(env, span_mm, catalog, grades, metric, section_type="rolled"):
    """Price every (catalog member, grade) pair WITHOUT an EC3 analysis.

    Returns a list of (metric_value, catalog_index, fy) sorted ascending.
    """
    prev_fy, prev_st = env.fy, env.section_type
    env.section_type = section_type
    out = []
    h = catalog["h_mm"].to_numpy(); b = catalog["b_mm"].to_numpy()
    tf = catalog["tf_mm"].to_numpy(); tw = catalog["tw_mm"].to_numpy()
    idx = catalog["catalog_index"].to_numpy()
    area = 2.0 * b * tf + (h - 2.0 * tf) * tw
    mass = area * float(span_mm) * _STEEL_DENSITY_FACTOR
    for fy in grades:
        env.fy = float(fy)
        for k in range(len(catalog)):
            if metric == "mass":
                val = float(mass[k])
            else:
                cost, co2, _ = env._calculate_cost_co2(float(mass[k]))
                val = float(cost if metric == "cost" else co2)
            out.append((val, int(idx[k]), float(fy)))
    env.fy, env.section_type = prev_fy, prev_st
    out.sort(key=lambda t: t[0])
    return out


def catalog_repair(env, span_mm, load, design, catalog, metric="cost", storey=20,
                   mode="catalog_snap", grades=None, section_type="rolled"):
    """Repair a catalog-arm design while keeping it a genuine catalog member.

    mode = "none"               : passthrough baseline.
    mode = "catalog_snap"       : cheapest feasible member at the POLICY'S
                                  chosen grade. The policy keeps ownership
                                  of the grade decision; only the section
                                  choice is projected onto feasibility.
    mode = "catalog_snap+grade" : cheapest feasible (member, grade) pair
                                  over the whole catalog. This is not an
                                  operator so much as the EXACT ACHIEVABLE
                                  CEILING of this catalog -- report it as a
                                  reference bound, not as a policy result.

    Same non-degradation contract as `repair`: the original is returned
    unless a candidate is feasible AND strictly better on `metric`.
    """
    base_ok, base = analyse(env, span_mm, load, design["h"], design["b"],
                            design["tf"], design["tw"], design["fy"],
                            section_type, storey)
    n_total = 1
    best = dict(base, n_ec3=n_total, repaired=False, scale=None,
                variant="original", feasible_before=base_ok,
                catalog_index=design.get("catalog_index", -1),
                catalog_label=design.get("catalog_label", "policy"))
    if mode == "none":
        return best
    best_val = base[metric] if base_ok else np.inf

    if grades is None:
        grades = [float(g) for g in env.grades]
    cand_grades = [float(design["fy"])] if mode == "catalog_snap" else grades

    rows = {int(r.catalog_index): r for r in catalog.itertuples()}
    for val, cidx, fy in _priced_candidates(env, span_mm, catalog, cand_grades,
                                            metric, section_type):
        if val >= best_val - 1e-9:
            break  # cost-ordered: nothing cheaper remains, stop early
        r = rows[cidx]
        n_total += 1
        ok, info = analyse(env, span_mm, load, r.h_mm, r.b_mm, r.tf_mm, r.tw_mm,
                           fy, section_type, storey)
        if ok:
            best = dict(info, n_ec3=0, repaired=True, scale=None,
                        variant=mode, feasible_before=base_ok,
                        catalog_index=cidx, catalog_label=str(r.label))
            best_val = info[metric]
            break  # first feasible in cost order == cheapest feasible
    best["n_ec3"] = n_total
    return best


def verify_membership(info, catalog, tol=1e-6):
    """True iff (h,b,tf,tw) exactly matches some catalog row."""
    d = ((catalog["h_mm"] - info["h"]).abs() < tol) & \
        ((catalog["b_mm"] - info["b"]).abs() < tol) & \
        ((catalog["tf_mm"] - info["tf"]).abs() < tol) & \
        ((catalog["tw_mm"] - info["tw"]).abs() < tol)
    return bool(d.any())
