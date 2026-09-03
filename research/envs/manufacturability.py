"""
research/envs/manufacturability.py
================================================================
Resolves the rolled-versus-welded costing inconsistency in the
fabrication model.

THE PROBLEM
-----------
`HSSBeamEnv._calculate_cost_co2` selects the fabrication cost factor
purely from the `section_type` label (rolled -> 0.15, welded -> 0.42),
and `section_type` is a FREE decision variable for the GA, the random
search and the policy. Nothing couples that label to whether the
geometry could actually be hot-rolled. The optimiser therefore learned
to claim the cheap rolled fabrication factor while using plate-girder
proportions, which no rolling mill produces.

Measured on the existing cost ground truth: all 142 contexts are
labelled "rolled", yet their median web slenderness is 123.2*eps and
their median web thickness is 6.3 mm.

THE EVIDENCE
------------
1) GEOMETRY. Across the 132 hot-rolled universal beam sections in the
   British Steel UB datasheet (h = 127-1026 mm), evaluated with this
   project's own slenderness definition (root radius r = 0.1*tf):

       d/tw  : 19.1 - 73.3 eps   (median 50.0)
       c/tf  :  3.3 - 10.3 eps   (median  6.6)
       tw/tf : 0.53 - 0.84       (median 0.61)
       b/h   : 0.30 - 0.66       (median 0.41)

   ZERO of the 132 real sections exceed the EC3 Class-3 web limit of
   124*eps, and only 2 exceed the Class-1 limit of 72*eps. A web at
   123*eps is simply not a rolled section.
   Source: British Steel, "Universal beams" datasheet.
   https://www.britishsteel.co.uk/wp-content/uploads/2026/02/british-steel-universal-beams-datasheet-190724.pdf

2) GRADE AVAILABILITY. EN 10025-6 ("Technical delivery conditions for
   FLAT PRODUCTS of high yield strength structural steels in the
   quenched and tempered condition") is the standard that covers
   S460-S960. It specifies flat products (plate, 3-200 mm) only; it
   does not cover hot-rolled sections. Hot-rolled I-sections to
   EN 10365 are supplied in the EN 10025-2/-3/-4 grades, i.e. up to
   S460. Grades S500 and above are therefore a plate product, and a
   beam in those grades must be a welded plate girder.
   Source: EN 10025-6:2019 scope.
   https://standards.iteh.ai/catalog/standards/cen/77ea2e36-0020-4c39-afef-337cb89046ca/en-10025-6-2019

THE FIX
-------
`section_type` stays a free decision variable, but it no longer decides
the cost on its own. A design may only be COSTED as rolled if it is
inside the rolled-manufacturability envelope above. Anything outside it
is costed as a welded plate girder, which is what it physically is.
This is a costing/labelling correction only -- no EC3 mechanics, no
capacity equations and no constraint definitions are touched.

Thresholds are the observed real-section extremes, rounded outward so
the envelope is permissive rather than tight (a design rejected as
rolled is one that no real section comes close to):

       d/tw  <=  75 eps        (observed max 73.3)
       c/tf  <=  11 eps        (observed max 10.3)
       tw/tf >=  0.50          (observed min 0.53)
       b/h   <=  1.05          (covers the near-square UC family)
       b/h   >=  0.25          (observed min 0.30)
       fy    <=  460 MPa       (EN 10025-6 is plate-only above S460)

`ROLLED_LIMITS` is a single dict so the sensitivity of any result to
these thresholds can be tested by editing one place.
================================================================
"""

from __future__ import annotations

import numpy as np

ROLLED_LIMITS = {
    "web_slenderness_max": 75.0,   # d/tw in units of eps
    "flange_slenderness_max": 11.0,  # c/tf in units of eps
    "tw_over_tf_min": 0.50,
    "b_over_h_max": 1.05,
    "b_over_h_min": 0.25,
    "fy_max": 460.0,               # MPa; EN 10025-6 is a flat-product standard
}


def rolled_violations(h, b, tf, tw, fy, limits=None):
    """Return the list of rolled-manufacturability limits this design breaks.

    Slenderness is computed with exactly the same definition as
    `HSSBeamEnv._ec3_analysis` (rolled root radius r = 0.1*tf), so the
    envelope is expressed in the same units as the EC3 class checks.
    """
    L = ROLLED_LIMITS if limits is None else limits
    eps = float(np.sqrt(235.0 / fy))
    r = 0.1 * tf
    c_flange = (b - tw) / 2.0 - r
    d_web = (h - 2.0 * tf) - 2.0 * r
    lam_f = c_flange / max(tf, 1e-9) / eps
    lam_w = d_web / max(tw, 1e-9) / eps

    v = []
    if fy > L["fy_max"] + 1e-9:
        v.append("grade_plate_only")
    if lam_w > L["web_slenderness_max"]:
        v.append("web_too_slender")
    if lam_f > L["flange_slenderness_max"]:
        v.append("flange_too_slender")
    if tw / max(tf, 1e-9) < L["tw_over_tf_min"]:
        v.append("web_too_thin_vs_flange")
    if not (L["b_over_h_min"] <= b / max(h, 1e-9) <= L["b_over_h_max"]):
        v.append("aspect_ratio")
    return v


def is_rolled_manufacturable(h, b, tf, tw, fy, limits=None) -> bool:
    """True iff this geometry/grade could plausibly be a hot-rolled section."""
    return not rolled_violations(h, b, tf, tw, fy, limits)


def effective_section_type(section_type, h, b, tf, tw, fy, limits=None) -> str:
    """The section type the design must be COSTED as.

    "welded" stays "welded". "rolled" is honoured only if the design is
    inside the rolled-manufacturability envelope; otherwise it is costed
    as the welded plate girder it actually is.
    """
    if section_type != "rolled":
        return section_type
    return "rolled" if is_rolled_manufacturable(h, b, tf, tw, fy, limits) else "welded"
