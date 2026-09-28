"""Golden-value regression test for the EC3 mechanics and the cost/CO2 model.

Pins `_ec3_analysis()` and `_calculate_cost_co2()` for five representative
sections (rolled/welded, S355-S690, compact and non-compact).

Provenance of the pinned numbers
--------------------------------
* util, mass, chi_lt, section class, co2 (cases 0-3): unchanged from the original
  research-branch goldens (pure EC3 physics; not affected by any later fix).
* cost, cases 1 (S460) and 3 (S690): updated for the 2026-09-26 material-cost-factor
  calibration (S460 1.15 -> 1.30, S690 1.85 -> 1.75; anchored to the supervisor's
  reference data). Case 0/2 (S355, factor 1.00) are unchanged.
* cost and co2, case 4 (S550, 'rolled' input): the section (h=700 mm) lies outside the
  hot-rolled envelope, so the E1 manufacturability check reclassifies it as welded.
  The original golden pre-dated E1. The physics fields for this case are unchanged.
"""
import pytest

from hssbeamgen.envs.hss_env import HSSBeamEnv

# (h, b, tf, tw, fy, span_mm, load_kN_per_m, input_type, expected_effective_type, expected values)
CASES = [
    dict(h=300, b=150, tf=12, tw=8, fy=355, span=6000, load=25, section_type="rolled", eff="rolled",
         util=0.539964, mass=287.234640, chi_lt=0.833313, cls=1, cost=410.745535, co2=737.331321),
    dict(h=450, b=190, tf=18, tw=11, fy=460, span=9000, load=55, section_type="rolled", eff="rolled",
         util=0.957998, mass=845.235405, chi_lt=0.617566, cls=1, cost=1500.292844, co2=1987.148437),
    dict(h=600, b=320, tf=22, tw=14, fy=355, span=12000, load=90, section_type="welded", eff="welded",
         util=1.202466, mass=2059.588800, chi_lt=0.736740, cls=1, cost=3501.300960, co2=5950.152043),
    dict(h=700, b=220, tf=28, tw=16, fy=690, span=13000, load=110, section_type="welded", eff="welded",
         util=1.977366, mass=2308.779200, chi_lt=0.293725, cls=1, cost=7742.185134, co2=5330.955473),
    dict(h=700, b=220, tf=20, tw=12, fy=550, span=14000, load=70, section_type="rolled", eff="welded",
         util=2.071762, mass=1929.404400, chi_lt=0.395727, cls=3, cost=5084.391171, co2=4859.935068),
]
TOL = 1e-4


@pytest.fixture(scope="module")
def env():
    return HSSBeamEnv(reward_mode="lagrangian")


@pytest.mark.parametrize("c", CASES, ids=lambda c: f"S{c['fy']}-{c['section_type']}-h{c['h']}")
def test_golden_case(env, c):
    env.h, env.b, env.tf, env.tw, env.fy = c["h"], c["b"], c["tf"], c["tw"], c["fy"]
    env.section_type = c["section_type"]
    env.span, env.load = c["span"], c["load"]
    env.use_storey_load_scaling = False

    util, mass, _pen, _cl, chi_lt, dbg = env._ec3_analysis()
    cost, co2, cdbg = env._calculate_cost_co2(mass)

    for name, actual, expected in [("util", util, c["util"]), ("mass", mass, c["mass"]),
                                   ("chi_lt", chi_lt, c["chi_lt"]), ("cost", cost, c["cost"]),
                                   ("co2", co2, c["co2"])]:
        assert abs(actual - expected) / (abs(expected) + 1e-9) < TOL, f"{name}: {actual} vs {expected}"
    assert dbg["section_class"] == c["cls"]
    assert cdbg["effective_section_type"] == c["eff"]


def test_material_cost_factors_are_calibrated_values():
    """Cost factors must equal the calibrated table (anchored: 355=1.00, 460=1.30, 690=1.75)."""
    import inspect
    src = inspect.getsource(HSSBeamEnv._calculate_cost_co2)
    assert "{355: 1.00, 460: 1.30, 500: 1.38, 550: 1.48, 620: 1.61, 690: 1.75}" in src
