"""
research/tests/test_ec3_golden_values.py
================================================================
Regression test for the EC3 mechanics and cost/CO2 model in
research/envs/hss_env.py. Pins the output of `_ec3_analysis()` and
`_calculate_cost_co2()` for a small set of representative geometries
(spanning rolled/welded, low/high grade, and both compact and
non-compact section classes) as golden reference values, computed once
and hardcoded below. Any future change to this file's EC3 formulas that
shifts these outputs by more than a tight numerical tolerance will fail
this test -- catching accidental regressions without depending on any
external codebase to compare against.

Run: python research/tests/test_ec3_golden_values.py
================================================================
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.envs.hss_env import HSSBeamEnv


GOLDEN_CASES = [
    dict(h=300, b=150, tf=12, tw=8, fy=355, span=6000, load=25, section_type='rolled',
         expected_util=0.539964, expected_mass=287.234640, expected_chi_lt=0.833313,
         expected_section_class=1, expected_cost=410.745535, expected_co2=737.331321),
    dict(h=450, b=190, tf=18, tw=11, fy=460, span=9000, load=55, section_type='rolled',
         expected_util=0.957998, expected_mass=845.235405, expected_chi_lt=0.617566,
         expected_section_class=1, expected_cost=1354.489737, expected_co2=1987.148437),
    dict(h=600, b=320, tf=22, tw=14, fy=355, span=12000, load=90, section_type='welded',
         expected_util=1.202466, expected_mass=2059.588800, expected_chi_lt=0.736740,
         expected_section_class=1, expected_cost=3501.300960, expected_co2=5950.152043),
    dict(h=700, b=220, tf=28, tw=16, fy=690, span=13000, load=110, section_type='welded',
         expected_util=1.977366, expected_mass=2308.779200, expected_chi_lt=0.293725,
         expected_section_class=1, expected_cost=8147.655246, expected_co2=5330.955473),
    dict(h=700, b=220, tf=20, tw=12, fy=550, span=14000, load=70, section_type='rolled',
         expected_util=2.071762, expected_mass=1929.404400, expected_chi_lt=0.395727,
         expected_section_class=3, expected_cost=3690.950617, expected_co2=4077.603259),
]

TOL = 1e-4


def run():
    env = HSSBeamEnv(reward_mode="lagrangian")
    n_pass = 0
    for i, c in enumerate(GOLDEN_CASES):
        env.h, env.b, env.tf, env.tw, env.fy = c["h"], c["b"], c["tf"], c["tw"], c["fy"]
        env.section_type = c["section_type"]
        env.span, env.load = c["span"], c["load"]
        env.use_storey_load_scaling = False

        util, mass, penalty, class_loss, chi_lt, dbg = env._ec3_analysis()
        cost, co2, _ = env._calculate_cost_co2(mass)

        checks = [
            ("util", util, c["expected_util"]),
            ("mass", mass, c["expected_mass"]),
            ("chi_lt", chi_lt, c["expected_chi_lt"]),
            ("cost", cost, c["expected_cost"]),
            ("co2", co2, c["expected_co2"]),
        ]
        for name, actual, expected in checks:
            rel_diff = abs(actual - expected) / (abs(expected) + 1e-9)
            assert rel_diff < TOL, (
                f"Case {i} ({c['section_type']}, fy={c['fy']}, h={c['h']}, b={c['b']}): "
                f"{name} mismatch. expected={expected}, got={actual}, rel_diff={rel_diff:.2e}"
            )
        assert dbg["section_class"] == c["expected_section_class"], (
            f"Case {i}: section_class mismatch. expected={c['expected_section_class']}, "
            f"got={dbg['section_class']}"
        )
        n_pass += 1
        print(f"  case {i} ({c['section_type']:6s}, S{int(c['fy'])}, h={c['h']}, b={c['b']}): OK "
              f"(util={util:.4f}, mass={mass:.1f}, class={dbg['section_class']})")

    print(f"\nAll {n_pass}/{len(GOLDEN_CASES)} golden-value cases passed (tolerance: {TOL:.0e} relative).")


if __name__ == "__main__":
    run()
