"""
research/scripts/expA7_handcalc_comparison.py
================================================================
Direct comparison against the supervisor-provided hand-calculated EC3
reference designs (span = 9 m, bay width = 6 m, factored UDL = 56.77
kN/m, M_Ed = 575 kNm, V_Ed = 256 kN), mirroring Table 11 of the
reference RC-beam paper (design comparison against a human designer).

Reuses evaluate_with_repair.py's rollout_design() and
research.algo.repair.repair() unchanged -- no new environment or policy
code, this only wires them at a single forced context instead of a full
grid, and prints the comparison against the four hand-calculated
reference designs (S355 / S460 / S550 / S690).

USAGE
------
    python research/scripts/expA7_handcalc_comparison.py \\
        --models research/models/gated_cost_merged_seed4{2,3,4,5,6}/final_model \\
        --economy_metric cost
================================================================
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.envs.hss_env import HSSBeamEnv
from research.algo.repair import repair
from research.scripts.evaluate_with_repair import rollout_design

SPAN_M = 9.0
LOAD_KNM = 56.77  # factored UDL, kN/m -- matches env's `load` convention exactly
STOREY = 20

# Hand-calculated reference designs (span=9m, same load case; see supervisor doc)
HAND_CALC = {
    355: dict(name="UKB 457x191x74 (rolled)", mass_per_m=74.0, section_type="rolled",
              h=457.0, b=190.4, tf=14.5, tw=9.0),
    460: dict(name="UKB 457x152x67 (rolled)", mass_per_m=67.0, section_type="rolled",
              h=458.0, b=153.8, tf=15.0, tw=9.0),
    550: dict(name="welded bf=162,hw=374,tf=15,tw=8", bf=162, hw=374, tf=15, tw=8,
              section_type="welded"),
    690: dict(name="welded bf=152,hw=334,tf=15,tw=8", bf=152, hw=334, tf=15, tw=8,
              section_type="welded"),
}


def hand_calc_costs(env, metric):
    rows = []
    for fy, r in HAND_CALC.items():
        if "mass_per_m" in r:
            mass = r["mass_per_m"] * SPAN_M
            h, b, tf, tw = r["h"], r["b"], r["tf"], r["tw"]
        else:
            mass = (2 * (r["bf"] * r["tf"]) + r["hw"] * r["tw"]) * 1e-6 * 7850.0 * SPAN_M
            h, b, tf, tw = r["hw"] + 2 * r["tf"], r["bf"], r["tf"], r["tw"]
        # _calculate_cost_co2 reads self.section_type/h/b/tf/tw internally (rolled vs
        # welded fabrication factor, and the >=S550 welded thickness penalty) -- these
        # MUST be set explicitly per reference design, or it silently reuses whatever
        # geometry/type was last left on the env, which is wrong for any row where the
        # true fabrication class differs from that leftover state.
        env.fy = float(fy)
        env.section_type = r["section_type"]
        env.h, env.b, env.tf, env.tw = h, b, tf, tw
        cost, co2, _ = env._calculate_cost_co2(mass)
        rows.append(dict(source="hand_calc", grade=fy, name=r["name"], mass=round(mass, 1),
                          **{metric: round(cost if metric == "cost" else (co2 if metric == "co2" else mass), 1)}))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", required=True)
    ap.add_argument("--economy_metric", default="cost")
    ap.add_argument("--modes", nargs="+", default=["none", "scale", "scale+thin"])
    ap.add_argument("--out_csv", default="research/results/expA7_handcalc_comparison.csv")
    a = ap.parse_args()

    from stable_baselines3 import PPO
    env = HSSBeamEnv(reward_mode="feasibility_gated", economy_metric=a.economy_metric)

    rows = hand_calc_costs(env, a.economy_metric)

    for mp in a.models:
        name = os.path.basename(os.path.dirname(mp))
        model = PPO.load(mp)
        policy_fn = lambda obs: model.predict(obs, deterministic=True)[0]
        design, was_feasible, n_gen = rollout_design(env, policy_fn, SPAN_M, LOAD_KNM,
                                                      storey=STOREY, max_steps=40, seed=0)
        for mode in a.modes:
            out = repair(env, SPAN_M * 1000.0, LOAD_KNM, design, metric=a.economy_metric,
                          storey=STOREY, mode=mode)
            rows.append(dict(
                source=f"ppo_{name}", grade=int(out["fy"]), name=f"mode={mode}, n_ec3={40 + out['n_ec3']}",
                mass=round(out["mass"], 1), feasible=out["feasible"],
                **{a.economy_metric: round(out[a.economy_metric], 1)},
            ))

    df = pd.DataFrame(rows)
    df.to_csv(a.out_csv, index=False)
    print(f"\n=== Table 11-style comparison: span={SPAN_M}m, UDL={LOAD_KNM}kN/m "
          f"(M_Ed=575 kNm, matches supervisor hand-calc exactly) ===")
    print(df.to_string(index=False))
    hc = df[df.source == "hand_calc"]
    best_hc = hc.loc[hc[a.economy_metric].idxmin()]
    print(f"\nBest hand-calculated design: grade {best_hc.grade} ({best_hc['name']}), "
          f"{a.economy_metric}={best_hc[a.economy_metric]}")
    print(f"wrote {a.out_csv}")


if __name__ == "__main__":
    main()
