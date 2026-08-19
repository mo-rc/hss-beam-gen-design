"""
inspect_pretrain_dataset.py — quick sanity checks on the generated
EC3-optimal dataset before using it for anything downstream.

USAGE:
    python inspect_pretrain_dataset.py --csv ./pretrain_data/ec3_optimal_designs.csv
"""
import argparse
import csv
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="../pretrain_data/ec3_optimal_designs.csv")
args = parser.parse_args()

rows = []
with open(args.csv) as f:
    for r in csv.DictReader(f):
        for k in ["span_m","load_kNm","grade","h","b","tf","tw","util","mass","cost","co2"]:
            r[k] = float(r[k])
        rows.append(r)

print(f"Total rows: {len(rows)}")

# ── Governance-aware monotonicity check ─────────────────────────────
# Deflection depends only on E and Iy — identical across ALL grades.
# When a section is deflection-governed, higher fy gives ZERO benefit,
# so mass should plateau (not decrease) across grades. Checking raw
# monotonicity everywhere conflates this correct physics with real
# inversions. Split by the *lower* grade's governing regime in each
# adjacent pair — if EITHER grade in the pair is deflection-governed,
# treat mass staying flat (within a tolerance) as CORRECT, not an
# inversion.
groups = defaultdict(list)
for r in rows:
    key = (round(r["span_m"],1), round(r["load_kNm"],1), r["section_type"])
    groups[key].append(r)

results = {"MOMENT": {"ok": 0, "inverted": 0},
           "DEFL":   {"ok": 0, "inverted": 0}}

MASS_TOL = 0.02   # 2% — treat as "flat" rather than a real inversion

for key, grp in groups.items():
    grp_sorted = sorted(grp, key=lambda x: x["grade"])
    if len(grp_sorted) < 2:
        continue
    for i in range(len(grp_sorted) - 1):
        lo, hi = grp_sorted[i], grp_sorted[i+1]
        regime = lo.get("governing", "MOMENT")   # regime of the lower grade
        mass_lo, mass_hi = lo["mass"], hi["mass"]

        if regime == "DEFL":
            # Expect roughly flat, not decreasing — only flag if hi is
            # MEANINGFULLY heavier than lo (a real problem), not if it's
            # merely not-lighter (which is physically correct here).
            if mass_hi > mass_lo * (1 + MASS_TOL):
                results["DEFL"]["inverted"] += 1
            else:
                results["DEFL"]["ok"] += 1
        else:
            # Moment-governed: genuinely expect mass to decrease with grade.
            if mass_hi > mass_lo + 1e-6:
                results["MOMENT"]["inverted"] += 1
            else:
                results["MOMENT"]["ok"] += 1

for regime, r in results.items():
    total = r["ok"] + r["inverted"]
    if total == 0:
        continue
    print(f"{regime:<8} pairs: {r['ok']}/{total} correctly ordered "
          f"({r['inverted']} true inversions, "
          f"{r['inverted']/total:.1%})")

print()

# Highest-demand row available
top = max(rows, key=lambda r: r["span_m"] * r["load_kNm"])
print(f"\nHighest-demand feasible row found:")
print(f"  span={top['span_m']:.1f}m load={top['load_kNm']:.0f}kN/m grade=S{top['grade']:.0f} "
      f"type={top['section_type']}")
print(f"  h={top['h']:.0f} b={top['b']:.0f} tf={top['tf']:.1f} tw={top['tw']:.1f} "
      f"util={top['util']:.3f} mass={top['mass']:.0f}kg")

# S690 rows near max span/load
s690_high = [r for r in rows if r["grade"]==690 and r["span_m"]>=12.0 and r["load_kNm"]>=100.0]
print(f"\nS690 rows at span>=12m, load>=100kN/m: {len(s690_high)} found")
for r in sorted(s690_high, key=lambda x: (x["span_m"], x["load_kNm"]))[:5]:
    print(f"  span={r['span_m']:.1f} load={r['load_kNm']:.0f} h={r['h']:.0f} "
          f"b={r['b']:.0f} mass={r['mass']:.0f}kg util={r['util']:.3f}")
