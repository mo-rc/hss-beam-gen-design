"""
research/envs/rolled_catalog.py
================================================================
Generates a representative catalog of rolled I-section (UB/UC-style)
geometries, used by HSSBeamCatalogEnv (research/envs/hss_catalog_env.py)
to constrain the "geometry" part of the action space to sections that
could plausibly be rolled, rather than any continuous (h,b,tf,tw).

IMPORTANT, SAME CAVEAT AS THE EARLIER AISC HSS CATALOG SCAFFOLD:
This catalog is PROCEDURALLY GENERATED from typical UB/UC proportion
rules (depth-to-flange-width ratio, flange-thickness-to-depth ratio,
web-thickness-to-flange-thickness ratio), calibrated to fall within the
h/b/tf/tw ranges already used throughout this project's continuous-action
environment (h: 250-750mm, b: 120-300mm, tf: 8-35mm, tw: 6-25mm), so the
two arms (continuous vs catalog-constrained) explore comparable design
envelopes. It is NOT a copy of a real manufacturer table.

FOR PUBLICATION: replace `generate_catalog()`'s output with an actual
section table (e.g. Tata Steel "Blue Book" UB/UC properties, or the
equivalent from your target market's steel supplier / EN 10365 standard
sizes) before reporting the catalog-constrained arm's results as
representative of real fabrication practice. The rest of this codebase
(HSSBeamCatalogEnv, training, evaluation) does not care where the CSV
comes from, only that it has the same columns (h_mm, b_mm, tf_mm, tw_mm,
label), so this is a drop-in data swap, not a code change.
================================================================
"""

import numpy as np
import pandas as pd


# Typical UB proportions: b/h roughly 0.35-0.45 for smaller sections,
# narrowing toward 0.30 for deeper sections (UB family); UC family is
# closer to square (b/h ~ 0.9-1.0). We generate both families.
def generate_catalog(h_min=250, h_max=750, h_step=25) -> pd.DataFrame:
    rows = []
    depths = np.arange(h_min, h_max + 1, h_step)

    for h in depths:
        # --- UB-style (beam): narrower flange, deeper section ---
        for b_ratio in (0.34, 0.40, 0.46):
            b = round(h * b_ratio / 2) * 2
            tf = round(np.clip(h * 0.028 + 3.0, 8, 35))
            tw = round(np.clip(tf * 0.62, 6, 25))
            if 120 <= b <= 300 and 8 <= tf <= 35 and 6 <= tw <= 25:
                rows.append(dict(h_mm=float(h), b_mm=float(b), tf_mm=float(tf), tw_mm=float(tw),
                                  family="UB", label=f"UB{int(h)}x{int(b)}x{int(tf)}"))

        # --- UC-style (column): near-square, thicker flanges ---
        for b_ratio in (0.85, 0.95, 1.02):
            b = round(h * b_ratio / 2) * 2
            tf = round(np.clip(h * 0.045 + 4.0, 8, 35))
            tw = round(np.clip(tf * 0.65, 6, 25))
            if 120 <= b <= 300 and 8 <= tf <= 35 and 6 <= tw <= 25:
                rows.append(dict(h_mm=float(h), b_mm=float(b), tf_mm=float(tf), tw_mm=float(tw),
                                  family="UC", label=f"UC{int(h)}x{int(b)}x{int(tf)}"))

    df = pd.DataFrame(rows).drop_duplicates(subset=["h_mm", "b_mm", "tf_mm", "tw_mm"])
    # Sort by an approximate mass-per-metre proxy (area) so "catalog index +-1"
    # in the environment corresponds to "next size up/down", matching how a
    # real designer navigates a size table.
    df["area_proxy"] = df.h_mm * df.tw_mm + 2 * df.b_mm * df.tf_mm
    df = df.sort_values("area_proxy").reset_index(drop=True)
    df["catalog_index"] = df.index
    return df


if __name__ == "__main__":
    cat = generate_catalog()
    cat.to_csv("rolled_catalog.csv", index=False)
    print(f"Generated {len(cat)} rolled sections ({(cat.family=='UB').sum()} UB, "
          f"{(cat.family=='UC').sum()} UC) -> rolled_catalog.csv")
    print(cat.head(8).to_string(index=False))
