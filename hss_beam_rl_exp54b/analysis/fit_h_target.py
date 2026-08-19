"""
fit_h_target.py

Fits a grade-independent, physics-informed h_target(span, load) formula to replace
the crude `h_target = span_m * 42` heuristic currently used in the environment's
h_target_norm / geometry_gap observation features (see PROJECT_SUMMARY_HSS_RL.md,
Section 5).

Approach
--------
1. Load the EC3-optimal design dataset produced by generate_ec3_pretrain_dataset.py
   (./pretrain_data/ec3_optimal_designs.csv), which contains one row per feasible
   (span, load, grade, section_type) combination that survived the min-mass sweep
   (util in [0.90, 1.02], section_class <= 2), tagged with a `governing` field
   (MOMENT / DEFL).
2. Collapse to "best-overall-row-per-context": for each (span, load) pair, aggregate
   across ALL grades and section types and keep the min-mass feasible row. This
   deliberately sidesteps both physics confounds documented in Section 5:
     - deflection-governed mass is grade-independent (higher fy gives no benefit),
     - epsilon = sqrt(235/fy) makes Class 1/2 limits stricter at higher grade,
       which can force higher-grade sections to be *heavier*, not lighter.
   h_target only needs to signal "what scale of section this demand calls for",
   not which grade wins, so aggregating over grade/type is the correct move.
3. Fit a quadratic regression:
       h_target = c0 + c1*span + c2*load + c3*span^2 + c4*span*load + c5*load^2
   via ordinary least squares on that aggregated subset.
4. Report R^2, mean/max residuals (both overall and split by `governing` regime),
   and a side-by-side comparison against the crude `span_m * 42` baseline.

Column-name assumptions (EDIT THE CONFIG BLOCK BELOW IF YOUR CSV DIFFERS)
---------------------------------------------------------------------
This script was written from the project summary alone -- the actual dataset
wasn't available when it was drafted. Column names are auto-detected against a
list of common aliases; if detection fails for a column, the script prints the
actual CSV columns and exits with a clear error so the CONFIG block can be fixed
in one place rather than hunting through the code.
"""

import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG -- edit these if your CSV uses different column names.
# Each entry is a list of acceptable aliases, checked in order.
# ---------------------------------------------------------------------------
COLUMN_ALIASES = {
    "span":       ["span", "span_m", "span_m_", "L", "L_m"],
    "load":       ["load", "load_kNm", "w", "w_kNm", "udl", "udl_kNm"],
    "grade":      ["grade", "steel_grade", "fy_grade"],
    "section_type": ["section_type", "sec_type", "type"],
    "h":          ["h", "h_mm", "depth", "depth_mm"],
    "mass":       ["mass", "mass_kg", "mass_per_m", "unit_mass"],
    "utilization": ["utilization", "util", "UC", "uc"],
    "section_class": ["section_class", "class", "sec_class"],
    "governing":  ["governing", "governance", "govern"],
    "feasible":   ["feasible", "is_feasible"],
}

CSV_PATH_DEFAULT = "../pretrain_data/ec3_optimal_designs.csv"
CRUDE_BASELINE_COEF = 42.0  # h_target = span_m * 42


def resolve_columns(df: pd.DataFrame) -> dict:
    resolved = {}
    missing = []
    for canonical, aliases in COLUMN_ALIASES.items():
        found = next((a for a in aliases if a in df.columns), None)
        if found is not None:
            resolved[canonical] = found
        else:
            missing.append(canonical)

    # governing/feasible/section_type/grade are optional for the core fit --
    # only span/load/h/mass are strictly required.
    hard_required = ["span", "load", "h", "mass"]
    hard_missing = [c for c in hard_required if c not in resolved]
    if hard_missing:
        print(f"ERROR: could not find required columns {hard_missing} in CSV.")
        print(f"Actual CSV columns: {list(df.columns)}")
        print("Edit COLUMN_ALIASES in fit_h_target.py to match your dataset, then re-run.")
        sys.exit(1)

    if missing:
        print(f"NOTE: optional columns not found (will skip related checks): {missing}")

    return resolved


def build_design_matrix(span: np.ndarray, load: np.ndarray) -> np.ndarray:
    """Quadratic feature matrix: [1, span, load, span^2, span*load, load^2]."""
    return np.column_stack([
        np.ones_like(span),
        span,
        load,
        span ** 2,
        span * load,
        load ** 2,
    ])


def fit_quadratic(span: np.ndarray, load: np.ndarray, h_target: np.ndarray):
    X = build_design_matrix(span, load)
    coefs, residuals_ss, rank, sv = np.linalg.lstsq(X, h_target, rcond=None)
    preds = X @ coefs
    resid = h_target - preds
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((h_target - h_target.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coefs, preds, resid, r2


def main(csv_path: str):
    print(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, columns: {list(df.columns)}")

    cols = resolve_columns(df)

    # If a 'feasible' flag exists, filter to feasible rows only.
    if "feasible" in cols:
        before = len(df)
        df = df[df[cols["feasible"]].astype(bool)]
        print(f"  Filtered to feasible rows: {before} -> {len(df)}")

    span_col, load_col, h_col, mass_col = cols["span"], cols["load"], cols["h"], cols["mass"]

    # --- Step 2: aggregate to best-overall-row-per-context (min mass across grade/type) ---
    group_cols = [span_col, load_col]
    idx_min_mass = df.groupby(group_cols)[mass_col].idxmin()
    best = df.loc[idx_min_mass].reset_index(drop=True)
    print(f"\nAggregated to best-overall-row-per-context: {len(best)} unique (span, load) contexts")

    if "grade" in cols:
        print("  Winning grade distribution at these contexts:")
        print(best[cols["grade"]].value_counts().to_string())

    span = best[span_col].to_numpy(dtype=float)
    load = best[load_col].to_numpy(dtype=float)
    h_target = best[h_col].to_numpy(dtype=float)

    # --- Step 3: fit quadratic regression ---
    coefs, preds, resid, r2 = fit_quadratic(span, load, h_target)
    c0, c1, c2, c3, c4, c5 = coefs

    print("\n=== Fitted quadratic h_target(span, load) ===")
    print(f"h_target = {c0:.4f} + {c1:.4f}*span + {c2:.4f}*load "
          f"+ {c3:.6f}*span^2 + {c4:.6f}*span*load + {c5:.6f}*load^2")
    print(f"R^2 = {r2:.4f}")
    print(f"Mean |residual| = {np.mean(np.abs(resid)):.2f}")
    print(f"Max |residual|  = {np.max(np.abs(resid)):.2f}")
    print(f"RMSE            = {np.sqrt(np.mean(resid ** 2)):.2f}")

    # --- Step 4: split by governing regime, if available ---
    if "governing" in cols:
        gov_col = cols["governing"]
        best["_resid"] = resid
        print("\n--- Residuals by governing regime ---")
        for regime, sub in best.groupby(gov_col):
            r = sub["_resid"].to_numpy()
            print(f"  {regime}: n={len(sub)}, mean|resid|={np.mean(np.abs(r)):.2f}, "
                  f"max|resid|={np.max(np.abs(r)):.2f}")

    # --- Step 5: compare against crude baseline h_target = span_m * 42 ---
    # NOTE: assumes `span` column is already in meters. If your dataset stores
    # span in mm, divide by 1000 here before comparing.
    crude_preds = span * CRUDE_BASELINE_COEF
    crude_resid = h_target - crude_preds
    crude_ss_res = np.sum(crude_resid ** 2)
    ss_tot = np.sum((h_target - h_target.mean()) ** 2)
    crude_r2 = 1 - crude_ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n=== Baseline comparison: h_target = span_m * 42 ===")
    print(f"Crude R^2 = {crude_r2:.4f}")
    print(f"Crude mean |residual| = {np.mean(np.abs(crude_resid)):.2f}")
    print(f"Crude max |residual|  = {np.max(np.abs(crude_resid)):.2f}")
    print(f"Crude RMSE            = {np.sqrt(np.mean(crude_resid ** 2)):.2f}")

    print("\n=== Summary ===")
    print(f"{'Model':<20}{'R^2':>10}{'Mean|resid|':>15}{'Max|resid|':>13}{'RMSE':>10}")
    print(f"{'Quadratic fit':<20}{r2:>10.4f}{np.mean(np.abs(resid)):>15.2f}"
          f"{np.max(np.abs(resid)):>13.2f}{np.sqrt(np.mean(resid**2)):>10.2f}")
    print(f"{'Crude span*42':<20}{crude_r2:>10.4f}{np.mean(np.abs(crude_resid)):>15.2f}"
          f"{np.max(np.abs(crude_resid)):>13.2f}{np.sqrt(np.mean(crude_resid**2)):>10.2f}")

    # --- Save fitted coefficients for direct copy-paste into the env file ---
    out_path = "../pretrain_data/h_target_fit_coefs.txt"
    try:
        with open(out_path, "w") as f:
            f.write("# Fitted h_target(span, load) quadratic coefficients\n")
            f.write(f"# h_target = c0 + c1*span + c2*load + c3*span^2 + c4*span*load + c5*load^2\n")
            f.write(f"c0 = {c0!r}\n")
            f.write(f"c1 = {c1!r}\n")
            f.write(f"c2 = {c2!r}\n")
            f.write(f"c3 = {c3!r}\n")
            f.write(f"c4 = {c4!r}\n")
            f.write(f"c5 = {c5!r}\n")
            f.write(f"# R^2 = {r2:.4f}, RMSE = {np.sqrt(np.mean(resid**2)):.2f}\n")
        print(f"\nCoefficients written to {out_path}")
    except OSError as e:
        print(f"\n(Could not write coefficients file: {e})")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH_DEFAULT
    main(path)
