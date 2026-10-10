# EC3 model: verification and scope

Script: `pipeline/14_ec3_crosscheck.py` (seconds, no training). Data: `results/ec3_reference_audit.csv`,
`results/ec3_benchmark_ipe500.csv`, `results/ec3_sections_fe.csv`. Hand-calculation worksheet with an empty column for an
independent tool: `docs/ec3_worksheet.md`. Tests: `tests/test_ec3_crosscheck.py`.

## Scope of the member check (`hss_env._ec3_analysis`)
| Check | Clause | Status |
|---|---|---|
| Section class (bending; web, outstand flange) | EN 1993-1-1 Table 5.2 | implemented; class 4 is penalised, not designed |
| Moment resistance, `W_pl` (class 1-2), `W_el` (class 3) | 6.2.5 | implemented, gamma_M0 = 1.0 |
| Lateral-torsional buckling, general method | 6.3.2.2, curves of Table 6.4, alpha_LT of Table 6.3 | implemented; `M_cr` closed form, C1 = 1.13 |
| Shear resistance, plastic | 6.2.6 | implemented (eta = 1.0, conservative) |
| Moment-shear interaction | 6.2.8 | simplified expression; applies only above V_Ed / V_pl,Rd = 0.5 (one reference design, at 0.502) |
| Deflection, `5 w L^4 / (384 E I_y)` against L/250 | serviceability | implemented; load = 0.5 x factored UDL |
| Shear buckling of the web | 6.2.6(6), EN 1993-1-5 5 | **not checked** (see below) |
| Patch loading, web crippling, flange-induced web buckling, torsion, moment gradient between restraints | | not checked |

## Assumptions that are not EC3 values
Effective LTB length `L_cr = 0.40 L` (restraint spacing); C1 = 1.13; serviceability load = 0.5 x the factored UDL with limit
L/250; generic fillet factors for "rolled" sections (area 1.05, I_y 1.02, W_pl 1.05, I_t 1.15) and a root radius of 0.1 t_f in
the class check; no weld allowance in the outstand width of welded sections (conservative); nominal f_y (the largest flange
thickness in any reference design is 35 mm, below the 40 mm limit of Table 3.1); cost and CO2 factors are calibrated model
values.

## Checks that do not use the model's own code
1. **Published benchmark** (IPE 500, S235, L = 3.75 m, general method, curve b; ECCS TC8 No. 119 example 5, as reproduced in
   SOFiSTiK benchmark DCE-EN24). With the published section constants the closed-form `M_cr,0` is 895.3 kNm (published 895);
   the chain lambda_LT 0.695, phi_LT 0.825, chi_LT 0.787 is reproduced to within 0.06%. The model's own chi_LT code equals the
   published formula at the same lambda. On the model's idealised IPE 500 section `M_cr,0` is 888.9 kNm (-0.7%), `W_pl` +0.9%,
   `I_w` +1.3%, `I_t` -7.6% (conservative).
2. **Finite-element section properties** (`sectionproperties`, true root radii, seven IPE / HEA / HEB sections; for the IPE 500 the FE area and second moment of
   area, 115.5 cm2 and 48212 cm4, match the published 115.5 cm2 and 48197 cm4). Ratio of FE value to the sharp-corner formula: A 1.027-1.059,
   I_y 1.028-1.057, W_pl 1.029-1.060, I_t 1.14-1.40, I_z 1.00; the model uses 1.05, 1.02, 1.05, 1.15. The model's `W_pl` is
   up to about 2% above the FE value (HEB 500) and 1% below it (HEA 300); I_y and I_t are on the conservative side.
3. **Reference designs against the omitted shear-buckling check** (EN 1993-1-5 5.3, end stiffeners only, eta = 1.0,
   gamma_M1 = 1.0). The web slenderness h_w / (t_w eps) exceeds 72 in 85 (cost), 141 (mass) and 141 (CO2) of 142 reference
   designs, up to 125, so the check would be required. The largest V_Ed / V_bw,Rd is 0.60, 0.80 and 0.68 (rigid end post) and
   0.60, 0.89, 0.76 (non-rigid); no reference design fails. A design that satisfies an added constraint stays best known for
   the stricter problem, so the reference set is unaffected.

## Not done
- The same check on the designs of the policy, kNN and searches (they are not stored; only aggregates are). Re-evaluate the
  headline checkpoints with `03_evaluate_agent.py --out_detail`, then the audit can be applied to those designs.
- A value from an independent tool for the three worksheet designs (column `Tool`).
- Section-property factors for rolled sections were checked on seven European sections only.

## Audit trail
- `tests/test_ec3_independent.py` restates the model's formulas (same fillet factors, same interaction expression); it checks
  that the code matches that restatement, not the standard. The benchmark above is the check against an external source.
- The comment in `hss_env.py` and the docstring of that test cite Table 6.5 for the LTB curve selection of the general method;
  the curves used (a, b for rolled, c, d for welded, by h/b) are those of Table 6.4. The code is left unchanged because the
  evaluation metas record a SHA-256 of `hss_env.py`.
