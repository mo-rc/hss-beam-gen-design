| operator | objective | search | ppo_gap | search_gap | diff_pp | p_raw | separated | ppo_feas | search_feas | p_holm | 95% CI (pp) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | cost | DE (B=40) | 49.4 | 87.9 | -38.5 | 0.008 | True | 100.0 | 83.2 | 0.024 | [-54.3, -20.6] |
| none | cost | GA (B=40) | 49.4 | 104.9 | -55.5 | 0.008 | True | 100.0 | 76.8 | 0.024 | [-70.9, -37.8] |
| none | cost | random (B=40) | 49.4 | 84.8 | -35.5 | 0.008 | True | 100.0 | 83.0 | 0.024 | [-50.7, -17.8] |
| none | mass | DE (B=40) | 48.2 | 54.6 | -6.4 | 0.032 | False | 100.0 | 83.2 | 0.063 | [-11.3, -2.3] |
| none | mass | GA (B=40) | 48.2 | 65.7 | -17.5 | 0.008 | True | 100.0 | 76.8 | 0.024 | [-22.8, -12.6] |
| none | mass | random (B=40) | 48.2 | 54.0 | -5.8 | 0.056 | False | 100.0 | 83.0 | 0.063 | [-11.0, -1.4] |
| none | co2 | DE (B=40) | 47.6 | 68.9 | -21.3 | 0.008 | True | 100.0 | 83.2 | 0.024 | [-27.7, -13.7] |
| none | co2 | GA (B=40) | 47.6 | 81.1 | -33.4 | 0.008 | True | 100.0 | 76.8 | 0.024 | [-40.6, -25.3] |
| none | co2 | random (B=40) | 47.6 | 67.8 | -20.1 | 0.008 | True | 100.0 | 83.0 | 0.024 | [-26.9, -12.5] |
| scale+thin | cost | DE (B=40) | 8.9 | 39.2 | -30.3 | 0.008 | True | 100.0 | 95.1 | 0.024 | [-32.4, -28.1] |
| scale+thin | cost | GA (B=40) | 8.9 | 43.6 | -34.7 | 0.008 | True | 100.0 | 94.8 | 0.024 | [-37.7, -31.7] |
| scale+thin | cost | random (B=40) | 8.9 | 34.6 | -25.7 | 0.008 | True | 100.0 | 97.2 | 0.024 | [-28.1, -23.2] |
| scale+thin | mass | DE (B=40) | 7.2 | 17.7 | -10.5 | 0.008 | True | 100.0 | 95.1 | 0.024 | [-13.0, -7.5] |
| scale+thin | mass | GA (B=40) | 7.2 | 17.9 | -10.7 | 0.008 | True | 100.0 | 94.8 | 0.024 | [-13.2, -7.8] |
| scale+thin | mass | random (B=40) | 7.2 | 15.4 | -8.2 | 0.008 | True | 100.0 | 97.2 | 0.024 | [-10.5, -5.5] |
| scale+thin | co2 | DE (B=40) | 4.6 | 26.5 | -22.0 | 0.008 | True | 100.0 | 95.1 | 0.024 | [-24.0, -19.6] |
| scale+thin | co2 | GA (B=40) | 4.6 | 25.9 | -21.4 | 0.008 | True | 100.0 | 94.8 | 0.024 | [-23.9, -18.6] |
| scale+thin | co2 | random (B=40) | 4.6 | 23.0 | -18.5 | 0.008 | True | 100.0 | 97.2 | 0.024 | [-20.7, -16.0] |

none: PPO 40 vs search 40 evaluations; scale+thin: PPO 100-112 vs search 106-112. 5 seeds vs 5 seeds, exact permutation test, Holm within (objective, operator) over the 3 searches; smallest attainable Holm p = 0.024. Gaps over feasible contexts only; feas = feasibility (%).
