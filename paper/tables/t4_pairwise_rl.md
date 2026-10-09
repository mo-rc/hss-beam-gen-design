| experiment | arm_a | arm_b | mean_diff_pct_points | ci95_lo_pct_points | ci95_hi_pct_points | p_raw | p_holm |
|---|---|---|---|---|---|---|---|
| 2a reward mode | feasibility_gated | lagrangian | -0.5 | -2.6 | +1.8 | 0.706 | 0.706 |
| 2a reward mode | feasibility_gated | shaped | -6.9 | -12.8 | -1.9 | 0.048 | 0.119 |
| 2a reward mode | lagrangian | shaped | -6.4 | -12.1 | -1.7 | 0.040 | 0.119 |
| 2b algorithm | ppo | sac | -16.0 | -27.5 | -4.1 | 0.095 | 0.381 |
| 2b algorithm | ppo | td3 | -27.2 | -38.2 | -14.7 | 0.008 | 0.048 |
| 2b algorithm | ppo | ddpg | -23.2 | -52.7 | -3.8 | 0.016 | 0.079 |
| 2b algorithm | sac | td3 | -11.2 | -27.8 | +5.3 | 0.214 | 0.643 |
| 2b algorithm | sac | ddpg | -7.2 | -38.3 | +17.3 | 0.714 | 1.000 |
| 2b algorithm | td3 | ddpg | +4.0 | -27.1 | +27.7 | 0.881 | 1.000 |

scale+thin; exact permutation test, Holm-corrected within each experiment; n = 5 seeds per arm.
