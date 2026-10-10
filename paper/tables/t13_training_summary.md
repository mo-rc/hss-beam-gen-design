| experiment | arm | n_seeds | reward_mid | reward_end | ep_len_mid | ep_len_end | n_seeds_end_worse_2x |
|---|---|---|---|---|---|---|---|
| 2a_reward_mode | feasibility_gated | 5 | -58.3 | -44.1 | 38.3 | 37.9 | 0 |
| 2a_reward_mode | lagrangian | 5 | -136.3 | -129.9 | 38.8 | 38.9 | 0 |
| 2a_reward_mode | shaped | 5 | -1332.5 | -269.5 | 38.1 | 38.9 | 0 |
| 2b_algorithm | ddpg | 5 | -62.1 | -64.6 | 33.9 | 36.3 | 0 |
| 2b_algorithm | sac | 5 | -44.1 | -76.4 | 9.6 | 12.2 | 1 |
| 2b_algorithm | td3 | 5 | -59.7 | -228.4 | 16.4 | 20.2 | 3 |
| 2c_objective | co2 | 5 | -56.6 | -40.2 | 38.5 | 39.0 | 0 |
| 2c_objective | mass | 5 | -53.2 | -42.1 | 38.5 | 38.9 | 0 |

Training-episode reward and episode length (mean over seeds) in the step windows 0.4-0.6M and 0.9-1.0M, and the number of seeds whose end reward is at least 2x more negative than the mid reward. Rewards are sums over the episode: they are not comparable between algorithms with different episode lengths, nor between reward modes.
