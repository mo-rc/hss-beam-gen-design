| objective | method | gap | feasibility | evals_per_design | labelling_evals | training_steps |
|---|---|---|---|---|---|---|
| cost | PPO (5 seeds) | 8.9 ± 2.4 | 100 | 112 | 0 | 1,000,000 |
| cost | kNN, pooled labels (LOO) | 1.1 | 95 | 63 | 13,536,000 | 0 |
| cost | kNN, DE labels B=1000 (LOO) | 3.2 ± 0.3 | 96 | 70 | 141,000 | 0 |
| cost | kNN, DE labels B=4800 (LOO) | 0.6 ± 0.1 | 95 | 61 | 676,800 | 0 |
| cost | DE search, B=40 | 39.2 ± 1.3 | 95 | 112 | 0 | 0 |
| cost | GA search, B=40 | 43.6 ± 2.9 | 95 | 108 | 0 | 0 |
| cost | random search, B=40 | 34.6 ± 2.0 | 97 | 112 | 0 | 0 |
| mass | PPO (5 seeds) | 7.2 ± 3.2 | 100 | 104 | 0 | 1,000,000 |
| mass | kNN, pooled labels (LOO) | 1.0 | 100 | 42 | 13,536,000 | 0 |
| mass | kNN, DE labels B=1000 (LOO) | 3.2 ± 0.2 | 100 | 42 | 141,000 | 0 |
| mass | kNN, DE labels B=4800 (LOO) | 0.7 ± 0.1 | 100 | 40 | 676,800 | 0 |
| mass | DE search, B=40 | 17.7 ± 1.5 | 95 | 109 | 0 | 0 |
| mass | GA search, B=40 | 17.9 ± 1.3 | 95 | 106 | 0 | 0 |
| mass | random search, B=40 | 15.4 ± 0.7 | 97 | 108 | 0 | 0 |
| co2 | PPO (5 seeds) | 4.6 ± 2.6 | 100 | 100 | 0 | 1,000,000 |
| co2 | kNN, pooled labels (LOO) | 0.5 | 100 | 41 | 13,536,000 | 0 |
| co2 | kNN, DE labels B=1000 (LOO) | 1.6 ± 0.1 | 100 | 42 | 141,000 | 0 |
| co2 | kNN, DE labels B=4800 (LOO) | 0.2 ± 0.0 | 100 | 39 | 676,800 | 0 |
| co2 | DE search, B=40 | 26.5 ± 1.1 | 95 | 108 | 0 | 0 |
| co2 | GA search, B=40 | 25.9 ± 2.1 | 95 | 106 | 0 | 0 |
| co2 | random search, B=40 | 23.0 ± 1.5 | 97 | 108 | 0 | 0 |

Main grid, scale+thin, leave-one-out for the kNN (141 labelled contexts). Pooled labels cost about 96,000 evaluations per context (12 fixed-grade GA searches x 2 restarts x 4,000); labelling cost for DE labels is 141 x B; RL training is 1,000,000 environment steps and needs no labels; search needs neither.
