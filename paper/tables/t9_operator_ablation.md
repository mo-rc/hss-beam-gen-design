| objective | method | operator | gap | feasibility | evals |
|---|---|---|---|---|---|
| cost | PPO | none | 49.4 ± 20.8 | 100 | 40 |
| cost | PPO | scale | 20.8 ± 9.8 | 100 | 57 |
| cost | PPO | scale+thin | 8.9 ± 2.4 | 100 | 112 |
| cost | kNN (pooled labels, LOO) | none | 5.4 | 44 | 1 |
| cost | kNN (pooled labels, LOO) | scale | 1.3 | 95 | 18 |
| cost | kNN (pooled labels, LOO) | scale+thin | 1.1 | 95 | 63 |
| cost | DE search, B=40 | none | 87.9 ± 5.4 | 83 | 40 |
| cost | DE search, B=40 | scale | 70.0 ± 2.3 | 95 | 58 |
| cost | DE search, B=40 | scale+thin | 39.2 ± 1.3 | 95 | 112 |
| cost | GA search, B=40 | none | 104.9 ± 3.9 | 77 | 36 |
| cost | GA search, B=40 | scale | 78.9 ± 3.7 | 95 | 54 |
| cost | GA search, B=40 | scale+thin | 43.6 ± 2.9 | 95 | 108 |
| cost | random search, B=40 | none | 84.8 ± 2.0 | 83 | 40 |
| cost | random search, B=40 | scale | 61.8 ± 3.2 | 97 | 58 |
| cost | random search, B=40 | scale+thin | 34.6 ± 2.0 | 97 | 112 |
| mass | PPO | none | 48.2 ± 5.7 | 100 | 40 |
| mass | PPO | scale | 24.3 ± 4.1 | 100 | 57 |
| mass | PPO | scale+thin | 7.2 ± 3.2 | 100 | 104 |
| mass | kNN (pooled labels, LOO) | none | 6.2 | 39 | 1 |
| mass | kNN (pooled labels, LOO) | scale | 1.3 | 100 | 19 |
| mass | kNN (pooled labels, LOO) | scale+thin | 1.0 | 100 | 42 |
| mass | DE search, B=40 | none | 54.6 ± 1.3 | 83 | 40 |
| mass | DE search, B=40 | scale | 41.4 ± 1.2 | 95 | 58 |
| mass | DE search, B=40 | scale+thin | 17.7 ± 1.5 | 95 | 109 |
| mass | GA search, B=40 | none | 65.7 ± 3.3 | 77 | 36 |
| mass | GA search, B=40 | scale | 45.4 ± 0.8 | 95 | 54 |
| mass | GA search, B=40 | scale+thin | 17.9 ± 1.3 | 95 | 106 |
| mass | random search, B=40 | none | 54.0 ± 2.4 | 83 | 40 |
| mass | random search, B=40 | scale | 36.1 ± 1.0 | 97 | 58 |
| mass | random search, B=40 | scale+thin | 15.4 ± 0.7 | 97 | 108 |
| co2 | PPO | none | 47.6 ± 8.5 | 100 | 40 |
| co2 | PPO | scale | 23.1 ± 5.6 | 100 | 57 |
| co2 | PPO | scale+thin | 4.6 ± 2.6 | 100 | 100 |
| co2 | kNN (pooled labels, LOO) | none | 5.8 | 42 | 1 |
| co2 | kNN (pooled labels, LOO) | scale | 0.7 | 100 | 19 |
| co2 | kNN (pooled labels, LOO) | scale+thin | 0.5 | 100 | 41 |
| co2 | DE search, B=40 | none | 68.9 ± 2.6 | 83 | 40 |
| co2 | DE search, B=40 | scale | 53.0 ± 2.0 | 95 | 58 |
| co2 | DE search, B=40 | scale+thin | 26.5 ± 1.1 | 95 | 108 |
| co2 | GA search, B=40 | none | 81.1 ± 4.6 | 77 | 36 |
| co2 | GA search, B=40 | scale | 56.8 ± 1.7 | 95 | 54 |
| co2 | GA search, B=40 | scale+thin | 25.9 ± 2.1 | 95 | 106 |
| co2 | random search, B=40 | none | 67.8 ± 3.6 | 83 | 40 |
| co2 | random search, B=40 | scale | 46.8 ± 1.8 | 97 | 58 |
| co2 | random search, B=40 | scale+thin | 23.0 ± 1.5 | 97 | 108 |

Main grid. Gaps are over FEASIBLE contexts only; read them together with the feasibility column.
