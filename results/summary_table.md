| Agent | Eval density | Mean reward | Crash rate | Mean speed (m/s) | Most common action | Seeds |
| --- | --- | --- | --- | --- | --- | --- |
| Random policy | low (0.5x) | 17.89 +/- 0.29 | 0.57 +/- 0.06 | 24.62 +/- 0.21 | LANE_RIGHT (21% of steps) | 3 |
| Random policy | medium (1x) | 8.28 +/- 0.47 | 0.97 +/- 0.02 | 23.70 +/- 0.27 | FASTER (21% of steps) | 3 |
| Random policy | high (1.5x) | 4.08 +/- 0.33 | 0.99 +/- 0.01 | 23.11 +/- 0.23 | LANE_RIGHT (22% of steps) | 3 |
| DQN | low (0.5x) | 21.15 +/- 2.05 | 0.65 +/- 0.26 | 28.57 +/- 1.15 | SLOWER (75% of steps) | 3 |
| DQN | medium (1x) | 12.53 +/- 4.99 | 0.75 +/- 0.32 | 25.92 +/- 2.27 | LANE_RIGHT (54% of steps) | 3 |
| DQN | high (1.5x) | 5.77 +/- 2.53 | 0.96 +/- 0.06 | 25.24 +/- 1.91 | LANE_RIGHT (54% of steps) | 3 |
| PPO | low (0.5x) | 20.88 +/- 0.01 | 0.00 +/- 0.00 | 20.03 +/- 0.01 | LANE_RIGHT (96% of steps) | 3 |
| PPO | medium (1x) | 20.81 +/- 0.01 | 0.02 +/- 0.00 | 20.03 +/- 0.02 | LANE_RIGHT (95% of steps) | 3 |
| PPO | high (1.5x) | 11.85 +/- 0.07 | 0.69 +/- 0.01 | 19.57 +/- 0.03 | LANE_RIGHT (96% of steps) | 3 |
