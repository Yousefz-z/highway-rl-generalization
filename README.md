# Training density is not evaluation density

**Does an RL agent trained at a single traffic density generalize to densities
it never saw, and does the answer differ between DQN and PPO?** Short version:
at a 50,000 timestep budget, no, and the metric you use decides which algorithm
looks better.

![Mean episode reward against evaluation traffic density, three seeds per
algorithm](results/reward_vs_density.png)

DQN and PPO are each trained at one traffic density in `highway-fast-v0`, then
evaluated at three, with a random policy as the control. Read on mean reward
alone, PPO both scores higher and degrades less, which looks like better
generalization. It is not: PPO takes the same action on 95 to 96% of steps at
every density, so its flat curve is an open-loop policy that the environment
tolerates until it does not, and its crash rate goes from 0.02 to 0.69 when
density rises 50%.

**[Read the report (PDF)](report/report.pdf)** ([markdown
source](report/report.md)). The limitations section says what this does and
does not establish.

## Results

| Agent | Eval density | Mean reward | Crash rate | Mean speed (m/s) | Most common action |
| --- | --- | --- | --- | --- | --- |
| Random policy | low (0.5x) | 17.89 +/- 0.29 | 0.57 +/- 0.06 | 24.62 +/- 0.21 | LANE_RIGHT (21% of steps) |
| Random policy | medium (1x) | 8.28 +/- 0.47 | 0.97 +/- 0.02 | 23.70 +/- 0.27 | FASTER (21% of steps) |
| Random policy | high (1.5x) | 4.08 +/- 0.33 | 0.99 +/- 0.01 | 23.11 +/- 0.23 | LANE_RIGHT (22% of steps) |
| DQN | low (0.5x) | 21.15 +/- 2.05 | 0.65 +/- 0.26 | 28.57 +/- 1.15 | SLOWER (75% of steps) |
| DQN | medium (1x) | 12.53 +/- 4.99 | 0.75 +/- 0.32 | 25.92 +/- 2.27 | LANE_RIGHT (54% of steps) |
| DQN | high (1.5x) | 5.77 +/- 2.53 | 0.96 +/- 0.06 | 25.24 +/- 1.91 | LANE_RIGHT (54% of steps) |
| PPO | low (0.5x) | 20.88 +/- 0.01 | 0.00 +/- 0.00 | 20.03 +/- 0.01 | LANE_RIGHT (96% of steps) |
| PPO | medium (1x) | 20.81 +/- 0.01 | 0.02 +/- 0.00 | 20.03 +/- 0.02 | LANE_RIGHT (95% of steps) |
| PPO | high (1.5x) | 11.85 +/- 0.07 | 0.69 +/- 0.01 | 19.57 +/- 0.03 | LANE_RIGHT (96% of steps) |

Mean and standard deviation over three training seeds, 100 evaluation episodes
each. `medium` is the training density; `low` and `high` were never seen during
training. Per-seed numbers are in
[`results/results.csv`](results/results.csv).

## Design

- **Independent variable**: `vehicles_density` at 0.5, 1.0, 1.5. Not
  `vehicles_count`: a pilot sweep found that raising it appends vehicles
  further down the road rather than packing them in, so 20, 40 and 80 give
  bit-identical rollouts inside a 30 second episode.
- **Training**: medium density only, DQN and PPO, stable-baselines3 defaults,
  no tuning, identical 50,000 timestep budget, seeds 0/1/2.
- **Evaluation**: all three densities, 100 episodes, deterministic actions, on
  a shared block of environment seeds so every agent meets the same traffic.
- **Control**: a uniform random policy through the identical harness.

## Reproducing it

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train the six agents. Each run is roughly 30 minutes on one CPU core, so about
three hours in total, or half that if you run two at a time:

```bash
for algo in dqn ppo; do
  for seed in 0 1 2; do
    python train.py --algo "$algo" --seed "$seed"
  done
done
```

Evaluate everything, including the random control, and build the figures:

```bash
for seed in 0 1 2; do python evaluate.py --algo random --seed "$seed"; done
for algo in dqn ppo; do
  for seed in 0 1 2; do
    python evaluate.py --algo "$algo" --seed "$seed"
  done
done
python make_figures.py
```

`evaluate.py` appends to `results/results.csv`, so delete that file first if
you are regenerating rather than adding to it. `make_figures.py` writes both
PNGs and `results/summary_table.md`.

Training curves are written to `runs/` in TensorBoard format:

```bash
tensorboard --logdir runs
```

## Layout

```
experiment.py     the independent variable, the budget, the seeds, the env factory
train.py          trains one agent, saves a checkpoint
evaluate.py       runs one agent at every density, appends rows to the results CSV
make_figures.py   reads the CSV, writes the figures and the summary table
results/          the CSV, the two figures, the summary table
models/           the six trained checkpoints (about 1 MB in total, committed)
report/           the report, markdown source and PDF
```

Everything configurable lives in `experiment.py`. Changing a density level or
the training budget there changes it for training, evaluation and figures at
once, so the three scripts cannot disagree about what the experiment is.

## Contributing

Checks, scope and setup are in [CONTRIBUTING.md](CONTRIBUTING.md). The
methodology issue template is for confounds and claims the evidence does not
support, which is the most useful thing to open here.
