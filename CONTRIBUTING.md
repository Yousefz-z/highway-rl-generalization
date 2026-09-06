# Contributing

This is a small research repository built around one experiment. The most
useful contributions are ones that make the result more trustworthy: a bug in
the metric computation, a confound in the experimental setup, or an
extra evaluation condition.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --group dev          # needs pip 25.1 or newer
```

`pip install --group dev` reads the `dev` group from `pyproject.toml`. With an
older pip, install the same pinned versions by hand:

```bash
pip install ruff==0.16.6 mypy==2.3.1 pandas-stubs==3.0.5.260730
```

## The checks CI runs

These are the exact commands in `.github/workflows/ci.yml`. Run all four
before opening a pull request and you will not get a red build:

```bash
ruff check .
ruff format --check .
mypy .
python train.py --algo dqn --seed 99 --timesteps 200
python evaluate.py --algo dqn --seed 99 --episodes 2 --csv /tmp/smoke.csv
python evaluate.py --algo random --seed 99 --episodes 2 --csv /tmp/smoke.csv
python make_figures.py --csv /tmp/smoke.csv --out-dir /tmp/smoke-figures
```

`ruff format .` (without `--check`) fixes formatting in place.

The smoke run is not a test suite. It exists so that a change which breaks
training, checkpoint loading or the metric computation fails in under a minute
rather than after a few hours of training.

## Scope

The experiment is deliberately narrow: one environment, one independent
variable, default hyperparameters, three seeds. Pull requests that widen it
(more environments, a hyperparameter search, a different simulator, a UI) will
be declined, because the value of the repository is that the claim it makes is
small enough to be checked. Widening the scope without widening the evidence
makes it worse, not better.

Changes that are in scope:

- fixes to how a metric is computed or aggregated
- additional seeds or evaluation episodes at the existing conditions
- corrections to the report, especially to the limitations section
- reproducibility problems: pinned versions that no longer resolve, a script
  that fails on a platform other than Linux

## Pull requests

One pull request per change. Fill in the template, including how you tested
it. If a change affects any number in `results/results.csv`, regenerate the
CSV and the figures in the same pull request and say which numbers moved.
