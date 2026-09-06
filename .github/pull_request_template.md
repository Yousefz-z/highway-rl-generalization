## What changed

<!-- One or two sentences. -->

## Why

<!-- The problem this solves. Link the issue if there is one. -->

## How it was tested

<!--
Required. Say what you actually ran, not what should work.

    ruff check .
    ruff format --check .
    mypy .
    python train.py --algo dqn --seed 99 --timesteps 200
    python evaluate.py --algo dqn --seed 99 --episodes 2 --csv /tmp/smoke.csv
    python evaluate.py --algo random --seed 99 --episodes 2 --csv /tmp/smoke.csv
    python make_figures.py --csv /tmp/smoke.csv --out-dir /tmp/smoke-figures
-->

## Does this change any published number?

- [ ] No, the results in `results/` are unaffected
- [ ] Yes, and this pull request regenerates `results/results.csv` and the
      figures. The numbers that moved:

<!-- If yes, list them: which rows, old value, new value, and why. -->

## Scope

- [ ] This stays inside the scope described in CONTRIBUTING.md
