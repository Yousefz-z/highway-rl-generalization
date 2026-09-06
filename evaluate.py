"""Evaluate one agent at every traffic density and append rows to the results CSV.

Usage:
    python evaluate.py --algo random --seed 0
    python evaluate.py --algo dqn --seed 0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Protocol

import numpy as np
from stable_baselines3 import DQN, PPO

import experiment

# Evaluation episodes use a fixed seed block, shared by every agent, so that all
# agents face the same traffic at a given density and the comparison is paired.
EVAL_SEED_BASE = 10_000

# The random policy has no training density; the column has to say so.
NO_TRAIN_DENSITY = "none"


class Policy(Protocol):
    """Anything that maps an observation to an action index."""

    def act(self, obs: np.ndarray) -> int: ...


class RandomPolicy:
    """Uniform random action, the control every trained agent is compared to."""

    def __init__(self, n_actions: int, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_actions = n_actions

    def act(self, obs: np.ndarray) -> int:
        return int(self._rng.integers(self._n_actions))


class SB3Policy:
    """Deterministic greedy action from a trained stable-baselines3 agent."""

    def __init__(self, model: DQN | PPO) -> None:
        self._model = model

    def act(self, obs: np.ndarray) -> int:
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


def load_policy(algo: str, seed: int, n_actions: int) -> Policy:
    """Build the policy for this algorithm and seed.

    Raises:
        ValueError: if `algo` is not "random" or a supported algorithm name.
        FileNotFoundError: if a trained checkpoint is missing.
    """
    if algo == "random":
        return RandomPolicy(n_actions, seed)
    if algo not in experiment.ALGOS:
        raise ValueError(
            f"unknown algo {algo!r}; expected 'random' or one of {experiment.ALGOS}"
        )
    path = Path("models") / f"{algo}_seed{seed}"
    if not path.with_suffix(".zip").exists():
        raise FileNotFoundError(
            f"no checkpoint at {path}.zip; run "
            f"'python train.py --algo {algo} --seed {seed}' first"
        )
    loader = DQN if algo == "dqn" else PPO
    return SB3Policy(loader.load(path))


def run_episodes(policy: Policy, density: str, episodes: int) -> dict[str, float]:
    """Roll out `episodes` episodes at one density and return the three metrics."""
    env = experiment.make_env(density)
    rewards: list[float] = []
    crashes: list[bool] = []
    speeds: list[float] = []

    for i in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        total = 0.0
        episode_speeds: list[float] = []
        crashed = False
        while True:
            obs, reward, terminated, truncated, info = env.step(policy.act(obs))
            total += float(reward)
            episode_speeds.append(float(info["speed"]))
            crashed = bool(info["crashed"])
            if terminated or truncated:
                break
        rewards.append(total)
        crashes.append(crashed)
        speeds.append(float(np.mean(episode_speeds)))

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "crash_rate": float(np.mean(crashes)),
        "mean_speed": float(np.mean(speeds)),
    }


def append_rows(rows: list[dict[str, object]], csv_path: Path) -> None:
    """Append result rows, writing the header if the file is new."""
    csv_path.parent.mkdir(exist_ok=True)
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(experiment.CSV_COLUMNS))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def evaluate(algo: str, seed: int, episodes: int, csv_path: Path) -> None:
    """Evaluate one agent at all three densities and append the rows."""
    n_actions = experiment.n_actions()
    train_density = NO_TRAIN_DENSITY if algo == "random" else experiment.TRAIN_DENSITY
    rows: list[dict[str, object]] = []
    for density in experiment.DENSITIES:
        policy = load_policy(algo, seed, n_actions)
        metrics = run_episodes(policy, density, episodes)
        row: dict[str, object] = {
            "algo": algo,
            "seed": seed,
            "train_density": train_density,
            "eval_density": density,
            **metrics,
        }
        rows.append(row)
        print(
            f"{algo} seed{seed} @ {density}: "
            f"reward {metrics['mean_reward']:.2f}, "
            f"crash {metrics['crash_rate']:.2f}, "
            f"speed {metrics['mean_speed']:.2f}"
        )
    append_rows(rows, csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=("random", *experiment.ALGOS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--episodes", type=int, default=experiment.EVAL_EPISODES)
    parser.add_argument("--csv", type=Path, default=Path(experiment.RESULTS_CSV))
    args = parser.parse_args()

    evaluate(args.algo, args.seed, args.episodes, args.csv)


if __name__ == "__main__":
    main()
