"""Train one agent at the training density and save its checkpoint.

Usage:
    python train.py --algo dqn --seed 0
    python train.py --algo ppo --seed 0 --timesteps 2000   # short smoke run
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

import experiment

MODELS_DIR = Path("models")
TENSORBOARD_DIR = Path("runs")


def build_agent(algo: str, seed: int) -> BaseAlgorithm:
    """Create an SB3 agent with default hyperparameters at the training density.

    Raises:
        ValueError: if `algo` is not a supported algorithm name.
    """
    env = experiment.make_env(experiment.TRAIN_DENSITY, seed=seed)
    log = str(TENSORBOARD_DIR)
    if algo == "dqn":
        return DQN("MlpPolicy", env, seed=seed, verbose=1, tensorboard_log=log)
    if algo == "ppo":
        return PPO("MlpPolicy", env, seed=seed, verbose=1, tensorboard_log=log)
    raise ValueError(f"unknown algo {algo!r}; expected one of {experiment.ALGOS}")


def checkpoint_path(algo: str, seed: int) -> Path:
    """Where the trained agent for this algorithm and seed is stored."""
    return MODELS_DIR / f"{algo}_seed{seed}"


def train(algo: str, seed: int, timesteps: int) -> Path:
    """Train one agent end to end and save it. Returns the checkpoint path."""
    MODELS_DIR.mkdir(exist_ok=True)
    agent = build_agent(algo, seed)
    agent.learn(
        total_timesteps=timesteps,
        tb_log_name=f"{algo}_seed{seed}",
        progress_bar=False,
    )
    path = checkpoint_path(algo, seed)
    agent.save(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", required=True, choices=experiment.ALGOS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=experiment.TOTAL_TIMESTEPS,
        help="override the fixed budget, for smoke runs only",
    )
    args = parser.parse_args()

    path = train(args.algo, args.seed, args.timesteps)
    print(f"saved {path}.zip after {args.timesteps} timesteps")


if __name__ == "__main__":
    main()
