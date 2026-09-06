"""Shared experiment constants and the environment factory.

Every script in this repository reads its configuration from here so that the
independent variable, the training budget and the seed set are defined exactly
once. Changing a value here changes it for training, evaluation and figures at
the same time.
"""

from __future__ import annotations

from typing import Final

import gymnasium as gym
import highway_env  # noqa: F401  (registers the highway-env environment ids)
import numpy as np

ENV_ID: Final = "highway-fast-v0"

# The independent variable: the `vehicles_density` config of highway-fast-v0,
# which scales the spacing between vehicles. "medium" is the stock environment,
# and the other two levels sit symmetrically around it at plus and minus 50%.
#
# `vehicles_count` is deliberately left at its default of 20. A pilot sweep
# showed that raising it does not make traffic denser: the extra vehicles are
# appended further down the road (20 vehicles reach about 475m ahead, 40 reach
# about 948m) and are never met inside a 30 second episode, so counts of 20, 40
# and 80 produce bit-identical rollouts. `vehicles_density` is the knob that
# actually changes how much traffic the agent encounters.
DENSITIES: Final[dict[str, float]] = {"low": 0.5, "medium": 1.0, "high": 1.5}

# Agents are trained at this density only, then evaluated at all three.
TRAIN_DENSITY: Final = "medium"

# Fixed for every run of both algorithms. Chosen once, from the measured
# throughput of the environment (roughly 25 env steps per second on one CPU
# core), so that six runs finish in a few hours. Do not change it: an unequal
# budget makes the DQN/PPO comparison meaningless.
TOTAL_TIMESTEPS: Final = 50_000

SEEDS: Final = (0, 1, 2)
ALGOS: Final = ("dqn", "ppo")

# Episodes per evaluation point.
EVAL_EPISODES: Final = 100

RESULTS_CSV: Final = "results/results.csv"
CSV_COLUMNS: Final = (
    "algo",
    "seed",
    "train_density",
    "eval_density",
    "mean_reward",
    "crash_rate",
    "mean_speed",
)


def make_env(density: str, seed: int | None = None) -> gym.Env[np.ndarray, int]:
    """Build a highway-fast-v0 environment at the named traffic density.

    Raises:
        KeyError: if `density` is not one of the three defined levels.
    """
    if density not in DENSITIES:
        raise KeyError(
            f"unknown density {density!r}; expected one of {sorted(DENSITIES)}"
        )
    env = gym.make(ENV_ID, config={"vehicles_density": DENSITIES[density]})
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def n_actions() -> int:
    """Size of the discrete action set, read from the environment itself.

    Raises:
        TypeError: if the environment is not configured for discrete actions.
    """
    space = make_env(TRAIN_DENSITY).action_space
    if not isinstance(space, gym.spaces.Discrete):
        raise TypeError(f"expected a Discrete action space, got {type(space).__name__}")
    return int(space.n)
