"""Turn the results CSV into the two report figures and the summary table.

Usage:
    python make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import experiment  # noqa: E402

# Colourblind-safe, and distinguishable when printed in greyscale.
STYLES: dict[str, tuple[str, str, str]] = {
    "random": ("#767676", "o", "Random policy"),
    "dqn": ("#0072B2", "s", "DQN"),
    "ppo": ("#D55E00", "^", "PPO"),
}
PLOT_ORDER = ("random", "dqn", "ppo")
DENSITY_ORDER = tuple(experiment.DENSITIES)


def load_results(csv_path: Path) -> pd.DataFrame:
    """Read the results CSV.

    Raises:
        FileNotFoundError: if no results have been produced yet.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"no results at {csv_path}; run evaluate.py for each agent first"
        )
    frame = pd.read_csv(csv_path)
    frame["eval_density"] = pd.Categorical(
        frame["eval_density"], categories=DENSITY_ORDER, ordered=True
    )
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation across seeds, per algorithm and density."""
    grouped = frame.groupby(["algo", "eval_density"], observed=True)
    summary: pd.DataFrame = grouped.agg(
        mean_reward_mean=("mean_reward", "mean"),
        mean_reward_std=("mean_reward", "std"),
        mean_reward_count=("mean_reward", "count"),
        crash_rate_mean=("crash_rate", "mean"),
        crash_rate_std=("crash_rate", "std"),
        mean_speed_mean=("mean_speed", "mean"),
        mean_speed_std=("mean_speed", "std"),
    )
    flat: pd.DataFrame = summary.reset_index()
    return flat


def _density_label(name: str) -> str:
    return f"{name}\n({experiment.DENSITIES[name]:g}x)"


def plot_metric(
    summary: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path
) -> None:
    """One line per algorithm across evaluation densities, with seed error bars."""
    fig, axes = plt.subplots(figsize=(6.0, 4.0))
    x = range(len(DENSITY_ORDER))

    for algo in PLOT_ORDER:
        rows = summary[summary["algo"] == algo].set_index("eval_density")
        if rows.empty:
            continue
        rows = rows.reindex(list(DENSITY_ORDER))
        colour, marker, label = STYLES[algo]
        axes.errorbar(
            x,
            rows[f"{metric}_mean"],
            yerr=rows[f"{metric}_std"],
            color=colour,
            marker=marker,
            capsize=4,
            linewidth=1.8,
            label=label,
        )

    train_index = DENSITY_ORDER.index(experiment.TRAIN_DENSITY)
    axes.axvline(train_index, color="black", linestyle=":", linewidth=1.0, zorder=0)
    axes.annotate(
        "trained here",
        xy=(train_index, axes.get_ylim()[1]),
        xytext=(0, -12),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="black",
    )

    axes.set_xticks(list(x))
    axes.set_xticklabels([_density_label(d) for d in DENSITY_ORDER])
    axes.set_xlabel("Evaluation traffic density (vehicles_density)")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.legend(frameon=False)
    axes.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary_table(summary: pd.DataFrame, out_path: Path) -> None:
    """Write the mean plus or minus standard deviation table as markdown."""
    header = (
        "| Agent | Eval density | Mean reward | Crash rate "
        "| Mean speed (m/s) | Seeds |",
        "| --- | --- | --- | --- | --- | --- |",
    )
    lines = list(header)
    for algo in PLOT_ORDER:
        rows = summary[summary["algo"] == algo].set_index("eval_density")
        if rows.empty:
            continue
        rows = rows.reindex(list(DENSITY_ORDER))
        for density in DENSITY_ORDER:
            row = rows.loc[density]
            lines.append(
                f"| {STYLES[algo][2]} | {density} ({experiment.DENSITIES[density]:g}x) "
                f"| {row['mean_reward_mean']:.2f} +/- {row['mean_reward_std']:.2f} "
                f"| {row['crash_rate_mean']:.2f} +/- {row['crash_rate_std']:.2f} "
                f"| {row['mean_speed_mean']:.2f} +/- {row['mean_speed_std']:.2f} "
                f"| {int(row['mean_reward_count'])} |"
            )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(experiment.RESULTS_CSV))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True)
    summary = aggregate(load_results(args.csv))

    plot_metric(
        summary,
        "mean_reward",
        "Mean episode reward",
        f"Reward vs traffic density (trained at {experiment.TRAIN_DENSITY})",
        args.out_dir / "reward_vs_density.png",
    )
    plot_metric(
        summary,
        "crash_rate",
        "Crash rate",
        f"Crash rate vs traffic density (trained at {experiment.TRAIN_DENSITY})",
        args.out_dir / "crash_rate_vs_density.png",
    )
    write_summary_table(summary, args.out_dir / "summary_table.md")
    print(f"wrote two figures and summary_table.md to {args.out_dir}")


if __name__ == "__main__":
    main()
