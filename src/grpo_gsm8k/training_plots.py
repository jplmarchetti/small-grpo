"""Training curves of one run, read from its train_log.csv (and val_curve.csv if present).

- Per-step values over 16 completions are noisy, so each panel shows the raw line faintly
  and a moving average on top.
- The reward panel also shows the cumulative mean reward: the average of every reward
  received so far, a smooth "is it improving?" line.

`moving_average` is adapted from rasbt/reasoning-from-scratch (ch07.py), Apache-2.0,
Copyright Sebastian Raschka.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from grpo_gsm8k.plot_style import C, apply_house_style, save_fig  # noqa: E402
from grpo_gsm8k.run_log import read_step_logs  # noqa: E402


def moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """Mean of the last `window` values at each step (shorter at the start)."""
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    starts = np.maximum(np.arange(1, len(values) + 1) - window, 0)
    ends = np.arange(1, len(values) + 1)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def cumulative_mean(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.cumsum(values) / np.arange(1, len(values) + 1)


def column(rows: Sequence[dict], name: str) -> np.ndarray:
    return np.array([float(row[name]) for row in rows])


def read_val_curve(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def plot_training_curves(run_dir: Path, window: int = 20, title: str | None = None) -> Path:
    """Three panels: reward, graded rates (lenient / format, + validation if any), completion length."""
    apply_house_style()
    logs = read_step_logs(run_dir / "train_log.csv")
    steps = column(logs, "step")
    validation = read_val_curve(run_dir / "val_curve.csv")

    fig, (reward_ax, rates_ax, tokens_ax) = plt.subplots(3, 1, figsize=(9.5, 10), sharex=True)

    reward = column(logs, "mean_reward")
    reward_ax.plot(steps, reward, color=C["blue"], alpha=0.25, lw=1, label="per step")
    reward_ax.plot(steps, moving_average(reward, window), color=C["blue"], lw=2.4, label=f"moving average ({window})")
    reward_ax.plot(steps, cumulative_mean(reward), color=C["orange"], lw=2.4, ls="--", label="cumulative mean")
    reward_ax.set_ylabel("mean reward")
    reward_ax.legend(loc="best")

    for name, color, label in (("lenient_rate", C["green"], "lenient (sampled)"), ("format_rate", C["purple"], "format (sampled)")):
        values = column(logs, name)
        rates_ax.plot(steps, values, color=color, alpha=0.2, lw=1)
        rates_ax.plot(steps, moving_average(values, window), color=color, lw=2.4, label=label)
    if validation:
        val_steps = column(validation, "step")
        rates_ax.plot(val_steps, column(validation, "lenient_accuracy"), "o-", color=C["green"], ms=7, lw=1.5,
                      mfc="white", label="lenient (validation, greedy)")
    rates_ax.set_ylim(0, 1.02)
    rates_ax.set_ylabel("share of completions")
    rates_ax.legend(loc="best")

    tokens = column(logs, "mean_tokens")
    tokens_ax.plot(steps, tokens, color=C["grey"], alpha=0.3, lw=1)
    tokens_ax.plot(steps, moving_average(tokens, window), color=C["grey"], lw=2.4)
    tokens_ax.set_ylabel("completion tokens")
    tokens_ax.set_xlabel("training step")
    tokens_ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(title or run_dir.name)
    return Path(save_fig(fig, str(run_dir / "training_curves.png")))
