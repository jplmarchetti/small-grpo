"""Stage 8: turn a grade into the cell's reward (METHODOLOGY §6).

This is the only place where the four cells differ. Every completion is always
graded with all three rules; the cell only chooses which grades become the reward.

Pattern adapted from rasbt/reasoning-from-scratch (`7_6_plus_format_reward.py`,
reward = correctness + weight * format), Apache-2.0, Copyright Sebastian Raschka.
Here the weight is fixed at 1 and format is never multiplied by correctness.
"""

from enum import Enum

import numpy as np

from grpo_gsm8k.graders import Grade


class RewardCell(Enum):
    R0 = "R0"  # random: 1 with probability p0, no information
    RF = "RF"  # format only
    RC = "RC"  # lenient correctness only
    RFC = "RFC"  # both


def random_reward(p0: float, reward_rng: np.random.Generator) -> float:
    """R0's coin flip. Its own generator, so it never disturbs token sampling."""
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must be in (0, 1), got {p0}")
    return float(reward_rng.random() < p0)


def cell_reward(
    grade: Grade,
    cell: RewardCell,
    p0: float | None = None,
    reward_rng: np.random.Generator | None = None,
) -> float:
    format_reward = float(grade.format_ok)
    correctness_reward = float(grade.lenient_correct)  # lenient, so RC is not also paid for format
    match cell:
        case RewardCell.R0:
            if p0 is None or reward_rng is None:
                raise ValueError("R0 needs p0 (from manifests/preflight.json) and the run's reward_rng")
            return random_reward(p0, reward_rng)
        case RewardCell.RF:
            return format_reward
        case RewardCell.RC:
            return correctness_reward
        case RewardCell.RFC:
            return correctness_reward + format_reward
    raise ValueError(f"unknown cell {cell!r}")
