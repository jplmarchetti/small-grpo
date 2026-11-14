"""Stage 11b: choose the one learning rate used by every cell and seed (METHODOLOGY §7, §13).

- RC is trained for 100 steps at each candidate rate, with greedy validation at steps 0, 50, 100.
- A rate "collapses" if at step 50 or 100, compared with its own step 0, lenient accuracy
  drops too much, truncation rises too much, or completions shrink below a fraction of
  their starting length. The thresholds sit in configs/base.yaml and were fixed before
  any pilot curve existed.
- The chosen rate is the largest one that does not collapse: the main risk of the study
  is that nothing moves in 500 steps, so we prefer the most movement that is still safe.

Pure functions; the script only runs the pilot and prints this module's verdicts.
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from grpo_gsm8k.config import LrPilotConfig


@dataclass(frozen=True)
class PilotVerdict:
    learning_rate: float
    collapsed: bool
    reasons: tuple[str, ...]


def read_val_curve(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def judge_collapse(learning_rate: float, val_curve: Sequence[Mapping[str, float]], rule: LrPilotConfig) -> PilotVerdict:
    """Compare every later validation point with step 0; any broken threshold is a collapse."""
    by_step = {int(row["step"]): row for row in val_curve}
    if 0 not in by_step or len(by_step) < 2:
        raise ValueError(f"lr {learning_rate}: need step 0 and at least one later validation point")
    start = by_step[0]
    reasons = []
    for step in sorted(s for s in by_step if s > 0):
        row = by_step[step]
        lenient_drop = start["lenient_accuracy"] - row["lenient_accuracy"]
        truncation_rise = row["truncation_rate"] - start["truncation_rate"]
        length_ratio = row["mean_tokens"] / start["mean_tokens"]
        if lenient_drop > rule.max_lenient_drop:
            reasons.append(f"step {step}: lenient accuracy fell {lenient_drop:.1%}")
        if truncation_rise > rule.max_truncation_rise:
            reasons.append(f"step {step}: truncation rose {truncation_rise:.1%}")
        if length_ratio < rule.min_length_ratio:
            reasons.append(f"step {step}: length fell to {length_ratio:.0%} of step 0")
    return PilotVerdict(learning_rate, bool(reasons), tuple(reasons))


def choose_learning_rate(verdicts: Sequence[PilotVerdict]) -> float | None:
    """The largest rate that did not collapse; None if every rate collapsed (then the pilot must be revisited)."""
    stable = [verdict.learning_rate for verdict in verdicts if not verdict.collapsed]
    return max(stable) if stable else None


def pilot_run_dir(root: Path, learning_rate: float) -> Path:
    return root / f"lr_{learning_rate:.0e}"
