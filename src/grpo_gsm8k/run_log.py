"""Stage 10 logging: one CSV row per training step, sample completions every few steps.

Pattern adapted from rasbt/reasoning-from-scratch (`append_step_metrics`,
`append_sample_logs`), Apache-2.0, Copyright Sebastian Raschka.
"""

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

SIGNAL_SOURCES = ("format_only", "correctness_only", "both", "none")


@dataclass(frozen=True)
class StepLog:
    step: int
    mean_reward: float
    format_rate: float  # F, logged in every cell
    lenient_rate: float  # C_l
    strict_rate: float  # C_s
    mean_tokens: float
    truncation_rate: float
    zero_variance_groups: float  # share of groups whose rewards are all equal
    groups_format_only: int  # groups that differ in format but not in correctness
    groups_correctness_only: int
    groups_both: int
    groups_none: int
    skipped_update: bool  # every advantage was 0, no optimizer step
    loss: float
    grad_norm: float
    seconds: float
    peak_vram_gb: float


STEP_LOG_COLUMNS = tuple(field.name for field in fields(StepLog))


def append_step_log(path: Path, log: StepLog) -> None:
    """Append one row; write the header when the file is new."""
    is_new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_LOG_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(asdict(log))


def read_step_logs(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def truncate_step_log(path: Path, last_step: int) -> int:
    """Drop rows after `last_step` (used when resuming); return how many were dropped."""
    if not path.exists():
        return 0
    rows = read_step_logs(path)
    kept = [row for row in rows if int(row["step"]) <= last_step]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows) - len(kept)


def append_sample_completions(
    path: Path, step: int, questions: Sequence[str], texts: Sequence[str], rewards: Sequence[float]
) -> None:
    """A few completions in plain text, to read what the policy is actually writing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(f"\n===== step {step} =====\n")
        for question, text, reward in zip(questions, texts, rewards, strict=True):
            handle.write(f"\n--- reward {reward:g} | {question[:100]}\n{text}\n")
