"""Stage 16: hand-checks of the lenient grader.

A person reads completions and judges "right answer or not" without the grader; we then
measure how often the lenient grader agrees. This is a check on C_l, not a result.
The grader's decision is the last column, so it can be hidden while reading.
"""

import csv
import random
from collections.abc import Sequence
from pathlib import Path

AUDIT_COLUMNS = ("problem_id", "question", "gold_answer", "completion", "human_correct", "grader_lenient")
YES, NO = "1", "0"


def draw_audit_sample(prediction_rows: Sequence[dict], n: int, seed: int) -> list[dict]:
    """`n` rows chosen with a fixed seed, kept in their original order."""
    chosen = sorted(random.Random(seed).sample(range(len(prediction_rows)), n))
    return [prediction_rows[index] for index in chosen]


def write_audit_sheet(sample_rows: Sequence[dict], questions: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(
                {
                    "problem_id": row["problem_id"],
                    "question": questions[row["problem_id"]],
                    "gold_answer": row["gold_answer"],
                    "completion": row["completion"],
                    "human_correct": "",
                    "grader_lenient": YES if row["lenient_correct"] else NO,
                }
            )


def read_audit_sheet(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def grader_agreement(filled_rows: Sequence[dict]) -> tuple[float, list[dict]]:
    """Share of rows where human and lenient grader agree, and the rows where they do not."""
    unfilled = [row["problem_id"] for row in filled_rows if row["human_correct"].strip() not in (YES, NO)]
    if unfilled:
        raise ValueError(f"human_correct must be 1 or 0; missing for {unfilled[:5]}")
    disagreements = [row for row in filled_rows if row["human_correct"].strip() != row["grader_lenient"]]
    return 1 - len(disagreements) / len(filled_rows), disagreements


# ---------------------------------------------------------------- post-training hand-check (§13, 2026-11-16)


def pool_seed_rows(rows_by_seed: dict[int, Sequence[dict]]) -> list[dict]:
    """One list over all seeds, each row's id tagged with its seed (a problem appears once per seed)."""
    return [
        {**row, "problem_id": f"{row['problem_id']}/seed{seed}"}
        for seed in sorted(rows_by_seed)
        for row in rows_by_seed[seed]
    ]


def draw_format_failure_slice(pooled_rows: Sequence[dict], n: int, seed: int, exclude: Sequence[dict]) -> list[dict]:
    """The targeted slice: completions without the FINAL line, not already in the uniform sample.

    Picked on purpose from the hard cases, so its agreement rate is reported apart from
    the uniform one and never pooled with it.
    """
    excluded = {row["problem_id"] for row in exclude}
    candidates = [row for row in pooled_rows if not row["format_ok"] and row["problem_id"] not in excluded]
    if len(candidates) < n:
        raise ValueError(f"only {len(candidates)} format failures to draw {n} from")
    return draw_audit_sample(candidates, n, seed)
