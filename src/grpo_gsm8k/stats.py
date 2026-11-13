"""Stage 14: from per-problem scores to the numbers the paper reports (METHODOLOGY §9).

- The core works on a plain `ScoreTable`: one row per test problem, one column per
  model, filled with 0/1 scores (or seed averages). Reading prediction files is a thin
  layer on top.
- Gains: the correctness gain averages "what the correctness reward adds" over the two
  rows of the 2x2; the format gain does the same over the two columns. They add up to
  RFC - R0 exactly.
- Bootstrap: resample whole rows (problems), so each draw keeps one problem's scores
  for all models together and shared difficulty cancels when models are subtracted.
- Per-problem count: of the problems RFC newly gets strict-correct, did B0 already have
  the right number somewhere (already right) or not (newly solved)?

Pure functions only; nothing here plots or writes files.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRADER_FIELDS = {"strict": "strict_correct", "lenient": "lenient_correct"}
CONFIDENCE_LEVEL = 0.95
RESAMPLES_PER_CHUNK = 500  # bounds memory: chunk x problems x models floats


# ---------------------------------------------------------------- score table


@dataclass(frozen=True)
class ScoreTable:
    problem_ids: list[str]
    models: tuple[str, ...]
    scores: np.ndarray  # (n_problems, n_models)

    def accuracies(self) -> dict[str, float]:
        return dict(zip(self.models, self.scores.mean(axis=0).tolist()))


def read_prediction_rows(path: Path) -> list[dict]:
    """One JSON object per line, as written by the evaluator."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_score_table(rows_by_model: Mapping[str, Sequence[dict]], grader: str) -> ScoreTable:
    """Align every model's rows by problem id; refuse models evaluated on different problems."""
    if grader not in GRADER_FIELDS:
        raise ValueError(f"grader must be one of {sorted(GRADER_FIELDS)}, got {grader!r}")
    field = GRADER_FIELDS[grader]
    scores_by_model = {
        model: {row["problem_id"]: float(row[field]) for row in rows} for model, rows in rows_by_model.items()
    }
    problem_ids = sorted(next(iter(scores_by_model.values())))
    for model, scores in scores_by_model.items():
        if sorted(scores) != problem_ids:
            raise ValueError(f"{model} was evaluated on a different set of problems")
    columns = [[scores[pid] for pid in problem_ids] for scores in scores_by_model.values()]
    return ScoreTable(problem_ids, tuple(scores_by_model), np.array(columns).T)


def average_over_seeds(seed_tables: Sequence[ScoreTable]) -> ScoreTable:
    """Mean score per problem over seeds (0, 1/3, 2/3 or 1 with three seeds)."""
    first = seed_tables[0]
    for table in seed_tables[1:]:
        if table.problem_ids != first.problem_ids or table.models != first.models:
            raise ValueError("seed tables must have the same problems and models in the same order")
    return ScoreTable(first.problem_ids, first.models, np.mean([t.scores for t in seed_tables], axis=0))


# ---------------------------------------------------------------- gains


@dataclass(frozen=True)
class GainSplit:
    correctness_without_format: float  # RC - R0
    correctness_with_format: float  # RFC - RF
    format_without_correctness: float  # RF - R0
    format_with_correctness: float  # RFC - RC
    correctness_gain: float
    format_gain: float
    total_gain: float  # RFC - R0 = format_gain + correctness_gain


def split_gain(accuracies: Mapping[str, float]) -> GainSplit:
    """Works on floats or on arrays of bootstrap resamples alike."""
    r0, rf, rc, rfc = (accuracies[cell] for cell in ("R0", "RF", "RC", "RFC"))
    correctness_without_format, correctness_with_format = rc - r0, rfc - rf
    format_without_correctness, format_with_correctness = rf - r0, rfc - rc
    return GainSplit(
        correctness_without_format=correctness_without_format,
        correctness_with_format=correctness_with_format,
        format_without_correctness=format_without_correctness,
        format_with_correctness=format_with_correctness,
        correctness_gain=(correctness_without_format + correctness_with_format) / 2,
        format_gain=(format_without_correctness + format_with_correctness) / 2,
        total_gain=rfc - r0,
    )


def compute_effects(accuracies: Mapping[str, float]) -> dict:
    """Every reported quantity by name: `acc_<model>`, the gain split, and R0 - B0 if B0 is present."""
    effects = {f"acc_{model}": accuracy for model, accuracy in accuracies.items()}
    effects.update(vars(split_gain(accuracies)))
    if "B0" in accuracies:
        effects["R0_minus_B0"] = accuracies["R0"] - accuracies["B0"]
    return effects


# ---------------------------------------------------------------- bootstrap


def bootstrap_samples(table: ScoreTable, n_resamples: int, analysis_seed: int) -> dict[str, np.ndarray]:
    """Every quantity of `compute_effects`, recomputed on `n_resamples` resampled problem sets."""
    rng = np.random.default_rng(analysis_seed)
    n_problems = len(table.problem_ids)
    resampled_accuracies = []
    for start in range(0, n_resamples, RESAMPLES_PER_CHUNK):
        chunk = min(RESAMPLES_PER_CHUNK, n_resamples - start)
        rows = rng.integers(0, n_problems, size=(chunk, n_problems))
        resampled_accuracies.append(table.scores[rows].mean(axis=1))  # (chunk, n_models)
    accuracy_columns = np.concatenate(resampled_accuracies).T
    return compute_effects(dict(zip(table.models, accuracy_columns)))


def bootstrap_intervals(table: ScoreTable, n_resamples: int, analysis_seed: int) -> dict[str, tuple[float, float]]:
    """95% percentile interval of every quantity."""
    tail = (1 - CONFIDENCE_LEVEL) / 2 * 100
    samples = bootstrap_samples(table, n_resamples, analysis_seed)
    return {
        name: tuple(float(v) for v in np.percentile(values, [tail, 100 - tail]))
        for name, values in samples.items()
    }


# ---------------------------------------------------------------- per-problem count


@dataclass(frozen=True)
class StrictGainSources:
    already_right_now_formatted: int  # RFC strict, B0 not strict but lenient
    newly_solved: int  # RFC strict, B0 lenient-wrong
    lost: int  # B0 strict, RFC not strict
    net_strict_gain: int  # = already_right_now_formatted + newly_solved - lost


def count_strict_gain_sources(b0_rows: Sequence[dict], rfc_rows: Sequence[dict]) -> StrictGainSources:
    rfc_by_id = {row["problem_id"]: row for row in rfc_rows}
    if sorted(rfc_by_id) != sorted(row["problem_id"] for row in b0_rows):
        raise ValueError("B0 and RFC were evaluated on different problems")
    already_right = newly_solved = lost = 0
    for b0 in b0_rows:
        rfc = rfc_by_id[b0["problem_id"]]
        if rfc["strict_correct"] and not b0["strict_correct"]:
            if b0["lenient_correct"]:
                already_right += 1
            else:
                newly_solved += 1
        if b0["strict_correct"] and not rfc["strict_correct"]:
            lost += 1
    return StrictGainSources(already_right, newly_solved, lost, already_right + newly_solved - lost)
