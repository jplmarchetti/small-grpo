"""Stage 15: the report's tables and figures, computed from saved files only (laptop).

- Computing and drawing are separate: `analyze` runs the §9 statistics once per grader,
  and every table and figure draws from its result, so a figure cannot disagree with
  the table next to it.
- Reporting rules (METHODOLOGY §13, 2026-11-16): lenient for correctness, strict for the
  benchmark view and format, always shown together; every gain also per seed, since
  the bootstrap resamples problems only; "B0" is the baseline of the trained prompt
  (few-shot), and the zero-shot baseline is shown as "what prompting alone buys".
- Tables are returned as Markdown text; figures are written as PNG in the house style.
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from grpo_gsm8k.plot_style import C, apply_house_style, save_fig  # noqa: E402
from grpo_gsm8k.stats import (  # noqa: E402
    ScoreTable,
    StrictGainSources,
    average_over_seeds,
    bootstrap_intervals,
    build_score_table,
    compute_effects,
    count_strict_gain_sources,
    read_prediction_rows,
)

CELLS = ("R0", "RF", "RC", "RFC")
GRADERS = ("lenient", "strict")
BASELINE = "B0"  # the trained prompt's baseline (few-shot)
ZERO_SHOT = "B0_zeroshot"
CELL_COLORS = {"R0": C["grey"], "RF": C["blue"], "RC": C["orange"], "RFC": C["green"]}

GAIN_ROWS = (
    ("correctness_gain", "Correctness gain"),
    ("correctness_without_format", "  RC − R0"),
    ("correctness_with_format", "  RFC − RF"),
    ("format_gain", "Format gain"),
    ("format_without_correctness", "  RF − R0"),
    ("format_with_correctness", "  RFC − RC"),
    ("total_gain", "Total (RFC − R0)"),
    ("R0_minus_B0", "R0 − B0 (no-signal drift)"),
)


# ---------------------------------------------------------------- loading


@dataclass(frozen=True)
class TestPredictions:
    baselines: dict[str, list[dict]]  # BASELINE and ZERO_SHOT
    runs: dict[tuple[str, int], list[dict]]  # (cell, seed) -> rows
    seeds: tuple[int, ...]


def load_test_predictions(test_root: Path, seeds: Sequence[int]) -> TestPredictions:
    """outputs/<config_tag>/test as written by run_final_eval.py."""
    baselines = {
        BASELINE: read_prediction_rows(test_root / "B0_fewshot" / "predictions.jsonl"),
        ZERO_SHOT: read_prediction_rows(test_root / "B0" / "predictions.jsonl"),
    }
    runs = {
        (cell, seed): read_prediction_rows(test_root / cell / f"seed{seed}" / "predictions.jsonl")
        for cell in CELLS
        for seed in seeds
    }
    return TestPredictions(baselines, runs, tuple(seeds))


def seed_score_table(predictions: TestPredictions, seed: int, grader: str) -> ScoreTable:
    rows = dict(predictions.baselines) | {cell: predictions.runs[(cell, seed)] for cell in CELLS}
    return build_score_table(rows, grader)


# ---------------------------------------------------------------- statistics


@dataclass(frozen=True)
class GraderResults:
    grader: str
    estimates: dict[str, float]  # compute_effects on the seed-averaged table
    intervals: dict[str, tuple[float, float]]  # bootstrap over problems only
    per_seed: dict[int, dict[str, float]]  # compute_effects on each seed's table


def analyze(predictions: TestPredictions, grader: str, n_resamples: int, analysis_seed: int) -> GraderResults:
    seed_tables = {seed: seed_score_table(predictions, seed, grader) for seed in predictions.seeds}
    pooled = average_over_seeds(list(seed_tables.values()))
    return GraderResults(
        grader=grader,
        estimates=compute_effects(pooled.accuracies()),
        intervals=bootstrap_intervals(pooled, n_resamples, analysis_seed),
        per_seed={seed: compute_effects(table.accuracies()) for seed, table in seed_tables.items()},
    )


def mean_of(rows: Sequence[dict], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def describe_model(rows_per_seed: Sequence[Sequence[dict]]) -> dict[str, float]:
    """Format, length and truncation, averaged over the given seeds."""
    return {
        field: float(np.mean([mean_of(rows, field) for rows in rows_per_seed]))
        for field in ("format_ok", "n_tokens", "truncated")
    }


@dataclass(frozen=True)
class Headroom:
    format_rate: float  # baseline completions in the required format
    lenient_not_strict: float  # right number, not on the FINAL line: what format can still fix


def format_headroom(baseline_rows: Sequence[dict]) -> Headroom:
    misplaced = [row["lenient_correct"] and not row["strict_correct"] for row in baseline_rows]
    return Headroom(mean_of(baseline_rows, "format_ok"), float(np.mean(misplaced)))


def strict_gain_sources_per_seed(predictions: TestPredictions) -> dict[int, StrictGainSources]:
    baseline = predictions.baselines[BASELINE]
    return {seed: count_strict_gain_sources(baseline, predictions.runs[("RFC", seed)]) for seed in predictions.seeds}


# ---------------------------------------------------------------- tables


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def with_interval(value: float, interval: tuple[float, float]) -> str:
    return f"{pct(value)} [{pct(interval[0])}, {pct(interval[1])}]"


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def make_results_table(predictions: TestPredictions, results: Mapping[str, GraderResults]) -> str:
    """One row per model: accuracies with intervals (seed means), per-seed lenient, format, length."""
    lenient, strict = results["lenient"], results["strict"]
    header = ["Model", "Lenient % [95% CI]", "Strict % [95% CI]", "Lenient per seed", "Format %", "Mean tokens",
              "Truncated %"]
    rows = []
    for model, label in ((ZERO_SHOT, "B0, zero-shot prompt"), (BASELINE, "B0, few-shot prompt (trained)")):
        traits = describe_model([predictions.baselines[model]])
        rows.append([label, with_interval(lenient.estimates[f"acc_{model}"], lenient.intervals[f"acc_{model}"]),
                     with_interval(strict.estimates[f"acc_{model}"], strict.intervals[f"acc_{model}"]), "–",
                     pct(traits["format_ok"]), f"{traits['n_tokens']:.0f}", pct(traits["truncated"])])
    for cell in CELLS:
        traits = describe_model([predictions.runs[(cell, seed)] for seed in predictions.seeds])
        per_seed = " / ".join(pct(lenient.per_seed[seed][f"acc_{cell}"]) for seed in predictions.seeds)
        rows.append([cell, with_interval(lenient.estimates[f"acc_{cell}"], lenient.intervals[f"acc_{cell}"]),
                     with_interval(strict.estimates[f"acc_{cell}"], strict.intervals[f"acc_{cell}"]), per_seed,
                     pct(traits["format_ok"]), f"{traits['n_tokens']:.0f}", pct(traits["truncated"])])
    return markdown_table(header, rows)


def make_effects_table(results: Mapping[str, GraderResults], seeds: Sequence[int]) -> str:
    """Gains in points, both graders, with the problem-only interval and every seed's own value."""
    header = ["Quantity (points)"]
    for grader in GRADERS:
        header += [f"{grader.capitalize()} [95% CI]", f"{grader.capitalize()} per seed"]
    rows = []
    for key, label in GAIN_ROWS:
        row = [label]
        for grader in GRADERS:
            result = results[grader]
            row.append(with_interval(result.estimates[key], result.intervals[key]))
            row.append(" / ".join(pct(result.per_seed[seed][key]) for seed in seeds))
        rows.append(row)
    return markdown_table(header, rows)


def make_strict_sources_table(sources: Mapping[int, StrictGainSources]) -> str:
    header = ["Seed", "Already right, now formatted", "Newly solved", "Lost", "Net strict gain"]
    rows = [[str(seed), str(s.already_right_now_formatted), str(s.newly_solved), str(s.lost), str(s.net_strict_gain)]
            for seed, s in sources.items()]
    return markdown_table(header, rows)


def make_cost_table(train_logs: Mapping[tuple[str, int], Sequence[dict]]) -> str:
    """Seconds per step, GPU-hours, peak VRAM and the share of steps with no update, per cell (seed means)."""
    header = ["Cell", "Seconds / step", "Training hours per run (no validation)", "Peak VRAM (GB)", "Steps skipped %"]
    rows = []
    for cell in CELLS:
        logs = [rows_ for (log_cell, _), rows_ in train_logs.items() if log_cell == cell]
        seconds = [sum(float(r["seconds"]) for r in log) for log in logs]
        steps = [len(log) for log in logs]
        skipped = [np.mean([r["skipped_update"] == "True" for r in log]) for log in logs]
        rows.append([cell, f"{np.sum(seconds) / np.sum(steps):.1f}", f"{np.mean(seconds) / 3600:.2f}",
                     f"{max(float(r['peak_vram_gb']) for log in logs for r in log):.1f}", pct(float(np.mean(skipped)))])
    return markdown_table(header, rows)


def read_csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------- example problem


def pick_example_problem(predictions: TestPredictions, seed: int) -> str:
    """Rule fixed in code, not by eye: the first test problem the trained-prompt B0 gets
    strict-wrong and RFC (this seed) gets strict-right."""
    rfc = {row["problem_id"]: row for row in predictions.runs[("RFC", seed)]}
    for row in sorted(predictions.baselines[BASELINE], key=lambda r: r["problem_id"]):
        if not row["strict_correct"] and rfc[row["problem_id"]]["strict_correct"]:
            return row["problem_id"]
    raise ValueError("no problem where RFC is right and B0 is wrong")


def show_example_problem(predictions: TestPredictions, problem_id: str, seed: int) -> str:
    models = [(BASELINE, predictions.baselines[BASELINE])] + [(cell, predictions.runs[(cell, seed)]) for cell in CELLS]
    blocks, gold = [], None
    for name, rows in models:
        row = next(r for r in rows if r["problem_id"] == problem_id)
        gold = row["gold_answer"]
        grades = f"format {int(row['format_ok'])}, strict {int(row['strict_correct'])}, lenient {int(row['lenient_correct'])}"
        blocks.append(f"**{name}** ({grades})\n\n```text\n{row['completion'].strip()}\n```")
    return f"Problem `{problem_id}`, gold answer **{gold}**, seed {seed}.\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------- figures


def plot_gain_split(results: Mapping[str, GraderResults], seeds: Sequence[int], path: Path) -> Path:
    """Correctness and format gain per grader: bar = seed mean, whisker = problem interval, dots = seeds."""
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, grader in zip(axes, GRADERS):
        result = results[grader]
        for x, (key, label, color) in enumerate((("correctness_gain", "Correctness", C["green"]),
                                                 ("format_gain", "Format", C["blue"]))):
            value, (low, high) = result.estimates[key], result.intervals[key]
            ax.bar(x, 100 * value, color=color, width=0.55, alpha=0.85)
            ax.errorbar(x, 100 * value, yerr=[[100 * (value - low)], [100 * (high - value)]], color="#222222",
                        capsize=6, lw=1.6)
            ax.scatter([x + 0.2] * len(seeds), [100 * result.per_seed[s][key] for s in seeds], color="#222222",
                       s=22, zorder=3)
            ax.annotate(f"{100 * value:+.1f}", (x - 0.2, 100 * high + 0.4), ha="center", fontweight="bold")
        ax.axhline(0, color="#444444", lw=1)
        ax.set_xticks([0, 1], ["Correctness gain", "Format gain"])
        ax.set_title(f"{grader.capitalize()} grader")
    axes[0].set_ylabel("Gain (accuracy points)")
    fig.suptitle("Gain split at step 150 (bars: seed mean; whiskers: 95% CI over problems; dots: seeds)")
    return Path(save_fig(fig, str(path)))


def plot_validation_curves(val_curves: Mapping[tuple[str, int], Sequence[dict]], path: Path) -> Path:
    """Validation lenient accuracy, format rate and length per step: thin lines = seeds, thick = mean."""
    apply_house_style()
    panels = (("lenient_accuracy", "Lenient accuracy (%)", 100), ("format_rate", "Format rate (%)", 100),
              ("mean_tokens", "Mean tokens", 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (field, label, scale) in zip(axes, panels):
        for cell in CELLS:
            curves = [curve for (c, _), curve in sorted(val_curves.items()) if c == cell]
            steps = [float(row["step"]) for row in curves[0]]
            values = np.array([[scale * float(row[field]) for row in curve] for curve in curves])
            for seed_values in values:
                ax.plot(steps, seed_values, color=CELL_COLORS[cell], alpha=0.3, lw=1)
            ax.plot(steps, values.mean(axis=0), "o-", color=CELL_COLORS[cell], lw=2.4, ms=6, label=cell)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_xticks(steps)
    axes[0].legend()
    fig.suptitle("Validation curves, config-v2 (500 problems, greedy; step 0 is the untrained model)")
    return Path(save_fig(fig, str(path)))


def plot_strict_gain_sources(sources: Mapping[int, StrictGainSources], path: Path) -> Path:
    apply_house_style()
    labels = ("Already right,\nnow formatted", "Newly solved", "Lost")
    means = [np.mean([getattr(s, f) for s in sources.values()]) for f in ("already_right_now_formatted",
                                                                          "newly_solved", "lost")]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(range(3), means, color=[C["blue"], C["green"], C["red"]], width=0.55)
    for bar, value in zip(bars, means):
        ax.annotate(f"{value:.0f}", (bar.get_x() + bar.get_width() / 2, value + 2), ha="center", fontweight="bold")
    ax.set_xticks(range(3), labels)
    ax.set_ylabel("Test problems (mean over seeds)")
    ax.set_title("RFC vs B0 (few-shot): where the strict gain comes from")
    return Path(save_fig(fig, str(path)))


def plot_config_v1_drift(v1_r0: Sequence[dict], v2_r0: Mapping[int, Sequence[dict]], path: Path) -> Path:
    """The stability lesson: R0 (random reward) under config-v1 vs config-v2, on validation."""
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
    for ax, (field, label) in zip(axes, (("lenient_accuracy", "Lenient accuracy (%)"),
                                         ("format_rate", "Format rate (%)"))):
        ax.plot([float(r["step"]) for r in v1_r0], [100 * float(r[field]) for r in v1_r0], "o-", color=C["red"],
                lw=2.4, label="config-v1: lr 3e-5, 500 steps (seed 42)")
        for i, (seed, curve) in enumerate(sorted(v2_r0.items())):
            ax.plot([float(r["step"]) for r in curve], [100 * float(r[field]) for r in curve], "o-", color=C["green"],
                    lw=1.6, alpha=0.8, label="config-v2: lr 1e-5, 150 steps (3 seeds)" if i == 0 else None)
        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_ylim(0, 100)
    axes[0].legend(loc="lower left")
    fig.suptitle("R0 gets a random reward, yet config-v1 drifted until the model could no longer answer")
    return Path(save_fig(fig, str(path)))
