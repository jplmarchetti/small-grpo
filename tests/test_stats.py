"""Stage 14 on made-up data: the analysis is finished before any real result exists."""

import json

import numpy as np
import pytest

from grpo_gsm8k.stats import (
    ScoreTable,
    average_over_seeds,
    bootstrap_intervals,
    bootstrap_samples,
    build_score_table,
    compute_effects,
    count_strict_gain_sources,
    read_prediction_rows,
    split_gain,
)

# METHODOLOGY §9, made-up accuracies in percent.
MADE_UP = {
    "strict": {"B0": 3, "R0": 5, "RF": 21, "RC": 12, "RFC": 36},
    "lenient": {"B0": 20, "R0": 22, "RF": 23, "RC": 35, "RFC": 38},
}
MODELS = ("B0", "R0", "RF", "RC", "RFC")


def table_with_accuracies(accuracies_percent: dict[str, int], n_problems: int = 100, seed: int = 0) -> ScoreTable:
    """A 0/1 table whose column means are exactly the given percentages (100 problems)."""
    rng = np.random.default_rng(seed)
    columns = []
    for model in MODELS:
        column = np.zeros(n_problems)
        column[rng.choice(n_problems, accuracies_percent[model], replace=False)] = 1.0
        columns.append(column)
    problem_ids = [f"test-{i:05d}" for i in range(n_problems)]
    return ScoreTable(problem_ids, MODELS, np.stack(columns, axis=1))


def rows(scores: dict[str, tuple[int, int]]) -> list[dict]:
    """{problem_id: (strict, lenient)} -> prediction rows."""
    return [
        {"problem_id": pid, "strict_correct": bool(s), "lenient_correct": bool(l)}
        for pid, (s, l) in scores.items()
    ]


# ---------------------------------------------------------------- gains


@pytest.mark.parametrize(
    "grader, format_gain, correctness_gain, total_gain",
    [("strict", 20, 11, 31), ("lenient", 2, 14, 16)],
)
def test_made_up_table_of_the_methodology(grader, format_gain, correctness_gain, total_gain):
    gains = split_gain(MADE_UP[grader])
    assert gains.format_gain == pytest.approx(format_gain)
    assert gains.correctness_gain == pytest.approx(correctness_gain)
    assert gains.total_gain == pytest.approx(total_gain)


def test_halves_are_reported():
    gains = split_gain(MADE_UP["strict"])
    assert (gains.correctness_without_format, gains.correctness_with_format) == (7, 15)
    assert (gains.format_without_correctness, gains.format_with_correctness) == (16, 24)


def test_gains_always_add_up_to_the_total():
    rng = np.random.default_rng(1)
    for _ in range(100):
        accuracies = dict(zip(("R0", "RF", "RC", "RFC"), rng.random(4)))
        gains = split_gain(accuracies)
        assert gains.format_gain + gains.correctness_gain == pytest.approx(gains.total_gain)


def test_effects_include_the_training_only_effect():
    effects = compute_effects(MADE_UP["lenient"])
    assert effects["R0_minus_B0"] == 2
    assert effects["acc_RFC"] == 38
    assert effects["correctness_gain"] == 14


def test_effects_without_b0_leave_it_out():
    accuracies = {k: v for k, v in MADE_UP["lenient"].items() if k != "B0"}
    assert "R0_minus_B0" not in compute_effects(accuracies)


# ---------------------------------------------------------------- score table


def test_made_up_score_table_reproduces_the_gains():
    table = table_with_accuracies(MADE_UP["strict"])
    effects = compute_effects(table.accuracies())
    assert effects["format_gain"] == pytest.approx(0.20)
    assert effects["correctness_gain"] == pytest.approx(0.11)


def test_build_score_table_aligns_models_by_problem_id():
    b0 = rows({"test-00000": (0, 1), "test-00001": (1, 1)})
    rfc = list(reversed(rows({"test-00000": (1, 1), "test-00001": (0, 0)})))
    table = build_score_table({"B0": b0, "RFC": rfc}, grader="strict")
    assert table.problem_ids == ["test-00000", "test-00001"]
    assert table.scores.tolist() == [[0.0, 1.0], [1.0, 0.0]]
    lenient = build_score_table({"B0": b0, "RFC": rfc}, grader="lenient")
    assert lenient.scores.tolist() == [[1.0, 1.0], [1.0, 0.0]]


def test_build_score_table_refuses_mismatched_problems():
    with pytest.raises(ValueError):
        build_score_table(
            {"B0": rows({"test-00000": (1, 1)}), "RFC": rows({"test-00001": (1, 1)})}, grader="strict"
        )


def test_build_score_table_refuses_unknown_grader():
    with pytest.raises(ValueError):
        build_score_table({"B0": rows({"test-00000": (1, 1)})}, grader="format")


def test_average_over_seeds_gives_fractional_scores():
    ids = ["a", "b"]
    seed_tables = [
        ScoreTable(ids, ("R0",), np.array([[1.0], [0.0]])),
        ScoreTable(ids, ("R0",), np.array([[1.0], [1.0]])),
        ScoreTable(ids, ("R0",), np.array([[0.0], [0.0]])),
    ]
    averaged = average_over_seeds(seed_tables)
    assert averaged.scores[:, 0] == pytest.approx([2 / 3, 1 / 3])


def test_average_over_seeds_refuses_different_layouts():
    with pytest.raises(ValueError):
        average_over_seeds(
            [ScoreTable(["a"], ("R0",), np.ones((1, 1))), ScoreTable(["b"], ("R0",), np.ones((1, 1)))]
        )


def test_read_prediction_rows(tmp_path):
    path = tmp_path / "test_predictions.jsonl"
    expected = rows({"test-00000": (1, 1), "test-00001": (0, 1)})
    path.write_text("".join(json.dumps(row) + "\n" for row in expected))
    assert read_prediction_rows(path) == expected


# ---------------------------------------------------------------- bootstrap


def test_bootstrap_is_reproducible_from_the_analysis_seed():
    table = table_with_accuracies(MADE_UP["lenient"])
    first = bootstrap_intervals(table, n_resamples=500, analysis_seed=42)
    second = bootstrap_intervals(table, n_resamples=500, analysis_seed=42)
    assert first == second
    assert first != bootstrap_intervals(table, n_resamples=500, analysis_seed=43)


def test_identity_holds_on_every_resample():
    samples = bootstrap_samples(table_with_accuracies(MADE_UP["strict"]), n_resamples=500, analysis_seed=0)
    assert np.allclose(samples["format_gain"] + samples["correctness_gain"], samples["total_gain"])


def test_intervals_contain_the_point_estimate():
    table = table_with_accuracies(MADE_UP["lenient"])
    point = compute_effects(table.accuracies())
    intervals = bootstrap_intervals(table, n_resamples=2000, analysis_seed=0)
    for name, (low, high) in intervals.items():
        assert low <= point[name] <= high, name


def test_identical_models_have_a_zero_width_gain_interval():
    """Resampling whole rows: two models with the same scores never differ."""
    column = table_with_accuracies(MADE_UP["lenient"]).scores[:, [0]]
    table = ScoreTable([f"p{i}" for i in range(100)], MODELS, np.repeat(column, 5, axis=1))
    low, high = bootstrap_intervals(table, n_resamples=500, analysis_seed=0)["total_gain"]
    assert low == high == 0.0


def test_paired_resampling_gives_tight_gain_intervals():
    """RFC = R0 plus 5 extra problems: the gain's interval is much tighter than either accuracy's."""
    rng = np.random.default_rng(3)
    r0 = (rng.random(1000) < 0.4).astype(float)
    rfc = r0.copy()
    rfc[np.flatnonzero(r0 == 0)[:50]] = 1.0
    scores = np.stack([r0, r0, r0, r0, rfc], axis=1)  # B0, R0, RF, RC, RFC
    table = ScoreTable([f"p{i}" for i in range(1000)], MODELS, scores)
    intervals = bootstrap_intervals(table, n_resamples=2000, analysis_seed=0)
    gain_width = np.subtract(*reversed(intervals["total_gain"]))
    accuracy_width = np.subtract(*reversed(intervals["acc_RFC"]))
    assert intervals["total_gain"][0] > 0
    assert gain_width < accuracy_width / 2


# ---------------------------------------------------------------- per-problem count


def test_strict_gain_sources():
    b0 = rows({
        "p0": (0, 1),  # had the answer, unformatted -> RFC strict: already right
        "p1": (0, 0),  # no answer -> RFC strict: newly solved
        "p2": (1, 1),  # B0 strict, RFC lost it
        "p3": (1, 1),  # both strict: not counted
        "p4": (0, 0),  # neither: not counted
    })
    rfc = rows({"p0": (1, 1), "p1": (1, 1), "p2": (0, 1), "p3": (1, 1), "p4": (0, 0)})
    sources = count_strict_gain_sources(b0, rfc)
    assert (sources.already_right_now_formatted, sources.newly_solved, sources.lost) == (1, 1, 1)
    assert sources.net_strict_gain == 1


def test_strict_gain_sources_add_up_on_random_rows():
    rng = np.random.default_rng(7)
    for _ in range(50):
        lenient_b0, lenient_rfc = rng.random(200) < 0.5, rng.random(200) < 0.5
        strict_b0 = lenient_b0 & (rng.random(200) < 0.3)  # strict implies lenient
        strict_rfc = lenient_rfc & (rng.random(200) < 0.8)
        b0 = rows({f"p{i}": (strict_b0[i], lenient_b0[i]) for i in range(200)})
        rfc = rows({f"p{i}": (strict_rfc[i], lenient_rfc[i]) for i in range(200)})
        s = count_strict_gain_sources(b0, rfc)
        assert s.already_right_now_formatted + s.newly_solved - s.lost == s.net_strict_gain
        assert s.net_strict_gain == int(strict_rfc.sum()) - int(strict_b0.sum())
