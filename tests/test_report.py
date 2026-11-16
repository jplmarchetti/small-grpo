import json

import pytest

from grpo_gsm8k.report import (
    BASELINE,
    CELLS,
    ZERO_SHOT,
    analyze,
    format_headroom,
    load_test_predictions,
    make_cost_table,
    make_effects_table,
    make_results_table,
    pick_example_problem,
    plot_gain_split,
    show_example_problem,
    strict_gain_sources_per_seed,
)

SEEDS = (42, 43)
N_PROBLEMS = 40
# lenient-correct problems per model (strict = lenient here, except where noted)
LENIENT_COUNT = {"B0": 16, "B0_fewshot": 16, "R0": 16, "RF": 16, "RC": 24, "RFC": 24}


def row(i, lenient, strict=None, format_ok=True):
    return {"problem_id": f"test-{i:05d}", "gold_answer": "7", "completion": f" answer {i}\nFINAL: 7",
            "predicted_number": "7", "format_ok": format_ok, "lenient_correct": lenient,
            "strict_correct": lenient if strict is None else strict, "n_tokens": 50, "truncated": False}


def write_model(directory, n_lenient, misplaced=0):
    """The first n_lenient problems are lenient-correct; the first `misplaced` of them are not strict."""
    directory.mkdir(parents=True)
    rows = [row(i, i < n_lenient, strict=(i >= misplaced and i < n_lenient)) for i in range(N_PROBLEMS)]
    (directory / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.fixture
def predictions(tmp_path):
    write_model(tmp_path / "B0", LENIENT_COUNT["B0"])
    write_model(tmp_path / "B0_fewshot", LENIENT_COUNT["B0_fewshot"], misplaced=2)
    for cell in CELLS:
        for seed in SEEDS:
            write_model(tmp_path / cell / f"seed{seed}", LENIENT_COUNT[cell])
    return load_test_predictions(tmp_path, SEEDS)


def test_gains_add_up_and_match_the_made_up_counts(predictions):
    lenient = analyze(predictions, "lenient", n_resamples=200, analysis_seed=0)
    assert lenient.estimates["correctness_gain"] == pytest.approx(8 / N_PROBLEMS)
    assert lenient.estimates["format_gain"] == pytest.approx(0.0)
    assert lenient.estimates["R0_minus_B0"] == pytest.approx(0.0)  # B0 = the few-shot (trained-prompt) baseline
    assert set(lenient.per_seed) == set(SEEDS)
    est = lenient.estimates
    assert est["correctness_gain"] + est["format_gain"] == pytest.approx(est["total_gain"])
    low, high = lenient.intervals["correctness_gain"]
    assert low <= est["correctness_gain"] <= high


def test_baseline_is_the_trained_prompt(predictions):
    strict = analyze(predictions, "strict", n_resamples=50, analysis_seed=0)
    assert strict.estimates[f"acc_{BASELINE}"] == pytest.approx(14 / N_PROBLEMS)  # 2 misplaced answers
    assert strict.estimates[f"acc_{ZERO_SHOT}"] == pytest.approx(16 / N_PROBLEMS)


def test_headroom_counts_right_numbers_in_the_wrong_place(predictions):
    headroom = format_headroom(predictions.baselines[BASELINE])
    assert headroom.lenient_not_strict == pytest.approx(2 / N_PROBLEMS)
    assert headroom.format_rate == 1.0


def test_strict_gain_sources_add_up_per_seed(predictions):
    for sources in strict_gain_sources_per_seed(predictions).values():
        assert (sources.already_right_now_formatted, sources.newly_solved, sources.lost) == (2, 8, 0)
        assert sources.net_strict_gain == 10


def test_tables_show_every_model_and_every_seed(predictions):
    results = {g: analyze(predictions, g, n_resamples=50, analysis_seed=0) for g in ("lenient", "strict")}
    table = make_results_table(predictions, results)
    for name in ("zero-shot", "few-shot", *CELLS):
        assert name in table
    effects = make_effects_table(results, SEEDS)
    assert "Correctness gain" in effects and "20.0 / 20.0" in effects


def test_reports_are_deterministic(predictions, tmp_path):
    first = analyze(predictions, "lenient", n_resamples=100, analysis_seed=42)
    second = analyze(predictions, "lenient", n_resamples=100, analysis_seed=42)
    assert first == second
    results = {"lenient": first, "strict": analyze(predictions, "strict", 100, 42)}
    assert plot_gain_split(results, SEEDS, tmp_path / "gain.png").exists()


def test_example_problem_follows_its_rule(predictions):
    problem_id = pick_example_problem(predictions, 42)
    assert problem_id == "test-00000"  # B0 misplaced it (strict-wrong); RFC is strict-right
    assert "gold answer **7**" in show_example_problem(predictions, problem_id, 42)


def test_cost_table_counts_skipped_steps():
    log = [{"seconds": "10", "skipped_update": "True", "peak_vram_gb": "3.0"},
           {"seconds": "6", "skipped_update": "False", "peak_vram_gb": "4.5"}]
    logs = {(cell, 42): log for cell in CELLS}
    table = make_cost_table(logs)
    assert "| R0 | 8.0 | 0.00 | 4.5 | 50.0 |" in table
