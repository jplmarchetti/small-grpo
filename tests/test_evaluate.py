import pytest

from grpo_gsm8k.data import Problem
from grpo_gsm8k.evaluate import (
    batches_by_length,
    build_prediction_row,
    evaluate_with,
    summarize_predictions,
    write_predictions,
)
from grpo_gsm8k.rollout import Completion
from grpo_gsm8k.stats import build_score_table, read_prediction_rows

PROBLEMS = [
    Problem("train-00001", "Short?", "18", "#### 18"),
    Problem("train-00002", "A much longer question about apples?", "7", "#### 7"),
    Problem("train-00003", "Medium length question?", "3", "#### 3"),
]


def fake_completion(text: str, truncated: bool = False) -> Completion:
    return Completion(prompt_ids=[1], completion_ids=[2, 3], text=text, n_tokens=2, truncated=truncated)


def fake_generate(answers: dict[str, str], calls: list):
    """Answers keyed by a substring of the question; records each batch."""

    def generate(prompts):
        calls.append(len(prompts))
        return [fake_completion(next(v for k, v in answers.items() if k in prompt)) for prompt in prompts]

    return generate


def test_batches_by_length_covers_every_prompt_once_longest_first():
    prompts = ["aa", "a", "aaaa", "aaa", "aaaaa"]
    batches = batches_by_length(prompts, batch_size=2)
    assert batches == [[4, 2], [3, 0], [1]]
    assert sorted(i for batch in batches for i in batch) == list(range(5))


def test_rows_come_back_in_problem_order():
    calls = []
    answers = {"Short": "FINAL: 18", "apples": "it is 7", "Medium": "FINAL: 4"}
    rows = evaluate_with(fake_generate(answers, calls), PROBLEMS, batch_size=2, show_progress=False)
    assert [row["problem_id"] for row in rows] == ["train-00001", "train-00002", "train-00003"]
    assert [(r["format_ok"], r["strict_correct"], r["lenient_correct"]) for r in rows] == [
        (True, True, True),
        (False, False, True),
        (True, False, False),
    ]
    assert calls == [2, 1]


def test_exemplars_reach_the_prompt():
    seen = []

    def generate(prompts):
        seen.extend(prompts)
        return [fake_completion("FINAL: 1") for _ in prompts]

    exemplar = Problem("train-09999", "Exemplar question?", "5", "2 + 3 = 5\n#### 5")
    evaluate_with(generate, PROBLEMS[:1], batch_size=4, exemplars=[exemplar], show_progress=False)
    assert "Exemplar question?" in seen[0] and "FINAL: 5" in seen[0]


def test_prediction_row_fields():
    row = build_prediction_row(PROBLEMS[0], fake_completion("so 18 left", truncated=True))
    assert row == {
        "problem_id": "train-00001",
        "gold_answer": "18",
        "completion": "so 18 left",
        "predicted_number": "18",
        "format_ok": False,
        "strict_correct": False,
        "lenient_correct": True,
        "n_tokens": 2,
        "truncated": True,
    }


def test_summary():
    rows = [
        build_prediction_row(PROBLEMS[0], fake_completion("FINAL: 18")),
        build_prediction_row(PROBLEMS[1], fake_completion("7", truncated=True)),
    ]
    summary = summarize_predictions(rows)
    assert summary["lenient_accuracy"] == 1.0
    assert summary["strict_accuracy"] == summary["format_rate"] == 0.5
    assert summary["truncation_rate"] == 0.5


def test_written_predictions_are_readable_by_the_statistics(tmp_path):
    rows = [build_prediction_row(p, fake_completion("FINAL: 18")) for p in PROBLEMS]
    path = tmp_path / "predictions.jsonl"
    write_predictions(rows, path)
    assert read_prediction_rows(path) == rows
    table = build_score_table({"B0": rows}, grader="strict")
    assert table.scores[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_evaluation_refuses_sampling():
    from grpo_gsm8k.evaluate import evaluate_policy
    from grpo_gsm8k.rollout import DecodingConfig

    with pytest.raises(ValueError):
        evaluate_policy(None, None, PROBLEMS, DecodingConfig(do_sample=True, max_new_tokens=8), batch_size=2)
