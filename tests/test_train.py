"""Pure helpers of train.py. train.py imports torch, so this runs on the GPU PC."""

import pytest

pytest.importorskip("torch")

from grpo_gsm8k.data import Problem  # noqa: E402
from grpo_gsm8k.graders import Grade  # noqa: E402
from grpo_gsm8k.train import classify_signal_source, make_problem_order, problems_for_step  # noqa: E402

PROBLEMS = [Problem(f"train-{i:05d}", f"q{i}", str(i), f"#### {i}") for i in range(1000)]


def grade(format_ok: bool, lenient: bool) -> Grade:
    return Grade(format_ok, format_ok and lenient, lenient, None)


def test_problem_order_depends_only_on_the_seed():
    assert make_problem_order(PROBLEMS, 42) == make_problem_order(PROBLEMS, 42)
    assert make_problem_order(PROBLEMS, 42) != make_problem_order(PROBLEMS, 43)
    assert sorted(p.problem_id for p in make_problem_order(PROBLEMS, 42)) == [p.problem_id for p in PROBLEMS]


def test_steps_walk_through_the_order_once():
    order = make_problem_order(PROBLEMS, 42)
    seen = [p.problem_id for step in range(1, 501) for p in problems_for_step(order, step, 2)]
    assert len(seen) == 1000 and len(set(seen)) == 1000
    assert problems_for_step(order, 1, 2) == order[:2]


@pytest.mark.parametrize(
    "grades, expected",
    [
        ([grade(True, False)] * 8, "none"),
        ([grade(True, True)] + [grade(True, False)] * 7, "correctness_only"),
        ([grade(False, False)] + [grade(True, False)] * 7, "format_only"),
        ([grade(False, True)] + [grade(True, False)] * 7, "both"),
    ],
)
def test_signal_source(grades, expected):
    assert classify_signal_source(grades) == expected
