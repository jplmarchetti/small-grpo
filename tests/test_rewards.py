import numpy as np
import pytest

from grpo_gsm8k.graders import Grade
from grpo_gsm8k.rewards import RewardCell, cell_reward


def make_grade(format_ok: bool, lenient_correct: bool) -> Grade:
    return Grade(
        format_ok=format_ok,
        strict_correct=format_ok and lenient_correct,
        lenient_correct=lenient_correct,
        predicted_number="18",
    )


# (format_ok, lenient_correct) -> expected reward for RF, RC, RFC
DETERMINISTIC_TABLE = [
    ((False, False), {RewardCell.RF: 0.0, RewardCell.RC: 0.0, RewardCell.RFC: 0.0}),
    ((True, False), {RewardCell.RF: 1.0, RewardCell.RC: 0.0, RewardCell.RFC: 1.0}),
    ((False, True), {RewardCell.RF: 0.0, RewardCell.RC: 1.0, RewardCell.RFC: 1.0}),
    ((True, True), {RewardCell.RF: 1.0, RewardCell.RC: 1.0, RewardCell.RFC: 2.0}),
]


@pytest.mark.parametrize("grade_flags, expected", DETERMINISTIC_TABLE)
def test_deterministic_cells(grade_flags, expected):
    grade = make_grade(*grade_flags)
    for cell, reward in expected.items():
        assert cell_reward(grade, cell) == reward


def test_rc_uses_lenient_not_strict():
    """Right answer in the wrong place: RC pays, so correctness and format stay independent."""
    assert cell_reward(make_grade(format_ok=False, lenient_correct=True), RewardCell.RC) == 1.0


@pytest.mark.parametrize("grade_flags", [flags for flags, _ in DETERMINISTIC_TABLE])
def test_r0_ignores_the_grade(grade_flags):
    """Same seed, any grade: the same coin flips."""
    draws = [cell_reward(make_grade(*grade_flags), RewardCell.R0, 0.3, np.random.default_rng(7)) for _ in range(5)]
    reference = [cell_reward(make_grade(False, False), RewardCell.R0, 0.3, np.random.default_rng(7)) for _ in range(5)]
    assert draws == reference


def test_r0_mean_matches_p0():
    rng = np.random.default_rng(42)
    grade = make_grade(False, False)
    draws = [cell_reward(grade, RewardCell.R0, 0.25, rng) for _ in range(10_000)]
    assert set(draws) == {0.0, 1.0}
    assert abs(np.mean(draws) - 0.25) < 0.02


def test_r0_is_reproducible_from_the_seed():
    grade = make_grade(False, False)
    first = [cell_reward(grade, RewardCell.R0, 0.25, rng) for rng in [np.random.default_rng(1)] for _ in range(50)]
    second = [cell_reward(grade, RewardCell.R0, 0.25, rng) for rng in [np.random.default_rng(1)] for _ in range(50)]
    assert first == second


def test_r0_without_p0_or_rng_fails_loudly():
    grade = make_grade(False, False)
    with pytest.raises(ValueError):
        cell_reward(grade, RewardCell.R0)
    with pytest.raises(ValueError):
        cell_reward(grade, RewardCell.R0, p0=0.2)


@pytest.mark.parametrize("p0", [0.0, 1.0, -0.1, 1.5])
def test_r0_rejects_degenerate_p0(p0):
    with pytest.raises(ValueError):
        cell_reward(make_grade(False, False), RewardCell.R0, p0, np.random.default_rng(0))


def test_cells_are_exactly_the_four_of_the_design():
    assert [cell.value for cell in RewardCell] == ["R0", "RF", "RC", "RFC"]
    with pytest.raises(ValueError):
        RewardCell("RCF")  # a typo cannot silently become a cell
