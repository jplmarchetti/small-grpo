"""The grader is the reward, the validation metric and the final result. Test it hard."""

from decimal import Decimal

import pytest

from grpo_gsm8k.graders import (
    Grade,
    find_final_lines,
    grade_completion,
    last_number_in,
    normalize_number,
    numbers_match,
)

# ---------------------------------------------------------------- normalize_number


@pytest.mark.parametrize(
    "text, expected",
    [
        ("18", Decimal("18")),
        ("1,000", Decimal("1000")),
        ("$1,000.", Decimal("1000")),
        ("50%", Decimal("50")),
        ("-4", Decimal("-4")),
        ("$-4", Decimal("-4")),
        ("-$4", Decimal("-4")),
        ("2.50", Decimal("2.5")),
        (" 7 ", Decimal("7")),
    ],
)
def test_normalize_number(text, expected):
    assert normalize_number(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "Infinity", "NaN", "3/4", "1e5", "--3", "."])
def test_normalize_number_rejects_non_numbers(text):
    assert normalize_number(text) is None


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("5", "5.0", True),
        ("5", "5.00", True),
        ("1000", "1,000", True),
        ("$18.", "18", True),
        ("18", "18.5", False),
        ("-3", "3", False),
        ("abc", "abc", False),  # nothing numeric never matches
    ],
)
def test_numbers_match(a, b, expected):
    assert numbers_match(a, b) is expected


# ---------------------------------------------------------------- helpers


def test_find_final_lines_only_counts_lines_starting_with_final():
    text = "Step 1\nFINAL: 3\nsee FINAL: 4\n  FINAL: 5  \nFinal: 6"
    assert find_final_lines(text) == ["FINAL: 3", "FINAL: 5"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("so she has 18 dollars left.", "18"),
        ("a total of 1,000 apples", "1,000"),
        ("from 1,2,3 we pick 3", "3"),
        ("pages 10-15", "15"),  # a hyphen after a digit is not a minus sign
        ("it is -7 degrees", "-7"),
        ("cost $2.50.", "$2.50"),
        ("3/4 of them", "4"),  # fractions are not parsed as one number
        ("no digits here", None),
        ("", None),
    ],
)
def test_last_number_in(text, expected):
    assert last_number_in(text) == expected


# ---------------------------------------------------------------- grade_completion


def grade(text: str, gold: str = "18") -> tuple[bool, bool, bool]:
    result = grade_completion(text, gold)
    return result.format_ok, result.strict_correct, result.lenient_correct


def test_methodology_worked_example():
    """METHODOLOGY §5: right answer in the wrong place."""
    assert grade("She spends 2 dollars, so she has 18 dollars left.") == (False, False, True)


def test_perfect_completion():
    assert grade("20 - 2 = 18\nFINAL: 18") == (True, True, True)


def test_formatted_but_wrong():
    assert grade("20 - 3 = 17\nFINAL: 17") == (True, False, False)


def test_decorated_number_on_final_line_is_formatted():
    assert grade("...\nFINAL: $18.") == (True, True, True)
    assert grade("...\nFINAL: 1,000", gold="1000") == (True, True, True)


def test_trailing_blank_lines_and_spaces_are_ignored():
    assert grade("work\nFINAL: 18\n\n   \n") == (True, True, True)


def test_final_line_must_be_last():
    """A lone FINAL line not at the end: F=0, but the lenient grader still reads it."""
    assert grade("FINAL: 18\nThis is because 20 - 2 = 18.") == (False, False, True)
    assert grade("FINAL: 18\nI made 5 mistakes.") == (False, False, True)


def test_two_final_lines_fail_format_and_lenient_uses_last_number():
    assert grade("FINAL: 17\nFINAL: 18") == (False, False, True)
    assert grade("FINAL: 18\nFINAL: 17") == (False, False, False)


def test_final_line_with_extra_words_fails_format():
    assert grade("...\nFINAL: 18 dollars") == (False, False, True)


def test_final_line_with_two_numbers_fails_format():
    """Lenient takes the last number on the single FINAL line."""
    assert grade("...\nFINAL: 20 - 2 = 18") == (False, False, True)
    assert grade("...\nFINAL: 18 or 19") == (False, False, False)


def test_final_line_without_number_falls_back_to_last_number_in_text():
    assert grade("the answer is 18\nFINAL: eighteen") == (False, False, True)


def test_final_is_case_sensitive():
    assert grade("...\nFinal: 18") == (False, False, True)


def test_no_space_after_colon_is_accepted():
    assert grade("...\nFINAL:18") == (True, True, True)


def test_markdown_decoration_is_not_format():
    assert grade("...\n**FINAL: 18**") == (False, False, True)


def test_decimal_gold_answers():
    assert grade("...\nFINAL: 2.50", gold="2.5") == (True, True, True)
    assert grade("...\nFINAL: -4", gold="-4") == (True, True, True)


def test_empty_and_numberless_completions_score_zero():
    assert grade("") == (False, False, False)
    assert grade("I don't know.") == (False, False, False)


def test_predicted_number_is_the_lenient_candidate():
    assert grade_completion("so 18 left", "18").predicted_number == "18"
    assert grade_completion("FINAL: 17\nthen 3 more", "18").predicted_number == "17"
    assert grade_completion("nothing", "18").predicted_number is None


def test_grade_is_three_booleans():
    result = grade_completion("FINAL: 18", "18")
    assert result == Grade(format_ok=True, strict_correct=True, lenient_correct=True, predicted_number="18")


def test_strict_implies_format_and_lenient():
    """C_s = 1 only when F = 1 and C_l = 1, on every example above."""
    texts = [
        "so 18 left", "FINAL: 18", "FINAL: 17", "FINAL: 18\nok 3", "FINAL: 17\nFINAL: 18",
        "...\nFINAL: 18 dollars", "", "x\nFINAL:18", "FINAL: 18 or 19",
    ]
    for text in texts:
        result = grade_completion(text, "18")
        assert result.strict_correct == (result.format_ok and result.lenient_correct), text
