"""Stage 3: grade one completion with the format, strict and lenient rules (METHODOLOGY §5).

This file is the single source of the three scores. It is the training reward, the
validation metric and the final result, so it is frozen with the git tag `grader-v1`.
Do not edit it after that without asking.

The "last number in the text" fallback is adapted from rasbt/reasoning-from-scratch
(`RE_NUMBER`, `extract_final_candidate`), Apache-2.0, Copyright Sebastian Raschka.
Changes: accepts thousands separators (`1,000`), does not parse fractions.
"""

import re
from dataclasses import dataclass
from decimal import Decimal

FINAL_PREFIX = "FINAL:"

# A number as it may appear in text: optional minus (not right after a digit, so
# "10-15" is two numbers), optional `$`, digits with optional `1,000` grouping and decimals.
NUMBER_IN_TEXT = re.compile(r"(?<!\d)-?\$?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
# A FINAL line that holds exactly one number, optionally decorated with `$`, `%` or a trailing dot.
FORMATTED_FINAL_LINE = re.compile(r"FINAL:\s*(-?\$?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?\.?)")
# What a number must look like once `,`, `$`, `%` and a trailing dot are removed.
PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class Grade:
    format_ok: bool  # F
    strict_correct: bool  # C_s
    lenient_correct: bool  # C_l
    predicted_number: str | None  # the lenient candidate, as written in the completion


def normalize_number(text: str) -> Decimal | None:
    """`$1,000.` -> Decimal('1000'); anything that is not a plain number -> None."""
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "").removesuffix(".")
    if not PLAIN_NUMBER.fullmatch(cleaned):
        return None
    return Decimal(cleaned)


def numbers_match(predicted: str, gold: str) -> bool:
    """Exact decimal equality after normalization: 5, 5.0 and 5.00 match."""
    predicted_value, gold_value = normalize_number(predicted), normalize_number(gold)
    return predicted_value is not None and predicted_value == gold_value


def non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def find_final_lines(text: str) -> list[str]:
    return [line for line in non_empty_lines(text) if line.startswith(FINAL_PREFIX)]


def last_number_in(text: str) -> str | None:
    matches = NUMBER_IN_TEXT.findall(text)
    return matches[-1] if matches else None


def formatted_final_number(text: str) -> str | None:
    """The number of a well-formatted answer, or None when F = 0.

    F = 1 when the last non-empty line is `FINAL: <one number>` and no other
    FINAL line appears.
    """
    lines = non_empty_lines(text)
    if not lines or len(find_final_lines(text)) != 1:
        return None
    match = FORMATTED_FINAL_LINE.fullmatch(lines[-1])
    return match.group(1) if match else None


def lenient_candidate(text: str) -> str | None:
    """The number on the single FINAL line if there is one; else the last number in the text."""
    final_lines = find_final_lines(text)
    if len(final_lines) == 1:
        number_on_final_line = last_number_in(final_lines[0])
        if number_on_final_line is not None:
            return number_on_final_line
    return last_number_in(text)


def grade_completion(text: str, gold_answer: str) -> Grade:
    final_number = formatted_final_number(text)
    candidate = lenient_candidate(text)
    format_ok = final_number is not None
    return Grade(
        format_ok=format_ok,
        strict_correct=format_ok and numbers_match(final_number, gold_answer),
        lenient_correct=candidate is not None and numbers_match(candidate, gold_answer),
        predicted_number=candidate,
    )
