"""Stage 2: the one plain-text prompt used by every model and every cell (METHODOLOGY §4).

Few-shot is the same prompt with solved examples inserted before the real question,
so the two versions cannot drift apart.
"""

import re
from collections.abc import Sequence

from grpo_gsm8k.data import GOLD_ANSWER_MARKER, Problem

PROMPT_PREAMBLE = (
    "A conversation between User and Assistant. The User asks a math question and the\n"
    "Assistant solves it step by step, then writes the final answer on its own last\n"
    'line in the form "FINAL: <number>".'
)
STOP_STRING = "\nUser:"
TURN_SEPARATOR = "\n\n"

CALCULATOR_NOTE = re.compile(r"<<[^>]*>>")  # GSM8K annotations such as <<5*3=15>>


def format_turn(question: str, answer: str = "") -> str:
    """`User: <question>` then `Assistant:` (with the answer after a space, if any)."""
    assistant_line = f"Assistant: {answer}" if answer else "Assistant:"
    return f"User: {question.strip()}\n{assistant_line}"


def format_exemplar(problem: Problem) -> str:
    """The GSM8K solution rewritten as the answer we want: no calculator notes, ends in `FINAL: <n>`."""
    reasoning = problem.solution.rsplit(GOLD_ANSWER_MARKER, 1)[0]
    reasoning = CALCULATOR_NOTE.sub("", reasoning).strip()
    return f"{reasoning}\nFINAL: {problem.gold_answer}"


def build_prompt(question: str, exemplars: Sequence[Problem] = ()) -> str:
    solved_turns = [format_turn(exemplar.question, format_exemplar(exemplar)) for exemplar in exemplars]
    return TURN_SEPARATOR.join([PROMPT_PREAMBLE, *solved_turns, format_turn(question)])
