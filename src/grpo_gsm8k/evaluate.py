"""Stage 6: measure a model on a set of problems with greedy decoding and both graders.

- One evaluator for B0, B0-fewshot, the validation curve during training and the final
  test, so a gain can never come from the evaluation changing.
- Prompts are batched by length (less padding, faster) with a fixed batch size from
  `configs/base.yaml`, so every model is evaluated with exactly the same batches.
- Returns one row per problem, in the original order; only scripts write files.
"""

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from grpo_gsm8k.data import Problem
from grpo_gsm8k.graders import grade_completion
from grpo_gsm8k.prompts import build_prompt
from grpo_gsm8k.rollout import Completion, DecodingConfig, generate_completions

GenerateFn = Callable[[Sequence[str]], list[Completion]]


def batches_by_length(prompts: Sequence[str], batch_size: int) -> list[list[int]]:
    """Indices of `prompts` grouped into batches of similar length (longest first, stable)."""
    order = sorted(range(len(prompts)), key=lambda index: (-len(prompts[index]), index))
    return [order[start : start + batch_size] for start in range(0, len(order), batch_size)]


def build_prediction_row(problem: Problem, completion: Completion) -> dict:
    grade = grade_completion(completion.text, problem.gold_answer)
    return {
        "problem_id": problem.problem_id,
        "gold_answer": problem.gold_answer,
        "completion": completion.text,
        "predicted_number": grade.predicted_number,
        "format_ok": grade.format_ok,
        "strict_correct": grade.strict_correct,
        "lenient_correct": grade.lenient_correct,
        "n_tokens": completion.n_tokens,
        "truncated": completion.truncated,
    }


def eta_progress_message(done: int, total: int, started_at: float) -> str:
    """Adapted from `eta_progress_message` in rasbt/reasoning-from-scratch (Apache-2.0)."""
    elapsed = time.time() - started_at
    remaining = elapsed / done * (total - done) if done else 0.0
    return f"{done}/{total} problems, {elapsed:.0f}s elapsed, ~{remaining:.0f}s left"


def evaluate_with(
    generate: GenerateFn,
    problems: Sequence[Problem],
    batch_size: int,
    exemplars: Sequence[Problem] = (),
    show_progress: bool = True,
) -> list[dict]:
    """Core loop with generation passed in, so it is testable without a model."""
    prompts = [build_prompt(problem.question, exemplars) for problem in problems]
    rows: list[dict | None] = [None] * len(problems)
    started_at, done = time.time(), 0
    for batch_indices in batches_by_length(prompts, batch_size):
        completions = generate([prompts[index] for index in batch_indices])
        for index, completion in zip(batch_indices, completions, strict=True):
            rows[index] = build_prediction_row(problems[index], completion)
        done += len(batch_indices)
        if show_progress:
            print(eta_progress_message(done, len(problems), started_at), flush=True)
    return rows


def evaluate_policy(
    policy,
    tokenizer,
    problems: Sequence[Problem],
    decoding: DecodingConfig,
    batch_size: int,
    exemplars: Sequence[Problem] = (),
    show_progress: bool = True,
) -> list[dict]:
    if decoding.do_sample:
        raise ValueError("evaluation is greedy; pass greedy_decoding(...)")

    def generate(prompts: Sequence[str]) -> list[Completion]:
        return generate_completions(policy, tokenizer, prompts, decoding)

    return evaluate_with(generate, problems, batch_size, exemplars, show_progress)


def summarize_predictions(rows: Sequence[dict]) -> dict:
    n = len(rows)
    return {
        "n_problems": n,
        "lenient_accuracy": sum(row["lenient_correct"] for row in rows) / n,
        "strict_accuracy": sum(row["strict_correct"] for row in rows) / n,
        "format_rate": sum(row["format_ok"] for row in rows) / n,
        "mean_tokens": sum(row["n_tokens"] for row in rows) / n,
        "truncation_rate": sum(row["truncated"] for row in rows) / n,
    }


def write_predictions(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
