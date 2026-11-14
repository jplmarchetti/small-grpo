"""Stage 7: pre-flight. Where does the base model start, and which prompt is trained?

- p0: the chance that one sampled B0 completion is lenient-correct. R0's coin flip uses it.
- Mixed groups: GRPO only learns from groups whose 8 rewards are not all equal. We count
  them for the correctness signal (RC) and for the format signal (RF).
- Go / no-go (METHODOLOGY §11): if the zero-shot p0 is below the threshold, every cell
  uses the few-shot prompt, and p0 is the few-shot value (judgment call 7).
- The result is one committed file, manifests/preflight.json, read by every later run.
"""

import json
from collections.abc import Sequence
from pathlib import Path

from grpo_gsm8k.graders import Grade
from grpo_gsm8k.paths import MANIFEST_DIR

PREFLIGHT_PATH = MANIFEST_DIR / "preflight.json"
GroupGrades = Sequence[Sequence[Grade]]  # problems x group_size


def estimate_base_success_rate(group_grades: GroupGrades) -> float:
    """Mean lenient score over every sampled completion."""
    scores = [grade.lenient_correct for group in group_grades for grade in group]
    return sum(scores) / len(scores)


def mixed_group_fraction(group_grades: GroupGrades, score: str = "lenient_correct") -> float:
    """Share of groups whose completions do not all have the same `score`."""
    mixed = sum(len({getattr(grade, score) for grade in group}) > 1 for group in group_grades)
    return mixed / len(group_grades)


def sampled_statistics(group_grades: GroupGrades, truncated: Sequence[Sequence[bool]]) -> dict:
    n_completions = sum(len(group) for group in group_grades)
    return {
        "n_problems": len(group_grades),
        "group_size": len(group_grades[0]),
        "p0": estimate_base_success_rate(group_grades),
        "format_rate": sum(g.format_ok for group in group_grades for g in group) / n_completions,
        "strict_rate": sum(g.strict_correct for group in group_grades for g in group) / n_completions,
        "mixed_group_fraction": mixed_group_fraction(group_grades, "lenient_correct"),
        "format_mixed_group_fraction": mixed_group_fraction(group_grades, "format_ok"),
        "truncation_rate": sum(flag for row in truncated for flag in row) / n_completions,
    }


def decide_fewshot_prompt(zero_shot_p0: float, threshold: float) -> bool:
    return zero_shot_p0 < threshold


def build_preflight_manifest(
    sampled: dict[str, dict],
    greedy_summaries: dict[str, dict],
    threshold: float,
    fewshot_override: bool | None,
    warnings: list[str],
    model_revision: str,
) -> dict:
    """Apply the go / no-go rule, then any documented override; p0 is the trained prompt's value."""
    rule_says_fewshot = decide_fewshot_prompt(sampled["B0"]["p0"], threshold)
    use_fewshot = rule_says_fewshot if fewshot_override is None else fewshot_override
    trained_prompt = "B0_fewshot" if use_fewshot else "B0"
    return {
        "p0": sampled[trained_prompt]["p0"],
        "use_fewshot_prompt": use_fewshot,
        "trained_prompt": trained_prompt,
        "rule_says_fewshot": rule_says_fewshot,
        "rule_overridden": use_fewshot != rule_says_fewshot,
        "p0_threshold": threshold,
        "sampled": sampled,
        "greedy_validation": greedy_summaries,
        "warnings": warnings,
        "model_revision": model_revision,
    }


def save_preflight(content: dict, path: Path = PREFLIGHT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def load_preflight(path: Path = PREFLIGHT_PATH) -> dict:
    """p0 and use_fewshot_prompt come from here, never from code."""
    content = json.loads(path.read_text())
    if not 0.0 < content["p0"] < 1.0:
        raise ValueError(f"p0 must be in (0, 1), got {content['p0']}")
    return content
