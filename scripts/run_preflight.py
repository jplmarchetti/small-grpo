"""Stage 7: pre-flight on validation (GPU PC). Writes manifests/preflight.json and the hand-check sheet.

    uv run --extra train python scripts/run_preflight.py

1. B0 and B0-fewshot, greedy, on all validation problems (per-problem rows to outputs/preflight/).
2. Sampled statistics on the first 200 validation problems, for both prompts (METHODOLOGY §13):
   p0, mixed groups for correctness and for format, truncation.
3. Go / no-go: few-shot in every cell if the zero-shot p0 is below the threshold, unless
   `preflight.fewshot_override` is set (METHODOLOGY §13); p0 is the trained prompt's value.
4. A 50-row hand-check sheet from the greedy completions of that prompt's B0.
"""

import time

import torch

from grpo_gsm8k.audit import draw_audit_sample, write_audit_sheet
from grpo_gsm8k.config import load_config
from grpo_gsm8k.data import load_split
from grpo_gsm8k.evaluate import evaluate_policy, summarize_predictions, write_predictions
from grpo_gsm8k.graders import grade_completion
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.policy import load_policy
from grpo_gsm8k.preflight import PREFLIGHT_PATH, build_preflight_manifest, sampled_statistics, save_preflight
from grpo_gsm8k.prompts import build_prompt
from grpo_gsm8k.rollout import greedy_decoding, sample_groups, sampling_decoding

OUTPUT_DIR = REPO_ROOT / "outputs" / "preflight"
AUDIT_DIR = REPO_ROOT / "audits"


def sample_and_grade(policy, tokenizer, problems, exemplars, config) -> dict:
    """Sample `group_size` completions per problem, grade them, return the sampled statistics."""
    group_size = config.training.group_size
    problems_per_call = max(1, config.evaluation.batch_size // group_size)
    decoding = sampling_decoding(config.training)
    torch.manual_seed(config.preflight.sampling_seed)
    group_grades, truncated = [], []
    started = time.time()
    for start in range(0, len(problems), problems_per_call):
        chunk = problems[start : start + problems_per_call]
        prompts = [build_prompt(problem.question, exemplars) for problem in chunk]
        for problem, group in zip(chunk, sample_groups(policy, tokenizer, prompts, group_size, decoding), strict=True):
            group_grades.append([grade_completion(c.text, problem.gold_answer) for c in group])
            truncated.append([c.truncated for c in group])
        print(f"sampled {len(group_grades)}/{len(problems)} problems, {time.time() - started:.0f}s", flush=True)
    return sampled_statistics(group_grades, truncated)


def main() -> None:
    config = load_config()
    validation = load_split("validation")
    exemplars = load_split("fewshot_exemplars")
    prompts_by_name = {"B0": (), "B0_fewshot": exemplars}

    policy, tokenizer = load_policy(config.model)
    policy.eval()
    greedy = greedy_decoding(config.training)

    greedy_rows, greedy_summaries = {}, {}
    for name, name_exemplars in prompts_by_name.items():
        print(f"\n== greedy {name} on {len(validation)} validation problems", flush=True)
        rows = evaluate_policy(policy, tokenizer, validation, greedy, config.evaluation.batch_size, name_exemplars)
        write_predictions(rows, OUTPUT_DIR / name / "predictions.jsonl")
        greedy_rows[name], greedy_summaries[name] = rows, summarize_predictions(rows)
        print(greedy_summaries[name], flush=True)

    sampled_problems = validation[: config.preflight.n_sampled_problems]
    sampled = {}
    for name, name_exemplars in prompts_by_name.items():
        print(f"\n== sampling {name}", flush=True)
        sampled[name] = sample_and_grade(policy, tokenizer, sampled_problems, name_exemplars, config)
        print(sampled[name], flush=True)

    warnings = [
        f"{name} {kind} truncation rate {rate:.1%} is above {config.preflight.truncation_alert:.0%}"
        for name in prompts_by_name
        for kind, rate in (("greedy", greedy_summaries[name]["truncation_rate"]), ("sampled", sampled[name]["truncation_rate"]))
        if rate > config.preflight.truncation_alert
    ]
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)
    manifest = build_preflight_manifest(
        sampled, greedy_summaries, config.preflight.p0_threshold, config.preflight.fewshot_override,
        warnings, config.model.revision,
    )
    save_preflight(manifest)
    trained_name = manifest["trained_prompt"]

    questions = {problem.problem_id: problem.question for problem in validation}
    audit_rows = draw_audit_sample(greedy_rows[trained_name], config.preflight.n_audit, config.preflight.audit_seed)
    audit_path = AUDIT_DIR / f"preflight_{trained_name}.csv"
    write_audit_sheet(audit_rows, questions, audit_path)
    print(f"\nwrote {PREFLIGHT_PATH} (p0={manifest['p0']:.3f}, trained prompt {trained_name}, rule overridden: {manifest['rule_overridden']})")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    main()
