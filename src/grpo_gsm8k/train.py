"""Stage 10: the GRPO training step and the training loop for one run (one cell x one seed).

- One step: take 2 problems, sample 8 completions each, grade them, turn grades into the
  cell's reward, compare within groups, and nudge the adapter (stages 5, 3, 8, 9 in order).
- The loop adds a fixed problem order per seed, one log row per step, sample completions
  every 10 steps, and every 50 steps a saved adapter plus a greedy validation evaluation.
- Everything that differs between cells is `cell_reward`. The prompt (few-shot or not) and
  p0 come from manifests/preflight.json.
- Resume (agreed 2026-11-14): restart from the latest checkpoint with the same problem
  order, adapter and optimizer state. Random states are restored too, but GPU sampling
  is not guaranteed to replay exactly; every resume is written to `resumes.log`.

The step order and the save-on-interrupt pattern follow `compute_grpo_loss` and
`train_rlvr_grpo` in rasbt/reasoning-from-scratch (Apache-2.0, Copyright Sebastian Raschka).
"""

import csv
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from grpo_gsm8k.config import ExperimentConfig
from grpo_gsm8k.data import Problem, load_split
from grpo_gsm8k.evaluate import evaluate_policy, summarize_predictions, write_predictions
from grpo_gsm8k.graders import Grade, grade_completion
from grpo_gsm8k.grpo import accumulate_policy_gradient, compute_group_advantages
from grpo_gsm8k.policy import load_policy, save_adapter
from grpo_gsm8k.preflight import load_preflight
from grpo_gsm8k.prompts import build_prompt
from grpo_gsm8k.rewards import RewardCell, cell_reward
from grpo_gsm8k.rollout import greedy_decoding, sample_groups, sampling_decoding
from grpo_gsm8k.run_log import StepLog, append_sample_completions, append_step_log, truncate_step_log

SAMPLES_EVERY = 10
N_SAMPLES_LOGGED = 3
VAL_CURVE_COLUMNS = ("step", "lenient_accuracy", "strict_accuracy", "format_rate", "mean_tokens", "truncation_rate")


@dataclass(frozen=True)
class RunSpec:
    cell: RewardCell
    seed: int
    output_dir: Path
    steps: int
    learning_rate: float
    validate: bool = True


@dataclass
class RunContext:
    """What every step needs; built once per run."""

    config: ExperimentConfig
    spec: RunSpec
    policy: torch.nn.Module
    tokenizer: object
    optimizer: torch.optim.Optimizer
    problem_order: list[Problem]
    exemplars: Sequence[Problem]
    p0: float
    reward_rng: np.random.Generator


# ---------------------------------------------------------------- pure helpers


def make_problem_order(problems: Sequence[Problem], seed: int) -> list[Problem]:
    """Depends only on the seed, so all four cells of a seed see the same problems at the same steps."""
    order = list(problems)
    random.Random(seed).shuffle(order)
    return order


def problems_for_step(problem_order: Sequence[Problem], step: int, problems_per_step: int) -> list[Problem]:
    start = (step - 1) * problems_per_step
    return list(problem_order[start : start + problems_per_step])


def classify_signal_source(group_grades: Sequence[Grade]) -> str:
    """Where a group's possible signal comes from: its completions differ in format, correctness, both or neither."""
    format_varies = len({grade.format_ok for grade in group_grades}) > 1
    correctness_varies = len({grade.lenient_correct for grade in group_grades}) > 1
    if format_varies and correctness_varies:
        return "both"
    if format_varies:
        return "format_only"
    if correctness_varies:
        return "correctness_only"
    return "none"


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


# ---------------------------------------------------------------- one step


def train_step(context: RunContext, step: int) -> tuple[StepLog, list[tuple[str, str, float]]]:
    config, spec = context.config, context.spec
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    problems = problems_for_step(context.problem_order, step, config.training.problems_per_step)
    prompts = [build_prompt(problem.question, context.exemplars) for problem in problems]
    context.policy.eval()
    groups = sample_groups(
        context.policy, context.tokenizer, prompts, config.training.group_size, sampling_decoding(config.training)
    )
    group_grades = [
        [grade_completion(completion.text, problem.gold_answer) for completion in group]
        for problem, group in zip(problems, groups, strict=True)
    ]
    rewards = [
        [cell_reward(grade, spec.cell, context.p0, context.reward_rng) for grade in grades] for grades in group_grades
    ]
    advantages = compute_group_advantages(torch.tensor(rewards), config.training.advantage_eps)

    completions = [completion for group in groups for completion in group]
    flat_advantages = advantages.flatten()
    skipped = bool((flat_advantages == 0).all())
    loss, grad_norm = 0.0, 0.0
    if not skipped:  # no signal, no change: AdamW momentum would otherwise still move the weights
        context.policy.train()
        loss = accumulate_policy_gradient(
            context.policy, completions, flat_advantages, config.training.micro_batch_size,
            context.tokenizer.pad_token_id,
        )
        trainable = [p for p in context.policy.parameters() if p.requires_grad]
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, config.training.grad_clip_norm))
        context.optimizer.step()
        context.optimizer.zero_grad(set_to_none=True)

    flat_grades = [grade for grades in group_grades for grade in grades]
    sources = [classify_signal_source(grades) for grades in group_grades]
    log = StepLog(
        step=step,
        mean_reward=mean([r for group in rewards for r in group]),
        format_rate=mean([g.format_ok for g in flat_grades]),
        lenient_rate=mean([g.lenient_correct for g in flat_grades]),
        strict_rate=mean([g.strict_correct for g in flat_grades]),
        mean_tokens=mean([c.n_tokens for c in completions]),
        truncation_rate=mean([c.truncated for c in completions]),
        zero_variance_groups=mean([len(set(group)) == 1 for group in rewards]),
        groups_format_only=sources.count("format_only"),
        groups_correctness_only=sources.count("correctness_only"),
        groups_both=sources.count("both"),
        groups_none=sources.count("none"),
        skipped_update=skipped,
        loss=loss,
        grad_norm=grad_norm,
        seconds=time.time() - started,
        peak_vram_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
    )
    sample_texts = [group[0].text for group in groups]
    sample_rewards = [group_rewards[0] for group_rewards in rewards]
    return log, list(zip([p.question for p in problems], sample_texts, sample_rewards))


# ---------------------------------------------------------------- checkpoints


def checkpoint_dir(output_dir: Path, step: int) -> Path:
    return output_dir / "checkpoints" / f"step_{step:03d}"


def save_checkpoint(context: RunContext, step: int) -> Path:
    directory = checkpoint_dir(context.spec.output_dir, step)
    save_adapter(context.policy, directory)
    torch.save(
        {
            "step": step,
            "optimizer": context.optimizer.state_dict(),
            "reward_rng": context.reward_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        directory / "training_state.pt",
    )
    return directory


def latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted((output_dir / "checkpoints").glob("step_*/training_state.pt"))
    return candidates[-1].parent if candidates else None


def restore_training_state(context: RunContext, directory: Path) -> int:
    state = torch.load(directory / "training_state.pt", weights_only=False)
    context.optimizer.load_state_dict(state["optimizer"])
    context.reward_rng.bit_generator.state = state["reward_rng"]
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["step"])


# ---------------------------------------------------------------- validation


def validate(context: RunContext, validation: Sequence[Problem], step: int) -> dict:
    context.policy.eval()
    rows = evaluate_policy(
        context.policy, context.tokenizer, validation, greedy_decoding(context.config.training),
        context.config.evaluation.batch_size, context.exemplars, show_progress=False,
    )
    write_predictions(rows, context.spec.output_dir / "val_predictions" / f"step_{step:03d}.jsonl")
    summary = summarize_predictions(rows)
    append_val_curve(context.spec.output_dir / "val_curve.csv", step, summary)
    return summary


def append_val_curve(path: Path, step: int, summary: dict) -> None:
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if int(row["step"]) != step]
    rows.append({"step": step, **{column: summary[column] for column in VAL_CURVE_COLUMNS[1:]}})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=VAL_CURVE_COLUMNS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["step"])))


# ---------------------------------------------------------------- the loop


def build_context(config: ExperimentConfig, spec: RunSpec) -> tuple[RunContext, int]:
    """Load (or resume) the policy and everything a step needs; return the context and the last finished step."""
    preflight = load_preflight()
    exemplars = load_split("fewshot_exemplars") if preflight["use_fewshot_prompt"] else ()
    resume_from = latest_checkpoint(spec.output_dir)
    torch.manual_seed(spec.seed)
    if resume_from is None:
        policy, tokenizer = load_policy(config.model, config.lora, lora_seed=spec.seed)
    else:
        policy, tokenizer = load_policy(config.model, adapter_dir=resume_from)
    # Recompute activations in the backward pass instead of storing them: same gradient,
    # roughly half the memory for 4 sequences of ~1,000 tokens (the few-shot prompt is long).
    # Only active in train mode; generation runs in eval mode and keeps its KV cache.
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=spec.learning_rate, weight_decay=config.training.weight_decay)
    context = RunContext(
        config=config,
        spec=spec,
        policy=policy,
        tokenizer=tokenizer,
        optimizer=optimizer,
        problem_order=make_problem_order(load_split("train"), spec.seed),
        exemplars=exemplars,
        p0=preflight["p0"],
        reward_rng=np.random.default_rng(spec.seed),
    )
    last_step = 0
    if resume_from is not None:
        last_step = restore_training_state(context, resume_from)
        dropped = truncate_step_log(spec.output_dir / "train_log.csv", last_step)
        with (spec.output_dir / "resumes.log").open("a") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} resumed from step {last_step}, dropped {dropped} log rows\n")
        print(f"resumed from {resume_from} (step {last_step})", flush=True)
    return context, last_step


def run_training(config: ExperimentConfig, spec: RunSpec) -> None:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    context, last_step = build_context(config, spec)
    validation = load_split("validation") if spec.validate else []
    every = config.training.checkpoint_every
    if spec.validate and last_step == 0:
        print(f"step 0 validation: {validate(context, validation, 0)}", flush=True)

    completed_step = last_step
    try:
        for step in range(last_step + 1, spec.steps + 1):
            log, samples = train_step(context, step)
            append_step_log(spec.output_dir / "train_log.csv", log)
            completed_step = step
            print(
                f"step {step}: reward {log.mean_reward:.3f} lenient {log.lenient_rate:.3f} format {log.format_rate:.3f} "
                f"tokens {log.mean_tokens:.0f} loss {log.loss:+.4f} {log.seconds:.1f}s {log.peak_vram_gb:.1f}GB"
                + (" (skipped)" if log.skipped_update else ""),
                flush=True,
            )
            if step % SAMPLES_EVERY == 0:
                questions, texts, rewards = zip(*samples[:N_SAMPLES_LOGGED])
                append_sample_completions(spec.output_dir / "samples.txt", step, questions, texts, rewards)
            if step % every == 0 or step == spec.steps:
                save_checkpoint(context, step)
                if spec.validate and step % every == 0:
                    print(f"step {step} validation: {validate(context, validation, step)}", flush=True)
    except KeyboardInterrupt:
        if completed_step > last_step:  # the step in progress may be half-applied, so save the last finished one
            print(f"interrupted, saving checkpoint at step {completed_step}", flush=True)
            save_checkpoint(context, completed_step)
        raise
