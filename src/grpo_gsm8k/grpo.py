"""Stage 9: GRPO core, from rewards to gradients (METHODOLOGY §7).

- Advantage: each completion's reward compared with its group (mean and spread of the
  8 rewards for the same problem).
- Loss: REINFORCE with the group mean as baseline. Each completion token's
  log-probability is weighted by the completion's advantage; the sum is divided by
  the number of completion tokens in the whole step, so long answers do not pull harder.
- Memory: completions are scored in micro-batches and `backward()` is called on each.
  Every micro-batch is divided by the same step-wide token count, so the accumulated
  gradient equals the full-batch gradient.

The shift-and-gather in `completion_token_log_probs` and the advantage formula are
adapted from rasbt/reasoning-from-scratch (`sequence_logprob`, `compute_grpo_loss`
in ch06.py), Apache-2.0, Copyright Sebastian Raschka. Changes: per-token values with a
completion mask, right padding, token normalization, micro-batched backward.
"""

from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn.functional as F


class TokenizedCompletion(Protocol):
    """What the GRPO core needs from a completion: the exact token ids that were sampled."""

    prompt_ids: Sequence[int]
    completion_ids: Sequence[int]


# ---------------------------------------------------------------- advantages


def compute_group_advantages(rewards: torch.Tensor, eps: float) -> torch.Tensor:
    """(n_problems, group_size) rewards -> advantages of the same shape.

    Sample standard deviation (unbiased), as in PyTorch's default, RfS and TRL.
    A group whose rewards are all equal gets all-zero advantages.
    """
    if rewards.dim() != 2:
        raise ValueError(f"rewards must be (n_problems, group_size), got shape {tuple(rewards.shape)}")
    group_mean = rewards.mean(dim=1, keepdim=True)
    group_std = rewards.std(dim=1, keepdim=True)
    return (rewards - group_mean) / (group_std + eps)


# ---------------------------------------------------------------- log-probabilities


def pad_prompt_and_completion(
    completions: Sequence[TokenizedCompletion], pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Join prompt + completion per row and pad on the LEFT, so every completion ends at the last position.

    Returns input_ids, attention_mask, position_ids and completion_mask, all (batch, length).
    completion_mask is 1 exactly on the completion tokens. Position ids count from 0 at each
    row's first real token, as during generation, so padding does not shift positions.
    """
    rows = [list(c.prompt_ids) + list(c.completion_ids) for c in completions]
    length = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
    completion_mask = torch.zeros((len(rows), length), dtype=torch.float32)
    for index, (row, completion) in enumerate(zip(rows, completions, strict=True)):
        input_ids[index, length - len(row) :] = torch.tensor(row)
        attention_mask[index, length - len(row) :] = 1
        completion_mask[index, length - len(completion.completion_ids) :] = 1.0
    position_ids = (attention_mask.cumsum(dim=1) - 1).clamp(min=0)
    return input_ids.to(device), attention_mask.to(device), position_ids.to(device), completion_mask.to(device)


def completion_token_log_probs(
    policy: torch.nn.Module, completions: Sequence[TokenizedCompletion], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probability of every sampled completion token, with gradients.

    Position t predicts token t+1. Because completions end at the last position (left
    padding), only the last `longest completion + 1` positions need logits: computing the
    full-vocabulary logits for the long few-shot prompt would cost several GB for nothing.
    Returns (log_probs, completion_mask), both (batch, longest completion); the mask is 0
    where a shorter row has prompt or padding.
    """
    device = next(policy.parameters()).device
    input_ids, attention_mask, position_ids, completion_mask = pad_prompt_and_completion(
        completions, pad_token_id, device
    )
    n_scored = max(len(c.completion_ids) for c in completions)
    logits = policy(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, logits_to_keep=n_scored + 1
    ).logits
    next_token_logits = logits[:, :-1, :].float()  # positions L-n-1 .. L-2 predict tokens L-n .. L-1
    next_tokens = input_ids[:, -n_scored:]
    log_probs = F.log_softmax(next_token_logits, dim=-1).gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)
    return log_probs, completion_mask[:, -n_scored:]


# ---------------------------------------------------------------- loss and gradient


def policy_gradient_loss(
    log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    total_completion_tokens: int,
) -> torch.Tensor:
    """-(sum over completions of advantage x sum of its token log-probs) / tokens in the step."""
    per_completion_log_prob = (log_probs * completion_mask).sum(dim=1)
    return -(advantages * per_completion_log_prob).sum() / total_completion_tokens


def accumulate_policy_gradient(
    policy: torch.nn.Module,
    completions: Sequence[TokenizedCompletion],
    advantages: torch.Tensor,
    micro_batch_size: int,
    pad_token_id: int,
) -> float:
    """Add the step's policy gradient to `.grad` of the policy parameters; return the loss.

    `advantages` is flat, one per completion. Completions with advantage 0 add nothing
    to the gradient and are not scored, but their tokens still count in the divisor.
    The caller clips, steps the optimizer and zeroes the gradients.
    """
    if len(completions) != advantages.numel():
        raise ValueError(f"{len(completions)} completions but {advantages.numel()} advantages")
    total_completion_tokens = sum(len(c.completion_ids) for c in completions)
    if total_completion_tokens == 0:
        return 0.0
    contributing = [index for index, advantage in enumerate(advantages.tolist()) if advantage != 0.0]
    step_loss = 0.0
    for start in range(0, len(contributing), micro_batch_size):
        batch_indices = contributing[start : start + micro_batch_size]
        log_probs, completion_mask = completion_token_log_probs(
            policy, [completions[i] for i in batch_indices], pad_token_id
        )
        micro_advantages = advantages[batch_indices].to(log_probs.device, log_probs.dtype)
        micro_loss = policy_gradient_loss(log_probs, completion_mask, micro_advantages, total_completion_tokens)
        micro_loss.backward()
        step_loss += micro_loss.item()
    return step_loss
