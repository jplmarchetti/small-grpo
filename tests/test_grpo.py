"""GRPO core on a tiny random Qwen2 model (no download). Needs the `train` extra, so it
runs on the GPU PC (`pcw run -- uv run --extra train pytest tests/test_grpo.py`) and is
skipped on the laptop."""

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from grpo_gsm8k.grpo import (  # noqa: E402
    accumulate_policy_gradient,
    completion_token_log_probs,
    compute_group_advantages,
    pad_prompt_and_completion,
    policy_gradient_loss,
)

PAD_TOKEN_ID = 0
EPS = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class FakeCompletion:
    prompt_ids: list[int]
    completion_ids: list[int]


@pytest.fixture
def tiny_policy():
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    # float32 and eager attention so gradient comparisons are tight
    model = transformers.Qwen2ForCausalLM(config).to(DEVICE, torch.float32)
    model.config._attn_implementation = "eager"
    return model


def make_completions(n: int, seed: int = 1) -> list[FakeCompletion]:
    generator = torch.Generator().manual_seed(seed)
    completions = []
    for _ in range(n):
        prompt_length = int(torch.randint(3, 8, (1,), generator=generator))
        completion_length = int(torch.randint(1, 10, (1,), generator=generator))
        ids = torch.randint(1, 64, (prompt_length + completion_length,), generator=generator).tolist()
        completions.append(FakeCompletion(ids[:prompt_length], ids[prompt_length:]))
    return completions


def flat_gradient(model) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


# ---------------------------------------------------------------- advantages


def test_worked_example_one_right_among_eight():
    """Blueprint stage 9: one correct answer gets +2.47, each wrong one -0.35."""
    rewards = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0]])
    advantages = compute_group_advantages(rewards, EPS)
    assert advantages[0, 0].item() == pytest.approx(2.47, abs=0.01)
    assert advantages[0, 1:].tolist() == pytest.approx([-0.35] * 7, abs=0.01)


def test_equal_rewards_give_zero_advantages():
    rewards = torch.tensor([[0.0] * 8, [1.0] * 8, [2.0] * 8])
    assert torch.equal(compute_group_advantages(rewards, EPS), torch.zeros(3, 8))


def test_groups_are_normalized_independently():
    rewards = torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0], [2.0, 1, 1, 1, 1, 1, 1, 1]])
    advantages = compute_group_advantages(rewards, EPS)
    assert torch.allclose(advantages[0], advantages[1])  # shifting a group's rewards changes nothing
    assert advantages.sum(dim=1).abs().max() < 1e-5  # each group's advantages sum to zero


def test_advantages_reject_flat_rewards():
    with pytest.raises(ValueError):
        compute_group_advantages(torch.zeros(8), EPS)


# ---------------------------------------------------------------- masks and log-probs


def test_completion_mask_covers_exactly_the_completion_tokens():
    """Left padding: every completion ends at the last position."""
    completions = [FakeCompletion([5, 6, 7], [8, 9]), FakeCompletion([5], [8, 9, 10, 11])]
    input_ids, attention_mask, position_ids, completion_mask = pad_prompt_and_completion(completions, PAD_TOKEN_ID, "cpu")
    assert input_ids.tolist() == [[5, 6, 7, 8, 9], [5, 8, 9, 10, 11]]
    assert completion_mask.tolist() == [[0, 0, 0, 1, 1], [0, 1, 1, 1, 1]]
    short = [FakeCompletion([5], [6])] + completions
    input_ids, attention_mask, position_ids, completion_mask = pad_prompt_and_completion(short, PAD_TOKEN_ID, "cpu")
    assert input_ids[0].tolist() == [0, 0, 0, 5, 6]
    assert completion_mask[0].tolist() == [0, 0, 0, 0, 1]  # padding is never a completion token
    assert attention_mask[0].tolist() == [0, 0, 0, 1, 1]
    assert position_ids[0].tolist() == [0, 0, 0, 0, 1]  # positions start at the first real token


def test_log_probs_match_a_manual_full_forward(tiny_policy):
    """Scoring only the completion positions gives the same values as full logits."""
    completion = FakeCompletion([3, 4, 5], [6, 7])
    log_probs, mask = completion_token_log_probs(tiny_policy, [completion], PAD_TOKEN_ID)
    ids = torch.tensor([[3, 4, 5, 6, 7]], device=DEVICE)
    all_log_probs = torch.log_softmax(tiny_policy(input_ids=ids).logits.float(), dim=-1)
    expected = [all_log_probs[0, 2, 6], all_log_probs[0, 3, 7]]  # token at t+1 predicted at t
    assert mask.tolist() == [[1, 1]]
    assert torch.allclose(log_probs[0], torch.stack(expected), atol=1e-6)


def test_padding_does_not_change_a_row(tiny_policy):
    """A row scored alone or next to a much longer one (heavy left padding) gives the same values."""
    completion = make_completions(1, seed=3)[0]
    long_neighbour = FakeCompletion([1] * 20, [2] * 12)
    alone, mask_alone = completion_token_log_probs(tiny_policy, [completion], PAD_TOKEN_ID)
    batched, mask_batched = completion_token_log_probs(tiny_policy, [completion, long_neighbour], PAD_TOKEN_ID)
    n = alone.shape[1]
    assert mask_alone[0].sum() == n == len(completion.completion_ids)
    assert torch.equal(mask_batched[0, -n:], mask_alone[0]) and mask_batched[0, :-n].sum() == 0
    assert torch.allclose(batched[0, -n:], alone[0], atol=1e-5)


# ---------------------------------------------------------------- loss and gradient


def test_loss_is_token_normalized_by_the_whole_step():
    log_probs = torch.tensor([[-1.0, -1.0, -1.0], [-2.0, -2.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    advantages = torch.tensor([1.0, -1.0])
    # -(1 * -3 + -1 * -4) / 10 = -0.1
    assert policy_gradient_loss(log_probs, mask, advantages, total_completion_tokens=10).item() == pytest.approx(-0.1)


def test_micro_batched_gradient_equals_full_batch(tiny_policy):
    completions = make_completions(16)
    advantages = torch.randn(16, generator=torch.Generator().manual_seed(5))
    full_loss = accumulate_policy_gradient(tiny_policy, completions, advantages, micro_batch_size=16, pad_token_id=PAD_TOKEN_ID)
    full_gradient = flat_gradient(tiny_policy).clone()
    tiny_policy.zero_grad()
    micro_loss = accumulate_policy_gradient(tiny_policy, completions, advantages, micro_batch_size=4, pad_token_id=PAD_TOKEN_ID)
    assert micro_loss == pytest.approx(full_loss, abs=1e-5)
    assert torch.allclose(flat_gradient(tiny_policy), full_gradient, atol=1e-5)


def test_zero_advantage_completions_add_nothing_but_count_tokens(tiny_policy):
    completions = make_completions(8)
    advantages = torch.tensor([1.0, -1.0, 0, 0, 0, 0, 0, 0])
    accumulate_policy_gradient(tiny_policy, completions, advantages, 4, PAD_TOKEN_ID)
    with_zeros = flat_gradient(tiny_policy).clone()
    tiny_policy.zero_grad()
    # same two contributing completions, divided by the same total token count
    total_tokens = sum(len(c.completion_ids) for c in completions)
    log_probs, mask = completion_token_log_probs(tiny_policy, completions[:2], PAD_TOKEN_ID)
    policy_gradient_loss(log_probs, mask, advantages[:2].to(DEVICE), total_tokens).backward()
    assert torch.allclose(flat_gradient(tiny_policy), with_zeros, atol=1e-6)


def test_all_zero_advantages_leave_no_gradient(tiny_policy):
    loss = accumulate_policy_gradient(tiny_policy, make_completions(8), torch.zeros(8), 4, PAD_TOKEN_ID)
    assert loss == 0.0
    assert all(p.grad is None for p in tiny_policy.parameters())


def test_positive_advantage_makes_the_completion_more_likely(tiny_policy):
    """One plain SGD step on a positive-advantage completion raises its log-probability."""
    completion = make_completions(1, seed=9)[0]
    before, mask = completion_token_log_probs(tiny_policy, [completion], PAD_TOKEN_ID)
    accumulate_policy_gradient(tiny_policy, [completion], torch.tensor([1.0]), 4, PAD_TOKEN_ID)
    with torch.no_grad():
        for parameter in tiny_policy.parameters():
            parameter -= 0.1 * parameter.grad
    after, _ = completion_token_log_probs(tiny_policy, [completion], PAD_TOKEN_ID)
    assert (after * mask).sum() > (before * mask).sum()


def test_mismatched_lengths_fail(tiny_policy):
    with pytest.raises(ValueError):
        accumulate_policy_gradient(tiny_policy, make_completions(4), torch.ones(3), 4, PAD_TOKEN_ID)
