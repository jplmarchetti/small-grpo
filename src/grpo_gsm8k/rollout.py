"""Stage 5: make the policy write, by sampling (training, pre-flight) or greedily (evaluation).

- A completion stops at an end-of-text token, at the string "\\nUser:" (the model
  starts inventing the next turn), or at `max_new_tokens` (then it is `truncated`).
- The stop tokens stay in `completion_ids` (they are trained on, judgment call 6);
  the text handed to the grader has the stop string removed.
- Token ids are kept, not just text: training must score exactly the sampled tokens.

The per-row cutting follows the idea at the end of `sample_responses_batched` in
rasbt/reasoning-from-scratch (Apache-2.0, Copyright Sebastian Raschka); the sampling
loop itself is Hugging Face `generate`.

torch is imported inside the generation functions only, so the pure trimming logic
is testable on the laptop.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from grpo_gsm8k.config import TrainingConfig
from grpo_gsm8k.prompts import STOP_STRING


@dataclass(frozen=True)
class DecodingConfig:
    do_sample: bool
    max_new_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0


def sampling_decoding(training_config: TrainingConfig) -> DecodingConfig:
    return DecodingConfig(
        do_sample=True,
        max_new_tokens=training_config.max_new_tokens,
        temperature=training_config.temperature,
        top_p=training_config.top_p,
    )


def greedy_decoding(training_config: TrainingConfig) -> DecodingConfig:
    return DecodingConfig(do_sample=False, max_new_tokens=training_config.max_new_tokens)


@dataclass(frozen=True)
class Completion:
    prompt_ids: list[int]
    completion_ids: list[int]  # includes the stop tokens
    text: str  # without the stop string; what the grader reads
    n_tokens: int
    truncated: bool


# ---------------------------------------------------------------- stopping (pure)


def shortest_prefix_containing(token_ids: list[int], decode: Callable[[list[int]], str], needle: str) -> int:
    """Smallest k such that decode(token_ids[:k]) contains `needle` (binary search; the text only grows)."""
    low, high = 1, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        if needle in decode(token_ids[:middle]):
            high = middle
        else:
            low = middle + 1
    return low


def trim_at_stop(
    token_ids: Sequence[int],
    decode: Callable[[list[int]], str],
    end_token_ids: set[int],
    stop_string: str = STOP_STRING,
) -> tuple[list[int], str, bool]:
    """Cut new tokens at the first stop. Returns (kept ids, grader text, stopped)."""
    ids = list(token_ids)
    first_end = next((index for index, token in enumerate(ids) if token in end_token_ids), None)
    if first_end is not None:
        ids = ids[: first_end + 1]
    text = decode(ids)
    if stop_string in text:
        ids = ids[: shortest_prefix_containing(ids, decode, stop_string)]
        text = decode(ids)
        return ids, text[: text.index(stop_string)], True
    return ids, text, first_end is not None


# ---------------------------------------------------------------- generation (GPU)


def end_token_ids(policy, tokenizer) -> set[int]:
    """The tokenizer's end-of-text plus any end tokens of the model's generation config."""
    ids = {tokenizer.eos_token_id}
    configured = getattr(getattr(policy, "generation_config", None), "eos_token_id", None)
    if configured is not None:
        ids.update(configured if isinstance(configured, list) else [configured])
    return ids


def generate_completions(policy, tokenizer, prompts: Sequence[str], decoding: DecodingConfig) -> list[Completion]:
    import torch
    from transformers import GenerationConfig

    device = next(policy.parameters()).device
    stop_ids = end_token_ids(policy, tokenizer)
    sampling = {"temperature": decoding.temperature, "top_p": decoding.top_p, "top_k": 0} if decoding.do_sample else {}
    generation_config = GenerationConfig(
        do_sample=decoding.do_sample,
        max_new_tokens=decoding.max_new_tokens,
        eos_token_id=sorted(stop_ids),
        pad_token_id=tokenizer.pad_token_id,
        stop_strings=[STOP_STRING],
        **sampling,
    )
    batch = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output = policy.generate(**batch, generation_config=generation_config, tokenizer=tokenizer)
    new_tokens = output[:, batch["input_ids"].shape[1] :].tolist()

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=True)

    completions = []
    for row, row_new_tokens in enumerate(new_tokens):
        prompt_ids = batch["input_ids"][row][batch["attention_mask"][row].bool()].tolist()
        kept_ids, text, stopped = trim_at_stop(row_new_tokens, decode, stop_ids)
        completions.append(Completion(prompt_ids, kept_ids, text, len(kept_ids), truncated=not stopped))
    return completions


def sample_groups(
    policy, tokenizer, prompts: Sequence[str], group_size: int, decoding: DecodingConfig
) -> list[list[Completion]]:
    """`group_size` sampled completions per prompt, arranged as problems x group_size."""
    repeated = [prompt for prompt in prompts for _ in range(group_size)]
    flat = generate_completions(policy, tokenizer, repeated, decoding)
    return [flat[start : start + group_size] for start in range(0, len(flat), group_size)]
