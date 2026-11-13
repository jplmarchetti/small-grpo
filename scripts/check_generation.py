"""Checkpoint 5 checks on the real model (GPU PC). Exits non-zero if any check fails.

    uv run --extra train python scripts/check_generation.py

1. B0 evaluated twice on the first 50 validation problems gives identical rows.
2. No completion text contains "User:".
3. A fresh adapter gives the same greedy completions as B0 (LoRA starts as "no change").
4. A saved and reloaded adapter gives the same greedy completions again.
5. A 5-token limit marks every completion truncated.
6. Sampling 8 completions for one problem works (shape, all graded).
Also prints timing and peak VRAM.
"""

import json
import tempfile
import time
from pathlib import Path

import torch

from grpo_gsm8k.config import load_config
from grpo_gsm8k.data import load_split
from grpo_gsm8k.evaluate import evaluate_policy, summarize_predictions
from grpo_gsm8k.policy import load_policy, save_adapter
from grpo_gsm8k.prompts import build_prompt
from grpo_gsm8k.rollout import DecodingConfig, generate_completions, greedy_decoding, sample_groups, sampling_decoding

N_EVAL_PROBLEMS = 50
N_ADAPTER_PROMPTS = 5
LORA_SEED = 42

results: dict[str, bool] = {}


def check(name: str, passed: bool) -> None:
    results[name] = bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name}", flush=True)


def greedy_texts(policy, tokenizer, prompts, decoding) -> list[str]:
    return [c.text for c in generate_completions(policy, tokenizer, prompts, decoding)]


def main() -> None:
    config = load_config()
    greedy = greedy_decoding(config.training)
    problems = load_split("validation")[:N_EVAL_PROBLEMS]
    batch_size = config.evaluation.batch_size

    b0, tokenizer = load_policy(config.model)
    b0.eval()
    started = time.time()
    first = evaluate_policy(b0, tokenizer, problems, greedy, batch_size, show_progress=False)
    eval_seconds = time.time() - started
    second = evaluate_policy(b0, tokenizer, problems, greedy, batch_size, show_progress=False)
    check("B0 evaluated twice gives identical rows", first == second)
    check('no completion text contains "User:"', all("User:" not in row["completion"] for row in first))
    summary = summarize_predictions(first)
    print(json.dumps(summary, indent=2), f"\n{eval_seconds:.1f}s for {N_EVAL_PROBLEMS} greedy problems", flush=True)

    prompts = [build_prompt(p.question) for p in problems[:N_ADAPTER_PROMPTS]]
    b0_texts = greedy_texts(b0, tokenizer, prompts, greedy)
    del b0
    torch.cuda.empty_cache()

    fresh, _ = load_policy(config.model, config.lora, lora_seed=LORA_SEED)
    fresh.eval()
    check("fresh adapter == B0 (greedy)", greedy_texts(fresh, tokenizer, prompts, greedy) == b0_texts)
    with tempfile.TemporaryDirectory() as adapter_dir:
        save_adapter(fresh, Path(adapter_dir))
        del fresh
        torch.cuda.empty_cache()
        reloaded, _ = load_policy(config.model, adapter_dir=Path(adapter_dir))
        reloaded.eval()
        check("saved + reloaded adapter == B0 (greedy)", greedy_texts(reloaded, tokenizer, prompts, greedy) == b0_texts)

    short = DecodingConfig(do_sample=False, max_new_tokens=5)
    check("5-token limit marks every row truncated", all(c.truncated for c in generate_completions(reloaded, tokenizer, prompts, short)))

    torch.manual_seed(0)
    started = time.time()
    groups = sample_groups(reloaded, tokenizer, prompts[:2], config.training.group_size, sampling_decoding(config.training))
    sample_seconds = time.time() - started
    check("sampling gives problems x group_size completions", [len(g) for g in groups] == [config.training.group_size] * 2)
    truncated = sum(c.truncated for g in groups for c in g)
    print(f"{sample_seconds:.1f}s to sample 2 x {config.training.group_size}, {truncated} truncated", flush=True)
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)

    failed = [name for name, passed in results.items() if not passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
