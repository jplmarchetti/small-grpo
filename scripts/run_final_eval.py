"""Stage 13: score every model once on the 1,319 GSM8K test problems (GPU PC).

    uv run --extra train python scripts/run_final_eval.py

Started by hand only, once all seeds are trained; no queue runs it. The ONLY caller of
`load_test_problems`. It refuses to run unless the frozen files match their tags
(grader-v1, config-v2) and the stability gate has fixed the reported step. Models:
- B0 (zero-shot prompt) and B0_fewshot: evaluated once, they have no seed;
- every finished run (cell x seed) at the reported step, with the trained prompt.
Writes outputs/<config_tag>/test/<model>[/seed<k>]/predictions.jsonl and metrics.json; a
model that already has predictions is skipped.
"""

import json
from pathlib import Path

import torch

from grpo_gsm8k.config import load_config
from grpo_gsm8k.data import load_split, load_test_problems
from grpo_gsm8k.evaluate import evaluate_policy, summarize_predictions, write_predictions
from grpo_gsm8k.frozen import require_frozen_files
from grpo_gsm8k.policy import load_policy
from grpo_gsm8k.preflight import load_preflight
from grpo_gsm8k.queue import run_dir, run_is_complete, runs_root
from grpo_gsm8k.rewards import RewardCell
from grpo_gsm8k.rollout import greedy_decoding
from grpo_gsm8k.stability_gate import load_reported_step


def evaluate_once(name: str, out_dir: Path, adapter_dir: Path | None, exemplars, problems, config) -> None:
    predictions_path = out_dir / "predictions.jsonl"
    if predictions_path.exists():
        print(f"{name}: already evaluated, skipping", flush=True)
        return
    print(f"\n== {name} on {len(problems)} test problems", flush=True)
    policy, tokenizer = load_policy(config.model, adapter_dir=adapter_dir)
    policy.eval()
    rows = evaluate_policy(policy, tokenizer, problems, greedy_decoding(config.training),
                           config.evaluation.batch_size, exemplars, show_progress=False)
    write_predictions(rows, predictions_path)
    summary = summarize_predictions(rows) | {"name": name, "adapter": str(adapter_dir) if adapter_dir else None}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)
    del policy
    torch.cuda.empty_cache()


def main() -> None:
    require_frozen_files()
    config = load_config()
    reported_step = load_reported_step(config.config_tag)
    test_root = runs_root() / "test"
    problems = load_test_problems()
    fewshot_exemplars = load_split("fewshot_exemplars")
    trained_exemplars = fewshot_exemplars if load_preflight()["use_fewshot_prompt"] else ()

    evaluate_once("B0", test_root / "B0", None, (), problems, config)
    evaluate_once("B0_fewshot", test_root / "B0_fewshot", None, fewshot_exemplars, problems, config)
    steps = config.training.steps
    for seed in config.training.seeds:
        for cell in RewardCell:
            directory = run_dir(cell.value, seed)
            if not run_is_complete(directory, steps):
                continue
            adapter_dir = directory / "checkpoints" / f"step_{reported_step:03d}"  # fixed by the gate, same for all
            evaluate_once(f"{cell.value} seed {seed}", test_root / cell.value / f"seed{seed}", adapter_dir,
                          trained_exemplars, problems, config)


if __name__ == "__main__":
    main()
