"""Stage 11a: 20 steps of RC, to prove the loop works and measure its cost (GPU PC).

    uv run --extra train python scripts/run_smoke_test.py

Checks:
1. The loss is finite at every step that updated.
2. The adapter changed (LoRA B matrices start at zero; at least one is no longer zero).
3. The step-20 adapter, saved and reloaded, gives identical greedy completions.
Prints s/step (after warm-up) and peak VRAM, and writes outputs/smoke_test/.
"""

import math
import shutil

import torch

from grpo_gsm8k.config import load_config
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.policy import load_policy
from grpo_gsm8k.prompts import build_prompt
from grpo_gsm8k.rewards import RewardCell
from grpo_gsm8k.rollout import generate_completions, greedy_decoding
from grpo_gsm8k.run_log import read_step_logs
from grpo_gsm8k.train import RunSpec, build_context, checkpoint_dir, run_training

SMOKE_STEPS = 20
WARMUP_STEPS = 3
SMOKE_LEARNING_RATE = 3e-5  # middle of the pilot grid; the real LR comes from the pilot
OUTPUT_DIR = REPO_ROOT / "outputs" / "smoke_test"
N_RELOAD_PROMPTS = 5


def adapter_has_changed(policy: torch.nn.Module) -> bool:
    return any(p.abs().sum() > 0 for name, p in policy.named_parameters() if "lora_B" in name)


def main() -> None:
    config = load_config()
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    spec = RunSpec(RewardCell.RC, 42, OUTPUT_DIR, SMOKE_STEPS, SMOKE_LEARNING_RATE, validate=False)
    run_training(config, spec)

    logs = read_step_logs(OUTPUT_DIR / "train_log.csv")
    updated = [row for row in logs if row["skipped_update"] == "False"]
    checks = {"loss finite at every updated step": all(math.isfinite(float(row["loss"])) for row in updated)}

    context, _ = build_context(config, spec)  # resumes from step 20
    checks["adapter changed from its zero start"] = adapter_has_changed(context.policy)
    context.policy.eval()
    greedy = greedy_decoding(config.training)
    prompts = [build_prompt(p.question, context.exemplars) for p in context.problem_order[:N_RELOAD_PROMPTS]]
    trained = [c.text for c in generate_completions(context.policy, context.tokenizer, prompts, greedy)]
    reloaded_policy, tokenizer = load_policy(config.model, adapter_dir=checkpoint_dir(OUTPUT_DIR, SMOKE_STEPS))
    reloaded_policy.eval()
    reloaded = [c.text for c in generate_completions(reloaded_policy, tokenizer, prompts, greedy)]
    checks["saved + reloaded adapter gives identical greedy completions"] = trained == reloaded

    timed = [float(row["seconds"]) for row in logs[WARMUP_STEPS:]]
    seconds_per_step = sum(timed) / len(timed)
    print(f"\n{len(updated)}/{len(logs)} steps updated the adapter")
    print(f"{seconds_per_step:.1f} s/step after warm-up -> ~{seconds_per_step * config.training.steps / 60:.0f} min "
          f"of training per {config.training.steps}-step run (plus validation)")
    print(f"peak VRAM {max(float(row['peak_vram_gb']) for row in logs):.2f} GB")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
