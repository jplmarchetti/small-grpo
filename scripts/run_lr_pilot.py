"""Stage 11b: the learning-rate pilot on the GPU PC (~16 min per rate).

    uv run --extra train python scripts/run_lr_pilot.py

Trains RC for `lr_pilot.steps` steps at each candidate rate into outputs/lr_pilot/lr_<rate>/,
with validation at steps 0, 50 and 100. Then it prints the collapse verdict of every rate
and the rate the rule chooses, and writes outputs/lr_pilot/verdicts.json. The script never
edits configs/base.yaml: the chosen rate is written there by hand, then tagged config-v1.
Re-running resumes unfinished rates and skips finished ones.
"""

import json
from dataclasses import asdict

from grpo_gsm8k.config import load_config
from grpo_gsm8k.lr_pilot import choose_learning_rate, judge_collapse, pilot_run_dir, read_val_curve
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.rewards import RewardCell
from grpo_gsm8k.train import RunSpec, checkpoint_dir, run_training

PILOT_ROOT = REPO_ROOT / "outputs" / "lr_pilot"


def main() -> None:
    config = load_config()
    rule = config.lr_pilot
    for learning_rate in rule.learning_rates:
        run_dir = pilot_run_dir(PILOT_ROOT, learning_rate)
        if (checkpoint_dir(run_dir, rule.steps) / "training_state.pt").exists():
            print(f"lr {learning_rate:g}: already finished, skipping", flush=True)
            continue
        print(f"\n===== lr {learning_rate:g} =====", flush=True)
        run_training(config, RunSpec(RewardCell.RC, rule.seed, run_dir, rule.steps, learning_rate, validate=True))

    verdicts = [
        judge_collapse(lr, read_val_curve(pilot_run_dir(PILOT_ROOT, lr) / "val_curve.csv"), rule)
        for lr in rule.learning_rates
    ]
    chosen = choose_learning_rate(verdicts)
    print("\n===== verdicts =====")
    for verdict in verdicts:
        status = "COLLAPSED: " + "; ".join(verdict.reasons) if verdict.collapsed else "stable"
        print(f"lr {verdict.learning_rate:g}: {status}")
    print(f"\nrule chooses: {chosen if chosen is not None else 'none (every rate collapsed)'}")
    (PILOT_ROOT / "verdicts.json").write_text(
        json.dumps({"verdicts": [asdict(v) for v in verdicts], "chosen_learning_rate": chosen}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
