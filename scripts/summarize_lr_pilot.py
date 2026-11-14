"""Compare the pilot rates side by side (laptop, after `pcw fetch outputs/lr_pilot`).

    uv run python scripts/summarize_lr_pilot.py

Prints the validation table at steps 0 / 50 / 100 with each rate's verdict, and writes
outputs/lr_pilot/lr_pilot.png, plus training_curves.png in each rate's folder.
"""

from grpo_gsm8k.config import load_config
from grpo_gsm8k.lr_pilot import choose_learning_rate, judge_collapse, pilot_run_dir, read_val_curve
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.training_plots import plot_lr_pilot, plot_training_curves

PILOT_ROOT = REPO_ROOT / "outputs" / "lr_pilot"


def main() -> None:
    rule = load_config().lr_pilot
    curves = {lr: read_val_curve(pilot_run_dir(PILOT_ROOT, lr) / "val_curve.csv") for lr in rule.learning_rates}
    verdicts = [judge_collapse(lr, curve, rule) for lr, curve in curves.items()]

    print(f"{'lr':>8} {'step':>5} {'lenient':>8} {'strict':>7} {'format':>7} {'tokens':>7} {'trunc':>6}")
    for lr, curve in curves.items():
        for row in curve:
            print(
                f"{lr:>8g} {int(row['step']):>5} {row['lenient_accuracy']:>8.1%} {row['strict_accuracy']:>7.1%} "
                f"{row['format_rate']:>7.1%} {row['mean_tokens']:>7.0f} {row['truncation_rate']:>6.1%}"
            )
    for verdict in verdicts:
        print(f"lr {verdict.learning_rate:g}: " + ("COLLAPSED: " + "; ".join(verdict.reasons) if verdict.collapsed else "stable"))
    print(f"rule chooses: {choose_learning_rate(verdicts)}")

    for lr in rule.learning_rates:
        plot_training_curves(pilot_run_dir(PILOT_ROOT, lr), window=10, title=f"LR pilot: RC, lr {lr:g}")
    print(plot_lr_pilot(PILOT_ROOT, rule.learning_rates))


if __name__ == "__main__":
    main()
