"""Stage 10: train one run (one cell x one seed) on the GPU PC.

    uv run --extra train python scripts/run_train.py --cell RC --seed 42

The learning rate comes from configs/base.yaml (set by the LR pilot). `--learning-rate`
exists only for the smoke test and the LR pilot, which write outside the core run folders.
Re-running the same command resumes from the latest checkpoint.
"""

import argparse
from pathlib import Path

from grpo_gsm8k.config import load_config
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.rewards import RewardCell
from grpo_gsm8k.train import RunSpec, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cell", required=True, choices=[cell.value for cell in RewardCell])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int, default=None, help="default: training.steps from the config")
    parser.add_argument("--learning-rate", type=float, default=None, help="pilot / smoke test only")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    learning_rate = args.learning_rate if args.learning_rate is not None else config.training.learning_rate
    if learning_rate is None:
        raise SystemExit("training.learning_rate is null in configs/base.yaml: run the LR pilot first")
    if args.learning_rate is not None and args.output_dir is None:
        raise SystemExit("--learning-rate overrides the config, so it needs an explicit --output-dir")
    spec = RunSpec(
        cell=RewardCell(args.cell),
        seed=args.seed,
        output_dir=args.output_dir or REPO_ROOT / "outputs" / args.cell / f"seed{args.seed}",
        steps=args.steps or config.training.steps,
        learning_rate=learning_rate,
        validate=not args.no_validation,
    )
    run_training(config, spec)


if __name__ == "__main__":
    main()
