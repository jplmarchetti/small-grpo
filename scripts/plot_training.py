"""Plot the training curves of one run (laptop, after `pcw fetch outputs/<run>`).

    uv run python scripts/plot_training.py outputs/smoke_test --window 5
"""

import argparse
from pathlib import Path

from grpo_gsm8k.training_plots import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--window", type=int, default=20, help="moving-average window in steps")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    print(plot_training_curves(args.run_dir, args.window, args.title))


if __name__ == "__main__":
    main()
