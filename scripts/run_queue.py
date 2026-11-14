"""Stage 12: train every cell of the given seeds, then (optionally) evaluate on test (GPU PC).

    uv run --extra train python scripts/run_queue.py --seeds 42 --cells RC RFC RF R0 --then-final-eval

Seeds run one at a time, all cells of a seed before the next seed. Finished runs are
skipped, interrupted ones resume. Re-running the same command is always safe.
"""

import argparse
import subprocess
import sys

from grpo_gsm8k.config import load_config
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.queue import run_queue
from grpo_gsm8k.rewards import RewardCell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--cells", nargs="+", default=[cell.value for cell in RewardCell],
                        choices=[cell.value for cell in RewardCell])
    parser.add_argument("--then-final-eval", action="store_true", help="run scripts/run_final_eval.py afterwards")
    args = parser.parse_args()

    status = run_queue(args.seeds, args.cells, load_config().training.steps)
    print("\n===== queue summary =====")
    for (cell, seed), outcome in status.items():
        print(f"{cell} seed {seed}: {outcome}")
    failed = [run for run, outcome in status.items() if outcome.startswith("FAILED")]
    if args.then_final_eval:
        if failed:
            print(f"not running the final evaluation: {len(failed)} run(s) failed")
        else:
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_final_eval.py")], check=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
