"""Stage 12: the run queue. No training logic of its own.

- Walks through (seed, cell) pairs in a fixed order; a finished run is skipped, an
  interrupted one resumes (run_training resumes from its latest checkpoint).
- Every run is its own process, so a crash cannot leave GPU memory or state behind for
  the next one. A failed run writes `error.txt` in its folder and the queue continues.
"""

import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from grpo_gsm8k.paths import REPO_ROOT

RUNS_ROOT = REPO_ROOT / "outputs"


def run_dir(cell: str, seed: int, root: Path = RUNS_ROOT) -> Path:
    return root / cell / f"seed{seed}"


def run_is_complete(directory: Path, steps: int) -> bool:
    return (directory / "checkpoints" / f"step_{steps:03d}" / "training_state.pt").exists()


def train_in_subprocess(cell: str, seed: int) -> int:
    command = [sys.executable, str(REPO_ROOT / "scripts" / "run_train.py"), "--cell", cell, "--seed", str(seed)]
    return subprocess.run(command, check=False).returncode


def run_queue(
    seeds: Sequence[int],
    cells: Sequence[str],
    steps: int,
    train: Callable[[str, int], int] = train_in_subprocess,
    root: Path = RUNS_ROOT,
) -> dict[tuple[str, int], str]:
    """Seed by seed, so every finished seed is a complete 2x2. Returns a status per run."""
    status = {}
    for seed in seeds:
        for cell in cells:
            directory = run_dir(cell, seed, root)
            if run_is_complete(directory, steps):
                status[(cell, seed)] = "already complete"
                continue
            print(f"\n===== {cell} seed {seed} ({time.strftime('%H:%M')}) =====", flush=True)
            started = time.time()
            return_code = train(cell, seed)
            if return_code == 0 and run_is_complete(directory, steps):
                status[(cell, seed)] = f"complete in {(time.time() - started) / 60:.0f} min"
            else:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "error.txt").write_text(f"{time.strftime('%Y-%m-%d %H:%M')} exit code {return_code}\n")
                status[(cell, seed)] = f"FAILED (exit code {return_code})"
            print(f"{cell} seed {seed}: {status[(cell, seed)]}", flush=True)
    return status
