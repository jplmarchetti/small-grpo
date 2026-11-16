"""Apply the config-v2 stability gate to seed 42's validation curves (laptop).

    pcw fetch outputs/config-v2        # val_curve.csv of every run
    uv run python scripts/judge_stability_gate.py

Writes manifests/stability_gate.json with the reported step (or none). Run it once, after
the four seed-42 runs finish and before seeds 43 and 44.
"""

from grpo_gsm8k.config import load_config
from grpo_gsm8k.lr_pilot import read_val_curve
from grpo_gsm8k.queue import run_dir
from grpo_gsm8k.rewards import RewardCell
from grpo_gsm8k.stability_gate import GATE_PATH, judge_stability, save_gate_verdict

GATE_SEED = 42


def main() -> None:
    config = load_config()
    curves = {cell.value: read_val_curve(run_dir(cell.value, GATE_SEED) / "val_curve.csv") for cell in RewardCell}
    verdict = judge_stability(curves, config.stability_gate)
    save_gate_verdict(verdict, config.config_tag)
    for reason in verdict.reasons:
        print(reason)
    print(f"reported step: {verdict.reported_step}  ->  {GATE_PATH}")


if __name__ == "__main__":
    main()
