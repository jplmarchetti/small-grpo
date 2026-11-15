"""Stability gate of config-v2 (METHODOLOGY §13, 2026-11-15): which step is the reported model.

- Fixed before any config-v2 run, applied once to seed 42's validation curves.
- It looks only at stability (truncation in every cell, drift of the no-signal R0 cell),
  never at the correctness gain, so it cannot select for the effect being measured.
- Passes at `check_step` -> report that step. Otherwise try `fallback_step`. Otherwise no
  step is reported and the negative result is written up instead.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from grpo_gsm8k.config import StabilityGateConfig
from grpo_gsm8k.paths import MANIFEST_DIR

GATE_PATH = MANIFEST_DIR / "stability_gate.json"

ValCurve = Sequence[Mapping[str, float]]


@dataclass(frozen=True)
class GateVerdict:
    reported_step: int | None  # None: the gate failed at both steps
    reasons: tuple[str, ...]  # why each failed step failed


def gate_failures(curves: Mapping[str, ValCurve], step: int, rule: StabilityGateConfig) -> list[str]:
    """Broken thresholds at `step`, compared with step 0 of the same run."""
    failures = []
    for cell, curve in curves.items():
        by_step = {int(row["step"]): row for row in curve}
        if 0 not in by_step or step not in by_step:
            raise ValueError(f"{cell}: validation at steps 0 and {step} is required")
        start, end = by_step[0], by_step[step]
        rise = end["truncation_rate"] - start["truncation_rate"]
        if rise > rule.max_truncation_rise:
            failures.append(f"step {step}: {cell} truncation rose {rise:.1%}")
        if cell == "R0":
            change = end["lenient_accuracy"] - start["lenient_accuracy"]
            if abs(change) > rule.max_r0_lenient_change:
                failures.append(f"step {step}: R0 lenient accuracy moved {change:+.1%}")
    return failures


def judge_stability(curves: Mapping[str, ValCurve], rule: StabilityGateConfig) -> GateVerdict:
    if "R0" not in curves:
        raise ValueError("the gate needs the R0 curve")
    reasons: list[str] = []
    for step in (rule.check_step, rule.fallback_step):
        failures = gate_failures(curves, step, rule)
        if not failures:
            return GateVerdict(step, tuple(reasons))
        reasons.extend(failures)
    return GateVerdict(None, tuple(reasons))


def save_gate_verdict(verdict: GateVerdict, config_tag: str, path: Path = GATE_PATH) -> None:
    record = {"config_tag": config_tag, **asdict(verdict)}
    path.write_text(json.dumps(record, indent=2) + "\n")


def load_reported_step(config_tag: str, path: Path = GATE_PATH) -> int:
    """The step every final evaluation uses; refuses if the gate has not run or failed."""
    if not path.exists():
        raise SystemExit(f"{path.name} is missing: run scripts/judge_stability_gate.py first")
    record = json.loads(path.read_text())
    if record["config_tag"] != config_tag:
        raise SystemExit(f"{path.name} is for {record['config_tag']}, not {config_tag}")
    if record["reported_step"] is None:
        raise SystemExit(f"the stability gate failed ({record['reasons']}): no model is reported")
    return int(record["reported_step"])
