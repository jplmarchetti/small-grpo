import pytest

from grpo_gsm8k.config import load_config
from grpo_gsm8k.stability_gate import GateVerdict, judge_stability, load_reported_step, save_gate_verdict

RULE = load_config().stability_gate


def point(step, lenient=0.494, truncation=0.008):
    return {"step": float(step), "lenient_accuracy": lenient, "truncation_rate": truncation}


def curves(r0=None, rc=None):
    stable = [point(0), point(50), point(100), point(150)]
    return {"R0": r0 or stable, "RF": stable, "RC": rc or stable, "RFC": stable}


def test_rule_is_the_one_fixed_in_methodology():
    assert (RULE.check_step, RULE.fallback_step) == (150, 100)
    assert (RULE.max_truncation_rise, RULE.max_r0_lenient_change) == (0.05, 0.05)
    assert load_config().training.steps == RULE.check_step


def test_stable_runs_report_the_last_step():
    assert judge_stability(curves(), RULE) == GateVerdict(150, ())


def test_r0_drift_in_either_direction_fails():
    down = [point(0), point(50), point(100), point(150, lenient=0.40)]
    up = [point(0), point(50), point(100), point(150, lenient=0.60)]
    assert judge_stability(curves(r0=down), RULE).reported_step == 100
    assert judge_stability(curves(r0=up), RULE).reported_step == 100


def test_correctness_gain_alone_never_fails_the_gate():
    better_rc = [point(0), point(50, 0.55), point(100, 0.58), point(150, 0.60)]
    assert judge_stability(curves(rc=better_rc), RULE).reported_step == 150


def test_truncation_in_any_cell_falls_back_then_fails():
    late = [point(0), point(50), point(100), point(150, truncation=0.10)]
    verdict = judge_stability(curves(rc=late), RULE)
    assert verdict.reported_step == 100 and "RC truncation" in verdict.reasons[0]
    early = [point(0), point(50), point(100, truncation=0.10), point(150, truncation=0.20)]
    assert judge_stability(curves(rc=early), RULE).reported_step is None


def test_missing_validation_point_is_an_error():
    with pytest.raises(ValueError):
        judge_stability(curves(rc=[point(0), point(50)]), RULE)


def test_reported_step_round_trip_and_refusals(tmp_path):
    path = tmp_path / "gate.json"
    with pytest.raises(SystemExit):
        load_reported_step("config-v2", path)
    save_gate_verdict(GateVerdict(150, ()), "config-v2", path)
    assert load_reported_step("config-v2", path) == 150
    with pytest.raises(SystemExit):
        load_reported_step("config-v3", path)
    save_gate_verdict(GateVerdict(None, ("x",)), "config-v2", path)
    with pytest.raises(SystemExit):
        load_reported_step("config-v2", path)
