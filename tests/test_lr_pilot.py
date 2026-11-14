import pytest

from grpo_gsm8k.config import load_config
from grpo_gsm8k.lr_pilot import PilotVerdict, choose_learning_rate, judge_collapse, pilot_run_dir

RULE = load_config().lr_pilot


def point(step: int, lenient: float = 0.49, truncation: float = 0.01, tokens: float = 100.0) -> dict:
    return {"step": step, "lenient_accuracy": lenient, "truncation_rate": truncation, "mean_tokens": tokens}


def test_rule_is_the_one_fixed_before_the_pilot():
    assert RULE.learning_rates == (1e-5, 3e-5, 1e-4)
    assert (RULE.steps, RULE.seed) == (100, 42)
    assert (RULE.max_lenient_drop, RULE.max_truncation_rise, RULE.min_length_ratio) == (0.05, 0.05, 0.5)


def test_a_stable_or_improving_curve_does_not_collapse():
    verdict = judge_collapse(3e-5, [point(0), point(50, lenient=0.47), point(100, lenient=0.55)], RULE)
    assert not verdict.collapsed and verdict.reasons == ()


def test_a_drop_within_noise_is_not_a_collapse():
    assert not judge_collapse(1e-4, [point(0, lenient=0.49), point(50, lenient=0.45)], RULE).collapsed


@pytest.mark.parametrize(
    "later, reason",
    [
        (point(100, lenient=0.40), "lenient accuracy fell"),
        (point(100, truncation=0.10), "truncation rose"),
        (point(100, tokens=40.0), "length fell"),
    ],
)
def test_each_threshold_triggers_a_collapse(later, reason):
    verdict = judge_collapse(1e-4, [point(0), point(50), later], RULE)
    assert verdict.collapsed and reason in verdict.reasons[0] and "step 100" in verdict.reasons[0]


def test_collapse_at_step_50_counts_even_if_step_100_recovers():
    assert judge_collapse(1e-4, [point(0), point(50, lenient=0.30), point(100)], RULE).collapsed


def test_judging_needs_step_0_and_a_later_point():
    with pytest.raises(ValueError):
        judge_collapse(1e-5, [point(50)], RULE)
    with pytest.raises(ValueError):
        judge_collapse(1e-5, [point(0)], RULE)


def test_choose_the_largest_rate_that_did_not_collapse():
    verdicts = [PilotVerdict(1e-5, False, ()), PilotVerdict(3e-5, False, ()), PilotVerdict(1e-4, True, ("x",))]
    assert choose_learning_rate(verdicts) == 3e-5
    assert choose_learning_rate([PilotVerdict(lr, False, ()) for lr in RULE.learning_rates]) == 1e-4
    assert choose_learning_rate([PilotVerdict(lr, True, ("x",)) for lr in RULE.learning_rates]) is None


def test_run_dirs_are_distinct(tmp_path):
    names = {pilot_run_dir(tmp_path, lr).name for lr in RULE.learning_rates}
    assert names == {"lr_1e-05", "lr_3e-05", "lr_1e-04"}
