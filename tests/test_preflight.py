import pytest

from grpo_gsm8k.audit import (
    draw_audit_sample,
    grader_agreement,
    read_audit_sheet,
    write_audit_sheet,
)
from grpo_gsm8k.config import load_config
from grpo_gsm8k.graders import Grade
from grpo_gsm8k.preflight import (
    build_preflight_manifest,
    decide_fewshot_prompt,
    estimate_base_success_rate,
    load_preflight,
    mixed_group_fraction,
    sampled_statistics,
    save_preflight,
)


def grade(format_ok: bool, lenient: bool) -> Grade:
    return Grade(format_ok, format_ok and lenient, lenient, "1")


ALL_WRONG = [grade(True, False)] * 8
ONE_RIGHT = [grade(True, True)] + [grade(True, False)] * 7
FORMAT_MIXED = [grade(False, False)] + [grade(True, False)] * 7


# ---------------------------------------------------------------- p0 and mixed groups


def test_p0_is_the_mean_lenient_score_over_all_completions():
    assert estimate_base_success_rate([ALL_WRONG, ONE_RIGHT]) == pytest.approx(1 / 16)


def test_mixed_group_fraction_counts_groups_that_can_teach():
    groups = [ALL_WRONG, ONE_RIGHT, FORMAT_MIXED, [grade(True, True)] * 8]
    assert mixed_group_fraction(groups, "lenient_correct") == pytest.approx(1 / 4)
    assert mixed_group_fraction(groups, "format_ok") == pytest.approx(1 / 4)


def test_sampled_statistics():
    truncated = [[False] * 8, [True] + [False] * 7]
    stats = sampled_statistics([ALL_WRONG, FORMAT_MIXED], truncated)
    assert stats["p0"] == 0.0
    assert stats["format_rate"] == pytest.approx(15 / 16)
    assert stats["mixed_group_fraction"] == 0.0
    assert stats["format_mixed_group_fraction"] == 0.5
    assert stats["truncation_rate"] == pytest.approx(1 / 16)
    assert (stats["n_problems"], stats["group_size"]) == (2, 8)


@pytest.mark.parametrize("p0, expected", [(0.05, True), (0.0999, True), (0.10, False), (0.3, False)])
def test_go_no_go_rule(p0, expected):
    assert decide_fewshot_prompt(p0, threshold=0.10) is expected


def test_preflight_config_matches_methodology():
    preflight = load_config().preflight
    assert preflight.p0_threshold == 0.10
    assert preflight.n_sampled_problems == 200
    assert preflight.truncation_alert == 0.10
    assert preflight.n_audit == 50
    assert preflight.fewshot_override is True  # METHODOLOGY §13


SAMPLED = {"B0": {"p0": 0.111}, "B0_fewshot": {"p0": 0.384}}


@pytest.mark.parametrize(
    "override, use_fewshot, p0, overridden",
    [(None, False, 0.111, False), (True, True, 0.384, True), (False, False, 0.111, False)],
)
def test_manifest_applies_rule_then_override(override, use_fewshot, p0, overridden):
    manifest = build_preflight_manifest(SAMPLED, {}, 0.10, override, [], "rev")
    assert manifest["rule_says_fewshot"] is False  # 0.111 >= 0.10
    assert (manifest["use_fewshot_prompt"], manifest["p0"], manifest["rule_overridden"]) == (use_fewshot, p0, overridden)


def test_rule_below_threshold_picks_fewshot_p0():
    manifest = build_preflight_manifest({"B0": {"p0": 0.05}, "B0_fewshot": {"p0": 0.3}}, {}, 0.10, None, [], "rev")
    assert manifest["use_fewshot_prompt"] and manifest["p0"] == 0.3 and not manifest["rule_overridden"]


# ---------------------------------------------------------------- manifest


def test_preflight_round_trip(tmp_path):
    path = tmp_path / "preflight.json"
    save_preflight({"p0": 0.3, "use_fewshot_prompt": True}, path)
    assert load_preflight(path) == {"p0": 0.3, "use_fewshot_prompt": True}


@pytest.mark.parametrize("p0", [0.0, 1.0])
def test_degenerate_p0_is_refused(tmp_path, p0):
    path = tmp_path / "preflight.json"
    save_preflight({"p0": p0, "use_fewshot_prompt": False}, path)
    with pytest.raises(ValueError):
        load_preflight(path)


# ---------------------------------------------------------------- audit sheet


def prediction_rows(n: int) -> list[dict]:
    return [
        {"problem_id": f"train-{i:05d}", "gold_answer": str(i), "completion": f"FINAL: {i}", "lenient_correct": i % 2 == 0}
        for i in range(n)
    ]


def test_audit_sample_is_reproducible_and_ordered():
    rows = prediction_rows(100)
    first = draw_audit_sample(rows, 10, seed=42)
    assert first == draw_audit_sample(rows, 10, seed=42)
    assert first != draw_audit_sample(rows, 10, seed=43)
    ids = [row["problem_id"] for row in first]
    assert ids == sorted(ids) and len(set(ids)) == 10


def test_audit_sheet_round_trip_and_agreement(tmp_path):
    rows = prediction_rows(4)
    path = tmp_path / "audit.csv"
    write_audit_sheet(rows, {row["problem_id"]: f"question {i}" for i, row in enumerate(rows)}, path)
    sheet = read_audit_sheet(path)
    assert list(sheet[0]) == ["problem_id", "question", "gold_answer", "completion", "human_correct", "grader_lenient"]
    assert [row["grader_lenient"] for row in sheet] == ["1", "0", "1", "0"]
    for row, human in zip(sheet, ["1", "0", "0", "0"], strict=True):
        row["human_correct"] = human
    agreement, disagreements = grader_agreement(sheet)
    assert agreement == 0.75
    assert [row["problem_id"] for row in disagreements] == ["train-00002"]


def test_agreement_refuses_an_unfilled_sheet(tmp_path):
    path = tmp_path / "audit.csv"
    write_audit_sheet(prediction_rows(2), {"train-00000": "q", "train-00001": "q"}, path)
    with pytest.raises(ValueError):
        grader_agreement(read_audit_sheet(path))
