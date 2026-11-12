import json

import pytest

from grpo_gsm8k.config import load_config
from grpo_gsm8k.data import (
    SPLIT_NAMES,
    Problem,
    drop_duplicate_questions,
    extract_gold_answer,
    load_split,
    make_splits,
    save_split_manifest,
    split_sizes,
)
from grpo_gsm8k.paths import SPLITS_PATH


def make_problems(n: int) -> list[Problem]:
    return [Problem(f"train-{i:05d}", f"question {i}", str(i), f"... #### {i}") for i in range(n)]


SMALL_SIZES = {"validation": 5, "train": 10, "fewshot_exemplars": 2}


# ---------------------------------------------------------------- gold answer


@pytest.mark.parametrize(
    "solution, expected",
    [
        ("She has 3 + 5 = <<3+5=8>>8 apples.\n#### 8", "8"),
        ("...\n#### 1,000", "1,000"),
        ("...\n#### -4 ", "-4"),
        ("a #### b\n#### 12", "12"),  # the last marker wins
    ],
)
def test_extract_gold_answer(solution, expected):
    assert extract_gold_answer(solution) == expected


def test_extract_gold_answer_without_marker_fails():
    with pytest.raises(ValueError):
        extract_gold_answer("no answer here")


# ---------------------------------------------------------------- duplicates


def test_drop_duplicate_questions_keeps_first_and_counts():
    problems = make_problems(3) + [Problem("train-00099", "question 1", "1", "#### 1")]
    kept, n_dropped = drop_duplicate_questions(problems)
    assert n_dropped == 1
    assert [p.problem_id for p in kept] == ["train-00000", "train-00001", "train-00002"]


# ---------------------------------------------------------------- splits


def test_make_splits_sizes_and_disjoint():
    splits = make_splits(make_problems(30), seed=42, sizes=SMALL_SIZES)
    assert {name: len(ids) for name, ids in splits.items()} == SMALL_SIZES
    all_ids = [i for ids in splits.values() for i in ids]
    assert len(all_ids) == len(set(all_ids))


def test_make_splits_is_deterministic_and_seed_dependent():
    problems = make_problems(30)
    assert make_splits(problems, 42, SMALL_SIZES) == make_splits(problems, 42, SMALL_SIZES)
    assert make_splits(problems, 42, SMALL_SIZES) != make_splits(problems, 43, SMALL_SIZES)


def test_make_splits_does_not_mutate_input():
    problems = make_problems(30)
    before = list(problems)
    make_splits(problems, 42, SMALL_SIZES)
    assert problems == before


def test_make_splits_refuses_too_few_problems():
    with pytest.raises(ValueError):
        make_splits(make_problems(16), 42, SMALL_SIZES)


# ---------------------------------------------------------------- manifest round trip


def test_manifest_is_byte_identical_when_saved_twice(tmp_path):
    splits = make_splits(make_problems(30), 42, SMALL_SIZES)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    save_split_manifest(splits, 30, 0, 42, first)
    save_split_manifest(splits, 30, 0, 42, second)
    assert first.read_bytes() == second.read_bytes()


def test_load_split_refuses_test():
    with pytest.raises(ValueError):
        load_split("test")


# ---------------------------------------------------------------- the committed manifest


@pytest.mark.skipif(not SPLITS_PATH.exists(), reason="manifest not generated yet")
def test_committed_manifest_matches_config():
    manifest = json.loads(SPLITS_PATH.read_text())
    config = load_config()
    expected_sizes = split_sizes(config.data)
    assert manifest["seed"] == config.data.split_seed
    assert set(manifest["splits"]) == set(SPLIT_NAMES)
    assert {name: len(ids) for name, ids in manifest["splits"].items()} == expected_sizes
    all_ids = [i for ids in manifest["splits"].values() for i in ids]
    assert len(all_ids) == len(set(all_ids))
    assert all(i.startswith("train-") for i in all_ids)
