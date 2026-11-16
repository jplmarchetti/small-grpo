import pytest

from grpo_gsm8k.audit import draw_audit_sample, draw_format_failure_slice, pool_seed_rows


def rows(n, format_every=3):
    return [{"problem_id": f"test-{i:05d}", "format_ok": i % format_every != 0, "lenient_correct": True}
            for i in range(n)]


def test_pooling_tags_every_row_with_its_seed():
    pooled = pool_seed_rows({43: rows(2), 42: rows(2)})
    assert [r["problem_id"] for r in pooled] == ["test-00000/seed42", "test-00001/seed42",
                                                 "test-00000/seed43", "test-00001/seed43"]


def test_slice_holds_only_format_failures_outside_the_uniform_sample():
    pooled = pool_seed_rows({42: rows(60), 43: rows(60)})
    uniform = draw_audit_sample(pooled, 25, seed=42)
    slice_ = draw_format_failure_slice(pooled, 20, seed=42, exclude=uniform)
    assert len(slice_) == 20
    assert all(not r["format_ok"] for r in slice_)
    assert not {r["problem_id"] for r in slice_} & {r["problem_id"] for r in uniform}
    assert slice_ == draw_format_failure_slice(pooled, 20, seed=42, exclude=uniform)  # reproducible


def test_slice_refuses_when_too_few_failures():
    with pytest.raises(ValueError):
        draw_format_failure_slice(pool_seed_rows({42: rows(9)}), 5, seed=0, exclude=[])
