import numpy as np
import pytest

from grpo_gsm8k.run_log import (
    STEP_LOG_COLUMNS,
    StepLog,
    append_sample_completions,
    append_step_log,
    read_step_logs,
    truncate_step_log,
)
from grpo_gsm8k.training_plots import cumulative_mean, moving_average, plot_training_curves


def make_log(step: int, reward: float = 0.5) -> StepLog:
    return StepLog(
        step=step, mean_reward=reward, format_rate=0.9, lenient_rate=0.4, strict_rate=0.38, mean_tokens=100.0,
        truncation_rate=0.0, zero_variance_groups=0.5, groups_format_only=0, groups_correctness_only=1,
        groups_both=0, groups_none=1, skipped_update=False, loss=-0.01, grad_norm=0.3, seconds=4.0, peak_vram_gb=5.0,
    )


def test_step_log_appends_with_one_header(tmp_path):
    path = tmp_path / "train_log.csv"
    for step in (1, 2, 3):
        append_step_log(path, make_log(step))
    rows = read_step_logs(path)
    assert [row["step"] for row in rows] == ["1", "2", "3"]
    assert tuple(rows[0]) == STEP_LOG_COLUMNS
    assert path.read_text().count("mean_reward") == 1


def test_truncate_drops_rows_after_the_resumed_step(tmp_path):
    path = tmp_path / "train_log.csv"
    for step in range(1, 8):
        append_step_log(path, make_log(step))
    assert truncate_step_log(path, last_step=5) == 2
    assert [row["step"] for row in read_step_logs(path)] == ["1", "2", "3", "4", "5"]
    append_step_log(path, make_log(6))
    assert [row["step"] for row in read_step_logs(path)][-1] == "6"


def test_truncate_without_a_log_is_harmless(tmp_path):
    assert truncate_step_log(tmp_path / "missing.csv", 3) == 0


def test_sample_completions_are_readable(tmp_path):
    path = tmp_path / "samples.txt"
    append_sample_completions(path, 10, ["What is 2+2?"], ["2 + 2 = 4\nFINAL: 4"], [1.0])
    text = path.read_text()
    assert "step 10" in text and "FINAL: 4" in text and "reward 1" in text


def test_moving_average_and_cumulative_mean():
    values = [0, 1, 0, 1, 1, 1]
    assert moving_average(values, 2).tolist() == pytest.approx([0, 0.5, 0.5, 0.5, 1, 1])
    assert cumulative_mean(values).tolist() == pytest.approx([0, 0.5, 1 / 3, 0.5, 0.6, 4 / 6])
    assert moving_average(values, 100).tolist() == pytest.approx(cumulative_mean(values).tolist())


def test_plot_writes_a_png(tmp_path):
    rng = np.random.default_rng(0)
    for step in range(1, 41):
        append_step_log(tmp_path / "train_log.csv", make_log(step, reward=float(rng.random())))
    (tmp_path / "val_curve.csv").write_text(
        "step,lenient_accuracy,strict_accuracy,format_rate,mean_tokens,truncation_rate\n0,0.49,0.49,0.97,100,0\n"
    )
    path = plot_training_curves(tmp_path, window=5)
    assert path.exists() and path.stat().st_size > 10_000


def test_lr_pilot_plot(tmp_path):
    from grpo_gsm8k.lr_pilot import pilot_run_dir
    from grpo_gsm8k.training_plots import plot_lr_pilot

    for lr in (1e-5, 3e-5):
        run_dir = pilot_run_dir(tmp_path, lr)
        for step in range(1, 21):
            append_step_log(run_dir / "train_log.csv", make_log(step))
        (run_dir / "val_curve.csv").write_text(
            "step,lenient_accuracy,strict_accuracy,format_rate,mean_tokens,truncation_rate\n"
            "0,0.49,0.49,0.97,100,0\n10,0.5,0.5,0.97,98,0\n"
        )
    assert plot_lr_pilot(tmp_path, (1e-5, 3e-5)).exists()
