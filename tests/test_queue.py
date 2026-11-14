import json
import subprocess

import pytest

from grpo_gsm8k.frozen import FROZEN_FILES, FROZEN_PATH, changed_frozen_files, sha256_of_bytes
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.queue import run_dir, run_is_complete, run_queue


def finish(directory, steps=500):
    checkpoint = directory / "checkpoints" / f"step_{steps:03d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "training_state.pt").write_bytes(b"x")


# ---------------------------------------------------------------- queue


def test_run_is_complete_only_with_the_final_checkpoint(tmp_path):
    directory = run_dir("RC", 42, tmp_path)
    assert not run_is_complete(directory, 500)
    finish(directory, steps=450)
    assert not run_is_complete(directory, 500)
    finish(directory)
    assert run_is_complete(directory, 500)


def test_queue_order_skip_and_failure(tmp_path):
    finish(run_dir("RF", 42, tmp_path))
    calls = []

    def fake_train(cell, seed):
        calls.append((cell, seed))
        if cell == "R0":
            return 1  # crashes
        finish(run_dir(cell, seed, tmp_path))
        return 0

    status = run_queue([42], ["RC", "RFC", "RF", "R0"], 500, fake_train, tmp_path)
    assert calls == [("RC", 42), ("RFC", 42), ("R0", 42)]  # RF skipped, order kept
    assert status[("RF", 42)] == "already complete"
    assert status[("RC", 42)].startswith("complete")
    assert status[("R0", 42)].startswith("FAILED")
    assert (run_dir("R0", 42, tmp_path) / "error.txt").exists()


def test_queue_finishes_a_seed_before_the_next(tmp_path):
    calls = []

    def fake_train(cell, seed):
        calls.append((cell, seed))
        finish(run_dir(cell, seed, tmp_path))
        return 0

    run_queue([42, 43], ["RC", "R0"], 500, fake_train, tmp_path)
    assert calls == [("RC", 42), ("R0", 42), ("RC", 43), ("R0", 43)]


def test_a_run_that_exits_cleanly_without_its_final_checkpoint_counts_as_failed(tmp_path):
    status = run_queue([42], ["RC"], 500, lambda cell, seed: 0, tmp_path)
    assert status[("RC", 42)].startswith("FAILED")


# ---------------------------------------------------------------- frozen files


def test_frozen_files_are_unchanged_since_their_tags():
    assert changed_frozen_files() == []


def test_frozen_manifest_matches_the_tagged_commits():
    """Run on the laptop, where .git exists: the recorded hashes are the tagged versions."""
    tags_resolve = all(
        subprocess.run(["git", "rev-parse", "--verify", "--quiet", tag], cwd=REPO_ROOT, capture_output=True).returncode == 0
        for tag in FROZEN_FILES
    )
    if not tags_resolve:
        pytest.skip("git tags not available here (e.g. the GPU PC mirror)")
    recorded = json.loads(FROZEN_PATH.read_text())
    assert set(recorded) == set(FROZEN_FILES)
    for tag, files in FROZEN_FILES.items():
        for path in files:
            tagged = subprocess.run(["git", "show", f"{tag}:{path}"], cwd=REPO_ROOT, capture_output=True, check=True).stdout
            assert recorded[tag][path] == sha256_of_bytes(tagged), f"{tag}:{path}"


def test_a_changed_file_is_detected(tmp_path):
    (tmp_path / "a.txt").write_text("frozen")
    manifest = tmp_path / "frozen.json"
    manifest.write_text(json.dumps({"tag-v1": {"a.txt": sha256_of_bytes(b"frozen")}}))
    assert changed_frozen_files(tmp_path, manifest) == []
    (tmp_path / "a.txt").write_text("edited")
    assert changed_frozen_files(tmp_path, manifest) == ["tag-v1: a.txt"]
