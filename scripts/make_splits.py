"""Stage 1: freeze the GSM8K splits into manifests/ (run once, on the laptop).

    uv run python scripts/make_splits.py

Downloads the pinned GSM8K files into data/, drops repeated train questions,
shuffles once with the configured seed, and writes manifests/splits.json and
manifests/revisions.json. Running it twice must give identical files.
"""

from grpo_gsm8k.config import load_config
from grpo_gsm8k.data import (
    DatasetSource,
    download_gsm8k_file,
    drop_duplicate_questions,
    file_sha256,
    make_splits,
    read_problems,
    save_revisions,
    save_split_manifest,
    split_sizes,
)
from grpo_gsm8k.paths import REVISIONS_PATH, SPLITS_PATH

HUB_SPLITS = ("train", "test")


def main() -> None:
    config = load_config()
    source = DatasetSource.from_config(config.data)
    parquet_paths = {hub_split: download_gsm8k_file(source, hub_split) for hub_split in HUB_SPLITS}

    train_problems = read_problems(parquet_paths["train"], "train")
    test_problems = read_problems(parquet_paths["test"], "test")
    unique_train_problems, n_duplicates_dropped = drop_duplicate_questions(train_problems)
    splits = make_splits(unique_train_problems, config.data.split_seed, split_sizes(config.data))

    save_split_manifest(splits, len(train_problems), n_duplicates_dropped, config.data.split_seed)
    file_hashes = {hub_split: file_sha256(path) for hub_split, path in parquet_paths.items()}
    save_revisions(source, file_hashes, config.model.name, config.model.revision)

    print(f"GSM8K train: {len(train_problems)} problems, {n_duplicates_dropped} duplicate questions dropped")
    print(f"GSM8K test: {len(test_problems)} problems (not split, not touched)")
    for name, problem_ids in splits.items():
        print(f"{name}: {len(problem_ids)}")
    print(f"wrote {SPLITS_PATH} and {REVISIONS_PATH}")


if __name__ == "__main__":
    main()
