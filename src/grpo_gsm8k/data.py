"""Stage 1: GSM8K problems and the frozen splits.

The splits are decided once by `scripts/make_splits.py` and saved as lists of
problem ids in `manifests/splits.json`. Every later stage reads that file through
`load_split`; nothing reshuffles. Only `scripts/run_final_eval.py` may call
`load_test_problems`.
"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from grpo_gsm8k.config import DataConfig
from grpo_gsm8k.paths import HF_CACHE_DIR, REVISIONS_PATH, SPLITS_PATH

GOLD_ANSWER_MARKER = "####"
SPLIT_NAMES = ("validation", "train", "fewshot_exemplars")  # carved from GSM8K train, in this order


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    gold_answer: str
    solution: str  # full GSM8K solution; only used to build few-shot exemplars


# ---------------------------------------------------------------- parsing


def extract_gold_answer(solution_text: str) -> str:
    """Return the text after the last `####`, stripped. Number cleaning is the grader's job."""
    if GOLD_ANSWER_MARKER not in solution_text:
        raise ValueError(f"no '{GOLD_ANSWER_MARKER}' in solution: {solution_text[-80:]!r}")
    return solution_text.rsplit(GOLD_ANSWER_MARKER, 1)[1].strip()


def make_problem_id(hub_split: str, row_index: int) -> str:
    """Ids are the row position in the pinned parquet file, e.g. `train-00042`."""
    return f"{hub_split}-{row_index:05d}"


def read_problems(parquet_path: Path, hub_split: str) -> list[Problem]:
    table = pq.read_table(parquet_path, columns=["question", "answer"])
    questions = table.column("question").to_pylist()
    solutions = table.column("answer").to_pylist()
    return [
        Problem(
            problem_id=make_problem_id(hub_split, index),
            question=question,
            gold_answer=extract_gold_answer(solution),
            solution=solution,
        )
        for index, (question, solution) in enumerate(zip(questions, solutions, strict=True))
    ]


# ---------------------------------------------------------------- splitting


def drop_duplicate_questions(problems: list[Problem]) -> tuple[list[Problem], int]:
    """Keep the first problem of every exact-repeat question; return them and the drop count."""
    seen_questions: set[str] = set()
    kept = []
    for problem in problems:
        if problem.question in seen_questions:
            continue
        seen_questions.add(problem.question)
        kept.append(problem)
    return kept, len(problems) - len(kept)


def split_sizes(data_config: DataConfig) -> dict[str, int]:
    return {
        "validation": data_config.validation_size,
        "train": data_config.train_size,
        "fewshot_exemplars": data_config.fewshot_size,
    }


def make_splits(problems: list[Problem], seed: int, sizes: dict[str, int]) -> dict[str, list[str]]:
    """Shuffle once with `seed` and slice consecutive pieces in `SPLIT_NAMES` order."""
    needed = sum(sizes[name] for name in SPLIT_NAMES)
    if needed > len(problems):
        raise ValueError(f"need {needed} problems, have {len(problems)}")
    shuffled_ids = [problem.problem_id for problem in problems]
    random.Random(seed).shuffle(shuffled_ids)
    splits, start = {}, 0
    for name in SPLIT_NAMES:
        splits[name] = shuffled_ids[start : start + sizes[name]]
        start += sizes[name]
    return splits


# ---------------------------------------------------------------- downloads


@dataclass(frozen=True)
class DatasetSource:
    """Where one pinned copy of GSM8K lives on the Hub."""

    repo: str
    config: str
    revision: str

    @classmethod
    def from_config(cls, data_config: DataConfig) -> "DatasetSource":
        return cls(data_config.dataset, data_config.config, data_config.revision)

    def parquet_filename(self, hub_split: str) -> str:
        return f"{self.config}/{hub_split}-00000-of-00001.parquet"


def download_gsm8k_file(source: DatasetSource, hub_split: str) -> Path:
    """Fetch one GSM8K parquet file at the pinned revision (cached under data/)."""
    path = hf_hub_download(
        repo_id=source.repo,
        filename=source.parquet_filename(hub_split),
        repo_type="dataset",
        revision=source.revision,
        cache_dir=HF_CACHE_DIR,
    )
    return Path(path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- manifests


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")


def save_split_manifest(
    splits: dict[str, list[str]],
    n_train_problems: int,
    n_duplicates_dropped: int,
    seed: int,
    path: Path = SPLITS_PATH,
) -> None:
    write_json(
        path,
        {
            "seed": seed,
            "n_train_problems": n_train_problems,
            "n_duplicates_dropped": n_duplicates_dropped,
            "splits": splits,
        },
    )


def save_revisions(
    source: DatasetSource,
    file_hashes: dict[str, str],
    model_name: str,
    model_revision: str,
    path: Path = REVISIONS_PATH,
) -> None:
    write_json(
        path,
        {
            "dataset": {
                "repo": source.repo,
                "config": source.config,
                "revision": source.revision,
                "sha256": file_hashes,
            },
            "model": {"repo": model_name, "revision": model_revision},
        },
    )


# ---------------------------------------------------------------- loading (later stages)


def load_hub_split(hub_split: str, revisions_path: Path = REVISIONS_PATH) -> list[Problem]:
    """Read one GSM8K file at the revision recorded in the manifest, refusing a changed file."""
    revisions = json.loads(revisions_path.read_text())["dataset"]
    source = DatasetSource(revisions["repo"], revisions["config"], revisions["revision"])
    path = download_gsm8k_file(source, hub_split)
    expected_hash = revisions["sha256"][hub_split]
    if file_sha256(path) != expected_hash:
        raise RuntimeError(f"{path} does not match the sha256 in {revisions_path}")
    return read_problems(path, hub_split)


def load_split(name: str, splits_path: Path = SPLITS_PATH, revisions_path: Path = REVISIONS_PATH) -> list[Problem]:
    """Problems of one frozen split, in manifest order. The test split is not available here."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}; expected one of {SPLIT_NAMES} (test: load_test_problems)")
    split_ids = json.loads(splits_path.read_text())["splits"][name]
    problems_by_id = {problem.problem_id: problem for problem in load_hub_split("train", revisions_path)}
    return [problems_by_id[problem_id] for problem_id in split_ids]


def load_test_problems(revisions_path: Path = REVISIONS_PATH) -> list[Problem]:
    """All 1,319 test problems. Only scripts/run_final_eval.py may call this."""
    return load_hub_split("test", revisions_path)
