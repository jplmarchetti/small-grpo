"""Typed view of configs/base.yaml.

The YAML file is the only place where fixed settings are written. Code reads them
through `load_config` so that no value is typed twice.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from grpo_gsm8k.paths import BASE_CONFIG_PATH


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    config: str
    revision: str
    split_seed: int
    validation_size: int
    train_size: int
    fewshot_size: int


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: int
    dropout: float
    target_modules: str


@dataclass(frozen=True)
class TrainingConfig:
    problems_per_step: int
    group_size: int
    steps: int
    temperature: float
    top_p: float
    max_new_tokens: int
    weight_decay: float
    grad_clip_norm: float
    advantage_eps: float
    micro_batch_size: int
    checkpoint_every: int
    seeds: tuple[int, ...]
    learning_rate: float | None  # None until the LR pilot has chosen it


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int  # fixed for every model, so all are evaluated with the same batches


@dataclass(frozen=True)
class AnalysisConfig:
    n_resamples: int
    analysis_seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: DataConfig
    lora: LoraConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    analysis: AnalysisConfig


def load_config(path: Path = BASE_CONFIG_PATH) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text())
    training = dict(raw["training"])
    training["seeds"] = tuple(training["seeds"])
    return ExperimentConfig(
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        lora=LoraConfig(**raw["lora"]),
        training=TrainingConfig(**training),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        analysis=AnalysisConfig(**raw["analysis"]),
    )
