import json

from grpo_gsm8k.config import load_config
from grpo_gsm8k.paths import MANIFEST_DIR


def test_base_config_matches_methodology():
    config = load_config()
    assert config.model.name == "Qwen/Qwen2.5-0.5B"
    assert len(config.model.revision) == 40 and len(config.data.revision) == 40  # pinned commit hashes
    assert (config.data.dataset, config.data.config) == ("openai/gsm8k", "main")
    assert (config.data.validation_size, config.data.train_size, config.data.fewshot_size) == (500, 1000, 4)
    assert config.data.split_seed == 42
    assert (config.lora.rank, config.lora.alpha, config.lora.dropout) == (16, 32, 0.0)
    training = config.training
    assert training.problems_per_step * training.group_size == 16
    # config-v2 (METHODOLOGY §13, 2026-11-15): 150 steps, 300 of the 1,000 training problems
    assert training.steps * training.problems_per_step <= config.data.train_size
    assert training.steps % training.checkpoint_every == 0
    assert (training.temperature, training.top_p, training.max_new_tokens) == (0.8, 0.95, 512)
    assert training.seeds == (42, 43, 44)
    # config-v2: 1e-5, a rate the LR pilot found stable (config-v1 used the pilot's choice, 3e-5)
    pilot = json.loads((MANIFEST_DIR / "lr_pilot.json").read_text())
    stable = {verdict["learning_rate"] for verdict in pilot["verdicts"] if not verdict["collapsed"]}
    assert training.learning_rate == 1e-5 and training.learning_rate in stable
    assert config.config_tag == "config-v2"


def test_analysis_config_matches_methodology():
    analysis = load_config().analysis
    assert analysis.n_resamples == 10_000
    assert isinstance(analysis.analysis_seed, int)
