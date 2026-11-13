from grpo_gsm8k.config import load_config


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
    assert training.steps * training.problems_per_step == config.data.train_size  # one pass
    assert (training.temperature, training.top_p, training.max_new_tokens) == (0.8, 0.95, 512)
    assert training.seeds == (42, 43, 44)
    assert training.learning_rate is None  # set only by the LR pilot


def test_analysis_config_matches_methodology():
    analysis = load_config().analysis
    assert analysis.n_resamples == 10_000
    assert isinstance(analysis.analysis_seed, int)
