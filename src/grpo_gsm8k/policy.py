"""Stage 4: load the policy (base model, optionally with a LoRA adapter).

- B0 is the base model with no adapter.
- A fresh adapter starts as "no change" (LoRA's B matrix is zero), so step 0 of every
  run equals B0. The seed is set right before the adapter is created, so all four
  cells of one seed start from the same adapter.
- Evaluation and training both build the model through `load_policy`.
"""

from pathlib import Path

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from grpo_gsm8k.config import LoraConfig, ModelConfig
from grpo_gsm8k.paths import HF_CACHE_DIR

MODEL_DTYPE = torch.bfloat16


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizerBase:
    """Left padding, needed to generate in batches."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name, revision=model_config.revision, cache_dir=HF_CACHE_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def attach_fresh_adapter(model: torch.nn.Module, lora_config: LoraConfig, lora_seed: int) -> PeftModel:
    torch.manual_seed(lora_seed)
    peft_config = PeftLoraConfig(
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        task_type="CAUSAL_LM",
    )
    policy = get_peft_model(model, peft_config)
    policy.print_trainable_parameters()
    return policy


def load_policy(
    model_config: ModelConfig,
    lora_config: LoraConfig | None = None,
    adapter_dir: Path | None = None,
    lora_seed: int | None = None,
    device: torch.device | None = None,
) -> tuple[torch.nn.Module, PreTrainedTokenizerBase]:
    """B0 (no adapter), a saved adapter (`adapter_dir`), or a fresh one (`lora_config` + `lora_seed`)."""
    if adapter_dir is not None and lora_seed is not None:
        raise ValueError("pass either adapter_dir (saved adapter) or lora_seed (fresh adapter), not both")
    tokenizer = load_tokenizer(model_config)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name, revision=model_config.revision, dtype=MODEL_DTYPE, cache_dir=HF_CACHE_DIR
    ).to(device or default_device())
    if adapter_dir is not None:
        return PeftModel.from_pretrained(model, adapter_dir, is_trainable=True), tokenizer
    if lora_seed is not None:
        if lora_config is None:
            raise ValueError("a fresh adapter needs lora_config")
        return attach_fresh_adapter(model, lora_config, lora_seed), tokenizer
    return model, tokenizer


def save_adapter(policy: PeftModel, directory: Path) -> None:
    """Only the LoRA weights (a few MB), not the base model."""
    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
