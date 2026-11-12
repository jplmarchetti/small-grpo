# Format vs. correctness in GRPO post-training

When GRPO is applied to a base language model on math problems, its benchmark score rises quickly. Part of that gain may only be the model learning to **write the answer where the grader looks for it**, not **answering more problems correctly**. This project measures how much is each.

**Status (2026-11-12):** design frozen, implementation not started. There is no training code, environment or measured GPU number yet.

## The experiment in short

The same base model is trained four times with GRPO. A **format reward** and a **correctness reward** are switched on and off independently:

|  | Correctness reward off | Correctness reward on |
|---|---|---|
| **Format reward off** | **R0**: random reward | **RC**: answer is right |
| **Format reward on** | **RF**: answer is in the `FINAL: <number>` format | **RFC**: both |

Every model is then scored on the GSM8K test set by two automatic graders: a **strict** one (right number, in the required format) and a **lenient** one (right number, anywhere). The total gain splits exactly into a **format gain** and a **correctness gain**.

| Item | Choice |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B` (base, not Instruct), LoRA r=16 on all linear layers, BF16 |
| Task | GSM8K `main`: 500 validation, 1,000 training, 1,319 test problems |
| Algorithm | GRPO written from scratch in PyTorch; 2 problems × 8 completions per step, 500 steps, no KL term |
| Reference models | B0 (untrained) and B0-fewshot (4 worked examples in the prompt) |
| Seeds | Target 3 per cell (42, 43, 44); fallback 1 |
| Hardware | One RTX 5070 (12 GB VRAM), 32 GB RAM |

## Documents

| File | What it is |
|---|---|
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | The experimental design. Source of truth for every scientific choice. |
| [docs/2026-11-12_format-vs-correctness-grpo_pipeline.md](docs/2026-11-12_format-vs-correctness-grpo_pipeline.md) | The implementation blueprint: 16 stages, each with intuition, pseudo-code, function names and a check. |
| [docs/SLIDES_PLAN.md](docs/SLIDES_PLAN.md) | Plan for the presentation. |
| [research-assignment-latex/](research-assignment-latex/) | The paper (IEEE template) and its references. |

## Planned layout

Nothing below exists yet; the blueprint defines it.

```text
configs/base.yaml      fixed settings and the chosen learning rate
manifests/             committed: data splits, model/dataset revisions, pre-flight results
src/grpo_gsm8k/        the package: data, prompts, graders, policy, rollout, evaluate,
                       preflight, rewards, grpo, train, stats, report, audit
scripts/               one thin entry point per stage that produces files
tests/                 graders, rewards, GRPO core, statistics
notebooks/             explain each stage by importing from src/
outputs/  reports/     generated, not committed
```

## Order of work

1. Splits, prompt and graders, with tests (laptop). The graders are frozen with a git tag before any training.
2. Rewards, GRPO core and statistics, with tests on a tiny model and on made-up numbers (laptop).
3. Model loading, generation and evaluation; then the **pre-flight** on validation: B0, B0-fewshot, and the base success rate `p0` (GPU).
4. Training loop, a 20-step smoke test, then a learning-rate pilot over {1e-5, 3e-5, 1e-4} (GPU).
5. The four cells for seed 42, test evaluation, tables and figures. The project has a complete result here.
6. Seeds 43 and 44, then the hand-check of the lenient grader.

## Hardware and cost (estimates, to be measured by the smoke test)

- **Memory:** about 3–5 GB peak. The weights take about 1 GB in BF16. The expensive part is the log-probability pass, which is done 4 sequences at a time: all 16 at once would need about 6 GB for the output table alone.
- **Time:** about 1–1.5 h per 500-step run, so 4–6 GPU-hours per seed, plus evaluation.
- **GPU:** the RTX 5070 needs a PyTorch build with CUDA 12.8 or newer.

## Where things run

- **Laptop:** editing, tests, splits, statistics, tables and figures. Use `uv run ...`, Python 3.11.
- **GPU PC:** anything that loads the model (evaluation, pre-flight, training). Do not run these on the laptop.

`small-model-grpo/` is a member of the parent `master-rl` uv workspace but does not have its own `pyproject.toml` yet. Creating it, with the training dependencies kept apart from the coursework ones, is the first implementation step.

## Attribution

The GRPO core follows the structure of chapters 6–7 of Sebastian Raschka's [reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) (Apache-2.0). Adapted functions carry an attribution comment. This project differs in what it measures: a controlled 2×2 comparison with separate graders, held-out test data and bootstrap intervals, using LoRA on Hugging Face models.
