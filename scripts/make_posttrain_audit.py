"""Stage 16: draw the post-training hand-check sheets (laptop, METHODOLOGY §11 and §13).

    rsync outputs/config-v2/test/*/seed*/predictions.jsonl from the GPU PC first
    uv run python scripts/make_posttrain_audit.py

Writes three sheets in audits/, from the test completions of the reported (step-150) models,
pooled over all seeds:
- posttrain_RC.csv and posttrain_RFC.csv: uniform samples, the honest agreement rate;
- posttrain_RC_no_format.csv: RC completions without the FINAL line (the targeted slice),
  reported apart and never pooled with the uniform rate.
The question column is left empty: the test split is read only by run_final_eval.py, and the
judgement "is the final answer the gold answer" needs only the gold answer and the completion.
"""

from grpo_gsm8k.config import load_config
from grpo_gsm8k.audit import draw_audit_sample, draw_format_failure_slice, pool_seed_rows, write_audit_sheet
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.stats import read_prediction_rows

AUDIT_DIR = REPO_ROOT / "audits"
N_UNIFORM = 25  # per cell (METHODOLOGY §11)
N_NO_FORMAT = 20  # targeted RC slice (METHODOLOGY §13, 2026-11-16)


def main() -> None:
    config = load_config()
    test_root = REPO_ROOT / "outputs" / config.config_tag / "test"
    seed = config.preflight.audit_seed
    pooled = {
        cell: pool_seed_rows({s: read_prediction_rows(test_root / cell / f"seed{s}" / "predictions.jsonl")
                              for s in config.training.seeds})
        for cell in ("RC", "RFC")
    }
    sheets = {cell: draw_audit_sample(rows, N_UNIFORM, seed) for cell, rows in pooled.items()}
    sheets["RC_no_format"] = draw_format_failure_slice(pooled["RC"], N_NO_FORMAT, seed, exclude=sheets["RC"])
    for name, sample in sheets.items():
        path = AUDIT_DIR / f"posttrain_{name}.csv"
        write_audit_sheet(sample, {row["problem_id"]: "" for row in sample}, path)
        print(f"wrote {path} ({len(sample)} completions)")


if __name__ == "__main__":
    main()
