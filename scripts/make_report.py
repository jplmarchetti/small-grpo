"""Stage 15: write the report from saved outputs (laptop, no GPU).

    rsync the small files (predictions, metrics, train_log.csv, val_curve.csv) into outputs/
    uv run python scripts/make_report.py

Reads outputs/<config_tag>/ (and outputs/ for the collapsed config-v1 runs) and writes
reports/<config_tag>/report.md plus its figures. It only reads and redraws: re-running it
on unchanged outputs gives the same files.
"""

import json
from pathlib import Path

from grpo_gsm8k.audit import grader_agreement, read_audit_sheet
from grpo_gsm8k.config import load_config
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.queue import run_dir
from grpo_gsm8k.report import (
    BASELINE,
    CELLS,
    GRADERS,
    analyze,
    format_headroom,
    load_test_predictions,
    make_cost_table,
    make_effects_table,
    make_results_table,
    make_strict_sources_table,
    pct,
    pick_example_problem,
    plot_config_v1_drift,
    plot_gain_split,
    plot_strict_gain_sources,
    plot_validation_curves,
    read_csv_rows,
    show_example_problem,
    strict_gain_sources_per_seed,
)
from grpo_gsm8k.stability_gate import load_reported_step

CONFIG_V1_ROOT = REPO_ROOT / "outputs"  # config-v1 runs predate the outputs/<config_tag>/ layout
EXAMPLE_SEED = 42


def config_v1_test_table() -> str:
    rows = []
    for name, folder in (("B0, few-shot", "B0_fewshot"), *((cell, f"{cell}/seed42") for cell in CELLS)):
        metrics = json.loads((CONFIG_V1_ROOT / "test" / folder / "metrics.json").read_text())
        rows.append(f"| {name} | {pct(metrics['lenient_accuracy'])} | {pct(metrics['strict_accuracy'])} | "
                    f"{pct(metrics['format_rate'])} | {pct(metrics['truncation_rate'])} |")
    return "\n".join(["| Model | Lenient % | Strict % | Format % | Truncated % |", "|---|---|---|---|---|", *rows])


HAND_CHECKS = (
    ("B0, few-shot, before training (validation)", "preflight_B0_fewshot.csv"),
    ("RC, step 150, uniform (test)", "posttrain_RC.csv"),
    ("RFC, step 150, uniform (test)", "posttrain_RFC.csv"),
    ("RC without FINAL line, targeted slice (test)", "posttrain_RC_no_format.csv"),
)


def hand_check_table() -> str:
    """Agreement of a blind human judgement with the lenient grader, one row per sheet."""
    rows = []
    for label, file in HAND_CHECKS:
        sheet = read_audit_sheet(REPO_ROOT / "audits" / file)
        rate, disagreements = grader_agreement(sheet)
        rows.append(f"| {label} | {len(sheet) - len(disagreements)} / {len(sheet)} | {100 * rate:.0f}% |")
    return "\n".join(["| Sample | Agree | Rate |", "|---|---|---|", *rows])


def main() -> None:
    config = load_config()
    seeds = config.training.seeds
    reported_step = load_reported_step(config.config_tag)
    out_dir = REPO_ROOT / "reports" / config.config_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_test_predictions(REPO_ROOT / "outputs" / config.config_tag / "test", seeds)
    results = {g: analyze(predictions, g, config.analysis.n_resamples, config.analysis.analysis_seed) for g in GRADERS}
    headroom = format_headroom(predictions.baselines[BASELINE])
    sources = strict_gain_sources_per_seed(predictions)
    val_curves = {(c, s): read_csv_rows(run_dir(c, s) / "val_curve.csv") for c in CELLS for s in seeds}
    train_logs = {(c, s): read_csv_rows(run_dir(c, s) / "train_log.csv") for c in CELLS for s in seeds}
    v1_r0 = read_csv_rows(CONFIG_V1_ROOT / "R0" / "seed42" / "val_curve.csv")
    example_id = pick_example_problem(predictions, EXAMPLE_SEED)

    plot_gain_split(results, seeds, out_dir / "gain_split.png")
    plot_validation_curves(val_curves, out_dir / "validation_curves.png")
    plot_strict_gain_sources(sources, out_dir / "strict_gain_sources.png")
    plot_config_v1_drift(v1_r0, {s: val_curves[("R0", s)] for s in seeds}, out_dir / "config_v1_drift.png")

    lenient, strict = results["lenient"], results["strict"]
    seed_list = " / ".join(str(s) for s in seeds)
    report = f"""# Results: {config.config_tag}

GSM8K test set (1,319 problems), greedy decoding. Every trained model is the step-{reported_step} adapter
(fixed by the stability gate, METHODOLOGY §13). Cells are averaged over seeds {seed_list}.
B0 is the base model with the trained (few-shot) prompt.

## Results

{make_results_table(predictions, results)}

Intervals: 95% paired bootstrap over test problems ({config.analysis.n_resamples:,} resamples), after averaging
seeds; they do not include seed-to-seed variation, which the per-seed column shows.

**What prompting alone buys:** the few-shot prompt raises lenient accuracy from
{pct(lenient.estimates['acc_B0_zeroshot'])}% to {pct(lenient.estimates['acc_B0'])}% with no training.

## Gain split (points)

{make_effects_table(results, seeds)}

Correctness claims use the lenient grader; the benchmark view and the format claim use the strict
grader (METHODOLOGY §9). Per-seed values are listed in seed order {seed_list}.

![Gain split](gain_split.png)

**Format headroom.** With the few-shot prompt the base model already uses the required format in
{pct(headroom.format_rate)}% of completions, and {pct(headroom.lenient_not_strict)}% of problems have the right
number outside the FINAL line, which is all a format reward could fix.

## Where the strict gain comes from (RFC vs B0)

{make_strict_sources_table(sources)}

The net gain hides a lot of churn: RFC newly solves many problems B0 missed, but also loses many that
B0 had. Almost none of the gain is a right answer that was only missing the FINAL line.

![Strict gain sources](strict_gain_sources.png)

## Training

![Validation curves](validation_curves.png)

{make_cost_table(train_logs)}

## A stability lesson: config-v1

config-v1 (learning rate 3e-5, 500 steps) degraded in every cell. Its test set was evaluated once,
before the change; config-v2 used none of these numbers.

{config_v1_test_table()}

![config-v1 drift](config_v1_drift.png)

## Hand-checks of the lenient grader

A person judged each completion blind (the grader's decision hidden) and the result was compared with the
lenient grader. The targeted slice is picked from the hard cases on purpose, so it is reported apart from
the uniform samples. Samples this small bound the error rate only loosely; they cannot rule out rare misreads.

{hand_check_table()}

## Example problem

Chosen by a fixed rule: the first test problem that B0 gets strict-wrong and RFC (seed {EXAMPLE_SEED})
gets strict-right.

{show_example_problem(predictions, example_id, EXAMPLE_SEED)}
"""
    (out_dir / "report.md").write_text(report)
    summary = {g: {"estimates": r.estimates, "intervals": r.intervals,
                   "per_seed": {str(s): v for s, v in r.per_seed.items()}} for g, r in results.items()}
    (out_dir / "effects.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
