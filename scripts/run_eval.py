"""Stage 6: evaluate B0, B0-fewshot or a saved adapter greedily on a non-test split (GPU PC).

    uv run --extra train python scripts/run_eval.py --name B0 --limit 50
    uv run --extra train python scripts/run_eval.py --name B0_fewshot --fewshot
    uv run --extra train python scripts/run_eval.py --name RC --adapter-dir outputs/RC/seed42/checkpoints/step_500

Writes outputs/eval/<name>/<split>/predictions.jsonl and metrics.json. The test split is
not reachable from here: only scripts/run_final_eval.py may load it.
"""

import argparse
import json
from pathlib import Path

from grpo_gsm8k.config import load_config
from grpo_gsm8k.data import SPLIT_NAMES, load_split
from grpo_gsm8k.evaluate import evaluate_policy, summarize_predictions, write_predictions
from grpo_gsm8k.paths import REPO_ROOT
from grpo_gsm8k.policy import load_policy
from grpo_gsm8k.rollout import greedy_decoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="model name used for the output folder, e.g. B0")
    parser.add_argument("--split", default="validation", choices=SPLIT_NAMES)
    parser.add_argument("--limit", type=int, default=None, help="first N problems only (debugging)")
    parser.add_argument("--fewshot", action="store_true", help="prepend the 4 frozen exemplars")
    parser.add_argument("--adapter-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    problems = load_split(args.split)[: args.limit]
    exemplars = load_split("fewshot_exemplars") if args.fewshot else ()
    policy, tokenizer = load_policy(config.model, adapter_dir=args.adapter_dir)
    policy.eval()

    rows = evaluate_policy(
        policy, tokenizer, problems, greedy_decoding(config.training), config.evaluation.batch_size, exemplars
    )
    summary = summarize_predictions(rows) | {"name": args.name, "split": args.split, "fewshot": args.fewshot}

    out_dir = args.out_dir or REPO_ROOT / "outputs" / "eval" / args.name / args.split
    write_predictions(rows, out_dir / "predictions.jsonl")
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
