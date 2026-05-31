"""Display past NovelCLF experiment results.

Usage examples
--------------
python scripts/show_experiments.py
python scripts/show_experiments.py --model nb --split chunk_random
python scripts/show_experiments.py --top 5
python scripts/show_experiments.py --best
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SUPPORTED_MODELS, SUPPORTED_SPLITS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query and display NovelCLF experiment history.",
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=sorted(SUPPORTED_MODELS),
        help="Filter experiments by model name.",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=sorted(SUPPORTED_SPLITS),
        help="Filter experiments by split strategy.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only the top N experiments (most recent first).",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Show the single best experiment (by accuracy) for each model.",
    )
    parser.add_argument(
        "--metric",
        default="accuracy",
        help="Metric to rank by when using --best (default: accuracy).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Lazy import to keep --help fast.
    from src.experiment import ExperimentLog  # noqa: E402

    log = ExperimentLog()

    if args.best:
        # Show the single best run per model.
        models = [args.model] if args.model else sorted(SUPPORTED_MODELS)
        best_entries: list[dict] = []
        for model in models:
            entry = log.best_experiment(metric=args.metric, model_name=model)
            if entry is not None:
                best_entries.append(entry)

        if not best_entries:
            print("No experiments found.")
            return

        print(f"Best experiment per model (ranked by {args.metric}):\n")
        print(log.summary(best_entries))
    else:
        entries = log.list_experiments(
            model_name=args.model,
            split=args.split,
            top_n=args.top,
        )
        if not entries:
            print("No experiments found.")
            return

        qualifier_parts: list[str] = []
        if args.model:
            qualifier_parts.append(f"model={args.model}")
        if args.split:
            qualifier_parts.append(f"split={args.split}")
        qualifier = f" ({', '.join(qualifier_parts)})" if qualifier_parts else ""
        count_label = f"top {args.top}" if args.top else f"{len(entries)}"
        print(f"Showing {count_label} experiment(s){qualifier}:\n")
        print(log.summary(entries))


if __name__ == "__main__":
    main()
