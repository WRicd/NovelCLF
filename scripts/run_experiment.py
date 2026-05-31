"""Batch experiment runner for NovelCLF.

Usage examples
--------------
python scripts/run_experiment.py --models nb,svm_tfidf --splits all --notes "baseline comparison"
python scripts/run_experiment.py --models all --splits book_holdout --notes "holdout sweep"
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SUPPORTED_MODELS, SUPPORTED_SPLITS  # noqa: E402

# TF-IDF-only models – the ones included when the user passes "all".
_DEFAULT_MODELS = sorted(SUPPORTED_MODELS - {"bert_svm", "bert_finetune"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple NovelCLF training experiments in one go.",
    )
    parser.add_argument(
        "--models",
        default="all",
        help=(
            'Comma-separated model names, or "all" for every TF-IDF model.  '
            "To include bert_svm / bert_finetune, list them explicitly."
        ),
    )
    parser.add_argument(
        "--splits",
        default="all",
        help='Comma-separated split names, or "all".',
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Optional note to tag every experiment in this batch.",
    )
    return parser.parse_args()


def resolve_names(raw: str, valid: set[str], default: list[str]) -> list[str]:
    """Parse a comma-separated string or 'all' into a sorted list of names."""
    if raw.strip().lower() == "all":
        return default
    names = [n.strip() for n in raw.split(",") if n.strip()]
    for name in names:
        if name not in valid:
            raise SystemExit(f"Error: unknown name '{name}'. Valid: {sorted(valid)}")
    return names


def main() -> None:
    args = parse_args()

    models = resolve_names(args.models, SUPPORTED_MODELS, _DEFAULT_MODELS)
    splits = resolve_names(args.splits, SUPPORTED_SPLITS, sorted(SUPPORTED_SPLITS))

    # Lazy imports to avoid slow startup when just checking --help.
    from src.experiment import ExperimentLog  # noqa: E402
    from src.train import train_model  # noqa: E402

    log = ExperimentLog()
    batch_entries: list[dict] = []
    total_start = time.perf_counter()

    combos = [(m, s) for m in models for s in splits]
    print(f"=== Running {len(combos)} experiment(s): {len(models)} model(s) × {len(splits)} split(s) ===\n")

    for idx, (model, split) in enumerate(combos, 1):
        tag = f"[{idx}/{len(combos)}]"
        print(f"{tag} Training {model} / {split} …", flush=True)
        run_start = time.perf_counter()
        try:
            metrics = train_model(model, split=split)
            duration = round(time.perf_counter() - run_start, 3)
            entry = log.log_experiment(
                model_name=model,
                split=split,
                metrics=metrics,
                notes=args.notes,
                duration=duration,
            )
            batch_entries.append(entry)
            acc = metrics.get("accuracy", 0)
            f1 = metrics.get("macro_f1", 0)
            print(f"{tag} Done  — accuracy={acc:.4f}  macro_f1={f1:.4f}  ({duration:.1f}s)\n")
        except Exception:
            duration = round(time.perf_counter() - run_start, 3)
            print(f"{tag} FAILED after {duration:.1f}s:")
            traceback.print_exc()
            print()

    total_elapsed = round(time.perf_counter() - total_start, 1)
    print(f"=== Batch complete — {len(batch_entries)}/{len(combos)} succeeded in {total_elapsed}s ===\n")

    if batch_entries:
        print(log.summary(batch_entries))


if __name__ == "__main__":
    main()
