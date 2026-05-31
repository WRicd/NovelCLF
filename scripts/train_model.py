from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a NovelCLF model.")
    parser.add_argument("--model", choices=["nb", "nb_word", "svm_tfidf", "svm_tfidf_word", "bert_svm", "bert_finetune"], default="nb")
    parser.add_argument("--split", choices=["chunk_random", "book_holdout"], default="chunk_random")
    parser.add_argument("--dataset", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(args.model, split=args.split, dataset_path=args.dataset) if args.dataset else train_model(args.model, split=args.split)
    summary = {
        "model": metrics["model"],
        "split": metrics["split"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "train_chunks": metrics["train_chunks"],
        "test_chunks": metrics["test_chunks"],
        "test_books": metrics["test_books"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
