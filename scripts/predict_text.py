from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_file, predict_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a novel genre from text or a .txt file.")
    parser.add_argument("--model", choices=["nb", "nb_word", "svm_tfidf", "svm_tfidf_word", "bert_svm"], default="nb")
    parser.add_argument("--voting", action="store_true", help="Predict file by chunk voting (whole book prediction)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", type=str)
    source.add_argument("--file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file:
        if args.voting:
            from src.predict import predict_book_voting
            result = predict_book_voting(args.file, model_name=args.model)
        else:
            result = predict_file(args.file, model_name=args.model)
    else:
        result = predict_text(args.text, model_name=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
