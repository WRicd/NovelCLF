from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import ChunkConfig, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paragraph-level NovelCLF dataset.")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chunks-per-book", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_dataset(
        config=ChunkConfig(
            chunk_size=args.chunk_size,
            min_chars=args.min_chars,
            max_chunks_per_book=args.max_chunks_per_book,
            seed=args.seed,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
