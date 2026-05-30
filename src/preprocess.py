"""Build paragraph-level datasets from raw Chinese novel files."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import CATEGORIES, DATASET_PATH, DATASET_REPORT_PATH, RANDOM_SEED, ensure_project_dirs


WHITESPACE_RE = re.compile(r"\s+")
NOISE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"请收藏本站.*",
        r"最新网址.*",
        r"手机用户请浏览.*",
        r"www\.[a-z0-9.-]+\.[a-z]{2,}",
        r"https?://\S+",
    ]
]


@dataclass(frozen=True)
class ChunkConfig:
    chunk_size: int = 1000
    min_chars: int = 200
    max_chunks_per_book: int = 400
    seed: int = RANDOM_SEED


def read_text_file(path: Path) -> str:
    """Read a text file with common Chinese encodings."""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_text(text: str) -> str:
    """Remove common boilerplate and normalize whitespace."""
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    text = text.replace("\u3000", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, min_chars: int) -> list[str]:
    """Split text into fixed-size chunks and drop very short fragments."""
    chunks = []
    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size].strip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
    return chunks


def stable_sample(items: list[str], limit: int, seed: int) -> list[str]:
    """Sample items deterministically while preserving a stable output order."""
    if limit <= 0 or len(items) <= limit:
        return items
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(items)), limit))
    return [items[index] for index in selected_indices]


def iter_novel_files(categories: dict[str, Path] | None = None) -> list[tuple[str, Path]]:
    """Return available novel text files as (label, path) pairs."""
    categories = categories or CATEGORIES
    files: list[tuple[str, Path]] = []
    for label, folder in categories.items():
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.txt")):
            files.append((label, path))
    return files


def build_dataset(
    output_path: Path = DATASET_PATH,
    report_path: Path = DATASET_REPORT_PATH,
    config: ChunkConfig | None = None,
) -> dict[str, object]:
    """Build the chunk dataset CSV and return a summary report."""
    ensure_project_dirs()
    config = config or ChunkConfig()
    rows: list[dict[str, object]] = []
    per_book: dict[str, int] = {}

    for label, path in iter_novel_files():
        raw_text = read_text_file(path)
        text = clean_text(raw_text)
        chunks = chunk_text(text, config.chunk_size, config.min_chars)
        chunks = stable_sample(chunks, config.max_chunks_per_book, config.seed + len(rows))

        book_name = path.stem
        per_book[f"{label}/{book_name}"] = len(chunks)

        for chunk_id, chunk in enumerate(chunks):
            text_hash = hashlib.md5(chunk.encode("utf-8")).hexdigest()[:12]
            rows.append(
                {
                    "text": chunk,
                    "label": label,
                    "book": book_name,
                    "book_path": str(path.relative_to(path.parents[1])),
                    "chunk_id": chunk_id,
                    "char_len": len(chunk),
                    "text_hash": text_hash,
                }
            )

    if not rows:
        raise FileNotFoundError("No novel .txt files were found in the configured category folders.")

    fieldnames = ["text", "label", "book", "book_path", "chunk_id", "char_len", "text_hash"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["label"] for row in rows)
    lengths = [int(row["char_len"]) for row in rows]
    report = {
        "dataset_path": str(output_path),
        "total_chunks": len(rows),
        "total_books": len(per_book),
        "chunk_size": config.chunk_size,
        "min_chars": config.min_chars,
        "max_chunks_per_book": config.max_chunks_per_book,
        "label_counts": dict(sorted(label_counts.items())),
        "book_counts": dict(sorted(per_book.items())),
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 2),
        "min_chunk_chars": min(lengths),
        "max_chunk_chars": max(lengths),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def summarize_dataset(dataset_path: Path = DATASET_PATH) -> dict[str, object]:
    """Load only metadata from the generated CSV and summarize it."""
    label_counts: Counter[str] = Counter()
    book_counts: defaultdict[str, int] = defaultdict(int)
    total = 0

    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            total += 1
            label_counts[row["label"]] += 1
            book_counts[f"{row['label']}/{row['book']}"] += 1

    return {
        "total_chunks": total,
        "label_counts": dict(sorted(label_counts.items())),
        "book_counts": dict(sorted(book_counts.items())),
    }
