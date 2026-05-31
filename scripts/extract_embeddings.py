"""Pre-compute and cache BERT embeddings for the NovelCLF dataset."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BERT_EMBEDDINGS_PATH,
    BERT_MODEL_DIR,
    DATASET_PATH,
    PROCESSED_DIR,
    ensure_project_dirs,
)
from src.features import extract_bert_embeddings
from src.train import load_dataset

EMBEDDING_META_PATH = PROCESSED_DIR / "bert_embedding_meta.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and cache BERT embeddings for all dataset chunks.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract embeddings even if the cache file already exists.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for BERT inference (default: 8).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token length for BERT tokenizer (default: 256).",
    )
    return parser.parse_args()


def _extract_with_progress(
    texts: list[str],
    bert_model_dir: Path,
    batch_size: int,
    max_length: int,
):
    """Wrap extract_bert_embeddings with a tqdm progress bar."""
    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import BertModel, BertTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(bert_model_dir))
    model = BertModel.from_pretrained(str(bert_model_dir)).to(device)
    model.eval()

    embeddings: list[np.ndarray] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Extracting"):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def main() -> None:
    """Entry point for BERT embedding extraction."""
    args = parse_args()
    ensure_project_dirs()

    # Check cache
    if BERT_EMBEDDINGS_PATH.exists() and not args.force:
        print(f"Cache already exists: {BERT_EMBEDDINGS_PATH}")
        print("Use --force to re-extract embeddings.")
        return

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH} ...")
    rows = load_dataset(DATASET_PATH)
    texts = [row["text"] for row in rows]
    print(f"Loaded {len(texts)} chunks.")

    # Extract embeddings
    print(f"Extracting BERT embeddings (batch_size={args.batch_size}, max_length={args.max_length}) ...")
    start_time = time.perf_counter()
    embeddings = _extract_with_progress(
        texts,
        BERT_MODEL_DIR,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    elapsed = time.perf_counter() - start_time

    # Save embeddings
    import numpy as np

    np.save(BERT_EMBEDDINGS_PATH, embeddings)
    print(f"Embeddings saved to {BERT_EMBEDDINGS_PATH}")

    # Save metadata CSV (text_hash -> row index)
    EMBEDDING_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EMBEDDING_META_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "text_hash"])
        for idx, row in enumerate(rows):
            writer.writerow([idx, row.get("text_hash", "")])
    print(f"Metadata saved to {EMBEDDING_META_PATH}")

    # Summary
    file_size_mb = BERT_EMBEDDINGS_PATH.stat().st_size / (1024 * 1024)
    print("\n--- Summary ---")
    print(f"Shape:     {embeddings.shape}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Time:      {elapsed:.1f}s")


if __name__ == "__main__":
    main()
