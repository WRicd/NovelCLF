"""Feature extraction helpers for TF-IDF and BERT-based models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def require_sklearn():
    """Import sklearn dependencies with a helpful error message."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing scikit-learn. Install project dependencies with: pip install -r requirements.txt"
        ) from exc
    return TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 20000):
    """Create the standard TF-IDF vectorizer used by baseline models."""
    TfidfVectorizer = require_sklearn()
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )


def _jieba_tokenizer(text: str) -> list[str]:
    """Tokenize Chinese text using jieba, keeping only multi-char words.

    Defined at module level so the function is picklable (required for
    serializing sklearn Pipeline objects with joblib/pickle).
    """
    import jieba
    return [w for w in jieba.lcut(text) if len(w.strip()) > 1]


def build_tfidf_vectorizer_word(max_features: int = 20000):
    """Create a word-level TF-IDF vectorizer using jieba segmentation.

    This allows comparing character n-gram vs. jieba word features to see
    which approach better captures genre-distinguishing patterns.
    """
    # Eagerly check jieba is available
    try:
        import jieba  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing jieba. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    TfidfVectorizer = require_sklearn()

    return TfidfVectorizer(
        tokenizer=_jieba_tokenizer,
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )


def extract_bert_embeddings(
    texts: Iterable[str],
    bert_model_dir: Path,
    batch_size: int = 8,
    max_length: int = 256,
):
    """Extract mean-pooled BERT embeddings for chunk-level texts."""
    try:
        import numpy as np
        import torch
        from transformers import BertModel, BertTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing BERT dependencies. Install torch and transformers from requirements.txt."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(bert_model_dir))
    model = BertModel.from_pretrained(str(bert_model_dir)).to(device)
    model.eval()

    texts = list(texts)
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
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
