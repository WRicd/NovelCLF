"""Prediction helpers for NovelCLF."""

from __future__ import annotations

from pathlib import Path

from .config import BERT_MODEL_DIR, MODEL_PATHS
from .features import extract_bert_embeddings
from .persistence import load_model
from .preprocess import clean_text, read_text_file


def _import_bert_finetune():
    """Lazy import for bert_finetune inference helpers."""
    from .bert_finetune import predict_bert_finetune, predict_bert_finetune_with_scores
    return predict_bert_finetune, predict_bert_finetune_with_scores


def load_trained_model(model_name: str):
    """Load a saved model bundle.

    For ``bert_finetune`` this returns ``None`` because inference loads the
    model directly from its checkpoint directory.
    """
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unsupported model: {model_name}")
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train it first.")
    if model_name == "bert_finetune":
        return None  # loaded on-demand inside predict helpers
    return load_model(model_path)


def predict_text(text: str, model_name: str = "nb") -> dict[str, object]:
    """Predict a label for raw text."""
    cleaned = clean_text(text)

    # bert_finetune manages its own model loading
    if model_name == "bert_finetune":
        _, predict_with_scores = _import_bert_finetune()
        result = predict_with_scores([cleaned], MODEL_PATHS["bert_finetune"])[0]
        return {"label": result["label"], "scores": result["scores"]}

    bundle = load_trained_model(model_name)
    model = bundle["model"]

    if model_name in {"nb", "nb_word", "svm_tfidf", "svm_tfidf_word"}:
        label = model.predict([cleaned])[0]
        scores = None
        if hasattr(model, "predict_scores"):
            scores = model.predict_scores([cleaned])[0]
        elif hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned])[0]
            scores = dict(zip(model.classes_, [float(value) for value in probabilities]))
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function([cleaned])
            values = raw_scores[0] if getattr(raw_scores, "ndim", 1) > 1 else raw_scores
            classes = model.classes_
            scores = dict(zip(classes, [float(value) for value in values]))
    elif model_name == "bert_svm":
        embeddings = extract_bert_embeddings([cleaned], BERT_MODEL_DIR)
        clf = model["classifier"]
        label = clf.predict(embeddings)[0]
        scores = dict(zip(clf.classes_, [float(value) for value in clf.predict_proba(embeddings)[0]]))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return {"label": label, "scores": scores}


def predict_file(path: Path, model_name: str = "nb") -> dict[str, object]:
    """Predict a label for a text file as a single text."""
    return predict_text(read_text_file(path), model_name=model_name)


def predict_book_voting(path: Path, model_name: str = "nb", max_chunks: int = 100) -> dict[str, object]:
    """Predict a whole book by chunking it and using majority voting.
    
    This splits the book into paragraphs, predicts each paragraph independently,
    and returns the most frequent label.
    """
    from collections import Counter
    from .preprocess import chunk_text, ChunkConfig, stable_sample
    
    raw_text = read_text_file(path)
    text = clean_text(raw_text)
    
    config = ChunkConfig()
    chunks = chunk_text(text, config.chunk_size, config.min_chars)
    
    if not chunks:
        return {"label": "Unknown", "error": "Text too short to extract any chunks."}
        
    # Sample up to max_chunks to speed up prediction
    if len(chunks) > max_chunks:
        chunks = stable_sample(chunks, max_chunks, seed=42)
        
    bundle = load_trained_model(model_name)
    model = bundle["model"] if bundle is not None else None
    
    # Batch predict
    if model_name in {"nb", "nb_word", "svm_tfidf", "svm_tfidf_word"}:
        preds = model.predict(chunks)
    elif model_name == "bert_svm":
        embeddings = extract_bert_embeddings(chunks, BERT_MODEL_DIR)
        preds = model["classifier"].predict(embeddings)
    elif model_name == "bert_finetune":
        predict_fn, _ = _import_bert_finetune()
        preds = predict_fn(chunks, MODEL_PATHS["bert_finetune"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    counts = Counter(preds)
    best_label = counts.most_common(1)[0][0]
    
    # Calculate confidence as percentage of votes
    total_votes = sum(counts.values())
    confidence = counts[best_label] / total_votes
    
    return {
        "label": best_label,
        "confidence": round(confidence, 4),
        "total_chunks_predicted": total_votes,
        "vote_distribution": dict(counts)
    }

