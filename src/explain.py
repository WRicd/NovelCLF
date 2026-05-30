"""Feature importance and model explainability helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import METRICS_DIR, ensure_project_dirs


def extract_tfidf_top_features(
    model,
    model_name: str,
    top_n: int = 20,
    save_path: Optional[Path] = None,
) -> dict[str, list[tuple[str, float]]]:
    """Extract the most important TF-IDF features per class.

    Works with sklearn Pipeline objects containing a TF-IDF vectorizer
    step named 'tfidf' and a classifier step named 'clf'.

    Returns a dict mapping each class label to a list of (feature, weight) tuples
    sorted by importance (descending).
    """
    try:
        import numpy as np
    except ModuleNotFoundError:
        return {}

    # Get feature names from the TF-IDF vectorizer
    if hasattr(model, "named_steps"):
        vectorizer = model.named_steps.get("tfidf")
        classifier = model.named_steps.get("clf")
    else:
        return {}

    if vectorizer is None or classifier is None:
        return {}

    feature_names = vectorizer.get_feature_names_out()

    results: dict[str, list[tuple[str, float]]] = {}

    if hasattr(classifier, "coef_"):
        # LinearSVC / MultinomialNB with multi-class
        coef = classifier.coef_
        classes = classifier.classes_

        if coef.ndim == 1:
            # Binary classification edge case
            classes = [classes[1]]
            coef = coef.reshape(1, -1)

        for i, label in enumerate(classes):
            if hasattr(coef, "toarray"):
                row = coef[i].toarray().flatten()
            else:
                row = np.asarray(coef[i]).flatten()
            top_indices = row.argsort()[-top_n:][::-1]
            results[label] = [
                (feature_names[j], round(float(row[j]), 6)) for j in top_indices
            ]

    elif hasattr(classifier, "feature_log_prob_"):
        # MultinomialNB
        for i, label in enumerate(classifier.classes_):
            log_probs = classifier.feature_log_prob_[i]
            top_indices = log_probs.argsort()[-top_n:][::-1]
            results[label] = [
                (feature_names[j], round(float(log_probs[j]), 6)) for j in top_indices
            ]

    if save_path is not None:
        _save_top_features(results, save_path, model_name)

    return results


def _save_top_features(
    results: dict[str, list[tuple[str, float]]],
    path: Path,
    model_name: str,
) -> None:
    """Save top features as a JSON file."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        label: [{"feature": feat, "weight": weight} for feat, weight in features]
        for label, features in results.items()
    }
    output = {"model": model_name, "top_features": serializable}
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def print_top_features(results: dict[str, list[tuple[str, float]]]) -> None:
    """Print top features per class in a readable format."""
    for label, features in results.items():
        feature_strs = [f"'{feat}'({weight:+.4f})" for feat, weight in features[:10]]
        print(f"  {label}: {', '.join(feature_strs)}")
