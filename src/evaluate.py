"""Evaluation and reporting helpers."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

# Force non-interactive backend before any matplotlib import
os.environ.setdefault("MPLBACKEND", "Agg")


def _fallback_confusion_matrix(y_true, y_pred, labels: list[str]) -> list[list[int]]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for actual, predicted in zip(y_true, y_pred):
        if actual in index and predicted in index:
            matrix[index[actual]][index[predicted]] += 1
    return matrix


def _fallback_report(y_true, y_pred, labels: list[str]) -> dict[str, object]:
    report: dict[str, object] = {}
    total = len(y_true)
    supports = Counter(y_true)
    macro_f1_values = []
    weighted_f1_total = 0.0

    for label in labels:
        tp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted == label)
        fp = sum(1 for actual, predicted in zip(y_true, y_pred) if actual != label and predicted == label)
        fn = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == label and predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = supports[label]
        macro_f1_values.append(f1)
        weighted_f1_total += f1 * support
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

    accuracy = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted) / total if total else 0.0
    macro_f1 = sum(macro_f1_values) / len(labels) if labels else 0.0
    weighted_f1 = weighted_f1_total / total if total else 0.0
    report["accuracy"] = accuracy
    report["macro avg"] = {"f1-score": macro_f1}
    report["weighted avg"] = {"f1-score": weighted_f1}
    return report


def _sklearn_metrics(y_true, y_pred, labels: list[str]) -> dict[str, object] | None:
    """Return sklearn metrics when sklearn is installed."""
    try:
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    except ModuleNotFoundError:
        return None

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 6),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
        "metrics_backend": "sklearn",
    }


def evaluate_predictions(y_true, y_pred, labels: list[str]) -> dict[str, object]:
    """Return common classification metrics."""
    sklearn_result = _sklearn_metrics(y_true, y_pred, labels)
    if sklearn_result is not None:
        return sklearn_result

    report = _fallback_report(y_true, y_pred, labels)
    return {
        "accuracy": round(float(report["accuracy"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 6),
        "classification_report": report,
        "confusion_matrix": _fallback_confusion_matrix(y_true, y_pred, labels),
        "labels": labels,
        "metrics_backend": "stdlib",
    }


def save_metrics(metrics: dict[str, object], path: Path) -> None:
    """Save metrics as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def save_confusion_matrix_figure(metrics: dict[str, object], path: Path, title: str) -> None:
    """Save a confusion matrix heatmap using matplotlib (no seaborn required)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.font_manager import FontProperties
    except ModuleNotFoundError:
        return

    from .config import resolve_font_path

    labels = metrics["labels"]
    matrix = np.array(metrics["confusion_matrix"])
    path.parent.mkdir(parents=True, exist_ok=True)

    # Configure Chinese font for labels
    font_file = resolve_font_path()
    font_props = FontProperties(fname=str(font_file)) if font_file else None

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    # Add text annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            # Use white text on dark cells for readability
            color = "white" if value > matrix.max() * 0.6 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10, fontproperties=font_props)
    ax.set_yticklabels(labels, fontsize=10, fontproperties=font_props)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


