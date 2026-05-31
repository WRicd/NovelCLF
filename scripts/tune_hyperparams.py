"""Hyperparameter tuning for NovelCLF TF-IDF models via GridSearchCV."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_PATH,
    METRICS_DIR,
    RANDOM_SEED,
    WORD_TFIDF_MODELS,
    ensure_project_dirs,
)
from src.train import load_dataset

_SUPPORTED_MODELS = {"nb", "nb_word", "svm_tfidf", "svm_tfidf_word"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for a TF-IDF model using book-level GroupKFold.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(_SUPPORTED_MODELS),
        help="Model to tune (nb, nb_word, svm_tfidf, svm_tfidf_word).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of GroupKFold folds (default: 3).",
    )
    return parser.parse_args()


def _build_pipeline_and_grid(model_name: str):
    """Return (Pipeline, param_grid) for the given model name."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    from src.features import _jieba_tokenizer

    is_word = model_name in WORD_TFIDF_MODELS
    base_name = model_name.replace("_word", "")

    # Build vectorizer
    if is_word:
        vectorizer = TfidfVectorizer(
            tokenizer=_jieba_tokenizer,
            sublinear_tf=True,
        )
    else:
        vectorizer = TfidfVectorizer(
            analyzer="char",
            sublinear_tf=True,
        )

    # Build classifier
    if base_name == "nb":
        classifier = MultinomialNB()
    elif base_name == "svm_tfidf":
        classifier = LinearSVC(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unsupported model for tuning: {model_name}")

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", classifier)])

    # Build param grid
    param_grid: dict[str, list] = {
        "tfidf__max_features": [5000, 10000, 20000, 40000],
        "tfidf__min_df": [1, 2, 5],
    }

    # char-level models can also tune ngram_range
    if not is_word:
        param_grid["tfidf__ngram_range"] = [(1, 2), (1, 3), (2, 3)]

    # SVM-specific: tune C
    if base_name == "svm_tfidf":
        param_grid["clf__C"] = [0.1, 1, 10]

    return pipeline, param_grid


def main() -> None:
    """Entry point for hyperparameter tuning."""
    args = parse_args()
    model_name: str = args.model
    n_folds: int = args.folds

    from sklearn.model_selection import GridSearchCV, GroupKFold

    ensure_project_dirs()

    # Load data
    print(f"Loading dataset from {DATASET_PATH} ...")
    rows = load_dataset(DATASET_PATH)
    texts = [row["text"] for row in rows]
    labels = [row["label"] for row in rows]
    groups = [f"{row['label']}/{row['book']}" for row in rows]
    print(f"Loaded {len(rows)} chunks, {len(set(groups))} unique books.")

    # Build pipeline and grid
    pipeline, param_grid = _build_pipeline_and_grid(model_name)

    total_combos = 1
    for values in param_grid.values():
        total_combos *= len(values)
    print(f"\nModel: {model_name}")
    print(f"Parameter grid ({total_combos} combinations):")
    for key, values in param_grid.items():
        print(f"  {key}: {values}")

    # Run GridSearchCV with book-level GroupKFold
    cv = GroupKFold(n_splits=n_folds)
    scoring = "f1_macro"

    print(f"\nRunning GridSearchCV ({n_folds}-fold GroupKFold, scoring={scoring}) ...")
    start_time = time.perf_counter()

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        refit=False,
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(texts, labels, groups=groups)

    elapsed = time.perf_counter() - start_time

    # Print summary table
    results = grid_search.cv_results_
    print(f"\n--- Tuning Results for {model_name} ---")
    print(f"{'Rank':<6} {'Mean Score':<12} {'Std':<10} {'Params'}")
    print("-" * 70)

    # Sort by rank
    sorted_indices = results["rank_test_score"].argsort()
    for idx in sorted_indices[:10]:  # Top 10
        rank = results["rank_test_score"][idx]
        mean = results["mean_test_score"][idx]
        std = results["std_test_score"][idx]
        params = results["params"][idx]
        print(f"{rank:<6} {mean:<12.6f} {std:<10.6f} {params}")

    print(f"\nBest score: {grid_search.best_score_:.6f}")
    print(f"Best params: {grid_search.best_params_}")
    print(f"Total time: {elapsed:.1f}s")

    # Save results to JSON
    output_path = METRICS_DIR / f"tuning_{model_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def _to_serializable(obj: object) -> object:
        """Convert numpy types to native Python types."""
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable_results = []
    for idx in sorted_indices:
        entry = {
            "rank": int(results["rank_test_score"][idx]),
            "mean_score": float(results["mean_test_score"][idx]),
            "std_score": float(results["std_test_score"][idx]),
            "params": {k: _to_serializable(v) for k, v in results["params"][idx].items()},
        }
        serializable_results.append(entry)

    report = {
        "model": model_name,
        "scoring": scoring,
        "n_folds": n_folds,
        "best_score": float(grid_search.best_score_),
        "best_params": {k: _to_serializable(v) for k, v in grid_search.best_params_.items()},
        "total_time_seconds": round(elapsed, 2),
        "all_results": serializable_results,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
