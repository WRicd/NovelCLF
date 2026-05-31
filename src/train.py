"""Train NovelCLF models."""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path

from .config import (
    BERT_EMBEDDINGS_PATH,
    BERT_MODEL_DIR,
    DATASET_PATH,
    FIGURES_DIR,
    METRICS_DIR,
    MODEL_PATHS,
    RANDOM_SEED,
    SUPPORTED_MODELS,
    SUPPORTED_SPLITS,
    WORD_TFIDF_MODELS,
    ensure_project_dirs,
)
from .evaluate import evaluate_predictions, save_confusion_matrix_figure, save_metrics
from .features import build_tfidf_vectorizer, build_tfidf_vectorizer_word, extract_bert_embeddings
from .persistence import dump_model
from .simple_nb import SimpleCharNaiveBayes


def _import_bert_finetune():
    """Lazy import for the bert_finetune module."""
    from .bert_finetune import predict_bert_finetune, train_bert_finetune
    return train_bert_finetune, predict_bert_finetune


def optional_training_deps():
    """Return sklearn training dependencies when available."""
    try:
        from sklearn.model_selection import GroupShuffleSplit, train_test_split
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC, SVC
    except ModuleNotFoundError:
        return None
    return GroupShuffleSplit, train_test_split, MultinomialNB, Pipeline, LinearSVC, SVC


def require_sklearn_for(model_name: str):
    deps = optional_training_deps()
    if deps is None:
        raise ModuleNotFoundError(
            f"Model '{model_name}' requires scikit-learn. Install dependencies with: pip install -r requirements.txt"
        )
    return deps


def load_dataset(dataset_path: Path = DATASET_PATH) -> list[dict[str, str]]:
    """Load the generated chunk dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}. Run scripts/build_dataset.py first."
        )
    with dataset_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_rows(rows: list[dict[str, str]], split: str):
    """Split rows either randomly by chunk or by held-out book."""
    if split not in SUPPORTED_SPLITS:
        raise ValueError(f"Unsupported split '{split}'. Use one of: {sorted(SUPPORTED_SPLITS)}")

    indices = list(range(len(rows)))
    labels = [row["label"] for row in rows]
    deps = optional_training_deps()
    if split == "chunk_random":
        if deps is not None:
            _, train_test_split, *_ = deps
            train_idx, test_idx = train_test_split(
                indices,
                test_size=0.2,
                random_state=RANDOM_SEED,
                stratify=labels,
            )
        else:
            rng = random.Random(RANDOM_SEED)
            train_idx = []
            test_idx = []
            by_label: dict[str, list[int]] = {}
            for index, label in zip(indices, labels):
                by_label.setdefault(label, []).append(index)
            for label_indices in by_label.values():
                shuffled = label_indices[:]
                rng.shuffle(shuffled)
                test_size = max(1, round(len(shuffled) * 0.2))
                test_idx.extend(shuffled[:test_size])
                train_idx.extend(shuffled[test_size:])
    else:
        if deps is not None:
            GroupShuffleSplit, *_ = deps
            groups = [f"{row['label']}/{row['book']}" for row in rows]
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
            train_idx, test_idx = next(splitter.split(indices, labels, groups))
        else:
            groups_by_label: dict[str, list[str]] = {}
            for row in rows:
                groups_by_label.setdefault(row["label"], [])
                group = f"{row['label']}/{row['book']}"
                if group not in groups_by_label[row["label"]]:
                    groups_by_label[row["label"]].append(group)

            rng = random.Random(RANDOM_SEED)
            test_groups = set()
            for groups in groups_by_label.values():
                shuffled = groups[:]
                rng.shuffle(shuffled)
                test_groups.update(shuffled[:1])

            train_idx = []
            test_idx = []
            for index, row in enumerate(rows):
                group = f"{row['label']}/{row['book']}"
                if group in test_groups:
                    test_idx.append(index)
                else:
                    train_idx.append(index)

    train_rows = [rows[index] for index in train_idx]
    test_rows = [rows[index] for index in test_idx]
    return (train_rows, test_rows), (train_idx, test_idx)


def train_tfidf_model(model_name: str, train_rows: list[dict[str, str]]):
    """Train a TF-IDF baseline model (char n-gram or jieba word)."""
    is_word = model_name in WORD_TFIDF_MODELS
    base_name = model_name.replace("_word", "")  # nb_word -> nb, svm_tfidf_word -> svm_tfidf

    deps = optional_training_deps()
    if deps is None:
        if base_name == "nb" and not is_word:
            model = SimpleCharNaiveBayes()
            model.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
            return model
        raise ModuleNotFoundError(
            f"Model '{model_name}' requires scikit-learn. Install dependencies with: pip install -r requirements.txt"
        )

    _, _, MultinomialNB, Pipeline, LinearSVC, _ = deps
    vectorizer = build_tfidf_vectorizer_word() if is_word else build_tfidf_vectorizer()

    if base_name == "nb":
        classifier = MultinomialNB()
    elif base_name == "svm_tfidf":
        classifier = LinearSVC(random_state=RANDOM_SEED)
    else:
        raise ValueError(f"Unsupported TF-IDF model: {model_name}")

    pipeline = Pipeline([("tfidf", vectorizer), ("clf", classifier)])
    pipeline.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
    return pipeline


def train_bert_svm(train_rows: list[dict[str, str]], train_embeddings):
    """Train an SVM on top of cached BERT-style embeddings."""
    *_, SVC = require_sklearn_for("bert_svm")
    labels = [row["label"] for row in train_rows]
    clf = SVC(kernel="linear", probability=True, random_state=RANDOM_SEED)
    clf.fit(train_embeddings, labels)
    return {"classifier": clf, "feature_type": "bert_mean_pooling"}


# All TF-IDF model names (char or word)
_TFIDF_MODELS = {"nb", "nb_word", "svm_tfidf", "svm_tfidf_word"}


def predict_model(model_name: str, model, rows: list[dict[str, str]]):
    """Predict labels for dataset rows with a trained model."""
    if model_name in _TFIDF_MODELS:
        return model.predict([row["text"] for row in rows])
    if model_name == "bert_svm":
        embeddings = extract_bert_embeddings([row["text"] for row in rows], BERT_MODEL_DIR)
        return model["classifier"].predict(embeddings)
    if model_name == "bert_finetune":
        _, predict_bert_finetune = _import_bert_finetune()
        return predict_bert_finetune([row["text"] for row in rows], MODEL_PATHS["bert_finetune"])
    raise ValueError(f"Unsupported model: {model_name}")


def train_model(model_name: str, split: str = "chunk_random", dataset_path: Path = DATASET_PATH) -> dict[str, object]:
    """Train, evaluate, save a model, and return its metrics."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Use one of: {sorted(SUPPORTED_MODELS)}")

    ensure_project_dirs()
    rows = load_dataset(dataset_path)
    (train_rows, test_rows), (train_idx, test_idx) = split_rows(rows, split)
    labels = sorted({row["label"] for row in rows})

    train_start = time.perf_counter()
    if model_name == "bert_finetune":
        # BERT fine-tuning has its own training + evaluation loop
        train_fn, predict_fn = _import_bert_finetune()
        save_dir = MODEL_PATHS["bert_finetune"]
        ft_result = train_fn(
            train_texts=[row["text"] for row in train_rows],
            train_labels=[row["label"] for row in train_rows],
            test_texts=[row["text"] for row in test_rows],
            test_labels=[row["label"] for row in test_rows],
            label_list=labels,
            bert_model_dir=BERT_MODEL_DIR,
            save_dir=save_dir,
            seed=RANDOM_SEED,
        )
        train_seconds = round(time.perf_counter() - train_start, 3)

        y_true = [row["label"] for row in test_rows]
        pred_start = time.perf_counter()
        y_pred = list(predict_fn([row["text"] for row in test_rows], save_dir))
        predict_seconds = round(time.perf_counter() - pred_start, 3)
        metrics = evaluate_predictions(y_true, y_pred, labels)

    elif model_name in _TFIDF_MODELS:
        model = train_tfidf_model(model_name, train_rows)
        train_seconds = round(time.perf_counter() - train_start, 3)

        y_true = [row["label"] for row in test_rows]
        pred_start = time.perf_counter()
        y_pred = list(predict_model(model_name, model, test_rows))
        predict_seconds = round(time.perf_counter() - pred_start, 3)
        metrics = evaluate_predictions(y_true, y_pred, labels)
    elif model_name == "bert_svm":
        # Load or compute full embeddings once
        if BERT_EMBEDDINGS_PATH.exists():
            import numpy as np
            full_embeddings = np.load(BERT_EMBEDDINGS_PATH)
        else:
            full_embeddings = extract_bert_embeddings([row["text"] for row in rows], BERT_MODEL_DIR)
            import numpy as np
            np.save(BERT_EMBEDDINGS_PATH, full_embeddings)

        train_embeddings = full_embeddings[train_idx]
        model = train_bert_svm(train_rows, train_embeddings)
        train_seconds = round(time.perf_counter() - train_start, 3)

        y_true = [row["label"] for row in test_rows]
        pred_start = time.perf_counter()
        test_embeddings = full_embeddings[test_idx]
        y_pred = list(model["classifier"].predict(test_embeddings))
        predict_seconds = round(time.perf_counter() - pred_start, 3)
        metrics = evaluate_predictions(y_true, y_pred, labels)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    metrics.update(
        {
            "model": model_name,
            "split": split,
            "train_chunks": len(train_rows),
            "test_chunks": len(test_rows),
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "train_books": sorted({f"{row['label']}/{row['book']}" for row in train_rows}),
            "test_books": sorted({f"{row['label']}/{row['book']}" for row in test_rows}),
        }
    )

    # bert_finetune saves its own checkpoint; other models use dump_model
    if model_name != "bert_finetune":
        model_path = MODEL_PATHS[model_name]
        dump_model(
            {
                "model": model,
                "model_name": model_name,
                "split": split,
                "labels": labels,
            },
            model_path,
        )

    metrics_path = METRICS_DIR / f"{model_name}_{split}.json"
    figure_path = FIGURES_DIR / f"{model_name}_{split}_confusion_matrix.png"
    save_metrics(metrics, metrics_path)
    save_confusion_matrix_figure(metrics, figure_path, f"{model_name} / {split}")

    # Extract and save feature importance for TF-IDF models
    if model_name in _TFIDF_MODELS:
        try:
            from .explain import extract_tfidf_top_features, print_top_features

            features_path = METRICS_DIR / f"{model_name}_{split}_top_features.json"
            top_features = extract_tfidf_top_features(
                model, model_name, top_n=20, save_path=features_path
            )
            if top_features:
                print(f"\n--- Top features per class ({model_name}) ---")
                print_top_features(top_features)
        except Exception:
            pass  # Feature importance is optional, don't block training


    return metrics
