"""Plot learning curves to diagnose model bias/variance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_PATH, FIGURES_DIR, RANDOM_SEED, resolve_font_path
from src.train import load_dataset, split_rows, train_tfidf_model, predict_model
from src.evaluate import evaluate_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Plot learning curve for a model.")
    parser.add_argument("--model", choices=["nb", "nb_word", "svm_tfidf", "svm_tfidf_word"], default="nb")
    parser.add_argument("--split", choices=["chunk_random", "book_holdout"], default="book_holdout")
    parser.add_argument("--steps", type=int, default=5, help="Number of training size steps")
    return parser.parse_args()


def plot_learning_curve(model_name: str, split: str, steps: int):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.font_manager import FontProperties
    except ModuleNotFoundError:
        print("matplotlib not installed. Skipping plot.")
        return

    rows = load_dataset(DATASET_PATH)
    (train_rows, test_rows), _ = split_rows(rows, split)
    labels = sorted({row["label"] for row in rows})
    y_test = [row["label"] for row in test_rows]

    # Generate training sizes
    total_train = len(train_rows)
    sizes = [int(s) for s in np.linspace(total_train // steps, total_train, steps)]
    
    train_scores = []
    test_scores = []

    print(f"Plotting learning curve for {model_name} ({split})...")
    for size in sizes:
        print(f"Training on {size} samples...")
        current_train = train_rows[:size]
        model = train_tfidf_model(model_name, current_train)
        
        # Eval on train
        y_train_pred = list(predict_model(model_name, model, current_train))
        y_train_true = [row["label"] for row in current_train]
        train_metrics = evaluate_predictions(y_train_true, y_train_pred, labels)
        train_scores.append(train_metrics["accuracy"])
        
        # Eval on test
        y_test_pred = list(predict_model(model_name, model, test_rows))
        test_metrics = evaluate_predictions(y_test, y_test_pred, labels)
        test_scores.append(test_metrics["accuracy"])

    # Plot
    font_file = resolve_font_path()
    font_props = FontProperties(fname=str(font_file)) if font_file else None

    plt.figure(figsize=(8, 6))
    plt.plot(sizes, train_scores, "o-", color="r", label="Training Accuracy")
    plt.plot(sizes, test_scores, "o-", color="g", label="Test Accuracy")
    
    plt.title(f"Learning Curve ({model_name} / {split})", fontproperties=font_props, fontsize=14)
    plt.xlabel("Training Examples", fontproperties=font_props, fontsize=12)
    plt.ylabel("Accuracy", fontproperties=font_props, fontsize=12)
    plt.legend(loc="best", prop=font_props)
    plt.grid(True)
    
    out_path = FIGURES_DIR / f"learning_curve_{model_name}_{split}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    
    print(f"Learning curve saved to {out_path}")


def main():
    args = parse_args()
    plot_learning_curve(args.model, args.split, args.steps)


if __name__ == "__main__":
    main()
