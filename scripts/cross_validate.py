"""Leave-One-Book-Out Cross Validation for NovelCLF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_PATH, METRICS_DIR
from src.train import load_dataset, train_tfidf_model, predict_model
from src.evaluate import evaluate_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Run Leave-One-Book-Out Cross Validation.")
    parser.add_argument("--model", choices=["nb", "nb_word", "svm_tfidf", "svm_tfidf_word"], default="nb")
    return parser.parse_args()


def run_lobo_cv(model_name: str):
    rows = load_dataset(DATASET_PATH)
    labels = sorted({row["label"] for row in rows})
    
    # Identify all unique books
    books = sorted({f"{row['label']}/{row['book']}" for row in rows})
    print(f"Starting LOBO CV with {len(books)} books for model {model_name}...")
    
    all_metrics = []
    
    start_time = time.perf_counter()
    
    for i, test_book in enumerate(books):
        print(f"[{i+1}/{len(books)}] Testing on {test_book}...")
        
        train_rows = [row for row in rows if f"{row['label']}/{row['book']}" != test_book]
        test_rows = [row for row in rows if f"{row['label']}/{row['book']}" == test_book]
        
        # Train
        model = train_tfidf_model(model_name, train_rows)
        
        # Predict
        y_pred = list(predict_model(model_name, model, test_rows))
        y_true = [row["label"] for row in test_rows]
        
        metrics = evaluate_predictions(y_true, y_pred, labels)
        metrics["test_book"] = test_book
        all_metrics.append(metrics)
        print(f"  Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}")

    total_time = time.perf_counter() - start_time
    
    # Aggregate results
    avg_accuracy = sum(m["accuracy"] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m["macro_f1"] for m in all_metrics) / len(all_metrics)
    
    print("\n--- LOBO CV Summary ---")
    print(f"Model: {model_name}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Macro-F1: {avg_f1:.4f}")
    print(f"Total Time: {total_time:.2f}s")
    
    # Save results
    out_path = METRICS_DIR / f"{model_name}_lobo_cv.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "model": model_name,
        "cv_type": "leave_one_book_out",
        "average_accuracy": avg_accuracy,
        "average_macro_f1": avg_f1,
        "total_time_seconds": round(total_time, 2),
        "fold_results": all_metrics
    }
    
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed results saved to {out_path}")


def main():
    args = parse_args()
    run_lobo_cv(args.model)


if __name__ == "__main__":
    main()
