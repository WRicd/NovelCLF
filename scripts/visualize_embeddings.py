"""Visualize embeddings with t-SNE and/or PCA projections."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Force non-interactive backend before any matplotlib import
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BERT_EMBEDDINGS_PATH,
    DATASET_PATH,
    FIGURES_DIR,
    RANDOM_SEED,
    ensure_project_dirs,
    resolve_font_path,
)
from src.train import load_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Produce t-SNE / PCA scatter plots of embeddings coloured by genre.",
    )
    parser.add_argument(
        "--method",
        choices=["tsne", "pca", "both"],
        default="both",
        help="Dimensionality reduction method (default: both).",
    )
    parser.add_argument(
        "--feature",
        choices=["bert", "tfidf"],
        default="tfidf",
        help="Feature type to visualise (default: tfidf).",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30,
        help="t-SNE perplexity parameter (default: 30).",
    )
    return parser.parse_args()


def _load_features(feature_type: str, rows: list[dict[str, str]]):
    """Return a feature matrix for the given feature type."""
    import numpy as np

    if feature_type == "bert":
        if not BERT_EMBEDDINGS_PATH.exists():
            raise FileNotFoundError(
                f"BERT embedding cache not found at {BERT_EMBEDDINGS_PATH}. "
                "Run scripts/extract_embeddings.py first."
            )
        embeddings = np.load(BERT_EMBEDDINGS_PATH)
        if embeddings.shape[0] != len(rows):
            raise ValueError(
                f"Embedding count ({embeddings.shape[0]}) does not match "
                f"dataset rows ({len(rows)}). Re-run extract_embeddings.py."
            )
        return embeddings

    # TF-IDF (char-level)
    from src.features import build_tfidf_vectorizer

    texts = [row["text"] for row in rows]
    vectorizer = build_tfidf_vectorizer()
    return vectorizer.fit_transform(texts)


def _plot_scatter(
    X_2d,
    labels: list[str],
    unique_labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Create and save a 2D scatter plot coloured by genre label."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.font_manager import FontProperties

    font_file = resolve_font_path()
    font_props = FontProperties(fname=str(font_file)) if font_file else None

    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    label_to_colour = {lbl: colours[i % len(colours)] for i, lbl in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for lbl in unique_labels:
        mask = np.array([l == lbl for l in labels])
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=label_to_colour[lbl],
            label=lbl,
            alpha=0.6,
            s=15,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=14, fontproperties=font_props)
    ax.set_xlabel("Component 1", fontsize=11)
    ax.set_ylabel("Component 2", fontsize=11)
    ax.legend(loc="best", prop=font_props, markerscale=2)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_pca(features, labels: list[str], unique_labels: list[str], feature_type: str) -> None:
    """Compute PCA and save scatter plot."""
    from sklearn.decomposition import PCA

    print("Computing PCA projection ...")
    start = time.perf_counter()
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_2d = pca.fit_transform(features if not hasattr(features, "toarray") else features.toarray())
    elapsed = time.perf_counter() - start
    print(f"PCA done in {elapsed:.1f}s  (explained variance: {pca.explained_variance_ratio_})")

    output_path = FIGURES_DIR / "embeddings_pca.png"
    title = f"PCA — {feature_type.upper()} embeddings"
    _plot_scatter(X_2d, labels, unique_labels, title, output_path)


def run_tsne(
    features,
    labels: list[str],
    unique_labels: list[str],
    feature_type: str,
    perplexity: float,
) -> None:
    """Compute t-SNE and save scatter plot."""
    from sklearn.manifold import TSNE

    print(f"Computing t-SNE projection (perplexity={perplexity}) ...")
    start = time.perf_counter()
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=RANDOM_SEED,
        init="pca",
        learning_rate="auto",
    )
    X_2d = tsne.fit_transform(features if not hasattr(features, "toarray") else features.toarray())
    elapsed = time.perf_counter() - start
    print(f"t-SNE done in {elapsed:.1f}s")

    output_path = FIGURES_DIR / "embeddings_tsne.png"
    title = f"t-SNE — {feature_type.upper()} embeddings"
    _plot_scatter(X_2d, labels, unique_labels, title, output_path)


def main() -> None:
    """Entry point for embedding visualisation."""
    args = parse_args()
    ensure_project_dirs()

    # Load dataset
    print(f"Loading dataset from {DATASET_PATH} ...")
    rows = load_dataset(DATASET_PATH)
    labels = [row["label"] for row in rows]
    unique_labels = sorted(set(labels))
    print(f"Loaded {len(rows)} chunks, {len(unique_labels)} genres: {unique_labels}")

    # Load features
    print(f"\nLoading {args.feature} features ...")
    features = _load_features(args.feature, rows)
    print(f"Feature matrix shape: {features.shape}")

    # Run projections
    if args.method in ("pca", "both"):
        run_pca(features, labels, unique_labels, args.feature)

    if args.method in ("tsne", "both"):
        run_tsne(features, labels, unique_labels, args.feature, args.perplexity)

    print("\nDone.")


if __name__ == "__main__":
    main()
