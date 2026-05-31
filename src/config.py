"""Central project configuration for NovelCLF."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

CATEGORIES = {
    "都市": RAW_DIR / "dushi",
    "科幻": RAW_DIR / "kehuan",
    "历史": RAW_DIR / "lishi",
    "玄幻": RAW_DIR / "xuanhuan",
}

RANDOM_SEED = 42

STOPWORDS_PATH = PROJECT_ROOT / "停用词表.txt"
BERT_MODEL_DIR = PROJECT_ROOT / "models" / "bert-base-chinese"

PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
FIGURES_DIR = OUTPUTS_DIR / "figures"
WORDCLOUD_DIR = OUTPUTS_DIR / "wordclouds"
MODELS_DIR = PROJECT_ROOT / "models"

DATASET_PATH = PROCESSED_DIR / "novel_chunks.csv"
DATASET_REPORT_PATH = PROCESSED_DIR / "split_report.json"
BERT_EMBEDDINGS_PATH = PROCESSED_DIR / "bert_embeddings.npy"

MODEL_PATHS = {
    "nb": MODELS_DIR / "nb_tfidf.joblib",
    "nb_word": MODELS_DIR / "nb_word.joblib",
    "svm_tfidf": MODELS_DIR / "svm_tfidf.joblib",
    "svm_tfidf_word": MODELS_DIR / "svm_tfidf_word.joblib",
    "bert_svm": MODELS_DIR / "bert_svm.joblib",
    "bert_finetune": MODELS_DIR / "bert_finetune",
}

SUPPORTED_SPLITS = {"chunk_random", "book_holdout"}
SUPPORTED_MODELS = {"nb", "nb_word", "svm_tfidf", "svm_tfidf_word", "bert_svm", "bert_finetune"}

# Models that use word-level (jieba) TF-IDF features
WORD_TFIDF_MODELS = {"nb_word", "svm_tfidf_word"}


def ensure_project_dirs() -> None:
    """Create generated-output directories used by the project."""
    for path in [
        PROCESSED_DIR,
        REPORTS_DIR,
        METRICS_DIR,
        EXPERIMENTS_DIR,
        FIGURES_DIR,
        WORDCLOUD_DIR,
        MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def resolve_font_path(font_path: str | None = None) -> Path | None:
    """Return a usable Chinese font path when one can be found."""
    candidates = []
    if font_path:
        candidates.append(Path(font_path))

    candidates.extend(
        [
            PROJECT_ROOT / "simhei.ttf",
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simsun.ttc"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
