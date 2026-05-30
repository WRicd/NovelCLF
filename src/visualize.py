"""Visualization helpers for NovelCLF."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .config import WORDCLOUD_DIR, resolve_font_path


def load_stopwords(path: Path) -> set[str]:
    """Load a stopword file with common encodings."""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return {line.strip() for line in path.read_text(encoding=encoding).splitlines() if line.strip()}
        except UnicodeDecodeError:
            continue
    return set()


def top_words(text: str, stopwords: set[str] | None = None, top_n: int = 20) -> list[tuple[str, int]]:
    """Return top words using jieba when available."""
    try:
        import jieba
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Missing jieba. Install dependencies with: pip install -r requirements.txt") from exc

    stopwords = stopwords or set()
    words = [
        word.strip()
        for word in jieba.lcut(text)
        if len(word.strip()) > 1 and word.strip() not in stopwords
    ]
    return Counter(words).most_common(top_n)


def save_wordcloud(
    frequencies: dict[str, int],
    filename: str,
    font_path: str | None = None,
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Save a wordcloud image from precomputed frequencies."""
    try:
        from wordcloud import WordCloud
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing wordcloud. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    font = resolve_font_path(font_path)
    if font is None:
        raise FileNotFoundError("No Chinese font was found. Pass --font-path or install a Chinese font.")

    WORDCLOUD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = WORDCLOUD_DIR / filename
    cloud = WordCloud(
        font_path=str(font),
        background_color="white",
        width=width,
        height=height,
        max_words=200,
        relative_scaling=0.4,
    )
    cloud.generate_from_frequencies(frequencies)
    cloud.to_file(str(output_path))
    return output_path
