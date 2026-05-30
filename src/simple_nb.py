"""A tiny dependency-free character n-gram Naive Bayes baseline."""

from __future__ import annotations

import math
from collections import Counter, defaultdict


class SimpleCharNaiveBayes:
    """Multinomial Naive Bayes over character n-grams.

    This fallback keeps the project runnable before scikit-learn is installed.
    It is intentionally small and educational, not a replacement for sklearn.
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 3), alpha: float = 1.0) -> None:
        self.ngram_range = ngram_range
        self.alpha = alpha
        self.classes_: list[str] = []
        self.class_log_prior_: dict[str, float] = {}
        self.feature_log_prob_: dict[str, dict[str, float]] = {}
        self.default_log_prob_: dict[str, float] = {}
        self.vocabulary_: set[str] = set()

    def _ngrams(self, text: str) -> list[str]:
        compact = "".join(text.split())
        tokens: list[str] = []
        start_n, end_n = self.ngram_range
        for n in range(start_n, end_n + 1):
            if len(compact) < n:
                continue
            tokens.extend(compact[index : index + n] for index in range(len(compact) - n + 1))
        return tokens

    def fit(self, texts: list[str], labels: list[str]):
        class_doc_counts = Counter(labels)
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        total_tokens: Counter[str] = Counter()

        for text, label in zip(texts, labels):
            counts = Counter(self._ngrams(text))
            token_counts[label].update(counts)
            total_tokens[label] += sum(counts.values())
            self.vocabulary_.update(counts)

        self.classes_ = sorted(class_doc_counts)
        total_docs = len(labels)
        vocab_size = max(len(self.vocabulary_), 1)

        for label in self.classes_:
            self.class_log_prior_[label] = math.log(class_doc_counts[label] / total_docs)
            denominator = total_tokens[label] + self.alpha * vocab_size
            self.default_log_prob_[label] = math.log(self.alpha / denominator)
            self.feature_log_prob_[label] = {
                token: math.log((count + self.alpha) / denominator)
                for token, count in token_counts[label].items()
            }
        return self

    def _joint_log_likelihood(self, text: str) -> dict[str, float]:
        counts = Counter(self._ngrams(text))
        scores = {}
        for label in self.classes_:
            score = self.class_log_prior_[label]
            probs = self.feature_log_prob_[label]
            default = self.default_log_prob_[label]
            for token, count in counts.items():
                score += count * probs.get(token, default)
            scores[label] = score
        return scores

    def predict(self, texts: list[str]) -> list[str]:
        predictions = []
        for text in texts:
            scores = self._joint_log_likelihood(text)
            predictions.append(max(scores, key=scores.get))
        return predictions

    def predict_scores(self, texts: list[str]) -> list[dict[str, float]]:
        """Return normalized probabilities for display."""
        results = []
        for text in texts:
            scores = self._joint_log_likelihood(text)
            max_score = max(scores.values())
            exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
            total = sum(exp_scores.values()) or 1.0
            results.append({label: value / total for label, value in exp_scores.items()})
        return results
