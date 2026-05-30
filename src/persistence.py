"""Model persistence helpers."""

from __future__ import annotations

import pickle
from pathlib import Path


def dump_model(obj, path: Path) -> None:
    """Save an object with joblib when available, otherwise pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except ModuleNotFoundError:
        with path.open("wb") as file:
            pickle.dump(obj, file)
    else:
        joblib.dump(obj, path)


def load_model(path: Path):
    """Load an object saved by dump_model."""
    try:
        import joblib
    except ModuleNotFoundError:
        with path.open("rb") as file:
            return pickle.load(file)
    try:
        return joblib.load(path)
    except Exception:
        with path.open("rb") as file:
            return pickle.load(file)
