"""Lightweight file-based experiment tracking for NovelCLF."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import EXPERIMENTS_DIR, MODEL_PATHS, ensure_project_dirs


_LOG_FILE = EXPERIMENTS_DIR / "experiment_log.json"


class ExperimentLog:
    """Append-only JSONL experiment log.

    Each line in the log file is a self-contained JSON object describing one
    training run.  The class provides helpers to query, filter, and summarise
    logged experiments.
    """

    def __init__(self, log_path: Path = _LOG_FILE) -> None:
        self.log_path = log_path

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def log_experiment(
        self,
        model_name: str,
        split: str,
        metrics: dict[str, Any],
        *,
        hyperparams: dict[str, Any] | None = None,
        notes: str | None = None,
        duration: float | None = None,
    ) -> dict[str, Any]:
        """Append a single experiment entry and return it."""
        ensure_project_dirs()

        model_path = MODEL_PATHS.get(model_name)
        entry: dict[str, Any] = {
            "experiment_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "split": split,
            "hyperparams": hyperparams or {},
            "metrics": {
                k: v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
            "model_path": str(model_path) if model_path else "",
            "notes": notes or "",
            "duration_seconds": duration,
        }

        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _load_all(self) -> list[dict[str, Any]]:
        """Read every entry from the JSONL log file."""
        if not self.log_path.exists():
            return []
        entries: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def list_experiments(
        self,
        model_name: str | None = None,
        split: str | None = None,
        top_n: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return experiments, optionally filtered and limited."""
        entries = self._load_all()
        if model_name:
            entries = [e for e in entries if e["model_name"] == model_name]
        if split:
            entries = [e for e in entries if e["split"] == split]
        # Most recent first
        entries.sort(key=lambda e: e["timestamp"], reverse=True)
        if top_n is not None:
            entries = entries[:top_n]
        return entries

    def best_experiment(
        self,
        metric: str = "accuracy",
        model_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the single best experiment by *metric*."""
        entries = self.list_experiments(model_name=model_name)
        if not entries:
            return None
        return max(entries, key=lambda e: e["metrics"].get(metric, -1))

    def to_dataframe(self):
        """Convert all experiments to a *pandas* DataFrame.

        Metrics are flattened into top-level columns prefixed with ``m_``.
        """
        import pandas as pd  # noqa: F811 – lazy import

        entries = self._load_all()
        rows: list[dict[str, Any]] = []
        for entry in entries:
            row = {
                "experiment_id": entry["experiment_id"],
                "timestamp": entry["timestamp"],
                "model_name": entry["model_name"],
                "split": entry["split"],
                "notes": entry.get("notes", ""),
                "duration_seconds": entry.get("duration_seconds"),
            }
            for k, v in entry.get("metrics", {}).items():
                row[f"m_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_table(
        headers: list[str],
        rows: list[list[str]],
        col_widths: list[int] | None = None,
    ) -> str:
        """Build a simple ASCII table string."""
        if col_widths is None:
            col_widths = [
                max(len(h), *(len(r[i]) for r in rows) if rows else 0)
                for i, h in enumerate(headers)
            ]
            # Ensure minimum width matches header
            col_widths = [max(w, len(h)) for w, h in zip(col_widths, headers)]

        sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
        hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
        lines = [sep, hdr, sep]
        for row in rows:
            line = "| " + " | ".join(
                cell.ljust(w) for cell, w in zip(row, col_widths)
            ) + " |"
            lines.append(line)
        lines.append(sep)
        return "\n".join(lines)

    def summary(self, entries: list[dict[str, Any]] | None = None) -> str:
        """Return a formatted summary table string."""
        if entries is None:
            entries = self._load_all()
        if not entries:
            return "(no experiments logged)"

        headers = ["ID (short)", "Timestamp", "Model", "Split", "Accuracy", "Macro-F1", "Duration", "Notes"]
        rows: list[list[str]] = []
        for e in entries:
            m = e.get("metrics", {})
            dur = e.get("duration_seconds")
            rows.append([
                e["experiment_id"][:8],
                e["timestamp"][:19],
                e["model_name"],
                e["split"],
                f"{m.get('accuracy', 0):.4f}",
                f"{m.get('macro_f1', 0):.4f}",
                f"{dur:.1f}s" if dur is not None else "–",
                (e.get("notes") or "")[:30],
            ])
        return self._fmt_table(headers, rows)
