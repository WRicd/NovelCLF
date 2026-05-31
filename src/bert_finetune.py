"""BERT end-to-end fine-tuning for NovelCLF genre classification."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence


def _require_torch_transformers():
    """Import torch and transformers with a helpful error message."""
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import (
            AdamW,
            BertForSequenceClassification,
            BertTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BERT fine-tuning requires torch and transformers. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    return (
        np,
        torch,
        DataLoader,
        Dataset,
        AdamW,
        BertForSequenceClassification,
        BertTokenizer,
        get_linear_schedule_with_warmup,
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NovelDataset:
    """A torch-compatible Dataset that tokenizes text chunks for BERT.

    Lazily imports torch so the class can be defined without torch installed.
    """

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        import torch

        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_bert_finetune(
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    label_list: list[str],
    bert_model_dir: Path,
    save_dir: Path,
    *,
    lr: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 256,
    warmup_ratio: float = 0.1,
    freeze_layers: int = 0,
    seed: int = 42,
) -> dict[str, object]:
    """Fine-tune ``bert-base-chinese`` for sequence classification.

    Parameters
    ----------
    train_texts, train_labels:
        Training data (raw text chunks and their string labels).
    test_texts, test_labels:
        Test data used for per-epoch evaluation.
    label_list:
        Sorted list of all label strings (e.g. ``["历史", "玄幻", ...]``).
    bert_model_dir:
        Path to the pre-trained bert-base-chinese directory.
    save_dir:
        Where to save the best checkpoint.
    lr, epochs, batch_size, max_length, warmup_ratio:
        Standard fine-tuning hyperparameters.
    freeze_layers:
        Number of bottom BERT encoder layers to freeze (0 = train all).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        A dictionary containing per-epoch metrics and best accuracy.
    """
    (
        np,
        torch,
        DataLoader,
        Dataset,
        AdamW,
        BertForSequenceClassification,
        BertTokenizer,
        get_linear_schedule_with_warmup,
    ) = _require_torch_transformers()

    # Reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bert_finetune] Using device: {device}")

    # Label encoding
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}

    # Tokenizer & model
    tokenizer = BertTokenizer.from_pretrained(str(bert_model_dir))
    model = BertForSequenceClassification.from_pretrained(
        str(bert_model_dir),
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # Freeze bottom layers if requested
    if freeze_layers > 0:
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for layer_idx in range(min(freeze_layers, len(model.bert.encoder.layer))):
            for param in model.bert.encoder.layer[layer_idx].parameters():
                param.requires_grad = False
        print(f"[bert_finetune] Froze embeddings + first {freeze_layers} encoder layers")

    # Datasets & loaders
    train_label_ids = [label2id[lb] for lb in train_labels]
    test_label_ids = [label2id[lb] for lb in test_labels]

    train_dataset = NovelDataset(train_texts, train_label_ids, tokenizer, max_length)
    test_dataset = NovelDataset(test_texts, test_label_ids, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Optimizer & scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Training loop
    best_accuracy = 0.0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if step % 50 == 0 or step == len(train_loader):
                print(
                    f"  Epoch {epoch}/{epochs}  Step {step}/{len(train_loader)}  "
                    f"Loss={loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader)
        train_acc = train_correct / train_total

        # --- Evaluate ---
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                eval_correct += (preds == labels).sum().item()
                eval_total += labels.size(0)

        eval_acc = eval_correct / eval_total
        epoch_info = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "train_acc": round(train_acc, 4),
            "eval_acc": round(eval_acc, 4),
        }
        history.append(epoch_info)
        print(
            f"[Epoch {epoch}] loss={avg_loss:.4f}  train_acc={train_acc:.4f}  "
            f"eval_acc={eval_acc:.4f}"
        )

        # Save best checkpoint
        if eval_acc > best_accuracy:
            best_accuracy = eval_acc
            _save_checkpoint(model, tokenizer, label_list, label2id, id2label, save_dir)
            print(f"  ✓ New best model saved (eval_acc={eval_acc:.4f})")

    return {
        "best_eval_accuracy": round(best_accuracy, 4),
        "history": history,
    }


def _save_checkpoint(model, tokenizer, label_list, label2id, id2label, save_dir: Path) -> None:
    """Save model weights, tokenizer, and label mapping to *save_dir*."""
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    meta = {
        "label_list": label_list,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }
    (save_dir / "label_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_bert_finetune(
    texts: list[str],
    model_dir: Path,
    *,
    batch_size: int = 8,
    max_length: int = 256,
) -> list[str]:
    """Run inference with a fine-tuned BERT checkpoint.

    Parameters
    ----------
    texts:
        Raw text strings to classify.
    model_dir:
        Directory containing saved model, tokenizer, and ``label_meta.json``.

    Returns
    -------
    list[str]
        Predicted label strings.
    """
    try:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BERT inference requires torch and transformers. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    meta_path = model_dir / "label_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Label metadata not found at {meta_path}. Is the model directory correct?"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in meta["id2label"].items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            predictions.extend(id2label[pid] for pid in pred_ids)

    return predictions


def predict_bert_finetune_with_scores(
    texts: list[str],
    model_dir: Path,
    *,
    batch_size: int = 8,
    max_length: int = 256,
) -> list[dict[str, object]]:
    """Run inference and return label + per-class probability scores.

    Returns a list of dicts, one per input text, each with keys:
    ``label`` (str) and ``scores`` (dict mapping label → float).
    """
    try:
        import torch
        from transformers import BertForSequenceClassification, BertTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "BERT inference requires torch and transformers. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    meta_path = model_dir / "label_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    label_list = meta["label_list"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(str(model_dir))
    model = BertForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    results: list[dict[str, object]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu()

            for i in range(len(batch)):
                pred_id = logits[i].argmax().item()
                scores = {
                    id2label[j]: round(probs[i, j].item(), 4)
                    for j in range(len(label_list))
                }
                results.append({"label": id2label[pred_id], "scores": scores})

    return results
