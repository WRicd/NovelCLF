"""FastAPI REST API for NovelCLF – Chinese novel genre classification."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``src`` package is importable.
# This mirrors the pattern used in scripts/predict_text.py.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import MODEL_PATHS, SUPPORTED_MODELS
    from src.predict import predict_book_voting, predict_text
    from src.preprocess import clean_text
except ImportError as exc:
    raise ImportError(
        "Could not import NovelCLF src modules. "
        "Make sure you are running from the project root and the src/ package exists. "
        f"Original error: {exc}"
    ) from exc

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ═══════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════


class PredictRequest(BaseModel):
    """Request body for the ``/predict`` endpoint."""

    text: str = Field(..., min_length=1, description="Raw Chinese text to classify.")
    model: str = Field(
        default="nb",
        description="Model name to use for prediction.",
    )


class PredictResponse(BaseModel):
    """Response for single-text prediction."""

    label: str = Field(..., description="Predicted genre label.")
    scores: dict[str, float] | None = Field(
        default=None,
        description="Per-class confidence scores (may be null for some models).",
    )
    model: str = Field(..., description="Model used for this prediction.")


class VotingResponse(BaseModel):
    """Response for whole-book voting prediction."""

    label: str = Field(..., description="Majority-voted genre label.")
    confidence: float = Field(..., description="Fraction of chunks that voted for the winning label.")
    total_chunks_predicted: int = Field(..., description="Number of text chunks that were classified.")
    vote_distribution: dict[str, int] = Field(
        ..., description="Vote counts per genre label."
    )
    model: str = Field(..., description="Model used for this prediction.")


class ModelInfo(BaseModel):
    """Information about a single available model."""

    name: str = Field(..., description="Short model identifier.")
    path: str = Field(..., description="On-disk path to the model artefact.")
    available: bool = Field(
        ..., description="Whether the model file/directory actually exists on disk."
    )


class ModelsResponse(BaseModel):
    """Response listing available models."""

    models: list[ModelInfo]


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    status: str
    project: str
    version: str


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI application
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="NovelCLF API",
    description=(
        "REST API for **NovelCLF** – a Chinese novel genre classifier.  "
        "Supports multiple model backends (Naive Bayes, SVM with TF-IDF, "
        "BERT+SVM, BERT fine-tune) and whole-book majority-voting prediction."
    ),
    version="0.1.0",
)

# Allow all origins so any frontend can integrate.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════


def _validate_model_name(model_name: str) -> None:
    """Raise 422 if the model name is unknown, 404 if the file is missing."""
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown model '{model_name}'. "
                f"Supported models: {sorted(SUPPORTED_MODELS)}"
            ),
        )
    model_path = MODEL_PATHS[model_name]
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Model '{model_name}' is recognised but its file "
                f"({model_path}) was not found on disk. Train it first."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════


@app.get("/", response_model=HealthResponse, tags=["General"])
async def root():
    """Health check / welcome endpoint.

    Returns basic project information and confirms the API is running.
    """
    return HealthResponse(
        status="ok",
        project="NovelCLF",
        version=app.version,
    )


@app.get("/models", response_model=ModelsResponse, tags=["Models"])
async def list_models():
    """List all registered models and whether their artefacts exist on disk.

    Only models whose files are present can be used for prediction.
    """
    items = []
    for name in sorted(SUPPORTED_MODELS):
        p = MODEL_PATHS[name]
        items.append(
            ModelInfo(name=name, path=str(p), available=p.exists())
        )
    return ModelsResponse(models=items)


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """Predict the genre of a raw Chinese text snippet.

    Accepts a JSON body with ``text`` (required) and ``model`` (optional,
    defaults to ``"nb"``).  Returns the predicted label and, when the model
    supports it, per-class confidence scores.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    _validate_model_name(request.model)

    try:
        result = predict_text(text, model_name=request.model)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    return PredictResponse(
        label=result["label"],
        scores=result.get("scores"),
        model=request.model,
    )


@app.post("/predict/file", tags=["Prediction"])
async def predict_file_upload(
    file: UploadFile = File(..., description="A .txt file containing Chinese novel text."),
    model: str = Query(default="nb", description="Model name to use for prediction."),
    voting: bool = Query(
        default=False,
        description="If true, split the file into chunks and use majority voting.",
    ),
):
    """Predict the genre of an uploaded ``.txt`` file.

    When ``voting=false`` (default) the entire file content is classified as a
    single text and a ``PredictResponse`` is returned.

    When ``voting=true`` the file is chunked and each chunk is classified
    independently; the final label is decided by majority vote, returning a
    ``VotingResponse``.
    """
    # Basic validation
    if file.filename and not file.filename.lower().endswith(".txt"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are accepted.",
        )

    _validate_model_name(model)

    # Read uploaded content
    try:
        raw_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}") from exc

    # Try to decode with common Chinese encodings (mirrors preprocess.read_text_file)
    content: str | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            content = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if content is None:
        content = raw_bytes.decode("utf-8", errors="ignore")

    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty or contains no readable text.")

    if voting:
        # Write to a temp file because predict_book_voting expects a file path.
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".txt", delete=False
            ) as tmp:
                tmp.write(content)
                tmp_path = Path(tmp.name)

            result = predict_book_voting(tmp_path, model_name=model)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Voting prediction failed: {exc}") from exc
        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()

        return VotingResponse(
            label=result["label"],
            confidence=result.get("confidence", 0.0),
            total_chunks_predicted=result.get("total_chunks_predicted", 0),
            vote_distribution=result.get("vote_distribution", {}),
            model=model,
        )
    else:
        try:
            result = predict_text(content, model_name=model)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

        return PredictResponse(
            label=result["label"],
            scores=result.get("scores"),
            model=model,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint (python api.py)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
