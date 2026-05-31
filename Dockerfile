# ---- Stage 1: Base with dependencies ----
FROM python:3.11-slim AS base

LABEL maintainer="NovelCLF" \
      description="Chinese novel genre classifier — Streamlit dashboard"

# System deps for matplotlib Chinese font rendering + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (CPU-only torch to keep image small)
COPY requirements.txt .
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# ---- Stage 2: Application ----
FROM base AS app

COPY src/ src/
COPY scripts/ scripts/
COPY app.py .
COPY api.py .
COPY 停用词表.txt .
COPY TUTORIAL.md .
COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt

# Create non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 8501 8000

# Default: run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
