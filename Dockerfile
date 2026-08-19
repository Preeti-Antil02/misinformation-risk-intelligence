<<<<<<< HEAD
# ============================================================================
# RiskLens v2.1.0 Enterprise Production Dockerfile for Hugging Face Spaces
# Multi-Process Container: Streamlit (7860), FastAPI (8000), Telegram Bot
# ============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    DATABASE_DIR=/app/databases \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HOME=/tmp/hf_cache

WORKDIR /app

# Install system libraries, build tools, and Tesseract OCR with English and Hindi packs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm || true

# Copy project modules and application structure
COPY src/ /app/src/
COPY risklens/ /app/risklens/
COPY app/ /app/app/
COPY models/ /app/models/
COPY scripts/ /app/scripts/
COPY api.py /app/api.py
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh

# Ensure persistent mount directories and executable permissions exist
RUN mkdir -p /app/databases /app/logs /app/scratch /app/results /data && \
    chmod +x /app/entrypoint.sh

# Expose Hugging Face Space default port
EXPOSE 7860

# Health probe for container readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
=======
FROM python:3.10-slim

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

EXPOSE 7860

HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.fileWatcherType=none"]
>>>>>>> origin/main
