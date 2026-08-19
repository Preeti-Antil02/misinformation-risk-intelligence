# ============================================================================
# RiskLens v2.1.0 Enterprise Production Dockerfile for Hugging Face Spaces
# Multi-Process Architecture:
# FastAPI Enterprise Gateway (0.0.0.0:7860) -> Streamlit Dashboard (127.0.0.1:8501)
# Fully compliant with Hugging Face non-root UID 1000 execution
# ============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    HOME=/tmp \
    DATABASE_DIR=/tmp/databases \
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

# Pre-cache base transformer weights during build so runtime startup is instant
RUN python -c "from transformers import RobertaTokenizer, RobertaForSequenceClassification; RobertaTokenizer.from_pretrained('roberta-base'); RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)"

# Copy project modules and application structure
COPY src/ /app/src/
COPY risklens/ /app/risklens/
COPY app/ /app/app/
COPY models/ /app/models/
COPY scripts/ /app/scripts/
COPY api.py /app/api.py
COPY app.py /app/app.py
COPY entrypoint.sh /app/entrypoint.sh

# Ensure persistent mount directories exist, sanitize Windows CRLF endings, and grant universal permissions
RUN mkdir -p /app/databases /app/logs /app/scratch /app/results /data /tmp/hf_cache /tmp/databases /tmp/logs && \
    sed -i 's/\r$//' /app/entrypoint.sh && \
    chmod -R 777 /app /tmp /data 2>/dev/null || true && \
    chmod +x /app/entrypoint.sh

# Expose Hugging Face Space default port
EXPOSE 7860

ENTRYPOINT ["/app/entrypoint.sh"]
