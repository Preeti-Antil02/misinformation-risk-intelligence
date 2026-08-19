# ============================================================================
# RiskLens v2.1.0 Enterprise Production Dockerfile for Hugging Face Spaces
# Multi-Process Container with Nginx Reverse Proxy:
# Nginx (Port 7860) -> FastAPI (Port 8000) + Streamlit (Port 8501) + Telegram Bot
# ============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    DATABASE_DIR=/app/databases \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/hub \
    HF_HOME=/root/.cache/huggingface/hub

WORKDIR /app

# Install system libraries, build tools, nginx, and Tesseract OCR with English and Hindi packs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    nginx \
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

# Copy Nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy project modules and application structure
COPY src/ /app/src/
COPY risklens/ /app/risklens/
COPY app/ /app/app/
COPY models/ /app/models/
COPY scripts/ /app/scripts/
COPY api.py /app/api.py
COPY app.py /app/app.py
COPY entrypoint.py /app/entrypoint.py
COPY entrypoint.sh /app/entrypoint.sh

# Ensure persistent mount directories and executable permissions exist
RUN mkdir -p /app/databases /app/logs /app/scratch /app/results /data && \
    chmod +x /app/entrypoint.sh /app/entrypoint.py

# Expose Hugging Face Space default port
EXPOSE 7860

# Health probe for container readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

ENTRYPOINT ["python", "/app/entrypoint.py"]
