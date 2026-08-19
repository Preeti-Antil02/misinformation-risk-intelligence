#!/bin/bash
set -e

echo "🛡️ Starting RiskLens v2.1.0 Enterprise Container..."

# Ensure writeable cache directories for HF non-root execution (UID 1000)
export HOME="${HOME:-/tmp}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf_cache}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"

if [ -d "/data" ] && [ -w "/data" ]; then
    export DATABASE_DIR="/data"
    echo "📦 Persistent storage detected at /data."
else
    export DATABASE_DIR="/tmp/databases"
    echo "⚠️ Using writeable storage at /tmp/databases."
fi

mkdir -p "$DATABASE_DIR" /tmp/logs /tmp/scratch /tmp/results /tmp/hf_cache /app/logs /app/databases 2>/dev/null || true

# 1. Start Streamlit Dashboard in background on port 8501
echo "🌐 [1/3] Launching Streamlit Dashboard on 127.0.0.1:8501..."
python -m streamlit run app.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none &

# 2. Start Telegram Bot in background (if configured)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "🤖 [2/3] Launching Telegram Bot in polling mode..."
    python risklens/telegram_bot.py &
else
    echo "ℹ️ [2/3] TELEGRAM_BOT_TOKEN not configured. Telegram bot worker skipped."
fi

# 3. Start FastAPI Enterprise Gateway in foreground on Space port 7860
PORT="${PORT:-7860}"
echo "🚀 [3/3] Launching FastAPI Enterprise Gateway on port $PORT..."
exec python -m uvicorn api:app --host 0.0.0.0 --port "$PORT" --log-level info
