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

mkdir -p "$DATABASE_DIR" /tmp/logs /tmp/scratch /tmp/results /tmp/hf_cache \
         /tmp/nginx_body /tmp/nginx_proxy /tmp/nginx_fastcgi /tmp/nginx_uwsgi /tmp/nginx_scgi \
         /app/logs /app/databases 2>/dev/null || true

# 1. Start Streamlit Dashboard in background on port 8501
echo "🌐 [1/4] Launching Streamlit Dashboard on 127.0.0.1:8501..."
python -m streamlit run app.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.enableWebsocketCompression=false \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none &
STREAMLIT_PID=$!

# 2. Start FastAPI on internal port 8000
echo "⚡ [2/4] Launching FastAPI Backend on 127.0.0.1:8000..."
python -m uvicorn api:app \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level info \
    --workers 1 &
FASTAPI_PID=$!

# 3. Start Telegram Bot in background (if configured and in polling mode)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    TGMODE="${TELEGRAM_MODE:-polling}"
    if [ "$TGMODE" = "webhook" ]; then
        echo "🤖 [3/4] Telegram Bot will run in webhook mode via FastAPI (no standalone worker)."
    else
        echo "🤖 [3/4] Launching Telegram Bot in polling mode..."
        python risklens/telegram_bot.py &
    fi
else
    echo "ℹ️ [3/4] TELEGRAM_BOT_TOKEN not configured. Telegram bot worker skipped."
fi

# 4. Wait for FastAPI and Streamlit to be ready, then start nginx as the public gateway
echo "⏳ [4/4] Waiting for services to be ready..."
sleep 8

echo "🚀 [4/4] Launching nginx reverse proxy on port 7860..."
exec nginx -g "daemon off;"
