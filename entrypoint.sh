#!/bin/bash
# ============================================================================
# RiskLens v2.1.0 Multi-Process Entrypoint for Hugging Face Spaces
# Manages:
# 1. FastAPI Backend (127.0.0.1:8000)
# 2. Streamlit Dashboard (127.0.0.1:8501)
# 3. Telegram Bot Worker (Background)
# 4. Nginx Reverse Proxy (0.0.0.0:7860 - Foreground)
# ============================================================================

set -e

echo "🛡️ Starting RiskLens v2.1.0 Enterprise System on Hugging Face Spaces..."

# Determine persistent database directory
if [ -d "/data" ]; then
    export DATABASE_DIR="/data"
    echo "📦 Persistent storage detected at /data. Using DATABASE_DIR=/data"
else
    export DATABASE_DIR="/app/databases"
    echo "⚠️ Running in ephemeral mode. Using DATABASE_DIR=/app/databases"
fi

mkdir -p "$DATABASE_DIR" /app/logs /app/scratch /app/results

# Graceful termination handler
cleanup() {
    echo "🛑 Received termination signal. Shutting down RiskLens child processes..."
    if [ -n "$API_PID" ] && kill -0 "$API_PID" 2>/dev/null; then
        kill -TERM "$API_PID" || true
    fi
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        kill -TERM "$STREAMLIT_PID" || true
    fi
    if [ -n "$BOT_PID" ] && kill -0 "$BOT_PID" 2>/dev/null; then
        kill -TERM "$BOT_PID" || true
    fi
    exit 0
}

trap cleanup SIGTERM SIGINT

# 1. Start FastAPI Backend in background on 127.0.0.1:8000
echo "🚀 Launching FastAPI Backend on 127.0.0.1:8000..."
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1 --log-level info &
API_PID=$!
echo "✓ FastAPI Backend started (PID: $API_PID)"

# 2. Start Streamlit Dashboard in background on 127.0.0.1:8501
echo "🌐 Launching Streamlit Dashboard on 127.0.0.1:8501..."
python3 -m streamlit run app.py \
    --server.port=8501 \
    --server.address=127.0.0.1 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --server.fileWatcherType=none &
STREAMLIT_PID=$!
echo "✓ Streamlit Dashboard started (PID: $STREAMLIT_PID)"

# 3. Start Telegram Bot in background (if configured)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "🤖 Launching Telegram Bot in polling mode..."
    python3 risklens/telegram_bot.py &
    BOT_PID=$!
    echo "✓ Telegram Bot started (PID: $BOT_PID)"
else
    echo "ℹ️ TELEGRAM_BOT_TOKEN not set. Telegram bot worker skipped."
fi

# 4. Wait for FastAPI backend to become ready before launching Nginx
echo "⏳ Waiting for FastAPI Backend to initialize on 127.0.0.1:8000..."
for i in $(seq 1 30); do
    if curl -s -f http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "✅ FastAPI Backend is active and healthy!"
        break
    fi
    echo "Waiting for FastAPI... ($i/30)"
    sleep 2
done

# 5. Start Nginx Reverse Proxy in foreground on port 7860
echo "⚡ Launching Nginx Reverse Proxy on port 7860..."
exec nginx -g "daemon off;"
