#!/bin/bash
set -e

echo "🛡️ Starting RiskLens v2.1.0 Enterprise Container..."

# Determine persistent database directory
if [ -d "/data" ]; then
    export DATABASE_DIR="/data"
    echo "📦 Persistent storage detected at /data. Using DATABASE_DIR=/data"
else
    export DATABASE_DIR="/app/databases"
    echo "⚠️ Running in ephemeral mode. Using DATABASE_DIR=/app/databases"
fi

mkdir -p "$DATABASE_DIR" /app/logs /app/scratch /app/results

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
