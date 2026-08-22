"""
entrypoint.py
=============
Enterprise Multi-Process Orchestrator for Hugging Face Spaces.
Spawns and monitors:
1. FastAPI Backend (127.0.0.1:8000)
2. Streamlit Dashboard (127.0.0.1:8501)
3. Telegram Polling Worker (if TELEGRAM_BOT_TOKEN set)
4. Nginx Reverse Proxy (0.0.0.0:7860)
"""

import os
import sys
import time
import subprocess
import signal
import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_DIR = Path(os.getenv("DATABASE_DIR", "/app/databases"))
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
Path("/app/logs").mkdir(parents=True, exist_ok=True)

processes = []

def cleanup(signum, frame):
    print("🛑 Received shutdown signal. Terminating child processes...")
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    sys.exit(0)

signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)

def main():
    print("🛡️ RiskLens v2.1.0 Multi-Process Container Initializing...")
    
    # 1. Launch FastAPI Backend
    print("🚀 [1/4] Launching FastAPI Backend on 127.0.0.1:8000...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
        cwd=BASE_DIR
    )
    processes.append(api_proc)
    
    # 2. Poll FastAPI Backend until healthy
    print("⏳ [2/4] Awaiting FastAPI backend initialization...")
    fastapi_healthy = False
    for i in range(45):
        if api_proc.poll() is not None:
            print(f"❌ FATAL: FastAPI process exited unexpectedly with code {api_proc.returncode}")
            break
        try:
            resp = urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
            if resp.status == 200:
                print("✅ FastAPI Backend is active and healthy!")
                fastapi_healthy = True
                break
        except Exception as e:
            print(f"Waiting for FastAPI... ({i+1}/45): {e}")
        time.sleep(2)
        
    if not fastapi_healthy:
        print("⚠️ Warning: FastAPI backend did not report healthy within timeout.")

    # 3. Launch Streamlit Dashboard
    print("🌐 [3/4] Launching Streamlit Dashboard on 127.0.0.1:8501...")
    st_proc = subprocess.Popen(
        [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=127.0.0.1",
            "--server.headless=true",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--server.fileWatcherType=none"
        ],
        cwd=BASE_DIR
    )
    processes.append(st_proc)
    
    # 4. Launch Telegram Bot (if token configured and in polling mode)
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_mode = os.getenv("TELEGRAM_MODE", "polling").lower()
    if tg_token:
        if tg_mode == "webhook":
            print("🤖 Telegram Bot will run in webhook mode via FastAPI (no standalone worker).")
        else:
            print("🤖 Launching Telegram Bot worker in polling mode...")
            tg_proc = subprocess.Popen([sys.executable, "risklens/telegram_bot.py"], cwd=BASE_DIR)
            processes.append(tg_proc)
    else:
        print("ℹ️ TELEGRAM_BOT_TOKEN not configured. Telegram bot worker skipped.")
        
    time.sleep(3)
    
    # 5. Launch Nginx in foreground on 7860
    print("⚡ [4/4] Starting Nginx Reverse Proxy on port 7860...")
    nginx_proc = subprocess.Popen(["nginx", "-g", "daemon off;"])
    processes.append(nginx_proc)
    
    # Wait on Nginx process
    nginx_proc.wait()

if __name__ == "__main__":
    main()
