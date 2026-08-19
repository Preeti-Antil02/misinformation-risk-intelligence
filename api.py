"""
<<<<<<< HEAD
api.py
======
Production FastAPI Application for RiskLens v2.1.0 Enterprise Intelligence.
Unified backend service providing:
1. Multi-model ensemble inference endpoint (POST /predict) with operational latency logging.
2. Hardened Telegram Webhook receiver (POST /telegram/webhook) with secret token verification.
3. System telemetry & live learning analytics (GET /analytics, GET /analytics/dashboard) with API key auth.
4. Deep Production Health Probe (GET /health) verifying DB, models, storage, scheduler, and webhook.
5. Operational Metrics Endpoint (GET /operations/metrics).
6. Background APScheduler for automated daily retraining, drift checks, and webhook probing.
"""

import os
import sys
import hmac
import time
import sqlite3
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

=======
Misinformation Risk Intelligence System — API Layer
FastAPI inference endpoint

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Example:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "Breaking news: shocking discovery doctors hate!"}'
"""

>>>>>>> origin/main
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
<<<<<<< HEAD
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Security
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security.api_key import APIKeyHeader, APIKeyQuery
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Try importing APScheduler
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    HAS_APSCHEDULER = True
except ImportError:
    HAS_APSCHEDULER = False

# Try importing telegram
try:
    from telegram import Update
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        MessageHandler,
        CallbackQueryHandler,
        filters,
        Application
    )
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    Update = Application = None
=======
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
>>>>>>> origin/main

from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.roberta_model import RobertaClassifier
from src.models.slm_model import QwenClassifier
from src.risk_scoring import RiskScorer
<<<<<<< HEAD
from risklens.feedback import (
    get_analytics_data,
    get_analytics_dashboard_html,
    check_and_retrain,
    init_feedback_db
)
from risklens.telegram_bot import (
    start as telegram_start,
    handle_message as telegram_handle_message,
    handle_callback as telegram_handle_callback,
    verify_telegram_secret_token
)
from risklens.logging_config import setup_logging
from risklens.monitoring import (
    init_error_tracking,
    init_telemetry_db,
    log_operational_event,
    get_operational_metrics,
    send_alert,
    check_telegram_webhook_health,
    check_model_drift_and_alert
)

load_dotenv()
setup_logging()
logger = logging.getLogger("risklens.api")

RISKLENS_API_KEY = os.getenv("RISKLENS_API_KEY", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
DATABASE_DIR = Path(os.getenv("DATABASE_DIR", str(BASE_DIR / "databases")))

# Global singletons
scheduler: Optional[Any] = None
telegram_app: Optional[Any] = None


# ============================================================================
# LIFESPAN CONTEXT: SCHEDULER & TELEGRAM LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages background scheduler, monitoring, and Telegram bot lifecycle."""
    global scheduler, telegram_app

    logger.info("Initializing RiskLens Enterprise Production Services & Observability...")
    init_feedback_db()
    init_telemetry_db()
    init_error_tracking()

    # 1. Start APScheduler for automated tasks
    if HAS_APSCHEDULER:
        try:
            scheduler = AsyncIOScheduler()

            # Job A: Daily 02:00 UTC active learning retraining
            scheduler.add_job(
                check_and_retrain,
                CronTrigger(hour=2, minute=0, timezone="UTC"),
                id="daily_model_retraining",
                name="Daily active learning retraining",
                replace_existing=True,
                kwargs={"min_samples": 500, "force": False}
            )

            # Job B: Daily 01:00 UTC model drift check
            scheduler.add_job(
                check_model_drift_and_alert,
                CronTrigger(hour=1, minute=0, timezone="UTC"),
                id="daily_model_drift_check",
                name="Daily model drift inspection",
                replace_existing=True
            )

            # Job C: Periodic Telegram Webhook Health Check (every 15 minutes)
            scheduler.add_job(
                check_telegram_webhook_health,
                "interval",
                minutes=15,
                id="telegram_webhook_health_monitor",
                name="Periodic Telegram webhook health check",
                replace_existing=True
            )

            scheduler.start()
            logger.info("APScheduler initialized: Retraining (02:00 UTC), Drift Monitor (01:00 UTC), Webhook Inspector (15m).")
        except Exception as e:
            logger.error(f"Failed to start APScheduler: {str(e)}")

    # 2. Initialize Telegram Application for Webhook processing
    if HAS_TELEGRAM and TELEGRAM_TOKEN:
        try:
            telegram_app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
            telegram_app.add_handler(CommandHandler("start", telegram_start))
            telegram_app.add_handler(MessageHandler(filters.TEXT | filters.PHOTO, telegram_handle_message))
            telegram_app.add_handler(CallbackQueryHandler(telegram_handle_callback))

            await telegram_app.initialize()
            await telegram_app.start()
            logger.info("Telegram Webhook Application initialized and ready.")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram Webhook Application: {str(e)}")
            telegram_app = None

    yield

    # Teardown
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("APScheduler shut down.")

    if telegram_app:
        try:
            await telegram_app.stop()
            await telegram_app.shutdown()
            logger.info("Telegram Application shut down.")
        except Exception as e:
            logger.warning(f"Telegram shutdown warning: {str(e)}")


# -------------------------------------------------------
# FastAPI Application & Security Middleware
# -------------------------------------------------------
app = FastAPI(
    title="RiskLens Enterprise Misinformation Intelligence API",
    description=(
        "Production AI verification backend: Multi-Model Ensemble, "
        "LangGraph Search Agent, Telegram Webhook, Observability, and Telemetry."
    ),
    version="2.1.0",
    lifespan=lifespan
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Enforces enterprise HTTP security headers across all endpoints (Item 18)."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# -------------------------------------------------------
# Authentication Dependency for Analytics & Operations
# -------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


async def verify_admin_auth(
    header_key: Optional[str] = Security(api_key_header),
    query_key: Optional[str] = Security(api_key_query)
):
    """Protects analytics and admin endpoints with API key authorization."""
    provided_key = header_key or query_key
    if RISKLENS_API_KEY:
        if provided_key and hmac.compare_digest(provided_key, RISKLENS_API_KEY):
            return True
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid 'X-API-Key' header or '?api_key=' parameter required."
        )
    return True


# -------------------------------------------------------
# Load Model Artifacts at Startup
# -------------------------------------------------------
MODELS_DIR = BASE_DIR / "models"
lr = joblib.load(str(MODELS_DIR / "baseline_logistic.pkl"))
xgb = joblib.load(str(MODELS_DIR / "xgboost_model.pkl"))
tfidf = joblib.load(str(MODELS_DIR / "tfidf_vectorizer.pkl"))
scaler = joblib.load(str(MODELS_DIR / "numeric_scaler.pkl"))

calibrated_model = None
if (MODELS_DIR / "calibrated_ensemble.pkl").exists():
    calibrated_model = joblib.load(str(MODELS_DIR / "calibrated_ensemble.pkl"))

ensemble_meta = None
if (MODELS_DIR / "ensemble_model.pkl").exists():
    ensemble_meta = joblib.load(str(MODELS_DIR / "ensemble_model.pkl"))

roberta = RobertaClassifier()
try:
    if (MODELS_DIR / "roberta_finetuned").exists():
        roberta.load(str(MODELS_DIR / "roberta_finetuned"))
except Exception:
    pass

qwen = QwenClassifier()
=======

# -------------------------------------------------------
# App
# -------------------------------------------------------
app = FastAPI(
    title="Misinformation Risk Intelligence API",
    description=(
        "Assesses misinformation risk using four models: "
        "Logistic Regression, XGBoost, fine-tuned RoBERTa, "
        "and Qwen2.5-3B zero-shot. Primary result is a weighted ensemble."
    ),
    version="2.0.0",
)

# -------------------------------------------------------
# Load models at startup
# -------------------------------------------------------
lr     = joblib.load("models/baseline_logistic.pkl")
xgb    = joblib.load("models/xgboost_model.pkl")
tfidf  = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/numeric_scaler.pkl")

roberta = RobertaClassifier()
roberta.load("models/roberta_finetuned")

qwen = QwenClassifier()   # lazy-loads on first call

>>>>>>> origin/main
tp = TextPreprocessor()
fb = FeatureBuilder()
rs = RiskScorer()

<<<<<<< HEAD

# -------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="News article or claim to analyse (10 to 50,000 characters)"
    )
=======
# -------------------------------------------------------
# Schemas
# -------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article or claim to analyse")
>>>>>>> origin/main


class ModelResult(BaseModel):
    probability_fake: float
    risk_level: str


class EnsembleResult(BaseModel):
    probability_fake: float
    risk_level: str
    source: str
<<<<<<< HEAD
    is_calibrated: bool
=======
>>>>>>> origin/main


class PredictResponse(BaseModel):
    input_text:          str
    ensemble:            EnsembleResult
    roberta:             ModelResult
    qwen_zero_shot:      ModelResult
    xgboost:             ModelResult
    logistic_regression: ModelResult


# -------------------------------------------------------
<<<<<<< HEAD
# System & Deep Health Probing Endpoints
=======
# Endpoints
>>>>>>> origin/main
# -------------------------------------------------------
@app.get("/")
def root():
    return {
<<<<<<< HEAD
        "system":    "RiskLens Misinformation Risk Intelligence System",
        "version":   "2.1.0",
        "status":    "operational",
        "models":    ["Stacking Ensemble (Platt Calibrated)", "RoBERTa", "Qwen2.5-3B", "XGBoost", "Logistic Regression"],
        "endpoints": ["/verify", "/predict", "/telegram/webhook", "/analytics", "/analytics/dashboard", "/operations/metrics", "/health", "/version", "/docs"],
    }


@app.get("/version")
def get_version():
    """Returns application release version and operational runtime configuration."""
    return {
        "version": "2.1.0",
        "release_name": "RiskLens Enterprise v2.1.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "telegram_mode": os.getenv("TELEGRAM_MODE", "polling"),
        "models_loaded": {
            "calibrated_ensemble": calibrated_model is not None,
            "logistic_regression": lr is not None,
            "xgboost": xgb is not None,
            "roberta": roberta is not None,
            "qwen_zero_shot": qwen is not None
        },
        "timestamp": time.time()
=======
        "system":    "Misinformation Risk Intelligence System",
        "version":   "2.0.0",
        "status":    "running",
        "models":    ["RoBERTa (fine-tuned)", "Qwen2.5-3B (zero-shot)", "XGBoost", "Logistic Regression"],
        "endpoints": ["/predict", "/health", "/docs"],
>>>>>>> origin/main
    }


@app.get("/health")
def health():
<<<<<<< HEAD
    """
    Comprehensive Deep Platform Health Probe.
    Actively checks:
    1. SQLite Database read/write accessibility on feedback.db and usage.db.
    2. In-memory model artifacts readiness.
    3. Storage persistent disk write permissions and mount path verification (/data vs /app/databases).
    4. APScheduler background engine status.
    5. Telegram Webhook/Polling subsystem status.
    6. Configuration presence of required secrets.
    Returns HTTP 200 (healthy) or HTTP 503 (degraded/unhealthy).
    """
    check_start = time.time()
    checks: Dict[str, Any] = {}
    is_healthy = True

    # 1. Database Check (Active probe)
    try:
        feedback_db = DATABASE_DIR / "feedback.db"
        usage_db = BASE_DIR / "usage.db"

        # Check feedback DB
        conn_fb = sqlite3.connect(str(feedback_db), timeout=2)
        cursor_fb = conn_fb.cursor()
        cursor_fb.execute("SELECT 1")
        conn_fb.close()

        # Check usage DB
        conn_us = sqlite3.connect(str(usage_db), timeout=2)
        cursor_us = conn_us.cursor()
        cursor_us.execute("SELECT 1")
        conn_us.close()

        checks["database"] = {"status": "ok", "message": "SQLite feedback and usage databases reachable"}
    except Exception as e:
        is_healthy = False
        checks["database"] = {"status": "error", "error": str(e)}

    # 2. Storage Directory Check
    try:
        DATABASE_DIR.mkdir(parents=True, exist_ok=True)
        probe_file = DATABASE_DIR / ".health_probe.tmp"
        probe_file.write_text("probe")
        probe_file.unlink()
        is_persistent = str(DATABASE_DIR) == "/data" and Path("/data").exists()
        checks["storage"] = {
            "status": "ok",
            "path": str(DATABASE_DIR),
            "writeable": True,
            "is_persistent": is_persistent,
            "mode": "persistent_volume" if is_persistent else "ephemeral_container"
        }
    except Exception as e:
        is_healthy = False
        checks["storage"] = {"status": "error", "error": str(e)}

    # 3. Model Readiness Check
    models_ready = all([
        calibrated_model is not None,
        lr is not None,
        xgb is not None,
        tfidf is not None,
        scaler is not None
    ])
    if models_ready:
        checks["models"] = {
            "status": "ok",
            "calibrated_ensemble": True,
            "logistic_regression": True,
            "xgboost": True
        }
    else:
        is_healthy = False
        checks["models"] = {"status": "error", "message": "One or more core model artifacts failed to load"}

    # 4. Background Scheduler Check
    scheduler_active = scheduler.running if scheduler else False
    checks["scheduler"] = {
        "status": "ok" if scheduler_active else "warning",
        "active": scheduler_active
    }

    # 5. Telegram Webhook / Polling Status
    tg_mode = os.getenv("TELEGRAM_MODE", "polling").lower()
    tg_configured = bool(os.getenv("TELEGRAM_BOT_TOKEN"))
    checks["telegram"] = {
        "status": "ok" if tg_configured else "unconfigured",
        "mode": tg_mode,
        "configured": tg_configured
    }

    # 6. Secret Configuration Presence
    checks["secrets_status"] = {
        "telegram_bot_token": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "google_factcheck_api_key": bool(os.getenv("GOOGLE_FACTCHECK_API_KEY")),
        "serper_api_key": bool(os.getenv("SERPER_API_KEY")),
        "risklens_api_key": bool(os.getenv("RISKLENS_API_KEY")),
        "user_id_salt": bool(os.getenv("USER_ID_SALT")),
        "sentry": bool(os.getenv("SENTRY_DSN"))
    }

    total_latency_ms = round((time.time() - check_start) * 1000, 2)
    payload = {
        "status": "healthy" if is_healthy else "unhealthy",
        "version": "2.1.0",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "timestamp": time.time(),
        "latency_ms": total_latency_ms,
        "components": checks
    }

    status_code = 200 if is_healthy else 503
    return JSONResponse(status_code=status_code, content=payload)


# -------------------------------------------------------
# Core Inference Endpoint (with Latency & Event Logging)
# -------------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_ts = time.time()
    text = request.text.strip()

    if len(text.split()) < 3:
        raise HTTPException(
            status_code=422,
            detail="Text is too short. Provide at least a full sentence."
        )

    try:
        # 1. Feature Extraction
        cleaned = tp.basic_clean(text)
        cleaned = tp.truncate(cleaned)

        X_tfidf = tfidf.transform([cleaned])

        temp_df = pd.DataFrame({"text": [text]})
        X_num = fb.build_features(temp_df).astype(np.float64)
        X_num_scaled = scaler.transform(X_num)
        X_combined = hstack([X_tfidf, csr_matrix(X_num_scaled)])

        # 2. Model Inferences
        lr_prob      = float(lr.predict_proba(X_tfidf)[0, 1])
        xgb_prob     = float(xgb.predict_proba(X_combined)[0, 1])
        roberta_prob = float(roberta.predict_proba([cleaned])[0])
        qwen_prob    = float(qwen.predict_proba([cleaned])[0])

        meta_features = np.array([[lr_prob, xgb_prob, roberta_prob, qwen_prob]])

        # 3. Calibrated Ensemble
        is_calibrated = False
        fallback_type = "none"
        if calibrated_model is not None:
            ensemble_prob = float(calibrated_model.predict_proba(meta_features)[0, 1])
            ensemble_source = "Stacking Meta-Learner (Platt Calibrated)"
            is_calibrated = True
        elif ensemble_meta is not None:
            ensemble_prob = float(ensemble_meta.predict_proba(meta_features)[0, 1])
            ensemble_source = "Stacking Meta-Learner (Raw)"
        else:
            ensemble_prob = (lr_prob * 0.10) + (xgb_prob * 0.35) + (roberta_prob * 0.30) + (qwen_prob * 0.25)
            ensemble_source = "Weighted Ensemble Fallback"
            fallback_type = "model_weighted_fallback"

        ensemble_risk = rs.score_ensemble(ensemble_prob)

        elapsed_ms = (time.time() - start_ts) * 1000
        log_operational_event(
            event_type="verify_request",
            latency_ms=elapsed_ms,
            status="success",
            fallback_type=fallback_type,
            details={"risk_level": ensemble_risk}
        )

        return PredictResponse(
            input_text=text[:500],
            ensemble=EnsembleResult(
                probability_fake=round(ensemble_prob, 4),
                risk_level=ensemble_risk,
                source=ensemble_source,
                is_calibrated=is_calibrated,
            ),
            roberta=ModelResult(
                probability_fake=round(roberta_prob, 4),
                risk_level=rs.score(roberta_prob),
            ),
            qwen_zero_shot=ModelResult(
                probability_fake=round(qwen_prob, 4),
                risk_level=rs.score(qwen_prob),
            ),
            xgboost=ModelResult(
                probability_fake=round(xgb_prob, 4),
                risk_level=rs.score(xgb_prob),
            ),
            logistic_regression=ModelResult(
                probability_fake=round(lr_prob, 4),
                risk_level=rs.score(lr_prob),
            ),
        )
    except Exception as e:
        elapsed_ms = (time.time() - start_ts) * 1000
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        log_operational_event(
            event_type="verify_request",
            latency_ms=elapsed_ms,
            status="error",
            fallback_type="none",
            details={"error": str(e)}
        )
        send_alert(
            title="Inference Pipeline Exception",
            message=f"Prediction request failed: {str(e)}",
            severity="error"
        )
        raise HTTPException(
            status_code=500,
            detail="Inference pipeline error. The request could not be processed."
        )


# -------------------------------------------------------
# Full Agentic Verification Endpoint (LangGraph + Web + Fact-Check)
# -------------------------------------------------------
class VerifyRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=50000, description="Claim or article text to verify")
    url: Optional[str] = Field(None, description="Optional web URL of the source article")


class VerifyResponse(BaseModel):
    claim: str
    verdict: str
    sources: List[Dict[str, Any]] = []
    risk_score: float
    risk_level: str
    explanation: Optional[Dict[str, Any]] = None
    fact_checker_available: bool = False
    latency_ms: float = 0.0


@app.post("/verify", response_model=VerifyResponse)
def full_verify(request: VerifyRequest):
    """
    Full Agentic Verification Pipeline (Exact path executed by Telegram Bot & UI):
    1. LangGraph Claim Extraction
    2. Multi-query Web Research (Google Serper API / DuckDuckGo search)
    3. Google Fact Check Tools API Verification
    4. Domain Credibility Reputation Scoring
    5. Calibrated Neural Model Baseline
    6. Evidence-Weighted Synthesis (70% Web Evidence + 30% Neural Risk)
    """
    from risklens.agent import verify as agent_verify

    start_ts = time.time()
    text = request.text.strip()
    if len(text.split()) < 2:
        raise HTTPException(status_code=422, detail="Text is too short. Provide at least a phrase or sentence.")

    try:
        result = agent_verify(text, url=request.url)
        elapsed_ms = round((time.time() - start_ts) * 1000, 2)
        result["latency_ms"] = elapsed_ms

        log_operational_event(
            event_type="full_verify_request",
            latency_ms=elapsed_ms,
            status="success",
            fallback_type="none" if result.get("sources") else "web_empty_fallback",
            details={
                "risk_level": result.get("risk_level"),
                "sources_count": len(result.get("sources", []))
            }
        )
        return VerifyResponse(**result)
    except Exception as e:
        elapsed_ms = round((time.time() - start_ts) * 1000, 2)
        logger.error(f"Full verification pipeline error: {str(e)}", exc_info=True)
        log_operational_event(
            event_type="full_verify_request",
            latency_ms=elapsed_ms,
            status="error",
            details={"error": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"Full verification pipeline failed: {str(e)}")


# -------------------------------------------------------
# Telegram Webhook Receiver (Item 20-Bonus)
# -------------------------------------------------------
@app.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    """
    Production Telegram Webhook Endpoint.
    Validates X-Telegram-Bot-Api-Secret-Token header and processes incoming updates asynchronously.
    """
    secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
    if not verify_telegram_secret_token(secret_token):
        logger.warning("Rejected unauthorized Telegram webhook request: Invalid secret token")
        raise HTTPException(status_code=403, detail="Forbidden: Invalid Telegram secret token")

    if not telegram_app:
        logger.error("Telegram Application is not initialized.")
        raise HTTPException(status_code=503, detail="Telegram service unavailable")

    try:
        data = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error handling Telegram webhook update: {str(e)}", exc_info=True)
        return {"status": "error"}


# -------------------------------------------------------
# Protected Analytics & Operational Metrics Endpoints
# -------------------------------------------------------
@app.get("/analytics", dependencies=[Depends(verify_admin_auth)])
def get_analytics():
    """Returns continuous learning and verification telemetry."""
    return JSONResponse(content=get_analytics_data())


@app.get("/analytics/dashboard", response_class=HTMLResponse, dependencies=[Depends(verify_admin_auth)])
def get_analytics_dashboard():
    """Serves standalone HTML dashboard for live accuracy and queue metrics."""
    return HTMLResponse(content=get_analytics_dashboard_html())


@app.get("/operations/metrics", dependencies=[Depends(verify_admin_auth)])
def get_ops_metrics(hours: int = 24):
    """Returns real-time operational health, latencies, error rates, and fallbacks."""
    return JSONResponse(content=get_operational_metrics(hours=hours))
=======
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()

    if len(text.split()) < 5:
        raise HTTPException(
            status_code=422,
            detail="Text is too short. Provide at least a few sentences."
        )

    cleaned = tp.basic_clean(text)
    cleaned = tp.truncate(cleaned)

    X_tfidf = tfidf.transform([cleaned])

    temp_df      = pd.DataFrame({"text": [cleaned]})
    X_num        = fb.build_features(temp_df).astype(np.float64)
    X_num_scaled = scaler.transform(X_num)
    X_combined   = hstack([X_tfidf, csr_matrix(X_num_scaled)])

    lr_prob      = float(lr.predict_proba(X_tfidf)[0, 1])
    xgb_prob     = float(xgb.predict_proba(X_combined)[0, 1])
    roberta_prob = float(roberta.predict_proba([cleaned])[0])
    qwen_prob    = float(qwen.predict_proba([cleaned])[0])

    # Ensemble: XGBoost 35% · RoBERTa 30% · Qwen 25% · LR 10%
    qwen_outlier = (
        qwen_prob > 0.7 and lr_prob < 0.4 and xgb_prob < 0.4 and roberta_prob < 0.4
    ) or (
        qwen_prob < 0.3 and lr_prob > 0.6 and xgb_prob > 0.6 and roberta_prob > 0.6
    )

    if qwen_outlier:
        ensemble_prob   = (lr_prob * 0.15) + (xgb_prob * 0.50) + (roberta_prob * 0.35)
        ensemble_source = "LR + XGBoost + RoBERTa (Qwen excluded — outlier)"
    else:
        ensemble_prob   = (lr_prob * 0.10) + (xgb_prob * 0.35) + (roberta_prob * 0.30) + (qwen_prob * 0.25)
        ensemble_source = "weighted ensemble"

    ensemble_risk = rs.score_ensemble(ensemble_prob)

    return PredictResponse(
        input_text=text,
        ensemble=EnsembleResult(
            probability_fake=round(ensemble_prob, 4),
            risk_level=ensemble_risk,
            source=ensemble_source,
        ),
        roberta=ModelResult(
            probability_fake=round(roberta_prob, 4),
            risk_level=rs.score(roberta_prob),
        ),
        qwen_zero_shot=ModelResult(
            probability_fake=round(qwen_prob, 4),
            risk_level=rs.score(qwen_prob),
        ),
        xgboost=ModelResult(
            probability_fake=round(xgb_prob, 4),
            risk_level=rs.score(xgb_prob),
        ),
        logistic_regression=ModelResult(
            probability_fake=round(lr_prob, 4),
            risk_level=rs.score(lr_prob),
        ),
    )
>>>>>>> origin/main
