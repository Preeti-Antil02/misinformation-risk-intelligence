"""
risklens/monitoring.py
======================
Production Observability, Error Tracking & Alert Dispatch Engine for RiskLens v2.1.0.
Features:
1. Sentry Error Tracking with strict PII and secret sanitization (before_send).
2. Multi-Channel Alert Dispatcher (Telegram Admin Chat, Alert Webhook, Sentry).
3. Operational Telemetry & Latency Logger (SQLite operational_telemetry table).
4. Automated Telegram Webhook Health & Backlog Inspector.
5. Model Quality Drift & Retraining Safety Monitors.
"""

import os
import sys
import json
import time
import logging
import sqlite3
import urllib.parse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

load_dotenv()
logger = logging.getLogger("risklens.monitoring")

DATABASES_DIR = Path(os.getenv("DATABASE_DIR", str(BASE_DIR / "databases")))
DATABASES_DIR.mkdir(parents=True, exist_ok=True)
TELEMETRY_DB_PATH = DATABASES_DIR / "feedback.db"

# Environment Variables
SENTRY_DSN = os.getenv("SENTRY_DSN", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_ADMIN_CHAT_ID = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").strip()
ACCURACY_ALERT_THRESHOLD = float(os.getenv("ACCURACY_ALERT_THRESHOLD", "0.75"))

# Optional Sentry import
try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    HAS_SENTRY = True
except ImportError:
    HAS_SENTRY = False
    sentry_sdk = None


# ============================================================================
# 1. SENTRY ERROR TRACKING WITH STRICT PII & SECRET SCRUBBING
# ============================================================================

def _scrub_sentry_event(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Sanitizes Sentry error payloads to prevent PII, tokens, and raw secrets leakage.
    - Strips authorization, API key, and secret token headers.
    - Truncates and sanitizes error breadcrumbs and user context.
    """
    try:
        # 1. Scrub Request Headers
        if "request" in event:
            req = event["request"]
            if "headers" in req and isinstance(req["headers"], dict):
                sensitive_headers = {
                    "authorization", "x-api-key", "x-telegram-bot-api-secret-token",
                    "cookie", "set-cookie", "token"
                }
                for h in list(req["headers"].keys()):
                    if h.lower() in sensitive_headers:
                        req["headers"][h] = "[REDACTED_SECRET]"

            # 2. Scrub query parameters & URL
            if "url" in req and isinstance(req["url"], str):
                parsed = urllib.parse.urlparse(req["url"])
                if parsed.query:
                    qs = urllib.parse.parse_qs(parsed.query)
                    for k in list(qs.keys()):
                        if any(s in k.lower() for s in ["key", "token", "secret", "auth", "pwd"]):
                            qs[k] = ["[REDACTED]"]
                    new_query = urllib.parse.urlencode(qs, doseq=True)
                    req["url"] = urllib.parse.urlunparse(parsed._replace(query=new_query))

        # 3. Anonymize user info
        if "user" in event and isinstance(event["user"], dict):
            event["user"] = {
                "id": "[ANONYMIZED_USER]",
                "ip_address": "[REDACTED_IP]"
            }

        # 4. Scrub Extra Data Context
        if "extra" in event and isinstance(event["extra"], dict):
            for k in list(event["extra"].keys()):
                if any(s in k.lower() for s in ["text", "message", "claim", "input", "body", "prompt"]):
                    event["extra"][k] = "[REDACTED_TEXT]"

        if "request" in event and "data" in event["request"]:
            event["request"]["data"] = "[REDACTED_BODY]"

        # 5. Scrub Breadcrumbs
        if "breadcrumbs" in event and "values" in event["breadcrumbs"]:
            for b in event["breadcrumbs"]["values"]:
                if "data" in b and isinstance(b["data"], dict):
                    for k in list(b["data"].keys()):
                        if any(s in k.lower() for s in ["token", "key", "secret", "password", "text"]):
                            b["data"][k] = "[REDACTED]"

    except Exception as e:
        logger.warning(f"Error during Sentry event scrubbing: {str(e)}")

    return event


def init_error_tracking():
    """Initializes Sentry error tracking with safety boundaries."""
    if not HAS_SENTRY:
        logger.info("sentry-sdk not installed. Skipping third-party error tracking.")
        return False

    if not SENTRY_DSN:
        logger.info("SENTRY_DSN not configured. Operating in local-only error logging mode.")
        return False

    try:
        sentry_logging = LoggingIntegration(
            level=logging.INFO,        # Capture info and above as breadcrumbs
            event_level=logging.ERROR   # Send errors as events
        )

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            integrations=[sentry_logging],
            environment=ENVIRONMENT,
            release="risklens@2.1.0",
            traces_sample_rate=0.2 if ENVIRONMENT == "production" else 1.0,
            before_send=_scrub_sentry_event,
            send_default_pii=False,
            max_breadcrumbs=50
        )
        logger.info("Sentry error tracking initialized successfully with active PII scrubbing.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {str(e)}")
        return False


# ============================================================================
# 2. OPERATIONAL TELEMETRY STORAGE (SQLite)
# ============================================================================

def init_telemetry_db():
    """Ensures operational telemetry tables exist in feedback.db."""
    try:
        conn = sqlite3.connect(str(TELEMETRY_DB_PATH), timeout=5)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS operational_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                latency_ms REAL DEFAULT 0.0,
                status TEXT NOT NULL,
                fallback_type TEXT DEFAULT 'none',
                details TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_alerts_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                channel_status TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON operational_telemetry(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_status ON operational_telemetry(status)")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to initialize telemetry database: {str(e)}")


def log_operational_event(
    event_type: str,
    latency_ms: float = 0.0,
    status: str = "success",
    fallback_type: str = "none",
    details: Optional[Dict[str, Any]] = None
):
    """
    Records an operational event (latency, fallback, error) into SQLite without blocking.
    """
    ts = datetime.utcnow().isoformat()
    details_str = json.dumps(details or {}) if details else ""
    try:
        conn = sqlite3.connect(str(TELEMETRY_DB_PATH), timeout=3)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO operational_telemetry (timestamp, event_type, latency_ms, status, fallback_type, details)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ts, event_type, float(latency_ms), status, fallback_type, details_str))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f"Failed to record operational event: {str(e)}")


def get_operational_metrics(hours: int = 24) -> Dict[str, Any]:
    """
    Aggregates operational health metrics over the specified time window.
    """
    since_ts = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
    default_res = {
        "timeframe_hours": hours,
        "total_requests": 0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "error_count": 0,
        "error_rate_pct": 0.0,
        "fallbacks": {
            "ocr_tesseract": 0,
            "serper_to_ddg": 0,
            "factcheck_to_cache": 0,
            "model_weighted": 0
        },
        "rate_limit_hits": 0,
        "system_status": "Healthy"
    }

    try:
        conn = sqlite3.connect(str(TELEMETRY_DB_PATH), timeout=5)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT latency_ms, status, fallback_type, event_type
            FROM operational_telemetry
            WHERE timestamp >= ?
        """, (since_ts,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return default_res

        latencies = [float(r["latency_ms"]) for r in rows if r["latency_ms"] > 0]
        errors = [r for r in rows if r["status"] == "error"]
        rate_limits = [r for r in rows if r["status"] == "rate_limited" or r["event_type"] == "rate_limit_exceeded"]

        fallback_counts = {
            "ocr_tesseract": sum(1 for r in rows if "tesseract" in str(r["fallback_type"]).lower()),
            "serper_to_ddg": sum(1 for r in rows if "ddg" in str(r["fallback_type"]).lower()),
            "factcheck_to_cache": sum(1 for r in rows if "cache" in str(r["fallback_type"]).lower()),
            "model_weighted": sum(1 for r in rows if "model" in str(r["fallback_type"]).lower())
        }

        total = len(rows)
        avg_lat = round(sum(latencies) / len(latencies), 1) if latencies else 0.0
        p95_lat = round(float(sorted(latencies)[int(len(latencies) * 0.95)]), 1) if latencies else 0.0
        err_rate = round((len(errors) / max(total, 1)) * 100, 2)

        status = "Healthy"
        if err_rate > 5.0 or (latencies and p95_lat > 8000):
            status = "Degraded"
        if err_rate > 20.0:
            status = "Critical"

        return {
            "timeframe_hours": hours,
            "total_requests": total,
            "avg_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
            "error_count": len(errors),
            "error_rate_pct": err_rate,
            "fallbacks": fallback_counts,
            "rate_limit_hits": len(rate_limits),
            "system_status": status
        }
    except Exception as e:
        logger.error(f"Failed to fetch operational metrics: {str(e)}")
        return default_res


# ============================================================================
# 3. MULTI-CHANNEL ALERT DISPATCHER
# ============================================================================

def send_alert(
    title: str,
    message: str,
    severity: str = "warning",
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, bool]:
    """
    Dispatches a critical operational alert across configured channels:
    1. Local log file (Structured INFO / WARNING / ERROR)
    2. Telegram Admin Chat (via direct Bot API)
    3. Alert Webhook (Slack / Discord / Custom webhook)
    4. Sentry (Captured Message / Event)
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    emoji_map = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨",
        "error": "❌",
        "success": "✅"
    }
    icon = emoji_map.get(severity.lower(), "⚠️")
    full_title = f"{icon} [RiskLens Alert - {severity.upper()}] {title}"

    # 1. Log locally
    log_msg = f"ALERT [{severity.upper()}]: {title} - {message}"
    if severity.lower() in ("critical", "error"):
        logger.error(log_msg)
    else:
        logger.warning(log_msg)

    delivery_status = {
        "telegram": False,
        "webhook": False,
        "sentry": False,
        "db_logged": False
    }

    # 2. Dispatch to Telegram Admin Chat
    if TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT_ID:
        try:
            tg_text = (
                f"{full_title}\n\n"
                f"**Time**: `{ts}`\n"
                f"**Environment**: `{ENVIRONMENT}`\n"
                f"**Details**: {message}\n"
            )
            if context:
                tg_text += f"\n**Context**:\n```json\n{json.dumps(context, indent=2)[:500]}\n```"

            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_ADMIN_CHAT_ID,
                "text": tg_text,
                "parse_mode": "Markdown"
            }
            resp = requests.post(url, json=payload, timeout=5)
            delivery_status["telegram"] = resp.status_code == 200
            if resp.status_code != 200:
                logger.warning(f"Telegram alert delivery returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Failed to dispatch Telegram alert: {str(e)}")

    # 3. Dispatch to Alert Webhook (Slack / Discord compatible payload)
    if ALERT_WEBHOOK_URL:
        try:
            webhook_payload = {
                "text": f"{full_title}\n{message}\nEnvironment: {ENVIRONMENT} | Time: {ts}",
                "attachments": [{
                    "title": title,
                    "text": message,
                    "color": "#EF4444" if severity == "critical" else "#F59E0B",
                    "fields": [
                        {"title": "Environment", "value": ENVIRONMENT, "short": True},
                        {"title": "Severity", "value": severity.upper(), "short": True}
                    ]
                }]
            }
            resp = requests.post(ALERT_WEBHOOK_URL, json=webhook_payload, timeout=5)
            delivery_status["webhook"] = resp.status_code in (200, 201, 204)
        except Exception as e:
            logger.error(f"Failed to dispatch Webhook alert: {str(e)}")

    # 4. Dispatch to Sentry
    if HAS_SENTRY and sentry_sdk and SENTRY_DSN:
        try:
            with sentry_sdk.push_scope() as scope:
                scope.set_level(severity.lower() if severity.lower() in ("info", "warning", "error", "fatal") else "warning")
                scope.set_tag("environment", ENVIRONMENT)
                if context:
                    scope.set_context("alert_context", context)
                sentry_sdk.capture_message(f"[{severity.upper()}] {title}: {message}")
            delivery_status["sentry"] = True
        except Exception as e:
            logger.error(f"Failed to capture Sentry alert: {str(e)}")

    # 5. Persist to system_alerts_log table
    try:
        conn = sqlite3.connect(str(TELEMETRY_DB_PATH), timeout=3)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO system_alerts_log (timestamp, severity, title, message, channel_status)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, severity, title, message, json.dumps(delivery_status)))
        conn.commit()
        conn.close()
        delivery_status["db_logged"] = True
    except Exception as e:
        logger.debug(f"Failed to log alert to DB: {str(e)}")

    return delivery_status


# ============================================================================
# 4. TELEGRAM WEBHOOK INSPECTOR & PROBING
# ============================================================================

def check_telegram_webhook_health() -> Dict[str, Any]:
    """
    Queries Telegram's getWebhookInfo API to detect delivery backlogs or silent drops.
    Alerts if pending update count exceeds threshold or if recent errors are detected.
    """
    if not TELEGRAM_BOT_TOKEN:
        return {"status": "unconfigured", "healthy": True, "message": "TELEGRAM_BOT_TOKEN not set"}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo"
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            err_msg = f"Telegram getWebhookInfo failed with HTTP {resp.status_code}"
            logger.warning(err_msg)
            return {"status": "error", "healthy": False, "error": err_msg}

        data = resp.json().get("result", {})
        pending_count = data.get("pending_update_count", 0)
        last_error_msg = data.get("last_error_message", "")
        last_error_date = data.get("last_error_date", 0)
        webhook_url = data.get("url", "")

        is_healthy = True
        issues = []

        # 1. Backlog check
        if pending_count > 20:
            is_healthy = False
            issues.append(f"High pending update backlog: {pending_count} updates waiting")

        # 2. Recent delivery error check (within last 15 minutes)
        if last_error_date and (time.time() - last_error_date) < 900:
            is_healthy = False
            issues.append(f"Recent delivery error: '{last_error_msg}'")

        # Alert if unhealthy
        if not is_healthy:
            send_alert(
                title="Telegram Webhook Delivery Degraded",
                message="; ".join(issues),
                severity="warning",
                context={
                    "pending_update_count": pending_count,
                    "webhook_url": webhook_url,
                    "last_error_message": last_error_msg
                }
            )

        return {
            "status": "healthy" if is_healthy else "degraded",
            "healthy": is_healthy,
            "url": webhook_url,
            "pending_update_count": pending_count,
            "last_error_message": last_error_msg,
            "last_error_date": last_error_date,
            "has_custom_certificate": data.get("has_custom_certificate", False)
        }
    except Exception as e:
        logger.error(f"Telegram webhook health check exception: {str(e)}")
        return {"status": "error", "healthy": False, "error": str(e)}


# ============================================================================
# 5. MODEL QUALITY DRIFT & RETRAINING HEALTH MONITOR
# ============================================================================

def check_model_drift_and_alert(
    min_evaluations: int = 15,
    threshold: float = ACCURACY_ALERT_THRESHOLD
) -> Dict[str, Any]:
    """
    Calculates live model accuracy and checks for quality regressions.
    Alerts if live accuracy drops below the configured threshold.
    """
    from risklens.feedback import calculate_live_accuracy, DatabaseManager

    try:
        acc_data = calculate_live_accuracy()
        total_fb = acc_data.get("total_feedback", 0)
        live_acc = acc_data.get("live_accuracy", 1.0)

        # Check queue surge (unreviewed corrections)
        with DatabaseManager.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM retraining_queue WHERE used_in_training = 0")
            queue_size = cursor.fetchone()[0]

        is_drift = False
        if total_fb >= min_evaluations and live_acc < threshold:
            is_drift = True
            send_alert(
                title="Model Quality Drift Detected",
                message=(
                    f"Live user accuracy has dropped to {live_acc * 100:.1f}% "
                    f"across {total_fb} feedback samples (Threshold: {threshold * 100:.1f}%). "
                    f"Active retraining queue has {queue_size} pending samples."
                ),
                severity="warning",
                context={
                    "live_accuracy": live_acc,
                    "total_feedback": total_fb,
                    "retraining_queue_size": queue_size,
                    "per_language_accuracy": acc_data.get("per_language", {})
                }
            )

        return {
            "checked": True,
            "drift_detected": is_drift,
            "live_accuracy": live_acc,
            "total_feedback": total_fb,
            "retraining_queue_size": queue_size
        }
    except Exception as e:
        logger.error(f"Drift monitoring check failed: {str(e)}")
        return {"checked": False, "error": str(e)}


# Initialize databases and Sentry on import
init_telemetry_db()
init_error_tracking()
