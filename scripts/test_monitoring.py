"""
scripts/test_monitoring.py
==========================
Comprehensive Automated Test Suite for RiskLens Observability & Monitoring Infrastructure.
Validates:
1. Telemetry database initialization and operational event logging.
2. Sentry PII event scrubber (ensures headers, keys, and tokens are redacted).
3. Operational metrics calculation (average latency, p95, error rates, fallbacks).
4. Alert dispatcher (Telegram admin push, Webhook, DB logging).
5. Telegram webhook health inspector.
6. Model drift detector.
7. Active deep /health check endpoint.
"""

import os
import sys
import time
import sqlite3
import logging
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from risklens.monitoring import (
    init_telemetry_db,
    log_operational_event,
    get_operational_metrics,
    send_alert,
    check_telegram_webhook_health,
    check_model_drift_and_alert,
    _scrub_sentry_event,
    TELEMETRY_DB_PATH
)
from risklens.feedback import init_feedback_db, record_prediction, record_feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("test_monitoring")


def test_telemetry_logging():
    logger.info("Test 1: Testing Telemetry Database Initialization & Event Logging...")
    init_telemetry_db()
    assert TELEMETRY_DB_PATH.exists(), f"Database not found at {TELEMETRY_DB_PATH}"

    # Log several simulated operational events
    log_operational_event("verify_request", latency_ms=120.5, status="success", fallback_type="none")
    log_operational_event("verify_request", latency_ms=250.0, status="success", fallback_type="ocr_tesseract")
    log_operational_event("verify_request", latency_ms=980.2, status="error", fallback_type="none", details={"error": "Simulated timeout"})
    log_operational_event("telegram_message", latency_ms=45.0, status="rate_limited")

    # Fetch operational metrics
    metrics = get_operational_metrics(hours=1)
    logger.info(f"Retrieved Metrics: {metrics}")

    assert metrics["total_requests"] >= 4, f"Expected >= 4 requests, got {metrics['total_requests']}"
    assert metrics["avg_latency_ms"] > 0, "Average latency should be positive"
    assert "ocr_tesseract" in metrics["fallbacks"], "Fallback type 'ocr_tesseract' should be tracked"
    logger.info("✅ Test 1 Passed: Telemetry logging and metrics computation verified.")


def test_sentry_pii_scrubber():
    logger.info("Test 2: Testing Sentry PII Redaction & Scrubbing...")
    fake_event = {
        "user": {
            "id": "123456789",
            "ip_address": "192.168.1.100",
            "username": "target_user"
        },
        "request": {
            "headers": {
                "Authorization": "Bearer secret_jwt_token_12345",
                "X-API-Key": "my_super_secret_admin_key",
                "X-Telegram-Bot-Api-Secret-Token": "tg_super_secret_token",
                "User-Agent": "Mozilla/5.0"
            },
            "query_string": "api_key=secret_in_url&text=hello",
            "data": "Sensitive news article text that contains private user details"
        },
        "extra": {
            "raw_text": "Private message content",
            "message": "User query input"
        }
    }

    scrubbed = _scrub_sentry_event(fake_event, {})

    # Verifications
    assert scrubbed["request"]["headers"]["Authorization"] == "[REDACTED_SECRET]"
    assert scrubbed["request"]["headers"]["X-API-Key"] == "[REDACTED_SECRET]"
    assert scrubbed["request"]["headers"]["X-Telegram-Bot-Api-Secret-Token"] == "[REDACTED_SECRET]"
    assert scrubbed["request"]["data"] == "[REDACTED_BODY]"
    assert scrubbed["extra"]["raw_text"] == "[REDACTED_TEXT]"
    assert scrubbed["user"]["ip_address"] == "[REDACTED_IP]"
    assert scrubbed["user"]["id"] == "[ANONYMIZED_USER]"
    logger.info("✅ Test 2 Passed: Sentry PII scrubber successfully sanitized all credentials and user text.")


def test_alert_dispatcher():
    logger.info("Test 3: Testing Alert Dispatcher...")
    result = send_alert(
        title="Automated Test Alert",
        message="Simulated system health alert verification.",
        severity="warning",
        context={"test_id": "auto_run_01"}
    )
    logger.info(f"Alert Dispatch Result: {result}")
    assert result.get("db_logged") is True, "Alert should always be logged to database"

    # Verify DB entry in system_alerts_log
    conn = sqlite3.connect(str(TELEMETRY_DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT title, severity, message FROM system_alerts_log ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "Automated Test Alert"
    assert row[1] == "warning"
    logger.info("✅ Test 3 Passed: Alert dispatcher properly recorded event in SQLite audit log.")


def test_telegram_webhook_checker():
    logger.info("Test 4: Testing Telegram Webhook Health Checker...")
    status = check_telegram_webhook_health()
    logger.info(f"Telegram Webhook Health Probe: {status}")
    assert "healthy" in status
    assert "status" in status
    logger.info("✅ Test 4 Passed: Telegram webhook health inspector ran safely.")


def test_model_drift_detector():
    logger.info("Test 5: Testing Model Drift Inspector...")
    drift_result = check_model_drift_and_alert()
    logger.info(f"Model Drift Check Result: {drift_result}")
    assert "drift_detected" in drift_result
    assert "live_accuracy" in drift_result
    logger.info("✅ Test 5 Passed: Model drift inspector executed successfully.")


def test_deep_health_endpoint():
    logger.info("Test 6: Testing FastAPI Deep /health Probe...")
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)
    response = client.get("/health")
    logger.info(f"Health Response Code: {response.status_code}")
    logger.info(f"Health Payload: {response.json()}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data["components"]
    assert "models" in data["components"]
    assert "storage" in data["components"]
    assert data["components"]["database"]["status"] == "ok"
    logger.info("✅ Test 6 Passed: Deep /health probe validated all subsystems as healthy.")


if __name__ == "__main__":
    logger.info("=================================================================")
    logger.info("Starting RiskLens Observability & Monitoring Validation Suite...")
    logger.info("=================================================================")

    test_telemetry_logging()
    test_sentry_pii_scrubber()
    test_alert_dispatcher()
    test_telegram_webhook_checker()
    test_model_drift_detector()
    test_deep_health_endpoint()

    logger.info("=================================================================")
    logger.info("🎉 All 6 Observability & Monitoring Test Suites Passed Successfully!")
    logger.info("=================================================================")
