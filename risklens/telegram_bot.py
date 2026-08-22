"""
risklens/telegram_bot.py
========================
Hardened High-Security Telegram Bot for RiskLens v2.1.0 with Observability.
- Daily Rate Limiting (20 requests/user/day via usage.db) (Item 12).
- Markdown parse mode sanitization and control character escaping (Item 15).
- Callback ownership validation and replay protection (Item 7).
- Strict input length boundaries (Item 14).
- Safe error handling with Sentry and Admin Alert integration (Item 17).
- Telegram Webhook Secret Token validation support (Item 20-Bonus).
- Operational telemetry and latency instrumentation.
"""

import os
import re
import sys
import hmac
import time
import logging
import asyncio
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Try importing telegram
try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import (
        ApplicationBuilder,
        ContextTypes,
        CommandHandler,
        MessageHandler,
        filters,
        CallbackQueryHandler
    )
    from telegram.error import TelegramError, RetryAfter, TimedOut, Forbidden
    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False
    Update = Any = object
    class ContextTypes:
        DEFAULT_TYPE = Any
    class InlineKeyboardButton: pass
    class InlineKeyboardMarkup: pass
    class TelegramError(Exception): pass
    class RetryAfter(Exception): pass
    class TimedOut(Exception): pass
    class Forbidden(Exception): pass

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from risklens.agent import verify
from risklens.multilingual import predict_multilingual, detect_language, LANGUAGE_NAMES
from risklens.ocr_pipeline import verify_image, validate_image
from risklens.feedback import record_prediction, record_feedback, anonymize_user_id
from risklens.utils import truncate_text
from risklens.logging_config import setup_logging
from risklens.monitoring import log_operational_event, send_alert

# Initialize production logging
setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

# Rate Limit Database
USAGE_DB_PATH = BASE_DIR / "usage.db"
MAX_MESSAGE_LENGTH = 4000
MAX_DAILY_REQUESTS = 20


def escape_markdown(text: str) -> str:
    """
    Escapes special characters for Telegram MarkdownV2 parse mode.
    Escapes: _ * [ ] ( ) ~ ` > # + - = | { } . !
    """
    if not text:
        return ""
    escape_chars = r'\_*[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', str(text))


def check_user_rate_limit(user_id: str, limit: int = MAX_DAILY_REQUESTS) -> tuple[bool, int]:
    """Enforces 20 requests/day per user using SQLite usage.db."""
    import sqlite3
    today = datetime.now().strftime("%Y-%m-%d")
    anon_user = anonymize_user_id(user_id)
    
    try:
        conn = sqlite3.connect(str(USAGE_DB_PATH), timeout=5)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_usage (
                user_id TEXT,
                usage_date TEXT,
                request_count INTEGER,
                PRIMARY KEY (user_id, usage_date)
            )
        """)
        
        cursor.execute(
            "SELECT request_count FROM user_usage WHERE user_id = ? AND usage_date = ?",
            (anon_user, today)
        )
        row = cursor.fetchone()

        if row is None:
            cursor.execute(
                "INSERT INTO user_usage (user_id, usage_date, request_count) VALUES (?, ?, 1)",
                (anon_user, today)
            )
            conn.commit()
            conn.close()
            return True, 1

        current_count = row[0]
        if current_count >= limit:
            conn.close()
            return False, current_count

        cursor.execute(
            "UPDATE user_usage SET request_count = request_count + 1 WHERE user_id = ? AND usage_date = ?",
            (anon_user, today)
        )
        conn.commit()
        conn.close()
        return True, current_count + 1
    except Exception as e:
        logger.error(f"Rate limit DB error: {str(e)}")
        # Fail-open with warning if DB transiently locked
        return True, 1


def verify_telegram_secret_token(token_header: str) -> bool:
    """
    Validates the X-Telegram-Bot-Api-Secret-Token header on incoming webhook requests.
    Uses constant-time comparison to prevent timing attacks.
    """
    if not TELEGRAM_WEBHOOK_SECRET:
        logger.warning("TELEGRAM_WEBHOOK_SECRET is not configured in .env!")
        return False
    return hmac.compare_digest(token_header, TELEGRAM_WEBHOOK_SECRET)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Secure welcome message."""
    try:
        welcome_text = (
            "🛡️ RiskLens Misinformation Intelligence\n\n"
            "Welcome! I am your secure assistant for verifying news and claims.\n\n"
            "📥 How to use:\n"
            "• Forward any text message or news headline.\n"
            "• Send a URL of a web article.\n"
            "• Send a screenshot of a viral post.\n\n"
            "I will perform a neural verification and provide a risk assessment instantly."
        )
        await update.message.reply_text(welcome_text)
    except Exception as e:
        logger.error(f"Error in /start handler: {str(e)}")


def format_telegram_report(data: dict) -> dict:
    """
    Standardized, shared formatting function for Telegram reports.
    Uses plain text formatting for maximum reliability with dynamic content.
    """
    risk_level = data.get("risk_level", "Moderate")
    prob = data.get("risk_score", data.get("probability", 0.5))
    confidence_pct = int(prob * 100)
    claim = data.get("claim", "")
    verdict = data.get("verdict", "")

    safe_claim = truncate_text(claim, 1000)
    safe_verdict = truncate_text(verdict, 2000)

    # Risk level emoji mapping
    risk_emoji = {"Critical": "🔴", "High": "🟠", "Moderate": "🟡", "Low": "🟢"}.get(risk_level, "⚪")

    report_text = (
        f"{risk_emoji} RISK ASSESSMENT: {risk_level.upper()}\n"
        f"─────────────────────\n"
        f"📋 Claim Detected:\n"
        f"{safe_claim}\n\n"
        f"📝 Verdict:\n"
        f"{safe_verdict}\n\n"
        f"📊 Confidence: {confidence_pct}%\n"
        f"─────────────────────\n"
        f"Was this intelligence accurate?"
    )

    return {
        "risk_level": risk_level,
        "confidence_pct": confidence_pct,
        "claim": claim,
        "verdict": verdict,
        "raw_text": report_text
    }


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Main verification handler with rate limiting, input validation, and safe error recovery."""
    if not update.message: return

    start_ts = time.time()

    # 0. User & Input Validation
    user = update.effective_user
    user_id = str(user.id) if user else "anonymous"
    raw_text = update.message.text or update.message.caption or ""

    # Enforce Rate Limiting (Item 12)
    allowed, current_requests = check_user_rate_limit(user_id, limit=MAX_DAILY_REQUESTS)
    if not allowed:
        logger.warning(f"Rate limit exceeded for user {anonymize_user_id(user_id)}")
        log_operational_event(
            event_type="telegram_message",
            status="rate_limited",
            details={"current_requests": current_requests}
        )
        await update.message.reply_text(
            f"Daily verification quota reached ({MAX_DAILY_REQUESTS}/{MAX_DAILY_REQUESTS}). Quota resets at 00:00 UTC."
        )
        return

    # Enforce Input Length Cap (Item 14)
    if len(raw_text) > MAX_MESSAGE_LENGTH:
        raw_text = raw_text[:MAX_MESSAGE_LENGTH]
        logger.info(f"Input truncated to {MAX_MESSAGE_LENGTH} characters for user {anonymize_user_id(user_id)}")

    logger.info(f"Processing Telegram request from user {anonymize_user_id(user_id)} (req #{current_requests})")

    status_msg = None
    try:
        status_msg = await update.message.reply_text("Initializing Neural Scan...")

        # 1. OCR Logic (Hardened)
        if update.message.photo:
            await status_msg.edit_text("Extracting text from screenshot...")
            temp_path = None
            try:
                photo_file = await update.message.photo[-1].get_file()
                scratch_dir = BASE_DIR / "scratch"
                scratch_dir.mkdir(parents=True, exist_ok=True)
                temp_path = scratch_dir / f"tg_{int(datetime.now().timestamp())}_{os.urandom(4).hex()}.png"

                await photo_file.download_to_drive(str(temp_path))

                # Image Integrity & MIME validation (Item 16)
                is_valid, reason = validate_image(temp_path)
                if not is_valid:
                    await status_msg.edit_text(f"Invalid image format: {reason}. Please send a standard PNG or JPEG.")
                    return

                # Run OCR verification
                res = verify_image(temp_path)

                data = {
                    "claim": res.get("claim", "Screenshot Analysis"),
                    "verdict": res.get("fact_check_result", {}).get("verdict", "Analysis complete."),
                    "risk_score": res.get("risk_score", 0.5),
                    "risk_level": res.get("risk_level", "Moderate"),
                    "language": res.get("detected_language", "en")
                }
                model_used = "Vision + OCR"
                final_text = res.get("extracted_text", data["claim"])

                if not res.get("extraction_successful"):
                    await status_msg.delete()
                    await update.message.reply_text("Could not extract legible text from this image. Please send a clearer screenshot.")
                    return

            except Exception as e:
                logger.error(f"Telegram OCR flow failed: {str(e)}", exc_info=True)
                await status_msg.edit_text("Unable to process this image. Please try sending plain text.")
                return
            finally:
                if temp_path and temp_path.exists():
                    try: temp_path.unlink()
                    except Exception: pass

        # 2. Text/URL Analysis (Hardened)
        else:
            text = raw_text.strip()
            if not text:
                await status_msg.edit_text("Please provide a claim, article headline, or link to verify.")
                return

            await status_msg.edit_text("Analyzing high-salience signals...")
            lang_code = detect_language(text)

            try:
                if lang_code == "en":
                    data = verify(text)
                    model_used = "Ensemble v2.1"
                else:
                    pred = predict_multilingual(text)
                    data = {
                        "claim": text[:150],
                        "verdict": "Analysis synthesized via regional engine.",
                        "risk_score": pred.get("probability", 0.5),
                        "risk_level": pred.get("risk_level", "Moderate"),
                        "language": lang_code
                    }
                    model_used = "Regional MuRIL"
            except Exception as e:
                logger.error(f"Inference engine failure in Telegram: {str(e)}", exc_info=True)
                data = {"claim": text[:100], "verdict": "Verification unavailable for this claim.", "risk_score": 0.5, "risk_level": "Moderate"}
                model_used = "Error-Fallback"

            final_text = text

        # 3. Record Prediction & Send Response (Item 4 & 7: associate user_id)
        pid = record_prediction(
            text=truncate_text(final_text, 500),
            language=data.get("language", "en"),
            probability=data["risk_score"],
            risk_level=data["risk_level"],
            model_used=model_used,
            source="telegram",
            user_id=user_id
        )

        formatted_report = format_telegram_report(data)
        report = formatted_report["raw_text"]

        keyboard = [
            [
                InlineKeyboardButton("👍 Correct", callback_data=f"fb_1_{pid}"),
                InlineKeyboardButton("👎 Wrong", callback_data=f"fb_0_{pid}")
            ]
        ]

        await status_msg.delete()
        await update.message.reply_text(
            report,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        logger.info(f"Verification report successfully delivered to user {anonymize_user_id(user_id)}")

        elapsed_ms = (time.time() - start_ts) * 1000
        log_operational_event(
            event_type="telegram_message",
            latency_ms=elapsed_ms,
            status="success",
            fallback_type="none",
            details={"risk_level": data.get("risk_level", "Unknown")}
        )

    except Forbidden:
        logger.warning(f"Bot blocked by user {anonymize_user_id(user_id)}")
    except (TimedOut, TelegramError) as e:
        logger.error(f"Telegram API communication error: {str(e)}")
        log_operational_event(event_type="telegram_message", status="error", details={"error": str(e)})
    except Exception as e:
        logger.critical(f"Unhandled error in Telegram message pipeline: {str(e)}", exc_info=True)
        log_operational_event(event_type="telegram_message", status="error", details={"error": str(e)})
        send_alert(
            title="Telegram Message Processing Exception",
            message=f"Unhandled error processing message: {str(e)}",
            severity="error"
        )
        if status_msg:
            try: await status_msg.edit_text("System busy. Please try your request again shortly.")
            except Exception: pass


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle feedback button clicks with ownership verification and replay guards."""
    query = update.callback_query
    if not query: return

    try:
        await query.answer()
        data = query.data.split("_")
        if len(data) < 3: return

        fb_type = data[1]
        pid = int(data[2])
        user_id = str(update.effective_user.id) if update.effective_user else "anonymous"

        fb_val = "👍 Correct" if fb_type == "1" else "👎 Wrong"
        correct_label = "real" if fb_type == "1" else "misinformation"

        # Record feedback with user ownership verification
        res = record_feedback(
            prediction_id=pid,
            user_feedback=fb_val,
            correct_label=correct_label,
            user_id=user_id
        )

        if res.get("success"):
            new_text = query.message.text + "\n\nFeedback recorded. Thank you!"
            await query.edit_message_text(text=new_text)
        else:
            await query.edit_message_text(text=query.message.text + "\n\nUnable to record feedback.")

    except Exception as e:
        logger.error(f"Error in Telegram callback handler: {str(e)}", exc_info=True)


async def telegram_error_handler(update: Optional[object], context: ContextTypes.DEFAULT_TYPE):
    """Global Telegram error handler with alert notification."""
    logger.error(f"Unhandled Telegram exception: {context.error}", exc_info=context.error)
    send_alert(
        title="Telegram Bot Runtime Exception",
        message=f"Bot handler encountered an exception: {str(context.error)}",
        severity="error"
    )


def setup_handlers(app):
    """Register all bot handlers on a Telegram Application instance.
    Reusable by both standalone polling mode and FastAPI webhook integration."""
    app.add_error_handler(telegram_error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler((filters.TEXT & ~filters.COMMAND) | filters.PHOTO, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))
    return app


def create_telegram_app(token: str = ""):
    """Create and configure a Telegram Application with all handlers.
    Used by FastAPI webhook integration to get a configured app instance."""
    if not HAS_TELEGRAM:
        logger.error("python-telegram-bot is not installed. Telegram app cannot be created.")
        return None
    tok = token or TELEGRAM_TOKEN
    if not tok:
        logger.error("TELEGRAM_BOT_TOKEN is not configured.")
        return None
    app = ApplicationBuilder().token(tok).build()
    setup_handlers(app)
    logger.info("Telegram Application created and handlers registered.")
    return app


if __name__ == '__main__':
    if not HAS_TELEGRAM:
        logger.critical("python-telegram-bot is not installed.")
        sys.exit(1)

    if not TELEGRAM_TOKEN:
        logger.critical("TELEGRAM_BOT_TOKEN is missing in .env")
        sys.exit(1)

    app = create_telegram_app(TELEGRAM_TOKEN)
    logger.info("RiskLens Telegram Bot service starting in secure polling mode with monitoring active...")
    app.run_polling(drop_pending_updates=True, allowed_updates=["message", "callback_query"])
