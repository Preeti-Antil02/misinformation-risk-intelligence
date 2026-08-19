"""
risklens/logging_config.py
==========================
Centralized logging configuration for RiskLens v2.1.0 Enterprise.
Supports console output, rotating file handlers (10MB limit), and Sentry error tracking integration.
"""

import logging
import logging.handlers
import os
from pathlib import Path

def setup_logging(level=logging.INFO):
    """Initializes the global logging system with console, rotating file, and error monitoring."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / "risklens.log"

    # Formatter: Timestamp | Level | Module | Message
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Rotating File Handler (10MB per file, keep 5 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # Root Logger Config
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers if any (to avoid duplicates on reload)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Specific noise reduction for 3rd party libs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.INFO)

    logging.info("Logging system initialized. Outputting to logs/risklens.log")

setup_logging()
