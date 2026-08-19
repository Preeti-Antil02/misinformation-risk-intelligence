"""
risklens/utils.py
=================
Production-grade utility functions for hardening.
Includes retry logic, timeouts, and network resilience helpers.
"""

import time
import random
import logging
import requests
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

def safe_network_call(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    timeout: tuple = (3.05, 15), # (connect, read)
    *args,
    **kwargs
) -> Any:
    """
    Executes a network call with exponential backoff, jitter, and timeouts.

    Parameters:
    -----------
    func : callable
        The function to execute (e.g., requests.get).
    max_retries : int
        Maximum number of retry attempts for transient errors.
    base_delay : float
        Starting delay for backoff in seconds.
    timeout : tuple
        Connection and read timeout.
    """
    last_exception = None

    # Inject timeout into kwargs if not present
    if 'timeout' not in kwargs:
        kwargs['timeout'] = timeout

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            # Don't retry on 4xx Client Errors (except 429)
            status_code = e.response.status_code if e.response is not None else 0
            if 400 <= status_code < 500 and status_code != 429:
                logger.error(f"Network call failed with fatal client error {status_code}. No retry.")
                raise
            last_exception = e
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            last_exception = e

        # If we reach here, a transient error occurred
        if attempt < max_retries - 1:
            # Exponential backoff with jitter
            delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
            logger.warning(f"Transient error in network call (Attempt {attempt+1}/{max_retries}). Retrying in {delay:.2f}s... Error: {str(last_exception)}")
            time.sleep(delay)

    logger.error(f"Network call failed after {max_retries} attempts.")
    raise last_exception

def truncate_text(text: str, limit: int = 100) -> str:
    """Safely truncates text for logging purposes to avoid PII leak."""
    if not text: return ""
    return (text[:limit] + "...") if len(text) > limit else text
