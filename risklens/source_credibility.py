"""
risklens/source_credibility.py
==============================
Hardened Source Credibility Scoring.
- Manages domain reputation database with loading guards.
- Exception-safe domain extraction and score retrieval.
- Integrated risk calculation with boundary safety.
"""

import os
import re
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List, Tuple

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Credibility Label to Score Mapping
CREDIBILITY_MAPPING = {
    "very high": 0.95,
    "high": 0.80,
    "mostly high": 0.80,
    "mixed": 0.50,
    "medium": 0.50,
    "low": 0.25,
    "very low": 0.05,
    "satire": 0.10,
    "conspiracy": 0.05,
    "fake news": 0.05
}

class SourceCredibilityEngine:
    """
    Domain credibility assessment engine with loading safeguards.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DATA_DIR / "domain_credibility.csv")
        self.domains_df = self._load_database()

    def _load_database(self) -> pd.DataFrame:
        """Loads domain credibility database with exception safety and fallback."""
        try:
            if self.db_path.exists():
                df = pd.read_csv(self.db_path)
                df["domain"] = df["domain"].str.lower().str.strip()
                logger.info(f"Loaded {len(df)} domain records from {self.db_path.name}")
                return df
        except Exception as e:
            logger.error(f"Failed to load domain database from {self.db_path}: {str(e)}")

        # Hardcoded High-Authority Fallback (Static Ground Truth)
        records = [
            {"domain": "reuters.com", "credibility_score": 0.95, "bias_label": "Least Biased", "category": "News"},
            {"domain": "apnews.com", "credibility_score": 0.95, "bias_label": "Least Biased", "category": "News"},
            {"domain": "bbc.com", "credibility_score": 0.95, "bias_label": "Left-Center", "category": "News"},
            {"domain": "who.int", "credibility_score": 0.95, "bias_label": "Least Biased", "category": "Health/Science"},
            {"domain": "snopes.com", "credibility_score": 0.95, "bias_label": "Least Biased", "category": "Fact Check"},
            {"domain": "theonion.com", "credibility_score": 0.10, "bias_label": "Satire", "category": "Satire"},
            {"domain": "infowars.com", "credibility_score": 0.05, "bias_label": "Conspiracy", "category": "Fake News"},
        ]
        logger.warning("Using hardcoded domain fallback records.")
        return pd.DataFrame(records)

    @staticmethod
    def extract_domain(url_or_domain: str) -> str:
        """Extracts clean registered hostname with parsing safety."""
        if not url_or_domain: return "unknown.com"
        target = url_or_domain.strip().lower()
        if not target.startswith(("http://", "https://")):
            target = "https://" + target

        try:
            parsed = urlparse(target)
            domain = parsed.netloc or parsed.path
            domain = domain.split(":")[0]
            if domain.startswith("www."):
                domain = domain[4:]
            return domain if domain else "unknown.com"
        except Exception:
            # Fallback regex parsing
            match = re.search(r'(?:https?://)?(?:www\.)?([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', url_or_domain)
            return match.group(1).lower() if match else "unknown.com"

    def get_source_credibility(self, url: str) -> Dict[str, Any]:
        """Retrieves domain data with total lookup safety."""
        try:
            domain = self.extract_domain(url)

            # 1. Exact Match
            match = self.domains_df[self.domains_df["domain"] == domain]

            # 2. Parent Domain Match (subdomain safety)
            if match.empty and "." in domain:
                parts = domain.split(".")
                if len(parts) > 2:
                    parent = ".".join(parts[-2:])
                    match = self.domains_df[self.domains_df["domain"] == parent]

            if not match.empty:
                row = match.iloc[0]
                return {
                    "domain": domain,
                    "credibility_score": round(float(row["credibility_score"]), 4),
                    "bias_label": str(row["bias_label"]),
                    "category": str(row.get("category", "General")),
                    "found_in_db": True
                }
        except Exception as e:
            logger.error(f"Source credibility lookup failed for {url}: {str(e)}")

        return {
            "domain": "unknown",
            "credibility_score": 0.50,
            "bias_label": "Unrated",
            "category": "Unknown",
            "found_in_db": False
        }

    def compute_integrated_risk(
        self,
        ensemble_probability: float,
        credibility_score: float,
        alpha: float = 0.70
    ) -> float:
        """Combines scores with boundary clamping."""
        p = max(0.0, min(1.0, float(ensemble_probability)))
        c = max(0.0, min(1.0, float(credibility_score)))
        final = (alpha * p) + ((1.0 - alpha) * (1.0 - c))
        return round(float(final), 4)

# Singleton
_default_engine: Optional[SourceCredibilityEngine] = None

def get_credibility_engine() -> SourceCredibilityEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = SourceCredibilityEngine()
    return _default_engine

def get_source_credibility(url: str) -> Dict[str, Any]:
    return get_credibility_engine().get_source_credibility(url)

def compute_integrated_risk(ensemble_probability: float, credibility_score: float) -> float:
    return get_credibility_engine().compute_integrated_risk(ensemble_probability, credibility_score)
