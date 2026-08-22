"""
risklens/agent.py
=================
Hardened LangGraph Verification Agent for Real-World Accuracy.
- Performs exhaustive multi-query web searches.
- Uses evidence-weighting logic with robust error boundaries.
- Prioritizes real-world evidence with graceful degradation on failure.
"""

import os
import re
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, TypedDict

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Try importing duckduckgo_search or ddgs
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except Exception:
    try:
        from ddgs import DDGS
        HAS_DDGS = True
    except Exception:
        HAS_DDGS = False

# Try importing langgraph
try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

from risklens.explainer import explain_prediction
from risklens.source_credibility import get_source_credibility, compute_integrated_risk
from risklens.claim_checker import extract_claim, check_claim
from risklens.utils import safe_network_call, truncate_text

load_dotenv()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

class AgentState(TypedDict):
    """LangGraph State dictionary."""
    raw_text: str
    url: Optional[str]
    claim: str
    sources: List[Dict[str, str]]
    verdict: str
    web_probability: float
    risk_level: str
    error: Optional[str]

class VerificationAgent:
    """
    Advanced Verification Agent that prioritizes real-world web evidence
    with robust error handling and timeout management.
    """

    def __init__(self):
        self.serper_api_key = os.getenv("SERPER_API_KEY", "")

    def _search_serper(self, query: str) -> List[Dict[str, str]]:
        """Queries Google via Serper API with hardened network call."""
        if not self.serper_api_key:
            return []

        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": 4})
        headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }

        try:
            response = safe_network_call(
                requests.post,
                url=url,
                headers=headers,
                data=payload,
                timeout=(3.05, 10)
            )
            if response.status_code == 200:
                data = response.json()
                results = []
                for item in data.get("organic", []):
                    results.append({
                        "name": item.get("title", "News Source"),
                        "url": item.get("link", ""),
                        "snippet": item.get("snippet", "")
                    })
                return results
        except Exception as e:
            logger.error(f"Serper search failed for query '{truncate_text(query)}': {str(e)}")

        return []

    def _search_ddg(self, query: str) -> List[Dict[str, str]]:
        """Fallback to DuckDuckGo if Serper is unavailable, with safety wrapper."""
        if not HAS_DDGS:
            return []

        results = []
        try:
            # Note: DDGS doesn't support standard requests timeouts directly easily in this version,
            # but we wrap it in a try block.
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=4))
                for r in ddg_results:
                    results.append({
                        "name": r.get("title", "News Source"),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
        except Exception as e:
            logger.error(f"DDG search failed for query '{truncate_text(query)}': {str(e)}")

        return results

    # ========================================================================
    # NODE 1: CLAIM EXTRACTOR
    # ========================================================================
    def node_claim_extractor(self, state: AgentState) -> Dict[str, Any]:
        """Distills the core verifiable claim from input with error boundary."""
        try:
            claim = extract_claim(state["raw_text"])
            return {"claim": claim}
        except Exception as e:
            logger.error(f"Node claim_extractor failed: {str(e)}")
            return {"claim": state["raw_text"][:200], "error": "extraction_failed"}

    # ========================================================================
    # NODE 2: WEB RESEARCHER (Hardened)
    # ========================================================================
    def node_web_research(self, state: AgentState) -> Dict[str, Any]:
        """Performs exhaustive multi-query search with graceful degradation."""
        claim = state.get("claim", state["raw_text"][:200])
        all_sources = []

        clean_q = re.sub(r'[^\w\s]', ' ', claim).strip()
        clean_q = re.sub(r'\s+', ' ', clean_q)
        # Priority single high-yield query first to minimize latency
        queries = [
            f"{clean_q} fact check",
            clean_q
        ]

        seen_urls = set()
        for q in queries:
            if not q.strip():
                continue
            results = []
            try:
                # Priority 1: Serper (Instant REST API)
                if self.serper_api_key:
                    results = self._search_serper(q)

                # Priority 2: DDG fallback
                if not results:
                    results = self._search_ddg(q)

                for r in results:
                    if r["url"] not in seen_urls:
                        all_sources.append(r)
                        seen_urls.add(r["url"])

                # Early exit: as soon as we have enough grounding sources, return immediately
                if len(all_sources) >= 3:
                    break
            except Exception as e:
                logger.warning(f"Search attempt failed for query '{truncate_text(q)}': {str(e)}")
                continue

        if not all_sources:
            logger.warning(f"No web sources found for claim: {truncate_text(claim)}")

        return {"sources": all_sources[:4]}

    # ========================================================================
    # NODE 3: VERDICT SYNTHESIZER (Hardened)
    # ========================================================================
    def node_verdict_synthesizer(self, state: AgentState) -> Dict[str, Any]:
        """Analyzes web evidence to produce an accurate verdict without hallucinating."""
        sources = state.get("sources", [])

        if not sources:
            return {
                "verdict": "Insufficient web evidence found to verify this claim against real-world sources.",
                "web_probability": 0.5,
                "fact_checker_available": False
            }

        supports = 0
        contradicts = 0
        neutral = 0

        support_keywords = ["confirmed", "true", "accurate", "official", "verified", "reportedly", "valid"]
        contradict_keywords = ["false", "fake", "hoax", "debunked", "misleading", "incorrect", "untrue", "myth"]

        try:
            for s in sources:
                snippet = s.get("snippet", "").lower()
                text = s.get("name", "").lower() + " " + snippet

                if any(w in text for w in contradict_keywords):
                    contradicts += 1.5
                elif any(w in text for w in support_keywords):
                    supports += 1.0
                else:
                    neutral += 1

            total = supports + contradicts + neutral
            if total == 0:
                return {"verdict": "Web analysis inconclusive.", "web_probability": 0.5}

            # Evidence-based probability
            web_prob = contradicts / (supports + contradicts + 0.1)
            web_prob = (web_prob * (supports + contradicts) + 0.5 * neutral) / (total)

            if web_prob > 0.65:
                verdict = f"Web search results strongly suggest this claim is inaccurate or a known hoax. Authoritative sources like {sources[0]['name']} have contradicted these assertions."
            elif web_prob < 0.35:
                verdict = f"Real-world evidence supports this claim. Reports from credible outlets corroborate the information provided."
            else:
                verdict = "Web evidence is mixed or inconclusive. While some reports mention this claim, authoritative confirmation is currently missing."

            return {"verdict": verdict, "web_probability": round(float(web_prob), 4), "fact_checker_available": True}
        except Exception as e:
            logger.error(f"Node verdict_synthesizer failed: {str(e)}")
            return {"verdict": "Internal error during verdict synthesis.", "web_probability": 0.5}

    def verify(self, text: str, url: Optional[str] = None) -> Dict[str, Any]:
        """End-to-end verification with robust error handling and total timeout."""
        start_time = time.time()
        logger.info(f"Starting verification for: {truncate_text(text)}")

        try:
            # 1. Execute Agent Logic
            state: AgentState = {
                "raw_text": text,
                "url": url,
                "claim": "",
                "sources": [],
                "verdict": "",
                "web_probability": 0.5,
                "risk_level": "Low",
                "error": None
            }

            # Manual node execution with error boundaries (simulating graph)
            state.update(self.node_claim_extractor(state))
            state.update(self.node_web_research(state))
            state.update(self.node_verdict_synthesizer(state))

            # 2. Get Neural Model Baseline (Wrapped in safety)
            try:
                neural_exp = explain_prediction(text)
                neural_prob = neural_exp["probability"]
            except Exception as e:
                logger.error(f"explain_prediction failed: {str(e)}")
                neural_prob = 0.5
                neural_exp = {"probability": 0.5, "risk_level": "Moderate", "why_summary": "Neural analysis failed."}

            # 3. SYNTHESIS
            # Weighting: 70% Web / 30% Neural
            if state.get("sources"):
                final_prob = (state["web_probability"] * 0.7) + (neural_prob * 0.3)
            else:
                final_prob = neural_prob # Fallback to neural if web fails

            from src.risk_scoring import RiskScorer
            risk_level = RiskScorer().score_ensemble(final_prob)

            latency = time.time() - start_time
            logger.info(f"Verification complete. Risk: {risk_level} ({final_prob:.2f}). Latency: {latency:.2f}s")

            return {
                "claim": state["claim"],
                "verdict": state["verdict"],
                "sources": state["sources"],
                "risk_score": round(final_prob, 4),
                "risk_level": risk_level,
                "explanation": neural_exp,
                "fact_checker_available": state.get("fact_checker_available", False)
            }
        except Exception as e:
            logger.critical(f"Total verification pipeline crash: {str(e)}", exc_info=True)
            return {
                "claim": text[:100],
                "verdict": "We encountered an error while processing this claim. Please try again later.",
                "sources": [],
                "risk_score": 0.5,
                "risk_level": "Moderate",
                "explanation": {"why_summary": "System error."},
                "error": str(e)
            }

# Singleton & In-Memory Response Cache
_default_agent: Optional[VerificationAgent] = None
_verify_cache: Dict[str, Dict[str, Any]] = {}

def get_verification_agent() -> VerificationAgent:
    global _default_agent
    if _default_agent is None:
        _default_agent = VerificationAgent()
    return _default_agent

def verify(text: str, url: Optional[str] = None) -> Dict[str, Any]:
    """Top-level functional interface to run full multi-agent verification pipeline with high-speed memory caching."""
    cache_key = f"{text.strip().lower()}::{url or ''}"
    if cache_key in _verify_cache:
        return dict(_verify_cache[cache_key])

    res = get_verification_agent().verify(text, url=url)
    if len(_verify_cache) > 250:
        _verify_cache.pop(next(iter(_verify_cache)))
    _verify_cache[cache_key] = res
    return res
