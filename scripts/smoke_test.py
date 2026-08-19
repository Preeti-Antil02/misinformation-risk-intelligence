"""
scripts/smoke_test.py
=====================
End-to-End Functional Smoke Test Suite for RiskLens v2.1.0 on Hugging Face Spaces.
Exercises:
1. Full LangGraph Agent Verification (/verify: claim extraction + web research + Google Fact Check + neural blend).
2. Neural 5-Model Ensemble Classification (/predict: calibrated Platt/Isotonic meta-learner).
3. Multilingual Indic Language Routing (Hindi claim verification via MuRIL).
4. Image Screenshot OCR & Extraction Pipeline (EasyOCR/Tesseract).
5. Database Persistence & Health Configuration (/health & /analytics).
6. Streamlit Dashboard Web Interface Availability.
"""

import os
import sys
import io
import time
import json
import argparse
import requests
from datetime import datetime
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_SPACE_URL = "https://preeti-antil-risklens.hf.space"


def test_full_agent_verification(base_url: str) -> dict:
    """Test 1: Tests the real LangGraph Verification Agent (exact path used by Telegram Bot)."""
    url = f"{base_url.rstrip('/')}/verify"
    test_claim = "World Health Organization declares global health emergency following new virus strain."
    start_t = time.time()
    try:
        resp = requests.post(url, json={"text": test_claim}, timeout=45)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)

        if resp.status_code != 200:
            return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"HTTP {resp.status_code}: {resp.text[:120]}"}

        data = resp.json()
        claim = data.get("claim", "")
        verdict = data.get("verdict", "")
        risk_score = data.get("risk_score")
        risk_level = data.get("risk_level")
        sources = data.get("sources", [])

        valid = (
            risk_score is not None and 0.0 <= risk_score <= 1.0 and
            risk_level in ["Low", "Medium", "High", "Critical", "Moderate"] and
            len(verdict) > 0
        )

        if valid:
            return {
                "pass": True,
                "elapsed_ms": elapsed_ms,
                "risk_level": risk_level,
                "risk_score": round(risk_score, 3),
                "verdict_snippet": verdict[:60] + "...",
                "sources_count": len(sources)
            }
        return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"Invalid verification schema: {data}"}
    except Exception as e:
        return {"pass": False, "elapsed_ms": round((time.time() - start_t) * 1000, 1), "error": str(e)}


def test_neural_ensemble_prediction(base_url: str) -> dict:
    """Test 2: Tests fast 5-model neural ensemble classification (/predict)."""
    url = f"{base_url.rstrip('/')}/predict"
    test_claim = "Scientists develop new malaria vaccine showing 90% efficacy in phase 3 clinical trials."
    start_t = time.time()
    try:
        resp = requests.post(url, json={"text": test_claim}, timeout=60)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)

        if resp.status_code != 200:
            return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"HTTP {resp.status_code}: {resp.text[:100]}"}

        data = resp.json()
        ens = data.get("ensemble", {})
        prob = ens.get("probability_fake")
        risk = ens.get("risk_level")

        valid_schema = (
            prob is not None and 0.0 <= prob <= 1.0 and
            risk in ["Low", "Medium", "High", "Critical"] and
            "roberta" in data and "xgboost" in data and "logistic_regression" in data
        )

        if valid_schema:
            return {
                "pass": True,
                "elapsed_ms": elapsed_ms,
                "risk_level": risk,
                "probability": round(prob, 3),
                "is_calibrated": ens.get("is_calibrated", False)
            }
        return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"Invalid response schema: {data}"}
    except Exception as e:
        return {"pass": False, "elapsed_ms": round((time.time() - start_t) * 1000, 1), "error": str(e)}


def test_multilingual_indic_claim(base_url: str) -> dict:
    """Test 3: Tests Hindi multilingual text routing."""
    hindi_claim = "नासा ने चंद्रमा के दक्षिणी ध्रुव पर विशाल जल स्रोतों की खोज की पुष्टि की है।"
    url = f"{base_url.rstrip('/')}/predict"
    start_t = time.time()
    try:
        resp = requests.post(url, json={"text": hindi_claim}, timeout=60)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)

        if resp.status_code == 200:
            data = resp.json()
            risk = data.get("ensemble", {}).get("risk_level", "Unknown")
            return {"pass": True, "elapsed_ms": elapsed_ms, "risk_level": risk}
        return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"HTTP {resp.status_code}: {resp.text[:100]}"}
    except Exception as e:
        return {"pass": False, "elapsed_ms": round((time.time() - start_t) * 1000, 1), "error": str(e)}


def test_ocr_pipeline_execution() -> dict:
    """Test 4: Tests dual-engine OCR pipeline locally / within container."""
    start_t = time.time()
    temp_img_path = BASE_DIR / "scratch" / "temp_smoke_ocr.png"
    try:
        from PIL import Image, ImageDraw
        from risklens.ocr_pipeline import extract_text_from_image

        temp_img_path.parent.mkdir(parents=True, exist_ok=True)

        # Create synthetic test image with crisp rendered text
        img = Image.new("RGB", (600, 150), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((20, 50), "BREAKING: Space Agency Announces Discovery", fill=(0, 0, 0))
        img.save(str(temp_img_path), format="PNG")

        # Run OCR extraction with file path
        extracted = extract_text_from_image(temp_img_path)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)

        text = extracted.get("raw_text", extracted.get("text", "")).strip()
        engine = extracted.get("engine_used", extracted.get("engine", "unknown"))

        if len(text) > 3:
            return {"pass": True, "elapsed_ms": elapsed_ms, "engine": engine, "extracted_text": text[:40]}

        return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"OCR produced empty text. Result: {extracted}"}
    except Exception as e:
        return {"pass": False, "elapsed_ms": round((time.time() - start_t) * 1000, 1), "error": str(e)}
    finally:
        if temp_img_path.exists():
            try: temp_img_path.unlink()
            except Exception: pass


def test_database_and_storage(base_url: str, api_key: str = "") -> dict:
    """Test 5: Verifies database persistence and /health storage mode."""
    start_t = time.time()
    try:
        # Check /health storage status
        h_resp = requests.get(f"{base_url.rstrip('/')}/health", timeout=15)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)

        if h_resp.status_code != 200:
            return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"/health returned HTTP {h_resp.status_code}"}

        h_data = h_resp.json()
        storage = h_data.get("components", {}).get("storage", {})
        storage_mode = storage.get("mode", "unknown")
        storage_path = storage.get("path", "unknown")
        is_persistent = storage.get("is_persistent", False)

        # If API key is provided, also test /analytics route
        analytics_ok = None
        if api_key:
            headers = {"X-API-Key": api_key}
            a_resp = requests.get(f"{base_url.rstrip('/')}/analytics", headers=headers, timeout=10)
            analytics_ok = (a_resp.status_code == 200)

        return {
            "pass": True,
            "elapsed_ms": elapsed_ms,
            "storage_mode": storage_mode,
            "storage_path": storage_path,
            "is_persistent": is_persistent,
            "analytics_verified": analytics_ok
        }
    except Exception as e:
        return {"pass": False, "elapsed_ms": round((time.time() - start_t) * 1000, 1), "error": str(e)}


def test_streamlit_dashboard(base_url: str) -> dict:
    """Test 6: Verifies Streamlit UI availability."""
    start_t = time.time()
    try:
        resp = requests.get(base_url.rstrip('/'), timeout=15)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)

        if resp.status_code == 200:
            return {"pass": True, "elapsed_ms": elapsed_ms, "status_code": 200}
        return {"pass": False, "elapsed_ms": elapsed_ms, "error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"pass": False, "elapsed_ms": round((time.time() - start_t) * 1000, 1), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="RiskLens End-to-End Functional Smoke Test Suite")
    parser.add_argument("--url", default=os.getenv("SPACE_URL", DEFAULT_SPACE_URL), help="Target Space URL")
    parser.add_argument("--api-key", default=os.getenv("RISKLENS_API_KEY", ""), help="Admin API Key for protected checks")
    args = parser.parse_args()

    space_url = args.url.rstrip("/")

    print("=" * 80)
    print("🧪 RiskLens v2.1.0 End-to-End Functional Smoke Test Suite")
    print(f"🎯 Target Endpoint : {space_url}")
    print(f"🕒 Timestamp       : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 80)

    results = []

    # 1. Full LangGraph Agent Verification (/verify)
    print("\n[1/6] 🤖 Testing Full LangGraph Agent Verification (/verify)...")
    res1 = test_full_agent_verification(space_url)
    results.append(("Full LangGraph Verification Agent", res1))
    if res1["pass"]:
        print(f"  ✅ PASS ({res1['elapsed_ms']} ms) — Verdict: {res1.get('risk_level')} (Score: {res1.get('risk_score')}) | Sources: {res1.get('sources_count')}")
        print(f"     • Synthesis: \"{res1.get('verdict_snippet')}\"")
    else:
        print(f"  ❌ FAIL ({res1['elapsed_ms']} ms) — {res1.get('error')}")

    # 2. Fast Neural Ensemble Prediction (/predict)
    print("\n[2/6] 🧠 Testing Calibrated Neural Ensemble Classification (/predict)...")
    res2 = test_neural_ensemble_prediction(space_url)
    results.append(("Calibrated Neural Ensemble", res2))
    if res2["pass"]:
        print(f"  ✅ PASS ({res2['elapsed_ms']} ms) — Verdict: {res2.get('risk_level')} (Prob: {res2.get('probability')}) [Calibrated: {res2.get('is_calibrated')}]")
    else:
        print(f"  ❌ FAIL ({res2['elapsed_ms']} ms) — {res2.get('error')}")

    # 3. Multilingual Indic Claim Verification
    print("\n[3/6] 🌐 Testing Multilingual Indic Claim Processing (Hindi)...")
    res3 = test_multilingual_indic_claim(space_url)
    results.append(("Multilingual Indic Routing", res3))
    if res3["pass"]:
        print(f"  ✅ PASS ({res3['elapsed_ms']} ms) — Verdict: {res3.get('risk_level')}")
    else:
        print(f"  ❌ FAIL ({res3['elapsed_ms']} ms) — {res3.get('error')}")

    # 4. OCR Pipeline Execution
    print("\n[4/6] 🖼️  Testing Dual-Engine OCR Text Extraction Pipeline...")
    res4 = test_ocr_pipeline_execution()
    results.append(("Dual-Engine OCR Extraction", res4))
    if res4["pass"]:
        print(f"  ✅ PASS ({res4['elapsed_ms']} ms) — Engine: '{res4.get('engine')}' | Extracted: \"{res4.get('extracted_text')}...\"")
    else:
        print(f"  ❌ FAIL ({res4['elapsed_ms']} ms) — {res4.get('error')}")

    # 5. Database Persistence & Health Probe
    print("\n[5/6] 💾 Testing Database Storage & Persistence Configuration (/health)...")
    res5 = test_database_and_storage(space_url, api_key=args.api_key)
    results.append(("Database & Storage Persistence", res5))
    if res5["pass"]:
        persist_str = "PERSISTENT (/data)" if res5.get("is_persistent") else "EPHEMERAL (/app/databases)"
        print(f"  ✅ PASS ({res5['elapsed_ms']} ms) — Storage Mode: {persist_str} at {res5.get('storage_path')}")
        if res5.get("analytics_verified") is not None:
            print(f"     • Protected /analytics verification: {'PASSED' if res5['analytics_verified'] else 'FAILED'}")
    else:
        print(f"  ❌ FAIL ({res5['elapsed_ms']} ms) — {res5.get('error')}")

    # 6. Streamlit Dashboard Availability
    print("\n[6/6] 🖥️  Testing Streamlit Dashboard Web Interface Availability...")
    res6 = test_streamlit_dashboard(space_url)
    results.append(("Streamlit Dashboard Web UI", res6))
    if res6["pass"]:
        print(f"  ✅ PASS ({res6['elapsed_ms']} ms) — Streamlit UI responded successfully (HTTP 200)")
    else:
        print(f"  ❌ FAIL ({res6['elapsed_ms']} ms) — {res6.get('error')}")

    # Final Summary Table
    print("\n" + "=" * 80)
    print("📊 FUNCTIONAL SMOKE TEST MATRIX")
    print("=" * 80)
    print(f"{'Test Feature':<35} | {'Status':<8} | {'Latency':<10} | {'Details'}")
    print("-" * 80)

    all_passed = True
    for name, r in results:
        status_icon = "✅ PASS" if r["pass"] else "❌ FAIL"
        if not r["pass"]:
            all_passed = False
        lat_str = f"{r.get('elapsed_ms', 0)} ms"
        detail = r.get("error", "OK") if not r["pass"] else f"Verdict: {r.get('risk_level', 'OK')}"
        print(f"{name:<35} | {status_icon:<8} | {lat_str:<10} | {detail}")

    print("=" * 80)
    if all_passed:
        print("🎉 ALL 6 END-TO-END SMOKE TESTS PASSED!")
    else:
        print("⚠️  ONE OR MORE SMOKE TESTS FAILED.")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
