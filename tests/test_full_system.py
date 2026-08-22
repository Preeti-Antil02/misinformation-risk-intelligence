"""
Comprehensive End-to-End System Audit for RiskLens v2.1.0.
Tests all backend modules, models, pipelines, databases, and APIs.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np


def test_1_models_and_feature_pipeline():
    print("\n--- TEST 1: Model & Feature Pipeline ---")
    import joblib
    from src.features.text_preprocessor import TextPreprocessor
    from src.features.feature_builder import FeatureBuilder
    from src.risk_scoring import RiskScorer
    from scipy.sparse import hstack, csr_matrix
    import pandas as pd

    models_dir = BASE_DIR / "models"
    lr = joblib.load(models_dir / "baseline_logistic.pkl")
    xgb = joblib.load(models_dir / "xgboost_model.pkl")
    tfidf = joblib.load(models_dir / "tfidf_vectorizer.pkl")
    scaler = joblib.load(models_dir / "numeric_scaler.pkl")

    tp = TextPreprocessor()
    fb = FeatureBuilder()
    rs = RiskScorer()

    sample = "SHOCKING: Secret cure for virus exposed by whistleblower!"
    cleaned = tp.truncate(tp.basic_clean(sample))
    X_tfidf = tfidf.transform([cleaned])
    X_num = fb.build_features(pd.DataFrame({"text": [sample]}))
    X_num_s = scaler.transform(X_num.values)
    X_comb = hstack([X_tfidf, csr_matrix(X_num_s)])

    p_lr = float(lr.predict_proba(X_tfidf)[0, 1])
    p_xgb = float(xgb.predict_proba(X_comb)[0, 1])

    risk_lr = rs.score(p_lr)
    risk_xgb = rs.score(p_xgb)

    print(f"Sample: '{sample}'")
    print(f"LR Prob: {p_lr:.4f} ({risk_lr}), XGB Prob: {p_xgb:.4f} ({risk_xgb})")
    assert 0.0 <= p_lr <= 1.0
    assert 0.0 <= p_xgb <= 1.0
    assert risk_lr in ["Low", "Moderate", "High", "Critical"]


def test_2_explainability_pipeline():
    print("\n--- TEST 2: Explainability Pipeline (SHAP) ---")
    from risklens.explainer import explain_prediction

    sample = "URGENT BOMBSHELL: Government hiding secret conspiracy guaranteed!"
    res = explain_prediction(sample)
    print("Probability:", res.get("probability"))
    print("Risk Level:", res.get("risk_level"))
    print("Top Features:", res.get("top_features"))
    assert "probability" in res
    assert "top_features" in res
    assert len(res.get("top_features", [])) > 0


def test_3_langgraph_agent_verification():
    print("\n--- TEST 3: Multi-Agent LangGraph Verification ---")
    from risklens.agent import verify

    sample = "COVID-19 vaccine contains microchips to track citizens worldwide."
    res = verify(sample)
    print("Claim:", res.get("claim"))
    print("Verdict:", res.get("verdict")[:80] + "...")
    print("Risk Score:", res.get("risk_score"))
    print("Risk Level:", res.get("risk_level"))
    print("Sources Count:", len(res.get("sources", [])))
    assert "claim" in res
    assert "verdict" in res
    assert "risk_level" in res


def test_4_multilingual_indic_engine():
    print("\n--- TEST 4: Multilingual & Indic Script Detection ---")
    from risklens.multilingual import detect_language, predict_multilingual

    samples = {
        "en": "Official election results announced by the electoral commission.",
        "hi": "भारत सरकार ने सभी नागरिकों के लिए नई योजना की घोषणा की।",
        "mr": "मुंबई आणि पुण्यातील नागरिकांसाठी महत्त्वाची बातमी समोर आली आहे.",
        "bn": "পশ্চিমবঙ্গের স্বাস্থ্য দফতর নতুন নির্দেশিকা জারি করেছে।"
    }

    for expected_lang, text in samples.items():
        detected = detect_language(text)
        print(f"Expected: {expected_lang} | Detected: {detected} | Text: {text[:35]}...")
        pred = predict_multilingual(text)
        assert "probability" in pred
        assert "risk_level" in pred


def test_5_url_reader_and_domain_credibility():
    print("\n--- TEST 5: Deep URL Intelligence & Domain Credibility ---")
    from risklens.source_credibility import get_source_credibility, compute_integrated_risk
    from risklens.url_reader import DeepURLReader

    # Test credibility lookup
    reuters_cred = get_source_credibility("https://www.reuters.com/article/technology-news")
    print(f"Reuters Credibility: {reuters_cred}")
    assert reuters_cred["credibility_score"] >= 0.85
    assert "credibility_tier" in reuters_cred

    fake_cred = get_source_credibility("https://www.infowars.com/breaking-news")
    print(f"Flagged Site Credibility: {fake_cred}")
    assert fake_cred["credibility_score"] <= 0.40

    integrated = compute_integrated_risk(0.80, reuters_cred["credibility_score"])
    print(f"Integrated Risk on High Authority Domain: {integrated:.4f}")
    assert integrated < 0.80 # Credible domain mitigates overall content risk


def test_6_conformal_uncertainty_quantification():
    print("\n--- TEST 6: Split Conformal Prediction ---")
    from risklens.conformal_predictor import ConformalPredictor

    cp = ConformalPredictor(alpha=0.10)
    res = cp.predict_set("Breaking news: Economy grows 3% this quarter.")
    print("Conformal Result:", res)
    assert "prediction_set" in res
    assert "confidence_guarantee" in res
    assert "is_ambiguous" in res


def test_7_feedback_and_telemetry_db():
    print("\n--- TEST 7: Feedback & Telemetry Database ---")
    from risklens.feedback import record_prediction, record_feedback, calculate_live_accuracy, get_recent_feedback

    pid = record_prediction(
        text="Audit test claim for pipeline verification",
        language="en",
        probability=0.75,
        risk_level="High",
        model_used="Audit Tester",
        source="system_audit",
        user_id="audit_tester_001"
    )
    assert pid > 0

    fb_res = record_feedback(
        prediction_id=pid,
        user_feedback="✅ Correct",
        correct_label="real",
        user_id="audit_tester_001"
    )
    assert fb_res.get("success") is True

    acc = calculate_live_accuracy()
    print("Live Accuracy Data:", acc)
    assert "total_predictions" in acc


def test_8_telegram_shared_formatter_and_webhook():
    print("\n--- TEST 8: Telegram Formatter & Webhook Direct Answer ---")
    from risklens.telegram_bot import format_telegram_report

    data = {
        "claim": "Test claim about climate change",
        "verdict": "Verified by multiple climate research agencies.",
        "risk_score": 0.12,
        "risk_level": "Low"
    }
    rep = format_telegram_report(data)
    print("Telegram Report Output:\n" + rep["raw_text"])
    assert "Misinformation probability: 12%" in rep["raw_text"]
    assert "Confidence:" not in rep["raw_text"]


def test_9_fastapi_endpoints():
    print("\n--- TEST 9: FastAPI Endpoints ---")
    from fastapi.testclient import TestClient
    from api import app

    client = TestClient(app)

    # 1. Health Probe
    resp_health = client.get("/health")
    print("Health Status Code:", resp_health.status_code)
    print("Health Payload:", resp_health.json())
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "healthy"

    # 2. Predict Endpoint
    resp_pred = client.post("/predict", json={"text": "Global leaders meet at international summit to discuss trade."})
    print("Predict Status Code:", resp_pred.status_code)
    assert resp_pred.status_code == 200
    pred_data = resp_pred.json()
    assert "ensemble" in pred_data
    assert "timing_breakdown" in pred_data

    # 3. Full Verify Endpoint
    resp_verify = client.post("/verify", json={"text": "Drinking hot lemon water cures 100 percent of cancer cases."})
    print("Verify Status Code:", resp_verify.status_code)
    assert resp_verify.status_code == 200
    verify_data = resp_verify.json()
    assert "verdict" in verify_data
    assert "risk_level" in verify_data

    # 4. Telegram Webhook Endpoint
    resp_webhook = client.post("/telegram/webhook", json={
        "message": {
            "chat": {"id": 999888777},
            "text": "Breaking news about economy",
            "from": {"id": 11223344}
        }
    })
    print("Webhook Status Code:", resp_webhook.status_code)
    assert resp_webhook.status_code == 200
    wh_data = resp_webhook.json()
    print("Webhook Direct Response Method:", wh_data.get("method"))
    assert wh_data.get("method") == "sendMessage"
    assert "reply_markup" in wh_data


if __name__ == "__main__":
    test_1_models_and_feature_pipeline()
    test_2_explainability_pipeline()
    test_3_langgraph_agent_verification()
    test_4_multilingual_indic_engine()
    test_5_url_reader_and_domain_credibility()
    test_6_conformal_uncertainty_quantification()
    test_7_feedback_and_telemetry_db()
    test_8_telegram_shared_formatter_and_webhook()
    test_9_fastapi_endpoints()
    print("\n=======================================================")
    print(">>> ALL 9 END-TO-END SUBSYSTEM TESTS PASSED WITH 100% SUCCESS <<<")
    print("=======================================================")
