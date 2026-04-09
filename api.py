"""
Misinformation Risk Intelligence System — API Layer
FastAPI inference endpoint

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Example:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{"text": "Breaking news: shocking discovery doctors hate!"}'
"""

import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.roberta_model import RobertaClassifier
from src.models.slm_model import QwenClassifier
from src.risk_scoring import RiskScorer

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

tp = TextPreprocessor()
fb = FeatureBuilder()
rs = RiskScorer()

# -------------------------------------------------------
# Schemas
# -------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=10, description="News article or claim to analyse")


class ModelResult(BaseModel):
    probability_fake: float
    risk_level: str


class EnsembleResult(BaseModel):
    probability_fake: float
    risk_level: str
    source: str


class PredictResponse(BaseModel):
    input_text:          str
    ensemble:            EnsembleResult
    roberta:             ModelResult
    qwen_zero_shot:      ModelResult
    xgboost:             ModelResult
    logistic_regression: ModelResult


# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "system":    "Misinformation Risk Intelligence System",
        "version":   "2.0.0",
        "status":    "running",
        "models":    ["RoBERTa (fine-tuned)", "Qwen2.5-3B (zero-shot)", "XGBoost", "Logistic Regression"],
        "endpoints": ["/predict", "/health", "/docs"],
    }


@app.get("/health")
def health():
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