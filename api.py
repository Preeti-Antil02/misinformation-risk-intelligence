"""
Misinformation Risk Intelligence System — API Layer
FastAPI inference endpoint

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Example request:
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
from src.models.bert_model import BertClassifier
from src.risk_scoring import RiskScorer

# -------------------------------------------------------
# App
# -------------------------------------------------------
app = FastAPI(
    title="Misinformation Risk Intelligence API",
    description=(
        "Assesses misinformation risk using three models: "
        "Logistic Regression, XGBoost, and fine-tuned BERT. "
        "Primary result is a weighted ensemble."
    ),
    version="1.1.0",
)

# -------------------------------------------------------
# Load models once at startup
# -------------------------------------------------------
lr     = joblib.load("models/baseline_logistic.pkl")
xgb    = joblib.load("models/xgboost_model.pkl")
tfidf  = joblib.load("models/tfidf_vectorizer.pkl")
scaler = joblib.load("models/numeric_scaler.pkl")

bert = BertClassifier()
bert.load("models/bert_finetuned")

tp = TextPreprocessor()
fb = FeatureBuilder()
rs = RiskScorer()

# -------------------------------------------------------
# Request / response schemas
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
    input_text: str
    ensemble: EnsembleResult
    bert: ModelResult
    xgboost: ModelResult
    logistic_regression: ModelResult


# -------------------------------------------------------
# Endpoints
# -------------------------------------------------------
@app.get("/")
def root():
    return {
        "system": "Misinformation Risk Intelligence System",
        "status": "running",
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

    temp_df = pd.DataFrame({"text": [cleaned]})
    X_num = fb.build_features(temp_df).astype(np.float64)
    X_num_scaled = scaler.transform(X_num)
    X_combined = hstack([X_tfidf, csr_matrix(X_num_scaled)])

    lr_prob   = float(lr.predict_proba(X_tfidf)[0, 1])
    xgb_prob  = float(xgb.predict_proba(X_combined)[0, 1])
    bert_prob = float(bert.predict_proba([cleaned])[0])

    # Ensemble with BERT outlier detection
    lr_and_xgb_avg = (lr_prob * 0.20 + xgb_prob * 0.40) / 0.60
    bert_outlier = (
        bert_prob > 0.7 and lr_prob < 0.4 and xgb_prob < 0.4
    ) or (
        bert_prob < 0.3 and lr_prob > 0.6 and xgb_prob > 0.6
    )

    if bert_outlier:
        ensemble_prob   = lr_and_xgb_avg
        ensemble_source = "LR + XGBoost consensus (BERT excluded — outlier)"
    else:
        ensemble_prob   = (lr_prob * 0.20) + (xgb_prob * 0.40) + (bert_prob * 0.40)
        ensemble_source = "weighted ensemble"

    ensemble_risk = rs.score_ensemble(ensemble_prob)

    return PredictResponse(
        input_text=text,
        ensemble=EnsembleResult(
            probability_fake=round(ensemble_prob, 4),
            risk_level=ensemble_risk,
            source=ensemble_source,
        ),
        bert=ModelResult(
            probability_fake=round(bert_prob, 4),
            risk_level=rs.score(bert_prob),
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