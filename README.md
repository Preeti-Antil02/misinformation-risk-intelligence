---
<<<<<<< HEAD
title: RiskLens Misinformation Risk Intelligence
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ RiskLens — Enterprise Misinformation Risk Intelligence System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Telegram Bot](https://img.shields.io/badge/Telegram_Bot-v21.1.1-2CA5E0.svg?logo=telegram&logoColor=white)](https://core.telegram.org/bots)
[![Docker Ready](https://img.shields.io/badge/Docker-Multi--Container-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Release](https://img.shields.io/badge/Release-v2.1.0-emerald.svg)]()

> **Real-time, multi-modal, and explainable misinformation detection powered by calibrated stacking ensembles, Indic multilingual intelligence, and agentic web verification.**

---

## 📌 Overview

**RiskLens** is an enterprise-grade NLP and computer vision intelligence system built to analyze, detect, and quantify misinformation risk across news articles, social media text, forwarded web links, and image screenshots. Rather than treating misinformation as a brittle binary label ("fake" vs. "real"), RiskLens treats it as a continuous, multidimensional risk assessment problem — outputting calibrated risk tiers (**Low**, **Medium**, **High**, **Critical**) paired with mathematical uncertainty guarantees, feature-attribution explainability, and evidence-grounded fact-checking.

RiskLens is designed for intelligence analysts, fact-checkers, newsrooms, content moderation desks, and everyday consumers receiving viral claims on encrypted messaging platforms like Telegram.

### What Makes RiskLens Different

1. **Multi-Model Stacking Ensemble with True Calibration**: Combines four diverse base classifiers (Logistic Regression baseline, XGBoost with engineered linguistic manipulation signals, fine-tuned RoBERTa transformer, and zero-shot Qwen2.5-3B SLM) under a meta-learner, calibrated via Platt Scaling and Isotonic Regression to eliminate overconfidence (Test AUC: **0.9744**, F1: **0.9169**).
2. **Mathematical Uncertainty via Conformal Prediction**: Implements Split Conformal Prediction to produce distribution-free statistical coverage guarantees ($90\%$ empirical coverage). Ambiguous prediction sets automatically escalate to deep verification.
3. **Agentic Web Verification & Source Credibility**: A 3-node LangGraph agent extracts discrete claims, queries the Google Fact Check Tools API and live web search engines (Serper Google Search / DuckDuckGo fallback), and cross-references a curated domain reputation database to synthesize evidence-weighted verdicts with cited debunks.
4. **Multilingual Indic & Multi-Modal OCR Coverage**: Native language routing and fine-tuned **MuRIL** (Multilingual Representations for Indic Languages) support for **Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, and English**, coupled with an **EasyOCR + Tesseract** image preprocessing pipeline for screenshot verification.
5. **Continuous Learning Flywheel**: Active learning queue with automated daily retraining (`APScheduler` at 02:00 UTC) on user feedback samples with strict data poisoning and sanity checks.

---

## ⚡ How It Works (Pipeline Architecture)

```
 [ User Input: Text / URL / Image Screenshot ]
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 1. Ingestion & Preprocessing                             │
 │    • Webhook / Streamlit Input                           │
 │    • Multi-tier URL Scraper (Chrome / Googlebot / DDG)   │
 │    • Dual-Engine OCR (EasyOCR + Tesseract + OpenCV)      │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 2. Language Detection & Routing                          │
 │    • Unicode Script Analyzer + langdetect                │
 │    ├─ Indic (hi, ta, te, bn, mr, gu) ──► MuRIL Fine-Tuned│
 │    └─ English (en) ───────────────────► Ensemble Models  │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 3. Multi-Model Inference & Feature Extraction            │
 │    • TF-IDF & Linguistic Manipulation Signals            │
 │    • Logistic Regression (AUC: 0.9302)                   │
 │    • XGBoost Classifier  (AUC: 0.9706)                   │
 │    • RoBERTa Transformer                                 │
 │    • Qwen2.5-3B Zero-Shot Classifier                     │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 4. Stacking Meta-Learner & Calibration                   │
 │    • Meta-Learner (Test AUC: 0.9744)                     │
 │    • Isotonic / Platt Probability Calibration (ECE: 0.009)│
 │    • Split Conformal Prediction (Coverage: 90.5%)        │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 5. LangGraph Agentic Verification & Source Credibility   │
 │    • Claim Extraction & Google Fact Check Tools API      │
 │    • Multi-Query Web Search (Serper / DuckDuckGo)        │
 │    • Domain Reputation Weighting (70% Web / 30% Model)   │
 └────────────────────┬─────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 6. Explainable Output & Telemetry Delivery               │
 │    • SHAP Feature Attribution + Attention Saliency       │
 │    • Interactive Streamlit Dashboard / Telegram Bot Card │
 │    • SQLite Telemetry & Active Learning Queue            │
 └──────────────────────────────────────────────────────────┘
```

> **Interactive Visualization**: To explore an interactive, clickable node-by-node breakdown of the complete pipeline with execution traces, launch the Streamlit app and navigate to the **Pipeline** tab.

---

## 🛠️ Tech Stack

| Domain | Technologies & Libraries |
|---|---|
| **Backend API & Web Server** | **FastAPI** (`0.110.0`), **Uvicorn** (`0.29.0`), **Pydantic** (`2.12.5`), **APScheduler** (`3.11.3`), **python-dotenv** (`1.2.2`), **Requests** (`2.32.5`) |
| **ML Models & NLP** | **Scikit-Learn** (`1.7.2`), **XGBoost** (`2.0.3`), **PyTorch** (`2.13.0`), **Hugging Face Transformers** (`4.38.2`), **MuRIL**, **RoBERTa**, **Qwen2.5-3B**, **SciPy** (`1.12.0`), **NumPy** (`1.26.4`), **Pandas** (`2.2.2`), **Joblib** (`1.3.2`) |
| **Explainability & Uncertainty** | **SHAP** (`0.45.1`), **Split Conformal Prediction**, **Platt Scaling & Isotonic Calibration** |
| **Agentic Verification & Search** | **LangGraph**, **Google Fact Check Tools API**, **Serper.dev API**, **DuckDuckGo Search** (`8.1.1`), **BeautifulSoup4** (`4.15.0`), **langdetect** (`1.0.9`), **NLTK** (`3.9.2`) |
| **Computer Vision & OCR** | **EasyOCR** (`1.7.2`), **PyTesseract** (`0.3.13`), **OpenCV Headless** (`4.9.0.80`), **Pillow** (`10.4.0`) |
| **Bot & User Interfaces** | **Python-Telegram-Bot** (`21.1.1`), **Streamlit** (`1.32.2`), Custom Glassmorphism CSS Design System |
| **Storage, Infra & Security** | **SQLite3** (WAL Mode, Persistent Disk), **Docker** (Multi-stage builds), **Pip-Audit** (`2.7.3`), HMAC-SHA256 User Pseudonymization |

---

## 💻 Getting Started (Local Development)

### 1. Prerequisites

- **Python**: `3.11.x` (Recommended)
- **Git**
- **System OCR Dependencies** (Required for screenshot analysis):
  - **Ubuntu / Debian**:
    ```bash
    sudo apt-get update && sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin libgl1 libglib2.0-0
    ```
  - **macOS** (Homebrew):
    ```bash
    brew install tesseract tesseract-lang
    ```
  - **Windows**:
    - Download and run the [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki), ensuring Hindi language data is selected, or use Chocolatey: `choco install tesseract`. Ensure the installation directory is added to your system `PATH`.

---

### 2. Clone & Install Dependencies

```bash
# Clone repository
git clone https://github.com/Preeti-Antil02/misinformation-risk-intelligence.git
cd misinformation-risk-intelligence

# Create and activate virtual environment
python -m venv venv

# On Linux / macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Upgrade build tools and install consolidated dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

### 3. Environment Configuration (`.env`)

Copy the template `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Open `.env` and configure each environment variable:

| Variable Name | Description | Where to Get It / Format |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Authentication token for the Telegram Bot | Create a bot via [@BotFather](https://t.me/BotFather) on Telegram |
| `TELEGRAM_WEBHOOK_SECRET` | Secret token verifying webhook authenticity (`X-Telegram-Bot-Api-Secret-Token`) | Generate any random 32+ char alphanumeric string |
| `GOOGLE_FACTCHECK_API_KEY` | API Key for Google Fact Check Tools API | Create a project in [Google Cloud Console](https://console.cloud.google.com/) and enable "Fact Check Tools API" |
| `SERPER_API_KEY` | API Key for live Google organic web search | Register for a free API key at [serper.dev](https://serper.dev) |
| `RISKLENS_API_KEY` | Custom admin bearer token protecting `/analytics` and `/operations/metrics` endpoints | Any secret string (e.g. `rk_admin_secret_key_2027`) |
| `USER_ID_SALT` | Secret salt for HMAC-SHA256 user ID pseudonymization | Any secret string (e.g. `salt_crypto_risklens_secret`) |
| `SENTRY_DSN` | *(Optional)* Sentry error tracking DSN with automatic PII scrubbing | [sentry.io](https://sentry.io) (Free tier: 5k events/mo) |
| `TELEGRAM_ADMIN_CHAT_ID` | *(Optional)* Telegram user/group ID for instant push alerts | Direct chat with bot or numeric admin ID |
| `ALERT_WEBHOOK_URL` | *(Optional)* Slack or Discord Webhook URL for team alert notifications | Webhook URL from Slack/Discord integration |
| `ACCURACY_ALERT_THRESHOLD` | Threshold for model drift alerting | Float (Default: `0.75` / 75%) |
| `DATABASE_DIR` | Directory path for persistent SQLite databases | Local default: `./databases` (or `/app/databases` in Docker) |
| `TWILIO_ACCOUNT_SID` | *(Optional)* Twilio Account SID for legacy WhatsApp sandbox | [Twilio Console](https://console.twilio.com/) (Leave blank if using Telegram only) |
| `TWILIO_AUTH_TOKEN` | *(Optional)* Twilio Auth Token | [Twilio Console](https://console.twilio.com/) |
| `TWILIO_WHATSAPP_NUMBER` | *(Optional)* Twilio Sandbox WhatsApp number | Format: `whatsapp:+14155238886` |
| `ENVIRONMENT` | Runtime environment flag | Set to `development` for local or `production` |
| `LOG_LEVEL` | Logging verbosity | `INFO` or `DEBUG` |
| `PORT` | Local FastAPI backend port | Default: `8000` |
| `BACKEND_API_URL` | Base URL of the running backend service | Local default: `http://127.0.0.1:8000` |

---

### 4. Running Locally

RiskLens can be run as individual components or simultaneously in separate terminal windows:

#### Option A: Run the FastAPI Backend Server
Starts the REST API, inference pipeline, rate-limiting store, and background retraining scheduler:
```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```
- API Documentation (Swagger UI): `http://127.0.0.1:8000/docs`
- Health Check Endpoint: `http://127.0.0.1:8000/health`
- Protected Telemetry Endpoint: `http://127.0.0.1:8000/analytics` (Requires `X-API-Key` header)

#### Option B: Run the Streamlit Enterprise Dashboard
Launches the full interactive UI (Verify, Analytics, Pipeline Architecture, History, Settings):
```bash
streamlit run app.py --server.port 8501
```
Open `http://localhost:8501` in your browser.

#### Option C: Run the Telegram Bot in Polling Mode (Development)
For local development without setting up public HTTPS webhooks, run the bot in polling mode:
```bash
python risklens/telegram_bot.py
```

---

### 5. Running Standalone Test Suites & Benchmarks

The repository includes standalone validation and benchmark scripts:

```bash
# 1. Multi-Modal Verification Suite (Simulates Text, URL, and Screenshot OCR verification)
python test_whatsapp_bot.py

# 2. Phase 1 Master Benchmark (Stacking Ensemble, Platt/Isotonic Calibration, Evaluation Matrix)
python run_phase1.py

# 3. Phase 2 Master Benchmark (SHAP Explainer, Source Credibility, Fact-Check Engine, LangGraph Agent)
python run_phase2.py

# 4. Phase 3 Master Benchmark (Multilingual MuRIL Evaluation, Image OCR Suite, Continuous Learning Flywheel)
python run_phase3.py

# 5. Dependency Vulnerability Security Scan
python scripts/security_scan.py
```

---

## 🚢 Deployment

RiskLens is architected for containerized deployment on platforms like **Render** or **Railway** using a dual-service topology with persistent disk storage:

- **`risklens-api`**: FastAPI Web Service (`Dockerfile.backend`) handling `/predict`, `/telegram/webhook`, `/analytics`, and the daily `APScheduler` at 02:00 UTC.
- **`risklens-dashboard`**: Streamlit Web Service (`Dockerfile.dashboard`) delivering the analyst UI.
- **`risklens-storage`**: 5 GB Persistent Disk Volume mounted at `/app/databases` guaranteeing SQLite databases (`feedback.db`, `usage.db`) persist across container redeployments and rollbacks.

### Deployment Blueprint
The repository includes a ready-to-deploy [`render.yaml`](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/render.yaml) blueprint. For complete step-by-step instructions, health check verification, webhook registration commands, and disaster recovery runbooks, see:

👉 **[Read the Full Production Deployment Runbook (DEPLOYMENT.md)](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/DEPLOYMENT.md)**

---

## 📁 Project Structure

```
misinformation-risk-intelligence/
├── .env.example                 # Production environment variable template with explanations
├── .gitignore                   # Ignores .env, models/weights, cache, databases, and venv
├── api.py                       # Production FastAPI application (Inference, Webhook, Analytics)
├── app.py                       # Root entry point for Streamlit Enterprise UI
├── Dockerfile.backend           # Hardened production container definition for FastAPI API
├── Dockerfile.dashboard         # Hardened production container definition for Streamlit UI
├── docker-compose.yml           # Local multi-container orchestration manifest
├── render.yaml                  # Infrastructure-as-Code Blueprint for Render deployment
├── requirements.txt             # Consolidated, pinned production dependencies
├── DEPLOYMENT.md                # Complete deployment runbook and rollback procedures
├── HARDENING_NOTES.md           # Security audit and resilience engineering documentation
├── LICENSE                      # Project license file (pending selection)
├── test_whatsapp_bot.py         # Multi-modal verification simulation test suite
├── run_phase1.py                # Stacking ensemble training & calibration benchmark runner
├── run_phase2.py                # Explainability, credibility & LangGraph agent test runner
├── run_phase3.py                # Multilingual MuRIL & OCR evaluation benchmark runner
│
├── app/                         # Modular Streamlit Application
│   ├── config.py                # UI constants, preset verification examples, and typography
│   ├── main.py                  # Streamlit secondary entrypoint
│   ├── state.py                 # Session state manager
│   ├── ui/                      # Refactored UI tab modules
│   │   ├── sidebar.py           # Real-time telemetry, model metrics, and system status sidebar
│   │   ├── tab_verify.py        # Core verification workspace (Text, URL, Image OCR)
│   │   ├── tab_analytics.py     # Live accuracy charts, confusion matrix, and feedback queue
│   │   ├── tab_pipeline.py      # Interactive pipeline architecture and execution trace
│   │   ├── tab_history.py       # Session prediction history with JSON/CSV export
│   │   ├── tab_settings.py      # Runtime configuration and API key management
│   │   └── theme.py             # Custom glassmorphism CSS theme engine
│   └── ui/components/           # Reusable UI widgets (Trust ring, Pipeline visualizer, etc.)
│
├── risklens/                    # Core Intelligence & Production Modules
│   ├── __init__.py              # Package initialization (v2.1.0)
│   ├── active_learning.py       # Human-in-the-loop uncertainty sampling & Level-1 retrainer
│   ├── agent.py                 # 3-node LangGraph verification agent (Extract, Search, Synthesize)
│   ├── claim_checker.py         # Google Fact Check Tools API client & verified debunks cache
│   ├── conformal_predictor.py   # Split Conformal Prediction engine for 90% statistical coverage
│   ├── explainer.py             # TreeSHAP feature attributions & RoBERTa attention token highlights
│   ├── feedback.py              # SQLite feedback logger, telemetry aggregation & model promotion
│   ├── logging_config.py        # Centralized structured logging with rotating file handlers
│   ├── monitoring.py            # Sentry error tracking, PII scrubbing, push alerting & telemetry
│   ├── multilingual.py          # Indic script detector, language router & MuRIL classifier
│   ├── ocr_pipeline.py          # Dual EasyOCR + Tesseract pipeline with OpenCV preprocessing
│   ├── ood_evaluator.py         # Out-of-domain benchmark evaluator (Health, Finance, Politics)
│   ├── source_credibility.py    # Domain reputation scorer & integrated risk calculator
│   ├── telegram_bot.py          # Production Telegram Bot with rate limiting & secret validation
│   ├── url_reader.py            # Multi-tier web scraper with browser spoofing & index fallback
│   ├── utils.py                 # Resilient network calls with exponential backoff & jitter
│   └── whatsapp_bot.py          # WhatsApp message card formatter & webhook helper
│
├── src/                         # Core Machine Learning Pipelines & Base Classifiers
│   ├── data_loader.py           # Dataset loading utilities (WELFake, ISOT, GossipCop, PolitiFact)
│   ├── preprocessing.py         # Data cleaning and tokenization helpers
│   ├── risk_scoring.py          # Risk level thresholding logic (Low, Medium, High, Critical)
│   ├── domain_testing.py        # Cross-domain validation scripts
│   ├── error_analysis.py        # Confusion matrix and error breakdown generation
│   ├── features/                # Feature extraction modules
│   │   ├── feature_builder.py   # Combined TF-IDF and manipulation feature matrix builder
│   │   ├── feature_engineering.py # Manipulation signal extractors (sentiment, caps, punctuation)
│   │   └── text_preprocessor.py # Text normalization, cleaning, and truncation
│   ├── models/                  # Base model implementations and trainers
│   │   ├── roberta_model.py     # Fine-tuned RoBERTa transformer classifier
│   │   ├── slm_model.py         # Zero-shot Small Language Model classifier (Qwen2.5-3B)
│   │   ├── stacking_ensemble.py # Level-1 Stacking Meta-Learner implementation
│   │   ├── calibration.py       # Platt scaling and Isotonic regression calibrators
│   │   ├── evaluate.py          # Multi-metric evaluation and matrix generation
│   │   └── evaluation.py        # Detailed cross-validation routines
│   └── explainability/          # Global and local SHAP explanation helpers
│       └── shap_explainer.py    # TreeSHAP explainer interface
│
├── data/                        # Datasets, Domain Mappings, and Local Feedback DB
│   ├── domain_credibility.csv   # Curated database of news domains and credibility rankings
│   ├── multilingual_dataset.csv # Curated Indic language benchmark dataset
│   ├── ood_benchmark_dataset.csv# Benchmark dataset across Health, Finance, Climate, Politics
│   └── active_learning_feedback.db # Local feedback collection SQLite database
│
├── models/                      # Trained model artifacts & scalers
│   ├── baseline_logistic.pkl    # TF-IDF Logistic Regression model
│   ├── xgboost_model.pkl        # XGBoost Classifier with linguistic manipulation features
│   ├── tfidf_vectorizer.pkl     # Fitted Scikit-Learn TF-IDF vectorizer
│   ├── numeric_scaler.pkl       # Fitted numeric feature scaler
│   ├── ensemble_model.pkl       # Stacking Meta-Learner model
│   ├── calibrated_ensemble.pkl  # Isotonic-calibrated Stacking Ensemble model
│   └── muril_finetuned/         # Fine-tuned MuRIL model configuration, tokenizer, and weights
│
├── results/                     # Diagnostic reports, evaluation matrices, and benchmark outputs
└── scripts/                     # Operational scripts
    └── security_scan.py         # Automated pip-audit dependency vulnerability scanner
```

---

## 🔒 Security & Production Hardening

RiskLens has undergone systematic security hardening across its codebase (summarized in [`HARDENING_NOTES.md`](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/HARDENING_NOTES.md)):

- **Zero Hardcoded Secrets**: All API tokens, secret keys, and credentials are read strictly from environment variables.
- **PII Protection & Cryptographic Pseudonymization**: User identifiers (phone numbers, Telegram IDs) are hashed using `HMAC-SHA256` with a server salt before being recorded in SQLite database tables.
- **Database Safety & Concurrency**: All SQLite interactions use parameterized queries (`?` placeholders) and contextual connection handlers with retry logic to prevent SQL injection and database locking issues.
- **Telegram Webhook Authenticity**: Validates the `X-Telegram-Bot-Api-Secret-Token` header on all incoming webhook requests to prevent replay and spoofing attacks.
- **Abuse Prevention**: User-level rate limiting enforced via SQLite `usage.db` (default: 20 requests per user per day).
- **HTTP Security Headers**: Enterprise HTTP headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Strict-Transport-Security`) injected into all FastAPI responses.
- **Automated Dependency Auditing**: Continuous dependency vulnerability scanning via `pip-audit` (`scripts/security_scan.py`).

---

## 🗺️ Roadmap & Deferred Features

The following features represent planned future expansions:

- [ ] **Synthesized Video & Deepfake Audio Detection**: Integration of spatial-temporal face manipulation detectors and audio frequency analysis for multi-modal broadcast clips.
- [ ] **Browser Extension**: Lightweight Chrome/Firefox extension for inline highlight verification while browsing social feeds.
- [ ] **On-Device Edge Mode**: Quantized ONNX / CoreML model export for offline on-device inference without server round-trips.
- [ ] **Native Meta WhatsApp Cloud API**: Direct webhook migration to Meta Cloud Business API from Telegram/Twilio sandbox.
- [ ] **Knowledge Graph Entity Linking**: Automated entity reconciliation linking claims to structured Wikidata and DBpedia knowledge graphs.

---

## 🤝 Contributing

RiskLens is currently developed as an internal research and production project. External contributions, pull requests, and feature submissions are not actively being accepted at this time.

---

## 📄 License

*License terms are currently pending formal selection.* See [`LICENSE`](file:///c:/Users/preet/Documents/misinformation-risk-intelligence/LICENSE) for future updates.
=======
title: RiskLens
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app/streamlit_app.py
pinned: false
---

🚀 RiskLens: Misinformation Risk Intelligence System
📌 Overview

RiskLens is an end-to-end NLP system designed to detect and quantify misinformation risk in news articles.
Unlike traditional fake news classifiers, it provides risk scoring and explainability, enabling more informed decision-making.

This is not just a classifier.
It’s a decision-support system for misinformation risk assessment.

🎯 Key Features
🔍 Multi-model pipeline
Combines Logistic Regression, XGBoost, and fine-tuned BERT for robust predictions

🧠 Hybrid feature engineering
Integrates TF-IDF with linguistic manipulation signals:
Sentiment polarity
Capitalization ratio
Extreme keyword detection

⚠️ Risk Scoring System
Converts predictions into actionable categories:
Low → Medium → High → Critical

📊 Explainable AI (SHAP)
Global feature importance
Local prediction explanations

📈 Robust Evaluation
Stratified Cross-Validation
Cross-dataset testing (GossipCop, PolitiFact)
Metrics: F1-score, ROC-AUC, Confusion Matrix


🌐 Real-time Deployment
Interactive web app with live predictions


🏗️ System Architecture
User Input (News Article)
        ↓
Text Preprocessing
        ↓
Feature Engineering (TF-IDF + Signals)
        ↓
Model Ensemble
(LogReg + XGBoost + BERT)
        ↓
Prediction + Confidence Score
        ↓
Risk Classification
        ↓
SHAP Explainability
        ↓
Frontend Display (Streamlit)


🛠️ Tech Stack
Languages: Python
ML/DL: Scikit-learn, XGBoost, Transformers (BERT)
Explainability: SHAP
Frontend: Streamlit
Deployment: Docker, Hugging Face

🎥 Demo

🔗 Live App: https://preeti-antil-risklens.hf.space/
🔗 GitHub Repo: https://github.com/Preeti-Antil02/misinformation-risk-intelligence

🚀 Deployment
Deployed as an interactive web application using Streamlit
Hosted on Hugging Face Spaces for real-time access
Lightweight setup focused on rapid prototyping and usability

🧠 Future Improvements
Real-time social media stream analysis
Multilingual misinformation detection
Knowledge graph integration for fact verification


📌 Applications
Social media monitoring systems
News verification platforms
Content moderation pipelines

👤 Author
    Preeti 
>>>>>>> origin/main
