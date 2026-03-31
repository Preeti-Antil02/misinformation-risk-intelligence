---
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