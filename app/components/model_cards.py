"""
app/components/model_cards.py
=============================
4-Model Comparative Analysis Grid.
"""

import streamlit as st
from app.config import RISK_CONFIG


def render_model_cards(results: dict):
    """
    Renders a 4-column responsive grid comparing all base Level-0 classifiers.
    """
    c1, c2, c3, c4 = st.columns(4)
    models = [
        (c1, "Logistic Regression", "TF-IDF N-grams", results["lr"]["prob"], results["lr"]["risk"]),
        (c2, "XGBoost Classifier", "TF-IDF + 7 Signals", results["xgb"]["prob"], results["xgb"]["risk"]),
        (c3, "RoBERTa Classifier", "Neural Transformer", results["roberta"]["prob"], results["roberta"]["risk"]),
        (c4, "Qwen2.5-3B SLM", "Zero-Shot Reasoning", results["qwen"]["prob"], results["qwen"]["risk"]),
    ]

    for col, name, desc, prob, risk in models:
        cfg = RISK_CONFIG.get(risk, RISK_CONFIG["Moderate"])
        with col:
            st.markdown(f"""
            <div style="background:#0d1422;border:1px solid #1c2a40;border-top:3px solid {cfg['border']};border-radius:12px;padding:16px;min-height:130px;">
                <div style="font-size:13px;font-weight:700;color:#ffffff;">{name}</div>
                <div style="font-size:11px;color:#6e7f96;margin-bottom:10px;">{desc}</div>
                <div style="display:flex;justify-content:space-between;align-items:flex-end;">
                    <div>
                        <div style="font-size:10px;color:#8b9bb4;text-transform:uppercase;">Tier</div>
                        <div style="font-size:13px;font-weight:700;color:{cfg['text']};">{risk}</div>
                    </div>
                    <div style="font-size:20px;font-weight:800;color:#ffffff;font-family:'JetBrains Mono',monospace;">{int(round(prob*100))}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
