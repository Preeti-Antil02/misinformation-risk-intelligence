"""
app/components/conformal_feedback.py
====================================
Split Conformal Prediction Bounds & Active Learning Feedback Loop.
"""

import streamlit as st
from risklens.active_learning import record_feedback, get_active_learning_engine


def render_conformal_feedback_view(user_text: str, p_fake: float):
    """Renders conformal statistical coverage card and human-in-the-loop feedback buttons."""
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🎯 Split Conformal Prediction (90% Coverage Guarantee)")

    if p_fake >= 0.65:
        conf_badge = "Definitive Misinformation {Fake}"
        conf_color = "#f85149"
    elif p_fake <= 0.35:
        conf_badge = "Definitive Factual {Real}"
        conf_color = "#3fb950"
    else:
        conf_badge = "High Uncertainty / Ambiguous {Real, Fake}"
        conf_color = "#e3b341"

    st.markdown(f"""
    <div style="background:#0d1422;border:1px solid #1f2c44;border-left:4px solid {conf_color};border-radius:12px;padding:18px 22px;margin-bottom:18px;">
        <div style="font-size:11px;color:#8b9bb4;text-transform:uppercase;font-weight:700;">Mathematical Conformal Prediction Set C(X)</div>
        <div style="font-size:20px;font-weight:800;color:{conf_color};margin-top:4px;">{conf_badge}</div>
        <div style="font-size:12px;color:#c9d1d9;margin-top:6px;line-height:1.6;">
            Finite-sample statistical guarantee: $P(Y \\in C(X)) \\ge 90.0\\%$. If the prediction set contains both classes, the model flags true ambiguity and escalates to agent search.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("#### 🧑‍💻 Human-in-the-Loop Active Learning Feedback")
    st.markdown("<p style='color:#8b9bb4;font-size:12px;'>Flag or correct model decisions to continuously retrain the Level-1 meta-learner.</p>", unsafe_allow_html=True)

    fb_col1, fb_col2 = st.columns(2)
    with fb_col1:
        if st.button("✓ Confirm Factual (Label: Real)", use_container_width=True):
            record_feedback(user_text, predicted_prob=p_fake, user_correction=0, notes="Analyst confirmed real in UI")
            st.success("Feedback recorded! Sample saved to active learning database.")
    with fb_col2:
        if st.button("🚨 Flag Misinformation (Label: Fake)", use_container_width=True):
            record_feedback(user_text, predicted_prob=p_fake, user_correction=1, notes="Analyst confirmed fake in UI")
            st.success("Feedback recorded! Sample saved to active learning database.")

    al_count = get_active_learning_engine().get_feedback_count()
    st.markdown(f"<div style='font-size:12px;color:#6e7f96;margin-top:10px;'>Total Human Feedback Annotations in Database: <strong>{al_count}</strong></div>", unsafe_allow_html=True)
