"""
app/main.py
===========
RiskLens Production Web Application Entry Point.
Modular, high-performance, and responsive multi-layer intelligence dashboard.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

st.set_page_config(
    page_title="RiskLens — Misinformation Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.config import CUSTOM_CSS, PRESET_EXAMPLES
from app.state import SessionStateManager
from app.utils.inference import load_all_models, run_fast_inference
from app.components.header import render_header
from app.components.stat_cards import render_stat_cards
from app.components.sidebar_analytics import render_sidebar_analytics
from app.components.input_section import render_input_section
from app.components.primary_card import render_primary_card
from app.components.model_cards import render_model_cards
from app.components.explainability import render_explainability_view
from app.components.agent_view import render_agent_view
from app.components.domain_view import render_domain_view
from app.components.conformal_feedback import render_conformal_feedback_view
from app.components.whatsapp_view import render_whatsapp_view

from risklens.explainer import explain_prediction
from risklens.source_credibility import get_source_credibility
from risklens.claim_checker import full_claim_pipeline
from risklens.agent import verify


def main():
    # Inject Custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Initialize State
    SessionStateManager.initialize()

    # Load Cached Models
    models_bundle = load_all_models()
    lr, xgb, tfidf, scaler, roberta, qwen, explainer, tp, fb, rs, calibrated_ensemble = models_bundle

    # -------------------------------------------------------
    # TOP BAR: Navigation & Branding
    # -------------------------------------------------------
    render_header()

    # -------------------------------------------------------
    # STAT CARDS: System Telemetry
    # -------------------------------------------------------
    render_stat_cards()
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # MAIN CONTENT: Two-Column Dashboard
    # -------------------------------------------------------
    main_col, side_col = st.columns([2.2, 1], gap="large")

    with main_col:
        # Input Section
        user_text, user_url, trigger_analysis = render_input_section()

        if not user_text.strip():
            st.info("💡 Enter a news headline, claim, or live web article URL above to begin intelligence analysis.")
            # Still render the sidebar even if no input
            with side_col:
                render_sidebar_analytics()
            return

        # ANALYSIS EXECUTION / CACHE
        if trigger_analysis or st.session_state.get("analysis_cache") is None or st.session_state.get("last_analyzed_text") != user_text:
            with st.spinner("Analyzing across 4 AI models, SHAP explainer & LangGraph agents..."):
                results = run_fast_inference(user_text, models_bundle)
                exp_data = explain_prediction(user_text)
                cred_data = get_source_credibility(user_url) if user_url else None
                claim_data = full_claim_pipeline(user_text)
                agent_data = verify(user_text, url=user_url if user_url else None)

                st.session_state.analysis_cache = {
                    "results": results,
                    "exp_data": exp_data,
                    "cred_data": cred_data,
                    "claim_data": claim_data,
                    "agent_data": agent_data,
                }
                st.session_state.last_analyzed_text = user_text

        cache = st.session_state.get("analysis_cache")
        if cache:
            results = cache["results"]
            exp_data = cache["exp_data"]
            cred_data = cache["cred_data"]
            agent_data = cache["agent_data"]
            claim_data = cache.get("claim_data", {})

            # Redesigned PRIMARY RISK CARD
            final_prob = agent_data.get("risk_score", results["ensemble"]["prob"])
            final_risk = agent_data.get("risk_level", results["ensemble"]["risk"])

            render_primary_card(
                risk_level=final_risk,
                probability=final_prob,
                claim_text=user_text,
                verdict=exp_data.get("why_summary", "Claim analysis complete."),
                suspicious_phrases=exp_data.get("top_features", []),
                shap_data=[{"feature": f, "value": v} for f, v in exp_data.get("feature_importance", {}).items()][:4],
                sources=[{"name": "Verified Source", "status": "Credible", "url": "#", "domain": "source.com"}]
            )

            # Secondary Tabs for detailed drills
            st.markdown("#### 🔍 Detailed Intelligence Drills")
            t1, t2, t3, t4 = st.tabs([
                "🤖 Live Verification",
                "📊 Model Diagnostics",
                "🌐 Source Reputation",
                "🎯 Conformal Bounds"
            ])

            with t1:
                render_agent_view(agent_data)
            with t2:
                render_model_cards(results)
                render_explainability_view(results, tfidf, fb)
            with t3:
                render_domain_view(cred_data, results["ensemble"]["prob"], agent_data.get("risk_score", results["ensemble"]["prob"]))
            with t4:
                render_conformal_feedback_view(user_text, results["ensemble"]["prob"])

    with side_col:
        render_sidebar_analytics()

        st.markdown("<hr style='border-color:rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
        st.markdown("### ⚡ Quick Presets")
        for key, p in PRESET_EXAMPLES.items():
            if st.button(p["title"], key=f"preset_{key}", use_container_width=True):
                SessionStateManager.load_preset(key)
                st.rerun()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
