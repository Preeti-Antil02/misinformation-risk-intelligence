"""
Misinformation Risk Intelligence System
This Streamlit app provides an interactive interface to assess the misinformation risk of news articles and claims using three independent models: BERT, XGBoost, and Logistic Regression. Users can paste text, run the analysis, and see a comprehensive risk assessment along with explainability insights.
"""

import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from scipy.sparse import hstack

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Misinformation Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------
# Path fix
# -------------------------------------------------------
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from src.features.text_preprocessor import TextPreprocessor
from src.features.feature_builder import FeatureBuilder
from src.models.bert_model import BertClassifier
from src.risk_scoring import RiskScorer

# -------------------------------------------------------
# Global CSS
# -------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

div.block-container {
    padding-top: 1rem !important;
}
            
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0d1117 100%) !important;
    border-right: 1px solid #21262d;
    padding-top: 0 !important;
}

.stApp { background: #0d1117; }

textarea {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 12px !important;
    color: #e6edf3 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    padding: 14px !important;
    transition: border-color 0.2s !important;
}
textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.1) !important;
}

div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%) !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 12px 32px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2ea043 0%, #3fb950 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(46,160,67,0.3) !important;
}

div[data-testid="stButton"] button[kind="secondary"] {
    background: #161b22 !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #21262d !important;
    color: #e6edf3 !important;
    border-color: #58a6ff !important;
}

hr { border-color: #21262d !important; margin: 24px 0 !important; }

div[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid #21262d;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    background: #1c1a00 !important;
    border: 1px solid #9e6a03 !important;
    border-radius: 12px !important;
    color: #d29922 !important;
}

div[data-testid="stSpinner"] p { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Risk config
# -------------------------------------------------------
RISK_CONFIG = {
    "Low": {
        "bg": "linear-gradient(135deg, #0d2818 0%, #0f3a1f 100%)",
        "border": "#2ea043",
        "text": "#3fb950",
        "badge_bg": "#0f3a1f",
        "badge_text": "#3fb950",
        "glow": "rgba(63,185,80,0.2)",
        "bar": "#3fb950",
    },
    "Moderate": {
        "bg": "linear-gradient(135deg, #1c1500 0%, #2a1e00 100%)",
        "border": "#d29922",
        "text": "#e3b341",
        "badge_bg": "#2a1e00",
        "badge_text": "#e3b341",
        "glow": "rgba(227,179,65,0.2)",
        "bar": "#e3b341",
    },
    "High": {
        "bg": "linear-gradient(135deg, #2d1800 0%, #3d2200 100%)",
        "border": "#db6d28",
        "text": "#f0883e",
        "badge_bg": "#3d2200",
        "badge_text": "#f0883e",
        "glow": "rgba(240,136,62,0.2)",
        "bar": "#f0883e",
    },
    "Critical": {
        "bg": "linear-gradient(135deg, #2d0f0f 0%, #3d1414 100%)",
        "border": "#da3633",
        "text": "#f85149",
        "badge_bg": "#3d1414",
        "badge_text": "#f85149",
        "glow": "rgba(248,81,73,0.25)",
        "bar": "#f85149",
    },
}

RISK_ICONS = {"Low": "✓", "Moderate": "◈", "High": "⚠", "Critical": "✕"}

POLITICAL_WORDS = [
    "trump", "clinton", "obama", "biden", "harris", "modi", "putin",
    "xi jinping", "macron", "zelensky", "netanyahu", "erdogan",
    "pelosi", "mcconnell", "gop", "democrat", "republican",
    "labour", "conservative", "liberal",
    "government", "minister", "president", "prime minister",
    "parliament", "congress", "senate", "white house", "kremlin",
    "pentagon", "administration", "federal", "cabinet",
    "supreme court", "legislation", "constitution", "chancellor",
    "election", "vote", "ballot", "campaign", "referendum",
    "political", "democracy", "autocracy", "inauguration",
    "midterm", "polling", "candidate", "party", "coalition",
    "diplomatic", "diplomacy", "sanctions", "treaty", "nato",
    "united nations", "g20", "g7", "embassy", "ambassador",
    "foreign", "bilateral", "multilateral", "summit", "ceasefire",
    "peace talks", "alliance", "geopolitical",
    "war", "military", "troops", "army", "navy", "missile",
    "nuclear", "weapons", "airstrike", "invasion", "occupation",
    "terrorist", "insurgent", "coup", "regime", "conflict",
    "border", "sovereignty", "annexation", "casualties",
    "economic", "trade", "tariff", "gdp", "inflation",
    "interest rate", "federal reserve", "central bank", "stimulus",
    "budget", "deficit", "debt", "policy", "reform", "subsidy",
    "china", "india", "russia", "iran", "ukraine", "israel",
    "gaza", "taiwan", "north korea", "pakistan", "afghanistan",
    "iraq", "syria", "europe", "asia", "middle east",
    "official", "spokesperson", "statement", "briefing",
    "investigation", "hearing", "testimony", "indictment",
    "impeachment", "scandal", "whistleblower", "classified",
    "leaked", "intelligence", "agency", "cia", "fbi",
]

FAKE_EXAMPLES = [
    "Ginger is 10,000x more effective at killing cancer than chemo, a new study finds. "
    "Doctors are refusing to tell patients because it would destroy the pharmaceutical industry.",

    "mRNA vaccines are gene therapy. A government insider has revealed the DNA-altering "
    "ingredients hidden in every dose. Share this before it gets deleted.",

    "Video shows White House unveiling secret plan to train bald eagles for border "
    "surveillance. Officials denied the program exists but footage proves otherwise.",

    "Putin intercepts secret shipments of adrenochrome in Ukraine. The globalist network "
    "has been exposed. This is what the mainstream media refuses to report.",

    "New pink salt diet drink burns fat overnight. A 72-year-old lost 20 lbs in 2 days. "
    "Doctors hate this one weird trick that big pharma has been suppressing for decades.",
]

REAL_EXAMPLES = [
    "Tiny gene mutations shape cancer behavior and immune response, according to a new "
    "study published in Nature. Researchers say the findings could improve targeted therapy.",

    "Bacteria that hunt and kill cancer cells have been discovered by researchers at the "
    "University of California. The organisms target tumours while leaving healthy tissue intact.",

    "Netanyahu video sparks AI debate after a disappearing ring moment went viral on social "
    "media. Experts remain divided on whether the footage was digitally altered.",

    "PSL 2026 is set to be played behind closed doors due to the escalating West Asia "
    "conflict, the Pakistan Cricket Board confirmed in an official statement on Friday.",

    "Researchers have uncovered the world's oldest known cave art, a 67,800-year-old hand "
    "stencil in Indonesia, hinting at early symbolic thinking and possibly spiritual beliefs.",
]

# -------------------------------------------------------
# Model loading
# -------------------------------------------------------
@st.cache_resource(show_spinner="Loading intelligence models...")
def load_all_models():
    lr     = joblib.load("models/baseline_logistic.pkl")
    xgb    = joblib.load("models/xgboost_model.pkl")
    tfidf  = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/numeric_scaler.pkl")

    bert = BertClassifier()
    bert.load("models/bert_finetuned")

    explainer = shap.TreeExplainer(xgb)
    tp = TextPreprocessor()
    fb = FeatureBuilder()
    rs = RiskScorer()

    return lr, xgb, tfidf, scaler, bert, explainer, tp, fb, rs

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------
def predict(text, lr, xgb, tfidf, scaler, bert, explainer, tp, fb, rs):
    cleaned = tp.basic_clean(text)
    cleaned = tp.truncate(cleaned)

    X_tfidf = tfidf.transform([cleaned])

    temp_df = pd.DataFrame({"text": [cleaned]})
    X_num_raw = fb.build_features(temp_df)

    def safe_to_float(val):
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        s = str(val).strip().strip("[]").split()[0]  
        try:
            return float(s)
        except (ValueError, IndexError):
            return 0.0

    X_num = pd.DataFrame(X_num_raw)
    X_num = X_num.applymap(safe_to_float).fillna(0.0)

    X_num_s = scaler.transform(X_num.values)

    from scipy.sparse import hstack, csr_matrix
    X_combined = hstack([X_tfidf, csr_matrix(X_num_s)])

    lr_prob   = float(lr.predict_proba(X_tfidf)[0, 1])
    xgb_prob  = float(xgb.predict_proba(X_combined)[0, 1])
    bert_prob = float(bert.predict_proba([cleaned])[0])

    lr_and_xgb_avg = (lr_prob * 0.35 + xgb_prob * 0.40) / 0.75

    bert_outlier = (
        bert_prob > 0.7 and lr_prob < 0.4 and xgb_prob < 0.4
    ) or (
        bert_prob < 0.3 and lr_prob > 0.6 and xgb_prob > 0.6
    )

    if bert_outlier:
        ensemble_prob   = lr_and_xgb_avg
        ensemble_source = "LR + XGBoost consensus (BERT excluded — outlier)"
    else:
        ensemble_prob   = (lr_prob * 0.35) + (xgb_prob * 0.40) + (bert_prob * 0.25)
        ensemble_source = "weighted ensemble"

    ensemble_risk = rs.score_ensemble(ensemble_prob)

    shap_vals = explainer.shap_values(X_combined)

    return {
        "lr":              {"prob": lr_prob,       "risk": rs.score(lr_prob)},
        "xgb":             {"prob": xgb_prob,      "risk": rs.score(xgb_prob)},
        "bert":            {"prob": bert_prob,     "risk": rs.score(bert_prob)},
        "ensemble":        {"prob": ensemble_prob, "risk": ensemble_risk},
        "ensemble_source": ensemble_source,
        "shap_values":     shap_vals,
        "X_combined":      X_combined,
        "cleaned_text":    cleaned,
    }
# -------------------------------------------------------
# SHAP helpers
# -------------------------------------------------------
def get_top_shap_words(shap_values, X_combined, tfidf, fb, n=12):
    all_names = tfidf.get_feature_names_out().tolist() + fb.feature_names

    if isinstance(shap_values, list):
        sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 2:
        sv = shap_values[0]
    else:
        sv = shap_values

    if hasattr(sv, "toarray"):
        sv = sv.toarray().flatten()
    elif hasattr(sv, "flatten"):
        sv = sv.flatten()

    pairs = [(all_names[i], float(sv[i]))
             for i in range(min(len(all_names), len(sv)))]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:n]


def highlight_text(text, top_words):
    result = text
    for word, contrib in top_words:
        if len(word) < 3:
            continue
        if contrib > 0.05:
            result = re.sub(
                rf'\b({re.escape(word)})\b',
                r'<mark style="background:rgba(248,81,73,0.18);'
                r'color:#f85149;border-radius:4px;padding:1px 5px;'
                r'border:1px solid rgba(248,81,73,0.3);">\1</mark>',
                result, flags=re.IGNORECASE
            )
        elif contrib < -0.05:
            result = re.sub(
                rf'\b({re.escape(word)})\b',
                r'<mark style="background:rgba(63,185,80,0.18);'
                r'color:#3fb950;border-radius:4px;padding:1px 5px;'
                r'border:1px solid rgba(63,185,80,0.3);">\1</mark>',
                result, flags=re.IGNORECASE
            )
    return result


def plot_shap_bar(top_words):
    words  = [w for w, _ in top_words[:10]]
    values = [v for _, v in top_words[:10]]
    colors = ["#f85149" if v > 0 else "#58a6ff" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(3, len(words) * 0.42)))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    ax.barh(words[::-1], values[::-1],
            color=colors[::-1], edgecolor="none", height=0.6)
    ax.axvline(0, color="#30363d", linewidth=1.2)
    ax.tick_params(colors="#8b949e", labelsize=10)
    ax.set_xlabel("SHAP contribution", color="#8b949e", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")

    fake_p = mpatches.Patch(color="#f85149", label="→ FAKE")
    real_p = mpatches.Patch(color="#58a6ff", label="→ REAL")
    ax.legend(handles=[fake_p, real_p], loc="lower right",
              fontsize=9, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="#8b949e")
    plt.tight_layout()
    return fig

# -------------------------------------------------------
# UI components
# -------------------------------------------------------
def section_title(text, subtitle=None):
    sub_html = f'<div style="font-size:13px;color:#8b949e;margin-top:4px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="margin-bottom:16px;">
        <div style="font-size:20px;font-weight:700;color:#e6edf3;
                    letter-spacing:-0.02em;">{text}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def primary_result_card(risk, prob, label="weighted ensemble"):
    cfg  = RISK_CONFIG[risk]
    icon = RISK_ICONS[risk]
    pct  = int(prob * 100)
    st.markdown(f"""
    <div style="
        background:{cfg['bg']};
        border:1px solid {cfg['border']};
        border-radius:16px;
        padding:28px 32px;
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:12px;
        box-shadow:0 0 40px {cfg['glow']};
    ">
        <div>
            <div style="
                display:inline-flex;align-items:center;gap:8px;
                background:{cfg['badge_bg']};
                border:1px solid {cfg['border']};
                border-radius:20px;padding:4px 14px;
                margin-bottom:12px;
            ">
                <span style="font-size:14px;color:{cfg['text']};font-weight:700;">
                    {icon}
                </span>
                <span style="font-size:13px;color:{cfg['text']};
                             font-weight:600;text-transform:uppercase;
                             letter-spacing:0.06em;">
                    {risk} Risk
                </span>
            </div>
            <div style="font-size:36px;font-weight:700;color:#e6edf3;
                        letter-spacing:-0.03em;line-height:1;">
                {prob:.1%}
            </div>
            <div style="font-size:13px;color:#8b949e;margin-top:6px;">
                fake news probability · {label}
            </div>
        </div>
        <div style="text-align:right;">
            <div style="
                width:100px;height:100px;
                border-radius:50%;
                background:conic-gradient(
                    {cfg['bar']} 0% {pct}%,
                    #21262d {pct}% 100%
                );
                display:flex;align-items:center;
                justify-content:center;
            ">
                <div style="
                    width:72px;height:72px;
                    border-radius:50%;
                    background:#0d1117;
                    display:flex;align-items:center;
                    justify-content:center;
                    font-size:18px;font-weight:700;
                    color:{cfg['text']};
                ">
                    {pct}%
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def model_card(label, risk, prob):
    cfg  = RISK_CONFIG[risk]
    icon = RISK_ICONS[risk]
    bar  = int(prob * 100)
    st.markdown(f"""
    <div style="
        background:#161b22;
        border:1px solid #21262d;
        border-radius:14px;
        padding:20px;
        height:100%;
    ">
        <div style="font-size:12px;font-weight:600;color:#8b949e;
                    text-transform:uppercase;letter-spacing:0.06em;
                    margin-bottom:14px;">
            {label}
        </div>
        <div style="font-size:28px;font-weight:700;color:#e6edf3;
                    letter-spacing:-0.02em;margin-bottom:6px;">
            {prob:.1%}
        </div>
        <div style="
            display:inline-flex;align-items:center;gap:6px;
            background:{cfg['badge_bg']};
            border:1px solid {cfg['border']};
            border-radius:12px;padding:3px 10px;
            margin-bottom:14px;
        ">
            <span style="font-size:11px;color:{cfg['text']};font-weight:600;">
                {icon} {risk}
            </span>
        </div>
        <div style="background:#21262d;border-radius:4px;height:4px;">
            <div style="background:{cfg['bar']};width:{bar}%;
                        height:100%;border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def disagreement_banner():
    st.markdown("""
    <div style="
        background:#1c1a00;
        border:1px solid #9e6a03;
        border-radius:12px;
        padding:14px 18px;
        margin-bottom:20px;
        display:flex;
        align-items:flex-start;
        gap:12px;
    ">
        <span style="font-size:18px;">⚡</span>
        <div>
            <div style="font-size:14px;font-weight:600;color:#e3b341;
                        margin-bottom:3px;">
                Model disagreement detected
            </div>
            <div style="font-size:13px;color:#9e6a03;line-height:1.5;">
                Models give conflicting predictions. This may indicate ambiguous language,
                political context, or dataset bias. Treat the result with caution.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def context_note(text):
    st.markdown(f"""
    <div style="
        background:#161b22;
        border-left:3px solid #30363d;
        border-radius:0 10px 10px 0;
        padding:12px 16px;
        font-size:13px;
        color:#8b949e;
        line-height:1.6;
        margin-top:10px;
        margin-bottom:20px;
    ">
        <span style="color:#58a6ff;font-weight:600;">ℹ</span>&nbsp; {text}
    </div>
    """, unsafe_allow_html=True)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:0 20px 20px;border-bottom:1px solid #21262d;margin-bottom:20px;">
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                <div style="width:44px;height:44px;border-radius:10px;
                            background:linear-gradient(135deg,#238636,#2ea043);
                            display:flex;align-items:center;
                            justify-content:center;font-size:24px;flex-shrink:0;">🛡</div>
                <div>
                    <div style="font-size:18px;font-weight:700;color:#e6edf3;
                                letter-spacing:-0.01em;">
                        Risk Intelligence
                    </div>
                    <div style="font-size:12px;color:#484f58;margin-top:2px;">
                        misinformation assessment
                    </div>
                </div>
            </div>
            <div style="font-size:12px;color:#8b949e;line-height:1.7;
                        padding:10px 12px;background:#161b22;
                        border-radius:8px;border:1px solid #21262d;">
                An AI-powered system that analyses news articles and claims
                using three independent models — BERT, XGBoost, and Logistic
                Regression — to assess misinformation risk and explain
                which words drive the prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<p style="font-size:11px;font-weight:600;color:#484f58;'
            'text-transform:uppercase;letter-spacing:0.08em;'
            'padding:0 16px;margin-bottom:8px;">Risk Scale</p>',
            unsafe_allow_html=True
        )

        for level, cfg in RISK_CONFIG.items():
            rng = {"Low":"0–30%","Moderate":"30–60%",
                   "High":"60–85%","Critical":"85–100%"}[level]
            st.markdown(f"""
            <div style="display:flex;align-items:center;padding:8px 16px;
                        margin-bottom:2px;border-radius:8px;">
                <div style="width:8px;height:8px;border-radius:50%;
                            background:{cfg['bar']};
                            margin-right:10px;flex-shrink:0;"></div>
                <span style="font-size:13px;color:{cfg['text']};
                             font-weight:500;flex:1;">{level}</span>
                <span style="font-size:12px;color:#484f58;">{rng}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin:20px 16px 0;
                    border-top:1px solid #21262d;padding-top:20px;">
            <div style="font-size:11px;font-weight:600;color:#484f58;
                        text-transform:uppercase;letter-spacing:0.08em;
                        margin-bottom:10px;">
                How to read results
            </div>
            <div style="font-size:12px;color:#8b949e;line-height:1.7;">
                <span style="color:#f85149;">■</span> Red words increase risk<br>
                <span style="color:#3fb950;">■</span> Green words lower risk<br><br>
                High confidence ≠ factual truth.<br>
                This system detects suspicious patterns, not verified facts.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    render_sidebar()

    st.markdown("""
    <div style="padding:32px 0 8px;">
        <div style="font-size:36px;font-weight:800;color:#e6edf3;
                    letter-spacing:-0.03em;line-height:1.1;margin-bottom:8px;">
            Misinformation Risk<br>
            <span style="
                background:linear-gradient(90deg,#58a6ff,#3fb950);
                -webkit-background-clip:text;
                -webkit-text-fill-color:transparent;
                background-clip:text;
            ">Intelligence System</span>
        </div>
        <div style="font-size:15px;color:#8b949e;max-width:520px;line-height:1.6;">
            Paste any news article or claim. Three independent models assess
            its misinformation risk and explain the decision.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Load models
    try:
        lr, xgb, tfidf, scaler, bert, explainer, tp, fb, rs = load_all_models()
    except Exception as e:
        st.markdown(f"""
        <div style="background:#2d0f0f;border:1px solid #da3633;
                    border-radius:12px;padding:20px;color:#f85149;
                    font-size:14px;">
            <strong>Failed to load models:</strong> {e}
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Input section
    st.markdown("""
    <div style="font-size:13px;font-weight:600;color:#8b949e;
                text-transform:uppercase;letter-spacing:0.06em;
                margin-bottom:8px;">
        Article or Claim
    </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "input",
        value=st.session_state.get("preload_text", ""),
        height=160,
        placeholder="Paste a news headline or full article here...",
        label_visibility="collapsed",
    )

    col_run, col_fake, col_real = st.columns([1.2, 1.4, 1.4])
    with col_run:
        run = st.button("🔍 Analyse", type="primary", use_container_width=True)
    with col_fake:
        if st.button("Load fake example", type="secondary", use_container_width=True):
            import random
            st.session_state["preload_text"] = random.choice(FAKE_EXAMPLES)
            st.rerun()
    with col_real:
        if st.button("Load real example", type="secondary", use_container_width=True):
            import random
            st.session_state["preload_text"] = random.choice(REAL_EXAMPLES)
            st.rerun()

    if not run or not text_input.strip():
        st.markdown("""
        <div style="text-align:center;padding:48px 0;color:#30363d;">
            <div style="font-size:40px;margin-bottom:12px;">🛡️</div>
            <div style="font-size:15px;">Enter text above and click Analyse</div>
        </div>
        """, unsafe_allow_html=True)
        return

    if len(text_input.split()) < 5:
        st.warning("Please enter at least a few sentences for meaningful analysis.")
        return

    with st.spinner("Running analysis across all models..."):
        results = predict(
            text_input, lr, xgb, tfidf, scaler,
            bert, explainer, tp, fb, rs
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Disagreement check across all models
    risks = {results["lr"]["risk"], results["xgb"]["risk"],
             results["bert"]["risk"]}
    if len(risks) > 1:
        disagreement_banner()

    # Primary result — ensemble
    section_title("Primary Assessment",
                  "Weighted ensemble — XGBoost 40% · Logistic Regression 35% · BERT 25%")

    # ── CHANGE 2: use ensemble_source as label ─────────────────────────────
    primary_result_card(
        results["ensemble"]["risk"],
        results["ensemble"]["prob"],
        label=results["ensemble_source"],
    )
    # ── END CHANGE 2 ──────────────────────────────────────────────────────

    is_political = any(w in text_input.lower() for w in POLITICAL_WORDS)

    # ── CHANGE 3: context note uses ensemble_risk ──────────────────────────
    ensemble_risk = results["ensemble"]["risk"]

    if ensemble_risk in ("High", "Critical") and is_political:
        context_note(
            "This text contains political or geopolitical terms. "
            "The model may reflect training data patterns rather than factual inaccuracy. "
            "Verify with a reliable source before drawing conclusions."
        )
    elif ensemble_risk in ("High", "Critical"):
        context_note(
            "High probability reflects learned patterns, not verified facts. "
            "This system is a risk indicator, not a fact-checker."
        )
    # ── END CHANGE 3 ──────────────────────────────────────────────────────

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model comparison
    section_title("Model Comparison",
                  "Three independent models — disagreement indicates uncertainty")
    c1, c2, c3 = st.columns(3)
    with c1:
        model_card("Logistic Regression · TF-IDF",
                   results["lr"]["risk"], results["lr"]["prob"])
    with c2:
        model_card("XGBoost · TF-IDF + Features",
                   results["xgb"]["risk"], results["xgb"]["prob"])
    with c3:
        model_card("BERT · Transformer",
                   results["bert"]["risk"], results["bert"]["prob"])

    st.markdown("<hr>", unsafe_allow_html=True)

    # SHAP explainability
    section_title("Explainability",
                  "XGBoost + SHAP — which words drive the prediction")

    top_words = get_top_shap_words(
        results["shap_values"], results["X_combined"], tfidf, fb, n=15
    )

    col_text, col_chart = st.columns([1, 1])

    with col_text:
        st.markdown("""
        <div style="font-size:12px;color:#8b949e;margin-bottom:10px;">
            <span style="color:#f85149;font-weight:600;">■ Red</span>
            &nbsp;pushes toward FAKE &nbsp;·&nbsp;
            <span style="color:#3fb950;font-weight:600;">■ Green</span>
            &nbsp;pushes toward REAL
        </div>
        """, unsafe_allow_html=True)

        highlighted = highlight_text(results["cleaned_text"], top_words)
        st.markdown(f"""
        <div style="
            background:#161b22;
            border:1px solid #21262d;
            border-radius:12px;
            padding:18px;
            font-size:14px;
            line-height:1.9;
            color:#c9d1d9;
        ">{highlighted}</div>
        """, unsafe_allow_html=True)

    with col_chart:
        fig = plot_shap_bar(top_words)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Contribution table
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Feature Contributions",
                  "Ranked by absolute SHAP value")

    contrib_df = pd.DataFrame(top_words, columns=["Feature", "SHAP Value"])
    contrib_df["Direction"] = contrib_df["SHAP Value"].apply(
        lambda v: "→ FAKE" if v > 0 else "→ REAL"
    )
    contrib_df["SHAP Value"] = contrib_df["SHAP Value"].round(4)
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # Domain warning
    if not is_political:
        st.markdown("<br>", unsafe_allow_html=True)
        st.warning(
            "⚠️ This text does not appear to be political news. "
            "Models were trained on the ISOT political news dataset — "
            "predictions on other domains may be unreliable."
        )


if __name__ == "__main__":
    main()