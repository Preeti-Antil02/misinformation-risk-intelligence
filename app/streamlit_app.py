"""
Misinformation Risk Intelligence System
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
    page_title="RiskLens · Misinformation Intelligence",
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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&family=Playfair+Display:ital,wght@1,700;1,800&display=swap');

div.block-container {
    padding-top: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080c12 0%, #0a0f18 100%) !important;
    border-right: 1px solid #1a2235;
    padding-top: 0 !important;
}

.stApp { background: #080c12; }

/* Animated background grid */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        linear-gradient(rgba(88,166,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(88,166,255,0.03) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

textarea {
    background: #0d1420 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 14px !important;
    color: #e6edf3 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    padding: 16px !important;
    transition: all 0.25s !important;
}
textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.08), 0 0 20px rgba(88,166,255,0.06) !important;
}
textarea::placeholder {
    color: #3a4a5c !important;
}

div[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #1a6b2e 0%, #238636 50%, #2ea043 100%) !important;
    color: #ffffff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    padding: 12px 32px !important;
    letter-spacing: 0.03em !important;
    transition: all 0.25s !important;
    box-shadow: 0 0 20px rgba(46,160,67,0.15) !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #238636 0%, #2ea043 50%, #3fb950 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(46,160,67,0.3) !important;
}

div[data-testid="stButton"] button[kind="secondary"] {
    background: #0d1420 !important;
    color: #6e7f96 !important;
    border: 1px solid #1e2d45 !important;
    border-radius: 12px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 13px !important;
    transition: all 0.25s !important;
}
div[data-testid="stButton"] button[kind="secondary"]:hover {
    background: #131d2e !important;
    color: #e6edf3 !important;
    border-color: #58a6ff !important;
    box-shadow: 0 0 12px rgba(88,166,255,0.1) !important;
}

hr { border-color: #1a2235 !important; margin: 28px 0 !important; }

div[data-testid="stDataFrame"] {
    border-radius: 14px;
    border: 1px solid #1a2235;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    background: #1a1600 !important;
    border: 1px solid #9e6a03 !important;
    border-radius: 14px !important;
    color: #d29922 !important;
}

div[data-testid="stSpinner"] p { color: #6e7f96 !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Risk config
# -------------------------------------------------------
RISK_CONFIG = {
    "Low": {
        "bg": "linear-gradient(135deg, #071a0f 0%, #0a2415 100%)",
        "border": "#2ea043",
        "text": "#3fb950",
        "badge_bg": "#0a2415",
        "badge_text": "#3fb950",
        "glow": "rgba(63,185,80,0.18)",
        "bar": "#3fb950",
    },
    "Moderate": {
        "bg": "linear-gradient(135deg, #161200 0%, #211800 100%)",
        "border": "#d29922",
        "text": "#e3b341",
        "badge_bg": "#211800",
        "badge_text": "#e3b341",
        "glow": "rgba(227,179,65,0.18)",
        "bar": "#e3b341",
    },
    "High": {
        "bg": "linear-gradient(135deg, #221200 0%, #2d1800 100%)",
        "border": "#db6d28",
        "text": "#f0883e",
        "badge_bg": "#2d1800",
        "badge_text": "#f0883e",
        "glow": "rgba(240,136,62,0.18)",
        "bar": "#f0883e",
    },
    "Critical": {
        "bg": "linear-gradient(135deg, #220a0a 0%, #2d0f0f 100%)",
        "border": "#da3633",
        "text": "#f85149",
        "badge_bg": "#2d0f0f",
        "badge_text": "#f85149",
        "glow": "rgba(248,81,73,0.22)",
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
    # Short
    "Scientists confirm drinking hot water every hour can completely cure cancer within days",
    "Government secretly approves mind-control chips in COVID booster shots, leaked documents reveal",
    "NASA discovers hidden city on Mars but refuses to release images to the public",
    "New study proves smartphones can read your thoughts using hidden sensors",
    "Eating only bananas for 3 days can detox your body and reverse aging, experts claim",

    # Long
    "A viral report circulating on social media claims that a group of international scientists has discovered that drinking hot water every hour can eliminate cancer cells completely within 72 hours, a finding that pharmaceutical companies are allegedly trying to suppress to protect chemotherapy profits.",

    "According to an anonymous government insider, recent COVID booster vaccines contain nanochips designed to monitor and influence human behavior, with several leaked documents suggesting a global surveillance program is already underway.",

    "Multiple online sources are claiming that NASA has found an ancient alien city buried beneath the surface of Mars, but the agency is deliberately withholding photographic evidence to avoid public panic and maintain control over extraterrestrial discoveries.",

    "A widely shared article suggests that modern smartphones are equipped with advanced sensors capable of interpreting brain signals, allowing companies to read users' thoughts and predict decisions without their consent.",

    "Health influencers are promoting a new 'banana detox' diet, claiming that consuming only bananas for three consecutive days can flush toxins, repair internal organs, and significantly slow down the aging process, despite lack of scientific evidence."
]

REAL_EXAMPLES = [
    # Short (Political-focused)
    "Government passes new data protection bill to strengthen user privacy laws",
    "Election Commission announces revised guidelines ahead of upcoming general elections",
    "Supreme Court hears petitions challenging recent electoral bond scheme",
    "Parliament debates new cybersecurity policy amid rising digital threats",
    "Finance Minister unveils budget with increased allocation for infrastructure and defense",

    # Long (Political-focused)
    "The government has passed a new data protection bill aimed at strengthening user privacy and regulating how companies collect and process personal data, with provisions for stricter penalties and oversight mechanisms.",

    "The Supreme Court is currently hearing multiple petitions challenging the legality and transparency of the electoral bond scheme, with petitioners arguing that it undermines accountability in political funding.",

    "Parliament witnessed intense debates over a proposed cybersecurity policy designed to address increasing digital threats, with lawmakers discussing measures for data security, surveillance, and national defense.",

    "The Finance Minister presented the annual budget outlining increased spending on infrastructure, defense, and public welfare programs, while also addressing concerns over fiscal deficit and economic growth."
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

    # Weights: XGBoost 40%, LR 20%, BERT 40%
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
    fig.patch.set_facecolor("#0d1420")
    ax.set_facecolor("#0d1420")

    ax.barh(words[::-1], values[::-1],
            color=colors[::-1], edgecolor="none", height=0.6)
    ax.axvline(0, color="#1e2d45", linewidth=1.2)
    ax.tick_params(colors="#6e7f96", labelsize=10)
    ax.set_xlabel("SHAP contribution", color="#6e7f96", fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2235")

    fake_p = mpatches.Patch(color="#f85149", label="→ FAKE")
    real_p = mpatches.Patch(color="#58a6ff", label="→ REAL")
    ax.legend(handles=[fake_p, real_p], loc="lower right",
              fontsize=9, facecolor="#0d1420",
              edgecolor="#1e2d45", labelcolor="#6e7f96")
    plt.tight_layout()
    return fig

# -------------------------------------------------------
# UI components
# -------------------------------------------------------
def section_title(text, subtitle=None):
    sub_html = (
        '<div style="font-size:13px;color:#6e7f96;margin-top:4px;'
        'font-family:Space Grotesk,sans-serif;">' + subtitle + '</div>'
    ) if subtitle else ""
    st.markdown(
        '<div style="margin-bottom:16px;">'
        '<div style="font-size:19px;font-weight:700;color:#e6edf3;'
        'letter-spacing:-0.02em;font-family:Syne,sans-serif;">' + text + '</div>'
        + sub_html +
        '</div>',
        unsafe_allow_html=True,
    )


def primary_result_card(risk, prob, label="weighted ensemble"):
    cfg  = RISK_CONFIG[risk]
    icon = RISK_ICONS[risk]
    pct  = int(prob * 100)
    st.markdown(
        '<div style="'
        'background:' + cfg["bg"] + ';'
        'border:1px solid ' + cfg["border"] + ';'
        'border-radius:18px;'
        'padding:28px 32px;'
        'display:flex;'
        'align-items:center;'
        'justify-content:space-between;'
        'margin-bottom:12px;'
        'box-shadow:0 0 48px ' + cfg["glow"] + ';'
        '">'
        '<div>'
        '<div style="'
        'display:inline-flex;align-items:center;gap:8px;'
        'background:' + cfg["badge_bg"] + ';'
        'border:1px solid ' + cfg["border"] + ';'
        'border-radius:20px;padding:4px 14px;'
        'margin-bottom:12px;">'
        '<span style="font-size:14px;color:' + cfg["text"] + ';font-weight:700;">' + icon + '</span>'
        '<span style="font-size:12px;color:' + cfg["text"] + ';'
        'font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.08em;font-family:JetBrains Mono,monospace;">' + risk + ' Risk</span>'
        '</div>'
        '<div style="font-size:40px;font-weight:400;color:#e6edf3;'
        'letter-spacing:-0.01em;line-height:1;font-family:Space Grotesk,sans-serif;">'
        + f"{prob:.1%}" +
        '</div>'
        '<div style="font-size:13px;color:#6e7f96;margin-top:8px;'
        'font-family:Space Grotesk,sans-serif;">fake news probability · ' + label + '</div>'
        '</div>'
        '<div style="text-align:right;">'
        '<div style="'
        'width:108px;height:108px;border-radius:50%;'
        'background:conic-gradient(' + cfg["bar"] + ' 0% ' + str(pct) + '%, #1a2235 ' + str(pct) + '% 100%);'
        'display:flex;align-items:center;justify-content:center;'
        'box-shadow:0 0 24px ' + cfg["glow"] + ';">'
        '<div style="'
        'width:78px;height:78px;border-radius:50%;background:#080c12;'
        'display:flex;align-items:center;justify-content:center;'
        'font-size:17px;font-weight:400;color:' + cfg["text"] + ';'
        'font-family:Space Grotesk,sans-serif;">' + str(pct) + '%</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def model_card(label, risk, prob):
    cfg  = RISK_CONFIG[risk]
    icon = RISK_ICONS[risk]
    bar  = int(prob * 100)
    st.markdown(
        '<div style="'
        'background:#0d1420;border:1px solid #1a2235;border-radius:16px;'
        'padding:22px;height:100%;transition:border-color 0.2s;">'
        '<div style="font-size:11px;font-weight:600;color:#6e7f96;'
        'text-transform:uppercase;letter-spacing:0.08em;'
        'margin-bottom:14px;font-family:JetBrains Mono,monospace;">' + label + '</div>'
        '<div style="font-size:30px;font-weight:400;color:#e6edf3;'
        'letter-spacing:-0.01em;margin-bottom:8px;font-family:Space Grotesk,sans-serif;">'
        + f"{prob:.1%}" + '</div>'
        '<div style="display:inline-flex;align-items:center;gap:6px;'
        'background:' + cfg["badge_bg"] + ';border:1px solid ' + cfg["border"] + ';'
        'border-radius:12px;padding:3px 10px;margin-bottom:16px;">'
        '<span style="font-size:11px;color:' + cfg["text"] + ';font-weight:600;'
        'font-family:JetBrains Mono,monospace;">' + icon + ' ' + risk + '</span>'
        '</div>'
        '<div style="background:#1a2235;border-radius:4px;height:3px;">'
        '<div style="background:' + cfg["bar"] + ';width:' + str(bar) + '%;'
        'height:100%;border-radius:4px;box-shadow:0 0 8px ' + cfg["glow"] + ';"></div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def disagreement_banner():
    st.markdown(
        '<div style="'
        'background:#151100;border:1px solid #9e6a03;border-radius:14px;'
        'padding:14px 18px;margin-bottom:20px;'
        'display:flex;align-items:flex-start;gap:12px;">'
        '<span style="font-size:18px;">⚡</span>'
        '<div>'
        '<div style="font-size:14px;font-weight:700;color:#e3b341;'
        'margin-bottom:3px;font-family:Syne,sans-serif;">Model disagreement detected</div>'
        '<div style="font-size:13px;color:#9e6a03;line-height:1.5;'
        'font-family:Space Grotesk,sans-serif;">'
        'Models give conflicting predictions. This may indicate ambiguous language, '
        'political context, or dataset bias. Treat the result with caution.'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def context_note(text):
    st.markdown(
        '<div style="'
        'background:#0d1420;border-left:3px solid #1e2d45;'
        'border-radius:0 12px 12px 0;padding:12px 16px;'
        'font-size:13px;color:#6e7f96;line-height:1.6;'
        'margin-top:10px;margin-bottom:20px;'
        'font-family:Space Grotesk,sans-serif;">'
        '<span style="color:#58a6ff;font-weight:600;">ℹ</span>&nbsp; ' + text +
        '</div>',
        unsafe_allow_html=True,
    )

# -------------------------------------------------------
# Sidebar — shield emoji bare (no box), RiskLens narrower
# -------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="padding:24px 20px 20px;border-bottom:1px solid #1a2235;margin-bottom:20px;">'
            '<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">'
            # CHANGED: bare emoji, no box wrapper
            '<span style="font-size:26px;flex-shrink:0;">🛡️</span>'
            '<div>'
            '<div style="'
            # CHANGED: font-size reduced from 22px → 17px to narrow the name
            'font-size:17px;font-weight:800;'
            'letter-spacing:-0.02em;'
            "font-family:'Syne',sans-serif;"
            'line-height:1;'
            'background:linear-gradient(90deg,#58a6ff 0%,#3fb950 100%);'
            '-webkit-background-clip:text;'
            '-webkit-text-fill-color:transparent;'
            'background-clip:text;">RiskLens</div>'
            '<div style="font-size:11px;color:#3a5070;margin-top:3px;'
            "font-family:'JetBrains Mono',monospace;"
            'letter-spacing:0.04em;">misinformation intelligence</div>'
            '</div>'
            '</div>'
            '<div style="font-size:12px;color:#6e7f96;line-height:1.7;'
            'padding:10px 12px;background:#0d1420;'
            'border-radius:10px;border:1px solid #1a2235;'
            "font-family:Space Grotesk,sans-serif;\">"
            'An AI-powered system that analyses news articles and claims '
            'using three independent models — BERT, XGBoost, and Logistic '
            'Regression — to assess misinformation risk and explain '
            'which words drive the prediction.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="font-size:10px;font-weight:600;color:#3a5070;'
            'text-transform:uppercase;letter-spacing:0.1em;'
            'padding:0 16px;margin-bottom:8px;'
            "font-family:JetBrains Mono,monospace;\">Risk Scale</p>",
            unsafe_allow_html=True,
        )

        for level, cfg in RISK_CONFIG.items():
            rng = {"Low": "0–30%", "Moderate": "30–60%",
                   "High": "60–85%", "Critical": "85–100%"}[level]
            st.markdown(
                '<div style="display:flex;align-items:center;padding:8px 16px;'
                'margin-bottom:2px;border-radius:8px;">'
                '<div style="width:7px;height:7px;border-radius:50%;'
                'background:' + cfg["bar"] + ';margin-right:10px;flex-shrink:0;'
                'box-shadow:0 0 6px ' + cfg["glow"] + ';"></div>'
                '<span style="font-size:13px;color:' + cfg["text"] + ';'
                'font-weight:500;flex:1;font-family:Space Grotesk,sans-serif;">' + level + '</span>'
                '<span style="font-size:11px;color:#3a5070;'
                'font-family:JetBrains Mono,monospace;">' + rng + '</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div style="margin:20px 16px 0;border-top:1px solid #1a2235;padding-top:20px;">'
            '<div style="font-size:10px;font-weight:600;color:#3a5070;'
            'text-transform:uppercase;letter-spacing:0.1em;'
            'margin-bottom:10px;font-family:JetBrains Mono,monospace;">How to read results</div>'
            '<div style="font-size:12px;color:#6e7f96;line-height:1.8;'
            'font-family:Space Grotesk,sans-serif;">'
            '<span style="color:#f85149;">■</span> Red words increase risk<br>'
            '<span style="color:#3fb950;">■</span> Green words lower risk<br><br>'
            'High confidence ≠ factual truth.<br>'
            'This system detects suspicious patterns, not verified facts.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

# -------------------------------------------------------
# Results renderer (extracted so it can be called from
# both the "just analysed" path and the "reload from
# session_state" path)
# -------------------------------------------------------
def render_results(results, text_input, tfidf, fb):
    st.markdown("<hr>", unsafe_allow_html=True)

    # Disagreement check across all models
    risks = {results["lr"]["risk"], results["xgb"]["risk"],
             results["bert"]["risk"]}
    if len(risks) > 1:
        disagreement_banner()

    # Primary result — ensemble
    section_title(
        "Primary Assessment",
        "Weighted ensemble — XGBoost 40% · BERT 40% · Logistic Regression 20%",
    )

    primary_result_card(
        results["ensemble"]["risk"],
        results["ensemble"]["prob"],
        label=results["ensemble_source"],
    )

    is_political = any(w in text_input.lower() for w in POLITICAL_WORDS)
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

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model comparison
    section_title(
        "Model Comparison",
        "Three independent models — disagreement indicates uncertainty",
    )
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
    section_title(
        "Explainability",
        "XGBoost + SHAP — which words drive the prediction",
    )

    top_words = get_top_shap_words(
        results["shap_values"], results["X_combined"], tfidf, fb, n=15
    )

    col_text, col_chart = st.columns([1, 1])

    with col_text:
        st.markdown(
            '<div style="font-size:12px;color:#6e7f96;margin-bottom:10px;'
            'font-family:Space Grotesk,sans-serif;">'
            '<span style="color:#f85149;font-weight:600;">■ Red</span>'
            '&nbsp;pushes toward FAKE &nbsp;·&nbsp;'
            '<span style="color:#3fb950;font-weight:600;">■ Green</span>'
            '&nbsp;pushes toward REAL'
            '</div>',
            unsafe_allow_html=True,
        )

        highlighted = highlight_text(results["cleaned_text"], top_words)
        st.markdown(
            '<div style="'
            'background:#0d1420;border:1px solid #1a2235;border-radius:14px;'
            'padding:18px;font-size:14px;line-height:1.9;color:#c9d1d9;'
            'font-family:Space Grotesk,sans-serif;">' + highlighted + '</div>',
            unsafe_allow_html=True,
        )

    with col_chart:
        fig = plot_shap_bar(top_words)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # Contribution table
    st.markdown("<br>", unsafe_allow_html=True)
    section_title("Feature Contributions", "Ranked by absolute SHAP value")

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

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    # ── Initialise session state keys ────────────────────────────────────
    if "preload_text" not in st.session_state:
        st.session_state["preload_text"] = ""
    if "last_results" not in st.session_state:
        st.session_state["last_results"] = None
    if "last_input" not in st.session_state:
        st.session_state["last_input"] = ""

    render_sidebar()

    # ── TOP NAV BAR — shield emoji bare (no box), RiskLens narrower ──────
    st.markdown(
        '<div style="'
        'display:flex;align-items:center;gap:12px;'
        'padding:18px 0 14px 0;'
        'border-bottom:1px solid #1a2235;margin-bottom:0;">'
        # CHANGED: bare emoji, no box wrapper
        '<span style="font-size:22px;flex-shrink:0;">🛡️</span>'
        '<div style="'
        # CHANGED: font-size reduced from 26px → 20px to narrow the name
        'font-size:20px;font-weight:800;'
        'letter-spacing:-0.03em;'
        "font-family:'Syne',sans-serif;"
        'line-height:1;'
        'background:linear-gradient(90deg,#58a6ff 0%,#3fb950 100%);'
        '-webkit-background-clip:text;'
        '-webkit-text-fill-color:transparent;'
        'background-clip:text;">RiskLens</div>'
        '<div style="'
        'margin-left:4px;font-size:11px;color:#3a5070;'
        "font-family:'JetBrains Mono',monospace;"
        'letter-spacing:0.05em;padding-top:4px;">'
        'v1.0 · misinformation intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── HERO SECTION ──────────────────────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:56px 20px 40px;position:relative;">'

        # Decorative glow blob
        '<div style="'
        'position:absolute;top:20px;left:50%;transform:translateX(-50%);'
        'width:500px;height:200px;'
        'background:radial-gradient(ellipse at center,'
        'rgba(88,166,255,0.06) 0%,transparent 70%);'
        'pointer-events:none;"></div>'

        # Eyebrow label
        '<div style="'
        'display:inline-flex;align-items:center;gap:8px;'
        'background:#0d1420;border:1px solid #1e2d45;border-radius:20px;'
        'padding:5px 16px;margin-bottom:24px;">'
        '<span style="'
        'width:6px;height:6px;border-radius:50%;background:#58a6ff;'
        'display:inline-block;box-shadow:0 0 8px rgba(88,166,255,0.6);"></span>'
        '<span style="'
        'font-size:11px;color:#58a6ff;font-weight:600;'
        'letter-spacing:0.1em;text-transform:uppercase;'
        "font-family:'JetBrains Mono',monospace;\">AI-Powered News Analysis</span>"
        '</div>'

        '<div style="max-width:600px;margin:0 auto;">'

        '<div style="'
        'font-size:52px;font-weight:800;color:#e6edf3;'
        'letter-spacing:-0.02em;line-height:1.1;margin-bottom:6px;'
        "font-family:'Playfair Display',Georgia,serif;"
        'font-style:italic;">Misinformation Risk</div>'

        '<div style="'
        'font-size:52px;font-weight:800;'
        'letter-spacing:-0.02em;line-height:1.1;margin-bottom:24px;'
        "font-family:'Playfair Display',Georgia,serif;"
        'font-style:italic;'
        'background:linear-gradient(90deg,#58a6ff 0%,#3fb950 60%,#58a6ff 100%);'
        'background-size:200% auto;'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;'
        'background-clip:text;">Intelligence System</div>'

        '</div>'

        # Subtitle
        '<div style="'
        'font-size:16px;color:#c9d1d9;max-width:480px;margin:0 auto;'
        'line-height:1.7;font-family:Space Grotesk,sans-serif;">'
        'Paste any news article or claim. Three independent models assess '
        'its misinformation risk and explain the decision.'
        '</div>'

        # Model badges
        '<div style="'
        'display:flex;justify-content:center;gap:10px;'
        'margin-top:24px;flex-wrap:wrap;">'
        '<div style="background:#0d1420;border:1px solid #1e2d45;'
        'border-radius:8px;padding:5px 14px;font-size:11px;color:#a0aec0;'
        "font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;\">BERT · Transformer</div>"
        '<div style="background:#0d1420;border:1px solid #1e2d45;'
        'border-radius:8px;padding:5px 14px;font-size:11px;color:#a0aec0;'
        "font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;\">XGBoost · TF-IDF</div>"
        '<div style="background:#0d1420;border:1px solid #1e2d45;'
        'border-radius:8px;padding:5px 14px;font-size:11px;color:#a0aec0;'
        "font-family:'JetBrains Mono',monospace;letter-spacing:0.04em;\">Logistic Regression</div>"
        '</div>'

        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Load models
    try:
        lr, xgb, tfidf, scaler, bert, explainer, tp, fb, rs = load_all_models()
    except Exception as e:
        st.markdown(
            '<div style="background:#1a0808;border:1px solid #da3633;'
            'border-radius:14px;padding:20px;color:#f85149;'
            'font-size:14px;font-family:Space Grotesk,sans-serif;">'
            '<strong>Failed to load models:</strong> ' + str(e) +
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown(
        '<div style="font-size:11px;font-weight:700;color:#e6edf3;'
        'text-transform:uppercase;letter-spacing:0.12em;'
        "margin-bottom:8px;font-family:'JetBrains Mono',monospace;\">Article or Claim</div>",
        unsafe_allow_html=True,
    )

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
            st.session_state["last_results"] = None  # clear stale results
            st.session_state["last_input"] = ""
            st.rerun()
    with col_real:
        if st.button("Load real example", type="secondary", use_container_width=True):
            import random
            st.session_state["preload_text"] = random.choice(REAL_EXAMPLES)
            st.session_state["last_results"] = None  # clear stale results
            st.session_state["last_input"] = ""
            st.rerun()

    # ── Run analysis and persist results in session_state ─────────────────
    if run:
        if not text_input.strip():
            pass  # fall through to the empty-state placeholder below
        elif len(text_input.split()) < 5:
            st.warning("Please enter at least a few sentences for meaningful analysis.")
        else:
            with st.spinner("Running analysis across all models..."):
                results = predict(
                    text_input, lr, xgb, tfidf, scaler,
                    bert, explainer, tp, fb, rs
                )
            # Persist so results survive reruns
            st.session_state["last_results"] = results
            st.session_state["last_input"]   = text_input

    # ── Display results (from session_state so they survive reruns) ───────
    if st.session_state["last_results"] is not None:
        render_results(
            st.session_state["last_results"],
            st.session_state["last_input"],
            tfidf,
            fb,
        )
    else:
        st.markdown(
            '<div style="text-align:center;padding:52px 0;color:#1a2235;">'
            '<div style="font-size:42px;margin-bottom:14px;'
            'filter:drop-shadow(0 0 12px rgba(88,166,255,0.2));">🛡️</div>'
            '<div style="font-size:15px;color:#3a5070;'
            'font-family:Space Grotesk,sans-serif;">Enter text above and click Analyse</div>'
            '</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()