"""
app/components/explainability.py
================================
SHAP Feature Attribution, Token Highlighting & Saliency Visualization.
"""

import re
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_top_shap_words(shap_values, X_combined, tfidf, fb, n: int = 12):
    """Extracts top active features sorted by absolute attribution."""
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

    pairs = [(all_names[i], float(sv[i])) for i in range(min(len(all_names), len(sv))) if abs(sv[i]) > 1e-5]
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)
    return pairs[:n]


def highlight_text(text: str, top_words: list) -> str:
    """Highlights high-risk words in red and mitigating words in green."""
    result = text
    for word, contrib in top_words:
        if len(word) < 3:
            continue
        if contrib > 0.04:
            result = re.sub(
                rf'\b({re.escape(word)})\b',
                r'<mark style="background:rgba(248,81,73,0.22);color:#f85149;border-radius:4px;padding:2px 6px;border:1px solid rgba(248,81,73,0.4);font-weight:600;">\1</mark>',
                result, flags=re.IGNORECASE)
        elif contrib < -0.04:
            result = re.sub(
                rf'\b({re.escape(word)})\b',
                r'<mark style="background:rgba(63,185,80,0.22);color:#3fb950;border-radius:4px;padding:2px 6px;border:1px solid rgba(63,185,80,0.4);font-weight:600;">\1</mark>',
                result, flags=re.IGNORECASE)
    return result


def render_explainability_view(results: dict, tfidf, fb):
    """Renders the side-by-side text highlighter and SHAP attribution chart."""
    top_words = get_top_shap_words(results["shap_values"], results["X_combined"], tfidf, fb, n=10)
    col_text, col_chart = st.columns([1, 1])

    with col_text:
        st.markdown("""
        <div style="font-size:12px;color:#8b9bb4;margin-bottom:10px;">
            <span style="color:#f85149;font-weight:700;">■ Red Highlight</span> increases fake risk &nbsp;|&nbsp;
            <span style="color:#3fb950;font-weight:700;">■ Green Highlight</span> decreases fake risk
        </div>
        """, unsafe_allow_html=True)
        highlighted = highlight_text(results["cleaned_text"], top_words)
        st.markdown(f"""
        <div style="background:#0d1422;border:1px solid #1f2c44;border-radius:12px;padding:18px;font-size:14px;line-height:1.9;color:#c9d1d9;max-height:280px;overflow-y:auto;">
            {highlighted}
        </div>
        """, unsafe_allow_html=True)

    with col_chart:
        features = [w[0] for w in reversed(top_words[:8])]
        scores = [w[1] for w in reversed(top_words[:8])]
        colors = ["#f85149" if s >= 0 else "#3fb950" for s in scores]

        fig, ax = plt.subplots(figsize=(6, 3.6))
        fig.patch.set_facecolor("#0a0e17")
        ax.set_facecolor("#0d1422")
        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores, color=colors, height=0.55)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, color="#e6edf3", fontsize=9, fontweight="bold")
        ax.axvline(0, color="#2d3b55", linestyle="--")
        ax.tick_params(colors="#8b9bb4", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1f2c44")
        ax.set_title("XGBoost Linguistic Feature Attribution (SHAP)", color="#ffffff", fontsize=10, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
