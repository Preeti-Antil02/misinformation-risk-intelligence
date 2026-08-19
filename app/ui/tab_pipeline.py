"""
app/ui/tab_pipeline.py
======================
Interactive Pipeline Architecture & Execution Trace Dashboard Tab for RiskLens.
Uses native Streamlit components for the info sections to ensure reliable rendering.
"""

import streamlit as st
import streamlit.components.v1 as components
from app.ui.components.pipeline_visualizer import get_pipeline_html


def render_tab_pipeline():
    """Renders the comprehensive interactive pipeline visualization tab."""

    # ── PAGE HEADER ──
    st.markdown(
        "<div class='page-head'>"
        "<h1>⚡ How RiskLens Works</h1>"
        "<p>Click any block in the diagram to learn what it does — no technical knowledge needed.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── MAIN INTERACTIVE PIPELINE ──
    theme = st.session_state.get("theme", "dark")
    html_code = get_pipeline_html(theme=theme)
    components.html(html_code, height=1020, scrolling=True)

    st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2  — Plain-English system summary
    # Uses only native Streamlit so nothing can break
    # ═══════════════════════════════════════════════════════════════
    st.divider()
    st.markdown("### 📋 System Summary")
    st.caption(
        "Everything below explains what RiskLens does — in plain language, "
        "no jargon required."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 How the Risk Score Is Calculated")
        st.info(
            "**Step 1 — Four AI detectors** each give a suspicion score (0–100%).\n\n"
            "**Step 2 — A voting system** combines their scores, giving more weight to "
            "whichever AI has historically been most accurate.\n\n"
            "**Step 3 — Website check** (only if you shared a link): trusted news outlets "
            "reduce the score; known conspiracy or satire sites raise it.\n\n"
            "**Final score** = 75% AI suspicion score + 25% website distrust score.\n\n"
            "📌 Scores below 25% → **Low Risk**  "
            "· 25–60% → **Medium**  "
            "· 60–85% → **High**  "
            "· 85%+ → **Critical**"
        )

        st.markdown("#### 🔄 How the AI Gets Smarter Over Time")
        st.success(
            "Every time you tap **👍 Correct** or **👎 Wrong** on a result:\n\n"
            "1. Your vote is saved with the original message and score.\n"
            "2. Cases where the AI was most wrong get the highest priority.\n"
            "3. **Every night at 2AM**, if 500+ corrections are collected, "
            "the AI re-trains itself on those corrections.\n"
            "4. The new model only goes live if it scores **better** than the old one — "
            "it can never get worse from user feedback.\n\n"
            "🌙 This is called the **Human-in-the-Loop Flywheel** — the more you use it, "
            "the smarter it gets."
        )

    with col2:
        st.markdown("#### 🛡️ What Happens When Things Go Wrong")
        st.warning(
            "RiskLens has automatic backups at every step:\n\n"
            "**📸 Can't read the image?**  \n"
            "Tries 3 times with different settings before giving up.\n\n"
            "**🌐 Web search times out?**  \n"
            "Falls back to a built-in library of 7 famous debunked viral hoaxes, "
            "then marks as inconclusive rather than guessing.\n\n"
            "**🇮🇳 Indian language message?**  \n"
            "Bypasses the English AI entirely and routes directly to a specialist AI "
            "trained only on Hindi, Tamil, Telugu, Bengali, Marathi, and Gujarati.\n\n"
            "**🤔 No web evidence found?**  \n"
            "The system explicitly says **'Inconclusive'** — it never fabricates a verdict."
        )

        st.markdown("#### 📊 Verified Performance Numbers")
        m1, m2 = st.columns(2)
        m1.metric("Ensemble Accuracy", "91.9%", "vs 85% target", delta_color="normal")
        m2.metric("Calibration Error", "0.91%", "was 1.11% before calibration", delta_color="inverse")
        m3, m4 = st.columns(2)
        m3.metric("Hindi AI Accuracy", "100%", "F1 = 1.0000")
        m4.metric("AUC Score", "0.9744", "Near-perfect discrimination")

    st.divider()

    # ── SCENARIO GUIDE ──
    st.markdown("### 🗺️ What Happens for Each Type of Message")

    tab_text, tab_url, tab_img, tab_retrain = st.tabs([
        "💬 Plain Text",
        "🌐 News Link",
        "📸 Screenshot",
        "🔄 Nightly Learning"
    ])

    with tab_text:
        st.markdown(
            "When you send a **plain text message**, here's the exact journey:\n\n"
            "1. 📩 **Received** by Telegram chatbot\n"
            "2. 🌐 **Language detected** — English, Hindi, etc.\n"
            "3. 📊 **11 writing signals measured** — capital letters, exclamation marks, emotional words\n"
            "4. 🤖 **4 AI detectors run in parallel** — Word Pattern AI · Pattern Trees AI · "
            "Deep Context AI · Step-by-Step Reasoning AI\n"
            "5. 🗳️ **All 4 votes combined** by a meta-AI\n"
            "6. 🎯 **Risk score calibrated** and tier assigned\n"
            "7. 🔎 **Key claim extracted** from your message\n"
            "8. 🌍 **Web search run** — Google Fact Check + live search\n"
            "9. ⚖️ **Verdict written** from evidence\n"
            "10. 🔬 **Top 3 red flags explained** in plain English\n"
            "11. 📤 **Full result card sent** to you\n"
            "12. 📝 **Your 👍/👎 vote recorded** for overnight retraining"
        )

    with tab_url:
        st.markdown(
            "When you share a **news link or URL**, two extra steps happen:\n\n"
            "- 🏛️ The **website's reputation** is looked up in a database of 100+ news outlets\n"
            "- The website trust score (25%) is **mixed into** the final risk score\n"
            "- A trusted outlet (BBC, Reuters) reduces risk; a known conspiracy site raises it\n\n"
            "Everything else is the same as a plain text message."
        )

    with tab_img:
        st.markdown(
            "When you send a **screenshot or photo**, two extra steps happen first:\n\n"
            "1. 🧹 **Image cleaned** — noise removed, text sharpened using computer vision\n"
            "2. 👁️ **Text read from image** — two OCR engines try up to 3 times each\n"
            "3. Once text is extracted, it continues normally through language detection\n\n"
            "**Indian language in the screenshot?**  \n"
            "Automatically routed to the specialist Indian language AI (MuRIL) "
            "which bypasses all English AI detectors."
        )

    with tab_retrain:
        st.markdown(
            "**Every night at 2AM**, the following automatic learning cycle runs:\n\n"
            "1. **Check:** Are there 500+ user corrections collected today?\n"
            "2. **Retrain:** The AI meta-model is retrained on those corrections\n"
            "3. **Test:** The new model is evaluated on a held-out test set\n"
            "4. **Gate:** New model only goes live if it scores higher than the current one\n"
            "5. **Deploy:** The updated model files are overwritten — live from next morning\n\n"
            "🌙 This runs automatically in the background, with no manual intervention needed.\n\n"
            "📌 *The amber dashed arrows in the pipeline diagram above show this learning loop visually.*"
        )
