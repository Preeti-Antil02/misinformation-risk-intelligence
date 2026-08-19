"""
app/components/input_section.py
===============================
Redesigned Input Panel with Language Detection & Modern Styling.
"""

import streamlit as st
from app.state import SessionStateManager
from risklens.url_reader import get_url_reader, DeepURLReader
from risklens.multilingual import detect_language, LANGUAGE_NAMES

def render_input_section():
    """
    Renders the modern input section with language auto-detection.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:13px; font-weight:600; color:#8b9bb4; margin-bottom:12px; display:flex; align-items:center; gap:8px;"><span>📋 Paste text, URL, or forward a WhatsApp message</span></div>', unsafe_allow_html=True)

    user_text = st.text_area(
        label="Input",
        value=st.session_state.get("text_input", ""),
        height=120,
        placeholder="Type or paste any statement to analyze...",
        label_visibility="collapsed"
    )

    # Language Detection Badge
    lang_code = detect_language(user_text) if user_text.strip() else "en"
    lang_name = LANGUAGE_NAMES.get(lang_code, "English")

    col_actions, col_btn = st.columns([2, 1])

    with col_actions:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:12px; height:100%;">
            <div style="background:rgba(88, 166, 255, 0.1); color:#58a6ff; border:1px solid rgba(88, 166, 255, 0.2); padding:6px 12px; border-radius:8px; font-size:12px; font-weight:700;">
                {lang_name} detected
            </div>
            <div style="font-size:13px; color:#8b9bb4;">or</div>
            <div style="color:#3fb950; font-size:13px; font-weight:600; cursor:pointer; display:flex; align-items:center; gap:6px;">
                🖼️ Upload screenshot
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_btn:
        analyze_btn = st.button("Verify now", type="primary", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Scraped Article Metadata Banner (if any)
    scraped_meta = st.session_state.get("scraped_meta", None)
    if scraped_meta and scraped_meta.get("success"):
        st.markdown(f"""
        <div style="background:rgba(46,160,67,0.12); border:1px solid #2ea043; border-radius:12px; padding:12px 18px; margin-bottom:16px; font-size:13px; color:#3fb950;">
            <strong>✓ Article Auto-Scraped:</strong> "{scraped_meta['title'][:70]}..."
            <div style="color:#8b9bb4; font-size:11px; margin-top:2px;">{scraped_meta['domain']} · {scraped_meta['word_count']} words</div>
        </div>
        """, unsafe_allow_html=True)

    trigger = analyze_btn or (user_text != st.session_state.get("last_analyzed_text", ""))
    # For now, return URL as empty if not explicitly handled in this minimal view
    return user_text, "", trigger
