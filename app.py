"""
app.py
======
Root Entry Point for RiskLens v2.1.0.
Enterprise Intelligence Platform with state-of-the-art multi-modal verification suite.
"""

import os
import sys
import textwrap
from pathlib import Path

# Force the project root into sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Initialize Hardened Logging
from risklens.logging_config import setup_logging
setup_logging()

import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="RiskLens — Enterprise Misinformation Intelligence",
    page_icon="https://raw.githubusercontent.com/feathericons/feather/master/icons/shield.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.ui.utils import render_html
from app.ui.theme import inject_theme
from app.ui.sidebar import render_sidebar
from app.ui.tab_verify import render_tab_verify
from app.ui.tab_analytics import render_tab_analytics
from app.ui.tab_pipeline import render_tab_pipeline
from app.ui.tab_history import render_tab_history
from app.ui.tab_settings import render_tab_settings


def main():
    inject_theme()
    
    from risklens import __version__ as APP_VERSION
    
    # 1. UNIFIED SINGLE TOP ROW: BRAND (LEFT) + THEME TOGGLE (RIGHT)
    # The centered pill navigation tabs are seamlessly aligned onto this exact same row on desktop
    brand_col1, brand_col2 = st.columns([7, 3])
    
    with brand_col1:
        render_html(f"""
            <div class="header-row-wrap">
                <div class="brand-box">
                    <div class="eyebrow">Enterprise intelligence platform</div>
                    <div class="brand-row">
                        <span class="brand-name">RiskLens</span>
                        <span class="badge">v{APP_VERSION}</span>
                    </div>
                </div>
            </div>
        """)

    with brand_col2:
        bot_username = os.getenv("TELEGRAM_BOT_USERNAME", "RiskLensIntelligenceBot").strip().lstrip("@")
        web_url = f"https://web.telegram.org/k/#@{bot_username}"
        is_dark = st.session_state.get("theme", "dark") == "dark"
        t_label = "Dark Mode" if is_dark else "Light Mode"

        c_tg, c_thm = st.columns([1.1, 1.2])
        with c_tg:
            render_html(f"""
                <div style="text-align: right; padding-top: 6px;">
                    <a href="{web_url}" target="_blank" class="tg-header-badge" id="header_tg_link" title="Open RiskLens Telegram Bot">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                        <span>Bot</span>
                        <span class="tg-pulse" style="width:5px; height:5px; margin-left: 2px;"></span>
                    </a>
                </div>
            """)
        with c_thm:
            st.markdown("<div class='theme-toggle-wrap' style='text-align: right; padding-top: 4px;'>", unsafe_allow_html=True)
            t_btn = st.button(t_label, key="top_theme_toggle", help="Toggle visual color mode")
            if t_btn:
                st.session_state.theme = "light" if is_dark else "dark"
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    # 2. PILL NAVIGATION TABS (Verify / Analytics / Pipeline / History / Settings)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Verify",
        "Analytics",
        "Pipeline",
        "History",
        "Settings"
    ])

    # 3. SIDEBAR TELEMETRY
    render_sidebar()

    with tab1:
        render_tab_verify()

    with tab2:
        render_tab_analytics()

    with tab3:
        render_tab_pipeline()

    with tab4:
        render_tab_history()

    with tab5:
        render_tab_settings()


if __name__ == "__main__":
    main()
