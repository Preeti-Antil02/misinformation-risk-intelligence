"""
app/state.py
============
Session State Manager for RiskLens Web Application.
Handles clean state transitions, resets previous scrape history on new inputs,
and ensures a pristine empty landing page on startup.
"""

from typing import Dict, Any, Optional
import streamlit as st
from app.config import PRESET_EXAMPLES


class SessionStateManager:
    """
    Manages Streamlit session state lifecycle with automatic history isolation.
    """

    @staticmethod
    def initialize():
        """Initializes default session state variables with a clean blank landing page."""
        if "text_input" not in st.session_state:
            st.session_state.text_input = ""
        if "url_input" not in st.session_state:
            st.session_state.url_input = ""
        if "scraped_meta" not in st.session_state:
            st.session_state.scraped_meta = None
        if "analysis_cache" not in st.session_state:
            st.session_state.analysis_cache = None
        if "last_analyzed_text" not in st.session_state:
            st.session_state.last_analyzed_text = ""
        if "last_scraped_url" not in st.session_state:
            st.session_state.last_scraped_url = ""

    @staticmethod
    def load_preset(preset_key: str):
        """Loads a preset example on-demand when clicked."""
        default_preset = next(iter(PRESET_EXAMPLES.values()))
        preset = PRESET_EXAMPLES.get(preset_key, default_preset)
        st.session_state.text_input = preset["text"]
        st.session_state.url_input = preset.get("url", "")
        st.session_state.scraped_meta = None  # Clear previous scrape history
        st.session_state.analysis_cache = None  # Force fresh analysis
        st.session_state.last_scraped_url = preset.get("url", "")

    @staticmethod
    def set_scraped_content(parsed: Dict[str, Any], url: str):
        """Sets freshly scraped live web article content into session state."""
        full_content = f"{parsed['title']}\n\n{parsed['full_text']}"
        st.session_state.text_input = full_content
        st.session_state.url_input = url
        st.session_state.scraped_meta = parsed
        st.session_state.analysis_cache = None  # Force fresh analysis
        st.session_state.last_scraped_url = url

    @staticmethod
    def clear_all():
        """Resets the entire input and analysis state."""
        st.session_state.text_input = ""
        st.session_state.url_input = ""
        st.session_state.scraped_meta = None
        st.session_state.analysis_cache = None
        st.session_state.last_analyzed_text = ""
        st.session_state.last_scraped_url = ""
