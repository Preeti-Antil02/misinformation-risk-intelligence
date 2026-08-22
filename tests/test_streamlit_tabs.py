"""
Test script verifying all Streamlit tabs and UI components load and compile cleanly.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import app.ui.tab_verify as tab_verify
import app.ui.tab_analytics as tab_analytics
import app.ui.tab_pipeline as tab_pipeline
import app.ui.tab_history as tab_history
import app.ui.tab_settings as tab_settings
import app.ui.sidebar as sidebar
import app.ui.theme as theme
import app.ui.components.telegram_popover as tg_popover
import app.ui.components.result_card as result_card
import app.ui.components.telegram_preview as tg_preview

print("All UI modules, components, and tabs imported and verified successfully!")
