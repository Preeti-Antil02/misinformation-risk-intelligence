"""
app/config.py
=============
Design Tokens, Custom CSS, Typography, and System Configuration.
"""

APP_VERSION = "v2.1.0"
APP_NAME = "RiskLens"
SYSTEM_TITLE = "Misinformation Risk Intelligence System"

# Risk Color Palette & Tokens (Updated for Modern Dashboard)
RISK_CONFIG = {
    "Low": {
        "bg": "rgba(46, 160, 67, 0.08)",
        "border": "#2ea043",
        "text": "#3fb950",
        "badge_bg": "rgba(46,160,67,0.15)",
        "badge_text": "#3fb950",
        "glow": "rgba(63,185,80,0.1)",
        "icon": "✓ Low Risk"
    },
    "Moderate": {
        "bg": "rgba(210, 153, 34, 0.08)",
        "border": "#d29922",
        "text": "#e3b341",
        "badge_bg": "rgba(210,153,34,0.15)",
        "badge_text": "#e3b341",
        "glow": "rgba(227,179,65,0.1)",
        "icon": "◈ Moderate Risk"
    },
    "High": {
        "bg": "rgba(219, 109, 40, 0.08)",
        "border": "#db6d28",
        "text": "#f0883e",
        "badge_bg": "rgba(219,109,40,0.15)",
        "badge_text": "#f0883e",
        "glow": "rgba(240,136,62,0.1)",
        "icon": "⚠ High Risk"
    },
    "Critical": {
        "bg": "rgba(218, 54, 51, 0.08)",
        "border": "#da3633",
        "text": "#f85149",
        "badge_bg": "rgba(218,54,51,0.15)",
        "badge_text": "#f85149",
        "glow": "rgba(248,81,73,0.15)",
        "icon": "✕ Critical Risk"
    }
}

# 4 Presets Distributed Across All 4 Risk Tiers
PRESET_EXAMPLES = {
    "low_risk": {
        "title": "🟢 Low Risk: WHO Health Advisory",
        "tier": "Low",
        "text": "The World Health Organization published updated clinical guidelines regarding seasonal influenza vaccinations and routine pediatric immunization schedules.",
        "url": "https://www.reuters.com/world/health-guidelines"
    },
    "moderate_risk": {
        "title": "🟡 Moderate Risk: Tech Market Speculation",
        "tier": "Moderate",
        "text": "Industry analysts and anonymous sources speculate that a major tech conglomerate may announce a multi-billion dollar semiconductor acquisition next quarter.",
        "url": "https://www.techcrunch.com/market-speculation"
    },
    "high_risk": {
        "title": "🟠 High Risk: Viral Utility Tax Leak",
        "tier": "High",
        "text": "Unverified leaked documents circulating on social media claim controversial new energy tariffs will double residential electricity bills starting next month.",
        "url": "https://www.thegatewaypundit.com/tariff-leak"
    },
    "critical_risk": {
        "title": "🔴 Critical Risk: 5G Vaccine Microchip Hoax",
        "tier": "Critical",
        "text": "SHOCKING BOMBSHELL: Medical whistleblowers confirm COVID-19 mRNA vaccines contain secret 5G microchips engineered to track global citizens worldwide!",
        "url": "https://www.naturalnews.com/health-hoax"
    }
}

# Global Custom CSS for Modern "Glass" UI
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* Base Overrides */
div.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px !important;
}
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #e6edf3;
}
.stApp { background: #080c12; }

/* Grid Background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle at 2px 2px, rgba(88,166,255,0.03) 1px, transparent 0);
    background-size: 32px 32px;
    pointer-events: none;
    z-index: 0;
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(13, 20, 34, 0.6) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 16px !important;
}

/* Stat Card Styling */
.stat-card {
    background: #0d1422;
    border: 1px solid #1f2c44;
    border-radius: 12px;
    padding: 16px;
    transition: all 0.2s ease;
}
.stat-card:hover {
    border-color: #388bfd;
    transform: translateY(-2px);
}

/* Chips/Badges */
.chip {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    margin-right: 6px;
    margin-bottom: 6px;
}

/* Inputs */
textarea, input[type="text"] {
    background: rgba(13, 20, 34, 0.8) !important;
    border: 1px solid #1f2c44 !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    padding: 14px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.1) !important;
}

/* Buttons */
div[data-testid="stButton"] button {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background: #238636 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    width: 100% !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #2ea043 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #070b12 !important;
    border-right: 1px solid #1a2436;
}

/* Hide Streamlit components */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* Custom Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: #080c12; }
::-webkit-scrollbar-thumb { background: #1f2c44; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #388bfd; }

/* Interactive Navigation Styling */
.nav-pill {
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}
.nav-pill:hover {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.05);
}
.nav-pill.active {
    background: #4f46e5 !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}

.upgrade-btn {
    transition: all 0.3s ease;
}
.upgrade-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4) !important;
    background: #5a52ff !important;
}
</style>
"""
