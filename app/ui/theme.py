import streamlit as st

def inject_theme():
    """
    Injects the state-of-the-art RiskLens Enterprise Intelligence theme:
    - Unified single-line top row: Brand (left), Pill Nav (center), Theme toggle (right)
    - Responsive collapse on narrow/mobile viewports
    - Headline rendered in bold Space Grotesk matching the RiskLens brand wordmark
    - Clean single-button resting state for 'Browse files' drag-and-drop uploader
    - Full color token system with Dark & Light mode contrast compliance
    - 100% SVG outline icons (zero emoji)
    """
    if "theme" not in st.session_state:
        st.session_state.theme = "dark"

    theme = st.session_state.theme

    if theme == "light":
        tokens = {
            "bg": "#F5F4FA",
            "surface": "#FFFFFF",
            "surface_alt": "#EFEBF7",
            "surface_raised": "#E6E1F2",
            "border": "#DCD7E8",
            "border_focus": "#6C63F0",
            "text": "#171524",
            "text_muted": "#5C5673",
            "text_faint": "#8B85A3",
            "primary": "#6C63F0",
            "primary_2": "#5B52E0",
            "purple": "#7E5CE6",
            "teal": "#1DB398",
            "critical": "#DC3832",
            "critical_soft": "#FDE8E7",
            "high": "#D97706",
            "high_soft": "#FEF3C7",
            "moderate": "#D97706",
            "moderate_soft": "#FEF3C7",
            "low": "#16A34A",
            "low_soft": "#DCFCE7",
            "shadow_sm": "0 2px 8px rgba(23, 21, 36, 0.05)",
            "shadow_md": "0 8px 24px rgba(23, 21, 36, 0.08)",
            "shadow_glow": "0 0 30px rgba(108, 99, 240, 0.12)",
            "grid_color": "rgba(108, 99, 240, 0.035)",
            "node_stroke": "rgba(108, 99, 240, 0.2)",
            "node_line": "rgba(108, 99, 240, 0.07)",
        }
    else:
        tokens = {
            "bg": "#0A0910",
            "surface": "#14121F",
            "surface_alt": "#1B1828",
            "surface_raised": "#1F1B2E",
            "border": "#2A2638",
            "border_focus": "#8B7CF6",
            "text": "#F2F0FA",
            "text_muted": "#9891B0",
            "text_faint": "#655F7D",
            "primary": "#6C63F0",
            "primary_2": "#8B7CF6",
            "purple": "#9B7CF5",
            "teal": "#4FD1B8",
            "critical": "#F0554F",
            "critical_soft": "#3A1B1E",
            "high": "#F0A339",
            "high_soft": "#3B2A14",
            "moderate": "#F0A339",
            "moderate_soft": "#3B2A14",
            "low": "#2FCC93",
            "low_soft": "#123128",
            "shadow_sm": "0 2px 8px rgba(0, 0, 0, 0.35)",
            "shadow_md": "0 10px 30px rgba(0, 0, 0, 0.55)",
            "shadow_glow": "0 0 40px rgba(108, 99, 240, 0.2)",
            "grid_color": "rgba(155, 124, 245, 0.03)",
            "node_stroke": "rgba(155, 124, 245, 0.35)",
            "node_line": "rgba(108, 99, 240, 0.08)",
        }

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {{
        --bg: {tokens['bg']};
        --surface: {tokens['surface']};
        --surface-alt: {tokens['surface_alt']};
        --surface-raised: {tokens['surface_raised']};
        --border: {tokens['border']};
        --border-focus: {tokens['border_focus']};
        --text: {tokens['text']};
        --text-muted: {tokens['text_muted']};
        --text-faint: {tokens['text_faint']};
        --primary: {tokens['primary']};
        --primary-2: {tokens['primary_2']};
        --purple: {tokens['purple']};
        --teal: {tokens['teal']};
        --critical: {tokens['critical']};
        --critical-soft: {tokens['critical_soft']};
        --high: {tokens['high']};
        --high-soft: {tokens['high_soft']};
        --moderate: {tokens['moderate']};
        --moderate-soft: {tokens['moderate_soft']};
        --low: {tokens['low']};
        --low-soft: {tokens['low_soft']};
        --shadow-sm: {tokens['shadow_sm']};
        --shadow-md: {tokens['shadow_md']};
        --shadow-glow: {tokens['shadow_glow']};
    }}

    * {{
        box-sizing: border-box;
    }}

    /* Global Streamlit App */
    .stApp {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
    }}

    div.block-container {{
        padding-top: 1rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1240px !important;
        margin: 0 auto !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
        position: relative;
        z-index: 2;
    }}

    /* Typography Defaults */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text) !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        letter-spacing: -0.015em !important;
    }}
    p, span, label, div {{
        color: inherit;
    }}

    svg.icon {{
        width: 15px;
        height: 15px;
        flex-shrink: 0;
        vertical-align: middle;
    }}

    /* ========================================================================
       BACKGROUND CANVAS MESH
       ======================================================================== */
    .stApp::before {{
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            radial-gradient(circle at 50% 15%, rgba(108, 99, 240, 0.08) 0%, rgba(155, 124, 245, 0.03) 40%, transparent 70%),
            radial-gradient(circle at 85% 65%, rgba(79, 209, 184, 0.035) 0%, transparent 45%),
            linear-gradient({tokens['grid_color']} 1px, transparent 1px),
            linear-gradient(90deg, {tokens['grid_color']} 1px, transparent 1px);
        background-size: 100% 100%, 100% 100%, 54px 54px, 54px 54px;
        pointer-events: none;
        z-index: 0;
    }}

    .stApp::after {{
        content: '';
        position: fixed;
        inset: 0;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='600' viewBox='0 0 800 600'%3E%3Cg fill='none' stroke='{tokens['node_line'].replace('#', '%23').replace('(', '%28').replace(')', '%29').replace(',', '%2C').replace(' ', '')}' stroke-width='1'%3E%3Cpath d='M100 80 L220 180 L380 120 L540 220 L700 140 M220 180 L280 340 L460 300 L540 220 M100 80 L160 280 L280 340 M460 300 L620 420 L700 140 M380 120 L460 300'/%3E%3C/g%3E%3Cg fill='{tokens['node_stroke'].replace('#', '%23').replace('(', '%28').replace(')', '%29').replace(',', '%2C').replace(' ', '')}'%3E%3Ccircle cx='100' cy='80' r='3.5'/%3E%3Ccircle cx='220' cy='180' r='4'/%3E%3Ccircle cx='380' cy='120' r='3'/%3E%3Ccircle cx='540' cy='220' r='4.5'/%3E%3Ccircle cx='700' cy='140' r='3.5'/%3E%3Ccircle cx='160' cy='280' r='3'/%3E%3Ccircle cx='280' cy='340' r='4'/%3E%3Ccircle cx='460' cy='300' r='4'/%3E%3Ccircle cx='620' cy='420' r='3.5'/%3E%3C/g%3E%3C/svg%3E");
        background-size: 800px 600px;
        opacity: 0.65;
        pointer-events: none;
        z-index: 0;
        animation: meshDrift 50s linear infinite alternate;
    }}

    @keyframes meshDrift {{
        0% {{ transform: translate(0, 0) scale(1); }}
        50% {{ transform: translate(-20px, 15px) scale(1.02); }}
        100% {{ transform: translate(25px, -15px) scale(0.98); }}
    }}

    /* ========================================================================
       1. UNIFIED SINGLE TOP ROW: BRAND (LEFT) + PILL NAV (CENTER) + THEME (RIGHT)
       ======================================================================== */
    .header-row-wrap {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 0 0;
        margin-bottom: 0;
        position: relative;
        z-index: 10;
        width: 100%;
    }}
    .brand-box {{
        display: flex;
        flex-direction: column;
        justify-content: center;
        flex-shrink: 0;
    }}
    .eyebrow {{
        font-size: 10.5px;
        letter-spacing: .12em;
        color: var(--primary-2);
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 2px;
        line-height: 1.2;
    }}
    .brand-row {{
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .brand-name {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        font-weight: 700;
        color: var(--text);
        line-height: 1.1;
        letter-spacing: -0.02em;
    }}
    .badge {{
        background: var(--surface-alt);
        border: 1px solid var(--border);
        color: var(--purple);
        font-size: 10.5px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 20px;
    }}
    .theme-toggle-wrap {{
        display: flex;
        align-items: center;
        justify-content: flex-end;
        flex-shrink: 0;
    }}

    /* Seamless Single-Line Pill Navigation Integration */
    .stTabs [data-baseweb="tab-list"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 26px !important;
        padding: 4px !important;
        gap: 4px !important;
        width: fit-content !important;
        backdrop-filter: blur(16px);
        box-shadow: var(--shadow-sm);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 22px !important;
        padding: 7px 16px !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 12.5px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: var(--text) !important;
        background: var(--surface-alt) !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, var(--primary), var(--purple)) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(108, 99, 240, 0.35) !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* Desktop Single-Row Merge */
    @media (min-width: 960px) {{
        .stTabs {{
            margin-top: -46px !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            margin: 0 auto 28px !important;
            position: relative;
            z-index: 20;
        }}
    }}

    /* Tablet / Mobile Clean Responsive Wrap */
    @media (max-width: 959px) {{
        .header-row-wrap {{
            flex-wrap: wrap;
            gap: 8px;
        }}
        .stTabs {{
            margin-top: 10px !important;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            margin: 8px auto 22px !important;
            width: 100% !important;
            justify-content: center !important;
            overflow-x: auto !important;
        }}
    }}

    /* ========================================================================
       2. HEADLINE & SUBHEAD (Matching 'Space Grotesk' & Bolder Weight)
       ======================================================================== */
    .hero-text {{
        text-align: center;
        padding: 24px 20px 0;
        margin-bottom: 24px;
        animation: fadeUp 0.5s ease both;
    }}
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .headline {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-style: normal !important;
        font-weight: 800 !important;
        font-size: clamp(30px, 4.6vw, 44px) !important;
        line-height: 1.2 !important;
        letter-spacing: -0.02em !important;
        background: linear-gradient(90deg, var(--text) 20%, #D9D4F5 65%, #B7ACF6 95%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 auto !important;
    }}
    .subhead {{
        font-size: 14.5px !important;
        color: var(--text-muted) !important;
        max-width: 560px !important;
        margin: 12px auto 0 !important;
        line-height: 1.6 !important;
    }}

    /* ========================================================================
       3. INPUT CARD & INTERACTION CONTROLS
       ======================================================================== */
    .tool-wrap {{
        max-width: 780px;
        margin: 0 auto 28px;
        padding: 0 4px;
    }}
    .input-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 4px;
        box-shadow: var(--shadow-sm);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .input-card:focus-within {{
        border-color: var(--border-focus);
        box-shadow: 0 0 0 3px rgba(108, 99, 240, 0.15);
    }}

    textarea, .stTextArea textarea {{
        width: 100% !important;
        min-height: 88px !important;
        background: transparent !important;
        border: none !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14.5px !important;
        padding: 14px 16px !important;
        resize: none !important;
        line-height: 1.6 !important;
    }}
    textarea:focus, .stTextArea textarea:focus {{
        outline: none !important;
        border: none !important;
        box-shadow: none !important;
    }}
    textarea::placeholder, .stTextArea textarea::placeholder {{
        color: var(--text-faint) !important;
    }}

    div[data-testid="stTextArea"] > div {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 2px !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stTextArea"] > div:focus-within {{
        border-color: var(--border-focus) !important;
        box-shadow: 0 0 0 3px rgba(108, 99, 240, 0.15) !important;
    }}

    /* Buttons: Primary & Secondary */
    div[data-testid="stButton"] button {{
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 13.5px !important;
        padding: 12px 20px !important;
        min-height: 44px !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        letter-spacing: 0.01em;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
    }}

    /* Coral / Red Warm Gradient on Primary */
    div[data-testid="stButton"] button[kind="primary"] {{
        background: linear-gradient(90deg, #F0554F 0%, #E0432E 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(240, 85, 79, 0.5) !important;
        box-shadow: 0 4px 14px rgba(240, 85, 79, 0.3) !important;
    }}
    div[data-testid="stButton"] button[kind="primary"]:hover {{
        background: linear-gradient(90deg, #F86E69 0%, #F0554F 100%) !important;
        transform: translateY(-1.5px);
        box-shadow: 0 6px 20px rgba(240, 85, 79, 0.42) !important;
    }}
    div[data-testid="stButton"] button[kind="primary"]:active {{
        transform: translateY(0);
    }}

    /* Outline Secondary Button */
    div[data-testid="stButton"] button[kind="secondary"] {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        box-shadow: var(--shadow-sm);
    }}
    div[data-testid="stButton"] button[kind="secondary"]:hover {{
        background: var(--surface-alt) !important;
        border-color: var(--primary-2) !important;
        color: var(--text) !important;
        transform: translateY(-1.5px);
    }}

    /* ========================================================================
       4. SIMPLIFIED 'BROWSE FILES' SINGLE-BUTTON UPLOADER
       ======================================================================== */
    div[data-testid="stFileUploader"] {{
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        gap: 6px !important;
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}

    /* Target the dropzone section to look identical to a 44px secondary button */
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"],
    div[data-testid="stFileUploader"] section {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 12px !important;
        padding: 0 14px !important;
        min-height: 44px !important;
        height: 44px !important;
        max-height: 44px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        cursor: pointer !important;
        box-shadow: var(--shadow-sm) !important;
        width: 100% !important;
        box-sizing: border-box !important;
        overflow: hidden !important;
    }}
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover,
    div[data-testid="stFileUploader"] section:hover {{
        background: var(--surface-alt) !important;
        border-color: var(--primary-2) !important;
        transform: translateY(-1.5px);
        box-shadow: var(--shadow-md) !important;
    }}

    /* Active Drag-over state styling */
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:focus-within,
    div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:active {{
        border-color: var(--primary-2) !important;
        background: var(--surface-alt) !important;
    }}

    /* Hide persistent verbose 'Drag and drop file here' text, subtext & default icons */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploader"] section [data-testid="stFileUploaderDropzoneInstructions"],
    div[data-testid="stFileUploader"] section small,
    div[data-testid="stFileUploader"] section > div > div > small,
    div[data-testid="stFileUploader"] section svg {{
        display: none !important;
    }}

    /* Keep inner container compact and centered */
    div[data-testid="stFileUploader"] section > div {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
    }}

    div[data-testid="stFileUploader"] section > div > div {{
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        margin: 0 !important;
        width: 100% !important;
    }}

    /* Style the inner 'Browse files' button text seamlessly */
    div[data-testid="stFileUploader"] section button {{
        background: transparent !important;
        border: none !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        min-height: auto !important;
        height: auto !important;
        cursor: pointer !important;
        width: 100% !important;
    }}
    div[data-testid="stFileUploader"] section button:hover {{
        color: var(--text) !important;
        background: transparent !important;
        transform: none !important;
    }}

    div[data-testid="stFileUploader"] section button::before {{
        content: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='%238B7CF6' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3v12M7 8l5-5 5 5M5 21h14'/%3E%3C/svg%3E");
        display: inline-block;
        vertical-align: middle;
        margin-right: 4px;
    }}

    /* Style the uploaded file list / error item if rendered by Streamlit */
    div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    div[data-testid="stFileUploader"] ul[data-testid="stFileUploaderPagination"] {{
        background: var(--surface-alt) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 4px 8px !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* Badges & Chips */
    .lang-badge {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: var(--surface-alt);
        color: var(--primary-2);
        border: 1px solid var(--border);
        font-weight: 600;
        font-size: 11.5px;
        padding: 4px 12px;
        border-radius: 20px;
    }}

    /* ========================================================================
       LIVE PROCESSING PANEL (State 1)
       ======================================================================== */
    .proc-wrap {{
        max-width: 780px;
        margin: 0 auto 32px;
        padding: 0 4px;
    }}
    .proc-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 6px 0;
        box-shadow: var(--shadow-md);
        overflow: hidden;
    }}
    .proc-head {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
    }}
    .proc-head-left {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .proc-spinner {{
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 2.5px solid var(--border);
        border-top-color: var(--primary-2);
        animation: spin 0.9s linear infinite;
        flex-shrink: 0;
    }}
    @keyframes spin {{
        to {{ transform: rotate(360deg); }}
    }}
    .proc-title {{
        font-size: 14px;
        font-weight: 600;
        color: var(--text);
    }}
    .proc-sub {{
        font-size: 11.5px;
        color: var(--text-faint);
        margin-top: 1px;
    }}
    .proc-chevron {{
        color: var(--text-faint);
    }}

    .proc-steps {{
        padding: 4px 20px 16px;
    }}
    .proc-step {{
        display: flex;
        gap: 12px;
        align-items: flex-start;
        padding: 10px 0;
        border-top: 1px solid var(--border);
    }}
    .proc-step:first-child {{
        border-top: none;
    }}
    .proc-node {{
        width: 26px;
        height: 26px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border: 1.5px solid var(--border);
        background: var(--surface-raised);
        margin-top: 1px;
    }}
    .proc-node.done {{
        border-color: var(--low);
        color: var(--low);
    }}
    .proc-node.active {{
        border-color: var(--primary-2);
        color: var(--primary-2);
    }}
    .proc-node.active svg {{
        animation: spin 1.2s linear infinite;
    }}
    .proc-node.pending {{
        color: var(--text-faint);
    }}
    .proc-step-text {{
        flex: 1;
    }}
    .proc-step-title {{
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
    }}
    .proc-step-title .accent {{
        color: var(--primary-2);
        font-weight: 700;
    }}
    .proc-step-sub {{
        font-size: 11.5px;
        color: var(--text-faint);
        margin-top: 2px;
    }}
    .proc-step.pending .proc-step-title {{
        color: var(--text-muted);
    }}

    /* ========================================================================
       FINAL REPORT & TELEGRAM PREVIEW (State 2)
       ======================================================================== */
    .report-wrap {{
        max-width: 1080px;
        margin: 0 auto;
        padding: 0;
    }}
    .report-card {{
        background: var(--surface);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: var(--shadow-md);
        transition: border-color 0.3s ease;
    }}
    .report-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 20px 24px;
        border-bottom: 1px solid var(--border);
    }}
    .report-header-left {{
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .risk-icon {{
        width: 38px;
        height: 38px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .risk-eyebrow {{
        font-size: 10px;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }}
    .risk-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        font-weight: 700;
        margin-top: 2px;
        color: var(--text);
    }}
    .risk-pct {{
        text-align: right;
    }}
    .risk-pct .n {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        line-height: 1;
    }}
    .risk-pct .l {{
        font-size: 10px;
        color: var(--text-faint);
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    .report-body {{
        padding: 22px 24px 0;
    }}
    .sec-label {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
        color: var(--text-faint);
        margin-bottom: 9px;
        display: flex;
        align-items: center;
        gap: 7px;
    }}
    .sec-block {{
        margin-bottom: 22px;
    }}
    .claim-quote {{
        font-size: 14px;
        font-style: italic;
        color: var(--text);
        line-height: 1.55;
        background: var(--surface-alt);
        border-left: 2.5px solid var(--primary-2);
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
    }}
    .evidence-box {{
        font-size: 13px;
        color: var(--text-muted);
        line-height: 1.65;
        background: var(--surface-alt);
        border-radius: 10px;
        padding: 14px 16px;
        border: 1px solid var(--border);
    }}

    .factor-row {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 9px;
        font-size: 12px;
    }}
    .factor-row:last-child {{
        margin-bottom: 0;
    }}
    .factor-label {{
        width: 120px;
        color: var(--text-muted);
        flex-shrink: 0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .factor-track {{
        flex: 1;
        height: 6px;
        background: var(--surface-alt);
        border-radius: 4px;
        overflow: hidden;
    }}
    .factor-fill {{
        height: 100%;
        border-radius: 4px;
    }}
    .factor-val {{
        width: 44px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        color: var(--text-faint);
        font-size: 10.5px;
    }}

    .source-list {{
        display: flex;
        flex-direction: column;
        gap: 0;
    }}
    .source-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid var(--border);
    }}
    .source-row:last-child {{
        border-bottom: none;
    }}
    .source-left {{
        display: flex;
        align-items: center;
        gap: 10px;
        overflow: hidden;
    }}
    .trust-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--low);
        flex-shrink: 0;
    }}
    .source-name {{
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
    }}
    .source-domain {{
        font-size: 11px;
        color: var(--text-faint);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .verified-pill {{
        font-size: 10px;
        font-weight: 700;
        color: var(--low);
        background: var(--low-soft);
        padding: 3px 9px;
        border-radius: 6px;
        border: 1px solid var(--border);
        flex-shrink: 0;
    }}

    .report-actions {{
        display: flex;
        gap: 10px;
        padding: 18px 24px;
        border-top: 1px solid var(--border);
        background: var(--surface-alt);
        margin-top: 22px;
    }}
    .rbtn {{
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 10px 14px;
        border-radius: 10px;
        border: 1.5px solid var(--border);
        background: var(--surface);
        color: var(--text-muted);
        font-weight: 600;
        font-size: 12.5px;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .rbtn:hover {{
        border-color: var(--primary-2);
        color: var(--text);
        transform: translateY(-1px);
    }}

    /* Telegram Panel */
    .tg-panel-label {{
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .06em;
        color: var(--text-faint);
        margin-bottom: 12px;
        text-align: center;
    }}
    .phone {{
        background: #0F1A24;
        border-radius: 28px;
        padding: 10px;
        border: 5px solid #161616;
        box-shadow: var(--shadow-md);
        max-width: 320px;
        margin: 0 auto;
    }}
    .phone-screen {{
        background: #0E1621;
        border-radius: 20px;
        overflow: hidden;
        min-height: 420px;
        display: flex;
        flex-direction: column;
    }}
    .tg-header {{
        background: #17212B;
        color: #fff;
        padding: 13px 15px;
        display: flex;
        align-items: center;
        gap: 9px;
        font-size: 12.5px;
        font-weight: 600;
        border-bottom: 1px solid #232E3C;
    }}
    .tg-avatar {{
        width: 26px;
        height: 26px;
        border-radius: 50%;
        background: linear-gradient(135deg, #2AABEE, #229ED9);
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }}
    .tg-avatar svg {{
        width: 13px;
        height: 13px;
        color: #fff;
    }}
    .tg-body {{
        padding: 13px 10px;
        flex: 1;
    }}
    .tg-bubble {{
        background: #182533;
        border-radius: 14px;
        padding: 11px 13px;
        font-size: 11.5px;
        line-height: 1.6;
        color: #E8EBEF;
    }}
    .tg-row {{
        display: flex;
        align-items: flex-start;
        gap: 7px;
        margin-bottom: 9px;
    }}
    .tg-row:last-child {{
        margin-bottom: 0;
    }}
    .tg-row svg {{
        width: 13px;
        height: 13px;
        flex-shrink: 0;
        margin-top: 1px;
        color: #5EB5F7;
    }}
    .tg-risk-tag {{
        display: inline-block;
        padding: 2px 9px;
        border-radius: 6px;
        font-size: 10.5px;
        font-weight: 700;
    }}
    .tg-kb {{
        display: flex;
        gap: 6px;
        margin-top: 11px;
    }}
    .tg-kb-btn {{
        flex: 1;
        background: #1F2C3A;
        border: 1px solid #2E3B4C;
        color: #5EB5F7;
        text-align: center;
        padding: 7px 4px;
        border-radius: 8px;
        font-size: 10.5px;
        font-weight: 600;
    }}

    /* ========================================================================
       FOOTER & LIVE TICKER PILL
       ======================================================================== */
    .footer {{
        margin-top: 48px;
        padding: 32px 24px 48px;
        text-align: center;
        border-top: 1px solid var(--border);
    }}
    .ticker {{
        display: inline-flex;
        align-items: center;
        gap: 9px;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 7px 18px;
        font-size: 12px;
        font-weight: 600;
        color: var(--text-muted);
        box-shadow: var(--shadow-sm);
    }}
    .ticker .dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--teal);
        animation: pulse 1.8s infinite;
        flex-shrink: 0;
    }}
    .ticker .num {{
        font-family: 'JetBrains Mono', monospace;
        color: var(--teal);
        font-weight: 700;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50% {{ opacity: .45; transform: scale(0.9); }}
    }}
    .footer-note {{
        margin-top: 10px;
        font-size: 11px;
        color: var(--text-faint);
    }}

    /* Sidebar Navigation */
    section[data-testid="stSidebar"] {{
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
        backdrop-filter: blur(20px);
    }}

    /* Mono Numbers */
    .mono {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Pulsing Signal Dot */
    .pulse-dot {{
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--primary-2);
        box-shadow: 0 0 0 0 rgba(108, 99, 240, 0.7);
        animation: pulseAnimation 2s infinite;
    }}
    @keyframes pulseAnimation {{
        0% {{ box-shadow: 0 0 0 0 rgba(108, 99, 240, 0.7); }}
        70% {{ box-shadow: 0 0 0 7px rgba(108, 99, 240, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(108, 99, 240, 0); }}
    }}

    /* Card System Generic */
    .card {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 20px;
        box-shadow: var(--shadow-sm);
        backdrop-filter: blur(16px);
        transition: all 0.2s ease;
        color: var(--text);
    }}

    /* Accessibility: Respect Reduced Motion */
    @media (prefers-reduced-motion: reduce) {{
        .stApp::after, .proc-spinner, .proc-node.active svg, .ticker .dot, .pulse-dot {{
            animation: none !important;
        }}
        div[data-testid="stButton"] button, textarea, .rbtn {{
            transition: none !important;
            transform: none !important;
        }}
    }}

    /* =========================================================================
       Telegram Direct Access Hero & Header Components
       ========================================================================= */
    .tg-hero-card {{
        background: linear-gradient(135deg, rgba(0, 136, 204, 0.12), rgba(0, 170, 255, 0.04)) !important;
        border: 1px solid rgba(0, 136, 204, 0.35) !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 20px rgba(0, 136, 204, 0.12) !important;
        backdrop-filter: blur(16px);
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        max-width: 380px;
        min-width: 280px;
    }}
    .tg-hero-card:hover {{
        border-color: rgba(0, 170, 255, 0.6) !important;
        box-shadow: 0 8px 28px rgba(0, 136, 204, 0.22) !important;
        transform: translateY(-2px);
    }}
    .tg-hero-header {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }}
    .tg-icon-circle {{
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #0088cc, #00b4d8);
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(0, 136, 204, 0.4);
        flex-shrink: 0;
    }}
    .tg-hero-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 14px;
        color: var(--text);
        display: flex;
        align-items: center;
    }}
    .tg-hero-handle {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        color: #00aaff;
        font-weight: 600;
    }}
    .tg-online-badge {{
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-size: 10px;
        font-weight: 700;
        color: #2FCC93;
        background: rgba(47, 204, 147, 0.15);
        padding: 2px 7px;
        border-radius: 12px;
        border: 1px solid rgba(47, 204, 147, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-left: 8px;
    }}
    .tg-pulse {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #2FCC93;
        box-shadow: 0 0 6px #2FCC93;
        animation: pulseAnimation 1.8s infinite;
    }}
    .tg-hero-desc {{
        font-size: 11.5px;
        color: var(--text-muted);
        line-height: 1.45;
        margin: 0 0 12px 0;
    }}
    .tg-hero-buttons {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }}
    .tg-btn-primary {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        background: linear-gradient(135deg, #0088cc, #00aaff) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 11.5px !important;
        padding: 7px 14px !important;
        border-radius: 8px !important;
        text-decoration: none !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 136, 204, 0.35) !important;
    }}
    .tg-btn-primary:hover {{
        box-shadow: 0 4px 14px rgba(0, 136, 204, 0.5) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }}
    .tg-btn-secondary {{
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        background: rgba(255, 255, 255, 0.06) !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 11.5px !important;
        padding: 7px 12px !important;
        border-radius: 8px !important;
        text-decoration: none !important;
        border: 1px solid var(--border) !important;
        transition: all 0.2s ease !important;
    }}
    .tg-btn-secondary:hover {{
        background: rgba(255, 255, 255, 0.12) !important;
        color: var(--text) !important;
        border-color: rgba(0, 136, 204, 0.4) !important;
    }}
    .tg-header-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(0, 136, 204, 0.16), rgba(0, 170, 255, 0.08));
        border: 1px solid rgba(0, 136, 204, 0.35);
        color: #00aaff !important;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 12px;
        font-weight: 700;
        text-decoration: none;
        transition: all 0.2s ease;
        margin-right: 8px;
    }}
    .tg-header-badge:hover {{
        background: linear-gradient(135deg, rgba(0, 136, 204, 0.28), rgba(0, 170, 255, 0.18));
        border-color: rgba(0, 170, 255, 0.6);
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 10px rgba(0, 136, 204, 0.25);
    }}

    /* Hide Streamlit Cruft */
    header[data-testid="stHeader"], #MainMenu, footer {{
        display: none !important;
        visibility: hidden !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
