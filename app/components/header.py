"""
app/components/header.py
========================
Modern Navigation Header with Interactive Components.
"""

import streamlit as st

def render_header():
    """Renders the top branding and navigation bar with enhanced interactivity."""
    html_content = """
<div style="display:flex; align-items:center; justify-content:space-between; padding: 10px 0 25px 0;">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="background:linear-gradient(135deg, #238636 0%, #2ea043 100%); width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-size:18px; box-shadow: 0 4px 12px rgba(46, 160, 67, 0.2);">🛡️</div>
        <div>
            <div style="font-size:22px; font-weight:800; color:#ffffff; letter-spacing:-0.03em; line-height:1;">RiskLens</div>
            <div style="font-size:11px; color:#8b9bb4; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-top:2px;">Intelligence Hub</div>
        </div>
    </div>

    <div style="display:flex; align-items:center; gap:12px;">
        <div style="display:flex; background:rgba(13, 20, 34, 0.6); padding:4px; border-radius:14px; border:1px solid rgba(255, 255, 255, 0.05); backdrop-filter: blur(8px);">
            <div class="nav-pill active" style="padding:7px 18px; border-radius:10px; font-size:13px; font-weight:700;">Verify</div>
            <div class="nav-pill" style="color:#8b9bb4; padding:7px 18px; border-radius:10px; font-size:13px; font-weight:600;">Analytics</div>
            <div class="nav-pill" style="color:#8b9bb4; padding:7px 18px; border-radius:10px; font-size:13px; font-weight:600;">History</div>
            <div class="nav-pill" style="color:#8b9bb4; padding:7px 18px; border-radius:10px; font-size:13px; font-weight:600;">Settings</div>
        </div>

        <div style="display:flex; align-items:center; gap:16px; margin-left:12px;">
            <div style="display:flex; flex-direction:column; align-items:flex-end;">
                <div style="font-size:12px; color:#ffffff; font-weight:700;">20 / 100</div>
                <div style="font-size:10px; color:#8b9bb4; font-weight:500;">daily free credits</div>
            </div>
            <div class="upgrade-btn" style="background:#4f46e5; color:white; padding:10px 24px; border-radius:12px; font-size:13px; font-weight:800; cursor:pointer; box-shadow:0 4px 12px rgba(79, 70, 229, 0.3); border:1px solid rgba(255,255,255,0.1);">Upgrade</div>
        </div>
    </div>
</div>
""".strip()
    st.markdown(html_content, unsafe_allow_html=True)
