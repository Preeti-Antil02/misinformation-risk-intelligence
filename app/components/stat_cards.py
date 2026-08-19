"""
app/components/stat_cards.py
============================
High-level system metrics and telemetry stat cards.
"""

import streamlit as st
from risklens.feedback import get_analytics_data

def render_stat_cards():
    """Renders 4 modern stat cards at the top of the dashboard."""
    data = get_analytics_data()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        html = f"""
<div class="stat-card">
    <div style="font-size:12px; color:#8b9bb4; font-weight:600; display:flex; align-items:center; gap:6px;">
        <span>Verified today</span>
    </div>
    <div style="font-size:28px; font-weight:800; color:#ffffff; margin-top:8px;">{data['verifications_today']}</div>
    <div style="font-size:11px; color:#3fb950; font-weight:600; margin-top:4px;">↑ 12 from yesterday</div>
</div>
""".strip()
        st.markdown(html, unsafe_allow_html=True)

    with col2:
        html = f"""
<div class="stat-card">
    <div style="font-size:12px; color:#8b9bb4; font-weight:600; display:flex; align-items:center; gap:6px;">
        <span>High/Critical</span>
    </div>
    <div style="font-size:28px; font-weight:800; color:#ffffff; margin-top:8px;">18</div>
    <div style="font-size:11px; color:#8b9bb4; font-weight:600; margin-top:4px;">41.8% of today's checks</div>
</div>
""".strip()
        st.markdown(html, unsafe_allow_html=True)

    with col3:
        acc_pct = f"{data['live_accuracy'] * 100:.1f}%"
        html = f"""
<div class="stat-card">
    <div style="font-size:12px; color:#8b9bb4; font-weight:600; display:flex; align-items:center; gap:6px;">
        <span>Live accuracy</span>
    </div>
    <div style="font-size:28px; font-weight:800; color:#38bdf8; margin-top:8px;">{acc_pct}</div>
    <div style="font-size:11px; color:#8b9bb4; font-weight:600; margin-top:4px;">From {data['total_verifications']} feedbacks</div>
</div>
""".strip()
        st.markdown(html, unsafe_allow_html=True)

    with col4:
        langs = ", ".join(list(data['per_language_accuracy'].keys())[:4]) or "en, hi, ta, te"
        html = f"""
<div class="stat-card">
    <div style="font-size:12px; color:#8b9bb4; font-weight:600; display:flex; align-items:center; gap:6px;">
        <span>Languages today</span>
    </div>
    <div style="font-size:28px; font-weight:800; color:#ffffff; margin-top:8px;">{len(data['per_language_accuracy']) or 4}</div>
    <div style="font-size:11px; color:#8b9bb4; font-weight:600; margin-top:4px; font-family:monospace;">{langs}</div>
</div>
""".strip()
        st.markdown(html, unsafe_allow_html=True)
