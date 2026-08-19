"""
app/components/sidebar_analytics.py
===================================
Right sidebar components: Accuracy ring, language charts, and recent activity.
"""

import streamlit as st
import plotly.graph_objects as go
from risklens.feedback import get_analytics_data

def render_sidebar_analytics():
    """Renders the right analytics panel."""
    data = get_analytics_data()

    # 1. Live Accuracy Ring
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;">Live accuracy</div>', unsafe_allow_html=True)

    acc = data['live_accuracy']
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = acc * 100,
        number = {'suffix': "%", 'font': {'size': 42, 'color': '#38bdf8', 'family': 'Plus Jakarta Sans'}},
        gauge = {
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': "#38bdf8"},
            'bgcolor': "rgba(31, 41, 55, 0.5)",
            'borderwidth': 0,
        },
        domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown(f"""
    <div style="text-align:center; color:#8b9bb4; font-size:12px; margin-top:-20px;">
        from user feedback<br>
        <span style="color:#ffffff;">{data['total_verifications']} corrections logged</span> · next retrain in 158
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 2. Languages Today
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;">Languages today</div>', unsafe_allow_html=True)

    lang_data = [
        {"name": "English", "pct": 60, "color": "#58a6ff"},
        {"name": "Hindi", "pct": 28, "color": "#8957e5"},
        {"name": "Tamil", "pct": 8, "color": "#f85149"},
        {"name": "Telugu", "pct": 4, "color": "#e3b341"},
    ]

    for l in lang_data:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <div style="font-size:12px; color:#e6edf3;">{l['name']}</div>
            <div style="width:120px; background:rgba(31, 41, 55, 0.5); height:6px; border-radius:3px; overflow:hidden; display:flex; align-items:center; gap:8px;">
                <div style="width:{l['pct']}%; background:{l['color']}; height:100%;"></div>
            </div>
            <div style="font-size:11px; color:#8b9bb4; width:30px; text-align:right;">{l['pct']}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Recent Verifications
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:12px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;">Recent verifications</div>', unsafe_allow_html=True)

    recents = [
        {"title": "Onions cure COVID...", "risk": "High", "color": "#f0883e"},
        {"title": "RBI rate cut 2026...", "risk": "Low", "color": "#3fb950"},
        {"title": "सरकार ने नई योजना...", "risk": "Mod", "color": "#e3b341"},
        {"title": "5G causes cancer...", "risk": "Critical", "color": "#f85149"},
        {"title": "ISRO moon mission...", "risk": "Low", "color": "#3fb950"},
    ]

    for r in recents:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.03);">
            <div style="display:flex; align-items:center; gap:8px;">
                <div style="width:8px; height:8px; border-radius:50%; background:{r['color']};"></div>
                <div style="font-size:12px; color:#e6edf3; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; width:140px;">{r['title']}</div>
            </div>
            <div style="font-size:10px; background:rgba(255,255,255,0.05); padding:2px 8px; border-radius:10px; color:#8b9bb4;">{r['risk']}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. WhatsApp Bot CTA
    st.markdown(f"""
    <div style="background:rgba(88, 166, 255, 0.05); border:1px solid rgba(88, 166, 255, 0.1); border-radius:16px; padding:20px;">
        <div style="font-size:13px; font-weight:700; color:#58a6ff; display:flex; align-items:center; gap:8px;">
            <span>📱 WhatsApp bot</span>
        </div>
        <p style="font-size:12px; color:#8b9bb4; margin:12px 0;">Forward any message to verify instantly. 500M+ Indian users, zero friction.</p>
        <div style="background:rgba(13, 20, 34, 0.8); border:1px solid #1f2c44; border-radius:10px; padding:10px; font-family:monospace; font-size:13px; color:#ffffff; text-align:center;">
            +91 98765 43210
        </div>
    </div>
    """, unsafe_allow_html=True)
