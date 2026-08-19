"""
app/components/whatsapp_view.py
===============================
Live WhatsApp Intelligence Card Simulator.
"""

import streamlit as st
from risklens.whatsapp_bot import format_whatsapp_card


def render_whatsapp_view(agent_data: dict, user_url: str = ""):
    """Renders the real-time preview of the WhatsApp intelligence card."""
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 📱 Live WhatsApp Message Card Preview")
    st.markdown("<p style='color:#8b9bb4;font-size:12px;'>This formatted card is generated in real time and distributed to end-users via Twilio WhatsApp Sandbox.</p>", unsafe_allow_html=True)

    wa_card = format_whatsapp_card(agent_data, is_image=False, url=user_url if user_url else None)
    st.markdown(f"""
    <div style="background:#071018;border:1px solid #128c7e;border-radius:14px;padding:20px;max-width:540px;font-family:'JetBrains Mono',monospace;font-size:12px;color:#e6edf3;white-space:pre-wrap;line-height:1.6;box-shadow:0 8px 30px rgba(18,140,126,0.15);">
{wa_card}
    </div>
    """, unsafe_allow_html=True)
