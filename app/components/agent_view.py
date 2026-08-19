"""
app/components/agent_view.py
============================
LangGraph Multi-Agent Verification & Web Evidence Viewer.
"""

import streamlit as st


def render_agent_view(agent_data: dict):
    """Renders the 3-node LangGraph claim, evidence, and synthesis pipeline."""
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    st.markdown("#### 🤖 3-Node LangGraph Evidence Agent")

    # Claim Box
    st.markdown(f"""
    <div style="background:#0d1422;border:1px solid #1f2c44;border-left:4px solid #58a6ff;border-radius:10px;padding:14px 18px;margin-bottom:16px;">
        <div style="font-size:11px;color:#8b9bb4;text-transform:uppercase;font-weight:700;">Node 1: Extracted Verifiable Claim</div>
        <div style="font-size:15px;color:#ffffff;font-weight:600;margin-top:4px;">"{agent_data.get('claim', 'No verifiable factual claim isolated.')}"</div>
    </div>
    """, unsafe_allow_html=True)

    # Verdict Box
    st.markdown(f"""
    <div style="background:#0d1422;border:1px solid #1f2c44;border-left:4px solid #f85149;border-radius:10px;padding:14px 18px;margin-bottom:16px;">
        <div style="font-size:11px;color:#8b9bb4;text-transform:uppercase;font-weight:700;">Node 3: Synthesized Evidence Verdict</div>
        <div style="font-size:13px;color:#e6edf3;line-height:1.7;margin-top:4px;">{agent_data.get('verdict', 'Analysis in progress.')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Node 2 Sources
    st.markdown("#### Node 2: Retrieved Evidence & Fact-Check Sources")
    sources = agent_data.get("sources", [])
    if sources:
        for s in sources:
            st.markdown(f"""
            <div style="background:#0a101b;border:1px solid #1a2538;border-radius:10px;padding:12px 16px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="color:#58a6ff;font-weight:700;font-size:13px;">{s['name']}</span>
                    <a href="{s['url']}" target="_blank" style="color:#8b9bb4;font-size:12px;text-decoration:none;">🔗 Open Source</a>
                </div>
                <div style="font-size:12px;color:#c9d1d9;margin-top:6px;line-height:1.5;">{s['snippet']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No external web fact-check matches required or found for this statement.")
