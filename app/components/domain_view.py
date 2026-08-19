"""
app/components/domain_view.py
=============================
Domain Reputation & Multi-Modal Composite Risk View.
"""

import streamlit as st


def render_domain_view(cred_data: dict, ensemble_prob: float, final_risk_score: float):
    """Renders the domain credibility breakdown and multi-modal risk calculation."""
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    if cred_data:
        c_score = cred_data["credibility_score"]
        c_pct = int(round(c_score * 100))
        badge_color = "#3fb950" if c_score >= 0.75 else ("#e3b341" if c_score >= 0.40 else "#f85149")

        st.markdown(f"""
        <div style="background:#0d1422;border:1px solid #1f2c44;border-radius:14px;padding:20px 24px;margin-bottom:18px;">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                <div>
                    <div style="font-size:11px;color:#8b9bb4;text-transform:uppercase;font-weight:700;">Analyzed Domain</div>
                    <div style="font-size:22px;font-weight:800;color:#ffffff;">{cred_data['domain']}</div>
                    <div style="font-size:12px;color:#8b9bb4;margin-top:2px;">Category: <span style="color:#ffffff;">{cred_data.get('category','News')}</span> | Bias: <span style="color:#ffffff;">{cred_data['bias_label']}</span></div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;color:#8b9bb4;text-transform:uppercase;font-weight:700;">Credibility Score</div>
                    <div style="font-size:32px;font-weight:800;color:{badge_color};font-family:'JetBrains Mono',monospace;">{c_pct}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Multi-Modal Composite Risk Integration Formula")
        st.markdown(f"""
        $$\\text{{Final Composite Risk}} = 0.75 \\times P_{{\\text{{ensemble}}}} + 0.25 \\times (1 - \\text{{Credibility}}) = 0.75 \\times {ensemble_prob:.2f} + 0.25 \\times (1 - {c_score:.2f}) = \\mathbf{{{final_risk_score:.2f}}}$$
        """)
    else:
        st.info("Enter a URL in the top input box to check domain reputation and credibility scoring.")
