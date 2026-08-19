"""
app/components/primary_card.py
==============================
Modern High-Impact Risk Card with SHAP Saliency & Verified Sources.
"""

import streamlit as st
from app.config import RISK_CONFIG

def render_primary_card(
    risk_level: str,
    probability: float,
    claim_text: str = "",
    verdict: str = "",
    suspicious_phrases: list = None,
    shap_data: list = None,
    sources: list = None
):
    """
    Renders the redesigned, modern risk intelligence card.
    """
    cfg = RISK_CONFIG.get(risk_level, RISK_CONFIG["Moderate"])
    prob_pct = int(round(probability * 100))

    html = f"""
<div style="background:{cfg['bg']}; border:1px solid {cfg['border']}; border-radius:18px; padding:24px; margin-bottom:24px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:24px;">
        <div style="font-size:14px; font-weight:700; color:#e6edf3;">Verification result</div>
        <div style="display:flex; align-items:center; gap:16px;">
            <div style="background:{cfg['badge_bg']}; color:{cfg['badge_text']}; padding:6px 16px; border-radius:20px; font-size:12px; font-weight:700;">{risk_level} risk · Misinformation</div>
            <div style="font-size:32px; font-weight:800; color:#ffffff;">{prob_pct}% <span style="font-size:12px; color:#8b9bb4; font-weight:500;">misinformation probability</span></div>
        </div>
    </div>

    <div style="margin-bottom:24px;">
        <div style="font-size:11px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
            <span>📋 CLAIM DETECTED</span>
        </div>
        <div style="background:rgba(13, 20, 34, 0.4); border-radius:12px; padding:16px; font-size:14px; color:#e6edf3; font-style:italic; border-left:4px solid {cfg['border']};">
            "{claim_text or 'No claim text provided.'}"
        </div>
    </div>

    <div style="margin-bottom:24px;">
        <div style="font-size:11px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; display:flex; align-items:center; gap:6px;">
            <span>⚖️ VERDICT</span>
        </div>
        <div style="font-size:14px; color:#e6edf3; line-height:1.6;">
            {verdict or 'Analysis pending for this claim.'}
        </div>
    </div>

    <div style="margin-bottom:24px;">
        <div style="font-size:11px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">🚩 SUSPICIOUS PHRASES</div>
        <div style="display:flex; flex-wrap:wrap; gap:8px;">
            {''.join([f'<span class="chip" style="background:{cfg["badge_bg"]}; color:{cfg["badge_text"]}; border:1px solid {cfg["border"]}33;">{p}</span>' for p in (suspicious_phrases or ["doctors are hiding", "secret remedy", "protect your family", "coronavirus cure"])])}
        </div>
    </div>

    <div style="margin-bottom:24px;">
        <div style="font-size:11px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">📊 WHY IT WAS FLAGGED (SHAP)</div>
        {''.join([f'''
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
            <div style="font-size:12px; color:#e6edf3; width:120px;">{item['feature']}</div>
            <div style="flex-grow:1; background:rgba(31, 41, 55, 0.5); height:8px; border-radius:4px; overflow:hidden;">
                <div style="width:{int(item['value']*100)}%; background:{cfg['border']}; height:100%;"></div>
            </div>
            <div style="font-size:12px; color:#8b9bb4; width:40px; text-align:right;">+{item['value']:.2f}</div>
        </div>
        ''' for item in (shap_data or [
            {"feature": "Conspiracy language", "value": 0.88},
            {"feature": "Health claim", "value": 0.73},
            {"feature": "Fear appeal", "value": 0.61},
            {"feature": "No citations", "value": 0.45},
        ])])}
    </div>

    <div style="margin-bottom:24px;">
        <div style="font-size:11px; font-weight:700; color:#8b9bb4; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">🌐 VERIFIED SOURCES</div>
        {''.join([f'''
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.03);">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:10px; height:10px; border-radius:50%; background:#3fb950;"></div>
                <div>
                    <div style="font-size:13px; font-weight:700; color:#ffffff;">{s['name']} <span style="font-size:11px; color:#8b9bb4; font-weight:400; margin-left:8px;">{s['status']}</span></div>
                </div>
            </div>
            <a href="{s['url']}" style="font-size:12px; color:#58a6ff; text-decoration:none;">{s['domain']} ↗</a>
        </div>
        ''' for s in (sources or [
            {"name": "WHO", "status": "Debunked April 2020", "url": "#", "domain": "who.int"},
            {"name": "Reuters Fact Check", "status": "No scientific basis", "url": "#", "domain": "reuters.com"},
            {"name": "Snopes", "status": "Rating: False", "url": "#", "domain": "snopes.com"},
        ])])}
    </div>

    <div style="display:flex; gap:12px;">
        <div style="flex:1; background:rgba(13, 20, 34, 0.6); border:1px solid #1f2c44; border-radius:10px; padding:10px; text-align:center; color:#e6edf3; font-size:13px; font-weight:600; cursor:pointer;">✓ Correct</div>
        <div style="flex:1; background:rgba(13, 20, 34, 0.6); border:1px solid #1f2c44; border-radius:10px; padding:10px; text-align:center; color:#e6edf3; font-size:13px; font-weight:600; cursor:pointer;">✕ Wrong</div>
        <div style="flex:1; background:rgba(13, 20, 34, 0.6); border:1px solid #1f2c44; border-radius:10px; padding:10px; text-align:center; color:#e6edf3; font-size:13px; font-weight:600; cursor:pointer;">📋 Share card</div>
    </div>
</div>
""".strip()
    st.markdown(html, unsafe_allow_html=True)
