import streamlit as st
from app.ui.utils import render_html

def render_whatsapp_preview(data):
    """Renders a responsive WhatsApp phone mockup with live verification data."""
    if not data:
        return

    risk_level = data.get("risk_level", "Unknown")
    probability = data.get("risk_score", 0.5)
    claim = data.get("claim", "")
    verdict = data.get("verdict", "")
    sources = data.get("sources", [])

    risk_emoji = "🛡️"
    if risk_level == "Critical": risk_emoji = "🔴"
    elif risk_level == "High": risk_emoji = "🟠"
    elif risk_level == "Moderate": risk_emoji = "🟡"
    elif risk_level == "Low": risk_emoji = "🟢"

    prob_pct = int(probability * 100)
    sources_text = "\n".join([f"• {s['name']}: {s['url']}" for s in sources[:2]])
    if not sources_text:
        sources_text = "• No external sources found."

    html = f"""
        <div class="phone-wrap" style="width: 100%; max-width: 290px; margin: 0 auto;">
            <div style="background: #0B141A; border: 6px solid #1a1a1a; border-radius: 36px; position: relative; padding: 10px; box-shadow: var(--shadow-lg);">
                <div style="height: 50px; background: #075E54; display: flex; align-items: center; padding: 0 10px; gap: 10px; border-radius: 20px 20px 0 0;">
                    <div style="width: 28px; height: 28px; background: linear-gradient(135deg, #6C63F0, #2FCC93); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 14px;">🔍</div>
                    <div style="color: white;">
                        <div style="font-weight: 700; font-size: 12px;">RiskLens Bot</div>
                        <div style="font-size: 10px; opacity: 0.8;">Online</div>
                    </div>
                </div>
                <div style="height: 440px; background: #E5DDD5; padding: 10px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; background-image: radial-gradient(circle,rgba(0,0,0,.02) 1px, transparent 1px); background-size: 14px 14px;">
                    <div style="align-self: flex-start; background: white; padding: 8px 12px; border-radius: 0 10px 10px 10px; font-size: 11.5px; max-width: 90%; box-shadow: 0 1px 1px rgba(0,0,0,0.1); white-space: pre-wrap; font-family: sans-serif; color: #111;">
                        <strong>🔍 RiskLens Verdict</strong>
                        <div style="border-top:1px dashed #ccc; margin:6px 0;"></div>
                        🎯 <b>Risk Level:</b> <span style="display:inline-block; padding:2px 8px; border-radius:6px; font-size:10.5px; font-weight:700; background:var(--high-soft); color:var(--high); margin:2px 0;">{risk_level} ({prob_pct}%)</span><br><br>
                        📋 <b>Claim Detected:</b><br>"{claim}"<br><br>
                        ⚠️ <b>Verdict:</b><br>{verdict}<br><br>
                        🔎 <b>Reasoning:</b> Evidence-grounded analysis complete.
                        <div style="border-top:1px dashed #ccc; margin:6px 0;"></div>
                        Was this helpful? 👍 Correct | 👎 Wrong
                        <div style="text-align:right; font-size:9px; color:#888; margin-top:4px;">10:42 AM ✓✓</div>
                    </div>
                </div>
                <div style="height: 40px; background: #ECE5DD; border-radius: 0 0 20px 20px; padding: 5px 10px; display: flex; gap: 5px; align-items: center;">
                    <div style="flex: 1; height: 28px; background: white; border-radius: 14px; padding: 0 12px; display: flex; align-items: center; color: #999; font-size: 12px;">Type a message</div>
                    <div style="width: 28px; height: 28px; background: #128C7E; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-size: 12px;">🎤</div>
                </div>
            </div>
        </div>
    """
    render_html(html)
