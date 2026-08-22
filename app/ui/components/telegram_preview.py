"""
app/ui/components/telegram_preview.py
=====================================
Pixel-perfect Telegram channel phone frame mockup for RiskLens v2.1.0.
Fully synchronized with risklens.telegram_bot formatting and labeling.
"""

from typing import Dict, Any
import streamlit as st
from app.ui.utils import render_html
from risklens.telegram_bot import format_telegram_report


def render_telegram_preview(data: Dict[str, Any]):
    """Renders the exact phone mockup preview matching the live Telegram bot output."""
    if not data:
        return

    # Use unified formatter to ensure 100% synchronization with real bot output
    formatted = format_telegram_report(data)
    risk_level = formatted["risk_level"]
    prob_pct = formatted["prob_pct"]
    claim = formatted["claim"]
    verdict = formatted["verdict"]

    if risk_level == "Critical":
        risk_color = "var(--critical)"
        risk_bg = "var(--critical-soft)"
    elif risk_level == "High":
        risk_color = "var(--high)"
        risk_bg = "var(--high-soft)"
    elif risk_level == "Moderate":
        risk_color = "var(--moderate)"
        risk_bg = "var(--moderate-soft)"
    else:  # Low
        risk_color = "var(--low)"
        risk_bg = "var(--low-soft)"

    # Truncate for clean mobile presentation
    display_claim = (claim[:130] + "...") if len(claim) > 130 else claim
    display_verdict = (verdict[:280] + "...") if len(verdict) > 280 else verdict

    html = f"""
    <div>
        <div class="tg-panel-label">Telegram channel preview</div>
        <div class="phone">
            <div class="phone-screen">
                <div class="tg-header">
                    <div class="tg-avatar">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2 3 6v6c0 5 3.8 8.7 9 10 5.2-1.3 9-5 9-10V6l-9-4Z"/></svg>
                    </div>
                    <div>
                        <div style="line-height: 1.2;">RiskLens Bot</div>
                        <div style="font-size: 10px; color: #5EB5F7; font-weight: 500;">verified channel</div>
                    </div>
                </div>
                <div class="tg-body">
                    <div class="tg-bubble">
                        <div class="tg-row">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4M12 17h.01M10.3 3.9 2.5 17a2 2 0 0 0 1.7 3h15.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/></svg>
                            <span><b>Risk assessment:</b> <span class="tg-risk-tag" style="background: {risk_bg}; color: {risk_color};">{risk_level}</span></span>
                        </div>
                        <div class="tg-row">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"/><path d="M14 2v6h6M9 13h6M9 17h6"/></svg>
                            <span><b>Claim detected:</b><br>"{display_claim}"</span>
                        </div>
                        <div class="tg-row">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10 13a5 5 0 0 0 7.5.5l2-2a5 5 0 0 0-7-7l-1.2 1.2M14 11a5 5 0 0 0-7.5-.5l-2 2a5 5 0 0 0 7 7l1.2-1.2"/></svg>
                            <span><b>Verdict:</b><br>{display_verdict}</span>
                        </div>
                        <div class="tg-row">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18M8 17V9m5 8V5m5 12v-5"/></svg>
                            <span><b>Misinformation probability:</b> {prob_pct}%</span>
                        </div>
                        <div style="font-size: 11px; color: #9BA7B4; margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06);">
                            Was this intelligence accurate?
                        </div>
                        <div class="tg-kb">
                            <div class="tg-kb-btn">✅ Correct</div>
                            <div class="tg-kb-btn">❌ Wrong</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    render_html(html)
