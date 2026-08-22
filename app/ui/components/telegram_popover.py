"""
app/ui/components/telegram_popover.py
======================================
Enterprise Viewport-Aware Popover & Dropdown Engine for RiskLens.
- Replaces previous large hero Telegram card with collapsed topbar 'Bot' pill popover.
- Uses HTML5 details/summary for 100% bulletproof client-side toggle without iframe script isolation issues.
- Smart responsive CSS: right-anchored on desktop, left-anchored on wrapped tablet, centered sheet on mobile (<520px).
"""

import os
from app.ui.utils import render_html


def render_telegram_popover():
    """
    Renders the Topbar Telegram Popover trigger and dynamic viewport-aware floating card.
    """
    bot_username = os.getenv("TELEGRAM_BOT_USERNAME", "RiskLensIntelligenceBot").strip().lstrip("@")
    app_deep_link = f"tg://resolve?domain={bot_username}"
    web_url = f"https://web.telegram.org/k/#@{bot_username}"

    html = f"""
    <div class="bot-pill-wrap">
        <details class="bot-pill-details">
            <summary class="bot-pill" id="botPillBtn" title="RiskLens Telegram Intelligence Bot">
                <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:14px; height:14px; flex-shrink:0;">
                    <path d="m22 2-7 20-4-9-9-4Z"/>
                </svg>
                <span>Bot</span>
                <span class="dot"></span>
            </summary>
            <div class="bot-popover" id="botPopover" role="dialog" aria-label="RiskLens Telegram Bot">
                <div class="bp-head">
                    <div class="bp-icon">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="width:16px; height:16px; color:#0A0910;">
                            <path d="m22 2-7 20-4-9-9-4Z"/>
                        </svg>
                    </div>
                    <div style="flex:1;">
                        <div class="bp-title" style="display:flex; align-items:center; justify-content:space-between;">
                            <span>Telegram Intelligence</span>
                            <span class="dot" style="width:5px; height:5px; border-radius:50%; background:var(--teal); display:inline-block;"></span>
                        </div>
                        <div class="bp-handle">@{bot_username}</div>
                    </div>
                </div>
                <p class="bp-desc">
                    Forward suspicious claims, articles, or screenshots directly to the bot for instant verification.
                </p>
                <div class="bp-actions">
                    <a href="{app_deep_link}" target="_blank" class="bp-btn primary" id="tg_popover_open_app" title="Launch Telegram Desktop/Mobile App directly">
                        Open in app
                    </a>
                    <a href="{web_url}" target="_blank" class="bp-btn secondary" id="tg_popover_open_web" title="Open Telegram Web client in browser (ISP-safe)">
                        Web client
                    </a>
                </div>
            </div>
        </details>
    </div>
    """
    render_html(html)
