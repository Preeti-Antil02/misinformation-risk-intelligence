"""
app/ui/components/telegram_popover.py
======================================
Enterprise Viewport-Aware Popover & Dropdown Engine for RiskLens.
- Pure client-side dynamic boundary detection.
- Flips horizontal alignment (right-anchor, left-anchor, or center) based on available screen space.
- Automatic mobile viewport centered sheet fallback (<520px).
- Click outside / Escape key dismissal.
- Zero parent-overflow clipping via fixed coordinate positioning.
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

    html_content = f"""
    <div class="tg-popover-wrapper" style="display:inline-block; text-align:right;">
        <!-- Popover Trigger Button -->
        <button type="button" class="tg-header-badge-btn" id="tg_popover_trigger" aria-haspopup="dialog" aria-expanded="false" title="RiskLens Telegram Intelligence Bot">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
            <span>Bot</span>
            <span class="tg-pulse" style="width:5px; height:5px; margin-left: 2px;"></span>
        </button>

        <!-- Popover Floating Card (Positioned Dynamically via JS) -->
        <div class="rl-popover" id="tg_popover_dialog" role="dialog" aria-modal="true" aria-label="RiskLens Telegram Bot">
            <div class="rl-popover-header">
                <div style="display:flex; align-items:center; gap:10px;">
                    <div class="rl-popover-avatar">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:13.5px; color:var(--text); display:flex; align-items:center; gap:6px;">
                            RiskLens Bot
                            <span class="tg-online-badge" style="margin-left:0; font-size:9px; padding:1px 6px;">
                                <span class="tg-pulse" style="width:5px; height:5px;"></span>Live
                            </span>
                        </div>
                        <div style="font-size:11px; color:var(--text-muted);">Neural Verification Suite</div>
                    </div>
                </div>
                <button type="button" class="rl-popover-close" id="tg_popover_close" aria-label="Close Popover">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                </button>
            </div>

            <p style="font-size:11.5px; color:var(--text-muted); line-height:1.45; margin:0 0 10px 0;">
                Forward viral forwards, claims, or screenshots directly for real-time meta-ensemble verification.
            </p>

            <div class="rl-popover-handle-box">
                <span>@{bot_username}</span>
                <button type="button" class="rl-popover-copy-btn" id="tg_copy_handle_btn" data-handle="@{bot_username}">
                    Copy
                </button>
            </div>

            <div class="rl-popover-actions">
                <a href="{app_deep_link}" target="_blank" class="rl-popover-btn-app" id="tg_popover_open_app" title="Launch Telegram Desktop/Mobile App directly">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                    <span>Open in Telegram App</span>
                </a>
                <a href="{web_url}" target="_blank" class="rl-popover-btn-web" id="tg_popover_open_web" title="Open Telegram Web client in browser (ISP-safe)">
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1 4-10z"/></svg>
                    <span>Web Client</span>
                </a>
            </div>
        </div>
    </div>
    """

    js_script = """
    <script>
    (function() {
        function initPopover() {
            var trigger = document.getElementById('tg_popover_trigger');
            var popover = document.getElementById('tg_popover_dialog');
            var closeBtn = document.getElementById('tg_popover_close');
            var copyBtn = document.getElementById('tg_copy_handle_btn');

            if (!trigger || !popover) return;

            // Move to body so it escapes all container overflow constraints
            if (popover.parentElement !== document.body) {
                document.body.appendChild(popover);
            }

            var PADDING = 12;
            var GAP = 8;
            var DESIRED_WIDTH = 320;

            function positionPopover() {
                if (!popover.classList.contains('open')) return;

                var triggerRect = trigger.getBoundingClientRect();
                var vw = window.innerWidth;
                var vh = window.innerHeight;

                var cardWidth = Math.min(DESIRED_WIDTH, vw - (PADDING * 2));
                popover.style.width = cardWidth + 'px';

                // MOBILE FALLBACK (< 520px): Center in viewport
                if (vw < 520) {
                    var left = Math.max(PADDING, (vw - cardWidth) / 2);
                    var top = triggerRect.bottom + GAP;

                    if (top + popover.offsetHeight > vh - PADDING) {
                        top = Math.max(PADDING, triggerRect.top - popover.offsetHeight - GAP);
                    }

                    popover.style.left = Math.round(left) + 'px';
                    popover.style.top = Math.round(top) + 'px';
                    popover.style.right = 'auto';
                    popover.style.bottom = 'auto';
                    return;
                }

                // TABLET & DESKTOP: Dynamic Boundary-Aware Anchoring
                var left = triggerRect.right - cardWidth;

                // Flip if overflows left
                if (left < PADDING) {
                    left = triggerRect.left;
                }

                // Adjust if overflows right
                if (left + cardWidth > vw - PADDING) {
                    left = vw - cardWidth - PADDING;
                }

                // Clamp strictly within viewport
                left = Math.max(PADDING, Math.min(left, vw - cardWidth - PADDING));

                var top = triggerRect.bottom + GAP;
                if (top + popover.offsetHeight > vh - PADDING) {
                    top = Math.max(PADDING, triggerRect.top - popover.offsetHeight - GAP);
                }

                popover.style.left = Math.round(left) + 'px';
                popover.style.top = Math.round(top) + 'px';
                popover.style.right = 'auto';
                popover.style.bottom = 'auto';
            }

            function openPopover() {
                popover.classList.add('open');
                trigger.setAttribute('aria-expanded', 'true');
                positionPopover();
            }

            function closePopover() {
                popover.classList.remove('open');
                trigger.setAttribute('aria-expanded', 'false');
            }

            function togglePopover(e) {
                e.stopPropagation();
                if (popover.classList.contains('open')) {
                    closePopover();
                } else {
                    openPopover();
                }
            }

            trigger.onclick = togglePopover;

            if (closeBtn) {
                closeBtn.onclick = function(e) {
                    e.stopPropagation();
                    closePopover();
                    trigger.focus();
                };
            }

            document.addEventListener('pointerdown', function(e) {
                if (!popover.classList.contains('open')) return;
                if (popover.contains(e.target) || trigger.contains(e.target)) return;
                closePopover();
            });

            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape' && popover.classList.contains('open')) {
                    closePopover();
                    trigger.focus();
                }
            });

            window.addEventListener('resize', positionPopover, { passive: true });
            window.addEventListener('scroll', positionPopover, { passive: true, capture: true });

            if (copyBtn) {
                copyBtn.onclick = function(e) {
                    e.stopPropagation();
                    var handle = copyBtn.getAttribute('data-handle') || '@RiskLensIntelligenceBot';
                    if (navigator.clipboard) {
                        navigator.clipboard.writeText(handle).then(function() {
                            copyBtn.textContent = '✓ Copied';
                            setTimeout(function() { copyBtn.textContent = 'Copy'; }, 2000);
                        });
                    }
                };
            }
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initPopover);
        } else {
            initPopover();
        }
        setTimeout(initPopover, 400);
    })();
    </script>
    """

    render_html(html_content + js_script)
