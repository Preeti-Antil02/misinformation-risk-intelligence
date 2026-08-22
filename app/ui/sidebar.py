import os
import streamlit as st
import sqlite3
import textwrap
from pathlib import Path
from datetime import date
from risklens import __version__ as APP_VERSION
from risklens.feedback import calculate_live_accuracy
from app.ui.utils import render_html

def get_db_path() -> Path:
    db_dir = Path(os.getenv("DATABASE_DIR", "databases"))
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "feedback.db"

def get_db_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn

def render_sidebar():
    """Renders the high-end production sidebar with live telemetry."""
    with st.sidebar:
        # 1. Brand Header
        html_head = f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:28px;">
                <div style="width:40px; height:40px; border-radius:12px; background:var(--primary-gradient); display:flex; align-items:center; justify-content:center; box-shadow:var(--shadow-sm);">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5"><path d="M12 2 3 6v6c0 5 3.8 8.7 9 10 5.2-1.3 9-5 9-10V6l-9-4Z"/><path d="m9 12 2 2 4-4"/></svg>
                </div>
                <div>
                    <div style="font-size:19px; font-weight:800; color:var(--text); letter-spacing:-.02em; line-height:1;">RiskLens</div>
                    <div style="font-size:11px; color:var(--text-faint); margin-top:3px; font-weight:600;">Intelligence v{APP_VERSION}</div>
                </div>
            </div>
        """
        render_html(html_head)

        # 2. Live Data Query (All-Time Primary + Today Secondary)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            today = date.today().isoformat()

            # All-time total verifications + today's count
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_verifs = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE timestamp LIKE ?", (f"{today}%",))
            verifs_today = cursor.fetchone()[0]

            # All-time risk alerts (High/Critical) + today's alerts
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE risk_level IN ('High', 'Critical')")
            total_risk_alerts = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE timestamp LIKE ? AND risk_level IN ('High', 'Critical')", (f"{today}%",))
            risk_alerts_today = cursor.fetchone()[0]

            # All-time language distribution and active languages
            cursor.execute("SELECT language, COUNT(*) as count FROM predictions GROUP BY language ORDER BY count DESC")
            all_lang_counts = cursor.fetchall()
            all_lang_counts = [dict(r) for r in all_lang_counts]
            total_langs_active = len(all_lang_counts)

            cursor.execute("SELECT text, risk_level, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 5")
            recent_verifs = cursor.fetchall()
            conn.close()
        except Exception:
            total_verifs, verifs_today = 0, 0
            total_risk_alerts, risk_alerts_today = 0, 0
            all_lang_counts, total_langs_active, recent_verifs = [], 0, []

        live_acc_data = calculate_live_accuracy()
        total_feedback = live_acc_data.get("total_feedback", 0)
        live_acc = live_acc_data.get("live_accuracy", 0.0)

        # Formatted card headline and subtext values
        c1_num = f"{total_verifs:,}" if total_verifs > 0 else "0"
        c1_sub = f"{verifs_today} today" if total_verifs > 0 else "No claims yet"

        c2_num = f"{total_risk_alerts:,}" if total_risk_alerts > 0 else "0"
        c2_sub = f"{risk_alerts_today} today" if total_verifs > 0 else "All-time alerts"

        if total_feedback == 0:
            c3_num = "Pending"
            c3_sub = "0 ratings yet"
        elif total_feedback < 3:
            c3_num = f"{live_acc*100:.0f}%*"
            c3_sub = f"{total_feedback} rating{'s' if total_feedback > 1 else ''}"
        else:
            c3_num = f"{live_acc*100:.1f}%"
            c3_sub = f"{total_feedback} ratings"

        c4_num = str(total_langs_active)
        c4_sub = "All-time regional"

        # 3. Stat Grid
        html_stats = f"""
            <div class="stat-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border);">
                    <div class="stat-num" style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:var(--text);">{c1_num}</div>
                    <div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:4px; text-transform:uppercase; font-weight:600;">Total Verified</div>
                    <div style="font-size:9.5px; color:var(--text-faint); margin-top:2px; font-weight:500;">{c1_sub}</div>
                </div>
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border); border-bottom: 2px solid var(--critical);">
                    <div class="stat-num" style="color:var(--critical); font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700;">{c2_num}</div>
                    <div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:4px; text-transform:uppercase; font-weight:600;">Risk Alerts</div>
                    <div style="font-size:9.5px; color:var(--text-faint); margin-top:2px; font-weight:500;">{c2_sub}</div>
                </div>
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border);">
                    <div class="stat-num" style="font-family:'JetBrains Mono',monospace; font-size:{'16px' if total_feedback == 0 else '20px'}; font-weight:700; color:{'var(--text-muted)' if total_feedback == 0 else 'var(--text)'};">{c3_num}</div>
                    <div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:4px; text-transform:uppercase; font-weight:600;">Live Accuracy</div>
                    <div style="font-size:9.5px; color:var(--text-faint); margin-top:2px; font-weight:500;">{c3_sub}</div>
                </div>
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border);">
                    <div class="stat-num" style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:var(--text);">{c4_num}</div>
                    <div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:4px; text-transform:uppercase; font-weight:600;">Langs Active</div>
                    <div style="font-size:9.5px; color:var(--text-faint); margin-top:2px; font-weight:500;">{c4_sub}</div>
                </div>
            </div>
        """
        render_html(html_stats)

        # 4. Accuracy Ring
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if total_feedback == 0:
            ring_bg = "background:conic-gradient(var(--border) 0deg 360deg);"
            center_label = "--"
            title_text = "Awaiting Feedback"
            sub_text = "Rate verifications to calibrate"
        elif total_feedback < 3:
            deg = max(1, int(live_acc * 360))
            ring_bg = f"background:conic-gradient(var(--low) 0deg {deg}deg, var(--border) {deg}deg 360deg);"
            center_label = f"{int(live_acc*100)}%"
            title_text = f"{live_acc*100:.0f}% accurate"
            sub_text = f"Early signal ({total_feedback} rating{'s' if total_feedback > 1 else ''})"
        else:
            deg = max(1, int(live_acc * 360))
            ring_bg = f"background:conic-gradient(var(--low) 0deg {deg}deg, var(--border) {deg}deg 360deg);"
            center_label = f"{int(live_acc*100)}%"
            title_text = f"{live_acc*100:.1f}% accurate"
            sub_text = f"Based on {total_feedback} live user ratings"

        html_fidelity = f"""
            <div class="side-block" style="background:var(--surface-alt); border-radius:var(--radius-md); padding:16px; border:1px solid var(--border);">
                <div class="side-block-title" style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); margin-bottom:12px;">System Fidelity</div>
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="width:64px; height:64px; border-radius:50%; flex-shrink:0; {ring_bg} display:flex; align-items:center; justify-content:center; position:relative;">
                        <div style="position:absolute; inset:6px; border-radius:50%; background:var(--surface-alt);"></div>
                        <div style="position:relative; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:14px; color:var(--text);">{center_label}</div>
                    </div>
                    <div>
                        <div style="font-size:15px; font-weight:700; color:var(--text);">{title_text}</div>
                        <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">{sub_text}</div>
                    </div>
                </div>
            </div>
        """
        render_html(html_fidelity)

        # 5. Language Usage
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='side-block-title' style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); margin-bottom:12px; margin-left: 5px;'>Language Distribution</div>", unsafe_allow_html=True)
        total_lang_samples = sum(row['count'] for row in all_lang_counts)
        if total_lang_samples == 0:
            render_html("""
                <div style="font-size:11.5px; color:var(--text-muted); padding: 4px 6px; line-height: 1.4;">
                    No claims recorded yet. Verify text, URLs, or screenshots above.
                </div>
            """)
        else:
            for row in all_lang_counts[:4]:
                pct = (row['count'] / total_lang_samples) * 100
                html_lang = f"""
                    <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px; padding: 0 5px;">
                        <span style="width:30px; color:var(--text-muted); font-weight:700; font-size:11px; text-transform:uppercase;">{row['language']}</span>
                        <div style="flex:1; height:6px; background:var(--border); border-radius:4px; overflow:hidden;"><div style="height:100%; width:{pct}%; border-radius:4px; background:linear-gradient(90deg,var(--primary),var(--low));"></div></div>
                        <span style="width:34px; text-align:right; color:var(--text-faint); font-family:'JetBrains Mono',monospace; font-size:10.5px;">{int(pct)}%</span>
                    </div>
                """
                render_html(html_lang)

        # 6. Telegram CTA (Replacing WhatsApp for Security)
        st.markdown("<div style='height:32px;'></div>", unsafe_allow_html=True)
        bot_username = os.getenv("TELEGRAM_BOT_USERNAME", "RiskLensVerifyBot").strip().lstrip("@")
        app_deep_link = f"tg://resolve?domain={bot_username}"
        web_url = f"https://web.telegram.org/k/#@{bot_username}"

        html_tg = f"""
            <div class="wa-cta" style="border-radius:var(--radius-md); padding:16px; background:linear-gradient(135deg,#0088cc,#00aaff); color:#fff; position:relative; overflow:hidden;">
                <div class="wa-cta-title" style="font-weight:700; font-size:13.5px; display:flex; align-items:center; gap:8px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                    Verify on Telegram
                </div>
                <p style="font-size:11.5px; opacity:.9; margin:8px 0 10px; line-height:1.4;">Instant neural verification for claims, forwarded messages, and screenshots.</p>
                <div style="font-size:11px; opacity:.85; font-family:'JetBrains Mono',monospace; margin-bottom:12px;">Handle: <b>@{bot_username}</b></div>
                <div style="display:flex; gap:8px; flex-wrap:wrap;">
                    <a href="{app_deep_link}" target="_blank" style="text-decoration:none;">
                        <div class="wa-cta-btn" style="background:#fff; color:#0088cc; font-size:11.5px; font-weight:800; padding:8px 14px; border-radius:8px; display:inline-flex; align-items:center; gap:5px; transition:transform 0.2s;">Open in App →</div>
                    </a>
                    <a href="{web_url}" target="_blank" style="text-decoration:none;">
                        <div class="wa-cta-btn" style="background:rgba(255,255,255,0.2); color:#fff; font-size:11.5px; font-weight:700; padding:8px 12px; border-radius:8px; display:inline-flex; align-items:center; gap:5px; border:1px solid rgba(255,255,255,0.4);">Web Client</div>
                    </a>
                </div>
            </div>
        """
        render_html(html_tg)
