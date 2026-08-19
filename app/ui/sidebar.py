import streamlit as st
import sqlite3
import textwrap
from pathlib import Path
from datetime import date
from risklens import __version__ as APP_VERSION
from risklens.feedback import calculate_live_accuracy
from app.ui.utils import render_html

DB_PATH = Path("databases/feedback.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
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

        # 2. Live Data Query
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            today = date.today().isoformat()

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE timestamp LIKE ?", (f"{today}%",))
            verifs_today = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM predictions WHERE timestamp LIKE ? AND (risk_level = 'High' OR risk_level = 'Critical')", (f"{today}%",))
            high_crit_today = cursor.fetchone()[0]

            cursor.execute("SELECT language, COUNT(*) as count FROM predictions WHERE timestamp LIKE ? GROUP BY language ORDER BY count DESC", (f"{today}%",))
            lang_counts = cursor.fetchall()

            cursor.execute("SELECT text, risk_level, timestamp FROM predictions ORDER BY timestamp DESC LIMIT 5")
            recent_verifs = cursor.fetchall()
            conn.close()
        except Exception:
            verifs_today, high_crit_today, lang_counts, recent_verifs = 0, 0, [], []

        live_acc_data = calculate_live_accuracy()
        live_acc = live_acc_data.get("live_accuracy", 0.0)

        # 3. Stat Grid
        html_stats = f"""
            <div class="stat-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border);"><div class="stat-num" style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:var(--text);">{verifs_today:,}</div><div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:5px; text-transform:uppercase; font-weight:600;">Verified today</div></div>
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border); border-bottom: 2px solid var(--critical);"><div class="stat-num" style="color:var(--critical); font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700;">{high_crit_today}</div><div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:5px; text-transform:uppercase; font-weight:600;">Risk Alerts</div></div>
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border);"><div class="stat-num" style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:var(--text);">{live_acc*100:.1f}%</div><div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:5px; text-transform:uppercase; font-weight:600;">Live Accuracy</div></div>
                <div class="stat-card" style="background:var(--surface-alt); border-radius:var(--radius-sm); padding:12px; border:1px solid var(--border);"><div class="stat-num" style="font-family:'JetBrains Mono',monospace; font-size:20px; font-weight:700; color:var(--text);">{len(lang_counts)}</div><div class="stat-label" style="font-size:10px; color:var(--text-muted); margin-top:5px; text-transform:uppercase; font-weight:600;">Langs Active</div></div>
            </div>
        """
        render_html(html_stats)

        # 4. Accuracy Ring
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        deg = int(live_acc * 360)
        html_fidelity = f"""
            <div class="side-block" style="background:var(--surface-alt); border-radius:var(--radius-md); padding:16px; border:1px solid var(--border);">
                <div class="side-block-title" style="font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); margin-bottom:12px;">System Fidelity</div>
                <div style="display:flex; align-items:center; gap:16px;">
                    <div style="width:64px; height:68px; border-radius:50%; flex-shrink:0; background:conic-gradient(var(--low) 0deg {deg}deg, var(--border) {deg}deg 360deg); display:flex; align-items:center; justify-content:center; position:relative;">
                        <div style="position:absolute; inset:6px; border-radius:50%; background:var(--surface-alt);"></div>
                        <div style="position:relative; font-family:'JetBrains Mono',monospace; font-weight:700; font-size:14px; color:var(--text);">{int(live_acc*100)}%</div>
                    </div>
                    <div>
                        <div style="font-size:15px; font-weight:700; color:var(--text);">{live_acc*100:.1f}% accurate</div>
                        <div style="font-size:11px; color:var(--text-muted); margin-top:2px;">Based on live user feedback</div>
                    </div>
                </div>
            </div>
        """
        render_html(html_fidelity)

        # 5. Language Usage
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='side-block-title' style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); margin-bottom:12px; margin-left: 5px;'>Language Distribution</div>", unsafe_allow_html=True)
        total_verifs = sum(row['count'] for row in lang_counts) if lang_counts else 1
        for row in lang_counts[:4]:
            pct = (row['count'] / total_verifs) * 100
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
        html_tg = f"""
            <div class="wa-cta" style="border-radius:var(--radius-md); padding:16px; background:linear-gradient(135deg,#0088cc,#00aaff); color:#fff; position:relative; overflow:hidden;">
                <div class="wa-cta-title" style="font-weight:700; font-size:13.5px; display:flex; align-items:center; gap:8px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg>
                    Verify on Telegram
                </div>
                <p style="font-size:11.5px; opacity:.9; margin:8px 0 14px; line-height:1.4;">Access high-security instant verification for claims and screenshots.</p>
                <a href="https://t.me/RiskLensBot" target="_blank" style="text-decoration:none;">
                    <div class="wa-cta-btn" style="background:#fff; color:#0088cc; font-size:12px; font-weight:800; padding:10px 16px; border-radius:10px; display:inline-flex; align-items:center; gap:6px; transition:transform 0.2s;">Launch Secure Bot →</div>
                </a>
            </div>
        """
        render_html(html_tg)
