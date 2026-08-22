import streamlit as st
import os
import sqlite3
import logging
from pathlib import Path
from dotenv import load_dotenv, set_key
from app.ui.utils import render_html

logger = logging.getLogger(__name__)

ENV_PATH = Path(".env")
USAGE_DB = Path("usage.db")

def get_usage_connection():
    """Returns usage DB connection with locking resilience and auto-schema initialization."""
    try:
        conn = sqlite3.connect(USAGE_DB, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                usage_date TEXT NOT NULL,
                request_count INTEGER DEFAULT 1,
                UNIQUE(user_id, usage_date)
            )
        """)
        conn.commit()
        return conn
    except Exception as e:
        logger.error(f"Usage DB connection failed: {str(e)}")
        return None

def render_tab_settings():
    """Renders the hardened settings and configuration tab."""
    st.markdown("<div class='page-head'><h1>System Configuration</h1><p>Manage production integrations, neural routing, and node security.</p></div>", unsafe_allow_html=True)

    # 1. API Integration (Telegram & Google Focus)
    st.markdown("### 🔑 Secure Integrations")

    load_dotenv()

    keys = {
        "TELEGRAM_BOT_TOKEN": "Telegram Bot Access Token",
        "GOOGLE_FACTCHECK_API_KEY": "Google Fact Check Intelligence",
        "SERPER_API_KEY": "Google Serper (Web Research) API"
    }

    with st.container():
        for key, label in keys.items():
            curr_val = os.getenv(key, "")
            new_val = st.text_input(label, value=curr_val, type="password", key=f"input_{key}", help=f"Persists to {ENV_PATH.name}")

            if new_val != curr_val:
                if st.button(f"Persist {label}", key=f"btn_{key}"):
                    try:
                        if not ENV_PATH.exists():
                            ENV_PATH.touch()
                        set_key(str(ENV_PATH), key, new_val)
                        st.success(f"✓ {label} synchronized successfully.")
                        logger.info(f"Configuration key {key} updated by user.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to write to .env: {str(e)}")
                        logger.error(f"Config write failure: {str(e)}")

        # Webhook Activation Shortcut
        if os.getenv("TELEGRAM_BOT_TOKEN"):
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            col_wh1, col_wh2 = st.columns([3, 1])
            with col_wh1:
                st.markdown(
                    "<div style='font-size: 13px; color: var(--text-secondary); padding-top: 8px;'>"
                    "🔗 <b>Telegram Bot Webhook</b>: Link your bot directly to this live Space for instant responses."
                    "</div>",
                    unsafe_allow_html=True
                )
            with col_wh2:
                if st.button("⚡ Activate Webhook", key="btn_sync_webhook", use_container_width=True):
                    with st.spinner("Connecting Telegram Bot to Webhook..."):
                        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
                        webhook_url = "https://preeti-antil-risklens.hf.space/telegram/webhook"
                        try:
                            import requests
                            r = requests.get(
                                f"https://api.telegram.org/bot{token}/setWebhook",
                                params={"url": webhook_url, "drop_pending_updates": True},
                                timeout=20
                            )
                            res = r.json()
                            if res.get("ok"):
                                st.success("✓ Telegram Bot webhook activated successfully!")
                            else:
                                st.error(f"Telegram API Error: {res.get('description')}")
                        except Exception as wh_err:
                            st.error(f"Failed to set webhook: {wh_err}")

    # 2. Intelligence Routing
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🌐 Intelligence Routing Control")
    st.markdown("<div style='font-size: 13px; color: var(--text-muted); margin-bottom: 20px;'>Toggle high-fidelity MuRIL transformer analysis for Indic regional flows.</div>", unsafe_allow_html=True)

    languages = {
        "hi": "Hindi Neural Cluster",
        "ta": "Tamil Neural Cluster",
        "te": "Telugu Neural Cluster",
        "bn": "Bengali Neural Cluster",
        "mr": "Marathi Neural Cluster"
    }

    cols = st.columns(2)
    for i, (code, label) in enumerate(languages.items()):
        with cols[i % 2]:
            st.toggle(label, value=True, key=f"route_{code}")

    # 3. Usage & Tiers (Hardened)
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Infrastructure Resilience")

    usage_today = 0
    conn = get_usage_connection()
    if conn:
        try:
            from datetime import date
            today = date.today().isoformat()
            row = conn.execute("SELECT SUM(request_count) FROM user_usage WHERE usage_date = ?", (today,)).fetchone()
            usage_today = row[0] if row and row[0] else 0
            conn.close()
        except Exception as e:
            logger.error(f"Settings usage query failed: {str(e)}")
    else:
        st.warning("⚠️ Usage monitoring node offline.")

    limit = 20
    pct = (usage_today / limit) * 100

    st.markdown(f"""
    <div class="card" style="padding: 24px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <div style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing:0.05em;">Security Tier</div>
                <div style="font-size: 18px; font-weight: 700; color: var(--primary);">Standard Node (Production)</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 12px; color: var(--text-muted); text-transform: uppercase; letter-spacing:0.05em;">Credit Consumption</div>
                <div style="font-size: 18px; font-weight: 700; color: var(--text);">{usage_today} / {limit}</div>
            </div>
        </div>
        <div style="background-color: var(--surface-alt); height: 8px; border-radius: 4px; overflow: hidden; margin-bottom: 12px;">
            <div style="background: linear-gradient(90deg, var(--primary) 0%, #34D399 100%); width: {min(pct, 100)}%; height: 100%;"></div>
        </div>
        <div style="font-size: 11px; color: var(--text-faint);">
            Credits reset at 00:00 UTC. Every scan consumes 1 neural compute credit.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4. Notifications
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🔔 Node Alerts")
    st.toggle("Push high-risk critical alerts to Telegram", value=True)
    st.toggle("Infrastructure health notifications", value=True)
    st.markdown("<div style='font-size: 11px; color: var(--text-faint); margin-top: 10px;'>Node alerts require the Telegram bot to be active.</div>", unsafe_allow_html=True)
