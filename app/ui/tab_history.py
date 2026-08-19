import streamlit as st
import sqlite3
import pandas as pd
import textwrap
from pathlib import Path

DB_PATH = Path("databases/feedback.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def render_tab_history():
    """Renders the high-end verification history interface."""
    st.markdown("<div class='page-head'><h1>History</h1><p>Audit trail of all system verifications and risk assessments.</p></div>", unsafe_allow_html=True)

    if "history_filter" not in st.session_state:
        st.session_state.history_filter = "All"

    # 1. Filter Chips
    filters = ["All", "Critical", "High", "Moderate", "Low"]
    st.markdown("<div class='filter-row' style='display:flex; gap:8px; margin-bottom:18px; flex-wrap:wrap;'>", unsafe_allow_html=True)
    cols = st.columns(len(filters))
    for i, f in enumerate(filters):
        with cols[i]:
            is_active = st.session_state.history_filter == f
            if st.button(f, key=f"filter_{f}", use_container_width=True, type="primary" if is_active else "secondary"):
                st.session_state.history_filter = f
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    # 2. Fetch History (Real Data Only)
    try:
        conn = get_db_connection()
        query = "SELECT * FROM predictions"
        params = []
        if st.session_state.history_filter != "All":
            query += " WHERE risk_level = ?"
            params.append(st.session_state.history_filter)
        query += " ORDER BY timestamp DESC LIMIT 100"
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
    except Exception:
        df = pd.DataFrame()

    # 3. Premium Table Header
    st.markdown(textwrap.dedent("""
        <div class='card history-table' style='border-radius:var(--radius-lg); overflow:hidden;'>
            <div class='h-row head' style='display:grid; grid-template-columns:90px 1fr 110px 130px 90px; align-items:center; padding:14px 20px; background:var(--surface-alt); font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:var(--text-muted); gap:12px;'>
                <span>Risk</span>
                <span>Claim / text</span>
                <span>Language</span>
                <span>Model</span>
                <span>Time</span>
            </div>
    """).strip(), unsafe_allow_html=True)

    if df.empty:
        st.markdown("<div style='padding: 80px; text-align: center; color: var(--text-faint); font-family:var(--font-display);'>No real verification records found in the database.</div>", unsafe_allow_html=True)
    else:
        for _, row in df.iterrows():
            risk_cls = row['risk_level'].lower()
            # De-indenting row HTML to prevent code block rendering
            row_html = textwrap.dedent(f"""
                <div class="h-row" style="display:grid; grid-template-columns:90px 1fr 110px 130px 90px; align-items:center; padding:14px 20px; border-bottom:1px solid var(--border); gap:12px;">
                    <span class="h-badge {risk_cls}" style="display:inline-flex; align-items:center; gap:5px; padding:4px 10px; border-radius:8px; font-size:11px; font-weight:700; width:fit-content; background:var(--{risk_cls}-soft); color:var(--{risk_cls});">{row['risk_level']}</span>
                    <span class="h-text" style="font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:var(--text);">{row['text']}</span>
                    <span class="h-lang" style="font-size:12px; color:var(--text-muted);">{row['language'].upper()}</span>
                    <span class="h-lang" style="font-size:12px; color:var(--text-muted); font-weight:600;">{row['model_used'].split()[-1]}</span>
                    <span class="h-time" style="font-size:11.5px; color:var(--text-faint); font-family:'JetBrains Mono',monospace;">{row['timestamp'][11:16]}</span>
                </div>
            """).strip()
            st.markdown(row_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
