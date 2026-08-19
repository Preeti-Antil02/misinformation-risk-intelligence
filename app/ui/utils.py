import streamlit as st
import re
import textwrap

def render_html(html_str):
    """
    Cleans and renders HTML string to ensure it doesn't trigger
    Streamlit's markdown code block formatting.
    """
    # 1. Dedent to remove leading whitespace
    clean = textwrap.dedent(html_str).strip()

    # 2. Remove HTML comments
    clean = re.sub(r'<!--.*?-->', '', clean, flags=re.DOTALL)

    # 3. Remove multiple newlines which break markdown blocks
    # We replace any sequence of 2 or more newlines with a single newline
    clean = re.sub(r'\n\s*\n', '\n', clean)

    # 4. Remove all leading whitespace from every line to be absolutely sure
    clean = "\n".join([line.strip() for line in clean.split("\n")])

    st.markdown(clean, unsafe_allow_html=True)
