"""
app/streamlit_app.py
====================
Backwards-compatible entry point that routes directly to app/main.py.
"""

from app.main import main

if __name__ == "__main__":
    main()