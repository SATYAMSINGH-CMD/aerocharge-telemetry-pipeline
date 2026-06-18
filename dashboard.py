"""Compatibility entry point for Streamlit Cloud or older run commands.

The main dashboard lives in app.py. Running `streamlit run dashboard.py` imports
that app so the project has one consistent analytics-first interface.
"""

import app  # noqa: F401
