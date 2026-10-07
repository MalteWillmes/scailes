@echo off
cd /d "%~dp0.."
uv run --no-project --with-requirements gui/requirements.txt streamlit run gui/app.py
