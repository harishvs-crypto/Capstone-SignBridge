@echo off
title SignBridge ISL System Launcher
echo ==============================================
echo       SignBridge ISL-to-Speech Launcher
echo ==============================================
echo.
echo Launching Streamlit dashboard in virtual environment...
.\venv\Scripts\python.exe -m streamlit run app.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start Streamlit. Please verify that your virtual
    echo         environment is set up at .\venv\
    pause
)
