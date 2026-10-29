@echo off
title AI-Based Network Attack Forecasting - Web Dashboard
cd /d "%~dp0"
echo =========================================================================
echo   AI-Based Network Attack Forecasting World Model - Web Server
echo =========================================================================
echo.
echo Starting Web Server on http://127.0.0.1:8080 ...
echo Opening your browser to http://127.0.0.1:8080 ...
start http://127.0.0.1:8080
python -m uvicorn app:app --host 127.0.0.1 --port 8080
pause
