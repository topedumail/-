@echo off
cd /d "%~dp0"
chcp 65001 >nul
echo Running evaluation set...
echo.
.venv\Scripts\python.exe backend\evals.py
echo.
pause
