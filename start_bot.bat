@echo off
cd /d "%~dp0"
echo ========================================
echo   Starting the bot - please wait...
echo ========================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"
echo.
echo ========================================
echo   The server stopped or an error occurred.
echo   See the messages above.
echo ========================================
pause
