@echo off
echo ============================================================
echo NEWSCAT - News Classification System
echo ============================================================
echo.

cd /d %~dp0

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)

echo [OK] Python found
echo [*] Starting NEWSCAT server...
echo [*] Open http://127.0.0.1:5000 in your browser
echo.

python server.py

pause
