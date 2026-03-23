@echo off
REM ============================================================
REM NEWSCAT - All-in-One Startup Script for Windows
REM Single file that starts everything for the News Classification System
REM ============================================================
REM
REM Usage: Double-click start.bat or run from command line
REM Requirements: Python 3.10+ installed and in PATH
REM
REM ============================================================

setlocal EnableDelayedExpansion

REM Get script directory (project root)
cd /d "%~dp0"

REM ------------------------------------------------------------
REM Display Banner
REM ------------------------------------------------------------
cls
echo.
echo  ================================================================
echo.
echo     N E W S C A T
echo.
echo     News Classification System
echo     Multi-Modal AI News Classification System v8.0.0 (FastAPI Optimized)
echo.
echo  ================================================================
echo.

REM ------------------------------------------------------------
REM Step 1: Check Python Installation
REM ------------------------------------------------------------
echo  [1/5] Checking Python installation...
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo        [ERROR] Python not found in PATH!
    echo.
    echo        Please install Python 3.10 or higher:
    echo          1. Download from https://www.python.org/downloads/
    echo          2. During installation, check "Add Python to PATH"
    echo          3. Restart this script after installation
    echo.
    goto :error_exit
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo        [OK] Python %PYTHON_VERSION% found
echo.

REM ------------------------------------------------------------
REM Step 2: Check Virtual Environment (optional)
REM ------------------------------------------------------------
echo  [2/5] Checking virtual environment...
echo.
if exist ".venv\Scripts\activate.bat" (
    echo        [OK] Found .venv - activating...
    call .venv\Scripts\activate.bat >nul 2>&1
) else (
    echo        [INFO] No virtual environment found - using system Python
    echo        [TIP] Run 'python -m venv .venv' to create one
)
echo.

REM ------------------------------------------------------------
REM Step 3: Check Required Files
REM ------------------------------------------------------------
echo  [3/5] Checking required files...
echo.
set FILES_OK=1

if not exist "server.py" (
    echo        [ERROR] server.py not found!
    set FILES_OK=0
) else (
    echo        [OK] server.py found
)

if not exist "backend\main.py" (
    echo        [ERROR] backend\main.py not found!
    set FILES_OK=0
) else (
    echo        [OK] backend\main.py found
)

if not exist "frontend\index.html" (
    echo        [WARNING] frontend\index.html not found - web UI may not work
) else (
    echo        [OK] frontend\index.html found
)

if !FILES_OK!==0 (
    echo.
    echo        [ERROR] Required files are missing!
    echo        Please ensure you are running from the project root.
    goto :error_exit
)
echo.

REM ------------------------------------------------------------
REM Step 4: Check Port Availability
REM ------------------------------------------------------------
echo  [4/5] Checking port 5000 availability...
echo.
netstat /an | findstr ":5000.*LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo        [ERROR] Port 5000 is already in use!
    echo.
    echo        Another application is using port 5000.
    echo        Please close it and try again.
    echo.
    echo        To find the process using port 5000:
    echo          netstat -ano ^| findstr :5000
    echo.
    goto :error_exit
)
echo        [OK] Port 5000 is available
echo.

REM ------------------------------------------------------------
REM Step 5: Start Server
REM ------------------------------------------------------------
echo  [5/5] Starting NEWSCAT server...
echo.
echo  ================================================================
echo.
echo    Server starting...
echo.
echo    URL:      http://127.0.0.1:5000
echo    Web UI:   http://127.0.0.1:5000
echo    Health:   http://127.0.0.1:5000/api/health
echo.
echo    Browser will open automatically in a few seconds...
echo    Press Ctrl+C to stop the server
echo.
echo  ================================================================
echo.

REM Run the server (server.py handles browser opening)
python server.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo        [ERROR] Server exited with an error!
    goto :error_exit
)

echo.
echo        [INFO] Server stopped normally
goto :end

REM ------------------------------------------------------------
REM Error Handler
REM ------------------------------------------------------------
:error_exit
echo.
echo  ================================================================
echo.
echo    [ERROR] NEWSCAT failed to start
echo.
echo  ================================================================
echo.
echo  Troubleshooting tips:
echo.
echo    1. Ensure Python 3.10+ is installed and in PATH
echo.
echo    2. Install dependencies:
echo       pip install -r backend/requirements.txt
echo.
echo    3. Check if port 5000 is already in use:
echo       netstat -ano ^| findstr :5000
echo.
echo    4. See logs/newscat.log for detailed error messages
echo.
echo    5. Try running manually:
echo       python server.py
echo.

REM ------------------------------------------------------------
REM End
REM ------------------------------------------------------------
:end
echo.
echo  Press any key to close this window...
pause >nul
