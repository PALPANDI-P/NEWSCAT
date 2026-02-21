# NEWSCAT - Startup Script (PowerShell)
# This script starts the Flask backend server

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[*] NEWSCAT - News Classification System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
$pythonPath = "C:/Users/PALPANDI/AppData/Local/Programs/Python/Python310/python.exe"
if (-not (Test-Path $pythonPath)) {
    # Try to find Python in PATH
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $pythonPath) {
        Write-Host "[ERROR] Python not found" -ForegroundColor Red
        Write-Host "Please install Python 3.10+ and add it to PATH" -ForegroundColor Yellow
        exit 1
    }
}

Write-Host "[*] Python: $pythonPath" -ForegroundColor Green
Write-Host "[*] Starting backend server..." -ForegroundColor Green
Write-Host ""

# Change to project directory
Push-Location $PSScriptRoot

# Run the Flask app using module syntax
& $pythonPath -m backend.app

Pop-Location
