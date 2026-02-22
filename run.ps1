# NEWSCAT - Startup Script (PowerShell)
# This script starts the Flask backend server

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "[*] NEWSCAT - News Classification System" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory (project root)
$ProjectRoot = $PSScriptRoot
if (-not $ProjectRoot) {
    $ProjectRoot = $PWD.Path
}

Write-Host "[*] Project Root: $ProjectRoot" -ForegroundColor Gray

# Check if Python is available
$pythonPath = $null

# Try common Python locations
$pythonLocations = @(
    "C:/Users/PALPANDI/AppData/Local/Programs/Python/Python310/python.exe",
    "C:/Users/PALPANDI/AppData/Local/Programs/Python/Python311/python.exe",
    "C:/Python310/python.exe",
    "C:/Python311/python.exe"
)

foreach ($loc in $pythonLocations) {
    if (Test-Path $loc) {
        $pythonPath = $loc
        break
    }
}

# Try to find Python in PATH
if (-not $pythonPath) {
    $pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
}

if (-not $pythonPath) {
    Write-Host "[ERROR] Python not found" -ForegroundColor Red
    Write-Host "Please install Python 3.10+ and add it to PATH" -ForegroundColor Yellow
    exit 1
}

Write-Host "[*] Python: $pythonPath" -ForegroundColor Green

# Change to project directory
Set-Location $ProjectRoot

# Ensure logs directory exists
$logsDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
    Write-Host "[*] Created logs directory" -ForegroundColor Gray
}

Write-Host "[*] Starting backend server..." -ForegroundColor Green
Write-Host "[*] Open http://127.0.0.1:5000 in your browser" -ForegroundColor Yellow
Write-Host ""

# Set PYTHONPATH and run the Flask app
$env:PYTHONPATH = $ProjectRoot
& $pythonPath -m backend.app
