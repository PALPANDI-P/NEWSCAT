# NEWSCAT VS Code Setup Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "🚀 NEWSCAT - VS Code Setup" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

# Create virtual environment
Write-Host "`n📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install requirements
Write-Host "📥 Installing requirements..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r backend/requirements.txt

# Download NLTK data
Write-Host "📚 Downloading NLTK data..." -ForegroundColor Yellow
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`nNext Steps:" -ForegroundColor Yellow
Write-Host "1. Open VS Code: code ." -ForegroundColor White
Write-Host "2. Press F5 to run the application" -ForegroundColor White
Write-Host "3. Open browser: http://localhost:5000" -ForegroundColor White
Write-Host "`nHappy Coding! 🚀" -ForegroundColor Cyan