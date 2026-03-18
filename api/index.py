import os
import sys
from pathlib import Path

# Add the project root to the sys.path so we can import from 'backend'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the Flask app from backend/app.py
from backend.app import app

# Vercel needs the 'app' variable to be the Flask instance
# This is already handled by importing it above.
