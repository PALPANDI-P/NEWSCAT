import sys
from pathlib import Path

# Add the project root to the sys.path so we can import from 'backend'
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import the FastAPI app from backend/main.py (live server)
# NOTE: backend/app.py is the old Flask version — not used
from backend.main import app  # noqa: F401

# Vercel serverless: expose the FastAPI 'app' instance
