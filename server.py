#!/usr/bin/env python
"""
NEWSCAT - Instant Startup Script
Optimized for sub-second startup
"""

import os
import sys
import threading
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

print("NEWSCAT v5.0 - Starting...")

# Fast import - minimal overhead
try:
    from backend.app import app
except Exception as e:
    print(f"[ERROR] Failed to import Flask app: {e}")
    sys.exit(1)

host = app.config.get('HOST', '127.0.0.1')
port = app.config.get('PORT', 5000)
if host == 'localhost':
    host = '127.0.0.1'

print(f"Server ready at http://{host}:{port}")

# Open browser in background thread (no delay)
def open_browser():
    import webbrowser
    try:
        webbrowser.open(f"http://{host}:{port}")
    except:
        pass

threading.Thread(target=open_browser, daemon=True).start()

# Run Flask app
app.run(
    host=host,
    port=port,
    debug=False,
    threaded=True,
    use_reloader=False,
    use_evalex=False
)
