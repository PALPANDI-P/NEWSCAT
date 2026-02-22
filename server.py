#!/usr/bin/env python
"""
NEWSCAT - Simple Startup Script
Run this to start the server
"""

import os
import sys
import webbrowser
import threading
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Change to project directory
os.chdir(str(project_root))

print("\n" + "="*70)
print("   NEWSCAT - Multi-Modal AI News Classification System v5.0")
print("="*70)
print(f"   Project Root: {project_root}")
print(f"   Starting server...")
print("-"*70)

# Import and run the app
try:
    from backend.app import app
    print("[OK] Flask app imported successfully")
except Exception as e:
    print(f"[ERROR] Failed to import Flask app: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Get configuration - use 127.0.0.1 for reliable binding
host = app.config.get('HOST', '127.0.0.1')
port = app.config.get('PORT', 5000)

# Override host to ensure it binds properly
if host == 'localhost':
    host = '127.0.0.1'

print(f"   URL: http://{host}:{port}")
print("="*70 + "\n")

# Open browser after delay
def open_browser():
    time.sleep(2)
    url = f"http://{host}:{port}"
    print(f"Opening browser: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser: {e}")

browser_thread = threading.Thread(target=open_browser)
browser_thread.daemon = True
browser_thread.start()

print("[INFO] Starting Flask server...")
print(f"[INFO] Host: {host}, Port: {port}")
print("[INFO] Press Ctrl+C to stop the server")
print("")

# Run Flask app with proper settings
try:
    app.run(
        host=host,
        port=port,
        debug=False,  # Disable debug mode for cleaner output
        threaded=True,
        use_reloader=False,  # Disable reloader to avoid duplicate processes
        use_evalex=False  # Disable evalex to avoid potential issues
    )
except OSError as e:
    if "Address already in use" in str(e) or "Only one usage of each socket address" in str(e):
        print(f"\n[ERROR] Port {port} is already in use!")
        print("[INFO] Please close any other application using this port.")
        print("[INFO] Or try a different port by setting PORT environment variable.")
        input("\nPress Enter to exit...")
    else:
        raise
except KeyboardInterrupt:
    print("\n[INFO] Server stopped by user")
except Exception as e:
    print(f"\n[ERROR] Server error: {e}")
    import traceback
    traceback.print_exc()
    input("\nPress Enter to exit...")
