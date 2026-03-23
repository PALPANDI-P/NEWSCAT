#!/usr/bin/env python
"""
NEWSCAT - Instant Startup Script
Optimized for sub-second startup

Changes:
- Ensures browser opens only once (no duplicate opens)
- Pre-loads all models at startup before opening browser
- Adds proper error handling for model loading
"""

import os
import sys
import threading
import time
import logging
import multiprocessing
from pathlib import Path

# Windows multiprocessing MUST have this before any other code runs.
# When ProcessPoolExecutor spawns workers on Windows, they re-run this file
# as a new process. freeze_support() ensures those worker re-spawns exit
# cleanly without running the server startup code again.
if __name__ != "__main__":
    # This file is being imported by a worker subprocess — do nothing.
    pass
else:
    multiprocessing.freeze_support()

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))
os.chdir(str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("NEWSCAT v5.0 - Starting...")


def free_port(target_port: int):
    """Kill any STALE process holding target_port so we can bind cleanly.
    Skips the current process PID to avoid self-termination."""
    import subprocess
    import os as _os
    current_pid = str(_os.getpid())
    killed_any = False
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f":{target_port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                # Skip self, invalid, and system PIDs
                if pid.isdigit() and pid != "0" and pid != current_pid:
                    try:
                        subprocess.run(["taskkill", "/F", "/PID", pid],
                                       capture_output=True, timeout=3)
                        print(f"  [CLEANUP] Freed port {target_port} (killed PID {pid})")
                        killed_any = True
                    except Exception:
                        pass
    except Exception:
        pass  # Port cleanup is best-effort
    if killed_any:
        time.sleep(1)  # Let OS release the port before uvicorn binds


# Fast import - minimal overhead
try:
    from backend.main import app
    import uvicorn
except Exception as e:
    print(f"[ERROR] Failed to import FastAPI app: {e}")
    sys.exit(1)

host = os.getenv("HOST", "127.0.0.1")
port = int(os.getenv("PORT", 5000))

# Free port before starting — prevents 'address already in use' errors
free_port(port)


# =============================================================================
# MODEL PRELOADING
# =============================================================================

def preload_models():
    """
    Pre-load all models at startup to ensure they're ready when the website opens.
    This prevents the first request from being slow due to lazy loading.
    
    Returns:
        dict: Status of each model type (loaded/error)
    """
    print("\n[MODEL PRELOAD] Initializing all models...")
    results = {
        "text": {"loaded": False, "error": None},
        "image": {"loaded": False, "error": None},
        "audio": {"loaded": False, "error": None},
        "video": {"loaded": False, "error": None},
    }
    
    # 1. Preload text classifier
    try:
        from backend.models.simple_classifier import SimpleNewsClassifier
        classifier = SimpleNewsClassifier(name="StartupClassifier")
        
        model_path = project_root / "backend" / "data" / "models" / "pretrained" / "simple_model.joblib"
        if model_path.exists():
            import joblib
            classifier.model = joblib.load(str(model_path))
            print(f"  [OK] Text model loaded from {model_path.name}")
        else:
            print(f"  [--] Text model not found at {model_path}, using fallback")
        
        results["text"]["loaded"] = True
    except Exception as e:
        results["text"]["error"] = str(e)
        print(f"  [FAIL] Text model error: {e}")
    
    # 2. Preload image processor
    try:
        from backend.models.image_processor import get_image_processor
        img_proc = get_image_processor()
        is_available = img_proc.is_available()
        print(f"  [{'OK' if is_available else '--'}] Image processor: {'ready' if is_available else 'unavailable (optional)'}")
        results["image"]["loaded"] = True
    except Exception as e:
        results["image"]["error"] = str(e)
        print(f"  [FAIL] Image processor error: {e}")
    
    # 3. Preload audio processor
    try:
        from backend.models.audio_processor import get_audio_processor
        audio_proc = get_audio_processor()
        is_available = audio_proc.is_available()
        print(f"  [{'OK' if is_available else '--'}] Audio processor: {'ready' if is_available else 'unavailable (optional)'}")
        results["audio"]["loaded"] = True
    except Exception as e:
        results["audio"]["error"] = str(e)
        print(f"  [FAIL] Audio processor error: {e}")
    
    # 4. Preload video processor
    try:
        from backend.models.video_processor import get_video_processor
        video_proc = get_video_processor()
        is_available = video_proc.is_available()
        print(f"  [{'OK' if is_available else '--'}] Video processor: {'ready' if is_available else 'unavailable (optional)'}")
        results["video"]["loaded"] = True
    except Exception as e:
        results["video"]["error"] = str(e)
        print(f"  [FAIL] Video processor error: {e}")
    
    # Summary
    loaded_count = sum(1 for r in results.values() if r["loaded"])
    print(f"[MODEL PRELOAD] Complete: {loaded_count}/4 model types initialized\n")
    
    return results


# =============================================================================
# BROWSER OPENING (SINGLETON PATTERN)
# =============================================================================

_browser_opened = False
_browser_lock = threading.Lock()


def open_browser_once():
    """
    Open browser only once using a thread-safe flag.
    This prevents the browser from opening multiple times even if
    uvicorn reloads or the module is imported multiple times.
    """
    global _browser_opened
    
    with _browser_lock:
        if _browser_opened:
            return  # Already opened, skip
        _browser_opened = True
    
    # Small delay to ensure server is up and ready
    time.sleep(1.5)
    
    try:
        import webbrowser
        url = f"http://{host}:{port}"
        webbrowser.open(url)
        print(f"[BROWSER] Opened {url}")
    except Exception as e:
        print(f"[BROWSER] Warning: Could not open browser: {e}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Step 1: Preload all models BEFORE starting server
    model_status = preload_models()
    
    # Check for critical model failures (text is required)
    if not model_status["text"]["loaded"]:
        print("[WARNING] Text classifier failed to load. Classification may not work properly.")
    
    # Step 2: Print server startup info
    print(f"Server ready at http://{host}:{port}")
    print("Press Ctrl+C to stop the server\n")
    
    # Step 3: Open browser in background thread (will only open once)
    threading.Thread(target=open_browser_once, daemon=True).start()
    
    # Step 4: Run FastAPI app with Uvicorn
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload to prevent duplicate browser opens
        workers=1,
    )
