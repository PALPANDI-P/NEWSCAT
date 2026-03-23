import os
import time
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

load_dotenv()

from backend.config import DevelopmentConfig, ProductionConfig
from backend.response_formatter import (
    format_classification_result,
    format_health_check,
    format_categories_response,
    format_model_info,
)
from backend.utils import TextValidator, ContentSummarizer
from backend.utils.cache import get_cache
from backend.models.parallel_processor import get_processor

# Configuration
ENV = os.getenv("FLASK_ENV", "development") # Keeping env name for compatibility
Config = ProductionConfig if ENV == "production" else DevelopmentConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO if ENV == "production" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NEWSCAT API", version="8.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")

@app.get("/")
async def read_index():
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")

# Serve static files from frontend directory
@app.get("/css/{file}")
async def serve_css(file: str):
    return FileResponse(PROJECT_ROOT / "frontend" / "css" / file)

@app.get("/js/{file}")
async def serve_js(file: str):
    return FileResponse(PROJECT_ROOT / "frontend" / "js" / file)

@app.get("/api/health")
async def health_check():
    """Health check — dynamically verifies processor availability"""
    from backend.models.image_processor import get_image_processor
    from backend.models.audio_processor import get_audio_processor
    from backend.models.video_processor import get_video_processor
    try:
        img_ok = get_image_processor().is_available()
    except Exception:
        img_ok = False
    try:
        audio_ok = get_audio_processor().is_available()
    except Exception:
        audio_ok = False
    try:
        vid_ok = get_video_processor().is_available()
    except Exception:
        vid_ok = False
    return format_health_check(
        classifications_available=True,
        image_processing_available=img_ok,
        audio_processing_available=audio_ok,
        video_processing_available=vid_ok,
        version="8.0.0"
    )

@app.post("/api/classify")
async def classify_text(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()
    
    is_valid, result = TextValidator.is_valid(text)
    if not is_valid:
        raise HTTPException(status_code=400, detail=result)
        
    start_time = time.perf_counter()
    processor = get_processor()
    cache = get_cache()
    
    # Check cache first
    cached_result = await cache.get(text)
    if cached_result:
        logger.info("Serving cached result for text classification")
        return cached_result

    # Process using the now-async parallel processor
    classification_result = await processor.process(text=text, models=["text"])
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    # Extract keywords and summary
    # Note: Using utility functions from backend
    from backend.utils import extract_keywords
    keywords = extract_keywords(text)
    content_summary = ContentSummarizer.summarize(text, max_sentences=2, max_words=30)
    
    main_topic = classification_result.primary_category
    # Workers return confidence as 0.0-1.0; scale to 0-100 for response_formatter
    confidence_pct = round(classification_result.confidence * 100, 2)
    
    response = format_classification_result(
        category=main_topic,
        confidence=confidence_pct,
        top_predictions=[],
        keywords=keywords,
        model_name="NewsCAT Optimized (FastAPI)",
        model_version="8.0",
        input_type="text",
        processing_time_ms=float(f"{total_time:.2f}"),
        main_topic=main_topic,
        main_topics=[main_topic],
        subtopic=main_topic,
        main_topic_summary=content_summary,
        category_display=main_topic.replace("_", " ").title(),
        content_summary=content_summary,
        content_metrics={
            "character_count": len(text),
            "word_count": len(text.split()),
        },
    )
    
    # Save to cache before returning
    await cache.set(text, response)
    
    return response

@app.post("/api/classify/image")
async def classify_image(image: UploadFile = File(...)):
    start_time = time.perf_counter()
    processor = get_processor()
    
    # Read image data
    image_data = await image.read()
    
    # Pass filename so the worker can use it as keyword context
    # e.g. 'police_arrest_suspect.jpg' -> 'police arrest suspect' for classification
    result = await processor.process(
        image_data=image_data,
        models=["image"],
        image_filename=image.filename or ""
    )
    
    total_time = (time.perf_counter() - start_time) * 1000
    # Scale confidence from 0-1 to 0-100 for response_formatter
    confidence_pct = round(result.confidence * 100, 2)
    
    response = format_classification_result(
        category=result.primary_category,
        confidence=confidence_pct,
        model_name="NewsCAT Image Engine",
        model_version="8.0",
        input_type="image",
        processing_time_ms=float(f"{total_time:.2f}"),
        main_topic=result.primary_category,
        main_topics=[result.primary_category],
        subtopic=result.primary_category,
        category_display=result.primary_category.replace("_", " ").title(),
    )
    return response

@app.post("/api/classify/audio")
async def classify_audio(audio: UploadFile = File(...)):
    start_time = time.perf_counter()
    processor = get_processor()
    
    audio_data = await audio.read()
    # Pass filename so the worker can use it as keyword context
    result = await processor.process(
        audio_data=audio_data,
        models=["audio"],
        audio_filename=audio.filename or ""
    )
    
    total_time = (time.perf_counter() - start_time) * 1000
    confidence_pct = round(result.confidence * 100, 2)
    
    response = format_classification_result(
        category=result.primary_category,
        confidence=confidence_pct,
        model_name="NewsCAT Audio Engine",
        model_version="8.0",
        input_type="audio",
        processing_time_ms=float(f"{total_time:.2f}"),
        main_topic=result.primary_category,
        main_topics=[result.primary_category],
        subtopic=result.primary_category,
        category_display=result.primary_category.replace("_", " ").title()
    )
    return response

@app.post("/api/classify/video")
async def classify_video(video: UploadFile = File(...)):
    start_time = time.perf_counter()
    processor = get_processor()
    
    video_data = await video.read()
    # Pass filename so the worker can use it as keyword context
    result = await processor.process(
        video_data=video_data,
        models=["video"],
        video_filename=video.filename or ""
    )
    
    total_time = (time.perf_counter() - start_time) * 1000
    confidence_pct = round(result.confidence * 100, 2)
    
    response = format_classification_result(
        category=result.primary_category,
        confidence=confidence_pct,
        model_name="NewsCAT Video Engine",
        model_version="8.0",
        input_type="video",
        processing_time_ms=float(f"{total_time:.2f}"),
        main_topic=result.primary_category,
        main_topics=[result.primary_category],
        subtopic=result.primary_category,
        category_display=result.primary_category.replace("_", " ").title()
    )
    return response

@app.get("/api/categories")
async def get_categories():
    """Return categories from authoritative Config.CATEGORIES source"""
    from backend.config import Config
    categories = {cat: display.replace("_", " ").title() 
                  for cat, display in Config.CATEGORIES.items()}
    return format_categories_response(categories)

@app.get("/api/model/info")
async def get_model_info_api():
    """Return model info using Config.CATEGORIES as authority"""
    from backend.config import Config
    return format_model_info(
        name="NewsCAT Multi-Modal",
        version="8.0.0",
        categories=sorted(Config.CATEGORIES.keys()),
        accuracy=0.95,
        trained=True
    )

@app.get("/api/processor-status")
async def get_processor_status():
    from backend.models.image_processor import get_image_processor
    from backend.models.audio_processor import get_audio_processor
    from backend.models.video_processor import get_video_processor
    
    img_proc = get_image_processor()
    audio_proc = get_audio_processor()
    vid_proc = get_video_processor()
    
    status = {
        "text": {"available": True, "message": "Text classification always available"},
        "image": {
            "available": img_proc.is_available(),
            "installation_instructions": img_proc.get_installation_instructions() if not img_proc.is_available() else None
        },
        "audio": {
            "available": audio_proc.is_available(),
            "installation_instructions": audio_proc.get_installation_instructions() if not audio_proc.is_available() else None
        },
        "video": {
            "available": vid_proc.is_available(),
            "installation_instructions": vid_proc.get_installation_instructions() if not vid_proc.is_available() else None
        }
    }
    return {"status": "success", "processors": status}

# Mock session storage
sessions = {}

@app.post("/api/auth/login")
async def auth_login(request: Request):
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    
    # Simple hardcoded check for demo (match app.py)
    if username == "admin" and password == "newsai2024":
        import secrets
        token = secrets.token_hex(32)
        sessions[token] = {"username": username, "created_at": datetime.now().isoformat()}
        return {"status": "success", "data": {"token": token, "username": username}}
    
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/api/auth/logout")
async def auth_logout(request: Request):
    data = await request.json()
    token = data.get("token")
    if token and token in sessions:
        sessions.pop(token, None)
    return {"status": "success", "message": "Logged out"}

@app.get("/api/auth/validate")
async def auth_validate(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if token in sessions:
        return {"status": "success", "data": {"username": sessions[token]["username"], "valid": True}}
    raise HTTPException(status_code=401, detail="Invalid session")

@app.get("/api/auth/session")
async def auth_session():
    return {"status": "success", "data": {"authenticated": True}}

@app.post("/api/classify/parallel")
async def classify_parallel(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
):
    start_time = time.perf_counter()
    processor = get_processor()
    
    # Prepare inputs
    image_data = await image.read() if image else None
    audio_data = await audio.read() if audio else None
    video_data = await video.read() if video else None
    
    # Determine input type for summary formatting
    inputs = []
    if text: inputs.append("text")
    if image_data: inputs.append("image")
    if audio_data: inputs.append("audio")
    if video_data: inputs.append("video")
    
    input_type = inputs[0] if len(inputs) == 1 else "mixed"
    
    result = await processor.process(
        text=text,
        image_data=image_data,
        audio_data=audio_data,
        video_data=video_data,
    )
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    # Use standard formatter for consistency
    response = format_classification_result(
        category=result.primary_category,
        confidence=result.confidence * 100,
        model_name="NewsCAT Parallel Engine",
        model_version="8.0",
        input_type=input_type,
        processing_time_ms=float(f"{total_time:.2f}"),
        category_display=result.primary_category.replace("_", " ").title(),
        models_run=list(result.model_results.keys())
    )
    return response

@app.get("/login.html")
async def get_login_page():
    return FileResponse(PROJECT_ROOT / "frontend" / "login.html")

@app.get("/landing.html")
async def get_landing_page():
    return FileResponse(PROJECT_ROOT / "frontend" / "landing.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
