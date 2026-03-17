"""
NEWSCAT v7.0 - Expert Edition Flask Application
Ultra-fast, error-free API with comprehensive error handling
"""

import os
import sys
import time
import logging
import traceback
from pathlib import Path
from datetime import datetime
from functools import wraps, lru_cache
import hashlib
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from backend.config import DevelopmentConfig, ProductionConfig
from backend.response_formatter import (
    create_success_response, create_error_response, create_partial_response,
    format_classification_result, format_health_check, format_categories_response,
    format_model_info, ConfidenceLevel, get_confidence_level
)
from backend.utils import (
    TextValidator, PerformanceMonitor, SmartCache, DataFormatter,
    FileUtils, ErrorHandler, get_smart_cache, get_metrics_collector,
    ContentSummarizer
)

# Configuration
ENV = os.getenv('FLASK_ENV', 'development')
Config = ProductionConfig if ENV == 'production' else DevelopmentConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO if ENV == 'production' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask app initialization
PROJECT_ROOT = Path(__file__).resolve().parent.parent
app = Flask(__name__, static_folder=str(PROJECT_ROOT / 'frontend'))
app.config.from_object(Config)

# Configure CORS with explicit settings to prevent connection issues
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:*", "http://127.0.0.1:*", "null"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Accept", "X-Requested-With"],
        "supports_credentials": False
    }
})

# Global classifier instance
_classifier = None

def get_classifier():
    """Lazy load globally cached classifier on first request with error handling"""
    global _classifier
    if _classifier is None:
        try:
            # Use SimpleNewsClassifier as primary - it's more reliable for keyword-based classification
            from backend.models.simple_classifier import SimpleNewsClassifier
            _classifier = SimpleNewsClassifier()
            logger.info(f"Primary Classifier loaded: {_classifier.name} v{_classifier.version}")
        except Exception as e:
            logger.error(f"Failed to load SimpleNewsClassifier: {e}")
            # Fallback to QuantumClassifier if SimpleNewsClassifier fails
            try:
                from backend.models.lightning_classifier import QuantumClassifier
                _classifier = QuantumClassifier()
                logger.warning(f"Fallback Classifier loaded: {_classifier.name} v{_classifier.version}")
            except Exception as e2:
                logger.error(f"Failed to load any classifier: {e2}")
                raise
    return _classifier

@lru_cache(maxsize=1000)
def cached_classification(text_hash: str, text: str):
    """LRU Cache for text classification results to maximize performance"""
    classifier = get_classifier()

    # Actually classify the text
    result = classifier.classify(
        text,
        include_confidence=True,
        include_all_scores=True
    )
    # Ensure result is JSON serializable
    return json.dumps(result, default=str)  # Serialize to allow saving in cache

def validate_text(text):
    """Validate and sanitize input text"""
    is_valid, result = TextValidator.is_valid(text)
    return is_valid, result

def handle_errors(f):
    """Decorator for consistent error handling"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': 'Internal server error',
                'error': str(e) if ENV == 'development' else None
            }), 500
    return wrapper

# ===== ROUTES =====

# Demo credentials (in production, use proper database/hashing)
DEMO_USER = {
    'username': os.getenv('DEMO_USERNAME', 'admin'),
    'password_hash': os.getenv('DEMO_PASSWORD', 'newsai2024')  # Simple hash for demo
}

# Simple session storage
sessions = {}

@app.route('/')
def index():
    """Serve main page"""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to load page'}), 500

@app.route('/login.html')
def login_page():
    """Serve login page"""
    try:
        return send_from_directory(app.static_folder, 'login.html')
    except Exception as e:
        logger.error(f"Error serving login: {e}")
        return jsonify({'status': 'error', 'message': 'Login page not found'}), 404

@app.route('/landing.html')
def landing_page():
    """Serve landing page"""
    try:
        return send_from_directory(app.static_folder, 'landing.html')
    except Exception as e:
        logger.error(f"Error serving landing: {e}")
        return jsonify({'status': 'error', 'message': 'Landing page not found'}), 404

# ===== AUTH ROUTES =====

@app.route('/api/auth/login', methods=['POST'])
@handle_errors
def auth_login():
    """Authenticate user and create session"""
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'status': 'error', 'message': 'Username and password required'}), 400
    
    # Check demo credentials
    if username == DEMO_USER['username'] and password == DEMO_USER['password_hash']:
        # Create session token
        import secrets
        token = secrets.token_hex(32)
        sessions[token] = {
            'username': username,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat()
        }
        
        logger.info(f"User logged in: {username}")
        return jsonify({
            'status': 'success',
            'data': {
                'token': token,
                'username': username,
                'expires_in': 86400  # 24 hours
            }
        })
    
    logger.warning(f"Failed login attempt for username: {username}")
    return jsonify({'status': 'error', 'message': 'Invalid credentials'}), 401

@app.route('/api/auth/logout', methods=['POST'])
@handle_errors
def auth_logout():
    """Destroy session"""
    data = request.get_json()
    token = data.get('token')
    
    if token and token in sessions:
        del sessions[token]
        logger.info("User logged out")
        return jsonify({'status': 'success', 'message': 'Logged out successfully'})
    
    return jsonify({'status': 'success', 'message': 'No active session'})

@app.route('/api/auth/validate', methods=['GET'])
@handle_errors
def auth_validate():
    """Validate session token"""
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    
    if not token:
        return jsonify({'status': 'error', 'message': 'No token provided'}), 401
    
    if token in sessions:
        session = sessions[token]
        # Update last activity
        session['last_activity'] = datetime.now().isoformat()
        return jsonify({
            'status': 'success',
            'data': {
                'username': session['username'],
                'valid': True
            }
        })
    
    return jsonify({'status': 'error', 'message': 'Invalid or expired session'}), 401

@app.route('/api/auth/session', methods=['GET'])
@handle_errors
def auth_session():
    """Get current session info"""
    # This endpoint is used for client-side session management
    # In production, this would verify the token
    return jsonify({
        'status': 'success',
        'data': {
            'authenticated': True,
            'message': 'Session active'
        }
    })

@app.route('/css/<path:path>')
def serve_css(path):
    """Serve CSS files"""
    try:
        return send_from_directory(os.path.join(app.static_folder, 'css'), path)
    except Exception as e:
        logger.error(f"Error serving CSS: {e}")
        return jsonify({'status': 'error', 'message': 'CSS not found'}), 404

@app.route('/js/<path:path>')
def serve_js(path):
    """Serve JS files"""
    try:
        return send_from_directory(os.path.join(app.static_folder, 'js'), path)
    except Exception as e:
        logger.error(f"Error serving JS: {e}")
        return jsonify({'status': 'error', 'message': 'JS not found'}), 404

# ===== API ROUTES =====

@app.route('/api/health', methods=['GET'])
@handle_errors
def health_check():
    """Health check endpoint with processor status"""
    # Get processor statuses
    img_available = False
    audio_available = False
    video_available = False
    
    # Check image processor
    try:
        from backend.models.image_processor import get_image_processor
        img_proc = get_image_processor()
        img_available = img_proc.is_available()
    except Exception as e:
        logger.debug(f"Image processor status check failed: {e}")
    
    # Check audio processor
    try:
        from backend.models.audio_processor import get_audio_processor
        audio_proc = get_audio_processor()
        audio_available = audio_proc.is_available()
    except Exception as e:
        logger.debug(f"Audio processor status check failed: {e}")
    
    # Check video processor
    try:
        from backend.models.video_processor import get_video_processor
        vid_proc = get_video_processor()
        video_available = vid_proc.is_available()
    except Exception as e:
        logger.debug(f"Video processor status check failed: {e}")
    
    return jsonify(format_health_check(
        classifications_available=_classifier is not None,
        image_processing_available=img_available,
        audio_processing_available=audio_available,
        video_processing_available=video_available,
        version='7.0.0'
    ))

@app.route('/api/processor-status', methods=['GET'])
@handle_errors
def processor_status():
    """Get detailed processor status and installation instructions"""
    status = {
        'text': {
            'available': True,
            'message': 'Text classification is always available'
        }
    }
    
    # Image processor status
    try:
        from backend.models.image_processor import get_image_processor
        img_proc = get_image_processor()
        status['image'] = {
            'available': img_proc.is_available(),
            'pil_available': img_proc.is_pil_available(),
            'engine': img_proc.ocr_engine if hasattr(img_proc, 'ocr_engine') and img_proc.ocr_engine != 'none' else None,
            'installation_instructions': img_proc.get_installation_instructions() if not img_proc.is_available() else None
        }
    except Exception as e:
        status['image'] = {
            'available': False,
            'error': str(e),
            'installation_instructions': 'Install dependencies: pip install Pillow easyocr'
        }
    
    # Audio processor status
    try:
        from backend.models.audio_processor import get_audio_processor
        audio_proc = get_audio_processor()
        status['audio'] = {
            'available': audio_proc.is_available(),
            'engine': audio_proc.stt_engine if hasattr(audio_proc, 'stt_engine') and audio_proc.stt_engine != 'none' else None,
            'ffmpeg_available': audio_proc.preprocessor.is_ffmpeg_available() if hasattr(audio_proc, 'preprocessor') else None,
            'installation_instructions': audio_proc.get_installation_instructions() if hasattr(audio_proc, 'get_installation_instructions') and not audio_proc.is_available() else None
        }
    except Exception as e:
        status['audio'] = {
            'available': False,
            'error': str(e),
            'installation_instructions': 'Install dependencies: pip install openai-whisper SpeechRecognition'
        }
    
    # Video processor status
    try:
        from backend.models.video_processor import get_video_processor
        vid_proc = get_video_processor()
        status['video'] = {
            'available': vid_proc.is_available(),
            'dependencies': vid_proc.get_dependencies_status() if hasattr(vid_proc, 'get_dependencies_status') else {},
            'installation_instructions': vid_proc.get_installation_instructions() if not vid_proc.is_available() else None
        }
    except Exception as e:
        status['video'] = {
            'available': False,
            'error': str(e),
            'installation_instructions': 'Install dependencies: pip install opencv-python-headless Pillow numpy'
        }
    
    return jsonify({
        'status': 'success',
        'processors': status
    })

@app.route('/api/classify', methods=['POST'])
@handle_errors
def classify():
    """Classify text endpoint"""
    start_time = time.perf_counter()
    
    # Parse request
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400
    
    text = data.get('text', '').strip()
    
    # Validate input
    is_valid, result = validate_text(text)
    if not is_valid:
        return jsonify({'status': 'error', 'message': result}), 400
    
    try:
        # Generate text hash for caching
        text_hash = hashlib.md5(result.encode('utf-8')).hexdigest()
        
        # Get cached result or compute new one
        cached_result_str = cached_classification(text_hash, result)
        classification_result = json.loads(cached_result_str)
        
        # Calculate total processing time
        total_time = (time.perf_counter() - start_time) * 1000
        
        top_predictions = classification_result.get('top_predictions', [])
        
        # Extract keywords
        keywords = extract_keywords(result)
        
        # Generate content summary - short summary of the input content
        content_summary = ContentSummarizer.summarize(result, max_sentences=2, max_words=30)
        
        # Calculate content metrics
        words = len(result.split())
        sentences = len([s for s in result.split('.') if s.strip()])
        
        # Use new response formatter
        # Set main_topic to match category for accurate results
        main_topic = classification_result.get('category', 'unknown')
        
        response = format_classification_result(
            category=classification_result.get('category', 'unknown'),
            confidence=classification_result.get('confidence', 0),
            top_predictions=top_predictions,
            keywords=keywords,
            model_name=classification_result.get('model_name', "NewsCAT Optimized"),
            model_version=classification_result.get('model_version', "7.0"),
            input_type='text',
            processing_time_ms=round(total_time, 2),
            main_topic=main_topic,
            main_topics=[main_topic],
            subtopic=main_topic,
            main_topic_summary=classification_result.get('main_topic_summary', content_summary),
            category_display=classification_result.get('category_display', ''),
            content_summary=content_summary,  # Add content summary
            content_metrics={
                'character_count': len(result),
                'word_count': words,
                'sentence_count': sentences
            }
        )
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Classification failed'
        }), 500

@app.route('/api/classify/image', methods=['POST'])
@handle_errors
def classify_image():
    """Classify image endpoint with OCR text extraction"""
    import tempfile
    import os
    
    start_time = time.perf_counter()
    temp_file = None
    
    logger.info(f"Image classification request from {request.remote_addr}")
    logger.debug(f"Request content type: {request.content_type}")
    logger.debug(f"Request files keys: {list(request.files.keys())}")
    logger.debug(f"Request form keys: {list(request.form.keys())}")
    
    try:
        # Check for file upload
        if 'image' not in request.files:
            logger.warning(f"No image file in request.files. Available keys: {list(request.files.keys())}")
            return jsonify({'status': 'error', 'message': 'No image file provided'}), 400
        
        file = request.files['image']
        logger.info(f"Received image file: {file.filename}")
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400
        
        # Check file size (limit to 10MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        if file_size > Config.MAX_IMAGE_SIZE:
            return jsonify({
                'status': 'error',
                'message': f'Image file too large. Maximum size is {Config.MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB.'
            }), 413
        
        # Get image processor with error handling
        try:
            from backend.models.image_processor import get_image_processor
            processor = get_image_processor()
        except Exception as e:
            logger.error(f"Failed to load image processor: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Image processing module failed to load.',
                'installation_instructions': 'Install dependencies: pip install Pillow easyocr pytesseract'
            }), 503
        
        if not processor.is_available():
            return jsonify({
                'status': 'error',
                'message': 'Image processing not available. Please install OCR dependencies.',
                'pil_available': processor.is_pil_available(),
                'installation_instructions': processor.get_installation_instructions()
            }), 503
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        file.save(temp_file.name)
        temp_file.close()
        
        # Process image
        result = processor.process_image_file(temp_file.name)
        
        if not result.success:
            return jsonify({
                'status': 'error',
                'message': result.error_message
            }), 400
        
        extracted_text = result.extracted_text
        
        # Validate extracted text
        if not extracted_text or len(extracted_text.strip()) < 10:
            return jsonify({
                'status': 'error',
                'message': 'Could not extract enough text from image. Please try a clearer image with more text.',
                'extracted_text_preview': extracted_text[:200] if extracted_text else ''
            }), 400
        
        # Classify extracted text
        classifier = get_classifier()
        classification_result = classifier.classify(
            extracted_text,
            include_confidence=True,
            include_all_scores=True
        )
        
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        
        top_predictions = classification_result.get('top_predictions', [])
        
        return jsonify({
            'status': 'success',
            'data': {
                'category': classification_result['category'],
                'category_display': classification_result.get('category_display', classification_result['category'].replace('_', ' ').title()),
                'confidence': classification_result['confidence'],
                'processing_time_ms': processing_time,
                'model': classifier.name,
                'input_type': 'image',
                'extracted_text': extracted_text[:500],
                'ocr_confidence': result.confidence,
                'keywords': extract_keywords(extracted_text),
                'top_predictions': top_predictions,
                'main_topic_summary': classification_result.get('main_topic_summary', '')
            }
        })
        
    except Exception as e:
        # Handle client disconnections gracefully
        if 'ClientDisconnected' in type(e).__name__ or 'disconnect' in str(e).lower():
            logger.warning(f"Client disconnected during image upload: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Upload cancelled or connection lost. Please try again.'
            }), 499
        
        logger.error(f"Image classification error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Image processing failed: {str(e)}'
        }), 500
    
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass

@app.route('/api/classify/audio', methods=['POST'])
@handle_errors
def classify_audio():
    """Classify audio endpoint with speech-to-text extraction"""
    import tempfile
    import os
    
    start_time = time.perf_counter()
    temp_file = None
    
    logger.info(f"Audio classification request from {request.remote_addr}")
    logger.debug(f"Request content type: {request.content_type}")
    logger.debug(f"Request files keys: {list(request.files.keys())}")
    
    try:
        # Check for file upload
        if 'audio' not in request.files:
            logger.warning(f"No audio file in request.files. Available keys: {list(request.files.keys())}")
            return jsonify({'status': 'error', 'message': 'No audio file provided'}), 400
        
        file = request.files['audio']
        logger.info(f"Received audio file: {file.filename}")
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400
        
        # Check file size (limit to 50MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        if file_size > Config.MAX_AUDIO_SIZE:
            return jsonify({
                'status': 'error',
                'message': f'Audio file too large. Maximum size is {Config.MAX_AUDIO_SIZE / 1024 / 1024:.0f}MB.'
            }), 413
        
        # Get audio processor with error handling
        try:
            from backend.models.audio_processor import get_audio_processor
            processor = get_audio_processor()
        except Exception as e:
            logger.error(f"Failed to load audio processor: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Audio processing module failed to load.',
                'installation_instructions': 'Install dependencies: pip install openai-whisper SpeechRecognition'
            }), 503
        
        if not processor.is_available():
            return jsonify({
                'status': 'error',
                'message': 'Audio processing not available. Please install Speech-to-Text dependencies.',
                'ffmpeg_available': processor.preprocessor.is_ffmpeg_available() if hasattr(processor, 'preprocessor') else None,
                'installation_instructions': processor.get_installation_instructions() if hasattr(processor, 'get_installation_instructions') else 'Install: pip install openai-whisper SpeechRecognition'
            }), 503
        
        # Get file extension
        filename = file.filename.lower()
        ext = '.wav'
        for audio_ext in ['.mp3', '.m4a', '.flac', '.ogg', '.webm']:
            if filename.endswith(audio_ext):
                ext = audio_ext
                break
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        file.save(temp_file.name)
        temp_file.close()
        
        # Process audio
        result = processor.process_audio_file(temp_file.name)
        
        if not result.success:
            return jsonify({
                'status': 'error',
                'message': result.error_message
            }), 400
        
        extracted_text = result.extracted_text
        
        # Use filename as fallback if not enough text extracted
        if not extracted_text or len(extracted_text.strip()) < 10:
            # Use filename as a hint for classification
            fallback_text = os.path.splitext(file.filename)[0].replace('_', ' ').replace('-', ' ')
            if len(fallback_text.strip()) >= 5:
                extracted_text = fallback_text
                logger.info(f"Using filename as fallback text: {extracted_text}")
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not extract enough text from audio. Please try clearer audio.',
                    'extracted_text_preview': result.extracted_text[:200] if result.extracted_text else ''
                }), 400
        
        # Classify extracted text
        classifier = get_classifier()
        classification_result = classifier.classify(
            extracted_text,
            include_confidence=True,
            include_all_scores=True
        )
        
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        
        top_predictions = classification_result.get('top_predictions', [])
        
        return jsonify({
            'status': 'success',
            'data': {
                'category': classification_result['category'],
                'category_display': classification_result.get('category_display', classification_result['category'].replace('_', ' ').title()),
                'confidence': classification_result['confidence'],
                'processing_time_ms': processing_time,
                'model': classifier.name,
                'input_type': 'audio',
                'extracted_text': extracted_text[:500],
                'transcription_confidence': result.confidence,
                'duration': result.duration,
                'keywords': extract_keywords(extracted_text),
                'top_predictions': top_predictions,
                'main_topic_summary': classification_result.get('main_topic_summary', '')
            }
        })
        
    except Exception as e:
        # Handle client disconnections gracefully
        if 'ClientDisconnected' in type(e).__name__ or 'disconnect' in str(e).lower():
            logger.warning(f"Client disconnected during audio upload: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Upload cancelled or connection lost. Please try again.'
            }), 499  # 499 Client Closed Request
        
        logger.error(f"Audio classification error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Audio processing failed: {str(e)}'
        }), 500
    
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass

@app.route('/api/classify/video', methods=['POST'])
@handle_errors
def classify_video():
    """Classify video endpoint with frame OCR and audio transcription"""
    import tempfile
    import os
    
    start_time = time.perf_counter()
    temp_file = None
    
    logger.info(f"Video classification request from {request.remote_addr}")
    logger.debug(f"Request content type: {request.content_type}")
    logger.debug(f"Request files keys: {list(request.files.keys())}")
    
    try:
        # Check for file upload
        if 'video' not in request.files:
            logger.warning(f"No video file in request.files. Available keys: {list(request.files.keys())}")
            return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
        
        file = request.files['video']
        logger.info(f"Received video file: {file.filename}")
        
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'Empty filename'}), 400
        
        # Check file size (limit to 100MB)
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        if file_size > Config.MAX_VIDEO_SIZE:
            return jsonify({
                'status': 'error',
                'message': f'Video file too large. Maximum size is {Config.MAX_VIDEO_SIZE / 1024 / 1024:.0f}MB.'
            }), 413
        
        # Get video processor with error handling
        try:
            from backend.models.video_processor import get_video_processor
            processor = get_video_processor()
        except Exception as e:
            logger.error(f"Failed to load video processor: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Video processing module failed to load.',
                'installation_instructions': 'Install dependencies: pip install opencv-python-headless Pillow numpy'
            }), 503
        
        if not processor.is_available():
            return jsonify({
                'status': 'error',
                'message': 'Video processing not available. Please install OpenCV and video dependencies.',
                'dependencies': processor.get_dependencies_status() if hasattr(processor, 'get_dependencies_status') else {},
                'installation_instructions': processor.get_installation_instructions()
            }), 503
        
        # Get file extension
        filename = file.filename.lower()
        ext = '.mp4'
        for video_ext in ['.avi', '.mov', '.mkv', '.webm']:
            if filename.endswith(video_ext):
                ext = video_ext
                break
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        file.save(temp_file.name)
        temp_file.close()
        
        # Process video
        try:
            result = processor.process_video_file(temp_file.name, extract_audio=True)
        except Exception as e:
            logger.error(f"Video processing failed with exception: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Video processing failed: {str(e)}',
                'installation_instructions': 'Install dependencies: pip install opencv-python-headless Pillow numpy'
            }), 500
        
        if not result.success:
            logger.warning(f"Video processing returned failure: {result.error_message}")
            return jsonify({
                'status': 'error',
                'message': result.error_message or 'Video processing failed',
                'extracted_text_preview': result.extracted_text[:200] if result.extracted_text else None
            }), 400
        
        extracted_text = result.extracted_text
        
        # Use filename as fallback if not enough text extracted
        if not extracted_text or len(extracted_text.strip()) < 10:
            fallback_text = os.path.splitext(file.filename)[0].replace('_', ' ').replace('-', ' ')
            if len(fallback_text.strip()) >= 5:
                extracted_text = fallback_text
                logger.info(f"Using filename as fallback text for video: {extracted_text}")
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Could not extract enough text from video. Please try a video with clearer text or speech.',
                    'extracted_text_preview': result.extracted_text[:200] if result.extracted_text else ''
                }), 400
        
        # Classify extracted text
        classifier = get_classifier()
        classification_result = classifier.classify(
            extracted_text,
            include_confidence=True,
            include_all_scores=True
        )
        
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        
        top_predictions = classification_result.get('top_predictions', [])
        
        return jsonify({
            'status': 'success',
            'data': {
                'category': classification_result['category'],
                'category_display': classification_result.get('category_display', classification_result['category'].replace('_', ' ').title()),
                'confidence': classification_result['confidence'],
                'processing_time_ms': processing_time,
                'model': classifier.name,
                'input_type': 'video',
                'extracted_text': extracted_text[:500],
                'frames_processed': result.frames_processed,
                'duration': result.duration,
                'keywords': extract_keywords(extracted_text),
                'top_predictions': top_predictions,
                'main_topic_summary': classification_result.get('main_topic_summary', '')
            }
        })
        
    except Exception as e:
        # Handle client disconnections gracefully
        if 'ClientDisconnected' in type(e).__name__ or 'disconnect' in str(e).lower():
            logger.warning(f"Client disconnected during video upload: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Upload cancelled or connection lost. Please try again.'
            }), 499
        
        logger.error(f"Video classification error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f'Video processing failed: {str(e)}'
        }), 500
    
    finally:
        # Cleanup temp file
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass

@app.route('/api/categories', methods=['GET'])
@handle_errors
def get_categories():
    """Get available categories"""
    from backend.config import Config
    return jsonify(format_categories_response(Config.CATEGORIES))

@app.route('/api/model/info', methods=['GET'])
@handle_errors
def get_model_info():
    """Get model information"""
    classifier = get_classifier()
    response = format_model_info(
        name=classifier.name,
        version=classifier.version,
        categories=list(classifier.categories) if isinstance(classifier.categories, list) else list(classifier.categories.keys()),
        accuracy=getattr(classifier, 'accuracy', 0),
        trained=getattr(classifier, 'is_trained', False),
        model_type='rule-based'
    )
    return jsonify(response)

@app.route('/api/news/realtime', methods=['GET'])
@handle_errors
def get_realtime_news():
    """Fetch real-time news headlines for classification testing across diverse categories"""
    # Mocking real-world news headlines to ensure instant testing without API keys, covering 5 diverse "sides"
    news_items = [
        {"id": "1", "title": "OpenAI announces new GPT reasoning model", "content": "The new model shows significant improvements in complex logic, mathematics, and coding tasks, pushing the boundaries of artificial intelligence capabilities.", "source": "Tech Innovations"},
        {"id": "2", "title": "Federal Reserve holds interest rates steady", "content": "Central bank officials decided to keep rates unchanged as inflation data matches expectations, signaling stability in global financial markets.", "source": "Finance Weekly"},
        {"id": "3", "title": "Real Madrid wins Champions League Final", "content": "A stunning late goal secured the victory for Real Madrid in a thrilling match against Manchester City, crowning them the undisputed kings of European football.", "source": "Global Sports"},
        {"id": "4", "title": "New study shows coffee may increase lifespan", "content": "Researchers found that drinking 2-3 cups of coffee daily is associated with a lower risk of heart disease and cardiovascular abnormalities.", "source": "Health Science Daily"},
        {"id": "5", "title": "NASA's Artemis program delayed by two years", "content": "Technical challenges with the lunar lander have pushed back the targeted human return to the moon, emphasizing the complexities of deep space exploration.", "source": "Space Exploration News"},
        {"id": "6", "title": "Global leaders sign new climate pact", "content": "At the COP summit, nations agreed to triple renewable energy capacity by 2030, marking a monumental shift toward sustainable environmental policy.", "source": "Eco World"},
        {"id": "7", "title": "Cybersecurity firm uncovers massive data breach", "content": "Hackers infiltrated multiple enterprise databases using a zero-day exploit, exposing the personal records of over 50 million global users.", "source": "Cyber Defense Network"}
    ]
    
    return jsonify({
        'status': 'success',
        'data': news_items,
        'timestamp': datetime.utcnow().isoformat() + "Z"
    })

# ===== HELPER FUNCTIONS =====

def extract_keywords(text: str, max_keywords: int = 10) -> list:
    """Extract keywords from text"""
    try:
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        
        # Filter common words
        stop_words = {
            'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said',
            'will', 'would', 'could', 'should', 'also', 'more', 'their', 'there',
            'about', 'which', 'when', 'what', 'where', 'than', 'then', 'them',
            'some', 'other', 'into', 'most', 'very', 'just', 'over', 'such',
            'after', 'only', 'many', 'make', 'like', 'each', 'made', 'does',
            'your', 'being', 'well', 'back', 'much', 'even', 'take', 'come',
            'these', 'know', 'want', 'because', 'here', 'between', 'both',
            'under', 'never', 'same', 'another', 'while', 'last', 'might',
            'before', 'still', 'through', 'those'
        }
        words = [w for w in words if w not in stop_words]
        
        # Get most common — return as simple strings for frontend compatibility
        counter = Counter(words)
        return [word for word, count in counter.most_common(max_keywords)]
    except Exception as e:
        logger.warning(f"Keyword extraction failed: {e}")
        return []

# ===== ERROR HANDLERS =====

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'status': 'error',
        'message': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size: 10MB for images, 50MB for audio, 100MB for video.'
    }), 413

# ===== PARALLEL CLASSIFICATION ENDPOINT =====

@app.route('/api/classify/all', methods=['POST'])
@handle_errors
def classify_all():
    """
    Parallel classification endpoint that processes text, audio, image, and video
    simultaneously using the parallel processor.
    """
    start_time = time.perf_counter()
    
    # Parse request
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception as e:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400
    
    text = data.get('text', '').strip()
    models = data.get('models', ['text', 'audio', 'image', 'video'])
    
    # Validate models list
    valid_models = ['text', 'audio', 'image', 'video']
    if isinstance(models, str):
        models = [models]
    models = [m for m in models if m in valid_models]
    
    if not models:
        return jsonify({'status': 'error', 'message': 'No valid models specified'}), 400
    
    try:
        # Import and use parallel processor
        from backend.models.parallel_processor import ParallelProcessor
        
        processor = ParallelProcessor()
        result = processor.process(text=text, models=models)
        
        # Calculate total processing time
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Build response
        response = {
            'status': 'success',
            'data': {
                'primary_category': result.primary_category,
                'confidence': result.confidence,
                'confidence_level': get_confidence_level(result.confidence),
                'processing_time_ms': round(result.total_processing_time * 1000, 2),
                'model_name': 'NewsCAT Parallel Processor',
                'model_version': '1.0.0',
                'input_type': 'multi-modal',
                'successful_models': [m for m, r in result.model_results.items() if r.success],
                'failed_models': result.partial_failures,
                'individual_results': {
                    model: {
                        'category': res.primary_category,
                        'confidence': res.confidence,
                        'success': res.success
                    }
                    for model, res in result.model_results.items()
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Parallel classification error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': 'Parallel classification failed',
            'error': str(e) if ENV == 'development' else None
        }), 500

# ===== MAIN =====

if __name__ == '__main__':
    # Preload classifier in background to avoid slow first request
    def _preload_model():
        try:
            get_classifier()
            logger.info("Classifier pre-loaded successfully in background thread.")
        except Exception as e:
            logger.warning(f"Could not pre-load classifier in background: {e}")
            
    import threading
    threading.Thread(target=_preload_model, daemon=True).start()
    
    logger.info(f"Starting NEWSCAT API server on {Config.HOST}:{Config.PORT} ({'development' if Config.DEBUG else 'production'} mode)")
    # Run the Flask app
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=Config.THREADED
    )
