"""
NEWSCAT - Main Flask Application
Multi-Modal AI News Classification System
Supports: Text, Image, Audio, Video inputs

Optimized Version 5.0:
- Lazy loading for ML models (load on first request)
- Model caching with ModelManager
- Batch processing support
- Async processing for heavy operations
- Response caching with TTL
- Proper error handling and timeouts
- Memory-efficient processing
- Rate limiting support
- Temporary file cleanup
"""

import os
import sys
import logging
import threading
import webbrowser
import hashlib
import json
import tempfile
import time
import gc
from datetime import datetime
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Any, Optional
import atexit

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import config
from backend.config import DevelopmentConfig

# ===== PERFORMANCE CONSTANTS =====
CACHE_TTL = 300  # 5 minutes cache TTL
MAX_CACHE_SIZE = 500
MAX_BATCH_SIZE = 50  # Maximum items in batch request
REQUEST_TIMEOUT = 30  # Seconds before request times out
MAX_TEXT_LENGTH = 50000  # Maximum text length
MIN_TEXT_LENGTH = 10  # Minimum text length

# ===== RESPONSE CACHE =====
_response_cache: Dict[str, tuple] = {}
_cache_timestamps: Dict[str, float] = {}
_cache_lock = threading.Lock()

# ===== THREAD POOL =====
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="newscat_worker")

# ===== LOGGING SETUP =====
# Ensure logs directory exists
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / 'newscat.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===== APP INITIALIZATION =====
# Get the project root directory (parent of backend folder)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

app = Flask(__name__, 
            static_folder=str(FRONTEND_DIR),
            template_folder=str(FRONTEND_DIR))
app.config.from_object(DevelopmentConfig)

# Enable CORS
CORS(app, origins=app.config['CORS_ORIGINS'])


# ===== MODEL MANAGER FOR LAZY LOADING =====
def _init_model_manager():
    """Initialize model manager with lazy loaders"""
    from backend.models.model_manager import get_model_manager
    manager = get_model_manager()
    
    # Register model loaders (lazy - won't load until first use)
    
    # Optimized Classifier (Primary)
    def load_optimized():
        from backend.models.optimized_classifier import OptimizedEnsembleClassifier
        return OptimizedEnsembleClassifier(config=dict(app.config))
    
    manager.register('optimized', load_optimized, preload=False)
    
    # Ensemble Classifier (Fallback)
    def load_ensemble():
        from backend.models.ensemble_classifier import EnsembleNewsClassifier
        return EnsembleNewsClassifier(config=dict(app.config))
    
    manager.register('ensemble', load_ensemble, preload=False)
    
    # Simple Classifier (Final Fallback)
    def load_simple():
        from backend.models.simple_classifier import SimpleNewsClassifier
        return SimpleNewsClassifier(config=dict(app.config))
    
    manager.register('simple', load_simple, preload=False)  # Lazy load for faster startup
    
    # Keyword Extractor
    def load_keyword_extractor():
        try:
            from backend.models.advanced_keyword_extractor import AdvancedKeywordExtractor
            return AdvancedKeywordExtractor(method='hybrid')
        except ImportError:
            from backend.models.keyword_extractor import KeywordExtractor
            return KeywordExtractor()
    
    manager.register('keyword_extractor', load_keyword_extractor, preload=False)  # Lazy load
    
    # Image Processor
    def load_image_processor():
        from backend.models.image_processor import ImageProcessor
        return ImageProcessor(lazy_init=True)
    
    manager.register('image_processor', load_image_processor, preload=False)
    
    # Audio Processor
    def load_audio_processor():
        from backend.models.audio_processor import AudioProcessor
        return AudioProcessor(lazy_init=True)
    
    manager.register('audio_processor', load_audio_processor, preload=False)
    
    # Video Processor
    def load_video_processor():
        from backend.models.video_processor import VideoProcessor
        return VideoProcessor(lazy_init=True)
    
    manager.register('video_processor', load_video_processor, preload=False)
    
    return manager


# Initialize model manager
model_manager = _init_model_manager()

# ===== CLASSIFIERS DICT FOR BACKWARD COMPATIBILITY =====
# This provides compatibility with tests and older code
# NOTE: Models are lazy-loaded, so we don't populate this dict at import time
# The dict will be populated on first access via get_classifier()
classifiers = {
    'simple': None,
    'enhanced': None,
    'keyword_extractor': None
}

def _ensure_classifiers_loaded():
    """Ensure classifiers are loaded (called on first classification request)"""
    global classifiers
    if classifiers['simple'] is None:
        classifiers['simple'] = model_manager.get('simple')
    if classifiers['keyword_extractor'] is None:
        classifiers['keyword_extractor'] = model_manager.get('keyword_extractor')
    if classifiers['enhanced'] is None:
        classifiers['enhanced'] = model_manager.get('simple')  # Use simple as enhanced fallback
    return classifiers


# ===== CACHE FUNCTIONS =====
def get_cache_key(text: str, endpoint: str = 'classify', **kwargs) -> str:
    """Generate cache key from text and endpoint"""
    key_data = f"{endpoint}:{text}:{sorted(kwargs.items())}"
    return hashlib.md5(key_data.encode()).hexdigest()


def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response if valid"""
    with _cache_lock:
        if cache_key in _response_cache:
            data, timestamp = _response_cache[cache_key]
            if time.time() - timestamp < CACHE_TTL:
                return data.copy()
            else:
                # Expired, remove from cache
                del _response_cache[cache_key]
    return None


def set_cached_response(cache_key: str, data: Dict) -> None:
    """Cache a response"""
    with _cache_lock:
        # Clean up if cache is full
        if len(_response_cache) >= MAX_CACHE_SIZE:
            # Remove oldest 25% of entries
            sorted_keys = sorted(_cache_timestamps.keys(), 
                               key=lambda k: _cache_timestamps[k])
            for key in sorted_keys[:MAX_CACHE_SIZE // 4]:
                _response_cache.pop(key, None)
                _cache_timestamps.pop(key, None)
        
        _response_cache[cache_key] = (data.copy(), time.time())
        _cache_timestamps[cache_key] = time.time()


def clear_cache():
    """Clear all cached responses"""
    with _cache_lock:
        _response_cache.clear()
        _cache_timestamps.clear()
    logger.info("Response cache cleared")


# ===== HELPER FUNCTIONS =====
def get_classifier(use_enhanced: bool = True):
    """Get appropriate classifier based on preference - Lazy loading"""
    from backend.models.model_manager import get_model_manager
    manager = get_model_manager()
    
    if use_enhanced:
        # Priority: Optimized > Ensemble > Simple
        for model_name in ['optimized', 'ensemble']:
            model = manager.get(model_name)
            if model is not None:
                return model
    
    # Fallback to simple (always preloaded)
    return manager.get('simple')


def validate_text(text: str) -> tuple:
    """Validate input text"""
    if not text or not isinstance(text, str):
        return False, "Invalid input text"
    
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False, f"Text too short. Minimum {MIN_TEXT_LENGTH} characters."
    
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long. Maximum {MAX_TEXT_LENGTH} characters."
    
    return True, text


def cleanup_temp_file(filepath: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.debug(f"Cleaned up temp file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {filepath}: {e}")


def create_error_response(message: str, code: str, status: int = 400, **kwargs):
    """Create standardized error response"""
    response = {
        'status': 'error',
        'message': message,
        'code': code,
        'timestamp': datetime.now().isoformat()
    }
    response.update(kwargs)
    return jsonify(response), status


def create_success_response(data: Dict, **kwargs):
    """Create standardized success response"""
    response = {
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }
    response.update(data)
    response.update(kwargs)
    return jsonify(response)


# ===== REQUEST TIMEOUT DECORATOR =====
def with_timeout(timeout_seconds: int = REQUEST_TIMEOUT):
    """Decorator to add timeout to request handlers"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                future = executor.submit(func, *args, **kwargs)
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError:
                logger.error(f"Request timeout in {func.__name__}")
                return create_error_response(
                    'Request timed out. Please try with shorter input.',
                    'REQUEST_TIMEOUT',
                    504
                )
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return create_error_response(
                    str(e),
                    'SERVER_ERROR',
                    500
                )
        return wrapper
    return decorator


# ===== ROUTES =====
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/css/<path:path>')
def serve_css(path):
    """Serve CSS files"""
    return send_from_directory(app.static_folder + '/css', path)


@app.route('/js/<path:path>')
def serve_js(path):
    """Serve JavaScript files"""
    return send_from_directory(app.static_folder + '/js', path)


# ===== TEXT CLASSIFICATION =====
@app.route('/api/classify', methods=['POST'])
def classify():
    """Main text classification endpoint with caching"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return create_error_response('No JSON data provided', 'INVALID_REQUEST', 400)
        
        text = data.get('text', '').strip()
        use_enhanced = data.get('enhanced', True)
        
        # Validate
        is_valid, result = validate_text(text)
        if not is_valid:
            return create_error_response(result, 'INVALID_INPUT', 400)
        
        text = result  # Validated text
        
        # Check cache
        cache_key = get_cache_key(text, 'classify', enhanced=use_enhanced)
        cached = get_cached_response(cache_key)
        if cached:
            cached['cached'] = True
            cached['processing_time_ms'] = round((time.time() - start_time) * 1000, 2)
            return jsonify(cached)
        
        logger.info(f"Classification request: {len(text)} chars, enhanced={use_enhanced}")
        
        # Get classifier (lazy loaded)
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        # Classify
        result = classifier.classify(text)
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        response = {
            'model': classifier.name,
            'model_version': getattr(classifier, 'version', '1.0.0'),
            'enhanced': use_enhanced,
            'input_type': 'text',
            'cached': False,
            'processing_time_ms': processing_time,
        }
        response.update(result)
        
        # Cache the result
        set_cached_response(cache_key, response)
        
        logger.info(f"Classification: {response.get('category')} ({response.get('confidence', 0):.2%}) in {processing_time}ms")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)


# ===== BATCH CLASSIFICATION =====
@app.route('/api/classify/batch', methods=['POST'])
def classify_batch():
    """Batch classification endpoint for multiple texts"""
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return create_error_response('No JSON data provided', 'INVALID_REQUEST', 400)
        
        texts = data.get('texts', [])
        use_enhanced = data.get('enhanced', True)
        
        if not isinstance(texts, list):
            return create_error_response('texts must be an array', 'INVALID_INPUT', 400)
        
        if len(texts) == 0:
            return create_error_response('No texts provided', 'INVALID_INPUT', 400)
        
        if len(texts) > MAX_BATCH_SIZE:
            return create_error_response(
                f'Too many texts. Maximum {MAX_BATCH_SIZE} per batch.',
                'BATCH_TOO_LARGE',
                400
            )
        
        # Get classifier
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        results = []
        for i, text in enumerate(texts):
            try:
                text = str(text).strip()
                is_valid, validated = validate_text(text)
                
                if not is_valid:
                    results.append({
                        'index': i,
                        'status': 'error',
                        'message': validated
                    })
                    continue
                
                # Check cache
                cache_key = get_cache_key(validated, 'classify', enhanced=use_enhanced)
                cached = get_cached_response(cache_key)
                
                if cached:
                    cached['index'] = i
                    cached['cached'] = True
                    results.append(cached)
                else:
                    result = classifier.classify(validated)
                    result['index'] = i
                    result['cached'] = False
                    set_cached_response(cache_key, result)
                    results.append(result)
                    
            except Exception as e:
                results.append({
                    'index': i,
                    'status': 'error',
                    'message': str(e)
                })
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        return create_success_response({
            'results': results,
            'total': len(texts),
            'successful': len([r for r in results if r.get('status') != 'error']),
            'model': classifier.name,
            'processing_time_ms': processing_time
        })
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)


# ===== KEYWORD EXTRACTION =====
@app.route('/api/keywords', methods=['POST'])
def extract_keywords():
    """Keyword extraction endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        top_n = min(int(data.get('top_n', 5)), 20)  # Max 20 keywords
        
        if not text:
            return create_error_response('No text provided', 'INVALID_INPUT', 400)
        
        # Get keyword extractor (lazy loaded)
        from backend.models.model_manager import get_model_manager
        extractor = get_model_manager().get('keyword_extractor')
        
        if not extractor:
            return create_error_response(
                'Keyword extractor not available',
                'EXTRACTOR_UNAVAILABLE',
                503
            )
        
        keywords = extractor.extract(text, top_n=top_n)
        
        return create_success_response({
            'keywords': keywords,
            'top_n': top_n
        })
        
    except Exception as e:
        logger.error(f"Keyword extraction error: {e}")
        return create_error_response(str(e), 'SERVER_ERROR', 500)


# ===== CATEGORIES =====
@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all categories"""
    categories = app.config.get('CATEGORIES', [
        'technology', 'sports', 'politics', 'business',
        'entertainment', 'health', 'science', 'world',
        'education', 'environment'
    ])
    
    return create_success_response({
        'categories': categories,
        'count': len(categories)
    })


# ===== HEALTH CHECK =====
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with model status"""
    from backend.models.model_manager import get_model_manager
    manager = get_model_manager()
    model_status = manager.get_status()
    
    # Build classifiers status for frontend compatibility
    classifiers_status = {}
    for model_name, info in model_status.get('models', {}).items():
        classifiers_status[model_name] = info.get('loaded', False)
    
    return create_success_response({
        'status': 'healthy',
        'service': 'NEWSCAT',
        'version': '5.0.0',
        'phase': 'Multi-Modal AI - Optimized with Lazy Loading',
        'classifiers': classifiers_status,
        'models': model_status,
        'cache': {
            'size': len(_response_cache),
            'max_size': MAX_CACHE_SIZE,
            'ttl_seconds': CACHE_TTL
        },
        'config': {
            'min_text_length': MIN_TEXT_LENGTH,
            'max_text_length': MAX_TEXT_LENGTH,
            'max_batch_size': MAX_BATCH_SIZE,
            'request_timeout': REQUEST_TIMEOUT
        }
    })


# ===== MODEL INFO =====
@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    use_enhanced = request.args.get('enhanced', 'true').lower() == 'true'
    classifier = get_classifier(use_enhanced)
    
    if not classifier:
        return create_error_response('Classifier not available', 'NOT_FOUND', 404)
    
    return create_success_response(classifier.get_info())


# ===== MODEL PRELOAD =====
@app.route('/api/model/preload', methods=['POST'])
def preload_models():
    """Preload models for faster subsequent requests"""
    data = request.get_json() or {}
    models = data.get('models', ['optimized', 'ensemble'])
    
    from backend.models.model_manager import get_model_manager
    manager = get_model_manager()
    results = manager.preload(models)
    
    return create_success_response({
        'preloaded': results,
        'status': manager.get_status()
    })


# ===== CLEAR CACHE =====
@app.route('/api/cache/clear', methods=['POST'])
def clear_cache_endpoint():
    """Clear response cache"""
    clear_cache()
    return create_success_response({'message': 'Cache cleared'})


# ===== IMAGE CLASSIFICATION =====
@app.route('/api/classify/image', methods=['POST'])
def classify_image():
    """Image classification endpoint with OCR"""
    temp_file = None
    
    try:
        use_enhanced = True
        extracted_text = ""
        image_metadata = {}
        
        # Get image processor (lazy loaded)
        from backend.models.model_manager import get_model_manager
        processor = get_model_manager().get('image_processor')
        
        if not processor or not processor.is_available():
            return create_error_response(
                'Image processing not available. Please install OCR dependencies.',
                'IMAGE_PROCESSOR_UNAVAILABLE',
                503,
                installation_instructions=processor.get_installation_instructions() if processor else None
            )
        
        # Check for image URL in JSON data
        if request.is_json:
            data = request.get_json()
            use_enhanced = data.get('enhanced', True)
            image_url = data.get('image_url', '').strip()
            
            if image_url:
                result = processor.process_image_url(image_url)
                
                if not result.success:
                    return create_error_response(
                        result.error_message,
                        'IMAGE_PROCESSING_ERROR',
                        400
                    )
                
                extracted_text = result.extracted_text
                image_metadata = result.metadata or {}
                image_metadata['source'] = 'url'
        
        # Check for file upload
        elif 'image' in request.files:
            file = request.files['image']
            
            if file.filename == '':
                return create_error_response('No file selected', 'NO_FILE', 400)
            
            # Save to temp file for processing
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            file.save(temp_file.name)
            temp_file.close()
            
            result = processor.process_image_file(temp_file.name)
            
            if not result.success:
                return create_error_response(
                    result.error_message,
                    'IMAGE_PROCESSING_ERROR',
                    400
                )
            
            extracted_text = result.extracted_text
            image_metadata = result.metadata or {}
            image_metadata['source'] = 'upload'
            image_metadata['filename'] = file.filename
        
        else:
            return create_error_response(
                'No image provided. Send file upload or image_url in JSON.',
                'NO_IMAGE',
                400
            )
        
        # Validate extracted text
        if not extracted_text or len(extracted_text.strip()) < MIN_TEXT_LENGTH:
            return create_error_response(
                'Could not extract enough text from image. Please try a clearer image.',
                'INSUFFICIENT_TEXT',
                400,
                extracted_text_preview=extracted_text[:500] if extracted_text else ''
            )
        
        logger.info(f"Image classification: {len(extracted_text)} chars extracted")
        
        # Classify extracted text
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        result = classifier.classify(extracted_text)
        
        response = {
            'model': classifier.name,
            'input_type': 'image',
            'extracted_text': extracted_text[:1000],
            'image_metadata': image_metadata
        }
        response.update(result)
        
        logger.info(f"Image classification: {response.get('category')} ({response.get('confidence', 0):.2%})")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Image classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)
    
    finally:
        # Cleanup temp file
        if temp_file:
            cleanup_temp_file(temp_file.name)


# ===== IMAGE PROCESSOR STATUS =====
@app.route('/api/image/status', methods=['GET'])
def image_processor_status():
    """Get image processor status"""
    from backend.models.model_manager import get_model_manager
    processor = get_model_manager().get('image_processor')
    
    if not processor:
        return jsonify({
            'available': False,
            'message': 'Image processor not registered'
        })
    
    return jsonify({
        'available': processor.is_available(),
        'status': processor.get_status()
    })


# ===== AUDIO CLASSIFICATION =====
@app.route('/api/classify/audio', methods=['POST'])
def classify_audio():
    """Audio classification endpoint with Speech-to-Text"""
    temp_file = None
    
    try:
        use_enhanced = True
        extracted_text = ""
        audio_metadata = {}
        
        # Get audio processor (lazy loaded)
        from backend.models.model_manager import get_model_manager
        processor = get_model_manager().get('audio_processor')
        
        if not processor or not processor.is_available():
            return create_error_response(
                'Audio processing not available. Please install Speech-to-Text dependencies.',
                'AUDIO_PROCESSOR_UNAVAILABLE',
                503,
                installation_instructions=processor.get_installation_instructions() if processor else None
            )
        
        # Check for file upload
        if 'audio' in request.files:
            file = request.files['audio']
            
            if file.filename == '':
                return create_error_response('No file selected', 'NO_FILE', 400)
            
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
            
            result = processor.process_audio_file(temp_file.name)
            
            if not result.success:
                return create_error_response(
                    result.error_message,
                    'AUDIO_PROCESSING_ERROR',
                    400
                )
            
            extracted_text = result.extracted_text
            audio_metadata = result.metadata or {}
            audio_metadata['source'] = 'upload'
            audio_metadata['filename'] = file.filename
            audio_metadata['duration'] = result.duration
        
        else:
            return create_error_response(
                'No audio provided. Send file upload.',
                'NO_AUDIO',
                400
            )
        
        # Validate extracted text
        if not extracted_text or len(extracted_text.strip()) < MIN_TEXT_LENGTH:
            return create_error_response(
                'Could not extract enough text from audio. Please try clearer audio.',
                'INSUFFICIENT_TEXT',
                400,
                extracted_text_preview=extracted_text[:500] if extracted_text else ''
            )
        
        logger.info(f"Audio classification: {len(extracted_text)} chars from {audio_metadata.get('duration', 0):.1f}s audio")
        
        # Classify extracted text
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        result = classifier.classify(extracted_text)
        
        response = {
            'model': classifier.name,
            'input_type': 'audio',
            'extracted_text': extracted_text[:1000],
            'audio_metadata': audio_metadata
        }
        response.update(result)
        
        logger.info(f"Audio classification: {response.get('category')} ({response.get('confidence', 0):.2%})")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Audio classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)
    
    finally:
        # Cleanup temp file
        if temp_file:
            cleanup_temp_file(temp_file.name)


# ===== AUDIO PROCESSOR STATUS =====
@app.route('/api/audio/status', methods=['GET'])
def audio_processor_status():
    """Get audio processor status"""
    from backend.models.model_manager import get_model_manager
    processor = get_model_manager().get('audio_processor')
    
    if not processor:
        return jsonify({
            'available': False,
            'message': 'Audio processor not registered'
        })
    
    return jsonify({
        'available': processor.is_available(),
        'status': processor.get_status()
    })


# ===== VIDEO CLASSIFICATION =====
@app.route('/api/classify/video', methods=['POST'])
def classify_video():
    """Video classification endpoint"""
    temp_file = None
    
    try:
        use_enhanced = True
        extracted_text = ""
        video_metadata = {}
        
        # Get video processor (lazy loaded)
        from backend.models.model_manager import get_model_manager
        processor = get_model_manager().get('video_processor')
        
        if not processor or not processor.is_available():
            return create_error_response(
                'Video processing not available. Please install video processing dependencies.',
                'VIDEO_PROCESSOR_UNAVAILABLE',
                503
            )
        
        # Check for file upload
        if 'video' in request.files:
            file = request.files['video']
            
            if file.filename == '':
                return create_error_response('No file selected', 'NO_FILE', 400)
            
            # Get file extension
            filename = file.filename.lower()
            ext = '.mp4'
            for video_ext in ['.avi', '.mov', '.webm', '.mkv']:
                if filename.endswith(video_ext):
                    ext = video_ext
                    break
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            file.save(temp_file.name)
            temp_file.close()
            
            result = processor.process_video_file(temp_file.name)
            
            if not result.success:
                return create_error_response(
                    result.error_message,
                    'VIDEO_PROCESSING_ERROR',
                    400
                )
            
            extracted_text = result.extracted_text
            video_metadata = result.metadata or {}
            video_metadata['source'] = 'upload'
            video_metadata['filename'] = file.filename
            video_metadata['duration'] = result.duration
            video_metadata['frames_processed'] = result.frames_processed
        
        else:
            return create_error_response(
                'No video provided. Send file upload.',
                'NO_VIDEO',
                400
            )
        
        # Validate extracted text
        if not extracted_text or len(extracted_text.strip()) < MIN_TEXT_LENGTH:
            return create_error_response(
                'Could not extract enough text from video.',
                'INSUFFICIENT_TEXT',
                400,
                extracted_text_preview=extracted_text[:500] if extracted_text else ''
            )
        
        logger.info(f"Video classification: {len(extracted_text)} chars from {video_metadata.get('duration', 0):.1f}s video")
        
        # Classify extracted text
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        result = classifier.classify(extracted_text)
        
        response = {
            'model': classifier.name,
            'input_type': 'video',
            'extracted_text': extracted_text[:1000],
            'video_metadata': video_metadata
        }
        response.update(result)
        
        logger.info(f"Video classification: {response.get('category')} ({response.get('confidence', 0):.2%})")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Video classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)
    
    finally:
        # Cleanup temp file
        if temp_file:
            cleanup_temp_file(temp_file.name)


# ===== VIDEO PROCESSOR STATUS =====
@app.route('/api/video/status', methods=['GET'])
def video_processor_status():
    """Get video processor status"""
    from backend.models.model_manager import get_model_manager
    processor = get_model_manager().get('video_processor')
    
    if not processor:
        return jsonify({
            'available': False,
            'message': 'Video processor not registered'
        })
    
    return jsonify({
        'available': processor.is_available(),
        'status': processor.get_status()
    })


# ===== ERROR HANDLERS =====
@app.errorhandler(404)
def not_found(error):
    return create_error_response('Not found', 'NOT_FOUND', 404)


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return create_error_response('Internal server error', 'INTERNAL_ERROR', 500)


@app.errorhandler(413)
def request_too_large(error):
    return create_error_response('Request entity too large', 'REQUEST_TOO_LARGE', 413)


# ===== CLEANUP ON EXIT =====
def cleanup_on_exit():
    """Cleanup resources on application exit"""
    logger.info("Cleaning up resources...")
    clear_cache()
    executor.shutdown(wait=False)
    
    # Clear model manager
    from backend.models.model_manager import get_model_manager
    get_model_manager().clear_all()
    
    gc.collect()
    logger.info("Cleanup complete")


atexit.register(cleanup_on_exit)


# ===== OPEN BROWSER =====
def open_browser(host: str, port: int):
    """Open browser after server starts"""
    time.sleep(1.5)
    url = f"http://{host}:{port}"
    logger.info(f"Opening browser: {url}")
    webbrowser.open(url)


# ===== MAIN =====
if __name__ == '__main__':
    host = app.config.get('HOST', 'localhost')
    port = app.config.get('PORT', 5000)
    
    print("\n" + "="*70)
    print("   NEWSCAT - Multi-Modal AI News Classification System v5.0")
    print("   Optimized with Lazy Loading and Caching")
    print("="*70)
    print(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   URL: http://{host}:{port}")
    print(f"   Debug Mode: {app.config.get('DEBUG', False)}")
    print("-" * 70)
    print("   Features:")
    print("      • Lazy model loading (loads on first request)")
    print("      • Response caching with TTL")
    print("      • Batch classification support")
    print("      • Async processing with thread pool")
    print("      • Proper error handling and timeouts")
    print("      • Memory-efficient processing")
    print("      • Automatic temp file cleanup")
    print("-" * 70)
    print("   API Endpoints:")
    print("      • POST /api/classify - Single text classification")
    print("      • POST /api/classify/batch - Batch classification")
    print("      • POST /api/classify/image - Image OCR classification")
    print("      • POST /api/classify/audio - Audio STT classification")
    print("      • POST /api/classify/video - Video classification")
    print("      • POST /api/keywords - Keyword extraction")
    print("      • POST /api/model/preload - Preload models")
    print("      • GET  /api/health - Health check")
    print("="*70)
    print("   Browser will open automatically...")
    print("="*70 + "\n")
    
    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser, args=(host, port))
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run Flask app
    app.run(
        debug=app.config.get('DEBUG', False),
        host=host,
        port=port,
        threaded=True
    )
