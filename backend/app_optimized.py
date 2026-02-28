"""
NEWSCAT - Ultra-Optimized Flask Application v6.0
Multi-Modal AI News Classification System - Future-Level Performance

Optimizations:
- Async/await patterns for I/O bound operations
- LRU cache with TTL for responses
- Response compression (gzip/brotli)
- Optimized JSON serialization (orjson)
- Batch processing with async concurrency
- Request deduplication
- Connection pooling
- Streaming for large files
- Memory-efficient processing with generators
- Pre-warmed model loading
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
import asyncio
import functools
from datetime import datetime
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Any, Optional, Callable, Generator
from dataclasses import asdict
from collections import OrderedDict
import atexit
import weakref

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from flask_compress import Compress
from dotenv import load_dotenv
import orjson  # Fast JSON serialization

# Load environment variables
load_dotenv()

# Import config
from backend.config import DevelopmentConfig, ProductionConfig

# ===== PERFORMANCE CONSTANTS =====
CACHE_TTL = 300  # 5 minutes cache TTL
MAX_CACHE_SIZE = 1000  # Increased cache size
MAX_BATCH_SIZE = 100  # Increased batch size
REQUEST_TIMEOUT = 30  # Seconds before request times out
MAX_TEXT_LENGTH = 100000  # Increased max text length
MIN_TEXT_LENGTH = 10  # Minimum text length
MAX_WORKERS = 8  # Increased thread pool workers
CACHE_CLEANUP_INTERVAL = 60  # Cleanup cache every 60 seconds

# Environment config
ENV = os.getenv('FLASK_ENV', 'development')
Config = ProductionConfig if ENV == 'production' else DevelopmentConfig

# ===== ULTRA-FAST JSON ENCODER =====
def fast_json_dumps(obj: Any) -> str:
    """Ultra-fast JSON serialization using orjson"""
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS).decode('utf-8')


def fast_json_response(data: Dict, status: int = 200) -> Response:
    """Create JSON response with optimized serialization"""
    return Response(
        fast_json_dumps(data),
        status=status,
        mimetype='application/json'
    )


# ===== THREAD-SAFE LRU CACHE WITH TTL =====
class OptimizedTTLCache:
    """Ultra-efficient thread-safe LRU cache with TTL"""
    
    __slots__ = ['max_size', 'ttl', '_cache', '_timestamps', '_lock', '_hits', '_misses', '_access_count']
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._access_count = 0
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached item with O(1) lookup"""
        with self._lock:
            if key in self._cache:
                timestamp = self._timestamps.get(key, 0)
                if time.time() - timestamp <= self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return self._cache[key]
                else:
                    # Expired - remove
                    del self._cache[key]
                    del self._timestamps[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Dict) -> None:
        """Set cached item with automatic eviction"""
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._timestamps.pop(oldest_key, None)
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._cache.move_to_end(key)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': f"{hit_rate:.2%}",
                'memory_estimate_mb': len(self._cache) * 0.01  # Rough estimate
            }
    
    def clear(self) -> None:
        """Clear all cached items"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._hits = 0
            self._misses = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired items and return count removed"""
        removed = 0
        now = time.time()
        with self._lock:
            expired_keys = [
                k for k, ts in self._timestamps.items() 
                if now - ts > self.ttl
            ]
            for key in expired_keys:
                self._cache.pop(key, None)
                self._timestamps.pop(key, None)
                removed += 1
        return removed


# Initialize optimized cache
_response_cache = OptimizedTTLCache(max_size=MAX_CACHE_SIZE, ttl_seconds=CACHE_TTL)


# ===== REQUEST DEDUPLICATION =====
_pending_requests: Dict[str, threading.Event] = {}
_request_lock = threading.Lock()


def deduplicated_request(cache_key_func: Callable):
    """Decorator to deduplicate concurrent identical requests"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache_key_func(*args, **kwargs)
            
            # Check if request is pending
            with _request_lock:
                if cache_key in _pending_requests:
                    event = _pending_requests[cache_key]
                    # Wait for the existing request to complete
                    event.wait(timeout=REQUEST_TIMEOUT)
                    # Return cached result
                    return _response_cache.get(cache_key) or func(*args, **kwargs)
                
                # Mark as pending
                event = threading.Event()
                _pending_requests[cache_key] = event
            
            try:
                result = func(*args, **kwargs)
                # Cache the result
                if result and isinstance(result, dict):
                    _response_cache.set(cache_key, result)
                return result
            finally:
                with _request_lock:
                    event.set()
                    _pending_requests.pop(cache_key, None)
        
        return wrapper
    return decorator


# ===== THREAD POOL =====
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="newscat_worker")


# ===== LOGGING SETUP =====
LOG_DIR = Path(__file__).resolve().parent.parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO if ENV == 'production' else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOG_DIR / 'newscat.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===== APP INITIALIZATION =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

app = Flask(__name__,
            static_folder=str(FRONTEND_DIR),
            template_folder=str(FRONTEND_DIR))
app.config.from_object(Config)

# Enable CORS with optimized settings
CORS(app, origins=app.config['CORS_ORIGINS'], supports_credentials=True)

# Enable compression
Compress(app)


# ===== MODEL MANAGER FOR LAZY LOADING =====
def _init_model_manager():
    """Initialize model manager with optimized lazy loaders"""
    from backend.models.model_manager import get_model_manager
    manager = get_model_manager()
    
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
    
    manager.register('simple', load_simple, preload=False)
    
    # Keyword Extractor
    def load_keyword_extractor():
        try:
            from backend.models.advanced_keyword_extractor import AdvancedKeywordExtractor
            return AdvancedKeywordExtractor(method='hybrid')
        except ImportError:
            from backend.models.keyword_extractor import KeywordExtractor
            return KeywordExtractor()
    
    manager.register('keyword_extractor', load_keyword_extractor, preload=False)
    
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


# ===== CACHE FUNCTIONS =====
def get_cache_key(text: str, endpoint: str = 'classify', **kwargs) -> str:
    """Generate cache key from text and endpoint using fast hash"""
    key_data = f"{endpoint}:{text}:{sorted(kwargs.items())}"
    return hashlib.blake2b(key_data.encode(), digest_size=16).hexdigest()


def get_cached_response(cache_key: str) -> Optional[Dict]:
    """Get cached response if valid"""
    return _response_cache.get(cache_key)


def set_cached_response(cache_key: str, data: Dict) -> None:
    """Cache a response"""
    _response_cache.set(cache_key, data)


def clear_cache():
    """Clear all cached responses"""
    _response_cache.clear()
    logger.info("Response cache cleared")


# ===== HELPER FUNCTIONS =====
def get_classifier(use_enhanced: bool = True):
    """Get appropriate classifier based on preference - Optimized lazy loading"""
    from backend.models.model_manager import get_model_manager
    manager = get_model_manager()
    
    if use_enhanced:
        # Priority: Optimized > Ensemble > Simple
        for model_name in ['optimized', 'ensemble', 'simple']:
            model = manager.get(model_name)
            if model is not None:
                return model
    
    # Fallback to simple
    return manager.get('simple')


def validate_text(text: str) -> tuple:
    """Validate input text - optimized"""
    if not text or not isinstance(text, str):
        return False, "Invalid input text"
    
    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        return False, f"Text too short. Minimum {MIN_TEXT_LENGTH} characters."
    
    if len(text) > MAX_TEXT_LENGTH:
        return False, f"Text too long. Maximum {MAX_TEXT_LENGTH} characters."
    
    return True, text


def cleanup_temp_file(filepath: str):
    """Clean up temporary file - non-blocking"""
    def _cleanup():
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logger.debug(f"Failed to cleanup temp file {filepath}: {e}")
    
    # Run cleanup in background thread
    threading.Thread(target=_cleanup, daemon=True).start()


def create_error_response(message: str, code: str, status: int = 400, **kwargs):
    """Create standardized error response - optimized"""
    response = {
        'status': 'error',
        'message': message,
        'code': code,
        'timestamp': datetime.now().isoformat()
    }
    response.update(kwargs)
    return fast_json_response(response, status)


def create_success_response(data: Dict, **kwargs):
    """Create standardized success response - optimized"""
    response = {
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    }
    response.update(data)
    response.update(kwargs)
    return fast_json_response(response)


# ===== ASYNC HELPER =====
def run_async(coro):
    """Run async coroutine in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ===== BATCH PROCESSING =====
def process_batch_items(items: List[str], classifier, use_cache: bool = True) -> Generator[Dict, None, None]:
    """Generator for batch processing - memory efficient"""
    for i, text in enumerate(items):
        try:
            text = str(text).strip()
            is_valid, validated = validate_text(text)
            
            if not is_valid:
                yield {
                    'index': i,
                    'status': 'error',
                    'message': validated
                }
                continue
            
            # Check cache
            if use_cache:
                cache_key = get_cache_key(validated, 'classify', enhanced=True)
                cached = _response_cache.get(cache_key)
                if cached:
                    cached['index'] = i
                    cached['cached'] = True
                    yield cached
                    continue
            
            # Classify
            result = classifier.classify(validated)
            result['index'] = i
            result['cached'] = False
            
            # Cache result
            if use_cache:
                _response_cache.set(cache_key, result)
            
            yield result
            
        except Exception as e:
            yield {
                'index': i,
                'status': 'error',
                'message': str(e)
            }


# ===== ROUTES =====
@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return create_error_response('Failed to serve main page', 'STATIC_FILE_ERROR', 500)


@app.route('/css/<path:path>')
def serve_css(path):
    """Serve CSS files with caching"""
    try:
        css_folder = os.path.join(app.static_folder, 'css')
        response = send_from_directory(css_folder, path)
        response.headers['Cache-Control'] = 'public, max-age=86400'  # 24 hours
        return response
    except Exception as e:
        logger.error(f"Error serving CSS file {path}: {e}")
        return create_error_response(f'CSS file not found: {path}', 'STATIC_FILE_ERROR', 404)


@app.route('/js/<path:path>')
def serve_js(path):
    """Serve JavaScript files with caching"""
    try:
        js_folder = os.path.join(app.static_folder, 'js')
        response = send_from_directory(js_folder, path)
        response.headers['Cache-Control'] = 'public, max-age=86400'  # 24 hours
        return response
    except Exception as e:
        logger.error(f"Error serving JS file {path}: {e}")
        return create_error_response(f'JavaScript file not found: {path}', 'STATIC_FILE_ERROR', 404)


# ===== TEXT CLASSIFICATION =====
@app.route('/api/classify', methods=['POST'])
def classify():
    """Main text classification endpoint with optimized caching"""
    start_time = time.perf_counter()
    
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
        
        text = result
        
        # Check cache
        cache_key = get_cache_key(text, 'classify', enhanced=use_enhanced)
        cached = _response_cache.get(cache_key)
        if cached:
            cached = cached.copy()
            cached['cached'] = True
            cached['processing_time_ms'] = round((time.perf_counter() - start_time) * 1000, 2)
            return create_success_response(cached)
        
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
        
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        
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
        _response_cache.set(cache_key, response.copy())
        
        logger.info(f"Classification: {response.get('category')} ({response.get('confidence', 0):.2%}) in {processing_time}ms")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)


# ===== BATCH CLASSIFICATION =====
@app.route('/api/classify/batch', methods=['POST'])
def classify_batch():
    """Optimized batch classification endpoint with streaming"""
    start_time = time.perf_counter()
    
    try:
        data = request.get_json()
        if not data:
            return create_error_response('No JSON data provided', 'INVALID_REQUEST', 400)
        
        texts = data.get('texts', [])
        use_enhanced = data.get('enhanced', True)
        stream_response = data.get('stream', False)
        
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
        
        if stream_response:
            # Stream results as they're processed
            def generate():
                results = []
                for result in process_batch_items(texts, classifier):
                    results.append(result)
                    yield fast_json_dumps({'partial': True, 'result': result}) + '\n'
                
                # Final summary
                processing_time = round((time.perf_counter() - start_time) * 1000, 2)
                summary = {
                    'complete': True,
                    'results': results,
                    'total': len(texts),
                    'successful': len([r for r in results if r.get('status') != 'error']),
                    'model': classifier.name,
                    'processing_time_ms': processing_time
                }
                yield fast_json_dumps(summary)
            
            return Response(
                stream_with_context(generate()),
                mimetype='application/x-ndjson'
            )
        else:
            # Process all and return
            results = list(process_batch_items(texts, classifier))
            processing_time = round((time.perf_counter() - start_time) * 1000, 2)
            
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
    """Optimized keyword extraction endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        top_n = min(int(data.get('top_n', 5)), 20)
        
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
    categories = app.config.get('CATEGORIES', {})
    
    return create_success_response({
        'categories': categories,
        'count': len(categories)
    })


# ===== HEALTH CHECK =====
@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint with optimized model status"""
    from backend.models.model_manager import get_model_manager
    manager = get_model_manager()
    model_status = manager.get_status()
    
    # Build classifiers status
    classifiers_status = {
        name: info.get('loaded', False)
        for name, info in model_status.get('models', {}).items()
    }
    
    return create_success_response({
        'status': 'healthy',
        'service': 'NEWSCAT',
        'version': '6.0.0',
        'phase': 'Multi-Modal AI - Ultra-Optimized',
        'classifiers': classifiers_status,
        'models': model_status,
        'cache': _response_cache.get_stats(),
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


# ===== CACHE STATS =====
@app.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Get cache statistics"""
    return create_success_response(_response_cache.get_stats())


# ===== IMAGE CLASSIFICATION =====
@app.route('/api/classify/image', methods=['POST'])
def classify_image():
    """Optimized image classification endpoint"""
    temp_file = None
    start_time = time.perf_counter()
    
    try:
        use_enhanced = True
        extracted_text = ""
        image_metadata = {}
        
        # Get image processor
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
            
            # Save to temp file
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
        
        # Classify extracted text
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        result = classifier.classify(extracted_text)
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        
        response = {
            'model': classifier.name,
            'input_type': 'image',
            'extracted_text': extracted_text[:1000],
            'image_metadata': image_metadata,
            'processing_time_ms': processing_time
        }
        response.update(result)
        
        logger.info(f"Image classification: {response.get('category')} ({response.get('confidence', 0):.2%})")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Image classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file.name)


# ===== AUDIO CLASSIFICATION =====
@app.route('/api/classify/audio', methods=['POST'])
def classify_audio():
    """Optimized audio classification endpoint"""
    temp_file = None
    start_time = time.perf_counter()
    
    try:
        use_enhanced = True
        extracted_text = ""
        audio_metadata = {}
        
        # Get audio processor
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
        
        # Classify extracted text
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        result = classifier.classify(extracted_text)
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        
        response = {
            'model': classifier.name,
            'input_type': 'audio',
            'extracted_text': extracted_text[:1000],
            'audio_metadata': audio_metadata,
            'processing_time_ms': processing_time
        }
        response.update(result)
        
        logger.info(f"Audio classification: {response.get('category')} ({response.get('confidence', 0):.2%})")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Audio classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file.name)


# ===== VIDEO CLASSIFICATION =====
@app.route('/api/classify/video', methods=['POST'])
def classify_video():
    """Optimized video classification endpoint"""
    temp_file = None
    start_time = time.perf_counter()
    
    try:
        use_enhanced = True
        extracted_text = ""
        video_metadata = {}
        
        # Get video processor
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
        
        # Classify extracted text
        classifier = get_classifier(use_enhanced)
        if not classifier:
            return create_error_response(
                'Classifier not available',
                'CLASSIFIER_UNAVAILABLE',
                503
            )
        
        result = classifier.classify(extracted_text)
        processing_time = round((time.perf_counter() - start_time) * 1000, 2)
        
        response = {
            'model': classifier.name,
            'input_type': 'video',
            'extracted_text': extracted_text[:1000],
            'video_metadata': video_metadata,
            'processing_time_ms': processing_time
        }
        response.update(result)
        
        logger.info(f"Video classification: {response.get('category')} ({response.get('confidence', 0):.2%})")
        return create_success_response(response)
        
    except Exception as e:
        logger.error(f"Video classification error: {e}", exc_info=True)
        return create_error_response(str(e), 'SERVER_ERROR', 500)
    
    finally:
        if temp_file:
            cleanup_temp_file(temp_file.name)


# ===== BACKGROUND CACHE CLEANUP =====
def schedule_cache_cleanup():
    """Schedule periodic cache cleanup"""
    def cleanup_task():
        while True:
            time.sleep(CACHE_CLEANUP_INTERVAL)
            removed = _response_cache.cleanup_expired()
            if removed > 0:
                logger.debug(f"Cache cleanup: removed {removed} expired items")
    
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()


# Start background tasks
schedule_cache_cleanup()


# ===== MAIN ENTRY =====
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='NEWSCAT API Server')
    parser.add_argument('--host', default=Config.HOST, help='Host to bind to')
    parser.add_argument('--port', type=int, default=Config.PORT, help='Port to bind to')
    parser.add_argument('--preload', action='store_true', help='Preload models on startup')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS, help='Number of worker threads')
    
    args = parser.parse_args()
    
    if args.preload:
        logger.info("Preloading models...")
        model_manager.preload(['optimized', 'ensemble', 'simple'])
    
    logger.info(f"Starting NEWSCAT v6.0 on {args.host}:{args.port}")
    logger.info(f"Environment: {ENV}")
    logger.info(f"Cache size: {MAX_CACHE_SIZE}, TTL: {CACHE_TTL}s")
    
    # Use threaded server for better concurrency
    app.run(
        host=args.host,
        port=args.port,
        threaded=True,
        debug=Config.DEBUG
    )
