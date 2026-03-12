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
from functools import wraps

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
    FileUtils, ErrorHandler, get_smart_cache, get_metrics_collector
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
    """Lazy load classifier on first request with error handling"""
    global _classifier
    if _classifier is None:
        try:
            from backend.models.lightning_classifier import LightningClassifier
            _classifier = LightningClassifier()
            logger.info(f"Classifier loaded: {_classifier.name} v{_classifier.version}")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise
    return _classifier

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

@app.route('/')
def index():
    """Serve main page"""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to load page'}), 500

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
            'ffmpeg_available': audio_proc._check_ffmpeg() if hasattr(audio_proc, '_check_ffmpeg') else None,
            'installation_instructions': audio_proc.get_installation_instructions() if not audio_proc.is_available() else None
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
        # Get classifier and classify
        classifier = get_classifier()
        classification_result = classifier.classify(
            result, 
            include_confidence=True,
            include_all_scores=True
        )
        
        # Calculate total processing time
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Format top_predictions to ensure confidence is 0-100 scale
        top_predictions = classification_result.get('top_predictions', [])
        for pred in top_predictions:
            if 'confidence' in pred and pred['confidence'] <= 1.0:
                pred['confidence'] = pred['confidence'] * 100
        
        # Extract keywords
        keywords = extract_keywords(result)
        
        # Calculate content metrics
        words = len(result.split())
        sentences = len([s for s in result.split('.') if s.strip()])
        
        # Use new response formatter
        response = format_classification_result(
            category=classification_result['category'],
            confidence=classification_result['confidence'],
            top_predictions=top_predictions,
            keywords=keywords,
            model_name=classifier.name,
            model_version=classifier.version,
            input_type='text',
            processing_time_ms=round(total_time, 2),
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
        
        # Format top_predictions to ensure confidence is 0-100 scale
        top_predictions = classification_result.get('top_predictions', [])
        for pred in top_predictions:
            if 'confidence' in pred and pred['confidence'] <= 1.0:
                pred['confidence'] = pred['confidence'] * 100
        
        return jsonify({
            'status': 'success',
            'data': {
                'category': classification_result['category'],
                'confidence': classification_result['confidence'],
                'processing_time_ms': processing_time,
                'model': classifier.name,
                'input_type': 'image',
                'extracted_text': extracted_text[:500],
                'ocr_confidence': result.confidence,
                'keywords': extract_keywords(extracted_text),
                'top_predictions': top_predictions
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
                'ffmpeg_available': processor._check_ffmpeg() if hasattr(processor, '_check_ffmpeg') else None,
                'installation_instructions': processor.get_installation_instructions()
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
        
        # Format top_predictions to ensure confidence is 0-100 scale
        top_predictions = classification_result.get('top_predictions', [])
        for pred in top_predictions:
            if 'confidence' in pred and pred['confidence'] <= 1.0:
                pred['confidence'] = pred['confidence'] * 100
        
        return jsonify({
            'status': 'success',
            'data': {
                'category': classification_result['category'],
                'confidence': classification_result['confidence'],
                'processing_time_ms': processing_time,
                'model': classifier.name,
                'input_type': 'audio',
                'extracted_text': extracted_text[:500],
                'transcription_confidence': result.confidence,
                'duration': result.duration,
                'keywords': extract_keywords(extracted_text),
                'top_predictions': top_predictions
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
        result = processor.process_video_file(temp_file.name, extract_audio=True)
        
        if not result.success:
            return jsonify({
                'status': 'error',
                'message': result.error_message
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
        
        # Format top_predictions to ensure confidence is 0-100 scale
        top_predictions = classification_result.get('top_predictions', [])
        for pred in top_predictions:
            if 'confidence' in pred and pred['confidence'] <= 1.0:
                pred['confidence'] = pred['confidence'] * 100
        
        return jsonify({
            'status': 'success',
            'data': {
                'category': classification_result['category'],
                'confidence': classification_result['confidence'],
                'processing_time_ms': processing_time,
                'model': classifier.name,
                'input_type': 'video',
                'extracted_text': extracted_text[:500],
                'frames_processed': result.frames_processed,
                'duration': result.duration,
                'keywords': extract_keywords(extracted_text),
                'top_predictions': top_predictions
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
        stop_words = {'this', 'that', 'with', 'from', 'they', 'have', 'been', 'were', 'said'}
        words = [w for w in words if w not in stop_words]
        
        # Get most common
        counter = Counter(words)
        return [{'word': word, 'count': count} for word, count in counter.most_common(max_keywords)]
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

# ===== MAIN =====

if __name__ == '__main__':
    # Pre-load classifier on startup for faster first request
    try:
        get_classifier()
        logger.info("Classifier pre-loaded successfully")
    except Exception as e:
        logger.warning(f"Could not pre-load classifier: {e}")
        logger.warning("Classifier will be loaded on first request")
    
    # Run the Flask app
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=Config.THREADED
    )
