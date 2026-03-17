"""
Image Worker for NEWSCAT
========================
Independent subprocess worker for image processing and classification.
Reuses the image_processor for image analysis.

Author: NEWSCAT Team
Version: 1.0.0
"""

import os
import sys
import time
import logging
import json
import traceback
from typing import Dict, Any, Optional, List
import tempfile
import base64

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the image processor
PROCESSOR = None
PROCESSOR_LOADED = False


def load_processor():
    """Load the image processor once"""
    global PROCESSOR, PROCESSOR_LOADED
    
    if PROCESSOR_LOADED:
        return PROCESSOR is not None
    
    try:
        from backend.models.image_processor import VisionAIProcessor
        
        # Initialize processor with fast configuration
        PROCESSOR = VisionAIProcessor(
            config={
                'use_clip': False,  # Skip heavy model for speed
                'use_ocr': True,
                'enable_object_detection': True,
                'max_image_size': 1024,
                'ocr_languages': ['en']
            }
        )
        
        PROCESSOR_LOADED = True
        logger.info("Image worker: Processor loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Image worker: Failed to load processor: {e}")
        PROCESSOR_LOADED = True
        return False


def process_image_file(image_path: str) -> Dict[str, Any]:
    """
    Process image file and extract content for classification.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Processing result with extracted content and classification
    """
    start_time = time.time()
    
    try:
        # Ensure processor is loaded
        if not load_processor():
            return _create_error_result("Failed to load image processor", start_time)
        
        if not image_path or not os.path.exists(image_path):
            return _create_error_result("Image file not found", start_time)
        
        # Process image
        result = PROCESSOR.process_image(image_path)
        
        processing_time = time.time() - start_time
        
        if result.success:
            # Extract text (from OCR) and classify
            extracted_content = result.extracted_text or ""
            detected_objects = [obj.label for obj in result.detected_objects]
            
            # Combine OCR text and detected objects for classification
            combined_content = extracted_content + " " + " ".join(detected_objects)
            
            # Classify based on extracted content
            categories = _classify_extracted_content(combined_content, detected_objects)
            
            return {
                'success': True,
                'model_type': 'image',
                'categories': categories,
                'primary_category': categories[0]['category'] if categories else 'unknown',
                'confidence': categories[0]['confidence'] if categories else 0.0,
                'processing_time': processing_time,
                'metadata': {
                    'image_width': result.width,
                    'image_height': result.height,
                    'detected_objects': detected_objects[:10],
                    'text_length': len(extracted_content)
                },
                'error': None,
                'extracted_text': extracted_content
            }
        else:
            return {
                'success': False,
                'model_type': 'image',
                'categories': [],
                'primary_category': 'unknown',
                'confidence': 0.0,
                'processing_time': processing_time,
                'metadata': {},
                'error': result.error_message or "Image processing failed"
            }
            
    except Exception as e:
        logger.error(f"Image worker: Processing error: {e}")
        logger.error(traceback.format_exc())
        return _create_error_result(str(e), start_time)


def process_image_data(image_data: bytes, format: str = 'jpg') -> Dict[str, Any]:
    """
    Process image data directly.
    
    Args:
        image_data: Image file bytes
        format: Image format (jpg, png, etc.)
        
    Returns:
        Processing result
    """
    start_time = time.time()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as f:
        f.write(image_data)
        temp_path = f.name
    
    try:
        return process_image_file(temp_path)
    finally:
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass


def process_image_base64(image_base64: str, format: str = 'jpg') -> Dict[str, Any]:
    """
    Process base64 encoded image.
    
    Args:
        image_base64: Base64 encoded image data
        format: Image format
        
    Returns:
        Processing result
    """
    try:
        image_bytes = base64.b64decode(image_base64)
        return process_image_data(image_bytes, format)
    except Exception as e:
        return _create_error_result(f"Failed to decode base64 image: {e}", time.time())


def _classify_extracted_content(
    text: str, 
    detected_objects: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Classify extracted image content using rule-based approach.
    
    Args:
        text: Extracted text from OCR
        detected_objects: List of detected object labels
        
    Returns:
        List of category predictions
    """
    if not text and not detected_objects:
        return []
    
    text_lower = text.lower()
    objects_lower = [obj.lower() for obj in (detected_objects or [])]
    
    # Object-based keywords (higher weight)
    object_keywords = {
        'technology': ['computer', 'laptop', 'phone', 'smartphone', 'tablet', 'screen', 'monitor', 'keyboard', 'mouse', 'camera', 'drone', 'robot'],
        'politics': ['flag', 'building', 'microphone', 'podium', 'person_suit', 'tie', 'flag_us', 'flag_uk'],
        'business': ['suit', 'briefcase', 'laptop', 'money', 'dollar', 'chart', 'graph', 'meeting', 'office'],
        'sports': ['ball', 'basketball', 'football', 'soccer_ball', 'tennis_racket', 'sports', 'player', 'stadium'],
        'entertainment': ['person', 'guitar', 'piano', 'microphone', 'camera', 'tv', 'stage', 'curtain'],
        'health': ['hospital', 'doctor', 'medical', 'syringe', 'pill', 'bandage', 'health', 'person_white_coat'],
        'science': ['telescope', 'microscope', 'rocket', 'space', 'satellite', 'flask', 'beaker', 'experiment'],
        'world': ['globe', 'map', 'airplane', 'ship', 'landmark', 'monument', 'building']
    }
    
    # Text-based keywords
    text_keywords = {
        'technology': [
            'tech', 'software', 'app', 'digital', 'ai', 'artificial intelligence', 'computer',
            'google', 'apple', 'microsoft', 'facebook', 'amazon', 'tesla', 'startup', 'cyber'
        ],
        'politics': [
            'government', 'president', 'congress', 'election', 'vote', 'political', 'senate',
            'parliament', 'minister', 'law', 'policy', 'party', 'democrat', 'republican', 'white house'
        ],
        'business': [
            'stock', 'market', 'economy', 'business', 'company', 'corporation', 'finance', 'investment',
            'trade', 'bank', 'revenue', 'profit', 'ipo', 'wall street', 'ceo', 'executive'
        ],
        'sports': [
            'game', 'match', 'team', 'player', 'sport', 'championship', 'league', 'win', 'score',
            'football', 'soccer', 'baseball', 'basketball', 'nba', 'nfl', 'tennis', 'olympics'
        ],
        'entertainment': [
            'movie', 'film', 'actor', 'actress', 'director', 'Hollywood', 'Bollywood', 'music',
            'celebrity', 'tv', 'show', 'Netflix', 'Oscar', 'premiere', 'concert', 'album', 'star'
        ],
        'health': [
            'health', 'medical', 'hospital', 'doctor', 'disease', 'treatment', 'vaccine', 'pandemic',
            'healthcare', 'medicine', 'patient', 'symptom', 'cancer', 'covid', 'clinical'
        ],
        'science': [
            'science', 'research', 'scientist', 'space', 'nasa', 'discovery', 'study', 'planet',
            'moon', 'mars', 'satellite', 'climate', 'experiment', 'physics', 'biology'
        ],
        'world': [
            'international', 'global', 'world', 'foreign', 'country', 'nation', 'war', 'conflict',
            'crisis', 'europe', 'asia', 'africa', 'summit', 'treaty', 'united nations'
        ]
    }
    
    category_scores = {}
    
    # Score based on detected objects (higher weight)
    for category, keywords in object_keywords.items():
        score = sum(2 for kw in keywords if any(kw in obj for obj in objects_lower))
        if score > 0:
            category_scores[category] = category_scores.get(category, 0) + score
    
    # Score based on text
    for category, keywords in text_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            category_scores[category] = category_scores.get(category, 0) + score
    
    # Sort and format
    sorted_categories = sorted(
        category_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        {'category': cat, 'confidence': min(score / 5, 1.0)}
        for cat, score in sorted_categories[:5]
        if score > 0
    ]


def _create_error_result(error_message: str, start_time: float) -> Dict[str, Any]:
    """Create an error result"""
    return {
        'success': False,
        'model_type': 'image',
        'categories': [],
        'primary_category': 'unknown',
        'confidence': 0.0,
        'processing_time': time.time() - start_time,
        'metadata': {},
        'error': error_message
    }


def process_image(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for image processing worker.
    
    Supports multiple input formats:
    - 'image_path': Path to image file
    - 'image_data': Raw image bytes
    - 'image_base64': Base64 encoded image
    
    Args:
        data: Input data dictionary
        
    Returns:
        Processing result
    """
    # Check for text fallback first (fast path - no need to load processor)
    if 'text' in data:
        return _classify_text_as_image(data['text'])
    
    # Check for different input formats
    if 'image_path' in data:
        return process_image_file(data['image_path'])
    
    if 'image_data' in data:
        return process_image_data(
            data['image_data'],
            data.get('format', 'jpg')
        )
    
    if 'image_base64' in data:
        return process_image_base64(
            data['image_base64'],
            data.get('format', 'jpg')
        )
    
    return _create_error_result("No valid image input provided", time.time())


def _classify_text_as_image(text: str) -> Dict[str, Any]:
    """Fallback: classify text as if it was extracted from image"""
    start_time = time.time()
    categories = _classify_extracted_content(text)
    
    return {
        'success': True,
        'model_type': 'image',
        'categories': categories,
        'primary_category': categories[0]['category'] if categories else 'unknown',
        'confidence': categories[0]['confidence'] if categories else 0.0,
        'processing_time': time.time() - start_time,
        'metadata': {
            'mode': 'text_fallback',
            'text_length': len(text)
        },
        'error': None
    }


# For direct testing
if __name__ == '__main__':
    test_text = "Stock market reaches new highs as tech companies report earnings"
    result = process_image({'text': test_text})
    print(json.dumps(result, indent=2))
