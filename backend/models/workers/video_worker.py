"""
Video Worker for NEWSCAT
========================
Independent subprocess worker for video processing and classification.
Reuses the video_processor for video analysis.

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

# Try to import the video processor
PROCESSOR = None
PROCESSOR_LOADED = False


def load_processor():
    """Load the video processor once"""
    global PROCESSOR, PROCESSOR_LOADED
    
    if PROCESSOR_LOADED:
        return PROCESSOR is not None
    
    try:
        from backend.models.video_processor import CinematicVideoProcessor
        
        # Initialize processor with fast configuration
        PROCESSOR = CinematicVideoProcessor(
            config={
                'use_deep_learning': False,  # Skip heavy model for speed
                'extract_keyframes': True,
                'max_keyframes': 5,
                'max_video_duration': 60,  # Limit processing time
                'scene_detection': True,
                'enable_ocr': True
            }
        )
        
        PROCESSOR_LOADED = True
        logger.info("Video worker: Processor loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Video worker: Failed to load processor: {e}")
        PROCESSOR_LOADED = True
        return False


def process_video_file(video_path: str) -> Dict[str, Any]:
    """
    Process video file and extract content for classification.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Processing result with extracted content and classification
    """
    start_time = time.time()
    
    try:
        # Ensure processor is loaded
        if not load_processor():
            return _create_error_result("Failed to load video processor", start_time)
        
        if not video_path or not os.path.exists(video_path):
            return _create_error_result("Video file not found", start_time)
        
        # Process video
        result = PROCESSOR.process_video(video_path)
        
        processing_time = time.time() - start_time
        
        if result.success:
            # Extract text from scenes and OCR
            extracted_content = result.summary or ""
            
            # Add text from keyframes
            for keyframe in result.keyframes:
                if keyframe.extracted_text:
                    extracted_content += " " + keyframe.extracted_text
            
            # Collect scene descriptions
            scene_descriptions = [scene.description for scene in result.scenes if scene.description]
            all_content = extracted_content + " " + " ".join(scene_descriptions)
            
            # Classify based on extracted content
            categories = _classify_extracted_content(all_content)
            
            return {
                'success': True,
                'model_type': 'video',
                'categories': categories,
                'primary_category': categories[0]['category'] if categories else 'unknown',
                'confidence': categories[0]['confidence'] if categories else 0.0,
                'processing_time': processing_time,
                'metadata': {
                    'video_duration': result.duration,
                    'scene_count': len(result.scenes),
                    'keyframe_count': len(result.keyframes),
                    'text_length': len(extracted_content)
                },
                'error': None,
                'extracted_text': extracted_content
            }
        else:
            return {
                'success': False,
                'model_type': 'video',
                'categories': [],
                'primary_category': 'unknown',
                'confidence': 0.0,
                'processing_time': processing_time,
                'metadata': {},
                'error': result.error_message or "Video processing failed"
            }
            
    except Exception as e:
        logger.error(f"Video worker: Processing error: {e}")
        logger.error(traceback.format_exc())
        return _create_error_result(str(e), start_time)


def process_video_data(video_data: bytes, format: str = 'mp4') -> Dict[str, Any]:
    """
    Process video data directly.
    
    Args:
        video_data: Video file bytes
        format: Video format (mp4, avi, etc.)
        
    Returns:
        Processing result
    """
    start_time = time.time()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as f:
        f.write(video_data)
        temp_path = f.name
    
    try:
        return process_video_file(temp_path)
    finally:
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass


def process_video_base64(video_base64: str, format: str = 'mp4') -> Dict[str, Any]:
    """
    Process base64 encoded video.
    
    Args:
        video_base64: Base64 encoded video data
        format: Video format
        
    Returns:
        Processing result
    """
    try:
        video_bytes = base64.b64decode(video_base64)
        return process_video_data(video_bytes, format)
    except Exception as e:
        return _create_error_result(f"Failed to decode base64 video: {e}", time.time())


def _classify_extracted_content(text: str) -> List[Dict[str, Any]]:
    """
    Classify extracted video content using rule-based approach.
    
    Args:
        text: Extracted text from video
        
    Returns:
        List of category predictions
    """
    if not text:
        return []
    
    text_lower = text.lower()
    
    # Define category keywords
    category_keywords = {
        'technology': [
            'tech', 'software', 'app', 'computer', 'digital', 'ai', 'artificial intelligence',
            'google', 'apple', 'microsoft', 'facebook', 'amazon', 'tesla', 'startup', 'robot',
            'cybersecurity', 'hacking', 'software', 'innovation', 'gadget', 'smartphone'
        ],
        'politics': [
            'government', 'president', 'congress', 'election', 'vote', 'political', 'senate',
            'parliament', 'minister', 'law', 'policy', 'party', 'democrat', 'republican',
            'white house', 'capitol', 'political', 'speech', 'press conference'
        ],
        'business': [
            'stock', 'market', 'economy', 'business', 'company', 'corporation', 'finance',
            'investment', 'trade', 'bank', 'revenue', 'profit', 'ipo', 'wall street',
            'ceo', 'executive', 'business news', 'earnings', 'quarterly', 'financial'
        ],
        'sports': [
            'game', 'match', 'team', 'player', 'sport', 'championship', 'league', 'win',
            'score', 'football', 'soccer', 'baseball', 'basketball', 'nba', 'nfl',
            'tennis', 'olympics', 'sports', 'coach', 'stadium', 'playoff', 'final'
        ],
        'entertainment': [
            'movie', 'film', 'actor', 'actress', 'director', 'Hollywood', 'Bollywood',
            'music', 'celebrity', 'tv', 'show', 'Netflix', 'Oscar', 'premiere', 'concert',
            'album', 'star', 'red carpet', 'premiere', 'trailer', 'blockbuster'
        ],
        'health': [
            'health', 'medical', 'hospital', 'doctor', 'disease', 'treatment', 'vaccine',
            'pandemic', 'healthcare', 'medicine', 'patient', 'symptom', 'cancer', 'covid',
            'clinical', 'health news', 'medical', 'wellness', 'epidemic'
        ],
        'science': [
            'science', 'research', 'scientist', 'space', 'nasa', 'discovery', 'study',
            'planet', 'moon', 'mars', 'satellite', 'climate', 'experiment', 'physics',
            'biology', 'astronomy', 'scientist', 'laboratory', 'research', 'space mission'
        ],
        'world': [
            'international', 'global', 'world', 'foreign', 'country', 'nation', 'war',
            'conflict', 'crisis', 'europe', 'asia', 'africa', 'summit', 'treaty',
            'united nations', 'diplomatic', 'breaking news', 'live', 'reporter'
        ]
    }
    
    # Score categories
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            category_scores[category] = score / len(keywords)
    
    # Sort and format
    sorted_categories = sorted(
        category_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return [
        {'category': cat, 'confidence': min(score * 3, 1.0)}
        for cat, score in sorted_categories[:5]
        if score > 0
    ]


def _create_error_result(error_message: str, start_time: float) -> Dict[str, Any]:
    """Create an error result"""
    return {
        'success': False,
        'model_type': 'video',
        'categories': [],
        'primary_category': 'unknown',
        'confidence': 0.0,
        'processing_time': time.time() - start_time,
        'metadata': {},
        'error': error_message
    }


def process_video(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for video processing worker.
    
    Supports multiple input formats:
    - 'video_path': Path to video file
    - 'video_data': Raw video bytes
    - 'video_base64': Base64 encoded video
    
    Args:
        data: Input data dictionary
        
    Returns:
        Processing result
    """
    # Check for text fallback first (fast path - no need to load processor)
    if 'text' in data:
        return _classify_text_as_video(data['text'])
    
    # Check for different input formats
    if 'video_path' in data:
        return process_video_file(data['video_path'])
    
    if 'video_data' in data:
        return process_video_data(
            data['video_data'],
            data.get('format', 'mp4')
        )
    
    if 'video_base64' in data:
        return process_video_base64(
            data['video_base64'],
            data.get('format', 'mp4')
        )
    
    return _create_error_result("No valid video input provided", time.time())


def _classify_text_as_video(text: str) -> Dict[str, Any]:
    """Fallback: classify text as if it was extracted from video"""
    start_time = time.time()
    categories = _classify_extracted_content(text)
    
    return {
        'success': True,
        'model_type': 'video',
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
    test_text = "The president delivers a speech about the economy and new policies"
    result = process_video({'text': test_text})
    print(json.dumps(result, indent=2))
