"""
Audio Worker for NEWSCAT
========================
Independent subprocess worker for audio processing and classification.
Reuses the audio_processor for audio analysis.

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

# Try to import the audio processor
PROCESSOR = None
PROCESSOR_LOADED = False


def load_processor():
    """Load the audio processor once"""
    global PROCESSOR, PROCESSOR_LOADED
    
    if PROCESSOR_LOADED:
        return PROCESSOR is not None
    
    try:
        from backend.models.audio_processor import NeuralAudioProcessor
        
        # Initialize processor with minimal configuration for speed
        PROCESSOR = NeuralAudioProcessor(
            config={
                'use_whisper': False,  # Skip heavy model for speed
                'enable_diarization': False,
                'enable_sentiment': False,
                'max_audio_duration': 60,  # Limit processing time
                'sample_rate': 16000
            }
        )
        
        PROCESSOR_LOADED = True
        logger.info("Audio worker: Processor loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Audio worker: Failed to load processor: {e}")
        PROCESSOR_LOADED = True
        return False


def process_audio_file(audio_path: str) -> Dict[str, Any]:
    """
    Process audio file and extract text for classification.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Processing result with extracted text and classification
    """
    start_time = time.time()
    
    try:
        # Ensure processor is loaded
        if not load_processor():
            return _create_error_result("Failed to load audio processor", start_time)
        
        if not audio_path or not os.path.exists(audio_path):
            return _create_error_result("Audio file not found", start_time)
        
        # Process audio
        result = PROCESSOR.process_audio(audio_path)
        
        processing_time = time.time() - start_time
        
        if result.success:
            # Extract text and classify
            extracted_text = result.extracted_text
            
            # Use simple text classification on extracted text
            categories = _classify_extracted_text(extracted_text)
            
            return {
                'success': True,
                'model_type': 'audio',
                'categories': categories,
                'primary_category': categories[0]['category'] if categories else 'unknown',
                'confidence': categories[0]['confidence'] if categories else 0.0,
                'processing_time': processing_time,
                'metadata': {
                    'audio_duration': result.duration,
                    'language': result.language,
                    'word_count': result.word_count,
                    'extracted_text_length': len(extracted_text)
                },
                'error': None,
                'extracted_text': extracted_text
            }
        else:
            return {
                'success': False,
                'model_type': 'audio',
                'categories': [],
                'primary_category': 'unknown',
                'confidence': 0.0,
                'processing_time': processing_time,
                'metadata': {},
                'error': result.error_message or "Audio processing failed"
            }
            
    except Exception as e:
        logger.error(f"Audio worker: Processing error: {e}")
        logger.error(traceback.format_exc())
        return _create_error_result(str(e), start_time)


def process_audio_data(audio_data: bytes, format: str = 'wav') -> Dict[str, Any]:
    """
    Process audio data directly.
    
    Args:
        audio_data: Audio file bytes
        format: Audio format (wav, mp3, etc.)
        
    Returns:
        Processing result
    """
    start_time = time.time()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}') as f:
        f.write(audio_data)
        temp_path = f.name
    
    try:
        return process_audio_file(temp_path)
    finally:
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass


def process_audio_base64(audio_base64: str, format: str = 'wav') -> Dict[str, Any]:
    """
    Process base64 encoded audio.
    
    Args:
        audio_base64: Base64 encoded audio data
        format: Audio format
        
    Returns:
        Processing result
    """
    try:
        audio_bytes = base64.b64decode(audio_base64)
        return process_audio_data(audio_bytes, format)
    except Exception as e:
        return _create_error_result(f"Failed to decode base64 audio: {e}", time.time())


def _classify_extracted_text(text: str) -> List[Dict[str, Any]]:
    """
    Classify extracted audio text using rule-based approach.
    
    Args:
        text: Extracted text from audio
        
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
            'google', 'apple', 'microsoft', 'facebook', 'amazon', 'tesla', 'startup'
        ],
        'politics': [
            'government', 'president', 'congress', 'election', 'vote', 'political', 'senate',
            'parliament', 'minister', 'law', 'policy', 'party', 'democrat', 'republican'
        ],
        'business': [
            'stock', 'market', 'economy', 'business', 'company', 'finance', 'investment',
            'trade', 'bank', 'revenue', 'profit', 'ipo', 'wall street', 'inflation'
        ],
        'sports': [
            'game', 'match', 'team', 'player', 'sport', 'championship', 'league', 'win',
            'score', 'football', 'soccer', 'baseball', 'basketball', 'tennis', 'olympics'
        ],
        'entertainment': [
            'movie', 'film', 'actor', 'actress', 'music', 'celebrity', 'tv', 'show', 'Hollywood',
            'Netflix', 'Oscar', 'premiere', 'concert', 'album'
        ],
        'health': [
            'health', 'medical', 'hospital', 'doctor', 'disease', 'treatment', 'vaccine',
            'pandemic', 'healthcare', 'medicine', 'patient', 'symptom', 'cancer'
        ],
        'science': [
            'science', 'research', 'space', 'nasa', 'scientist', 'discovery', 'study',
            'planet', 'moon', 'mars', 'satellite', 'climate', 'experiment'
        ],
        'world': [
            'international', 'global', 'world', 'foreign', 'country', 'war', 'conflict',
            'crisis', 'europe', 'asia', 'africa', 'summit', 'treaty'
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
        'model_type': 'audio',
        'categories': [],
        'primary_category': 'unknown',
        'confidence': 0.0,
        'processing_time': time.time() - start_time,
        'metadata': {},
        'error': error_message
    }


def process_audio(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for audio processing worker.
    
    Supports multiple input formats:
    - 'audio_path': Path to audio file
    - 'audio_data': Raw audio bytes
    - 'audio_base64': Base64 encoded audio
    
    Args:
        data: Input data dictionary
        
    Returns:
        Processing result
    """
    # Check for text fallback first (fast path - no need to load processor)
    if 'text' in data:
        return _classify_text_as_audio(data['text'])
    
    # Check for different input formats
    if 'audio_path' in data:
        return process_audio_file(data['audio_path'])
    
    if 'audio_data' in data:
        return process_audio_data(
            data['audio_data'],
            data.get('format', 'wav')
        )
    
    if 'audio_base64' in data:
        return process_audio_base64(
            data['audio_base64'],
            data.get('format', 'wav')
        )
    
    return _create_error_result("No valid audio input provided", time.time())


def _classify_text_as_audio(text: str) -> Dict[str, Any]:
    """Fallback: classify text as if it was extracted from audio"""
    start_time = time.time()
    categories = _classify_extracted_text(text)
    
    return {
        'success': True,
        'model_type': 'audio',
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
    test_text = "The president announced new economic policies today"
    result = process_audio({'text': test_text})
    print(json.dumps(result, indent=2))
