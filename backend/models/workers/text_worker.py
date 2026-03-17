"""
Text Worker for NEWSCAT
=======================
Independent subprocess worker for text classification.
Reuses the simple_classifier for fast and reliable text classification.

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

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import the classifier
CLASSIFIER = None
CLASSIFIER_LOADED = False


def load_classifier():
    """Load the text classifier once"""
    global CLASSIFIER, CLASSIFIER_LOADED
    
    if CLASSIFIER_LOADED:
        return CLASSIFIER is not None
    
    try:
        from backend.models.simple_classifier import SimpleNewsClassifier
        
        # Initialize classifier
        CLASSIFIER = SimpleNewsClassifier(name="TextWorkerClassifier")
        
        # Try to load pre-trained model
        model_path = os.path.join(
            os.path.dirname(__file__), 
            '..', '..', 'data', 'models', 'pretrained', 'simple_model.joblib'
        )
        
        if os.path.exists(model_path):
            import joblib
            try:
                data = joblib.load(model_path)
                # Handle both dict format and direct pipeline format
                if isinstance(data, dict):
                    CLASSIFIER.pipeline = data.get('pipeline', data)
                else:
                    CLASSIFIER.pipeline = data
                CLASSIFIER.is_trained = True
                logger.info("Text worker: Loaded pre-trained model")
            except Exception as e:
                logger.warning(f"Text worker: Could not load model: {e}")
        
        CLASSIFIER_LOADED = True
        logger.info("Text worker: Classifier loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Text worker: Failed to load classifier: {e}")
        CLASSIFIER_LOADED = True
        return False


def classify_text(text: str) -> Dict[str, Any]:
    """
    Classify text content.
    
    Args:
        text: Text to classify
        
    Returns:
        Classification result dictionary
    """
    start_time = time.time()
    
    try:
        # Ensure classifier is loaded
        if not load_classifier():
            return _create_error_result("Failed to load classifier", start_time)
        
        if not text or not text.strip():
            return _create_error_result("Empty text provided", start_time)
        
        # Perform classification
        if CLASSIFIER and CLASSIFIER.is_trained:
            result = CLASSIFIER.classify(text)
            # Check if ML classification is reliable
            # Note: simple_classifier returns 'category' not 'primary_category'
            ml_category = result.get('category', 'unknown')
            ml_confidence = result.get('confidence', 0.0)
            
            # If ML returns unknown or low confidence, use rule-based
            if ml_category == 'unknown' or ml_confidence < 0.5:
                rule_result = _rule_based_classification(text)
                # Use rule-based result if it has higher confidence
                if rule_result.get('confidence', 0) > ml_confidence:
                    result = rule_result
                    result['method'] = 'rule_based'
        else:
            # Use rule-based classification as fallback
            result = _rule_based_classification(text)
            result['method'] = 'rule_based'
        
        processing_time = time.time() - start_time
        
        # Normalize the result - simple_classifier uses 'category', workers expect 'primary_category'
        categories = result.get('categories', [])
        if not categories:
            # If no categories list, create one from the primary category
            primary_cat = result.get('category', result.get('primary_category', 'unknown'))
            conf = result.get('confidence', 0.0)
            if primary_cat and primary_cat != 'unknown':
                categories = [{'category': primary_cat, 'confidence': conf}]
        
        return {
            'success': True,
            'model_type': 'text',
            'categories': categories,
            'primary_category': result.get('category', result.get('primary_category', 'unknown')),
            'confidence': result.get('confidence', 0.0),
            'processing_time': processing_time,
            'metadata': {
                'text_length': len(text),
                'model_used': result.get('method', 'simple_classifier')
            },
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Text worker: Classification error: {e}")
        logger.error(traceback.format_exc())
        return _create_error_result(str(e), start_time)


def _rule_based_classification(text: str) -> Dict[str, Any]:
    """
    Fallback rule-based classification.
    
    Args:
        text: Text to classify
        
    Returns:
        Classification result
    """
    text_lower = text.lower()
    
    # Define category keywords
    category_keywords = {
        'technology': [
            'ai', 'artificial intelligence', 'machine learning', 'tech', 'technology',
            'software', 'digital', 'app', 'application', 'cyber', 'data', 'computer',
            'robot', 'algorithm', 'startup', 'google', 'apple', 'microsoft', 'amazon',
            'facebook', 'meta', 'twitter', 'instagram', 'whatsapp', 'tesla', 'spacex',
            'neural', 'deep learning', 'automation', 'cloud', 'internet', '5g', '6g',
            'semiconductor', 'chip', 'processor', 'nvidia', 'intel', 'amd', 'gpu'
        ],
        'politics': [
            'government', 'president', 'congress', 'senate', 'parliament', 'election',
            'vote', 'political', 'party', 'law', 'policy', 'minister', 'governor',
            'legislation', 'bill', 'amendment', 'democrat', 'republican', 'liberal',
            'conservative', 'prime minister', 'chancellor', 'mp', 'representative',
            'diplomat', 'embassy', 'treaty', 'summit', 'g20', 'un', 'nato'
        ],
        'business': [
            'stock', 'market', 'economy', 'business', 'company', 'corporation',
            'finance', 'investment', 'bank', 'trade', 'commercial', 'industry',
            'revenue', 'profit', 'loss', 'earnings', 'quarterly', 'annual',
            'merger', 'acquisition', 'ipo', 'share', 'investor', 'wall street',
            'dow', 'nasdaq', 's&p', 'inflation', 'recession', 'gdp', 'gdp growth'
        ],
        'sports': [
            'sport', 'game', 'match', 'team', 'player', 'championship', 'league',
            'tournament', 'football', 'soccer', 'baseball', 'basketball', 'nba',
            'nfl', 'mlb', 'tennis', 'golf', 'olympics', 'medal', 'score', 'win',
            'lose', 'coach', 'stadium', 'fan', 'season', 'playoff', 'final'
        ],
        'entertainment': [
            'movie', 'film', 'actor', 'actress', 'director', 'Hollywood', 'Bollywood',
            'music', 'album', 'song', 'concert', 'celebrity', 'star', 'famous',
            'tv', 'television', 'show', 'series', 'netflix', 'streaming', 'box office',
            'oscar', 'grammy', 'award', 'festival', 'premiere', 'cast'
        ],
        'health': [
            'health', 'medical', 'hospital', 'doctor', 'disease', 'patient', 'treatment',
            'vaccine', 'pandemic', 'covid', 'coronavirus', 'healthcare', 'medicine',
            'drug', 'fda', 'clinical', 'symptom', 'diagnosis', 'therapy', 'surgery',
            'mental health', 'depression', 'anxiety', 'cancer', 'diabetes', 'heart'
        ],
        'science': [
            'science', 'research', 'scientist', 'study', 'discovery', 'space', 'nasa',
            'moon', 'mars', 'satellite', 'astronaut', 'physics', 'chemistry', 'biology',
            'gene', 'dna', 'climate', 'environment', 'earth', 'solar', 'planet',
            'telescope', 'experiment', 'laboratory', 'academic', 'journal'
        ],
        'world': [
            'international', 'global', 'world', 'foreign', 'country', 'nation',
            'europe', 'asia', 'africa', 'americas', 'middle east', 'european',
            'chinese', 'russian', 'british', 'french', 'german', 'japanese', 'indian',
            'war', 'conflict', 'crisis', 'refugee', 'humanitarian', 'united nations'
        ]
    }
    
    # Score each category
    category_scores = {}
    text_words = set(text_lower.split())
    
    for category, keywords in category_keywords.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        # Normalize by keyword count
        if score > 0:
            category_scores[category] = score / len(keywords)
    
    # Sort by score
    sorted_categories = sorted(
        category_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Format results
    if sorted_categories:
        top_category = sorted_categories[0][0]
        confidence = min(sorted_categories[0][1] * 3, 1.0)  # Scale and cap at 1.0
        
        categories = [
            {
                'category': cat,
                'confidence': min(score * 3, 1.0)
            }
            for cat, score in sorted_categories[:5] if score > 0
        ]
        
        return {
            'primary_category': top_category,
            'confidence': confidence,
            'categories': categories
        }
    
    return {
        'primary_category': 'other',
        'confidence': 0.3,
        'categories': [{'category': 'other', 'confidence': 0.3}]
    }


def _create_error_result(error_message: str, start_time: float) -> Dict[str, Any]:
    """Create an error result"""
    return {
        'success': False,
        'model_type': 'text',
        'categories': [],
        'primary_category': 'unknown',
        'confidence': 0.0,
        'processing_time': time.time() - start_time,
        'metadata': {},
        'error': error_message
    }


def process_text(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for text processing worker.
    
    Args:
        data: Input data containing 'text' key
        
    Returns:
        Processing result
    """
    text = data.get('text', '')
    
    if not text:
        return _create_error_result("No text provided", time.time())
    
    return classify_text(text)


# For direct testing
if __name__ == '__main__':
    test_text = "Apple announces new iPhone with advanced AI features"
    result = process_text({'text': test_text})
    print(json.dumps(result, indent=2))
