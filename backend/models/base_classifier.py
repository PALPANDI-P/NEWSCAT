"""
Base Classifier - Abstract base class for all classifiers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseNewsClassifier(ABC):
    """Abstract base class for all news classifiers"""
    
    def __init__(self, name: str = "BaseClassifier", config: Dict = None):
        self.name = name
        self.version = "2.0.0"
        self.config = config or {}
        self.categories = self._get_categories()
        self.training_date = None
        self.accuracy = None
        self.is_trained = False
        
    @abstractmethod
    def classify(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Classify a news article
        
        Args:
            text: Input news text
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with classification results
        """
        pass
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[str], **kwargs) -> bool:
        """
        Train the classifier
        
        Args:
            texts: Training texts
            labels: Corresponding labels
            **kwargs: Training parameters
            
        Returns:
            True if training successful
        """
        pass
    
    def _get_categories(self) -> Dict[str, str]:
        """Get news categories"""
        from backend.config import Config
        return Config.CATEGORIES
    
    def _validate_input(self, text: str) -> bool:
        """Validate input text"""
        if not text or not isinstance(text, str):
            logger.warning("Invalid input: not a string")
            return False
        
        text = text.strip()
        if len(text) < self.config.get('MIN_TEXT_LENGTH', 20):
            logger.warning(f"Text too short: {len(text)} chars")
            return False
        
        if len(text) > self.config.get('MAX_TEXT_LENGTH', 10000):
            logger.warning(f"Text too long: {len(text)} chars")
            return False
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """Get classifier information"""
        return {
            'name': self.name,
            'version': self.version,
            'categories': list(self.categories.keys()),
            'trained': self.is_trained,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'accuracy': self.accuracy
        }
    
    def _create_response(self, category: str, confidence: float, 
                        features: Optional[Dict] = None) -> Dict[str, Any]:
        """Create standardized response"""
        response = {
            'status': 'success',
            'category': category,
            'category_name': self.categories.get(category, category),
            'confidence': float(confidence),
            'model': self.name,
            'model_version': self.version,
            'timestamp': datetime.now().isoformat()
        }
        
        if features:
            response['features'] = features
            
        return response
    
    def save_model(self, path: str) -> bool:
        """Save model to disk"""
        import joblib
        try:
            joblib.dump(self, path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load_model(cls, path: str):
        """Load model from disk"""
        import joblib
        try:
            model = joblib.load(path)
            logger.info(f"Model loaded from {path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None