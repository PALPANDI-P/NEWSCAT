"""
NEWSCAT Configuration Module
Centralized configuration management
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Base configuration class"""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    FLASK_APP = os.getenv('FLASK_APP', 'app.py')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    TESTING = False
    
    # Server
    HOST = os.getenv('HOST', 'localhost')
    PORT = int(os.getenv('PORT', 5000))
    
    # Paths
    BASE_DIR = BASE_DIR
    BACKEND_DIR = BASE_DIR / 'backend'
    FRONTEND_DIR = BASE_DIR / 'frontend'
    DATA_DIR = BACKEND_DIR / 'data'
    MODEL_DIR = DATA_DIR / 'models' / 'pretrained'
    TRAINING_DATA_DIR = DATA_DIR / 'training'
    LOG_DIR = BASE_DIR / 'logs'
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model settings
    DEFAULT_CLASSIFIER = os.getenv('DEFAULT_CLASSIFIER', 'enhanced')
    MODEL_CACHE_DIR = MODEL_DIR
    
    # Text processing
    MIN_TEXT_LENGTH = int(os.getenv('MIN_TEXT_LENGTH', 20))
    MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 10000))
    MAX_WORDS = int(os.getenv('MAX_WORDS', 2000))
    
    # Feature extraction
    TFIDF_MAX_FEATURES = int(os.getenv('TFIDF_MAX_FEATURES', 5000))
    NGRAM_RANGE = tuple(map(int, os.getenv('NGRAM_RANGE', '1,3').split(',')))
    USE_LEMMATIZATION = os.getenv('USE_LEMMATIZATION', 'True').lower() == 'true'
    
    # Training data
    TRAINING_DATA_PATH = Path(os.getenv('TRAINING_DATA_PATH', 
                                       TRAINING_DATA_DIR / 'news_samples.json'))
    
    # Categories
    CATEGORIES = {
        'politics': 'Government, elections, policies',
        'sports': 'Sports events, athletes, teams',
        'technology': 'Tech innovations, AI, software',
        'business': 'Markets, companies, economy',
        'entertainment': 'Movies, music, celebrities',
        'health': 'Medical news, healthcare',
        'science': 'Scientific discoveries, research',
        'world': 'International news, global events',
        'education': 'Schools, universities, learning',
        'environment': 'Climate, nature, conservation'
    }
    
    # Performance
    ENABLE_CACHING = os.getenv('ENABLE_CACHING', 'True').lower() == 'true'
    CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', 3600))
    WORKERS = int(os.getenv('WORKERS', 4))
    
    # Security
    CORS_ORIGINS = ['http://localhost:5000', 'http://127.0.0.1:5000']
    
    # Feature flags
    ENABLE_ADVANCED_FEATURES = True
    ENABLE_SENTIMENT_ANALYSIS = True
    ENABLE_ENTITY_EXTRACTION = True
    ENABLE_KEYWORD_EXTRACTION = True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    FLASK_ENV = 'development'
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'
    LOG_LEVEL = 'WARNING'
    SECRET_KEY = os.getenv('SECRET_KEY')  # Must be set


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}