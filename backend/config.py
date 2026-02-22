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
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5000))
    
    # Paths
    BASE_DIR = BASE_DIR
    BACKEND_DIR = BASE_DIR / 'backend'
    FRONTEND_DIR = BASE_DIR / 'frontend'
    DATA_DIR = BACKEND_DIR / 'data'
    MODEL_DIR = DATA_DIR / 'models' / 'pretrained'
    TRAINING_DATA_DIR = DATA_DIR / 'training'
    LOG_DIR = BASE_DIR / 'logs'
    
    # Model settings
    DEFAULT_CLASSIFIER = os.getenv('DEFAULT_CLASSIFIER', 'enhanced')
    MODEL_CACHE_DIR = MODEL_DIR
    
    # Text processing
    MIN_TEXT_LENGTH = int(os.getenv('MIN_TEXT_LENGTH', 20))
    MAX_TEXT_LENGTH = int(os.getenv('MAX_TEXT_LENGTH', 10000))
    MAX_WORDS = int(os.getenv('MAX_WORDS', 2000))
    
    # Feature extraction
    TFIDF_MAX_FEATURES = int(os.getenv('TFIDF_MAX_FEATURES', 5000))
    NGRAM_RANGE = (1, 2)  # Fixed tuple for ngram range
    USE_LEMMATIZATION = os.getenv('USE_LEMMATIZATION', 'True').lower() == 'true'
    
    # Training data
    TRAINING_DATA_PATH = Path(os.getenv('TRAINING_DATA_PATH', 
                                       TRAINING_DATA_DIR / 'news_samples.json'))
    
    # Categories - Extended to 35 categories
    CATEGORIES = {
        # Core Categories
        'technology': 'Tech innovations, AI, software, gadgets',
        'sports': 'Sports events, athletes, teams, championships',
        'politics': 'Government, elections, policies, legislation',
        'business': 'Markets, companies, economy, corporate news',
        'entertainment': 'Movies, music, celebrities, streaming',
        
        # Extended Categories
        'health': 'Medical news, healthcare, wellness, diseases',
        'science': 'Scientific discoveries, research, space, physics',
        'world': 'International news, global events, diplomacy',
        'education': 'Schools, universities, learning, academic',
        'environment': 'Climate, nature, conservation, sustainability',
        
        # Specialized Categories
        'finance': 'Cryptocurrency, investing, banking, markets',
        'automotive': 'Cars, electric vehicles, auto industry',
        'travel': 'Tourism, destinations, airlines, hotels',
        'food': 'Restaurants, cooking, nutrition, food industry',
        'fashion': 'Clothing, designers, trends, luxury brands',
        
        # Niche Categories
        'realestate': 'Housing market, property, mortgages',
        'legal': 'Lawsuits, court cases, legal proceedings',
        'religion': 'Faith, spirituality, religious news',
        'lifestyle': 'Wellness, relationships, personal growth',
        'opinion': 'Editorials, commentary, analysis',
        
        # Real Incident Categories
        'accidents': 'Traffic accidents, industrial incidents, crashes, derailments',
        'crime': 'Criminal activities, investigations, arrests, theft, assault',
        'disasters': 'Natural disasters, earthquakes, floods, hurricanes, wildfires',
        'protests': 'Demonstrations, rallies, civil unrest, activism, strikes',
        
        # Human-Centric Categories
        'career': 'Jobs, employment, workplace, hiring, layoffs, professional development',
        'relationships': 'Dating, marriage, family dynamics, divorce, parenting',
        'mentalhealth': 'Mental health awareness, psychology, therapy, depression, anxiety',
        
        # Specialized News Types
        'investigative': 'In-depth reporting, exposés, whistleblower stories, deep dives',
        'breaking': 'Breaking news, urgent alerts, developing stories, live updates',
        'weather': 'Weather forecasts, storms, meteorological news, climate patterns',
        
        # Additional Categories
        'infrastructure': 'Construction, public works, utilities, transportation systems',
        'socialmedia': 'Social media trends, platform news, viral content, influencers',
        'gaming': 'Video games, esports, gaming industry, console news',
        'space': 'Space exploration, satellites, rockets, space missions, astronomy',
        'agriculture': 'Farming, crops, livestock, agricultural policy, food production'
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
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist"""
        try:
            cls.MODEL_DIR.mkdir(parents=True, exist_ok=True)
            cls.TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
            cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directories: {e}")


# Create directories on module load (with error handling)
try:
    Config.ensure_directories()
except Exception as e:
    print(f"Warning: Directory creation failed: {e}")


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