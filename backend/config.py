"""
NEWSCAT Configuration - Fast Loading
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    """Fast configuration class"""
    
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = False
    TESTING = False
    
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5000))
    THREADED = True
    PROCESSES = 1
    
    BASE_DIR = BASE_DIR
    BACKEND_DIR = BASE_DIR / 'backend'
    FRONTEND_DIR = BASE_DIR / 'frontend'
    DATA_DIR = BACKEND_DIR / 'data'
    MODEL_DIR = DATA_DIR / 'models' / 'pretrained'
    
    DEFAULT_CLASSIFIER = 'lightning'
    MIN_TEXT_LENGTH = 5
    MAX_TEXT_LENGTH = 50000
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://127.0.0.1:5000,http://localhost:5000,http://localhost:3000,http://127.0.0.1:3000').split(',')
    
    # File upload limits (in bytes)
    # 10MB for images, 50MB for audio, 100MB for video
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max total request size
    
    # File type specific limits
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB
    
    # News categories with display names (50+ topics)
    CATEGORIES = {
        # Core Topics
        'technology': 'Technology',
        'sports': 'Sports',
        'politics': 'Politics',
        'business': 'Business',
        'entertainment': 'Entertainment',
        'health': 'Health',
        'science': 'Science',
        'world': 'World News',
        'education': 'Education',
        'environment': 'Environment',
        
        # Finance & Economy
        'finance': 'Finance',
        'cryptocurrency': 'Cryptocurrency',
        'stock_market': 'Stock Market',
        'economy': 'Economy',
        'real_estate': 'Real Estate',
        'banking': 'Banking',
        'insurance': 'Insurance',
        'taxation': 'Taxation',
        'retail': 'Retail',
        'ecommerce': 'E-Commerce',
        
        # Technology Deep Dive
        'artificial_intelligence': 'Artificial Intelligence',
        'cybersecurity': 'Cybersecurity',
        'blockchain': 'Blockchain',
        'iot': 'Internet of Things',
        'cloud_computing': 'Cloud Computing',
        'software_dev': 'Software Development',
        'hardware': 'Hardware',
        'gaming': 'Gaming',
        'social_media': 'Social Media',
        'startups': 'Startups',
        
        # Science & Research
        'space': 'Space',
        'physics': 'Physics',
        'biology': 'Biology',
        'chemistry': 'Chemistry',
        'medicine': 'Medicine',
        'neuroscience': 'Neuroscience',
        'climate_science': 'Climate Science',
        'genetics': 'Genetics',
        'astronomy': 'Astronomy',
        'oceanography': 'Oceanography',
        
        # Society & Culture
        'travel': 'Travel',
        'food': 'Food',
        'fashion': 'Fashion',
        'art': 'Art',
        'music': 'Music',
        'film': 'Film',
        'literature': 'Literature',
        'photography': 'Photography',
        'dance': 'Dance',
        'theater': 'Theater',
        
        # Crime & Security
        'crime': 'Crime',
        'law_enforcement': 'Law Enforcement',
        'national_security': 'National Security',
        'intelligence': 'Intelligence',
        'cybercrime': 'Cybercrime',
        'fraud': 'Fraud',
        'corruption': 'Corruption',
        'terrorism': 'Terrorism',
        'border_security': 'Border Security',
        'emergency_services': 'Emergency Services',
        
        # Lifestyle & Wellness
        'fitness': 'Fitness',
        'nutrition': 'Nutrition',
        'mental_health': 'Mental Health',
        'relationships': 'Relationships',
        'parenting': 'Parenting',
        'home_living': 'Home Living',
        'pets': 'Pets',
        'hobbies': 'Hobbies',
        'spirituality': 'Spirituality',
        'self_improvement': 'Self Improvement'
    }

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
