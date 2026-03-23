"""
NEWSCAT Configuration - Fast Loading
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Fast configuration class"""

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-key-change-in-production")
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    DEBUG = False
    TESTING = False

    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))
    THREADED = True
    PROCESSES = 1

    BASE_DIR = BASE_DIR
    BACKEND_DIR = BASE_DIR / "backend"
    FRONTEND_DIR = BASE_DIR / "frontend"
    DATA_DIR = BACKEND_DIR / "data"
    MODEL_DIR = DATA_DIR / "models" / "pretrained"

    DEFAULT_CLASSIFIER = "simple"
    MIN_TEXT_LENGTH = 5
    MAX_TEXT_LENGTH = 50000

    # CORS Configuration
    CORS_ORIGINS = os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5000,http://localhost:5000,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")

    # File upload limits (in bytes)
    # 10MB for images, 50MB for audio, 100MB for video
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max total request size

    # File type specific limits
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_AUDIO_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_VIDEO_SIZE = 100 * 1024 * 1024  # 100MB

    # News categories with display names (50+ topics)
    CATEGORIES = {
        'technology': 'Technology',
        'artificial_intelligence': 'Artificial Intelligence',
        'cybersecurity': 'Cybersecurity',
        'software_development': 'Software Development',
        'hardware_devices': 'Hardware & Devices',
        'cloud_computing': 'Cloud Computing',
        'telecommunications': 'Telecommunications',
        'robotics': 'Robotics & Automation',
        'internet_of_things': 'Internet of Things',
        'virtual_reality': 'Virtual & Augmented Reality',
        'data_science': 'Data Science & Analytics',
        'blockchain_tech': 'Blockchain Technology',
        'gaming_tech': 'Gaming Technology',
        'social_media_tech': 'Social Media Tech',
        'consumer_electronics': 'Consumer Electronics',
        'semiconductors': 'Semiconductors',
        'nanotechnology': 'Nanotechnology',
        'biotechnology': 'Biotechnology',
        'business': 'Business & Finance',
        'finance': 'Finance & Markets',
        'startups': 'Startups & VC',
        'economy': 'Economy',
        'real_estate': 'Real Estate',
        'marketing': 'Marketing & Advertising',
        'entrepreneurship': 'Entrepreneurship',
        'ecommerce': 'E-Commerce',
        'cryptocurrency': 'Cryptocurrency',
        'banking': 'Banking & Lending',
        'corporate_governance': 'Corporate Governance',
        'human_resources': 'Human Resources',
        'leadership': 'Leadership & Management',
        'supply_chain': 'Supply Chain & Logistics',
        'insurance': 'Insurance',
        'accounting': 'Accounting & Tax',
        'investments': 'Investments & Wealth',
        'international_trade': 'International Trade',
        'health': 'Health & Wellness',
        'medicine': 'Medicine & Clinical',
        'mental_health': 'Mental Health',
        'fitness': 'Fitness & Exercise',
        'nutrition': 'Nutrition & Diet',
        'public_health': 'Public Health',
        'healthcare_policy': 'Healthcare Policy',
        'alternative_medicine': 'Alternative Medicine',
        'pharmaceuticals': 'Pharmaceuticals',
        'pediatrics': 'Pediatrics',
        'aging_geriatrics': 'Aging & Geriatrics',
        "womens_health": "Women's Health",
        "mens_health": "Men's Health",
        'dentistry': 'Dentistry & Oral Care',
        'veterinary': 'Veterinary Medicine',
        'dermatology': 'Dermatology',
        'science': 'Science & Environment',
        'space': 'Space & Astronomy',
        'climate_change': 'Climate Change',
        'environment': 'Environment & Ecology',
        'physics': 'Physics',
        'biology': 'Biology',
        'chemistry': 'Chemistry',
        'genetics': 'Genetics & DNA',
        'archaeology': 'Archaeology & Anthropology',
        'oceanography': 'Oceanography',
        'geology': 'Geology & Earth Sciences',
        'paleontology': 'Paleontology',
        'meteorology': 'Meteorology & Weather',
        'zoology': 'Zoology & Animals',
        'botany': 'Botany & Plants',
        'energy': 'Energy & Power',
        'renewable_energy': 'Renewable Energy',
        'materials_science': 'Materials Science',
        'politics': 'Politics & Government',
        'elections': 'Elections & Campaigns',
        'geopolitics': 'Geopolitics',
        'international_relations': 'International Relations',
        'public_policy': 'Public Policy',
        'law_justice': 'Law & Justice',
        'war_conflict': 'War & Conflict',
        'human_rights': 'Human Rights',
        'immigration': 'Immigration & Borders',
        'civil_rights': 'Civil Rights',
        'diplomacy': 'Diplomacy',
        'national_security': 'National Security',
        'political_scandals': 'Political Scandals',
        'local_government': 'Local Government',
        'global_organizations': 'Global Organizations',
        'activism': 'Activism & Protests',
        'entertainment': 'Entertainment & Arts',
        'film_tv': 'Film & Television',
        'music': 'Music',
        'celebrity': 'Celebrity News',
        'pop_culture': 'Pop Culture',
        'video_games': 'Video Games',
        'books_literature': 'Books & Literature',
        'performing_arts': 'Performing Arts',
        'fine_arts': 'Fine Arts',
        'photography': 'Photography',
        'fashion': 'Fashion & Style',
        'anime_manga': 'Anime & Manga',
        'podcasts': 'Podcasts & Radio',
        'awards_shows': 'Awards Shows',
        'streaming': 'Streaming Platforms',
        'comics': 'Comics & Graphic Novels',
        'sports': 'Sports',
        'football_soccer': 'Football (Soccer)',
        'american_football': 'American Football',
        'basketball': 'Basketball',
        'baseball': 'Baseball',
        'tennis': 'Tennis',
        'golf': 'Golf',
        'motorsports': 'Motorsports',
        'combat_sports': 'Combat Sports',
        'athletics_olympics': 'Olympics & Athletics',
        'hockey': 'Hockey',
        'cricket': 'Cricket',
        'extreme_sports': 'Extreme Sports',
        'cycling': 'Cycling',
        'rugby': 'Rugby',
        'esports': 'E-Sports',
        'lifestyle': 'Lifestyle & Society',
        'travel': 'Travel & Tourism',
        'food_dining': 'Food & Dining',
        'education': 'Education & Learning',
        'parenting': 'Parenting & Family',
        'relationships': 'Relationships',
        'home_garden': 'Home & Garden',
        'pets_animals': 'Pets & Animals',
        'religion_spirituality': 'Religion & Spirituality',
        'crime': 'Crime & True Crime',
        'culture_trends': 'Culture & Trends',
        'social_issues': 'Social Issues',
        'personal_finance': 'Personal Finance',
        'diy_crafts': 'DIY & Crafts',
        'automotive': 'Automotive & Cars',
        'beauty': 'Beauty & Cosmetics',
        # New Real-Time & Live News Categories
        'breaking_news': 'Breaking News & Alerts',
        'real_time_events': 'Live Events & Real-Time Coverage',
        'crisis_response': 'Crisis & Emergency Response',
        'investigative_journalism': 'Investigative Journalism',
        'weather_alerts': 'Severe Weather Alerts',
        'market_movers': 'Market Movers',
        'press_releases': 'Press Releases',
        'trending_topics': 'Trending Topics',
        'sports_live': 'Sports Live Updates',
        'opinion_editorial': 'Opinion & Editorial',
        'fact_check': 'Fact Check',
        # Additional Specialized Categories
        'disability_accessibility': 'Disability & Accessibility',
        'quantum_computing': 'Quantum Computing',
        'space_tourism': 'Space Tourism',
        'food_safety': 'Food Safety',
        'digital_privacy': 'Digital Privacy',
        'workforce_automation': 'Workforce Automation',
    }

    # Real-time categories for priority classification
    REAL_TIME_CATEGORIES = {
        'breaking_news': {
            'name': 'Breaking News',
            'priority_keywords': ['urgent', 'breaking', 'developing story', 'just in', 'news alert', 'flash'],
            'weight': 1.0,
        },
        'live_updates': {
            'name': 'Live Updates',
            'priority_keywords': ['live coverage', 'real-time', 'ongoing', 'minute-by-minute', 'live blog'],
            'weight': 0.95,
        },
        'market_movers': {
            'name': 'Market Movers',
            'priority_keywords': ['stock surge', 'market plunge', 'rally', 'selloff', 'stocks jump', 'stocks fall'],
            'weight': 0.90,
        },
        'weather_alerts': {
            'name': 'Weather Alerts',
            'priority_keywords': ['tornado warning', 'flood alert', 'storm surge', 'hurricane watch', 'severe weather'],
            'weight': 1.0,
        },
        'press_releases': {
            'name': 'Press Releases',
            'priority_keywords': ['official announcement', 'press conference', 'announces', 'unveils', 'reveals'],
            'weight': 0.85,
        },
        'trending_topics': {
            'name': 'Trending Topics',
            'priority_keywords': ['viral', 'trending', 'trending now', 'going viral', 'buzzing'],
            'weight': 0.80,
        },
        'sports_live': {
            'name': 'Sports Live',
            'priority_keywords': ['live score', 'in-game', 'half-time', 'final score', 'game time'],
            'weight': 0.90,
        },
        'investigative_journalism': {
            'name': 'Investigative Journalism',
            'priority_keywords': ['investigation', 'exposé', 'inquiry', 'probe', 'exclusive', 'uncovered'],
            'weight': 0.85,
        },
        'opinion_editorial': {
            'name': 'Opinion & Editorial',
            'priority_keywords': ['opinion', 'editorial', 'viewpoint', 'commentary', 'analysis', ' perspective'],
            'weight': 0.80,
        },
        'fact_check': {
            'name': 'Fact Check',
            'priority_keywords': ['fact check', 'false', 'verified', 'misleading', 'true or false', 'verification'],
            'weight': 0.90,
        },
    }


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False
