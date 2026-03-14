"""
Fetch Training Data Script

This script fetches news articles from various public RSS feeds, 
cleans the HTML content, and categorizes them based on the feed source.
The scraped data is saved as a JSON dataset to be used for model training.
"""

import os
import sys
import json
import logging
from datetime import datetime
import time
import feedparser
from bs4 import BeautifulSoup
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FetchTrainingData')

# Ensure we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from backend.config import Config
except ImportError:
    # Fallback if running directly from scripts dir without proper PYTHONPATH
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from backend.config import Config

# RSS Feeds mapped to our categories
# Format: { 'RSS_URL': 'category_id' }
RSS_FEEDS = {
    # Technology & AI
    'http://feeds.bbci.co.uk/news/technology/rss.xml': 'technology',
    'https://techcrunch.com/feed/': 'technology',
    'https://www.wired.com/feed/category/science/latest/rss': 'science',
    'https://www.wired.com/feed/category/security/latest/rss': 'cybersecurity',
    'https://www.theverge.com/rss/index.xml': 'technology',
    'https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml': 'technology',
    'https://www.zdnet.com/topic/artificial-intelligence/rss.xml': 'artificial_intelligence',
    'https://cointelegraph.com/rss': 'cryptocurrency',

    # Business & Finance
    'http://feeds.bbci.co.uk/news/business/rss.xml': 'business',
    'https://rss.nytimes.com/services/xml/rss/nyt/Business.xml': 'business',
    'https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml': 'economy',
    'https://www.wsj.com/xml/rss/3_7014.xml': 'business', # WSJ Business
    'https://www.wsj.com/xml/rss/3_7031.xml': 'finance', # WSJ Markets
    'https://search.cnbc.com/rs/search/combinedcms/view.xml?profile=1069': 'business', # CNBC Business
    'https://search.cnbc.com/rs/search/combinedcms/view.xml?profile=1097': 'finance', # CNBC Finance

    # Science & Health
    'http://feeds.bbci.co.uk/news/science_and_environment/rss.xml': 'science',
    'https://rss.nytimes.com/services/xml/rss/nyt/Science.xml': 'science',
    'https://www.sciencemag.org/rss/news_current.xml': 'science',
    'http://feeds.bbci.co.uk/news/health/rss.xml': 'health',
    'https://rss.nytimes.com/services/xml/rss/nyt/Health.xml': 'health',
    'https://www.sciencedaily.com/rss/health_medicine.xml': 'medicine',
    'https://www.sciencedaily.com/rss/space_time.xml': 'space',
    'https://www.sciencedaily.com/rss/earth_climate/climate.xml': 'climate_change',

    # Politics & World
    'http://feeds.bbci.co.uk/news/world/rss.xml': 'world',
    'http://feeds.bbci.co.uk/news/politics/rss.xml': 'politics',
    'https://rss.nytimes.com/services/xml/rss/nyt/World.xml': 'world',
    'https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml': 'politics',
    'https://www.aljazeera.com/xml/rss/all.xml': 'world',
    'https://feeds.npr.org/1004/rss.xml': 'world',
    'https://feeds.npr.org/1014/rss.xml': 'politics',

    # Sports
    'https://www.espn.com/espn/rss/news': 'sports',
    'https://sports.yahoo.com/rss/': 'sports',
    'https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml': 'sports',

    # Entertainment & Culture
    'http://feeds.bbci.co.uk/news/entertainment_and_arts/rss.xml': 'entertainment',
    'https://rss.nytimes.com/services/xml/rss/nyt/Movies.xml': 'film_tv',
    'https://rss.nytimes.com/services/xml/rss/nyt/Music.xml': 'music',
    'https://www.billboard.com/feed/': 'music',
    'https://variety.com/feed/': 'entertainment',
    
    # Environment
    'https://www.theguardian.com/environment/rss': 'environment',
    
    # Education
    'https://www.theguardian.com/education/rss': 'education',
}

def clean_html(raw_html: str) -> str:
    """Remove HTML tags and clean up text."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=' ')
    # Clean up whitespace
    text = ' '.join(text.split())
    return text

def fetch_feed(url: str, category: str, max_items: int = 50) -> list:
    """Fetch entries from a single RSS feed."""
    logger.info(f"Fetching {url} -> mapped to '{category}'")
    
    try:
        # Use requests with a user agent to avoid some blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {url}: HTTP {response.status_code}")
            return []
            
        feed = feedparser.parse(response.content)
        
        articles = []
        for entry in feed.entries[:max_items]:
            # Extract content (prefer 'content' standard, fallback to 'summary' or 'description')
            content = ""
            if hasattr(entry, 'content'):
                content = entry.content[0].value
            elif hasattr(entry, 'summary'):
                content = entry.summary
            elif hasattr(entry, 'description'):
                content = entry.description
                
            title = entry.title if hasattr(entry, 'title') else ""
            
            # Combine title and content for better context
            full_text = clean_html(f"{title}. {content}")
            
            # Only keep articles with sufficient length
            if len(full_text) > 100:
                articles.append({
                    "text": full_text,
                    "label": category,
                    "source": url,
                    "title": title
                })
                
        logger.info(f"  Got {len(articles)} articles")
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return []

def collect_data(output_file: str):
    """Collect data from all mapped RSS feeds."""
    all_articles = []
    category_counts = {}
    
    logger.info("Starting RSS feed collection...")
    
    for url, category in RSS_FEEDS.items():
        if category not in CONFIG_CATEGORIES:
            logger.warning(f"Warning: mapped category '{category}' is not in Config.CATEGORIES. Fallback to 'world'")
            category = 'world'
            
        articles = fetch_feed(url, category)
        all_articles.extend(articles)
        
        category_counts[category] = category_counts.get(category, 0) + len(articles)
        # Sleep briefly to avoid hammering servers too fast
        time.sleep(1)
        
    logger.info(f"Collection complete. Total articles: {len(all_articles)}")
    logger.info("Category breakdown:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cat}: {count}")
        
    # Save to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Dataset saved to {output_path}")
    return all_articles

if __name__ == "__main__":
    # Validate categories against config
    try:
        CONFIG_CATEGORIES = list(Config.CATEGORIES.keys())
        logger.info(f"Loaded {len(CONFIG_CATEGORIES)} categories from Config")
    except Exception as e:
        logger.error(f"Failed to load categories from Config: {e}")
        CONFIG_CATEGORIES = list(set(RSS_FEEDS.values())) # fallback
        
    # Create output directory
    project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = project_root / 'data' / 'training'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = data_dir / 'online_dataset.json'
    
    collect_data(output_file)
