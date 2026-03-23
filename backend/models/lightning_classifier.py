"""
Lightning Classifier — CategoryKnowledgeGraph + QuantumClassifier for NEWSCAT.
Provides hierarchical topic lookup (used by response_formatter.py) and a
fallback classifier that mirrors the SimpleNewsClassifier interface.
"""

import logging
from typing import Dict, List, Any, Optional

from backend.models.taxonomy import TAXONOMY_KEYWORDS

logger = logging.getLogger(__name__)


class CategoryKnowledgeGraph:
    """
    Static knowledge graph mapping each category slug to its metadata.
    Used by response_formatter.py to derive parent/child topic relationships.
    """

    # Auto-build from TAXONOMY_KEYWORDS + parent hierarchy
    CATEGORIES: Dict[str, Dict[str, Any]] = {}

    # Parent mapping — top-level categories map to None; sub-categories map to parent
    _PARENT_MAP = {
        # Top-level Umbrella Mapping
        "technology": "Technology & Innovation",
        "business": "Business & Economy",
        "health": "Health & Wellness",
        "science": "Science & Discovery",
        "politics": "Politics & Government",
        "entertainment": "Entertainment & Media",
        "sports": "Sports & Athletics",
        "lifestyle": "Lifestyle & Society",
        "breaking_news": "Global Events",
        
        # Technology sub-categories
        "artificial_intelligence": "technology",
        "cybersecurity": "technology",
        "software_development": "technology",
        "hardware_devices": "technology",
        "cloud_computing": "technology",
        "telecommunications": "technology",
        "robotics": "technology",
        "internet_of_things": "technology",
        "virtual_reality": "technology",
        "data_science": "technology",
        "blockchain_tech": "technology",
        "gaming_tech": "technology",
        "social_media_tech": "technology",
        "consumer_electronics": "technology",
        "semiconductors": "technology",
        "nanotechnology": "technology",
        "biotechnology": "technology",
        # Business sub-categories
        "finance": "business",
        "startups": "business",
        "economy": "business",
        "real_estate": "business",
        "marketing": "business",
        "entrepreneurship": "business",
        "ecommerce": "business",
        "cryptocurrency": "business",
        "banking": "business",
        "corporate_governance": "business",
        "human_resources": "business",
        "leadership": "business",
        "supply_chain": "business",
        "insurance": "business",
        "accounting": "business",
        "investments": "business",
        "international_trade": "business",
        # Health sub-categories
        "medicine": "health",
        "mental_health": "health",
        "fitness": "health",
        "nutrition": "health",
        "public_health": "health",
        "healthcare_policy": "health",
        "alternative_medicine": "health",
        "pharmaceuticals": "health",
        "pediatrics": "health",
        "aging_geriatrics": "health",
        "womens_health": "health",
        "mens_health": "health",
        "dentistry": "health",
        "veterinary": "health",
        "dermatology": "health",
        # Science sub-categories
        "space": "science",
        "climate_change": "science",
        "environment": "science",
        "physics": "science",
        "biology": "science",
        "chemistry": "science",
        "genetics": "science",
        "archaeology": "science",
        "oceanography": "science",
        "geology": "science",
        "paleontology": "science",
        "meteorology": "science",
        "zoology": "science",
        "botany": "science",
        "energy": "science",
        "renewable_energy": "science",
        "materials_science": "science",
        # Politics sub-categories
        "elections": "politics",
        "geopolitics": "politics",
        "international_relations": "politics",
        "public_policy": "politics",
        "law_justice": "politics",
        "war_conflict": "politics",
        "human_rights": "politics",
        "immigration": "politics",
        "civil_rights": "politics",
        "diplomacy": "politics",
        "national_security": "politics",
        "political_scandals": "politics",
        "local_government": "politics",
        "global_organizations": "politics",
        "activism": "politics",
        # Entertainment sub-categories
        "film_tv": "entertainment",
        "music": "entertainment",
        "celebrity": "entertainment",
        "pop_culture": "entertainment",
        "video_games": "entertainment",
        "books_literature": "entertainment",
        "performing_arts": "entertainment",
        "fine_arts": "entertainment",
        "photography": "entertainment",
        "fashion": "entertainment",
        "anime_manga": "entertainment",
        "podcasts": "entertainment",
        "awards_shows": "entertainment",
        "streaming": "entertainment",
        "comics": "entertainment",
        # Sports sub-categories
        "football_soccer": "sports",
        "american_football": "sports",
        "basketball": "sports",
        "baseball": "sports",
        "tennis": "sports",
        "golf": "sports",
        "motorsports": "sports",
        "combat_sports": "sports",
        "athletics_olympics": "sports",
        "hockey": "sports",
        "cricket": "sports",
        "extreme_sports": "sports",
        "cycling": "sports",
        "rugby": "sports",
        "esports": "sports",
        # Lifestyle sub-categories
        "travel": "lifestyle",
        "food_dining": "lifestyle",
        "education": "lifestyle",
        "parenting": "lifestyle",
        "relationships": "lifestyle",
        "home_garden": "lifestyle",
        "pets_animals": "lifestyle",
        "religion_spirituality": "lifestyle",
        "crime": "lifestyle",
        "culture_trends": "lifestyle",
        "social_issues": "lifestyle",
        "personal_finance": "lifestyle",
        "diy_crafts": "lifestyle",
        "automotive": "lifestyle",
        "beauty": "lifestyle",
        # Breaking / Real-time sub-categories
        "real_time_events": "breaking_news",
        "crisis_response": "breaking_news",
        "market_movers": "business",
        "weather_alerts": "science",
        "press_releases": "business",
        "trending_topics": "entertainment",
        "sports_live": "sports",
        # Additional specialized
        "disability_accessibility": "lifestyle",
        "quantum_computing": "technology",
        "space_tourism": "science",
        "food_safety": "health",
        "digital_privacy": "technology",
        "workforce_automation": "business",
    }

    @classmethod
    def _build(cls):
        """Build the CATEGORIES graph from taxonomy + parent map."""
        for cat_slug, keywords in TAXONOMY_KEYWORDS.items():
            parent = cls._PARENT_MAP.get(cat_slug)
            core_words = [kw for kw in keywords[:5]]
            cls.CATEGORIES[cat_slug] = {
                "parent": parent,
                "neural_weight": 0.9,
                "embeddings": {
                    "core": core_words,
                    "related": [f"{w} news" for w in core_words[:3]],
                },
                "semantic_context": keywords[:3],
                "confidence_multipliers": {
                    "high": 3.5,
                    "medium": 2.0,
                    "low": 1.0,
                },
            }


# Build the graph on import
CategoryKnowledgeGraph._build()


class QuantumClassifier:
    """
    Fallback classifier using the CategoryKnowledgeGraph.
    Uses the same API as SimpleNewsClassifier to maintain compatibility.
    """

    def __init__(self, name: str = "QuantumClassifier", version: str = "9.0"):
        self.name = name
        self.version = version
        self._graph = CategoryKnowledgeGraph.CATEGORIES
        logger.info(
            f"{self.name} v{self.version} initialized with "
            f"{len(self._graph)} categories"
        )

    def classify(
        self,
        text: str,
        include_confidence: bool = True,
        include_all_scores: bool = False,
    ) -> Dict[str, Any]:
        """Classify text using SimpleNewsClassifier (delegate)."""
        # Import here to avoid circular dependency at module level
        from backend.models.simple_classifier import SimpleNewsClassifier

        classifier = SimpleNewsClassifier(name=self.name, version=self.version)
        return classifier.classify(text, include_confidence, include_all_scores)
