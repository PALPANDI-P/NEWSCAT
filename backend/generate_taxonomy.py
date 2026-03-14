import re
import json
import json

CATEGORIES_DEF = {
    # 1. TECHNOLOGY (18)
    "technology": {"parent": None, "name": "Technology"},
    "artificial_intelligence": {"parent": "technology", "name": "Artificial Intelligence"},
    "cybersecurity": {"parent": "technology", "name": "Cybersecurity"},
    "software_development": {"parent": "technology", "name": "Software Development"},
    "hardware_devices": {"parent": "technology", "name": "Hardware & Devices"},
    "cloud_computing": {"parent": "technology", "name": "Cloud Computing"},
    "telecommunications": {"parent": "technology", "name": "Telecommunications"},
    "robotics": {"parent": "technology", "name": "Robotics & Automation"},
    "internet_of_things": {"parent": "technology", "name": "Internet of Things"},
    "virtual_reality": {"parent": "technology", "name": "Virtual & Augmented Reality"},
    "data_science": {"parent": "technology", "name": "Data Science & Analytics"},
    "blockchain_tech": {"parent": "technology", "name": "Blockchain Technology"},
    "gaming_tech": {"parent": "technology", "name": "Gaming Technology"},
    "social_media_tech": {"parent": "technology", "name": "Social Media Tech"},
    "consumer_electronics": {"parent": "technology", "name": "Consumer Electronics"},
    "semiconductors": {"parent": "technology", "name": "Semiconductors"},
    "nanotechnology": {"parent": "technology", "name": "Nanotechnology"},
    "biotechnology": {"parent": "technology", "name": "Biotechnology"},

    # 2. BUSINESS & FINANCE (18)
    "business": {"parent": None, "name": "Business & Finance"},
    "finance": {"parent": "business", "name": "Finance & Markets"},
    "startups": {"parent": "business", "name": "Startups & VC"},
    "economy": {"parent": "business", "name": "Economy"},
    "real_estate": {"parent": "business", "name": "Real Estate"},
    "marketing": {"parent": "business", "name": "Marketing & Advertising"},
    "entrepreneurship": {"parent": "business", "name": "Entrepreneurship"},
    "ecommerce": {"parent": "business", "name": "E-Commerce"},
    "cryptocurrency": {"parent": "business", "name": "Cryptocurrency"},
    "banking": {"parent": "business", "name": "Banking & Lending"},
    "corporate_governance": {"parent": "business", "name": "Corporate Governance"},
    "human_resources": {"parent": "business", "name": "Human Resources"},
    "leadership": {"parent": "business", "name": "Leadership & Management"},
    "supply_chain": {"parent": "business", "name": "Supply Chain & Logistics"},
    "insurance": {"parent": "business", "name": "Insurance"},
    "accounting": {"parent": "business", "name": "Accounting & Tax"},
    "investments": {"parent": "business", "name": "Investments & Wealth"},
    "international_trade": {"parent": "business", "name": "International Trade"},

    # 3. HEALTH & WELLNESS (16)
    "health": {"parent": None, "name": "Health & Wellness"},
    "medicine": {"parent": "health", "name": "Medicine & Clinical"},
    "mental_health": {"parent": "health", "name": "Mental Health"},
    "fitness": {"parent": "health", "name": "Fitness & Exercise"},
    "nutrition": {"parent": "health", "name": "Nutrition & Diet"},
    "public_health": {"parent": "health", "name": "Public Health"},
    "healthcare_policy": {"parent": "health", "name": "Healthcare Policy"},
    "alternative_medicine": {"parent": "health", "name": "Alternative Medicine"},
    "pharmaceuticals": {"parent": "health", "name": "Pharmaceuticals"},
    "pediatrics": {"parent": "health", "name": "Pediatrics"},
    "aging_geriatrics": {"parent": "health", "name": "Aging & Geriatrics"},
    "womens_health": {"parent": "health", "name": "Women's Health"},
    "mens_health": {"parent": "health", "name": "Men's Health"},
    "dentistry": {"parent": "health", "name": "Dentistry & Oral Care"},
    "veterinary": {"parent": "health", "name": "Veterinary Medicine"},
    "dermatology": {"parent": "health", "name": "Dermatology"},

    # 4. SCIENCE & ENVIRONMENT (18)
    "science": {"parent": None, "name": "Science & Environment"},
    "space": {"parent": "science", "name": "Space & Astronomy"},
    "climate_change": {"parent": "science", "name": "Climate Change"},
    "environment": {"parent": "science", "name": "Environment & Ecology"},
    "physics": {"parent": "science", "name": "Physics"},
    "biology": {"parent": "science", "name": "Biology"},
    "chemistry": {"parent": "science", "name": "Chemistry"},
    "genetics": {"parent": "science", "name": "Genetics & DNA"},
    "archaeology": {"parent": "science", "name": "Archaeology & Anthropology"},
    "oceanography": {"parent": "science", "name": "Oceanography"},
    "geology": {"parent": "science", "name": "Geology & Earth Sciences"},
    "paleontology": {"parent": "science", "name": "Paleontology"},
    "meteorology": {"parent": "science", "name": "Meteorology & Weather"},
    "zoology": {"parent": "science", "name": "Zoology & Animals"},
    "botany": {"parent": "science", "name": "Botany & Plants"},
    "energy": {"parent": "science", "name": "Energy & Power"},
    "renewable_energy": {"parent": "science", "name": "Renewable Energy"},
    "materials_science": {"parent": "science", "name": "Materials Science"},

    # 5. POLITICS & WORLD (16)
    "politics": {"parent": None, "name": "Politics & Government"},
    "elections": {"parent": "politics", "name": "Elections & Campaigns"},
    "geopolitics": {"parent": "politics", "name": "Geopolitics"},
    "international_relations": {"parent": "politics", "name": "International Relations"},
    "public_policy": {"parent": "politics", "name": "Public Policy"},
    "law_justice": {"parent": "politics", "name": "Law & Justice"},
    "war_conflict": {"parent": "politics", "name": "War & Conflict"},
    "human_rights": {"parent": "politics", "name": "Human Rights"},
    "immigration": {"parent": "politics", "name": "Immigration & Borders"},
    "civil_rights": {"parent": "politics", "name": "Civil Rights"},
    "diplomacy": {"parent": "politics", "name": "Diplomacy"},
    "national_security": {"parent": "politics", "name": "National Security"},
    "political_scandals": {"parent": "politics", "name": "Political Scandals"},
    "local_government": {"parent": "politics", "name": "Local Government"},
    "global_organizations": {"parent": "politics", "name": "Global Organizations"},
    "activism": {"parent": "politics", "name": "Activism & Protests"},

    # 6. ENTERTAINMENT & ARTS (16)
    "entertainment": {"parent": None, "name": "Entertainment & Arts"},
    "film_tv": {"parent": "entertainment", "name": "Film & Television"},
    "music": {"parent": "entertainment", "name": "Music"},
    "celebrity": {"parent": "entertainment", "name": "Celebrity News"},
    "pop_culture": {"parent": "entertainment", "name": "Pop Culture"},
    "video_games": {"parent": "entertainment", "name": "Video Games"},
    "books_literature": {"parent": "entertainment", "name": "Books & Literature"},
    "performing_arts": {"parent": "entertainment", "name": "Performing Arts"},
    "fine_arts": {"parent": "entertainment", "name": "Fine Arts"},
    "photography": {"parent": "entertainment", "name": "Photography"},
    "fashion": {"parent": "entertainment", "name": "Fashion & Style"},
    "anime_manga": {"parent": "entertainment", "name": "Anime & Manga"},
    "podcasts": {"parent": "entertainment", "name": "Podcasts & Radio"},
    "awards_shows": {"parent": "entertainment", "name": "Awards Shows"},
    "streaming": {"parent": "entertainment", "name": "Streaming Platforms"},
    "comics": {"parent": "entertainment", "name": "Comics & Graphic Novels"},

    # 7. SPORTS (15)
    "sports": {"parent": None, "name": "Sports"},
    "football_soccer": {"parent": "sports", "name": "Football (Soccer)"},
    "american_football": {"parent": "sports", "name": "American Football"},
    "basketball": {"parent": "sports", "name": "Basketball"},
    "baseball": {"parent": "sports", "name": "Baseball"},
    "tennis": {"parent": "sports", "name": "Tennis"},
    "golf": {"parent": "sports", "name": "Golf"},
    "motorsports": {"parent": "sports", "name": "Motorsports"},
    "combat_sports": {"parent": "sports", "name": "Combat Sports"},
    "athletics_olympics": {"parent": "sports", "name": "Olympics & Athletics"},
    "hockey": {"parent": "sports", "name": "Hockey"},
    "cricket": {"parent": "sports", "name": "Cricket"},
    "extreme_sports": {"parent": "sports", "name": "Extreme Sports"},
    "cycling": {"parent": "sports", "name": "Cycling"},
    "rugby": {"parent": "sports", "name": "Rugby"},
    "esports": {"parent": "sports", "name": "E-Sports"},

    # 8. LIFESTYLE & SOCIETY (16)
    "lifestyle": {"parent": None, "name": "Lifestyle & Society"},
    "travel": {"parent": "lifestyle", "name": "Travel & Tourism"},
    "food_dining": {"parent": "lifestyle", "name": "Food & Dining"},
    "education": {"parent": "lifestyle", "name": "Education & Learning"},
    "parenting": {"parent": "lifestyle", "name": "Parenting & Family"},
    "relationships": {"parent": "lifestyle", "name": "Relationships"},
    "home_garden": {"parent": "lifestyle", "name": "Home & Garden"},
    "pets_animals": {"parent": "lifestyle", "name": "Pets & Animals"},
    "religion_spirituality": {"parent": "lifestyle", "name": "Religion & Spirituality"},
    "crime": {"parent": "lifestyle", "name": "Crime & True Crime"},
    "culture_trends": {"parent": "lifestyle", "name": "Culture & Trends"},
    "social_issues": {"parent": "lifestyle", "name": "Social Issues"},
    "personal_finance": {"parent": "lifestyle", "name": "Personal Finance"},
    "diy_crafts": {"parent": "lifestyle", "name": "DIY & Crafts"},
    "automotive": {"parent": "lifestyle", "name": "Automotive & Cars"},
    "beauty": {"parent": "lifestyle", "name": "Beauty & Cosmetics"}
}

# Generate config.py CATEGORIES dictionary text
config_opts = "CATEGORIES = {\n"
for k, v in CATEGORIES_DEF.items():
    config_opts += f"        '{k}': '{v['name']}',\n"
config_opts += "    }"

# Overwrite config.py
with open(r'e:\\NEWSCAT\\backend\\config.py', 'r', encoding='utf-8') as f:
    config_content = f.read()

new_config_content = re.sub(
    r'    CATEGORIES = \{[\s\S]*?\n    \}',
    "    " + config_opts,
    config_content
)

with open(r'e:\\NEWSCAT\\backend\\config.py', 'w', encoding='utf-8') as f:
    f.write(new_config_content)
print("Updated config.py")

# Generate lightning_classifier.py CategoryKnowledgeGraph
graph_opts = "CATEGORIES = {\n"
for k, v in CATEGORIES_DEF.items():
    parent_str = f"'{v['parent']}'" if v['parent'] else "None"
    
    # Generic embeddings generator based on name logic
    parts = [p for p in v['name'].lower().replace('&', '').replace('(', '').replace(')', '').split() if len(p) > 2]
    core_kw = [k.replace('_', ' ')] + parts + [v['name'].lower()]
    
    graph_opts += f"""        '{k}': {{
            'parent': {parent_str}, 'neural_weight': 0.9,
            'embeddings': {{
                'core': {json.dumps(list(set(core_kw)))},
                'related': {json.dumps([p + ' news' for p in parts])}
            }},
            'semantic_context': {json.dumps(parts)},
            'confidence_multipliers': {{'high': 3.5, 'medium': 2.0, 'low': 1.0}}
        }},\n"""
graph_opts += "    }"

with open(r'e:\\NEWSCAT\\backend\\models\\lightning_classifier.py', 'r', encoding='utf-8') as f:
    lightning_content = f.read()

new_lightning_content = re.sub(
    r'    CATEGORIES = \{[\s\S]*?\n    \}',
    "    " + graph_opts,
    lightning_content
)

with open(r'e:\\NEWSCAT\\backend\\models\\lightning_classifier.py', 'w', encoding='utf-8') as f:
    f.write(new_lightning_content)
print("Updated lightning_classifier.py")
