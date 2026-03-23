"""
NEWSCAT Taxonomy Keywords — Single Source of Truth
Maps 150+ category slugs to weighted keyword lists used by all classifiers and processors.
"""

TAXONOMY_KEYWORDS = {
    # =========================================================================
    # 1. TECHNOLOGY (18 categories)
    # =========================================================================
    "technology": [
        "technology", "tech", "digital", "innovation", "gadget", "silicon valley",
        "startup tech", "tech industry", "disruption", "tech news", "computing",
        "information technology", "IT infrastructure", "tech giant", "tech company",
    ],
    "artificial_intelligence": [
        "artificial intelligence", "AI", "machine learning", "deep learning",
        "neural network", "GPT", "LLM", "generative AI", "chatbot", "NLP",
        "natural language processing", "computer vision", "reinforcement learning",
        "transformer model", "AI ethics", "AGI", "autonomous", "AI model",
        "transformer architecture", "parameter count", "RAG", "fine-tuning", 
        "inference", "compute cluster", "tokenization", "attention mechanism",
    ],
    "cybersecurity": [
        "cybersecurity", "hacking", "data breach", "ransomware", "malware",
        "phishing", "firewall", "encryption", "vulnerability", "cyber attack",
        "zero-day", "security patch", "DDoS", "identity theft", "cyber threat",
        "infosec", "penetration testing", "SOC", "SIEM", "threat intelligence",
    ],
    "software_development": [
        "software development", "programming", "coding", "developer", "open source",
        "API", "SDK", "DevOps", "agile", "git", "github", "CI/CD", "microservices",
        "full stack", "frontend", "backend", "software engineer", "code review",
        "debugging", "framework", "library", "runtime", "IDE",
    ],
    "hardware_devices": [
        "hardware", "processor", "CPU", "GPU", "motherboard", "RAM", "SSD",
        "laptop", "desktop", "tablet", "wearable", "smartwatch", "peripheral",
        "display", "monitor", "keyboard", "mouse", "chip", "circuit board",
    ],
    "cloud_computing": [
        "cloud computing", "AWS", "Azure", "Google Cloud", "SaaS", "PaaS", "IaaS",
        "serverless", "container", "kubernetes", "docker", "cloud migration",
        "multi-cloud", "hybrid cloud", "edge computing", "cloud storage",
    ],
    "telecommunications": [
        "telecommunications", "5G", "6G", "LTE", "broadband", "fiber optic",
        "mobile network", "telecom", "satellite communication", "spectrum",
        "wireless", "ISP", "internet provider", "cell tower", "bandwidth",
    ],
    "robotics": [
        "robotics", "robot", "automation", "autonomous robot", "industrial robot",
        "drone", "unmanned", "actuator", "sensor fusion", "robotic process",
        "cobots", "humanoid robot", "surgical robot", "warehouse robot",
    ],
    "internet_of_things": [
        "internet of things", "IoT", "smart home", "connected device", "smart sensor",
        "wearable tech", "smart city", "industrial IoT", "IIoT", "embedded system",
        "smart appliance", "home automation", "smart thermostat", "smart speaker",
    ],
    "virtual_reality": [
        "virtual reality", "VR", "augmented reality", "AR", "mixed reality", "MR",
        "metaverse", "XR", "headset", "immersive", "spatial computing",
        "hologram", "haptic", "3D environment", "VR gaming", "AR glasses",
    ],
    "data_science": [
        "data science", "big data", "data analytics", "data mining", "data pipeline",
        "ETL", "data warehouse", "business intelligence", "BI", "data visualization",
        "statistical analysis", "predictive analytics", "data lake", "Hadoop", "Spark",
    ],
    "blockchain_tech": [
        "blockchain", "distributed ledger", "smart contract", "DeFi", "Web3",
        "decentralized", "consensus mechanism", "Ethereum", "Solana", "NFT",
        "tokenization", "DAO", "proof of work", "proof of stake", "dApp",
    ],
    "gaming_tech": [
        "gaming technology", "game engine", "Unreal Engine", "Unity", "ray tracing",
        "game development", "graphics card", "gaming console", "PS5", "Xbox",
        "Nintendo", "cloud gaming", "game streaming", "FPS", "rendering",
    ],
    "social_media_tech": [
        "social media technology", "algorithm", "content moderation", "feed ranking",
        "social network", "platform governance", "recommendation engine",
        "user engagement", "social graph", "viral content", "influencer tech",
    ],
    "consumer_electronics": [
        "consumer electronics", "smartphone", "iPhone", "Samsung Galaxy", "Pixel",
        "smart TV", "earbuds", "AirPods", "e-reader", "Kindle", "camera",
        "home theater", "sound bar", "portable speaker", "electronic gadget",
    ],
    "semiconductors": [
        "semiconductor", "chip manufacturing", "TSMC", "Intel", "AMD", "NVIDIA",
        "wafer", "nanometer", "fabrication", "foundry", "chip shortage",
        "Moore's law", "transistor", "silicon", "photolithography", "EUV",
    ],
    "nanotechnology": [
        "nanotechnology", "nanomaterial", "nanoparticle", "nanoscale", "nanorobot",
        "molecular engineering", "quantum dot", "carbon nanotube", "nano fabrication",
        "nanofiber", "nanocomposite", "nano medicine", "nano sensor",
    ],
    "biotechnology": [
        "biotechnology", "biotech", "gene editing", "CRISPR", "bioengineering",
        "synthetic biology", "bioinformatics", "genomics", "proteomics",
        "biomolecular", "cell therapy", "gene therapy", "biomanufacturing",
    ],

    # =========================================================================
    # 2. BUSINESS & FINANCE (18 categories)
    # =========================================================================
    "business": [
        "business", "corporation", "company", "enterprise", "CEO", "executive",
        "revenue", "profit", "quarterly earnings", "merger", "acquisition",
        "B2B", "B2C", "shareholder", "stakeholder", "corporate strategy",
    ],
    "finance": [
        "finance", "stock market", "Wall Street", "stock exchange", "equity",
        "bond", "IPO", "mutual fund", "ETF", "hedge fund", "portfolio",
        "capital market", "financial analysis", "bull market", "bear market",
        "trading", "securities", "dividend", "S&P 500", "Nasdaq", "Dow Jones",
    ],
    "startups": [
        "startup", "venture capital", "VC", "funding round", "Series A",
        "Series B", "unicorn", "accelerator", "incubator", "seed funding",
        "pitch deck", "bootstrapping", "pre-seed", "valuation", "pivot",
    ],
    "economy": [
        "economy", "GDP", "inflation", "recession", "economic growth",
        "unemployment", "fiscal policy", "monetary policy", "central bank",
        "Federal Reserve", "interest rate", "consumer spending", "CPI",
        "economic indicator", "deficit", "surplus", "stimulus",
        "macroeconomic", "quantitative easing", "fiscal stimulus", "bond yield", "hyperinflation",
    ],
    "real_estate": [
        "real estate", "property", "housing market", "mortgage", "rent",
        "commercial property", "residential", "real estate investment",
        "REIT", "home price", "housing bubble", "foreclosure", "landlord",
        "apartment", "condo", "home sale", "property value",
    ],
    "marketing": [
        "marketing", "advertising", "branding", "digital marketing", "SEO",
        "SEM", "social media marketing", "content marketing", "influencer marketing",
        "campaign", "ad spend", "conversion rate", "click-through rate", "PPC",
        "brand awareness", "market research", "target audience",
    ],
    "entrepreneurship": [
        "entrepreneurship", "entrepreneur", "founder", "co-founder", "business plan",
        "small business", "side hustle", "solopreneur", "business model",
        "lean startup", "MVP", "market opportunity", "scaling", "growth hacking",
    ],
    "ecommerce": [
        "ecommerce", "e-commerce", "online shopping", "Amazon", "Shopify",
        "dropshipping", "marketplace", "digital commerce", "checkout",
        "cart abandonment", "fulfillment", "last mile delivery", "D2C",
    ],
    "cryptocurrency": [
        "cryptocurrency", "crypto", "Bitcoin", "Ethereum", "altcoin", "stablecoin",
        "crypto exchange", "Binance", "Coinbase", "crypto wallet", "mining",
        "token", "ICO", "crypto regulation", "digital currency", "CBDC",
    ],
    "banking": [
        "banking", "bank", "lending", "loan", "credit", "deposit", "fintech",
        "neobank", "digital banking", "mortgage rate", "savings account",
        "checking account", "credit score", "APR", "FDIC", "banking regulation",
    ],
    "corporate_governance": [
        "corporate governance", "board of directors", "shareholder activism",
        "proxy vote", "ESG", "compliance", "audit", "fiduciary duty",
        "executive compensation", "whistleblower", "transparency", "accountability",
    ],
    "human_resources": [
        "human resources", "HR", "recruitment", "hiring", "talent acquisition",
        "employee engagement", "workplace culture", "remote work", "hybrid work",
        "DEI", "diversity", "inclusion", "employee benefits", "onboarding",
        "turnover", "retention", "payroll", "workforce", "layoffs",
    ],
    "leadership": [
        "leadership", "management", "executive leadership", "team management",
        "organizational leadership", "decision making", "strategic leadership",
        "servant leadership", "management style", "coaching", "mentorship",
    ],
    "supply_chain": [
        "supply chain", "logistics", "freight", "shipping", "warehousing",
        "inventory management", "procurement", "distribution", "supply chain disruption",
        "just-in-time", "port congestion", "cargo", "fleet management",
    ],
    "insurance": [
        "insurance", "insurance policy", "premium", "claim", "underwriting",
        "life insurance", "health insurance", "auto insurance", "property insurance",
        "reinsurance", "actuarial", "deductible", "coverage", "insurer",
    ],
    "accounting": [
        "accounting", "tax", "taxation", "audit", "bookkeeping", "CPA",
        "financial statement", "balance sheet", "income statement", "cash flow",
        "tax return", "tax reform", "GAAP", "IFRS", "revenue recognition",
    ],
    "investments": [
        "investment", "wealth management", "asset management", "private equity",
        "angel investor", "retirement fund", "401k", "IRA", "pension",
        "return on investment", "ROI", "portfolio diversification", "asset allocation",
    ],
    "international_trade": [
        "international trade", "trade war", "tariff", "import", "export",
        "trade deficit", "trade agreement", "WTO", "sanctions", "embargo",
        "free trade", "protectionism", "customs", "trade policy", "globalization",
    ],

    # =========================================================================
    # 3. HEALTH & WELLNESS (16 categories)
    # =========================================================================
    "health": [
        "health", "wellness", "healthcare", "medical", "patient", "hospital",
        "clinic", "doctor", "diagnosis", "treatment", "disease", "therapy",
        "health care system", "health outcome", "preventive care",
    ],
    "medicine": [
        "medicine", "clinical trial", "FDA", "drug approval", "medical device",
        "surgical procedure", "oncology", "cardiology", "neurology", "pathology",
        "radiology", "emergency medicine", "intensive care", "ICU", "medical research",
    ],
    "mental_health": [
        "mental health", "anxiety", "depression", "PTSD", "therapy", "counseling",
        "psychiatry", "psychology", "mindfulness", "stress management",
        "mental illness", "bipolar", "schizophrenia", "emotional wellbeing",
        "suicide prevention", "mental wellness", "behavioral health",
    ],
    "fitness": [
        "fitness", "exercise", "workout", "gym", "strength training",
        "cardio", "running", "marathon", "yoga", "pilates", "CrossFit",
        "personal trainer", "bodybuilding", "HIIT", "physical activity",
    ],
    "nutrition": [
        "nutrition", "diet", "calorie", "protein", "vitamin", "supplement",
        "healthy eating", "meal plan", "keto", "vegan diet", "intermittent fasting",
        "macronutrient", "micronutrient", "superfood", "nutritionist",
    ],
    "public_health": [
        "public health", "pandemic", "epidemic", "CDC", "WHO", "vaccination",
        "vaccine", "immunization", "outbreak", "quarantine", "contact tracing",
        "herd immunity", "disease prevention", "health policy", "sanitation",
    ],
    "healthcare_policy": [
        "healthcare policy", "health reform", "Medicare", "Medicaid", "ACA",
        "Obamacare", "universal healthcare", "health insurance mandate",
        "healthcare spending", "hospital regulation", "drug pricing",
    ],
    "alternative_medicine": [
        "alternative medicine", "herbal medicine", "acupuncture", "homeopathy",
        "naturopathy", "chiropractic", "Ayurveda", "traditional medicine",
        "holistic health", "aromatherapy", "reflexology", "meditation healing",
    ],
    "pharmaceuticals": [
        "pharmaceutical", "drug development", "pharma", "clinical trial",
        "FDA approval", "generic drug", "biologic", "prescription drug",
        "drug manufacturer", "Pfizer", "Moderna", "Johnson & Johnson",
        "drug safety", "pharmacology", "dosage", "side effects",
    ],
    "pediatrics": [
        "pediatrics", "child health", "infant", "newborn", "childhood disease",
        "vaccination schedule", "developmental milestone", "neonatal",
        "pediatric surgery", "children's hospital", "NICU", "childhood obesity",
    ],
    "aging_geriatrics": [
        "aging", "geriatrics", "elderly", "senior health", "dementia",
        "Alzheimer's", "assisted living", "nursing home", "longevity",
        "age-related disease", "retirement health", "elder care",
    ],
    "womens_health": [
        "women's health", "pregnancy", "maternal health", "reproductive health",
        "gynecology", "obstetrics", "breast cancer", "menopause", "fertility",
        "prenatal care", "postpartum", "cervical cancer", "PCOS",
    ],
    "mens_health": [
        "men's health", "prostate", "testosterone", "erectile dysfunction",
        "men's fitness", "male pattern baldness", "prostate cancer",
        "men's wellness", "andrology", "male fertility",
    ],
    "dentistry": [
        "dentistry", "dental", "oral health", "orthodontics", "braces",
        "dental implant", "root canal", "cavity", "gum disease", "periodontal",
        "teeth whitening", "dental hygiene", "wisdom teeth", "dentist",
    ],
    "veterinary": [
        "veterinary", "animal health", "vet", "veterinarian", "pet health",
        "animal surgery", "veterinary medicine", "animal clinic",
        "livestock health", "animal disease", "pet vaccination",
    ],
    "dermatology": [
        "dermatology", "skin care", "acne", "eczema", "psoriasis",
        "skin cancer", "melanoma", "dermatologist", "rash", "skin condition",
        "cosmetic dermatology", "botox", "collagen", "skin treatment",
    ],

    # =========================================================================
    # 4. SCIENCE & ENVIRONMENT (18 categories)
    # =========================================================================
    "science": [
        "science", "scientific discovery", "research", "laboratory", "experiment",
        "peer review", "scientific journal", "Nature", "Science magazine",
        "hypothesis", "theory", "observation", "STEM", "researcher",
    ],
    "space": [
        "space", "NASA", "SpaceX", "astronaut", "rocket", "satellite",
        "space station", "ISS", "Mars mission", "moon landing", "exoplanet",
        "telescope", "Hubble", "James Webb", "asteroid", "comet", "orbit",
        "space exploration", "cosmos", "galaxy", "black hole", "supernova",
    ],
    "climate_change": [
        "climate change", "global warming", "greenhouse gas", "carbon emission",
        "carbon footprint", "climate crisis", "Paris Agreement", "net zero",
        "sea level rise", "ice cap melting", "climate policy", "carbon tax",
        "climate adaptation", "CO2", "methane emission", "deforestation",
    ],
    "environment": [
        "environment", "ecology", "biodiversity", "conservation", "wildlife",
        "pollution", "air quality", "water quality", "ecosystem", "deforestation",
        "endangered species", "habitat", "sustainability", "reforestation",
        "environmental protection", "EPA", "waste management", "recycling",
    ],
    "physics": [
        "physics", "quantum physics", "relativity", "particle physics",
        "CERN", "Higgs boson", "dark matter", "dark energy", "string theory",
        "quantum mechanics", "thermodynamics", "electromagnetic", "photon",
        "neutron", "proton", "nuclear physics", "plasma physics",
    ],
    "biology": [
        "biology", "cell biology", "microbiology", "molecular biology",
        "organism", "evolution", "natural selection", "ecosystem biology",
        "marine biology", "developmental biology", "immunology", "virology",
        "bacteria", "virus", "protein", "cell division", "mitosis",
    ],
    "chemistry": [
        "chemistry", "chemical reaction", "molecule", "compound", "element",
        "periodic table", "organic chemistry", "inorganic chemistry",
        "biochemistry", "polymer", "catalyst", "chemical bond", "isotope",
        "pH", "acid", "base", "solvent", "solution",
    ],
    "genetics": [
        "genetics", "DNA", "RNA", "genome", "gene", "mutation", "heredity",
        "chromosomes", "genetic engineering", "epigenetics", "gene sequencing",
        "Human Genome Project", "allele", "genetic disorder", "gene expression",
    ],
    "archaeology": [
        "archaeology", "anthropology", "excavation", "artifact", "ancient civilization",
        "fossil record", "cultural heritage", "prehistoric", "Bronze Age",
        "Iron Age", "archaeological site", "tomb", "ruins", "antiquity",
    ],
    "oceanography": [
        "oceanography", "ocean", "marine science", "deep sea", "coral reef",
        "ocean current", "marine ecosystem", "submarine", "ocean floor",
        "tidal", "tsunami", "ocean temperature", "marine pollution",
    ],
    "geology": [
        "geology", "earth science", "tectonic plate", "earthquake", "volcano",
        "mineral", "rock formation", "sediment", "fossil", "geologist",
        "geological survey", "fault line", "magma", "erosion", "seismology",
    ],
    "paleontology": [
        "paleontology", "dinosaur", "fossil", "extinction", "Jurassic",
        "Cretaceous", "prehistoric animal", "amber", "paleontologist",
        "fossil discovery", "mass extinction", "T-Rex", "trilobite",
    ],
    "meteorology": [
        "meteorology", "weather", "forecast", "hurricane", "tornado",
        "thunderstorm", "blizzard", "heat wave", "cold front", "warm front",
        "barometric pressure", "precipitation", "climate model", "El Nino",
        "La Nina", "atmospheric", "wind speed", "weather pattern",
    ],
    "zoology": [
        "zoology", "animal", "wildlife", "mammal", "reptile", "bird",
        "insect", "marine animal", "primate", "predator", "prey",
        "animal behavior", "migration", "endangered animal", "species",
    ],
    "botany": [
        "botany", "plant", "flower", "tree", "photosynthesis", "seed",
        "pollination", "botanical garden", "plant biology", "crop science",
        "herbarium", "plant species", "root system", "leaf", "vegetation",
    ],
    "energy": [
        "energy", "power", "electricity", "power grid", "energy policy",
        "oil", "gas", "natural gas", "petroleum", "OPEC", "energy market",
        "power plant", "nuclear energy", "energy storage", "battery",
        "fuel cell", "energy crisis", "energy transition",
    ],
    "renewable_energy": [
        "renewable energy", "solar power", "wind power", "hydroelectric",
        "geothermal", "solar panel", "wind turbine", "clean energy",
        "green energy", "sustainable energy", "biofuel", "biomass",
        "tidal energy", "energy efficiency", "net zero energy",
    ],
    "materials_science": [
        "materials science", "composite material", "alloy", "ceramic",
        "polymer science", "superconductor", "graphene", "metamaterial",
        "material engineering", "corrosion", "tensile strength", "elasticity",
    ],

    # =========================================================================
    # 5. POLITICS & GOVERNMENT (16 categories)
    # =========================================================================
    "politics": [
        "politics", "government", "political", "legislation", "congress",
        "parliament", "senator", "representative", "policy", "political party",
        "Democrat", "Republican", "liberal", "conservative", "bipartisan",
        "bill", "law", "regulation", "executive order", "White House",
    ],
    "elections": [
        "election", "voting", "ballot", "campaign", "candidate", "primary",
        "general election", "midterm", "electoral college", "poll",
        "swing state", "debate", "running mate", "vote count", "recount",
        "voter turnout", "political advertising", "PAC", "Super PAC",
    ],
    "geopolitics": [
        "geopolitics", "geopolitical", "territorial dispute", "sphere of influence",
        "power struggle", "strategic alliance", "Cold War", "proxy war",
        "regional conflict", "global power", "hegemony", "superpower",
        "sovereignty", "sanctions regime", "multilateralism", "hegemonic", "non-proliferation",
    ],
    "international_relations": [
        "international relations", "foreign policy", "diplomacy", "treaty",
        "bilateral", "multilateral", "United Nations", "NATO", "G7", "G20",
        "summit", "ambassador", "embassy", "foreign affairs", "alliance",
    ],
    "public_policy": [
        "public policy", "policy reform", "regulation", "deregulation",
        "government spending", "budget", "welfare", "social security",
        "policy analysis", "think tank", "lobbying", "government program",
    ],
    "law_justice": [
        "law", "justice", "court", "judge", "trial", "verdict", "lawsuit",
        "Supreme Court", "attorney", "lawyer", "prosecutor", "defendant",
        "legislation", "constitutional", "civil law", "criminal law",
        "legal precedent", "jury", "appeal", "injunction", "ruling",
    ],
    "war_conflict": [
        "war", "conflict", "military", "combat", "battlefield", "troops",
        "invasion", "missile", "airstrike", "ceasefire", "peace talks",
        "armed forces", "artillery", "insurgency", "guerrilla", "siege",
        "casualties", "refugee", "displacement", "defense", "weapon",
    ],
    "human_rights": [
        "human rights", "civil liberties", "freedom of speech", "censorship",
        "persecution", "asylum", "refugee rights", "discrimination",
        "torture", "Amnesty International", "Human Rights Watch",
        "universal declaration", "political prisoner", "genocide",
    ],
    "immigration": [
        "immigration", "immigrant", "migrant", "border", "visa", "deportation",
        "asylum seeker", "green card", "citizenship", "illegal immigration",
        "border patrol", "immigration reform", "DACA", "immigration policy",
    ],
    "civil_rights": [
        "civil rights", "racial equality", "discrimination", "segregation",
        "affirmative action", "voting rights", "equal opportunity",
        "civil rights movement", "racial justice", "police reform",
        "systemic racism", "anti-discrimination", "equality",
    ],
    "diplomacy": [
        "diplomacy", "diplomatic", "negotiation", "peace agreement",
        "diplomatic relations", "consulate", "diplomatic immunity",
        "foreign minister", "state visit", "diplomatic crisis",
    ],
    "national_security": [
        "national security", "intelligence", "CIA", "FBI", "NSA", "espionage",
        "surveillance", "counterterrorism", "homeland security", "classified",
        "security clearance", "defense budget", "military intelligence",
    ],
    "political_scandals": [
        "political scandal", "corruption", "impeachment", "investigation",
        "cover-up", "abuse of power", "ethics violation", "bribery",
        "political controversy", "resignation", "indictment", "subpoena",
    ],
    "local_government": [
        "local government", "city council", "mayor", "municipal", "county",
        "township", "zoning", "city ordinance", "public hearing",
        "local election", "school board", "city budget", "town hall",
    ],
    "global_organizations": [
        "global organization", "United Nations", "UN", "World Bank", "IMF",
        "WHO", "UNESCO", "UNICEF", "WTO", "ICC", "International Court",
        "peacekeeping", "international aid", "humanitarian", "NGO",
    ],
    "activism": [
        "activism", "protest", "demonstration", "rally", "march", "sit-in",
        "boycott", "petition", "social movement", "grassroots", "strike",
        "civil disobedience", "activist", "advocacy", "campaign",
    ],

    # =========================================================================
    # 6. ENTERTAINMENT & ARTS (16 categories)
    # =========================================================================
    "entertainment": [
        "entertainment", "showbiz", "show business", "media", "celebrity",
        "Hollywood", "Bollywood", "premiere", "blockbuster", "box office",
        "entertainment industry", "talent", "agent", "publicity",
    ],
    "film_tv": [
        "film", "movie", "television", "TV show", "series", "documentary",
        "director", "actor", "actress", "screenplay", "cinematography",
        "Netflix", "HBO", "Disney", "Warner Bros", "studio", "box office",
        "trailer", "sequel", "prequel", "Oscar", "Emmy",
    ],
    "music": [
        "music", "song", "album", "concert", "musician", "singer", "band",
        "Grammy", "record label", "streaming music", "Spotify", "iTunes",
        "music video", "tour", "genre", "hip hop", "rock", "pop music",
        "classical music", "jazz", "R&B", "EDM", "country music",
    ],
    "celebrity": [
        "celebrity", "famous", "star", "A-list", "red carpet", "paparazzi",
        "tabloid", "gossip", "celebrity news", "influencer", "personal life",
        "scandal", "fame", "public figure", "celebrity couple",
    ],
    "pop_culture": [
        "pop culture", "viral", "meme", "trend", "trending", "fandom",
        "cosplay", "fan fiction", "popular culture", "cultural phenomenon",
        "pop icon", "mainstream", "cultural moment", "zeitgeist",
    ],
    "video_games": [
        "video game", "gaming", "gamer", "game release", "PlayStation",
        "Xbox", "Nintendo Switch", "Steam", "PC gaming", "mobile gaming",
        "esports tournament", "game review", "RPG", "FPS", "battle royale",
        "MMO", "MMORPG", "game developer", "indie game",
    ],
    "books_literature": [
        "book", "literature", "novel", "author", "publisher", "bestseller",
        "fiction", "non-fiction", "poetry", "memoir", "biography",
        "book review", "literary award", "Pulitzer", "Booker Prize",
        "reading", "library", "e-book", "audiobook",
    ],
    "performing_arts": [
        "performing arts", "theater", "theatre", "Broadway", "musical",
        "opera", "dance", "ballet", "symphony", "orchestra", "playwright",
        "stage performance", "live performance", "drama", "comedy show",
    ],
    "fine_arts": [
        "fine arts", "art", "painting", "sculpture", "gallery", "museum",
        "exhibition", "art auction", "contemporary art", "modern art",
        "abstract art", "Sotheby's", "Christie's", "art collection",
    ],
    "photography": [
        "photography", "photographer", "photo", "portrait", "landscape photography",
        "photojournalism", "camera", "DSLR", "mirrorless", "lens",
        "exposure", "composition", "photo editing", "Lightroom",
    ],
    "fashion": [
        "fashion", "style", "designer", "fashion week", "runway", "haute couture",
        "clothing", "apparel", "Vogue", "fashion brand", "Gucci", "Prada",
        "Louis Vuitton", "fashion trend", "wardrobe", "dress code", "textile",
    ],
    "anime_manga": [
        "anime", "manga", "otaku", "Japanese animation", "shonen", "shojo",
        "Studio Ghibli", "Naruto", "One Piece", "Dragon Ball", "Attack on Titan",
        "cosplay", "anime convention", "light novel", "isekai",
    ],
    "podcasts": [
        "podcast", "radio", "audio show", "podcast host", "episode",
        "podcast network", "radio broadcast", "talk show", "interview",
        "podcast platform", "Apple Podcasts", "Spotify podcast", "FM", "AM",
    ],
    "awards_shows": [
        "awards show", "Oscar", "Academy Awards", "Grammy Awards", "Emmy Awards",
        "Golden Globe", "BAFTA", "Tony Awards", "MTV Awards", "nomination",
        "red carpet", "acceptance speech", "best picture", "award ceremony",
    ],
    "streaming": [
        "streaming", "Netflix", "Disney+", "Hulu", "Amazon Prime Video",
        "HBO Max", "Apple TV+", "Paramount+", "Peacock", "streaming service",
        "binge watching", "original series", "streaming platform", "subscriber",
    ],
    "comics": [
        "comic", "graphic novel", "comic book", "Marvel", "DC Comics",
        "superhero", "manga artist", "comic strip", "comic convention",
        "Comic-Con", "comic publisher", "illustrated", "webcomic",
    ],

    # =========================================================================
    # 7. SPORTS (15 categories)
    # =========================================================================
    "sports": [
        "sports", "athlete", "championship", "league", "tournament", "match",
        "game", "score", "playoff", "season", "team", "coach", "stadium",
        "trophy", "medal", "record", "sports news", "competition",
    ],
    "football_soccer": [
        "football", "soccer", "FIFA", "World Cup", "Premier League", "La Liga",
        "Champions League", "Serie A", "Bundesliga", "goal", "penalty kick",
        "midfielder", "striker", "goalkeeper", "transfer window", "Messi",
        "Ronaldo", "UEFA", "MLS",
    ],
    "american_football": [
        "american football", "NFL", "Super Bowl", "quarterback", "touchdown",
        "field goal", "running back", "wide receiver", "interception", "fumble",
        "draft pick", "NFC", "AFC", "franchise", "Tom Brady", "Patrick Mahomes",
    ],
    "basketball": [
        "basketball", "NBA", "WNBA", "slam dunk", "three-pointer", "rebound",
        "point guard", "center", "power forward", "NBA Finals", "All-Star",
        "LeBron James", "Stephen Curry", "free throw", "basketball court",
    ],
    "baseball": [
        "baseball", "MLB", "World Series", "home run", "pitcher", "batter",
        "inning", "strikeout", "grand slam", "batting average", "bullpen",
        "outfield", "shortstop", "catcher", "base hit", "RBI",
    ],
    "tennis": [
        "tennis", "Grand Slam", "Wimbledon", "US Open", "French Open",
        "Australian Open", "ATP", "WTA", "serve", "volley", "forehand",
        "backhand", "match point", "set", "deuce", "ace",
    ],
    "golf": [
        "golf", "PGA", "birdie", "eagle", "bogey", "par", "hole-in-one",
        "Masters", "US Open golf", "British Open", "Ryder Cup",
        "golf course", "caddy", "fairway", "putt", "driving range",
    ],
    "motorsports": [
        "motorsport", "Formula 1", "F1", "NASCAR", "IndyCar", "MotoGP",
        "race car", "pit stop", "lap time", "pole position", "Grand Prix",
        "circuit", "drag racing", "rally", "Le Mans", "Ferrari", "Mercedes F1",
    ],
    "combat_sports": [
        "combat sports", "boxing", "MMA", "UFC", "wrestling", "martial arts",
        "judo", "karate", "taekwondo", "knockout", "submission",
        "weight class", "title fight", "championship belt", "bout",
    ],
    "athletics_olympics": [
        "Olympics", "Olympic Games", "athletics", "track and field", "sprint",
        "marathon", "swimming", "gymnastics", "medal count", "gold medal",
        "silver medal", "bronze medal", "Olympic record", "world record",
        "decathlon", "IOC", "Winter Olympics", "Summer Olympics",
    ],
    "hockey": [
        "hockey", "NHL", "ice hockey", "puck", "goal", "assist", "power play",
        "penalty box", "Stanley Cup", "defenseman", "goaltender", "wrist shot",
        "slapshot", "faceoff", "hat trick", "rink",
    ],
    "cricket": [
        "cricket", "IPL", "Test match", "ODI", "T20", "batting", "bowling",
        "wicket", "century", "six", "four", "run out", "LBW", "Ashes",
        "Cricket World Cup", "batsman", "bowler", "crease", "pitch",
        "DRS", "powerplay", "duckworth-lewis", "maiden over", "hat-trick", "super over",
    ],
    "extreme_sports": [
        "extreme sports", "surfing", "skateboarding", "snowboarding", "BMX",
        "rock climbing", "bungee jumping", "skydiving", "parkour", "base jumping",
        "X Games", "adventure sports", "motocross", "wakeboarding",
    ],
    "cycling": [
        "cycling", "Tour de France", "Giro d'Italia", "Vuelta", "road cycling",
        "mountain biking", "velodrome", "sprint cycling", "peloton",
        "time trial", "stage race", "UCI", "bicycle", "cyclist",
    ],
    "rugby": [
        "rugby", "Rugby World Cup", "rugby union", "rugby league", "try",
        "scrum", "lineout", "conversion", "penalty kick rugby", "fullback",
        "fly-half", "ruck", "maul", "Six Nations", "Super Rugby",
    ],
    "esports": [
        "esports", "competitive gaming", "League of Legends", "Dota 2",
        "Counter-Strike", "Fortnite", "Overwatch", "Valorant", "esports team",
        "esports tournament", "pro gamer", "twitch", "game streaming",
    ],

    # =========================================================================
    # 8. LIFESTYLE & SOCIETY (16 categories)
    # =========================================================================
    "lifestyle": [
        "lifestyle", "quality of life", "work-life balance", "self-improvement",
        "personal development", "habits", "routine", "minimalism", "wellness",
        "social life", "community", "urban living", "suburban", "rural life",
    ],
    "travel": [
        "travel", "tourism", "vacation", "destination", "hotel", "resort",
        "airline", "flight", "airport", "travel guide", "backpacking",
        "cruise", "road trip", "travel tips", "tourist attraction",
        "passport", "visa travel", "hostel", "Airbnb",
    ],
    "food_dining": [
        "food", "dining", "restaurant", "recipe", "cooking", "chef",
        "cuisine", "Michelin star", "food review", "culinary", "baking",
        "gourmet", "fast food", "brunch", "dinner", "food truck",
        "food festival", "wine", "cocktail", "beer", "organic food",
    ],
    "education": [
        "education", "school", "university", "college", "student", "teacher",
        "professor", "curriculum", "degree", "scholarship", "online learning",
        "e-learning", "MOOC", "academic", "campus", "enrollment",
        "graduate", "undergraduate", "PhD", "research university",
    ],
    "parenting": [
        "parenting", "family", "child", "parent", "toddler", "baby",
        "motherhood", "fatherhood", "childcare", "homeschooling",
        "family planning", "adoption", "foster care", "parental leave",
    ],
    "relationships": [
        "relationship", "dating", "marriage", "divorce", "love",
        "online dating", "Tinder", "Bumble", "partnership", "couple",
        "wedding", "engagement", "breakup", "long-distance relationship",
    ],
    "home_garden": [
        "home", "garden", "interior design", "home improvement", "renovation",
        "landscaping", "furniture", "decor", "DIY home", "real estate home",
        "architecture", "home renovation", "kitchen design", "bathroom remodel",
    ],
    "pets_animals": [
        "pet", "dog", "cat", "pet care", "pet adoption", "animal rescue",
        "pet food", "grooming", "puppy", "kitten", "fish tank", "aquarium",
        "bird pet", "hamster", "rabbit", "pet health", "leash", "collar",
    ],
    "religion_spirituality": [
        "religion", "spirituality", "faith", "church", "mosque", "temple",
        "synagogue", "prayer", "worship", "bible", "quran", "scripture",
        "theology", "pastor", "imam", "rabbi", "religious freedom",
        "Christianity", "Islam", "Hindu", "Buddhism", "Judaism",
    ],
    "crime": [
        "crime", "criminal", "murder", "robbery", "theft", "burglary",
        "assault", "homicide", "arrest", "suspect", "police", "investigation",
        "forensic", "crime scene", "prison", "sentence", "parole",
        "organized crime", "gang", "drug trafficking", "fraud", "scam",
    ],
    "culture_trends": [
        "culture", "cultural trend", "generational", "Gen Z", "millennial",
        "boomer", "cultural shift", "social norm", "tradition", "heritage",
        "cultural identity", "cultural exchange", "subculture",
    ],
    "social_issues": [
        "social issues", "inequality", "poverty", "homelessness",
        "gender equality", "racial inequality", "income gap", "wealth gap",
        "social justice", "welfare reform", "food insecurity", "housing crisis",
        "gun control", "gun violence", "domestic violence", "human trafficking",
    ],
    "personal_finance": [
        "personal finance", "budgeting", "saving", "debt", "credit card",
        "student loan", "mortgage payment", "financial planning", "retirement planning",
        "emergency fund", "net worth", "financial literacy", "compound interest",
    ],
    "diy_crafts": [
        "DIY", "crafts", "handmade", "knitting", "sewing", "woodworking",
        "upcycling", "scrapbooking", "crochet", "pottery", "origami",
        "craft project", "maker", "makerspace", "hobby",
    ],
    "automotive": [
        "automotive", "car", "vehicle", "auto industry", "electric vehicle", "EV",
        "Tesla", "Ford", "Toyota", "BMW", "Mercedes", "Audi", "self-driving car",
        "autonomous vehicle", "car review", "engine", "horsepower", "SUV",
        "truck", "sedan", "hybrid car", "fuel efficiency",
    ],
    "beauty": [
        "beauty", "cosmetics", "makeup", "skincare", "beauty product",
        "lipstick", "foundation", "mascara", "beauty brand", "Sephora",
        "beauty influencer", "cruelty-free", "anti-aging", "moisturizer",
    ],

    # =========================================================================
    # 9. REAL-TIME & BREAKING NEWS (11 categories)
    # =========================================================================
    "breaking_news": [
        "breaking news", "breaking", "urgent", "developing story", "just in",
        "news alert", "flash", "emergency", "breaking report", "happening now",
        "latest update", "alert", "critical update",
    ],
    "real_time_events": [
        "live event", "real-time coverage", "live coverage", "ongoing",
        "minute-by-minute", "live blog", "live stream", "real-time update",
        "live reporting", "on-scene", "unfolding", "developing",
    ],
    "crisis_response": [
        "crisis", "emergency response", "disaster relief", "humanitarian crisis",
        "natural disaster", "earthquake response", "flood relief", "rescue operation",
        "evacuation", "emergency services", "FEMA", "Red Cross", "first responder",
    ],
    "investigative_journalism": [
        "investigation", "investigative journalism", "exposé", "inquiry",
        "probe", "exclusive", "uncovered", "leaked documents", "whistleblower",
        "undercover", "deep dive", "in-depth report", "special report",
    ],
    "weather_alerts": [
        "weather alert", "tornado warning", "flood alert", "storm surge",
        "hurricane watch", "severe weather", "typhoon", "cyclone",
        "winter storm warning", "heat advisory", "weather emergency",
        "flash flood", "weather warning", "tropical storm",
    ],
    "market_movers": [
        "market mover", "stock surge", "market plunge", "rally", "selloff",
        "stocks jump", "stocks fall", "market crash", "trading halt",
        "circuit breaker", "market volatility", "flash crash", "market rally",
    ],
    "press_releases": [
        "press release", "official announcement", "press conference",
        "announces", "unveils", "reveals", "company statement",
        "official statement", "product launch", "news release",
    ],
    "trending_topics": [
        "trending", "trending now", "viral", "going viral", "buzzing",
        "hot topic", "social media trend", "hashtag", "most discussed",
        "popular topic", "internet sensation", "trending story",
    ],
    "sports_live": [
        "sports live", "live score", "in-game", "half-time", "final score",
        "game time", "live match", "live game", "score update",
        "sports update", "injury update", "lineup announcement",
    ],
    "opinion_editorial": [
        "opinion", "editorial", "op-ed", "viewpoint", "commentary",
        "analysis", "perspective", "column", "think piece", "debate",
        "personal opinion", "letter to editor", "columnist",
    ],
    "fact_check": [
        "fact check", "false claim", "verified", "misleading", "true or false",
        "verification", "debunked", "rumor", "misinformation", "disinformation",
        "fake news", "claim review", "truth meter", "fact-checking",
    ],

    # =========================================================================
    # 10. ADDITIONAL SPECIALIZED CATEGORIES (6 categories)
    # =========================================================================
    "disability_accessibility": [
        "disability", "accessibility", "ADA", "wheelchair", "assistive technology",
        "inclusion", "adaptive", "hearing impaired", "visually impaired",
        "neurodiversity", "accommodation", "barrier-free", "accessible design",
    ],
    "quantum_computing": [
        "quantum computing", "qubit", "quantum supremacy", "quantum entanglement",
        "quantum algorithm", "IBM Quantum", "Google Quantum", "quantum processor",
        "quantum cryptography", "quantum gate", "superposition", "decoherence",
    ],
    "space_tourism": [
        "space tourism", "Blue Origin", "Virgin Galactic", "SpaceX Starship",
        "suborbital flight", "space hotel", "commercial spaceflight",
        "space passenger", "zero gravity", "orbital tourism", "lunar tourism",
    ],
    "food_safety": [
        "food safety", "food recall", "contamination", "foodborne illness",
        "FDA inspection", "food regulation", "salmonella", "E. coli",
        "food labeling", "pesticide", "food supply chain", "food quality",
    ],
    "digital_privacy": [
        "digital privacy", "data privacy", "GDPR", "CCPA", "privacy policy",
        "data collection", "tracking", "cookies", "surveillance capitalism",
        "right to be forgotten", "privacy law", "data protection", "PII",
    ],
    "workforce_automation": [
        "workforce automation", "job displacement", "reskilling", "upskilling",
        "future of work", "gig economy", "freelance", "coworking",
        "workplace automation", "AI job impact", "labor market", "remote workforce",
    ],
}

# Total category count for verification
CATEGORY_COUNT = len(TAXONOMY_KEYWORDS)
assert CATEGORY_COUNT >= 150, f"Expected 150+ categories, got {CATEGORY_COUNT}"
