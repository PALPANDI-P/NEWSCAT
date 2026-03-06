/**
 * NEWSCAT v5 Professional - AI News Classification Frontend
 * Clean, modern, and efficient user interface
 *
 * Features:
 * - Request caching with LRU eviction
 * - Request deduplication
 * - Smooth animations
 * - Accessible design
 * - Responsive layout
 * - File upload support (Image, Audio, Video)
 */

'use strict';

// =============================================================================
// CONFIGURATION
// =============================================================================
const CONFIG = {
    API_BASE: '/api',
    MIN_CHARS: 10,
    DEBOUNCE_DELAY: 50,
    CACHE_TTL: 10 * 60 * 1000, // 10 minutes
    MAX_FILE_SIZES: {
        image: 50 * 1024 * 1024,  // 50MB
        audio: 100 * 1024 * 1024,  // 100MB
        video: 200 * 1024 * 1024  // 200MB
    },
    ANIMATION_DURATION: 300
};

// =============================================================================
// CACHE SYSTEM
// =============================================================================
class RequestCache {
    constructor(maxSize = 200) {
        this.cache = new Map();
        this.accessTimes = new Map();
        this.maxSize = maxSize;
        this.hits = 0;
        this.misses = 0;
        this.evictions = 0;
    }

    generateKey(endpoint, data) {
        const str = `${endpoint}:${JSON.stringify(data || {})}`;
        return this.hashString(str);
    }

    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString(36);
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) {
            this.misses++;
            return null;
        }

        // Check TTL
        if (Date.now() - item.timestamp > CONFIG.CACHE_TTL) {
            this.cache.delete(key);
            this.accessTimes.delete(key);
            this.misses++;
            return null;
        }

        // Update access time (LRU)
        this.accessTimes.set(key, Date.now());
        this.hits++;
        return item.data;
    }

    set(key, data) {
        // Evict oldest if at capacity
        if (this.cache.size >= this.maxSize) {
            const oldestKey = this.getOldestKey();
            if (oldestKey) {
                this.cache.delete(oldestKey);
                this.accessTimes.delete(oldestKey);
                this.evictions++;
            }
        }

        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
        this.accessTimes.set(key, Date.now());
    }

    getOldestKey() {
        let oldestKey = null;
        let oldestTime = Infinity;

        for (const [key, time] of this.accessTimes) {
            if (time < oldestTime) {
                oldestTime = time;
                oldestKey = key;
            }
        }
        return oldestKey;
    }

    clear() {
        this.cache.clear();
        this.accessTimes.clear();
    }

    getStats() {
        const total = this.hits + this.misses;
        return {
            size: this.cache.size,
            maxSize: this.maxSize,
            hitRate: total > 0 ? ((this.hits / total) * 100).toFixed(1) + '%' : '0%',
            efficiency: this.cache.size > 0 ? 'active' : 'empty',
            evictions: this.evictions
        };
    }
}

const requestCache = new RequestCache();

// =============================================================================
// STATE MANAGEMENT
// =============================================================================
const state = {
    currentInputType: 'text',
    selectedFiles: { image: null, audio: null, video: null },
    isAnalyzing: false,
    classificationHistory: [],
    modelInfo: null,
    abortController: null,
    pendingRequests: new Map(),
    domElements: {}
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================
const utils = {
    debounce(fn, delay = CONFIG.DEBOUNCE_DELAY) {
        let timeoutId;
        return (...args) => {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => fn.apply(this, args), delay);
        };
    },

    throttle(fn, limit) {
        let inThrottle;
        return (...args) => {
            if (!inThrottle) {
                fn.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    capitalizeFirst(str) {
        if (!str || typeof str !== 'string') return '';
        return str.charAt(0).toUpperCase() + str.slice(1);
    },

    sanitizeText(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Generate random number in range
    random(min, max) {
        return Math.random() * (max - min) + min;
    }
};

// =============================================================================
// CATEGORY STYLES - NEON PALETTE
// =============================================================================
const categoryStyles = {
    artificial_intelligence: {
        icon: 'fa-brain',
        color: '#00f5ff',
        gradient: 'linear-gradient(135deg, #00f5ff, #bf00ff)',
        glow: '0 0 20px rgba(0, 245, 255, 0.5)'
    },
    technology: {
        icon: 'fa-microchip',
        color: '#00f5ff',
        gradient: 'linear-gradient(135deg, #00f5ff, #6366f1)',
        glow: '0 0 20px rgba(0, 245, 255, 0.5)'
    },
    cybersecurity: {
        icon: 'fa-shield-alt',
        color: '#00ff88',
        gradient: 'linear-gradient(135deg, #00ff88, #00f5ff)',
        glow: '0 0 20px rgba(0, 255, 136, 0.5)'
    },
    cryptocurrency: {
        icon: 'fa-bitcoin',
        color: '#ff6b00',
        gradient: 'linear-gradient(135deg, #ff6b00, #ff00ff)',
        glow: '0 0 20px rgba(255, 107, 0, 0.5)'
    },
    business: {
        icon: 'fa-chart-line',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #ff6b00)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    finance: {
        icon: 'fa-coins',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #fcd34d)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    science: {
        icon: 'fa-flask',
        color: '#bf00ff',
        gradient: 'linear-gradient(135deg, #bf00ff, #00f5ff)',
        glow: '0 0 20px rgba(191, 0, 255, 0.5)'
    },
    space: {
        icon: 'fa-rocket',
        color: '#3b82f6',
        gradient: 'linear-gradient(135deg, #3b82f6, #bf00ff)',
        glow: '0 0 20px rgba(59, 130, 246, 0.5)'
    },
    health: {
        icon: 'fa-heartbeat',
        color: '#ff0040',
        gradient: 'linear-gradient(135deg, #ff0040, #ff00ff)',
        glow: '0 0 20px rgba(255, 0, 64, 0.5)'
    },
    politics: {
        icon: 'fa-landmark',
        color: '#bf00ff',
        gradient: 'linear-gradient(135deg, #bf00ff, #6366f1)',
        glow: '0 0 20px rgba(191, 0, 255, 0.5)'
    },
    world: {
        icon: 'fa-globe',
        color: '#60a5fa',
        gradient: 'linear-gradient(135deg, #60a5fa, #00f5ff)',
        glow: '0 0 20px rgba(96, 165, 250, 0.5)'
    },
    sports: {
        icon: 'fa-futbol',
        color: '#4ade80',
        gradient: 'linear-gradient(135deg, #4ade80, #00ff88)',
        glow: '0 0 20px rgba(74, 222, 128, 0.5)'
    },
    entertainment: {
        icon: 'fa-film',
        color: '#ff00ff',
        gradient: 'linear-gradient(135deg, #ff00ff, #ff0040)',
        glow: '0 0 20px rgba(255, 0, 255, 0.5)'
    },
    gaming: {
        icon: 'fa-gamepad',
        color: '#a855f7',
        gradient: 'linear-gradient(135deg, #a855f7, #bf00ff)',
        glow: '0 0 20px rgba(168, 85, 247, 0.5)'
    },
    environment: {
        icon: 'fa-leaf',
        color: '#34d399',
        gradient: 'linear-gradient(135deg, #34d399, #00ff88)',
        glow: '0 0 20px rgba(52, 211, 153, 0.5)'
    },
    education: {
        icon: 'fa-graduation-cap',
        color: '#818cf8',
        gradient: 'linear-gradient(135deg, #818cf8, #bf00ff)',
        glow: '0 0 20px rgba(129, 140, 248, 0.5)'
    },
    travel: {
        icon: 'fa-plane',
        color: '#2dd4bf',
        gradient: 'linear-gradient(135deg, #2dd4bf, #00f5ff)',
        glow: '0 0 20px rgba(45, 212, 191, 0.5)'
    },
    food: {
        icon: 'fa-utensils',
        color: '#fb923c',
        gradient: 'linear-gradient(135deg, #fb923c, #ff6b00)',
        glow: '0 0 20px rgba(251, 146, 60, 0.5)'
    },
    crime: {
        icon: 'fa-user-secret',
        color: '#ff0040',
        gradient: 'linear-gradient(135deg, #ff0040, #ff00ff)',
        glow: '0 0 20px rgba(255, 0, 64, 0.5)'
    },
    // Finance & Economy
    stock_market: {
        icon: 'fa-chart-line',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #f59e0b)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    economy: {
        icon: 'fa-chart-pie',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #fcd34d)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    real_estate: {
        icon: 'fa-building',
        color: '#60a5fa',
        gradient: 'linear-gradient(135deg, #60a5fa, #3b82f6)',
        glow: '0 0 20px rgba(96, 165, 250, 0.5)'
    },
    banking: {
        icon: 'fa-university',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #f59e0b)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    insurance: {
        icon: 'fa-shield-alt',
        color: '#34d399',
        gradient: 'linear-gradient(135deg, #34d399, #10b981)',
        glow: '0 0 20px rgba(52, 211, 153, 0.5)'
    },
    taxation: {
        icon: 'fa-file-invoice-dollar',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #fcd34d)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    retail: {
        icon: 'fa-shopping-cart',
        color: '#f472b6',
        gradient: 'linear-gradient(135deg, #f472b6, #ec4899)',
        glow: '0 0 20px rgba(244, 114, 182, 0.5)'
    },
    ecommerce: {
        icon: 'fa-shopping-bag',
        color: '#f472b6',
        gradient: 'linear-gradient(135deg, #f472b6, #db2777)',
        glow: '0 0 20px rgba(244, 114, 182, 0.5)'
    },
    // Technology Deep Dive
    blockchain: {
        icon: 'fa-link',
        color: '#8b5cf6',
        gradient: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
        glow: '0 0 20px rgba(139, 92, 246, 0.5)'
    },
    iot: {
        icon: 'fa-network-wired',
        color: '#00f5ff',
        gradient: 'linear-gradient(135deg, #00f5ff, #06b6d4)',
        glow: '0 0 20px rgba(0, 245, 255, 0.5)'
    },
    cloud_computing: {
        icon: 'fa-cloud',
        color: '#60a5fa',
        gradient: 'linear-gradient(135deg, #60a5fa, #3b82f6)',
        glow: '0 0 20px rgba(96, 165, 250, 0.5)'
    },
    software_dev: {
        icon: 'fa-code',
        color: '#00f5ff',
        gradient: 'linear-gradient(135deg, #00f5ff, #6366f1)',
        glow: '0 0 20px rgba(0, 245, 255, 0.5)'
    },
    hardware: {
        icon: 'fa-microchip',
        color: '#818cf8',
        gradient: 'linear-gradient(135deg, #818cf8, #6366f1)',
        glow: '0 0 20px rgba(129, 140, 248, 0.5)'
    },
    social_media: {
        icon: 'fa-hashtag',
        color: '#f472b6',
        gradient: 'linear-gradient(135deg, #f472b6, #ec4899)',
        glow: '0 0 20px rgba(244, 114, 182, 0.5)'
    },
    startups: {
        icon: 'fa-rocket',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #f59e0b)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    // Science & Research
    physics: {
        icon: 'fa-atom',
        color: '#bf00ff',
        gradient: 'linear-gradient(135deg, #bf00ff, #8b5cf6)',
        glow: '0 0 20px rgba(191, 0, 255, 0.5)'
    },
    biology: {
        icon: 'fa-dna',
        color: '#34d399',
        gradient: 'linear-gradient(135deg, #34d399, #10b981)',
        glow: '0 0 20px rgba(52, 211, 153, 0.5)'
    },
    chemistry: {
        icon: 'fa-flask',
        color: '#22c55e',
        gradient: 'linear-gradient(135deg, #22c55e, #10b981)',
        glow: '0 0 20px rgba(34, 197, 94, 0.5)'
    },
    medicine: {
        icon: 'fa-heartbeat',
        color: '#ff0040',
        gradient: 'linear-gradient(135deg, #ff0040, #f43f5e)',
        glow: '0 0 20px rgba(255, 0, 64, 0.5)'
    },
    neuroscience: {
        icon: 'fa-brain',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #f59e0b)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    climate_science: {
        icon: 'fa-cloud-sun',
        color: '#34d399',
        gradient: 'linear-gradient(135deg, #34d399, #06b6d4)',
        glow: '0 0 20px rgba(52, 211, 153, 0.5)'
    },
    genetics: {
        icon: 'fa-dna',
        color: '#a855f7',
        gradient: 'linear-gradient(135deg, #a855f7, #8b5cf6)',
        glow: '0 0 20px rgba(168, 85, 247, 0.5)'
    },
    astronomy: {
        icon: 'fa-star',
        color: '#3b82f6',
        gradient: 'linear-gradient(135deg, #3b82f6, #6366f1)',
        glow: '0 0 20px rgba(59, 130, 246, 0.5)'
    },
    oceanography: {
        icon: 'fa-water',
        color: '#06b6d4',
        gradient: 'linear-gradient(135deg, #06b6d4, #0ea5e9)',
        glow: '0 0 20px rgba(6, 182, 212, 0.5)'
    },
    // Society & Culture
    fashion: {
        icon: 'fa-tshirt',
        color: '#f472b6',
        gradient: 'linear-gradient(135deg, #f472b6, #ec4899)',
        glow: '0 0 20px rgba(244, 114, 182, 0.5)'
    },
    art: {
        icon: 'fa-palette',
        color: '#a855f7',
        gradient: 'linear-gradient(135deg, #a855f7, #ec4899)',
        glow: '0 0 20px rgba(168, 85, 247, 0.5)'
    },
    music: {
        icon: 'fa-music',
        color: '#f472b6',
        gradient: 'linear-gradient(135deg, #f472b6, #a855f7)',
        glow: '0 0 20px rgba(244, 114, 182, 0.5)'
    },
    film: {
        icon: 'fa-film',
        color: '#bf00ff',
        gradient: 'linear-gradient(135deg, #bf00ff, #a855f7)',
        glow: '0 0 20px rgba(191, 0, 255, 0.5)'
    },
    literature: {
        icon: 'fa-book',
        color: '#60a5fa',
        gradient: 'linear-gradient(135deg, #60a5fa, #3b82f6)',
        glow: '0 0 20px rgba(96, 165, 250, 0.5)'
    },
    photography: {
        icon: 'fa-camera',
        color: '#818cf8',
        gradient: 'linear-gradient(135deg, #818cf8, #6366f1)',
        glow: '0 0 20px rgba(129, 140, 248, 0.5)'
    },
    dance: {
        icon: 'fa-music',
        color: '#f472b6',
        gradient: 'linear-gradient(135deg, #f472b6, #ec4899)',
        glow: '0 0 20px rgba(244, 114, 182, 0.5)'
    },
    theater: {
        icon: 'fa-masks-theater',
        color: '#a855f7',
        gradient: 'linear-gradient(135deg, #a855f7, #bf00ff)',
        glow: '0 0 20px rgba(168, 85, 247, 0.5)'
    },
    // Crime & Security
    law_enforcement: {
        icon: 'fa-shield-alt',
        color: '#ff0040',
        gradient: 'linear-gradient(135deg, #ff0040, #f43f5e)',
        glow: '0 0 20px rgba(255, 0, 64, 0.5)'
    },
    national_security: {
        icon: 'fa-flag-usa',
        color: '#ef4444',
        gradient: 'linear-gradient(135deg, #ef4444, #dc2626)',
        glow: '0 0 20px rgba(239, 68, 68, 0.5)'
    },
    intelligence: {
        icon: 'fa-eye',
        color: '#6b7280',
        gradient: 'linear-gradient(135deg, #6b7280, #4b5563)',
        glow: '0 0 20px rgba(107, 114, 128, 0.5)'
    },
    cybercrime: {
        icon: 'fa-user-secret',
        color: '#00ff88',
        gradient: 'linear-gradient(135deg, #00ff88, #00f5ff)',
        glow: '0 0 20px rgba(0, 255, 136, 0.5)'
    },
    fraud: {
        icon: 'fa-user-secret',
        color: '#f59e0b',
        gradient: 'linear-gradient(135deg, #f59e0b, #ef4444)',
        glow: '0 0 20px rgba(245, 158, 11, 0.5)'
    },
    corruption: {
        icon: 'fa-gavel',
        color: '#ef4444',
        gradient: 'linear-gradient(135deg, #ef4444, #dc2626)',
        glow: '0 0 20px rgba(239, 68, 68, 0.5)'
    },
    terrorism: {
        icon: 'fa-bomb',
        color: '#dc2626',
        gradient: 'linear-gradient(135deg, #dc2626, #991b1b)',
        glow: '0 0 20px rgba(220, 38, 38, 0.5)'
    },
    border_security: {
        icon: 'fa-fence',
        color: '#ef4444',
        gradient: 'linear-gradient(135deg, #ef4444, #f97316)',
        glow: '0 0 20px rgba(239, 68, 68, 0.5)'
    },
    emergency_services: {
        icon: 'fa-ambulance',
        color: '#ef4444',
        gradient: 'linear-gradient(135deg, #ef4444, #f59e0b)',
        glow: '0 0 20px rgba(239, 68, 68, 0.5)'
    },
    // Lifestyle & Wellness
    fitness: {
        icon: 'fa-dumbbell',
        color: '#4ade80',
        gradient: 'linear-gradient(135deg, #4ade80, #22c55e)',
        glow: '0 0 20px rgba(74, 222, 128, 0.5)'
    },
    nutrition: {
        icon: 'fa-apple-alt',
        color: '#34d399',
        gradient: 'linear-gradient(135deg, #34d399, #10b981)',
        glow: '0 0 20px rgba(52, 211, 153, 0.5)'
    },
    mental_health: {
        icon: 'fa-brain',
        color: '#a855f7',
        gradient: 'linear-gradient(135deg, #a855f7, #8b5cf6)',
        glow: '0 0 20px rgba(168, 85, 247, 0.5)'
    },
    relationships: {
        icon: 'fa-heart',
        color: '#f472b6',
        gradient: 'linear-gradient(135deg, #f472b6, #ec4899)',
        glow: '0 0 20px rgba(244, 114, 182, 0.5)'
    },
    parenting: {
        icon: 'fa-baby',
        color: '#fbbf24',
        gradient: 'linear-gradient(135deg, #fbbf24, #fcd34d)',
        glow: '0 0 20px rgba(251, 191, 36, 0.5)'
    },
    home_living: {
        icon: 'fa-home',
        color: '#60a5fa',
        gradient: 'linear-gradient(135deg, #60a5fa, #3b82f6)',
        glow: '0 0 20px rgba(96, 165, 250, 0.5)'
    },
    pets: {
        icon: 'fa-paw',
        color: '#f59e0b',
        gradient: 'linear-gradient(135deg, #f59e0b, #fbbf24)',
        glow: '0 0 20px rgba(245, 158, 11, 0.5)'
    },
    hobbies: {
        icon: 'fa-puzzle-piece',
        color: '#818cf8',
        gradient: 'linear-gradient(135deg, #818cf8, #6366f1)',
        glow: '0 0 20px rgba(129, 140, 248, 0.5)'
    },
    spirituality: {
        icon: 'fa-pray',
        color: '#a855f7',
        gradient: 'linear-gradient(135deg, #a855f7, #8b5cf6)',
        glow: '0 0 20px rgba(168, 85, 247, 0.5)'
    },
    self_improvement: {
        icon: 'fa-chart-line',
        color: '#4ade80',
        gradient: 'linear-gradient(135deg, #4ade80, #22c55e)',
        glow: '0 0 20px rgba(74, 222, 128, 0.5)'
    },
    unknown: {
        icon: 'fa-question',
        color: '#94a3b8',
        gradient: 'linear-gradient(135deg, #94a3b8, #64748b)',
        glow: 'none'
    }
};

// Sample articles
const sampleArticles = {
    tech: "OpenAI has unveiled GPT-5, demonstrating unprecedented natural language capabilities. The model features enhanced reasoning and improved contextual awareness, potentially revolutionizing healthcare and education.",
    sports: "In a stunning Wimbledon upset, unseeded Maria Rodriguez defeated reigning champion Novak Djokovic in straight sets 6-4, 7-5. The 21-year-old Spanish player showcased exceptional athleticism.",
    politics: "The Senate passed a landmark $500 billion climate change bill today with bipartisan support, allocating funds for renewable energy initiatives and carbon capture technology.",
    business: "Apple reported record quarterly earnings of $120 billion driven by strong iPhone sales and growing Services revenue. The company announced a $90 billion stock buyback program.",
    entertainment: "The 96th Academy Awards ceremony celebrated a diverse range of films, with Oppenheimer taking home Best Picture. The event featured moving tributes and surprise appearances.",
    health: "A groundbreaking clinical trial has shown promising results for a new Alzheimer's treatment, with patients demonstrating significant cognitive improvement.",
    science: "NASA's James Webb Space Telescope has captured unprecedented images of distant galaxies, revealing new insights into the early universe.",
    world: "The G20 summit concluded with a historic agreement on climate finance, with developed nations pledging $100 billion annually to support developing countries.",
    education: "Harvard University announced a new tuition-free program for families earning under $85,000 annually, expanding access to higher education for millions of students.",
    environment: "Scientists report record-breaking renewable energy adoption, with solar and wind power now comprising 30% of global electricity generation.",
    finance: "The Federal Reserve announced interest rate adjustments to combat inflation while maintaining economic growth and employment stability.",
    cryptocurrency: "Bitcoin reached new highs as institutional investors embrace digital assets, with major banks launching cryptocurrency trading platforms.",
    travel: "International tourism rebounds as countries ease travel restrictions, with airlines adding new routes to meet surging demand for vacation destinations.",
    food: "Michelin-starred chefs are revolutionizing plant-based cuisine, creating sustainable dining experiences that rival traditional fine dining establishments.",
    gaming: "The highly anticipated video game release broke sales records, with millions of players engaging in the new multiplayer online battle arena experience.",
    crime: "Federal investigators announced arrests in a major cybercrime operation, dismantling a criminal network responsible for millions in financial fraud.",
    artificial_intelligence: "DeepMind's latest artificial intelligence breakthrough demonstrates human-level reasoning in complex scientific problem-solving tasks.",
    space: "SpaceX successfully launched its Starship rocket, marking a milestone in reusable spacecraft technology and Mars exploration missions.",
    cybersecurity: "A major data breach exposed millions of user records, prompting calls for stronger encryption and enhanced security protocols across the tech industry."
};

// =============================================================================
// PARTICLE SYSTEM
// Particle system and cursor glow removed for cleaner professional design

// Scroll animations simplified for professional design

// =============================================================================
// API FUNCTIONS
// =============================================================================
const api = {
    async request(endpoint, options = {}) {
        const cacheKey = requestCache.generateKey(endpoint, options.body);

        // Check cache
        const cached = requestCache.get(cacheKey);
        if (cached && !options.skipCache) {
            return { ...cached, fromCache: true };
        }

        // Cancel previous request if exists (only for text classification requests)
        if (state.abortController) {
            state.abortController.abort();
        }
        state.abortController = new AbortController();

        try {
            // Build headers - don't set Content-Type for FormData (let browser set it with boundary)
            const headers = { ...options.headers };
            const isFormData = options.body instanceof FormData;
            if (!isFormData) {
                headers['Content-Type'] = 'application/json';
            }

            const response = await fetch(`${CONFIG.API_BASE}${endpoint}`, {
                ...options,
                signal: state.abortController.signal,
                headers
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ message: 'Unknown error' }));
                throw new Error(error.message || `HTTP ${response.status}`);
            }

            const data = await response.json();

            // Cache successful responses
            if (data.status === 'success') {
                requestCache.set(cacheKey, data);
            }

            return data;
        } catch (error) {
            if (error.name === 'AbortError') {
                return null;
            }
            throw error;
        }
    },

    async classifyText(text, enhanced = true) {
        return this.request('/classify', {
            method: 'POST',
            body: JSON.stringify({ text: text.trim(), enhanced })
        });
    },

    async classifyFile(file, type) {
        console.log(`[API classifyFile] Starting upload - type: ${type}, file:`, file?.name, 'size:', file?.size);

        if (!file) {
            console.error('[API classifyFile] No file provided');
            throw new Error('No file selected');
        }

        const formData = new FormData();
        // Backend expects 'file' as the field name for all upload types
        formData.append('file', file);

        // Verify FormData contents
        console.log(`[API classifyFile] FormData created, checking entries:`);
        for (let pair of formData.entries()) {
            console.log(`  - ${pair[0]}:`, pair[1] instanceof File ? `File(${pair[1].name})` : pair[1]);
        }

        // Use a separate AbortController for file uploads with longer timeout
        const fileAbortController = new AbortController();

        // Set a longer timeout for file uploads (10 minutes for large files/video processing)
        const timeoutId = setTimeout(() => {
            console.log(`[API classifyFile] Timeout reached, aborting upload`);
            fileAbortController.abort();
        }, 600000); // 10 minutes

        try {
            const url = `${CONFIG.API_BASE}/classify/${type}`;
            console.log(`[API classifyFile] Sending POST to: ${url}`);
            console.log(`[API classifyFile] File details: name=${file.name}, size=${file.size}, type=${file.type}`);

            const response = await fetch(url, {
                method: 'POST',
                body: formData,
                signal: fileAbortController.signal
            });

            clearTimeout(timeoutId);

            console.log(`[API classifyFile] Response received - status: ${response.status}, ok: ${response.ok}`);

            if (!response.ok) {
                // Handle 499 Client Closed Request specifically
                if (response.status === 499) {
                    throw new Error('Upload was cancelled or connection was lost. Please try again.');
                }

                // Try to get error details from response
                let errorData;
                try {
                    errorData = await response.json();
                    console.error(`[API classifyFile] Error response:`, errorData);
                } catch (e) {
                    errorData = { message: `Server error (${response.status})` };
                }

                throw new Error(errorData.message || errorData.error || `${type} classification failed`);
            }

            const result = await response.json();
            console.log(`[API classifyFile] Success result:`, result);
            return result;
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                console.log(`[API classifyFile] Upload aborted`);
                return null;
            }
            // Enhance network error messages
            if (error.message?.includes('NetworkError') || error.message?.includes('Failed to fetch')) {
                throw new Error('Network connection failed. Please check your connection and try again.');
            }
            console.error(`[API classifyFile] Upload error:`, error);
            throw error;
        }
    },

    async getHealth() {
        return this.request('/health', { skipCache: true });
    },

    async getModelInfo() {
        return this.request('/model/info');
    }
};

// =============================================================================
// LOADING MANAGER
// =============================================================================
const loadingManager = {
    overlay: null,
    text: null,
    subtext: null,

    init() {
        this.overlay = document.getElementById('loading-overlay');
        this.text = document.getElementById('loading-text');
        this.subtext = document.getElementById('loading-subtext');
    },

    show(message = 'Analyzing...', subMessage = 'Processing your content') {
        if (!this.overlay) this.init();

        if (this.text) this.text.textContent = message;
        if (this.subtext) this.subtext.textContent = subMessage;

        this.overlay?.classList.add('active');
        document.body.style.overflow = 'hidden';
    },

    hide() {
        this.overlay?.classList.remove('active');
        document.body.style.overflow = '';
    },

    update(message, subMessage = null) {
        if (message && this.text) {
            this.animateTextChange(this.text, message);
        }
        if (subMessage && this.subtext) {
            this.animateTextChange(this.subtext, subMessage);
        }
    },

    animateTextChange(element, newText) {
        element.style.opacity = '0';
        setTimeout(() => {
            element.textContent = newText;
            element.style.opacity = '1';
        }, 150);
    }
};

// =============================================================================
// NOTIFICATION MANAGER
// =============================================================================
const notificationManager = {
    container: null,

    init() {
        this.container = document.createElement('div');
        this.container.className = 'notification-container';
        document.body.appendChild(this.container);
    },

    show(message, type = 'info', duration = 4000) {
        if (!this.container) this.init();

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;

        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            info: 'info-circle',
            warning: 'exclamation-triangle'
        };

        notification.innerHTML = `
            <i class="fas fa-${icons[type] || icons.info}"></i>
            <span>${utils.sanitizeText(message)}</span>
        `;

        this.container.appendChild(notification);

        // Animate in
        requestAnimationFrame(() => {
            notification.classList.add('show');
        });

        // Auto remove
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }
};

// =============================================================================
// RESULTS MANAGER - Clean v3/v4 Style Display
// =============================================================================
const resultsManager = {
    display(data) {
        const section = document.getElementById('results-section');
        if (!section) return;

        section.style.display = 'block';
        section.classList.remove('hidden');
        section.classList.add('animate-fade-in');

        setTimeout(() => {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        const category = data.category || 'unknown';
        const confidence = Math.min((data.confidence || 0), 100);
        const style = categoryStyles[category] || categoryStyles.unknown;

        // Update main result card
        this.updateResultCard(data, style);

        // Update summary
        this.updateSummary(data);

        // Update key metrics
        this.updateMetrics(data);

        // Display topics grid (v3/v4 style)
        this.displayTopicsGrid(data);

        // Display extracted text/media info if available
        this.displayExtractedInfo(data);

        // Display model info
        this.displayModelInfo(data);

        notificationManager.show(
            `Analysis complete! ${utils.capitalizeFirst(category)} (${confidence.toFixed(1)}% confidence)`,
            'success'
        );
    },

    updateResultCard(data, style) {
        // Update category display
        const categoryEl = document.getElementById('result-category');
        if (categoryEl) {
            categoryEl.innerHTML = `
                <div class="result-category-main" style="color: ${style.color}">
                    ${utils.capitalizeFirst(data.category_display || data.category)}
                </div>
            `;
        }

        // Update confidence badge
        const badge = document.getElementById('result-confidence-badge');
        if (badge) {
            badge.style.background = `linear-gradient(135deg, ${style.color}30, ${style.color}10)`;
            badge.style.borderColor = style.color;
            badge.style.color = style.color;
            const badgeText = badge.querySelector('span');
            if (badgeText) badgeText.textContent = `${(data.confidence || 0).toFixed(1)}%`;
        }

        // Update confidence level indicator
        const confidenceLevel = document.getElementById('result-confidence-level');
        if (confidenceLevel) {
            const level = data.confidence_level || this.getConfidenceLevel(data.confidence);
            const levelText = this.formatConfidenceLevel(level);
            confidenceLevel.textContent = levelText;
            confidenceLevel.style.color = this.getConfidenceColor(data.confidence);
        }
    },

    updateSummary(data) {
        const summaryEl = document.getElementById('result-summary');
        if (!summaryEl) return;

        // Use provided summary or generate one
        let summary = data.summary || '';
        if (!summary) {
            summary = this.generateSummary(data);
        }

        summaryEl.textContent = summary;
    },

    generateSummary(data) {
        const category = utils.capitalizeFirst(data.category_display || data.category || 'Unknown');
        const confidence = (data.confidence || 0).toFixed(1);
        const wordCount = data.word_count || (data.original_text?.split(/\s+/).length || 0);

        let summary = `Classified as ${category} with ${confidence}% confidence. `;
        summary += `Processed ${wordCount} words.`;

        // Add secondary topics if available
        if (data.topics && data.topics.length > 0) {
            const secondary = data.topics.slice(0, 3).map(t => utils.capitalizeFirst(t.category_display || t.category)).join(', ');
            if (secondary) {
                summary += ` Related topics: ${secondary}.`;
            }
        }

        return summary;
    },

    updateMetrics(data) {
        // Word count
        const wordCountEl = document.getElementById('metric-word-count');
        if (wordCountEl) {
            wordCountEl.textContent = data.word_count || 0;
        }

        // Processing time
        const timeEl = document.getElementById('metric-processing-time');
        if (timeEl) {
            const timeMs = data.processing_time_ms || 0;
            timeEl.textContent = timeMs > 0 ? `${timeMs.toFixed(0)}ms` : 'N/A';
        }

        // Keywords count
        const keywordsEl = document.getElementById('metric-keywords');
        if (keywordsEl) {
            const count = data.keywords ? data.keywords.length : 0;
            keywordsEl.textContent = count;
        }

        // Entities count
        const entitiesEl = document.getElementById('metric-entities');
        if (entitiesEl) {
            const count = data.entities ? data.entities.length : 0;
            entitiesEl.textContent = count;
        }
    },

    displayKeywords(data) {
        // For now, we'll keep the topics grid but make it cleaner
        // This method can be expanded later if we want a separate keyword list
        return;
    },

    displayTopicsGrid(data) {
        const container = document.getElementById('topics-grid');
        if (!container) return;

        const predictions = data.topics || data.top_predictions || [];
        if (predictions.length === 0) {
            container.innerHTML = '<div class="topic-chip">No additional topics detected</div>';
            return;
        }

        // Show only top 5-8 predictions for cleaner display (v3/v4 style)
        const displayPredictions = predictions.slice(0, 6);
        let topicsHtml = '';

        displayPredictions.forEach((pred, index) => {
            const confidence = Math.min(pred.confidence || 0, 100);
            const category = pred.category_display || pred.category;
            const style = categoryStyles[pred.category] || categoryStyles.unknown;
            const icon = this.getCategoryIcon(pred.category);

            topicsHtml += `
                <div class="topic-chip" style="animation-delay: ${index * 0.05}s">
                    <i class="fas ${icon}" style="color: ${style.color}"></i>
                    <span>${utils.capitalizeFirst(category)}</span>
                    <span class="topic-confidence">${confidence.toFixed(1)}%</span>
                </div>
            `;
        });

        container.innerHTML = topicsHtml;
    },

    getConfidenceLevel(confidence) {
        if (confidence >= 90) return 'very_high';
        if (confidence >= 75) return 'high';
        if (confidence >= 60) return 'moderate';
        if (confidence >= 40) return 'low';
        return 'very_low';
    },

    formatConfidenceLevel(level) {
        const labels = {
            'very_high': 'Very High Confidence',
            'high': 'High Confidence',
            'moderate': 'Moderate Confidence',
            'low': 'Low Confidence',
            'very_low': 'Very Low Confidence'
        };
        return labels[level] || level;
    },

    getConfidenceColor(confidence) {
        if (confidence >= 90) return '#10b981'; // green
        if (confidence >= 75) return '#3b82f6'; // blue
        if (confidence >= 60) return '#f59e0b'; // amber
        if (confidence >= 40) return '#f97316'; // orange
        return '#ef4444'; // red
    },

    getCategoryIcon(category) {
        const style = categoryStyles[category] || categoryStyles.unknown;
        return style.icon;
    },

    displayExtractedInfo(data) {
        const container = document.getElementById('extracted-info-container');
        if (!container) return;

        let html = '';

        // Show extracted text for image/audio/video
        if (data.extracted_text) {
            const inputType = data.input_type || 'content';
            const sourceLabel = {
                'image': 'OCR Extracted Text',
                'audio': 'Transcribed Audio',
                'video': 'Extracted Video Content'
            }[inputType] || 'Extracted Text';

            html += `
                <div class="extracted-info-card">
                    <h4><i class="fas fa-file-alt"></i> ${sourceLabel}</h4>
                    <div class="extracted-text-content">
                        ${utils.sanitizeText(data.extracted_text)}
                        ${data.extracted_text.length >= 500 ? '<span class="text-truncated">... (truncated)</span>' : ''}
                    </div>
                    ${data.ocr_confidence ? `<div class="extraction-confidence">OCR Confidence: ${(data.ocr_confidence * 100).toFixed(1)}%</div>` : ''}
                    ${data.transcription_confidence ? `<div class="extraction-confidence">Transcription Confidence: ${(data.transcription_confidence * 100).toFixed(1)}%</div>` : ''}
                </div>
            `;
        }

        // Show video/audio specific info
        if (data.duration) {
            html += `
                <div class="media-info-item">
                    <i class="fas fa-clock"></i>
                    <span>Duration: ${data.duration.toFixed(1)}s</span>
                </div>
            `;
        }

        if (data.frames_processed) {
            html += `
                <div class="media-info-item">
                    <i class="fas fa-images"></i>
                    <span>Frames Processed: ${data.frames_processed}</span>
                </div>
            `;
        }

        container.innerHTML = html || '';
        container.classList.toggle('hidden', !html);
    },

    displayModelInfo(data) {
        const container = document.getElementById('model-info-container');
        if (!container) return;

        const modelName = data.model || 'Unknown Model';
        const processingTime = data.processing_time_ms ? `${data.processing_time_ms.toFixed(0)}ms` : 'N/A';
        const inputType = data.input_type || 'text';

        container.innerHTML = `
            <div class="model-info-grid">
                <div class="model-info-item">
                    <i class="fas fa-robot"></i>
                    <span>Model: ${utils.sanitizeText(modelName)}</span>
                </div>
                <div class="model-info-item">
                    <i class="fas fa-tachometer-alt"></i>
                    <span>Processing Time: ${processingTime}</span>
                </div>
                <div class="model-info-item">
                    <i class="fas fa-keyboard"></i>
                    <span>Input Type: ${utils.capitalizeFirst(inputType)}</span>
                </div>
            </div>
        `;
    },

    // Topics grid and summary methods implemented above
};

// =============================================================================
// UI MANAGER
// =============================================================================
const uiManager = {
    switchInputType(type) {
        state.currentInputType = type;

        // Update tab buttons
        document.querySelectorAll('.input-type-btn').forEach(btn => {
            const isActive = btn.dataset.type === type;
            btn.classList.toggle('active', isActive);
        });

        // Update panels
        document.querySelectorAll('.input-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.type === type);
        });

        this.updateAnalyzeButton();
    },

    updateFilePreview(type, file) {
        const infoEl = document.getElementById(`${type}-info`);
        const nameEl = document.getElementById(`${type}-filename`);
        const sizeEl = document.getElementById(`${type}-filesize`);

        if (nameEl) nameEl.textContent = file.name;
        if (sizeEl) sizeEl.textContent = utils.formatFileSize(file.size);
        if (infoEl) infoEl.classList.remove('hidden');

        // Hide upload zone
        const uploadZone = document.getElementById(`${type}-upload-zone`);
        if (uploadZone) uploadZone.classList.add('hidden');
    },

    removeFile(type) {
        state.selectedFiles[type] = null;

        const infoEl = document.getElementById(`${type}-info`);
        const uploadZone = document.getElementById(`${type}-upload-zone`);
        const input = document.getElementById(`${type}-input`);

        if (infoEl) infoEl.classList.add('hidden');
        if (uploadZone) uploadZone.classList.remove('hidden');
        if (input) input.value = '';

        this.updateAnalyzeButton();
    },

    updateAnalyzeButton() {
        const btn = document.getElementById('analyze-btn');
        if (!btn) return;

        let isValid = false;
        let debugLength = 0;

        switch (state.currentInputType) {
            case 'text':
                const textarea = document.getElementById('news-text');
                if (textarea) {
                    debugLength = textarea.value.trim().length;
                    isValid = debugLength >= CONFIG.MIN_CHARS;
                }
                console.log(`[Debug] Text length: ${debugLength}, MIN: ${CONFIG.MIN_CHARS}, Valid: ${isValid}`);
                break;
            case 'image':
            case 'audio':
            case 'video':
                isValid = state.selectedFiles[state.currentInputType] !== null;
                console.log(`[Debug] File selected: ${isValid}`);
                break;
        }

        btn.disabled = !isValid || state.isAnalyzing;
        btn.classList.toggle('disabled', !isValid);
        console.log(`[Debug] Button disabled: ${btn.disabled}, isAnalyzing: ${state.isAnalyzing}`);
    },

    updateCharCount() {
        const textarea = document.getElementById('news-text');
        const countEl = document.getElementById('char-count');

        if (textarea && countEl) {
            countEl.textContent = textarea.value.length;
        }

        this.updateAnalyzeButton();
    }
};

// =============================================================================
// EVENT HANDLERS
// =============================================================================
const eventHandlers = {
    handleFileSelect(e, type) {
        console.log(`[FileSelect] Event triggered for ${type}`, e.target);

        const file = e.target.files?.[0];
        if (!file) {
            console.warn(`[FileSelect] No file selected for ${type}`);
            return;
        }

        console.log(`[FileSelect] File selected: ${file.name}, size: ${file.size}, type: ${file.type} `);

        const maxSize = CONFIG.MAX_FILE_SIZES[type];
        if (file.size > maxSize) {
            const maxMB = Math.round(maxSize / 1024 / 1024);
            notificationManager.show(`File too large.Maximum ${maxMB}MB allowed.`, 'error');
            console.warn(`[FileSelect] File too large: ${file.size} > ${maxSize} `);
            return;
        }

        state.selectedFiles[type] = file;
        console.log(`[FileSelect] File stored in state.selectedFiles[${type}]`);

        uiManager.updateFilePreview(type, file);
        uiManager.updateAnalyzeButton();

        notificationManager.show(`${utils.capitalizeFirst(type)} selected: ${file.name} `, 'success');
    },

    handleDrop(e, type) {
        e.preventDefault();
        e.stopPropagation();

        const dropZone = document.getElementById(`${type}-upload - zone`);
        dropZone?.classList.remove('drag-over');

        const files = e.dataTransfer.files;
        console.log(`[Drop] Files dropped: ${files.length} files`);

        if (files.length > 0) {
            const file = files[0];
            const input = document.getElementById(`${type}-input`);

            // Create a DataTransfer to simulate file input
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;

            console.log(`[Drop] File set on input: ${file.name} `);
            this.handleFileSelect({ target: input }, type);
        }
    },

    handleUploadZoneClick(type) {
        console.log(`[UploadZone] Click triggered for ${type}`);
        const input = document.getElementById(`${type}-input`);
        if (input) {
            console.log(`[UploadZone] Triggering click on file input`);
            input.click();
        } else {
            console.error(`[UploadZone] Input element not found for ${type}`);
        }
    }
};

// =============================================================================
// ANALYSIS FUNCTION
// =============================================================================
async function analyzeContent() {
    if (state.isAnalyzing) {
        console.log('[Analyze] Already analyzing, skipping');
        return;
    }

    console.log(`[Analyze] Starting analysis for type: ${state.currentInputType} `);
    state.isAnalyzing = true;
    uiManager.updateAnalyzeButton();

    const messages = {
        text: { main: 'Analyzing...', sub: 'Processing text content' },
        image: { main: 'Analyzing...', sub: 'Processing image content' },
        audio: { main: 'Audio Analysis...', sub: 'Transcribing with Whisper' },
        video: { main: 'Video Analysis...', sub: 'Scene detection in progress' }
    };

    const msg = messages[state.currentInputType];
    loadingManager.show(msg.main, msg.sub);

    try {
        let result;
        let inputText = '';

        switch (state.currentInputType) {
            case 'text':
                const textarea = document.getElementById('news-text');
                inputText = textarea?.value.trim() || '';
                console.log(`[Analyze] Text length: ${inputText.length}, content: "${inputText.substring(0, 50)}..."`);
                loadingManager.update('Processing...', 'Analyzing text content');
                result = await api.classifyText(inputText);
                console.log('[Analyze] API response received:', result);
                break;

            case 'image':
            case 'audio':
            case 'video':
                const file = state.selectedFiles[state.currentInputType];
                console.log(`[Analyze] Selected file for ${state.currentInputType}: `, file);

                if (!file) {
                    throw new Error(`No ${state.currentInputType} file selected`);
                }

                loadingManager.update('Processing...', 'AI models analyzing content');
                result = await api.classifyFile(file, state.currentInputType);
                inputText = file?.name || 'File';
                break;
        }

        // Handle result - check for null (aborted), error status, or success
        if (result === null) {
            // Request was aborted - don't show error, just stop
            loadingManager.hide();
            return;
        }

        if (result?.status === 'success') {
            loadingManager.update('Finalizing...', 'Preparing results display');
            addToHistory(result.data || result, inputText, state.currentInputType);

            setTimeout(() => {
                resultsManager.display(result.data || result);
                loadingManager.hide();
            }, 300);
        } else if (result?.status === 'error') {
            // API returned an error status
            throw new Error(result.message || 'Server returned an error');
        } else {
            // Unexpected response format
            throw new Error('Invalid response from server');
        }
    } catch (error) {
        loadingManager.hide();

        // Provide user-friendly error messages based on error type
        let errorMessage = error.message || 'Unknown error occurred';

        if (error.name === 'AbortError') {
            // Request was cancelled - no need to show error
            console.log('Request was cancelled');
            return;
        } else if (error.message?.includes('NetworkError') ||
            error.message?.includes('Failed to fetch') ||
            error.message?.includes('network')) {
            errorMessage = 'Network connection failed. Please check your connection and try again.';
        } else if (error.message?.includes('499') || error.message?.includes('cancelled')) {
            errorMessage = 'Upload was cancelled or connection was lost. Please try again.';
        } else if (error.message?.includes('HTTP')) {
            errorMessage = `Server error: ${error.message} `;
        }

        notificationManager.show(`Analysis failed: ${errorMessage} `, 'error');
        console.error('Analysis error:', error);
    } finally {
        state.isAnalyzing = false;
        uiManager.updateAnalyzeButton();
    }
}

// =============================================================================
// HISTORY MANAGEMENT
// =============================================================================
function addToHistory(result, inputText, inputType) {
    const entry = {
        timestamp: new Date().toISOString(),
        category: result.category,
        confidence: result.confidence,
        inputType,
        inputPreview: inputType === 'text' ? inputText.substring(0, 100) + '...' : inputText
    };

    state.classificationHistory.unshift(entry);
    if (state.classificationHistory.length > 50) {
        state.classificationHistory.pop();
    }

    try {
        localStorage.setItem('newscat_history', JSON.stringify(state.classificationHistory));
    } catch (e) {
        console.warn('Failed to save history:', e);
    }
}

function loadHistory() {
    try {
        const saved = localStorage.getItem('newscat_history');
        if (saved) {
            state.classificationHistory = JSON.parse(saved);
        }
    } catch (e) {
        console.warn('Failed to load history:', e);
    }
}

// =============================================================================
// SAMPLE LOADER
// =============================================================================
function loadSample(type) {
    const textarea = document.getElementById('news-text');
    if (!textarea || !sampleArticles[type]) return;

    textarea.value = '';
    uiManager.updateCharCount();

    const text = sampleArticles[type];
    let index = 0;

    const typeInterval = setInterval(() => {
        if (index < text.length) {
            textarea.value += text.charAt(index);
            index++;
            uiManager.updateCharCount();
        } else {
            clearInterval(typeInterval);
        }
    }, 3);
}

// =============================================================================
// INITIALIZATION
// =============================================================================
function initEventListeners() {
    // Text input
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.addEventListener('input', () => uiManager.updateCharCount());
    }

    // Tab switching
    document.querySelectorAll('.input-type-btn').forEach(btn => {
        btn.addEventListener('click', () => uiManager.switchInputType(btn.dataset.type));
    });

    // Analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeContent);
    }

    // File inputs
    ['image', 'audio', 'video'].forEach(type => {
        const input = document.getElementById(`${type}-input`);
        if (input) {
            console.log(`[Init] Attaching change listener to ${type}-input`);
            input.addEventListener('change', (e) => eventHandlers.handleFileSelect(e, type));
        } else {
            console.error(`[Init] Input element not found: ${type}-input`);
        }

        // Drag and drop
        const dropZone = document.getElementById(`${type}-upload-zone`);
        if (dropZone) {
            console.log(`[Init] Setting up dropZone for ${type}`);

            // Click to browse files - use the dedicated handler
            dropZone.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                eventHandlers.handleUploadZoneClick(type);
            });

            // Keyboard support for accessibility
            dropZone.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    eventHandlers.handleUploadZoneClick(type);
                }
            });

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                });
            });

            ['dragenter', 'dragover'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.add('drag-over');
                });
            });

            ['dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, () => {
                    dropZone.classList.remove('drag-over');
                });
            });

            dropZone.addEventListener('drop', (e) => eventHandlers.handleDrop(e, type));
        } else {
            console.error(`[Init] DropZone element not found: ${type}-upload - zone`);
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const btn = document.getElementById('analyze-btn');
            if (btn && !btn.disabled) {
                analyzeContent();
            }
        }
    });
}

function initializeApp() {
    // Initialize systems
    loadingManager.init();

    loadHistory();
    initEventListeners();
    uiManager.updateAnalyzeButton();

    // Load sample on first visit
    if (!localStorage.getItem('newscat_visited')) {
        localStorage.setItem('newscat_visited', 'true');
        loadSample('tech');
    }

    console.log('%c NEWSCAT v5 Professional ', 'background: linear-gradient(135deg, #6366f1, #a855f7); color: #fff; font-weight: bold; padding: 4px 8px; border-radius: 4px;');
    console.log('%c AI News Classification System ', 'color: #818cf8;');
}

// DOM Ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// Export for global access
window.newscat = {
    loadSample,
    removeFile: (type) => uiManager.removeFile(type),
    switchInputType: (type) => uiManager.switchInputType(type),
    getCacheStats: () => requestCache.getStats(),
    api,
    utils
};
