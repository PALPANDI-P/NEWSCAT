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
                let errMessage = `HTTP ${response.status}`;
                try {
                    const errorJson = await response.json();
                    errMessage = errorJson.message || errMessage;
                } catch (e) { }
                throw new Error(errMessage);
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
        // Backend specifically expects the key to match the type ('image', 'audio', 'video')
        formData.append(type, file);

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
            if (error.message?.includes('NetworkError') || error.message?.includes('Failed to fetch')) {
                throw new Error('Network connection failed. Please check your connection and try again.');
            }
            throw new Error(error.message || 'Error uploading file');
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
// RESULTS MANAGER - Refactored for Dashboard UI
// =============================================================================
const resultsManager = {
    display(data, inputText = 'Content analysis complete', inputType = 'text') {
        // Show results state without hiding hero state
        const resultsEl = document.getElementById('results-state');
        resultsEl.classList.remove('hidden');
        
        // Smooth scroll to results after a tiny delay to ensure DOM is updated
        setTimeout(() => {
            resultsEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
        
        // 0. Update Input Summary
        const summaryTextEl = document.getElementById('summary-text');
        const summaryBadgeEl = document.getElementById('summary-badge');
        const summaryIconEl = document.getElementById('summary-icon');
        
        if (summaryTextEl) {
            summaryTextEl.textContent = inputType === 'text' && inputText.length > 60 
                ? inputText.substring(0, 60) + '...' 
                : inputText;
        }
        
        if (summaryBadgeEl) {
            summaryBadgeEl.textContent = `${inputType.toUpperCase()} MODE`;
            // Color based on type
            summaryBadgeEl.className = 'text-[10px] uppercase tracking-widest px-3 py-1.5 rounded-full border shadow-[0_0_10px_rgba(255,255,255,0.05)] whitespace-nowrap font-bold ';
            if (inputType === 'text') summaryBadgeEl.classList.add('text-indigo-400', 'bg-indigo-500/10', 'border-indigo-500/20');
            else if (inputType === 'image') summaryBadgeEl.classList.add('text-purple-400', 'bg-purple-500/10', 'border-purple-500/20');
            else if (inputType === 'audio') summaryBadgeEl.classList.add('text-green-400', 'bg-green-500/10', 'border-green-500/20');
            else if (inputType === 'video') summaryBadgeEl.classList.add('text-cyan-400', 'bg-cyan-500/10', 'border-cyan-500/20');
        }
        
        if (summaryIconEl) {
            let iconClass = 'fa-file';
            if (inputType === 'text') iconClass = 'fa-keyboard text-indigo-400';
            else if (inputType === 'image') iconClass = 'fa-image text-purple-400';
            else if (inputType === 'audio') iconClass = 'fa-microphone text-green-400';
            else if (inputType === 'video') iconClass = 'fa-video text-cyan-400';
            summaryIconEl.className = `fas ${iconClass}`;
        }
        
        // 1. Main Category Update
        const category = data.category || 'unknown';
        const style = categoryStyles[category] || categoryStyles.unknown;
        const icon = this.getCategoryIcon(category);
        
        const mainIconEl = document.getElementById('result-main-icon');
        const iconContainerEl = document.getElementById('result-icon-container');
        const mainCategoryEl = document.getElementById('result-main-category');
        const subCategoryEl = document.getElementById('result-sub-category');
        
        mainIconEl.className = `fas ${icon}`;
        iconContainerEl.style.backgroundColor = `${style.color}33`; // 20% opacity
        iconContainerEl.style.color = style.color;
        iconContainerEl.style.borderColor = `${style.color}50`;
        
        mainCategoryEl.textContent = utils.capitalizeFirst(data.category_display || data.category);
        
        // Try mapping category to subtext
        const subtexts = {
            'politics': 'Government, elections, policies',
            'technology': 'Tech innovations, AI, software',
            'sports': 'Sports events, athletes, teams',
            'business': 'Markets, companies, economy',
            'entertainment': 'Movies, music, celebrities',
            'science': 'Scientific discoveries, research',
            'health': 'Medical news, healthcare',
            'environment': 'Climate, nature, conservation',
            'education': 'Schools, universities, learning',
            'world': 'International news, global events'
        };
        subCategoryEl.textContent = subtexts[category.toLowerCase()] || 'News Classification Result';
        
        // 2. Confidence Circle Update
        const confidence = Math.min((data.confidence || 0), 100);
        const confidenceCircle = document.getElementById('confidence-circle');
        const confidenceText = document.getElementById('confidence-text');
        
        const circumference = 2 * Math.PI * 40; // r=40
        const offset = circumference - (confidence / 100) * circumference;
        
        confidenceCircle.style.strokeDashoffset = offset;
        confidenceCircle.style.stroke = this.getConfidenceColor(confidence);
        confidenceText.textContent = `${confidence.toFixed(1)}%`;
        
        // 3. Execution Summary Panel Update
        this.updateExecutionSummary(data);
        
        // 4. Text Analysis Panel Update
        this.updateTextAnalysisGrid(data);

        notificationManager.show(`Analysis complete! ${utils.capitalizeFirst(category)}`, 'success');
    },

    updateExecutionSummary(data) {
        const textEl = document.getElementById('topic-summary-text');
        const indicatorEl = document.getElementById('topic-summary-indicator');
        
        if (!textEl) return;
        
        let summary = "Classification complete. Category matched successfully.";
        if (data.main_topic_summary) {
            summary = data.main_topic_summary;
        } else if (data.summary) {
            summary = data.summary;
        }
        
        // Typewriter effect
        textEl.textContent = "";
        let i = 0;
        textEl.style.opacity = '1';
        
        // Optional indicator color match
        if(indicatorEl && data.category) {
            const style = categoryStyles[data.category] || categoryStyles.unknown;
            indicatorEl.style.background = `linear-gradient(to bottom, ${style.color}, ${style.color}88)`;
        }

        function typeWriter() {
            if (i < summary.length) {
                textEl.textContent += summary.charAt(i);
                i++;
                setTimeout(typeWriter, 15);
            }
        }
        
        // slight delay to let the panel slide in
        setTimeout(typeWriter, 400);
    },

    updateTextAnalysisGrid(data) {
        // Only show text analysis for text input type, or simulate it if we wanted to
        const panel = document.getElementById('text-analysis-panel');
        if (!panel) return;
        
        if (state.currentInputType === 'text') {
            panel.classList.remove('hidden');
            const wordsMatch = data.original_text ? data.original_text.match(/\\b\\w+\\b/g) : [];
            const words = wordsMatch ? wordsMatch.length : 0;
            const chars = data.original_text ? Math.min(data.original_text.length, 10000) : 0;
            const sentences = data.original_text ? Math.max(1, (data.original_text.match(/[.!?]+/g) || []).length) : 0;
            const readability = Math.min(20, Math.max(1, Math.round(words / Math.max(1, sentences) * 0.5))); // Fake simplified readability
            
            this.animateValue('analysis-words', 0, words, 1000);
            this.animateValue('analysis-chars', 0, chars, 1000);
            this.animateValue('analysis-sentences', 0, sentences, 1000);
            this.animateValue('analysis-readability', 0, readability, 1000);
        } else {
            panel.classList.add('hidden'); // Hide text analysis metrics if not text mode
        }
    },
    
    animateValue(id, start, end, duration) {
        const obj = document.getElementById(id);
        if (!obj) return;
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            obj.innerHTML = Math.floor(progress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    },

    getConfidenceColor(confidence) {
        if (confidence >= 90) return '#22c55e'; // green-500
        if (confidence >= 75) return '#3b82f6'; // blue-500
        if (confidence >= 60) return '#f59e0b'; // amber-500
        if (confidence >= 40) return '#f97316'; // orange-500
        return '#ef4444'; // red-500
    },

    getCategoryIcon(category) {
        const style = categoryStyles[category] || categoryStyles.unknown;
        return style.icon;
    }
};

// =============================================================================
// UI MANAGER
// =============================================================================
const uiManager = {
    switchInputType(type) {
        state.currentInputType = type;

        // Reset display to hero
        document.getElementById('results-state').classList.add('hidden');
        document.getElementById('hero-state').classList.remove('hidden');

        // Note: Tab highlight handled by inline script in index.html
        
        this.updateAnalyzeButton();
    },

    updateAnalyzeButton() {
        const btn = document.getElementById('analyze-btn');
        if (!btn) return;

        let isValid = false;

        switch (state.currentInputType) {
            case 'text':
                const textarea = document.getElementById('text-input');
                if (textarea) {
                    isValid = textarea.value.trim().length >= CONFIG.MIN_CHARS;
                }
                break;
            case 'image':
            case 'audio':
            case 'video':
                isValid = state.selectedFiles[state.currentInputType] !== null;
                break;
        }

        btn.disabled = !isValid || state.isAnalyzing;
        btn.classList.toggle('opacity-50', !isValid || state.isAnalyzing);
        btn.classList.toggle('cursor-not-allowed', !isValid || state.isAnalyzing);
    },

    updateCharCount() {
        const textarea = document.getElementById('text-input');
        const countEl = document.getElementById('char-count');
        const warningEl = document.getElementById('char-warning');

        if (textarea && countEl) {
            const len = textarea.value.length;
            const trimLen = textarea.value.trim().length;
            countEl.textContent = len;
            
            if (warningEl) {
                if (trimLen > 0 && trimLen < CONFIG.MIN_CHARS) {
                    warningEl.classList.remove('hidden');
                } else {
                    warningEl.classList.add('hidden');
                }
            }
        }

        this.updateAnalyzeButton();
    },
    
    updateFilePreview(type, file) {
        // Custom simple preview implementation based on our new dashboard UI
        const previewContainer = document.getElementById(`${type}-preview-container`);
        const previewEl = document.getElementById(`${type}-preview`);
        
        if (previewContainer && previewEl && file) {
            if (type === 'image') {
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewEl.src = e.target.result;
                    previewContainer.classList.remove('hidden');
                };
                reader.readAsDataURL(file);
            } else if (type === 'audio' || type === 'video') {
                const url = URL.createObjectURL(file);
                previewEl.src = url;
                previewContainer.classList.remove('hidden');
            }
        }
    }
};

// =============================================================================
// EVENT HANDLERS
// =============================================================================
const eventHandlers = {
    handleFileSelect(e, type) {
        console.log(`[FileSelect] Event triggered for ${type}`);

        const file = e.target.files?.[0];
        if (!file) return;

        const maxSize = CONFIG.MAX_FILE_SIZES[type];
        if (file.size > maxSize) {
            const maxMB = Math.round(maxSize / 1024 / 1024);
            notificationManager.show(`File too large. Maximum ${maxMB}MB allowed.`, 'error');
            return;
        }

        state.selectedFiles[type] = file;
        uiManager.updateFilePreview(type, file);
        uiManager.updateAnalyzeButton();
        
        notificationManager.show(`${utils.capitalizeFirst(type)} ready to analyze`, 'success');
    }
};

// =============================================================================
// ANALYSIS FUNCTION
// =============================================================================
async function analyzeContent() {
    if (state.isAnalyzing) return;

    state.isAnalyzing = true;
    uiManager.updateAnalyzeButton();

    loadingManager.show('Analyzing Data', 'Ensemble ML models processing content...');

    try {
        let result;
        let inputText = '';

        switch (state.currentInputType) {
            case 'text':
                const textarea = document.getElementById('text-input');
                inputText = textarea?.value.trim() || '';
                result = await api.classifyText(inputText);
                break;

            case 'image':
            case 'audio':
            case 'video':
                const file = state.selectedFiles[state.currentInputType];
                if (!file) throw new Error(`No ${state.currentInputType} file selected`);
                result = await api.classifyFile(file, state.currentInputType);
                inputText = file.name;
                break;
        }

        if (result === null) {
            loadingManager.hide();
            return; // Aborted
        }

        if (result?.status === 'success') {
            // Simulate minimal processing time if backend too fast for UX
            setTimeout(() => {
                resultsManager.display(result.data || result, inputText, state.currentInputType);
                loadingManager.hide();
                addToHistory(result.data || result, inputText, state.currentInputType);
            }, 500);
        } else {
            throw new Error(result?.message || 'Server returned an error');
        }
    } catch (error) {
        loadingManager.hide();
        let errorMessage = error.message || 'Unknown error occurred';
        if (error.name !== 'AbortError') {
             notificationManager.show(`Analysis failed: ${errorMessage}`, 'error');
        }
    } finally {
        state.isAnalyzing = false;
        uiManager.updateAnalyzeButton();
    }
}


function addToHistory(result, inputText, inputType) {
    // Determine category based on response logic
    let categoryToSave = result.category;
    if (result.main_topic && result.main_topic.toLowerCase() !== result.category.toLowerCase()) {
        categoryToSave = `${utils.capitalizeFirst(result.main_topic)} > ${utils.capitalizeFirst(result.category)}`;
    }

    const entry = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        category: categoryToSave,
        confidence: result.confidence,
        inputType,
        inputPreview: inputType === 'text' ? inputText.substring(0, 80) + '...' : inputText
    };

    if (!state.classificationHistory) state.classificationHistory = [];
    state.classificationHistory.unshift(entry);
    
    if (state.classificationHistory.length > CONFIG.MAX_HISTORY) {
        state.classificationHistory.pop();
    }

    try {
        localStorage.setItem(CONFIG.HISTORY_KEY, JSON.stringify(state.classificationHistory));
        renderHistory();
    } catch (e) {
        console.warn('Failed to save history:', e);
    }
}

function loadHistory() {
    try {
        const saved = localStorage.getItem(CONFIG.HISTORY_KEY);
        if (saved) {
            state.classificationHistory = JSON.parse(saved);
        } else {
            state.classificationHistory = [];
        }
        renderHistory();
    } catch (e) {
        console.warn('Failed to load history:', e);
        state.classificationHistory = [];
    }
}

function clearHistory() {
    state.classificationHistory = [];
    try {
        localStorage.removeItem(CONFIG.HISTORY_KEY);
        renderHistory();
        notificationManager.show('History cleared successfully', 'success');
    } catch (e) {
        console.warn('Failed to clear history:', e);
    }
}

function renderHistory() {
    const historyList = document.getElementById('history-list');
    if (!historyList) return;

    if (!state.classificationHistory || state.classificationHistory.length === 0) {
        historyList.innerHTML = `
            <div class="history-empty p-8 text-center rounded-xl bg-dark-900 border border-dark-800">
                <i class="fas fa-inbox text-4xl mb-4 text-dark-500"></i>
                <p class="text-dark-400 font-medium">No classifications yet. Start by analyzing some content!</p>
            </div>
        `;
        return;
    }

    let html = '<div class="flex flex-col gap-3">';
    state.classificationHistory.forEach(item => {
        const date = new Date(item.timestamp).toLocaleDateString(undefined, { 
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
        });
        
        let iconHtml = '';
        switch(item.inputType) {
            case 'text': iconHtml = '<i class="fas fa-keyboard text-brand-purple"></i>'; break;
            case 'image': iconHtml = '<i class="fas fa-image text-blue-400"></i>'; break;
            case 'audio': iconHtml = '<i class="fas fa-music text-green-400"></i>'; break;
            case 'video': iconHtml = '<i class="fas fa-video text-red-400"></i>'; break;
            default: iconHtml = '<i class="fas fa-file text-gray-400"></i>';
        }

        html += `
            <div class="history-item flex flex-col sm:flex-row sm:items-center justify-between p-4 rounded-xl bg-dark-900/50 hover:bg-dark-800 border border-dark-800 hover:border-brand-purple/30 transition-all duration-300">
                <div class="flex items-start gap-4 mb-3 sm:mb-0 max-w-full sm:max-w-[70%]">
                    <div class="mt-1 bg-dark-950 p-2 rounded-lg border border-white/5 flex-shrink-0">
                        ${iconHtml}
                    </div>
                    <div class="min-w-0">
                        <p class="text-white font-medium truncate text-sm sm:text-base">${utils.sanitizeText(item.inputPreview)}</p>
                        <p class="text-dark-400 text-xs mt-1"><i class="far fa-clock mr-1"></i>${date}</p>
                    </div>
                </div>
                <div class="flex items-center gap-3 flex-shrink-0 ml-12 sm:ml-0">
                    <span class="px-3 py-1 bg-brand-purple/20 border border-brand-purple/30 text-brand-purpleLight rounded-full text-xs font-semibold whitespace-nowrap">
                        ${item.category}
                    </span>
                    <span class="text-xs font-bold ${item.confidence > 80 ? 'text-green-400' : 'text-yellow-400'}">
                        ${Number(item.confidence).toFixed(1)}%
                    </span>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
    // Add brief animation when updating
    historyList.style.opacity = '0';
    setTimeout(() => {
        historyList.innerHTML = html;
        historyList.style.transition = 'opacity 0.3s ease';
        historyList.style.opacity = '1';
    }, 150);
}

// =============================================================================
// SAMPLE LOADER & REAL-WORLD NEWS
// =============================================================================
function loadTextCharByChar(text) {
    const textarea = document.getElementById('text-input') || document.getElementById('news-text');
    if (!textarea) return;

    textarea.value = '';
    // Use the char-count element specific to where the textarea is found
    const charCount = document.getElementById('char-count');
    if (charCount) charCount.textContent = '0 characters';
    
    // Simulate typing
    let index = 0;
    const typeInterval = setInterval(() => {
        if (index < text.length) {
            textarea.value += text.charAt(index);
            index++;
            if (charCount) charCount.textContent = `${textarea.value.length} characters`;
        } else {
            clearInterval(typeInterval);
            validateInput(); // Validate input after text is loaded
        }
    }, 3);
}

function loadSample(type) {
    if (!samples[type]) return;
    loadTextCharByChar(samples[type]);
}

async function fetchRealtimeNews() {
    try {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('active');
            loadingOverlay.querySelector('.loading-text').textContent = 'Fetching Live News...';
            loadingOverlay.querySelector('.loading-subtext').textContent = 'Connecting to real-world sources';
        }
        
        const response = await api.request('/news/realtime');
        
        if (loadingOverlay) loadingOverlay.classList.remove('active');
        
        if (response.status === 'success' && response.data && response.data.length > 0) {
            // Pick a random news item
            const randomIndex = Math.floor(Math.random() * response.data.length);
            const newsItem = response.data[randomIndex];
            
            // Format the content
            const fullText = `${newsItem.title}\n\n${newsItem.content}\n\nSource: ${newsItem.source}`;
            
            // Load it smoothly
            loadTextCharByChar(fullText);
            notificationManager.show('Loaded real-world news from ' + newsItem.source, 'success');
        } else {
            notificationManager.show('Failed to fetch real-time news.', 'error');
        }
    } catch (e) {
        const loadingOverlay = document.getElementById('loading-overlay');
        if (loadingOverlay) loadingOverlay.classList.remove('active');
        console.error('Real-world news fetch failed:', e);
        notificationManager.show('Failed to connect to real-world news stream.', 'error');
    }
}

// =============================================================================
// HISTORY MANAGER
// =============================================================================
let classificationHistory = [];

function loadHistory() {
    try {
        const stored = localStorage.getItem('newscat_history');
        if (stored) {
            classificationHistory = JSON.parse(stored);
            renderHistory();
        }
    } catch (e) {
        console.error('Failed to load history', e);
    }
}

function saveHistory() {
    try {
        localStorage.setItem('newscat_history', JSON.stringify(classificationHistory));
    } catch (e) {
        console.error('Failed to save history', e);
    }
}

function addToHistory(data, inputText, inputType) {
    const entry = {
        id: Date.now().toString(),
        category: data.category,
        category_display: data.category_display || data.category,
        confidence: data.confidence,
        timestamp: new Date().toISOString(),
        inputText: inputText,
        inputType: inputType
    };
    
    // Add to beginning, keep max 10
    classificationHistory.unshift(entry);
    if (classificationHistory.length > 10) {
        classificationHistory.pop();
    }
    
    saveHistory();
    renderHistory();
}

function renderHistory() {
    const container = document.getElementById('history-list');
    if (!container) return;
    
    if (classificationHistory.length === 0) {
        container.innerHTML = `
            <div class="flex flex-col items-center justify-center p-6 text-gray-500">
                <i class="fas fa-folder-open text-2xl mb-2 opacity-50"></i>
                <span class="text-xs">No classification history yet</span>
            </div>
        `;
        return;
    }
    
    let html = '';
    classificationHistory.forEach(item => {
        const style = categoryStyles[item.category] || categoryStyles.unknown;
        const conf = Math.min(item.confidence || 0, 100).toFixed(1);
        const date = new Date(item.timestamp).toLocaleString(undefined, {
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
        });
        
        let iconClass = 'fa-file';
        if (item.inputType === 'text') iconClass = 'fa-keyboard';
        else if (item.inputType === 'image') iconClass = 'fa-image';
        else if (item.inputType === 'audio') iconClass = 'fa-microphone';
        else if (item.inputType === 'video') iconClass = 'fa-video';
        
        html += `
            <div class="flex flex-col p-4 rounded-xl bg-dark-800 border border-white/5 hover:border-white/10 transition-colors gap-3 relative overflow-hidden group">
                <div class="absolute top-0 left-0 w-1 h-full" style="background-color: ${style.color}"></div>
                <div class="flex items-center justify-between">
                    <div class="flex items-center gap-3">
                        <div class="w-8 h-8 rounded-lg flex items-center justify-center text-sm" style="background-color: ${style.color}22; color: ${style.color};">
                            <i class="fas ${style.icon}"></i>
                        </div>
                        <span class="text-white font-bold text-sm tracking-wide uppercase">${utils.capitalizeFirst(item.category_display)}</span>
                    </div>
                    <span class="text-xs font-bold px-2 py-1 rounded bg-dark-950/50" style="color: ${style.color}">${conf}%</span>
                </div>
                <div class="flex flex-col mt-1">
                    <span class="text-xs text-gray-500 line-clamp-2 leading-relaxed"><i class="fas ${iconClass} mr-1"></i> ${item.inputText || 'Media File'}</span>
                    <span class="text-[10px] text-gray-600 font-medium mt-2">${date}</span>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

// Side Panel Toggle Logic
function toggleHistorySidebar(show) {
    const sidebar = document.getElementById('history-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if(!sidebar || !overlay) return;

    if (show) {
        sidebar.classList.remove('translate-x-full');
        overlay.classList.remove('hidden');
        // Small delay for transition
        setTimeout(() => overlay.classList.remove('opacity-0'), 10);
        document.body.style.overflow = 'hidden'; // Prevent background scrolling
    } else {
        sidebar.classList.add('translate-x-full');
        overlay.classList.add('opacity-0');
        setTimeout(() => overlay.classList.add('hidden'), 500); // Wait for transition
        document.body.style.overflow = '';
    }
}

window.clearHistory = function() {
    if (confirm('Are you sure you want to clear your classification history?')) {
        classificationHistory = [];
        saveHistory();
        renderHistory();
    }
};

// =============================================================================
// INITIALIZATION
// =============================================================================
function initEventListeners() {
    // Expose setInputState for inline tab switching in HTML
    window.setInputState = (mode) => uiManager.switchInputType(mode);

    // Text input
    const textarea = document.getElementById('text-input');
    if (textarea) {
        textarea.addEventListener('input', () => uiManager.updateCharCount());
    }

    // Tab switching
    document.querySelectorAll('.input-type-btn').forEach(btn => {
        btn.addEventListener('click', () => uiManager.switchInputType(btn.dataset.type));
    });

    // History Sidebar Toggles
    const openBtn = document.getElementById('history-toggle-btn');
    const closeBtn = document.getElementById('close-history-btn');
    const overlay = document.getElementById('sidebar-overlay');

    if (openBtn) openBtn.addEventListener('click', () => toggleHistorySidebar(true));
    if (closeBtn) closeBtn.addEventListener('click', () => toggleHistorySidebar(false));
    if (overlay) overlay.addEventListener('click', () => toggleHistorySidebar(false));

    // Analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeContent);
    }

    // File inputs
    ['image', 'audio', 'video'].forEach(type => {
        const input = document.getElementById(`${type}-upload`);
        if (input) {
            console.log(`[Init] Attaching change listener to ${type}-upload`);
            input.addEventListener('change', (e) => eventHandlers.handleFileSelect(e, type));
        } else {
            console.error(`[Init] Input element not found: ${type}-upload`);
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
    fetchRealtimeNews,
    removeFile: (type) => uiManager.removeFile(type),
    switchInputType: (type) => uiManager.switchInputType(type),
    getCacheStats: () => requestCache.getStats(),
    api,
    utils
};
