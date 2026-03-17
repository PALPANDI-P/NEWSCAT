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
    ANIMATION_DURATION: 300,
    MAX_HISTORY: 50,
    HISTORY_KEY: 'newscat_history'
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
    fileAbortController: null,  // For aborting file uploads
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
        // Use the abort controller from state for cancellation support
        const textAbortController = state.abortController || new AbortController();
        state.abortController = textAbortController;
        
        // Set timeout for text requests (2 minutes)
        const timeoutId = setTimeout(() => {
            console.log('[API classifyText] Timeout reached, aborting request');
            textAbortController.abort();
        }, 120000); // 2 minutes
        
        try {
            const result = await this.request('/classify', {
                method: 'POST',
                body: JSON.stringify({ text: text.trim(), enhanced }),
                signal: textAbortController.signal
            });
            clearTimeout(timeoutId);
            return result;
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
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
        // Store in state so cancel button can access it
        const fileAbortController = new AbortController();
        state.fileAbortController = fileAbortController;

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
// LOADING MANAGER - Premium AI SaaS Style
// =============================================================================
const loadingManager = {
    overlay: null,
    titleEl: null,
    subtitleEl: null,
    stepEl: null,
    progressEl: null,
    dynamicMsgEl: null,
    stepDots: null,
    _interval: null,
    _stepIndex: 0,
    _isVisible: false,
    _initialized: false,

    // Mode-specific loading steps
    _modeSteps: {
        text: [
            { title: 'Analyzing Text', sub: 'Processing text content...', step: 'Step 1/4', message: 'Analyzing text content...' },
            { title: 'Classifying Content', sub: 'Running classification model...', step: 'Step 2/4', message: 'Classifying content...' },
            { title: 'Scoring Results', sub: 'Computing confidence scores...', step: 'Step 3/4', message: 'Scoring results...' },
            { title: 'Complete', sub: 'Finalizing results...', step: 'Step 4/4', message: 'Finalizing...' }
        ],
        image: [
            { title: 'Extracting Features', sub: 'Processing image features...', step: 'Step 1/4', message: 'Extracting image features...' },
            { title: 'Analyzing Content', sub: 'Analyzing visual data...', step: 'Step 2/4', message: 'Analyzing content...' },
            { title: 'Classifying Image', sub: 'Running classification model...', step: 'Step 3/4', message: 'Classifying image...' },
            { title: 'Complete', sub: 'Finalizing results...', step: 'Step 4/4', message: 'Finalizing...' }
        ],
        audio: [
            { title: 'Extracting Audio', sub: 'Processing audio data...', step: 'Step 1/4', message: 'Extracting audio...' },
            { title: 'Processing Audio', sub: 'Analyzing audio features...', step: 'Step 2/4', message: 'Processing audio...' },
            { title: 'Classifying Audio', sub: 'Running classification model...', step: 'Step 3/4', message: 'Classifying audio...' },
            { title: 'Complete', sub: 'Finalizing results...', step: 'Step 4/4', message: 'Finalizing...' }
        ],
        video: [
            { title: 'Extracting Frames', sub: 'Processing video frames...', step: 'Step 1/4', message: 'Extracting frames...' },
            { title: 'Analyzing Video', sub: 'Analyzing visual data...', step: 'Step 2/4', message: 'Analyzing video...' },
            { title: 'Classifying Video', sub: 'Running classification model...', step: 'Step 3/4', message: 'Classifying video...' },
            { title: 'Complete', sub: 'Finalizing results...', step: 'Step 4/4', message: 'Finalizing...' }
        ]
    },

    // Default text steps for backward compatibility
    _steps: [
        { 
            title: 'Running Text Analysis', 
            sub: 'Processing text content with ML classifier...', 
            step: 'Step 1/4',
            message: 'Running text classification model...'
        },
        { 
            title: 'Running Audio Analysis', 
            sub: 'Processing audio content...', 
            step: 'Step 2/4',
            message: 'Running audio classification model...'
        },
        { 
            title: 'Running Image Analysis', 
            sub: 'Processing image content...', 
            step: 'Step 3/4',
            message: 'Running image classification model...'
        },
        { 
            title: 'Running Video Analysis', 
            sub: 'Processing video content...', 
            step: 'Step 4/4',
            message: 'Running video classification model...'
        }
    ],

    // Additional cycling messages for variety
    _cyclingMessages: [
        'Running text classification...',
        'Running audio classification...',
        'Running image classification...',
        'Running video classification...',
        'Analyzing content patterns...',
        'Evaluating category scores...',
        'Computing confidence metrics...',
        'Merging model results...',
        'Applying ensemble voting...',
        'Finalizing classification results...'
    ],

    init() {
        // Prevent multiple initializations
        if (this._initialized) {
            // Re-fetch elements in case DOM was modified
            this.overlay = document.getElementById('loading-overlay');
            this.dynamicMsgEl = this.overlay ? this.overlay.querySelector('.loading-text') : null;
            this.subTextEl = this.overlay ? this.overlay.querySelector('.loading-subtext') : null;
            this.progressEl = this.overlay ? this.overlay.querySelector('.loading-progress') : null;
            this.stepEl = this.overlay ? this.overlay.querySelector('#loading-step') : null;
            this.stepDots = this.overlay ? this.overlay.querySelectorAll('.step-dot') : [];
            return;
        }
        
        this.overlay = document.getElementById('loading-overlay');
        this.dynamicMsgEl = this.overlay ? this.overlay.querySelector('.loading-text') : null;
        this.subTextEl = this.overlay ? this.overlay.querySelector('.loading-subtext') : null;
        this.progressEl = this.overlay ? this.overlay.querySelector('.loading-progress') : null;
        this.stepEl = this.overlay ? this.overlay.querySelector('#loading-step') : null;
        // Get all step dots
        this.stepDots = this.overlay ? this.overlay.querySelectorAll('.step-dot') : [];
        
        this._initialized = true;
    },

    // Update the step dots to reflect current progress
    _updateStepDots(step) {
        if (!this.stepDots || this.stepDots.length === 0) return;
        
        this.stepDots.forEach((dot, index) => {
            const dotStep = index + 1;
            if (dotStep <= step) {
                dot.classList.add('active');
                dot.classList.remove('bg-dark-700');
                dot.classList.add('bg-indigo-500');
            } else {
                dot.classList.remove('active');
                dot.classList.remove('bg-indigo-500');
                dot.classList.add('bg-dark-700');
            }
        });
        
        // Update step text
        if (this.stepEl) {
            this.stepEl.textContent = `Step ${step}/4`;
        }
    },

    show(title, subTitle, inputType = 'text') {
        // Ensure DOM elements are initialized before use
        if (!this._initialized) this.init();
        this._stepIndex = 0;
        this._isVisible = true;
        
        // Get mode-specific steps and messages
        const modeKey = inputType && this._modeSteps[inputType] ? inputType : 'text';
        const currentModeSteps = this._modeSteps[modeKey];

        if (title && this.dynamicMsgEl) {
            this.dynamicMsgEl.textContent = title;
        } else if (currentModeSteps && currentModeSteps[0] && this.dynamicMsgEl) {
            // Use mode-specific title if no custom title provided
            this.dynamicMsgEl.textContent = currentModeSteps[0].message;
        }
        if (subTitle && this.subTextEl) {
            this.subTextEl.textContent = subTitle;
        } else if (currentModeSteps && currentModeSteps[0] && this.subTextEl) {
            // Use mode-specific subtitle
            this.subTextEl.textContent = currentModeSteps[0].sub;
        }

        if (this.overlay) {
            this.overlay.style.display = 'flex';
            this.overlay.style.flexDirection = 'column';
            this.overlay.style.alignItems = 'center';
            this.overlay.style.justifyContent = 'center';
        }
        
        // Initialize step dots to step 1
        this._updateStepDots(1);
        
        // Reset progress with animation
        if (this.progressEl) {
            this.progressEl.style.width = '0%';
            setTimeout(() => {
                if(this.progressEl) this.progressEl.style.width = '80%';
            }, 100);
        }
        
        document.body.style.overflow = 'hidden';
        
        // Cycle through messages and update step dots based on elapsed time
        let messageIndex = 0;
        this._messageInterval = setInterval(() => {
            if (!this._isVisible) return;
            
            // Update step dots every 900ms (cycling through 4 steps in ~3.6s) - slower for better UX
            messageIndex++;
            const currentStep = Math.min(messageIndex, 4);
            this._updateStepDots(currentStep);
            
            if (this.dynamicMsgEl) {
                // Use mode-specific cycling messages if available
                const modeMsgIndex = (messageIndex - 1) % currentModeSteps.length;
                const modeMessage = currentModeSteps[modeMsgIndex]?.message;
                const randomMsg = this._cyclingMessages[Math.floor(Math.random() * this._cyclingMessages.length)];
                // Prefer mode-specific message, fall back to cycling messages
                this._animateMessageChange(this.dynamicMsgEl, modeMessage || randomMsg);
            }
        }, 900);
    },

    hide() {
        this._isVisible = false;
        if (this._messageInterval) {
            clearInterval(this._messageInterval);
            this._messageInterval = null;
        }
        
        if (this.progressEl) {
            this.progressEl.style.width = '100%';
        }
        
        // Reset step dots to all active (complete)
        this._updateStepDots(4);
        
        setTimeout(() => {
            if (this.overlay) this.overlay.style.display = 'none';
            document.body.style.overflow = '';
            // Reset dots for next time
            this._updateStepDots(0);
        }, 300); // Wait for progress bar to finish CSS transition
    },

    _animateMessageChange(element, newText) {
        element.style.opacity = '0';
        element.style.transition = 'opacity 0.2s ease';
        setTimeout(() => {
            element.textContent = newText;
            element.style.opacity = '1';
        }, 200);
    },

    update(message, subMessage = null) {
        if (message && this.dynamicMsgEl) {
            this._animateMessageChange(this.dynamicMsgEl, message);
        }
        if (subMessage && this.subTextEl) {
             this._animateMessageChange(this.subTextEl, subMessage);
        }
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
        
        // Hide the How It Works section when results are shown
        const howItWorks = document.getElementById('how-it-works-section');
        if (howItWorks) howItWorks.style.display = 'none';
        
        // Apply clean horizontal layout consistently
        resultsEl.classList.add('result-horizontal-layout');
        
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
        
        const coreTopicEl = document.getElementById('result-core-topic');
        const mainTopicsListEl = document.getElementById('result-main-topics-list');
        
        const coreTopic = data.subtopic || data.category || 'General';
        const mainTopics = data.main_topics || [data.main_topic || data.category || 'General'];
        
        if (coreTopicEl) {
            // Style: Matching the professional blue gradient from the image
            const userImageBlue = 'linear-gradient(135deg, #4A90E2 0%, #357ABD 100%)';
            coreTopicEl.innerHTML = `<span style="background: ${userImageBlue}; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; filter: drop-shadow(0 0 10px rgba(74, 144, 226, 0.3));">${utils.capitalizeFirst(coreTopic.replace(/_/g, ' '))}</span>`;
        }
        
        const mainTopicTextEl = document.getElementById('result-main-topic-text');
        const mainTopicIconEl = document.getElementById('result-main-topic-icon');
        
        if (mainTopicTextEl) {
            // Only show the single top main topic as requested
            const topMainTopic = Array.isArray(mainTopics) ? mainTopics[0] : mainTopics.split(',')[0].trim();
            mainTopicTextEl.textContent = utils.capitalizeFirst(topMainTopic.replace(/_/g, ' '));
            
            // Set a contextual icon for the main topic if available, otherwise use default
            if (mainTopicIconEl) {
                // If the main topic matches a known category, use its icon
                const mainTopicKey = topMainTopic.toLowerCase().replace(/ /g, '_');
                const mainTopicStyle = categoryStyles[mainTopicKey];
                if (mainTopicStyle && mainTopicStyle.icon) {
                    mainTopicIconEl.className = `fas ${mainTopicStyle.icon} text-[#f6b93b] text-sm shadow-glow`;
                } else {
                    mainTopicIconEl.className = 'fas fa-dot-circle text-[#f6b93b] text-sm shadow-glow';
                }
            }
        }
        
        // 2. Confidence Circle Update
        const confidence = Math.min((data.confidence || 0), 100);
        const confidenceCircle = document.getElementById('confidence-circle');
        const confidenceText = document.getElementById('confidence-text');
        
        // Updated circumference for r=20: 2 * PI * 20 = 125.6
        const circumference = 2 * Math.PI * 20;
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
        // Only show text analysis for text input type
        const panel = document.getElementById('text-analysis-panel');
        if (!panel) return;
        
        if (state.currentInputType === 'text') {
            panel.classList.remove('hidden');
            // Use the current textarea value since the backend doesn't return original_text
            const textarea = document.getElementById('text-input');
            const inputText = textarea ? textarea.value : '';
            const wordsMatch = inputText ? inputText.match(/\b\w+\b/g) : [];
            const words = wordsMatch ? wordsMatch.length : 0;
            const chars = inputText ? Math.min(inputText.length, 10000) : 0;
            const sentences = inputText ? Math.max(1, (inputText.match(/[.!?]+/g) || []).length) : 0;
            const readability = Math.min(20, Math.max(1, Math.round(words / Math.max(1, sentences) * 0.5))); // Simplified readability
            
            this.animateValue('analysis-words', 0, words, 1000);
            this.animateValue('analysis-chars', 0, chars, 1000);
            this.animateValue('analysis-sentences', 0, sentences, 1000);
            this.animateValue('analysis-readability', 0, readability, 1000);
        } else {
            panel.classList.add('hidden'); // Hide text analysis metrics if not text mode
        }
        
        // Update parallel model results
        this.updateParallelModelResults(data);
    },
    
    updateParallelModelResults(data) {
        const panel = document.getElementById('parallel-models-panel');
        if (!panel) return;
        
        // Get model_results from the API response
        const modelResults = data.model_results || {};
        
        // Update each model's result
        const models = ['text', 'audio', 'image', 'video'];
        models.forEach(model => {
            const modelData = modelResults[model];
            const categoryEl = document.getElementById(`model-${model}-category`);
            const confidenceEl = document.getElementById(`model-${model}-confidence`);
            
            if (categoryEl && confidenceEl) {
                if (modelData && modelData.success) {
                    categoryEl.textContent = this.capitalizeFirst(modelData.primary_category || '-');
                    const confidence = (modelData.confidence || 0) * 100;
                    confidenceEl.textContent = `${confidence.toFixed(0)}%`;
                    
                    // Color based on confidence
                    if (confidence >= 70) {
                        confidenceEl.className = 'text-xs text-green-400';
                    } else if (confidence >= 40) {
                        confidenceEl.className = 'text-xs text-yellow-400';
                    } else {
                        confidenceEl.className = 'text-xs text-red-400';
                    }
                } else {
                    categoryEl.textContent = 'Failed';
                    confidenceEl.textContent = '0%';
                    confidenceEl.className = 'text-xs text-gray-500';
                }
            }
        });
    },
    
    capitalizeFirst(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();
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
        
        // Show the How It Works section when returning to landing
        const howItWorks = document.getElementById('how-it-works-section');
        if (howItWorks) howItWorks.style.display = '';

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

    loadingManager.show('Analyzing Data', 'Ensemble ML models processing content...', state.currentInputType);

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

        // Handle both wrapped ({status: 'success', data: ...}) and direct responses
        let classificationData = null;
        if (result?.status === 'success' && result?.data) {
            classificationData = result.data;
        } else if (result?.status === 'success') {
            // Direct result without data wrapper
            classificationData = result;
        } else if (result?.category) {
            // Result is directly the classification data
            classificationData = result;
        } else {
            throw new Error(result?.message || 'Server returned an error');
        }
        
        // Display results and save to history
        resultsManager.display(classificationData, inputText, state.currentInputType);
        loadingManager.hide();
        addToHistory(classificationData, inputText, state.currentInputType);
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
    const mainTopic = result.main_topic || result.category || 'unknown';
    const subtopic = result.subtopic || result.category || 'unknown';

    const entry = {
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        date: new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        }),
        time: new Date().toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        }),
        category: result.category_display || result.category || 'unknown',
        main_topic: utils.capitalizeFirst(mainTopic.replace(/_/g, ' ')),
        subtopic: utils.capitalizeFirst(subtopic.replace(/_/g, ' ')),
        confidence: result.confidence,
        inputType,
        // Store complete content text for text classifications
        fullContent: inputType === 'text' ? inputText : (inputText || ''),
        inputPreview: inputType === 'text' ? inputText.substring(0, 80) + '...' : inputText,
        // Store additional metadata
        modelName: result.model_name || 'Ensemble Classifier',
        processingTime: result.processing_time_ms || 0
    };

    if (!state.classificationHistory || !Array.isArray(state.classificationHistory)) {
        state.classificationHistory = [];
    }
    
    state.classificationHistory.unshift(entry);
    
    // Hardcoded max history check
    if (state.classificationHistory.length > CONFIG.MAX_HISTORY) {
        state.classificationHistory = state.classificationHistory.slice(0, CONFIG.MAX_HISTORY);
    }

    try {
        const serialized = JSON.stringify(state.classificationHistory);
        localStorage.setItem(CONFIG.HISTORY_KEY, serialized);
        
        // Verification step
        const verify = localStorage.getItem(CONFIG.HISTORY_KEY);
        if (!verify || verify === 'null') throw new Error("History storage write failed silently.");
        
        renderHistory();
    } catch (e) {
        console.warn('Failed to save history securely, storage may be full:', e);
        // Recovery mechanism: shrink history aggressive
        try {
            state.classificationHistory = state.classificationHistory.slice(0, 10);
            localStorage.setItem(CONFIG.HISTORY_KEY, JSON.stringify(state.classificationHistory));
        } catch(fallbackErr) {
            console.error('Critical History Storage Failure:', fallbackErr);
        }
    }
}

function loadHistory() {
    try {
        const saved = localStorage.getItem(CONFIG.HISTORY_KEY);
        if (saved && saved !== 'null') {
            const parsed = JSON.parse(saved);
            // Defensive typing check constraint
            if (Array.isArray(parsed)) {
                // Filter out broken objects
                state.classificationHistory = parsed.filter(item => item && item.id && item.category);
            } else {
                state.classificationHistory = [];
            }
        } else {
            state.classificationHistory = [];
        }
        renderHistory();
    } catch (e) {
        console.warn('Failed to load history, state corrupted. Resetting:', e);
        state.classificationHistory = [];
        localStorage.removeItem(CONFIG.HISTORY_KEY); // Clean up bad data completely
        renderHistory();
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
                <p class="text-dark-400 font-medium">No classifications yet. Start by classifying some content!</p>
            </div>
        `;
        return;
    }

    let html = '<div class="flex flex-col gap-3">';
    state.classificationHistory.forEach(item => {
        // Use stored date and time if available, otherwise parse from timestamp
        const displayDate = item.date || (item.timestamp ? new Date(item.timestamp).toLocaleDateString(undefined, { 
            month: 'short', day: 'numeric', year: 'numeric' 
        }) : '');
        const displayTime = item.time || (item.timestamp ? new Date(item.timestamp).toLocaleTimeString(undefined, {
            hour: '2-digit', minute: '2-digit' 
        }) : '');
        
        let iconHtml = '';
        switch(item.inputType) {
            case 'text': iconHtml = '<i class="fas fa-keyboard text-brand-purple"></i>'; break;
            case 'image': iconHtml = '<i class="fas fa-image text-blue-400"></i>'; break;
            case 'audio': iconHtml = '<i class="fas fa-music text-green-400"></i>'; break;
            case 'video': iconHtml = '<i class="fas fa-video text-red-400"></i>'; break;
            default: iconHtml = '<i class="fas fa-file text-gray-400"></i>';
        }

        const mainTopicBadge = item.main_topic ? `<span class="text-[9px] font-bold text-indigo-400 bg-indigo-500/10 px-1.5 py-0.5 rounded-full border border-indigo-500/20">${utils.sanitizeText(item.main_topic)}</span>` : '';
        const subtopicBadge = item.subtopic ? `<span class="text-[9px] font-bold text-purple-400 bg-purple-500/10 px-1.5 py-0.5 rounded-full border border-purple-500/20">${utils.sanitizeText(item.subtopic)}</span>` : '';

        // Confidence color based on level
        const confidenceColor = item.confidence >= 80 ? 'text-green-400' : (item.confidence >= 60 ? 'text-yellow-400' : 'text-orange-400');

        html += `
            <div class="history-item flex flex-col p-4 rounded-xl bg-dark-900/50 hover:bg-dark-800 border border-dark-800 hover:border-brand-purple/30 transition-all duration-300">
                <div class="flex items-start gap-3">
                    <div class="mt-1 bg-dark-950 p-2 rounded-lg border border-white/5 flex-shrink-0 shadow-inner">
                        ${iconHtml}
                    </div>
                    <div class="min-w-0 flex-1">
                        <div class="flex items-center justify-between mb-1">
                            <h4 class="text-white font-bold text-sm tracking-wide capitalize truncate pr-2">${utils.sanitizeText(item.subtopic)}</h4>
                            <span class="text-xs font-bold ${confidenceColor} whitespace-nowrap">${Number(item.confidence).toFixed(1)}%</span>
                        </div>
                        <p class="text-gray-400 text-xs line-clamp-2 leading-relaxed italic mb-2" title="${utils.sanitizeText(item.fullContent || item.inputPreview).replace(/"/g, '&quot;')}">
                            "${utils.sanitizeText(item.inputPreview)}"
                        </p>
                        <div class="flex items-center justify-between border-t border-white/5 pt-2 mt-1">
                            <p class="text-gray-500 text-[10px] flex items-center gap-2 font-medium">
                                <span><i class="far fa-calendar-alt mr-1"></i>${displayDate}</span>
                                <span><i class="far fa-clock mr-1"></i>${displayTime}</span>
                            </p>
                            <div class="flex items-center">
                                ${mainTopicBadge}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    html += '</div>';
    
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
    }, 1); // 1ms per character for snappy loading
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

// (Duplicate history functions removed — unified into single addToHistory/renderHistory above)

// =============================================================================
// HISTORY UI FUNCTIONS
// =============================================================================
function toggleHistorySidebar(show) {
    const sidebar = document.getElementById('history-sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (!sidebar || !overlay) return;

    if (show) {
        sidebar.classList.remove('translate-x-full');
        sidebar.classList.add('translate-x-0');
        overlay.classList.remove('hidden');
        // Small delay for fade in
        setTimeout(() => overlay.classList.add('opacity-100'), 10);
    } else {
        sidebar.classList.remove('translate-x-0');
        sidebar.classList.add('translate-x-full');
        overlay.classList.remove('opacity-100');
        setTimeout(() => overlay.classList.add('hidden'), 500); // Wait for transition
    }
}

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

    // Cancel button
    const cancelBtn = document.getElementById('cancel-classify-btn');
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            // Abort text classification request
            if (state.abortController) {
                state.abortController.abort();
                state.abortController = null;
            }
            // Abort file upload requests (image, audio, video)
            if (state.fileAbortController) {
                state.fileAbortController.abort();
                state.fileAbortController = null;
            }
            // Hide loading
            loadingManager.hide();
            // Show notification
            notificationManager.show('Classification cancelled', 'info');
            console.log('[Cancel] Classification cancelled by user');
        });
    }

    // Analyze button
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', (e) => {
            // Create ripple effect
            const ripple = document.createElement('span');
            ripple.classList.add('ripple');
            const rect = analyzeBtn.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
            ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
            analyzeBtn.appendChild(ripple);
            setTimeout(() => ripple.remove(), 600);
            analyzeContent();
        });
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

function loadCategories() {
    const list = document.getElementById('categories-list');
    if (!list) return;

    // Group categories by their color theme for visual organization - Enhanced with more categories
    const colorGroups = {
        'Technology': { categories: ['technology', 'artificial_intelligence', 'cybersecurity', 'blockchain_tech', 'internet_of_things', 'cloud_computing', 'software_development', 'robotics', 'virtual_reality', 'data_science'], color: '#3b82f6' },
        'Business & Finance': { categories: ['business', 'finance', 'economy', 'cryptocurrency', 'real_estate', 'banking', 'investments', 'startups', 'marketing', 'ecommerce'], color: '#4f46e5' },
        'Science & Environment': { categories: ['science', 'space', 'biology', 'chemistry', 'physics', 'genetics', 'climate_change', 'environment', 'astronomy'], color: '#8b5cf6' },
        'Health': { categories: ['health', 'medicine', 'mental_health', 'fitness', 'nutrition', 'public_health'], color: '#2dd4bf' },
        'Politics': { categories: ['politics', 'elections', 'geopolitics', 'international_relations', 'public_policy', 'law_justice'], color: '#64748b' },
        'Sports': { categories: ['sports', 'football_soccer', 'basketball', 'tennis', 'golf', 'motorsports', 'cricket'], color: '#0284c7' },
        'Entertainment': { categories: ['entertainment', 'film_tv', 'music', 'celebrity', 'video_games', 'streaming', 'fashion'], color: '#7c3aed' },
        'Lifestyle': { categories: ['lifestyle', 'travel', 'food_dining', 'education', 'relationships', 'automotive', 'beauty'], color: '#0ea5e9' }
    };

    let html = '<div class="categories-container flex flex-col gap-3">';
    let count = 0;
    
    for (const [groupName, groupData] of Object.entries(colorGroups)) {
        const groupColor = groupData.color;
        
        html += `
            <div class="taxonomy-group rounded-xl overflow-hidden border border-white/5">
                <div class="taxonomy-header px-4 py-2.5 flex items-center justify-between" style="background: linear-gradient(135deg, ${groupColor}20, ${groupColor}08);">
                    <div class="flex items-center gap-2">
                        <div class="w-2.5 h-2.5 rounded-full" style="background: ${groupColor}; box-shadow: 0 0 6px ${groupColor}60;"></div>
                        <span class="text-[11px] font-extrabold uppercase tracking-[0.12em]" style="color: ${groupColor}">${groupName}</span>
                    </div>
                    <span class="text-[9px] font-bold px-2 py-0.5 rounded-full" style="color: ${groupColor}; background: ${groupColor}10; border: 1px solid ${groupColor}20;">${groupData.categories.length}</span>
                </div>
                <div class="grid grid-cols-2 gap-1.5 p-2.5">`;
        
        groupData.categories.forEach(key => {
            const style = categoryStyles[key];
            if (!style) return;
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            html += `
                <div class="taxonomy-chip flex items-center gap-2 px-3 py-2 rounded-lg text-[10.5px] font-semibold transition-all duration-200 cursor-pointer group" style="background: ${groupColor}08; border: 1px solid ${groupColor}12; color: ${groupColor};" data-category="${key}">
                    <i class="fas ${style.icon} w-3.5 text-center text-[9px] opacity-70 group-hover:opacity-100 transition-opacity" style="color: ${groupColor}"></i>
                    <span class="truncate leading-tight" style="color: var(--chip-text, #94a3b8);">${label}</span>
                </div>`;
            count++;
        });
        
        html += '</div></div>';
    }
    
    html += '</div>';
    list.innerHTML = html;

    // Update counter badge
    const badge = document.getElementById('categories-count-badge');
    if (badge) badge.textContent = `${count} Topics`;
}

function initThemeToggle() {
    const themeBtn = document.getElementById('theme-toggle-btn');
    const themeIcon = document.getElementById('theme-icon');
    if (!themeBtn || !themeIcon) return;
    
    // Check saved preference
    const isLightMode = localStorage.getItem('newscat_light_theme') === 'true';
    if (isLightMode) {
        document.body.classList.add('light-theme');
        themeIcon.className = 'fas fa-sun';
    }
    
    themeBtn.addEventListener('click', () => {
        const isLight = document.body.classList.toggle('light-theme');
        localStorage.setItem('newscat_light_theme', isLight);
        themeIcon.className = isLight ? 'fas fa-sun' : 'fas fa-moon';
        
        // Visual feedback
        const msg = isLight ? 'Light Theme Enabled' : 'Dark Theme Enabled';
        notificationManager.show(msg, 'success');
    });
}

function initializeApp() {
    // Initialize systems
    loadingManager.init();

    loadHistory();
    loadCategories();
    initEventListeners();
    initThemeToggle();
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

// =============================================================================
// ENHANCED FUNCTIONALITIES
// =============================================================================

// Auto-refresh session every 30 minutes
let sessionRefreshInterval;

function startSessionRefresh() {
    sessionRefreshInterval = setInterval(() => {
        const sessionLocal = localStorage.getItem('newsauth_session');
        const sessionSession = sessionStorage.getItem('newsauth_session');
        const session = sessionLocal || sessionSession;
        
        if (session) {
            try {
                const sessionData = JSON.parse(session);
                // Update timestamp to keep session alive
                sessionData.timestamp = Date.now();
                
                if (sessionLocal) {
                    localStorage.setItem('newsauth_session', JSON.stringify(sessionData));
                } else {
                    sessionStorage.setItem('newsauth_session', JSON.stringify(sessionData));
                }
                console.log('[Session] Auto-refreshed');
            } catch (e) {
                console.warn('[Session] Refresh failed');
            }
        }
    }, 30 * 60 * 1000); // 30 minutes
}

// Activity tracking for auto-logout warning
let activityTimeout;
const INACTIVITY_WARNING = 20 * 60 * 1000; // 20 minutes
const INACTIVITY_LOGOUT = 25 * 60 * 1000; // 25 minutes

function resetActivityTimer() {
    clearTimeout(activityTimeout);
    
    // Show warning at 20 minutes
    activityTimeout = setTimeout(() => {
        showInactivityWarning();
    }, INACTIVITY_WARNING);
}

function showInactivityWarning() {
    notificationManager.show('Your session will expire in 5 minutes due to inactivity', 'warning');
    
    // Auto logout at 25 minutes
    setTimeout(() => {
        notificationManager.show('Session expired due to inactivity', 'error');
        setTimeout(logout, 2000);
    }, 5 * 60 * 1000);
}

// Track user activity
['mousedown', 'keydown', 'scroll', 'touchstart'].forEach(event => {
    document.addEventListener(event, resetActivityTimer, { passive: true });
});

// Initialize session management when auth is confirmed
function initSessionManagement() {
    startSessionRefresh();
    resetActivityTimer();
}

// Enhanced keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + L: Logout
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        if (confirm('Are you sure you want to logout?')) {
            logout();
        }
    }
    
    // Escape: Close modals/sidebars
    if (e.key === 'Escape') {
        const historySidebar = document.getElementById('history-sidebar');
        if (historySidebar && historySidebar.classList.contains('translate-x-0')) {
            toggleHistorySidebar(false);
        }
    }
});

// Auto-initialize session management
if (checkAuth()) {
    initSessionManagement();
}
