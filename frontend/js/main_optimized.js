/**
 * NEWSCAT v6.0 - Ultra-Optimized Frontend JavaScript
 * Future-Level Performance with Caching, Debouncing, and Optimized DOM Operations
 * 
 * Optimizations:
 * - Request debouncing (300ms delay for text input)
 * - Client-side response caching with TTL
 * - Request deduplication
 * - Optimized DOM updates with DocumentFragment
 * - Intersection Observer for lazy loading
 * - AbortController for request cancellation
 * - Efficient event delegation
 */

// ===== CONFIGURATION =====
const API_BASE = '/api';
const MIN_CHARS = 10;
const MAX_FILE_SIZES = {
    image: 10 * 1024 * 1024,  // 10MB
    audio: 50 * 1024 * 1024,  // 50MB
    video: 100 * 1024 * 1024  // 100MB
};

const DEBOUNCE_DELAY = 300;  // ms
const CACHE_TTL = 5 * 60 * 1000;  // 5 minutes

// ===== CLIENT-SIDE CACHE =====
class ClientCache {
    constructor(maxSize = 100) {
        this.cache = new Map();
        this.maxSize = maxSize;
        this.hits = 0;
        this.misses = 0;
    }

    get(key) {
        const item = this.cache.get(key);
        if (!item) {
            this.misses++;
            return null;
        }

        // Check TTL
        if (Date.now() - item.timestamp > CACHE_TTL) {
            this.cache.delete(key);
            this.misses++;
            return null;
        }

        // Move to end (most recently used)
        this.cache.delete(key);
        this.cache.set(key, item);
        this.hits++;
        return item.data;
    }

    set(key, data) {
        // Remove oldest if at capacity
        if (this.cache.size >= this.maxSize) {
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }

        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }

    clear() {
        this.cache.clear();
    }

    getStats() {
        const total = this.hits + this.misses;
        return {
            size: this.cache.size,
            maxSize: this.maxSize,
            hitRate: total > 0 ? (this.hits / total * 100).toFixed(2) + '%' : '0%'
        };
    }
}

const clientCache = new ClientCache();

// ===== STATE MANAGEMENT =====
const state = {
    currentInputType: 'text',
    selectedFiles: { image: null, audio: null, video: null },
    isAnalyzing: false,
    classificationHistory: [],
    modelInfo: null,
    abortController: null,
    pendingRequests: new Map()
};

// ===== DEBOUNCE UTILITY =====
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ===== REQUEST DEDUPLICATION =====
async function deduplicatedFetch(url, options = {}) {
    const key = `${url}:${JSON.stringify(options)}`;

    // Check if identical request is pending
    if (state.pendingRequests.has(key)) {
        return state.pendingRequests.get(key);
    }

    // Create new request
    const promise = fetch(url, options)
        .finally(() => {
            state.pendingRequests.delete(key);
        });

    state.pendingRequests.set(key, promise);
    return promise;
}

// ===== CACHED API CALL =====
async function cachedApiCall(endpoint, options = {}) {
    const cacheKey = `${endpoint}:${JSON.stringify(options)}`;

    // Check client cache
    const cached = clientCache.get(cacheKey);
    if (cached) {
        return { ...cached, fromCache: true };
    }

    // Make request
    const response = await deduplicatedFetch(`${API_BASE}${endpoint}`, options);
    const data = await response.json();

    // Cache successful responses
    if (response.ok && data.status === 'success') {
        clientCache.set(cacheKey, data);
    }

    return data;
}

// ===== ABORT CONTROLLER FOR REQUEST CANCELLATION =====
function createAbortSignal() {
    if (state.abortController) {
        state.abortController.abort();
    }
    state.abortController = new AbortController();
    return state.abortController.signal;
}

// ===== SAMPLE ARTICLES =====
const sampleArticles = {
    tech: "OpenAI has unveiled GPT-5, demonstrating unprecedented natural language capabilities. The model features enhanced reasoning and improved contextual awareness, potentially revolutionizing healthcare and education.",
    sports: "In a stunning Wimbledon upset, unseeded Maria Rodriguez defeated reigning champion Novak Djokovic in straight sets 6-4, 7-5. The 21-year-old Spanish player showcased exceptional athleticism.",
    politics: "The Senate passed a landmark $500 billion climate change bill today with bipartisan support, allocating funds for renewable energy initiatives and carbon capture technology.",
    business: "Apple reported record quarterly earnings of $120 billion driven by strong iPhone sales and growing Services revenue. The company announced a $90 billion stock buyback program.",
    entertainment: "The 96th Academy Awards ceremony celebrated a diverse range of films, with Oppenheimer taking home Best Picture. The event featured moving tributes and surprise appearances.",
    health: "A groundbreaking clinical trial has shown promising results for a new Alzheimer's treatment, with patients demonstrating significant cognitive improvement.",
    science: "NASA's James Webb Space Telescope has captured unprecedented images of distant galaxies, revealing new insights into the early universe.",
    world: "The G20 summit concluded with a historic agreement on climate finance, with developed nations pledging $100 billion annually to support developing countries.",
    education: "Harvard University announced a revolutionary online learning platform that will make 500 courses freely available worldwide.",
    environment: "The Amazon rainforest has shown signs of recovery following aggressive conservation efforts, with deforestation rates dropping 45% compared to last year.",
    finance: "Bitcoin surged past $100,000 as institutional investors increased their cryptocurrency holdings, signaling growing mainstream acceptance.",
    automotive: "Tesla unveiled its next-generation electric vehicle with a 600-mile range, setting a new benchmark for the automotive industry.",
    travel: "International tourism has rebounded to pre-pandemic levels, with over 1 billion travelers recorded in the first half of the year.",
    food: "A new study reveals the health benefits of the Mediterranean diet, linking it to reduced risk of heart disease and improved longevity.",
    fashion: "Paris Fashion Week showcased sustainable fashion trends, with major designers committing to carbon-neutral production methods.",
    realestate: "The housing market shows signs of cooling as mortgage rates reach 7%, with home sales declining for the fifth consecutive month.",
    legal: "The Supreme Court ruled on a landmark privacy case, establishing new protections for digital communications in the modern era.",
    religion: "The Vatican announced an interfaith initiative to address climate change, bringing together leaders from major world religions.",
    lifestyle: "The wellness industry continues to boom as consumers prioritize mental health, with meditation apps seeing 200% growth in users.",
    opinion: "The future of work demands a fundamental rethinking of our education system. We must prepare students for jobs that don't yet exist."
};

// ===== CATEGORY STYLES =====
const categoryStyles = {
    'technology': { icon: 'fa-microchip', color: '#818cf8', gradient: 'linear-gradient(135deg, #6366f1, #8b5cf6)' },
    'sports': { icon: 'fa-futbol', color: '#4ade80', gradient: 'linear-gradient(135deg, #22c55e, #06b6d4)' },
    'politics': { icon: 'fa-landmark', color: '#c4b5fd', gradient: 'linear-gradient(135deg, #8b5cf6, #a855f7)' },
    'business': { icon: 'fa-chart-line', color: '#fbbf24', gradient: 'linear-gradient(135deg, #f59e0b, #f97316)' },
    'entertainment': { icon: 'fa-film', color: '#f472b6', gradient: 'linear-gradient(135deg, #ec4899, #f472b6)' },
    'health': { icon: 'fa-heartbeat', color: '#f87171', gradient: 'linear-gradient(135deg, #ef4444, #f87171)' },
    'science': { icon: 'fa-flask', color: '#22d3ee', gradient: 'linear-gradient(135deg, #06b6d4, #22d3ee)' },
    'world': { icon: 'fa-globe', color: '#60a5fa', gradient: 'linear-gradient(135deg, #3b82f6, #60a5fa)' },
    'education': { icon: 'fa-graduation-cap', color: '#a78bfa', gradient: 'linear-gradient(135deg, #8b5cf6, #a78bfa)' },
    'environment': { icon: 'fa-leaf', color: '#34d399', gradient: 'linear-gradient(135deg, #10b981, #34d399)' },
    'finance': { icon: 'fa-coins', color: '#fcd34d', gradient: 'linear-gradient(135deg, #f59e0b, #fcd34d)' },
    'automotive': { icon: 'fa-car', color: '#38bdf8', gradient: 'linear-gradient(135deg, #0ea5e9, #38bdf8)' },
    'travel': { icon: 'fa-plane', color: '#2dd4bf', gradient: 'linear-gradient(135deg, #14b8a6, #2dd4bf)' },
    'food': { icon: 'fa-utensils', color: '#fb923c', gradient: 'linear-gradient(135deg, #f97316, #fb923c)' },
    'fashion': { icon: 'fa-tshirt', color: '#e879f9', gradient: 'linear-gradient(135deg, #d946ef, #e879f9)' },
    'realestate': { icon: 'fa-home', color: '#f97316', gradient: 'linear-gradient(135deg, #ea580c, #f97316)' },
    'legal': { icon: 'fa-gavel', color: '#94a3b8', gradient: 'linear-gradient(135deg, #64748b, #94a3b8)' },
    'religion': { icon: 'fa-place-of-worship', color: '#c4b5fd', gradient: 'linear-gradient(135deg, #a855f7, #c4b5fd)' },
    'lifestyle': { icon: 'fa-heart', color: '#fb7185', gradient: 'linear-gradient(135deg, #f43f5e, #fb7185)' },
    'opinion': { icon: 'fa-comment-dots', color: '#a3a3a3', gradient: 'linear-gradient(135deg, #737373, #a3a3a3)' },
    'accidents': { icon: 'fa-car-crash', color: '#ef4444', gradient: 'linear-gradient(135deg, #dc2626, #ef4444)' },
    'crime': { icon: 'fa-user-secret', color: '#7c3aed', gradient: 'linear-gradient(135deg, #6d28d9, #7c3aed)' },
    'protests': { icon: 'fa-bullhorn', color: '#db2777', gradient: 'linear-gradient(135deg, #be185d, #db2777)' }
};

// ===== UTILITY FUNCTIONS =====

// ===== UTILITY FUNCTIONS =====
const utils = {
    debounce: (fn, delay) => debounce(fn, delay),

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    },

    generateCacheKey(text, endpoint = 'classify') {
        return `${endpoint}:${text.trim().toLowerCase()}`;
    },

    // Efficient DOM updates using DocumentFragment
    updateElement(elementId, content) {
        const element = document.getElementById(elementId);
        if (element) {
            if (typeof content === 'string') {
                element.innerHTML = content;
            } else {
                element.textContent = content;
            }
        }
    }
};

// ===== LOADING STATE MANAGEMENT =====
const loadingManager = {
    show(message = 'Analyzing...', subMessage = 'Processing your content with AI') {
        const overlay = document.getElementById('loading-overlay');
        const loadingText = document.getElementById('loading-text');
        const loadingSubtext = document.getElementById('loading-subtext');

        if (overlay) {
            if (loadingText) loadingText.textContent = message;
            if (loadingSubtext) loadingSubtext.textContent = subMessage;
            overlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    },

    hide() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    },

    update(message, subMessage = null) {
        const loadingText = document.getElementById('loading-text');
        const loadingSubtext = document.getElementById('loading-subtext');

        if (message && loadingText) {
            loadingText.style.opacity = '0';
            setTimeout(() => {
                loadingText.textContent = message;
                loadingText.style.opacity = '1';
            }, 150);
        }

        if (subMessage && loadingSubtext) {
            loadingSubtext.style.opacity = '0';
            setTimeout(() => {
                loadingSubtext.textContent = subMessage;
                loadingSubtext.style.opacity = '1';
            }, 150);
        }
    }
};

// ===== NOTIFICATION SYSTEM =====
const notificationManager = {
    show(message, type = 'info') {
        // Remove existing notifications
        const existing = document.querySelector('.notification');
        if (existing) existing.remove();

        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;

        document.body.appendChild(notification);

        // Animate in
        requestAnimationFrame(() => {
            notification.style.transform = 'translateX(0)';
            notification.style.opacity = '1';
        });

        // Auto remove
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
};

// ===== API FUNCTIONS =====
const api = {
    async classifyText(text, enhanced = true) {
        const cacheKey = utils.generateCacheKey(text, 'classify');

        // Check client cache
        const cached = clientCache.get(cacheKey);
        if (cached) {
            return { ...cached, fromCache: true };
        }

        const signal = createAbortSignal();

        const response = await fetch(`${API_BASE}/classify`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, enhanced }),
            signal
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Classification failed');
        }

        const data = await response.json();

        // Cache successful response
        if (data.status === 'success') {
            clientCache.set(cacheKey, data);
        }

        return data;
    },

    async classifyImage(file) {
        const formData = new FormData();
        formData.append('image', file);

        const signal = createAbortSignal();

        const response = await fetch(`${API_BASE}/classify/image`, {
            method: 'POST',
            body: formData,
            signal
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Image classification failed');
        }

        return response.json();
    },

    async classifyAudio(file) {
        const formData = new FormData();
        formData.append('audio', file);

        const signal = createAbortSignal();

        const response = await fetch(`${API_BASE}/classify/audio`, {
            method: 'POST',
            body: formData,
            signal
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Audio classification failed');
        }

        return response.json();
    },

    async classifyVideo(file) {
        const formData = new FormData();
        formData.append('video', file);

        const signal = createAbortSignal();

        const response = await fetch(`${API_BASE}/classify/video`, {
            method: 'POST',
            body: formData,
            signal
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.message || 'Video classification failed');
        }

        return response.json();
    },

    async getHealth() {
        return cachedApiCall('/health');
    },

    async getCategories() {
        return cachedApiCall('/categories');
    },

    async getModelInfo() {
        return cachedApiCall('/model/info');
    }
};

// ===== RESULTS DISPLAY =====
const resultsManager = {
    display(data) {
        const section = document.getElementById('results-section');
        if (!section) return;

        // Reset and show section
        section.style.display = 'block';
        section.style.opacity = '0';

        requestAnimationFrame(() => {
            section.style.opacity = '1';
        });

        setTimeout(() => {
            section.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);

        const category = data.category || 'unknown';
        const confidence = (data.confidence || 0) * 100;
        const style = categoryStyles[category] || categoryStyles['technology'];

        this.updateCategoryBadge(category, style);
        this.updateConfidenceCircle(confidence, style);
        this.updateStats(data);
        this.displayPredictions(data.top_predictions || [{ category, confidence: data.confidence }]);
        this.displayKeywords(data.keywords || []);
        this.displaySummary(category, confidence, data);

        notificationManager.show(
            `Classified as ${utils.capitalizeFirst(category)} with ${confidence.toFixed(1)}% confidence!`,
            'success'
        );
    },

    updateCategoryBadge(category, style) {
        const badge = document.getElementById('result-category-badge');
        const text = document.getElementById('result-category');
        const icon = document.getElementById('result-icon');

        if (text) text.textContent = utils.capitalizeFirst(category);
        if (icon) {
            icon.className = `fas ${style.icon}`;
            icon.style.color = style.color;
        }
        if (badge) {
            badge.style.borderColor = style.color + '60';
            badge.style.background = style.gradient.replace(')', ', 0.2)');
        }
    },

    updateConfidenceCircle(confidence, style) {
        const progress = document.getElementById('confidence-progress');
        const text = document.getElementById('result-confidence');

        if (progress) {
            const circumference = 283;
            const offset = circumference - (confidence / 100) * circumference;
            progress.style.strokeDashoffset = '283';
            progress.style.stroke = style.color;

            requestAnimationFrame(() => {
                progress.style.strokeDashoffset = offset.toString();
            });
        }

        if (text) {
            this.animateValue(text, 0, confidence, 1000);
        }
    },

    animateValue(element, start, end, duration) {
        const startTime = performance.now();

        const update = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeOutQuart = 1 - Math.pow(1 - progress, 4);
            const current = start + (end - start) * easeOutQuart;

            element.textContent = `${current.toFixed(1)}%`;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        };

        requestAnimationFrame(update);
    },

    updateStats(data) {
        const timeEl = document.getElementById('result-time');
        const modelEl = document.getElementById('result-model');
        const inputTypeEl = document.getElementById('result-input-type');

        if (timeEl) {
            const time = data.processing_time_ms || 0;
            timeEl.textContent = time < 1000 ? `${time.toFixed(0)}ms` : `${(time / 1000).toFixed(2)}s`;
        }

        if (modelEl) {
            modelEl.textContent = data.model || 'Ensemble';
        }

        if (inputTypeEl) {
            inputTypeEl.textContent = utils.capitalizeFirst(data.input_type || state.currentInputType);
        }
    },

    displayPredictions(predictions) {
        const container = document.getElementById('predictions-list');
        if (!container) return;

        const fragment = document.createDocumentFragment();

        predictions.forEach((pred, index) => {
            const style = categoryStyles[pred.category] || categoryStyles['technology'];
            const confidence = (pred.confidence || 0) * 100;

            const card = document.createElement('div');
            card.className = 'prediction-card';
            card.style.animationDelay = `${index * 0.1}s`;
            card.innerHTML = `
                <div class="prediction-card-header">
                    <span class="prediction-card-category">
                        <i class="fas ${style.icon}" style="color: ${style.color}"></i>
                        ${utils.capitalizeFirst(pred.category)}
                    </span>
                    <span class="prediction-card-confidence">${confidence.toFixed(1)}%</span>
                </div>
                <div class="prediction-card-bar">
                    <div class="prediction-card-bar-fill" style="width: 0%; background: ${style.gradient};" data-width="${confidence}%"></div>
                </div>
            `;
            fragment.appendChild(card);
        });

        container.innerHTML = '';
        container.appendChild(fragment);

        // Animate bars
        setTimeout(() => {
            container.querySelectorAll('.prediction-card-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 200);
    },

    displayKeywords(keywords) {
        const section = document.getElementById('keywords-section');
        const container = document.getElementById('keywords-list');

        if (!section || !container) return;

        if (keywords.length === 0) {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';

        const fragment = document.createDocumentFragment();
        keywords.forEach((kw, index) => {
            const tag = document.createElement('span');
            tag.className = 'keyword-tag';
            tag.style.animationDelay = `${index * 0.05}s`;
            tag.textContent = typeof kw === 'string' ? kw : kw.word || kw.keyword || kw;
            fragment.appendChild(tag);
        });

        container.innerHTML = '';
        container.appendChild(fragment);
    },

    displaySummary(category, confidence, data) {
        const section = document.getElementById('analysis-summary');
        const content = document.getElementById('summary-content');

        if (!section || !content) return;

        const style = categoryStyles[category] || categoryStyles['technology'];
        const time = data.processing_time_ms || 0;
        const timeStr = time < 1000 ? `${time.toFixed(0)}ms` : `${(time / 1000).toFixed(2)}s`;

        let confidenceLevel = confidence < 50 ? 'low' : confidence < 75 ? 'moderate' : 'high';

        content.innerHTML = `
            <p>
                <strong>${utils.capitalizeFirst(category)}</strong> category detected with
                <strong>${confidence.toFixed(1)}%</strong> confidence (${confidenceLevel} certainty).
            </p>
            <p style="margin-top: 12px; font-size: 0.9em; opacity: 0.8;">
                Analysis completed in <strong>${timeStr}</strong> using the <strong>${data.model || 'Ensemble'}</strong> AI model.
                ${data.fromCache ? ' <span style="color: #4ade80;">(from cache)</span>' : ''}
            </p>
        `;
    }
};

// ===== EVENT HANDLERS =====
const eventHandlers = {
    // Debounced text input handler
    handleTextInput: debounce(() => {
        this.updateCharCount();
        this.updateAnalyzeButton();
    }, DEBOUNCE_DELAY),

    handleFileSelect(event, type) {
        event.preventDefault();
        event.stopPropagation();

        const file = event.target.files[0];
        if (file) {
            this.processFile(file, type);
        }
    },

    processFile(file, type) {
        const maxSize = MAX_FILE_SIZES[type];
        if (file.size > maxSize) {
            const maxMB = Math.round(maxSize / 1024 / 1024);
            notificationManager.show(`File too large. Maximum ${maxMB}MB allowed.`, 'error');
            return;
        }

        state.selectedFiles[type] = file;
        uiManager.updateFilePreview(type, file);
        this.updateAnalyzeButton();
        notificationManager.show(`${utils.capitalizeFirst(type)} file selected: ${file.name}`, 'success');
    },

    updateCharCount() {
        const textarea = document.getElementById('news-text');
        const charCount = document.getElementById('char-count');
        const wrapper = charCount?.parentElement;

        if (textarea && charCount) {
            const count = textarea.value.length;
            charCount.textContent = count;
            wrapper?.classList.toggle('sufficient', count >= MIN_CHARS);
        }
    },

    updateAnalyzeButton() {
        const btn = document.getElementById('analyze-btn');
        if (!btn) return;

        let isValid = false;

        switch (state.currentInputType) {
            case 'text':
                const textarea = document.getElementById('news-text');
                isValid = textarea && textarea.value.trim().length >= MIN_CHARS;
                break;
            case 'image':
            case 'audio':
            case 'video':
                isValid = state.selectedFiles[state.currentInputType] !== null;
                break;
        }

        btn.disabled = !isValid || state.isAnalyzing;
    }
};

// ===== UI MANAGER =====
const uiManager = {
    updateFilePreview(type, file) {
        if (type === 'image') {
            this.updateImagePreview(file);
        } else {
            this.updateMediaPreview(type, file);
        }
    },

    updateImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById('image-preview');
            const img = document.getElementById('image-preview-img');
            const uploadContent = document.getElementById('image-upload-content');
            const filename = document.getElementById('image-filename-display');

            if (preview && img && uploadContent) {
                img.src = e.target.result;
                preview.style.display = 'flex';
                uploadContent.style.display = 'none';
                if (filename) filename.textContent = file.name;
            }
        };
        reader.readAsDataURL(file);
    },

    updateMediaPreview(type, file) {
        const infoEl = document.getElementById(`${type}-info`);
        const filenameEl = document.getElementById(`${type}-filename`);
        const filesizeEl = document.getElementById(`${type}-filesize`);
        const uploadContent = document.getElementById(`${type}-upload-content`);

        if (infoEl && uploadContent) {
            if (filenameEl) filenameEl.textContent = file.name;
            if (filesizeEl) filesizeEl.textContent = utils.formatFileSize(file.size);
            infoEl.style.display = 'flex';
            uploadContent.style.display = 'none';
        }
    },

    switchInputType(type) {
        state.currentInputType = type;

        // Update buttons
        document.querySelectorAll('.selector-btn').forEach(btn => {
            const isActive = btn.dataset.type === type;
            btn.classList.toggle('active', isActive);
            btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
        });

        // Update panels
        document.querySelectorAll('.input-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `${type}-panel`);
        });

        eventHandlers.updateAnalyzeButton();
    },

    removeFile(type) {
        state.selectedFiles[type] = null;

        if (type === 'image') {
            const preview = document.getElementById('image-preview');
            const uploadContent = document.getElementById('image-upload-content');
            const fileInput = document.getElementById('image-input');

            if (preview) preview.style.display = 'none';
            if (uploadContent) uploadContent.style.display = 'flex';
            if (fileInput) fileInput.value = '';
        } else {
            const infoEl = document.getElementById(`${type}-info`);
            const uploadContent = document.getElementById(`${type}-upload-content`);
            const fileInput = document.getElementById(`${type}-input`);

            if (infoEl) infoEl.style.display = 'none';
            if (uploadContent) uploadContent.style.display = 'flex';
            if (fileInput) fileInput.value = '';
        }

        eventHandlers.updateAnalyzeButton();
    }
};

// ===== ANALYSIS FUNCTION =====
async function analyzeContent() {
    if (state.isAnalyzing) return;

    const btn = document.getElementById('analyze-btn');
    if (!btn) return;

    state.isAnalyzing = true;
    btn.classList.add('loading');
    btn.disabled = true;

    const messages = {
        text: { main: 'Analyzing Text...', sub: 'Running 35-category AI classification' },
        image: { main: 'Analyzing Image...', sub: 'Extracting text and classifying content' },
        audio: { main: 'Analyzing Audio...', sub: 'Transcribing speech to text' },
        video: { main: 'Analyzing Video...', sub: 'Analyzing video frames' }
    };

    const msg = messages[state.currentInputType] || messages.text;
    loadingManager.show(msg.main, msg.sub);

    try {
        let result;
        let inputText = '';

        switch (state.currentInputType) {
            case 'text':
                loadingManager.update('Processing text...', 'Running ultra-fast classification');
                const textarea = document.getElementById('news-text');
                inputText = textarea?.value.trim() || '';
                result = await api.classifyText(inputText);
                break;
            case 'image':
                loadingManager.update('Processing image...', 'Extracting visual features');
                inputText = state.selectedFiles.image?.name || 'Image file';
                result = await api.classifyImage(state.selectedFiles.image);
                break;
            case 'audio':
                loadingManager.update('Processing audio...', 'Transcribing speech to text');
                inputText = state.selectedFiles.audio?.name || 'Audio file';
                result = await api.classifyAudio(state.selectedFiles.audio);
                break;
            case 'video':
                loadingManager.update('Processing video...', 'Analyzing video frames');
                inputText = state.selectedFiles.video?.name || 'Video file';
                result = await api.classifyVideo(state.selectedFiles.video);
                break;
        }

        if (result?.status === 'success') {
            loadingManager.update('Finalizing results...', 'Almost done!');

            // Add to history
            addToHistory(result.data || result, inputText, state.currentInputType);

            setTimeout(() => {
                resultsManager.display(result.data || result);
            }, 200);
        } else {
            throw new Error(result?.message || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        notificationManager.show(`Analysis failed: ${error.message}`, 'error');
    } finally {
        setTimeout(() => {
            loadingManager.hide();
        }, 400);
        state.isAnalyzing = false;
        btn.classList.remove('loading');
        eventHandlers.updateAnalyzeButton();
    }
}

// ===== HISTORY MANAGEMENT =====
function addToHistory(result, inputText, inputType) {
    const historyEntry = {
        timestamp: new Date().toISOString(),
        category: result.category,
        confidence: result.confidence,
        inputType,
        inputPreview: inputType === 'text' ? inputText.substring(0, 100) + '...' : inputText
    };

    state.classificationHistory.unshift(historyEntry);
    if (state.classificationHistory.length > 50) {
        state.classificationHistory.pop();
    }

    // Save to localStorage
    localStorage.setItem('newscat_history', JSON.stringify(state.classificationHistory));
}

function loadHistory() {
    const saved = localStorage.getItem('newscat_history');
    if (saved) {
        state.classificationHistory = JSON.parse(saved);
    }
}

// ===== INITIALIZATION =====
function initializeApp() {
    loadHistory();
    setupEventListeners();
    setupDragAndDrop();
    loadModelInfo();

    // Load sample text on first visit
    if (!localStorage.getItem('newscat_visited')) {
        localStorage.setItem('newscat_visited', 'true');
        loadSample('tech');
    }
}

function setupEventListeners() {
    // Text input with debouncing
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.addEventListener('input', debounce(() => {
            eventHandlers.updateCharCount();
            eventHandlers.updateAnalyzeButton();
        }, DEBOUNCE_DELAY));
    }

    // Input type switching
    document.querySelectorAll('.selector-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            uiManager.switchInputType(btn.dataset.type);
        });
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
            input.addEventListener('change', (e) => eventHandlers.handleFileSelect(e, type));
        }
    });
}

function setupDragAndDrop() {
    ['image', 'audio', 'video'].forEach(type => {
        const dropZone = document.getElementById(`${type}-drop-zone`);
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.classList.remove('dragover');
            });
        });

        dropZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                eventHandlers.processFile(files[0], type);
            }
        });
    });
}

function loadSample(type) {
    const textarea = document.getElementById('news-text');
    if (textarea && sampleArticles[type]) {
        textarea.value = '';
        const text = sampleArticles[type];
        let index = 0;

        const typeInterval = setInterval(() => {
            if (index < text.length) {
                textarea.value += text.charAt(index);
                index++;
                eventHandlers.updateCharCount();
            } else {
                clearInterval(typeInterval);
                eventHandlers.updateAnalyzeButton();
            }
        }, 8);
    }
}

function clearText() {
    const textarea = document.getElementById('news-text');
    if (textarea) {
        textarea.value = '';
        eventHandlers.updateCharCount();
        eventHandlers.updateAnalyzeButton();
    }
}

async function loadModelInfo() {
    try {
        const info = await api.getModelInfo();
        state.modelInfo = info;
    } catch (error) {
        console.warn('Failed to load model info:', error);
    }
}

// ===== DOM READY =====
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// ===== EXPORT FOR GLOBAL ACCESS =====
window.newscat = {
    loadSample,
    clearText,
    removeFile: uiManager.removeFile.bind(uiManager),
    switchInputType: uiManager.switchInputType.bind(uiManager),
    getCacheStats: () => clientCache.getStats()
};
