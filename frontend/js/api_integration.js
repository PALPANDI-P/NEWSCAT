/**
 * NEWSCAT v7.0 Professional - Enhanced API Integration
 * Advanced response handlers and UI integration for new response format
 *
 * Features:
 * - Professional response parsing and handling
 * - Advanced result visualization
 * - Error recovery and graceful degradation
 * - Performance metrics display
 * - Confidence level visualization
 */

'use strict';

// =============================================================================
// RESPONSE HANDLERS & FORMATTERS
// =============================================================================

class ResponseHandler {
    /**
     * Handle successful classification response
     * Backend returns flat structure - extract classification data directly
     */
    static handleSuccess(response) {
        if (!response || response.status !== 'success') {
            return ResponseHandler.handleError({
                error: {
                    code: 'INVALID_RESPONSE',
                    message: 'Invalid response format from server'
                }
            });
        }

        // Backend returns flat structure - extract classification fields directly
        // Extract all classification-related data from the flat response
        const data = {
            category: response.category,
            category_display: response.category_display,
            confidence: response.confidence,
            confidence_level: response.confidence_level,
            processing_time_ms: response.processing_time_ms,
            model_name: response.model,
            model_version: response.model_version,
            input_type: response.input_type,
            subcategory: response.subcategory,
            summary: response.summary,
            keywords: response.keywords,
            entities: response.entities,
            topics: response.topics,
            analysis: response.analysis,
            content_length: response.content_length,
            word_count: response.word_count,
            cached: response.cached,
            enhanced: response.enhanced
        };

        const html = ResponseHandler.buildResultHTML(data, response);

        return {
            success: true,
            html,
            data,
            confidence: data.confidence,
            category: data.category,
            processingTime: data.processing_time_ms
        };
    }

    /**
     * Build professional result HTML
     */
    static buildResultHTML(data, fullResponse) {
        const conf = data.confidence;
        const confLevel = ResponseHandler.getConfidenceLevel(conf);
        const categoryDisplay = data.category_display || data.category;
        const inputType = data.input_type || 'text';

        // Get icon based on input type
        const typeIcon = {
            text: 'fa-file-alt',
            image: 'fa-image',
            audio: 'fa-microphone',
            video: 'fa-video'
        }[inputType] || 'fa-file';

        // Get color based on confidence level
        const getConfColor = (confidence) => {
            if (confidence >= 70) return 'var(--color-accent-emerald)';
            if (confidence >= 50) return 'var(--color-accent-amber)';
            if (confidence >= 30) return 'var(--color-accent-orange)';
            return 'var(--color-accent-rose)';
        };

        const confColor = getConfColor(conf);

        let html = `
            <div class="result-card">
                <div class="result-header">
                    <div class="category-badge">
                        <i class="fas ${typeIcon}"></i>
                        <span>${categoryDisplay}</span>
                        <span class="input-type-indicator" title="${inputType} input">
                            <i class="fas fa-${typeIcon}"></i>
                        </span>
                    </div>
                    <div class="confidence-indicator">
                        <div class="confidence-value-wrapper">
                            <span class="confidence-value" style="color: ${confColor}">${conf.toFixed(1)}%</span>
                            <span class="confidence-label">${confLevel}</span>
                        </div>
                        <div class="confidence-bar">
                            <div class="confidence-bar-fill" style="width: ${conf}%; background: ${confColor}; animation-duration: ${conf / 20}s;"></div>
                        </div>
                    </div>
                </div>
        `;

        // Add analysis section with top predictions
        if (data.analysis && data.analysis.top_predictions && data.analysis.top_predictions.length > 0) {
            html += ResponseHandler.buildPredictionsHTML(data.analysis.top_predictions);
        }

        // Add keywords
        if (data.analysis && data.analysis.keywords && data.analysis.keywords.length > 0) {
            html += ResponseHandler.buildKeywordsHTML(data.analysis.keywords);
        }

        // Add metrics
        if (data.metrics) {
            html += ResponseHandler.buildMetricsHTML(data.metrics);
        }

        // Add processing info
        html += `
                <div class="metrics-section">
                    <div class="metric">
                        <div class="metric-label">Model</div>
                        <div class="metric-value">${data.model_name}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Processing Time</div>
                        <div class="metric-value">${ResponseHandler.formatDuration(data.processing_time_ms)}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">Input Type</div>
                        <div class="metric-value">${ResponseHandler.capitalizeFirst(data.input_type)}</div>
                    </div>
                </div>
            </div>
        `;

        return html;
    }

    /**
     * Build predictions HTML
     */
    static buildPredictionsHTML(predictions) {
        let html = `
            <div class="predictions-section">
                <div class="predictions-title">Top Predictions</div>
        `;

        predictions.slice(0, 5).forEach((pred, idx) => {
            const categoryDisplay = pred.category_display || pred.category;
            html += `
                <div class="prediction-item">
                    <div class="prediction-rank">${idx + 1}</div>
                    <div class="prediction-category">${categoryDisplay}</div>
                    <div class="prediction-confidence">${pred.confidence.toFixed(1)}%</div>
                </div>
            `;
        });

        html += `</div>`;
        return html;
    }

    /**
     * Build keywords HTML
     */
    static buildKeywordsHTML(keywords) {
        let html = `
            <div class="keywords-section">
                <div class="keywords-title">Extracted Keywords</div>
                <div class="keywords-list">
        `;

        keywords.forEach(keyword => {
            html += `<span class="keyword-tag"><i class="fas fa-bolt"></i>${keyword}</span>`;
        });

        html += `</div></div>`;
        return html;
    }

    /**
     * Build metrics HTML
     */
    static buildMetricsHTML(metrics) {
        let html = `<div class="metrics-section">`;

        if (metrics.character_count !== undefined) {
            html += `
                <div class="metric">
                    <div class="metric-label">Characters</div>
                    <div class="metric-value">${ResponseHandler.formatNumber(metrics.character_count)}</div>
                </div>
            `;
        }

        if (metrics.word_count !== undefined) {
            html += `
                <div class="metric">
                    <div class="metric-label">Words</div>
                    <div class="metric-value">${ResponseHandler.formatNumber(metrics.word_count)}</div>
                </div>
            `;
        }

        if (metrics.sentence_count !== undefined) {
            html += `
                <div class="metric">
                    <div class="metric-label">Sentences</div>
                    <div class="metric-value">${ResponseHandler.formatNumber(metrics.sentence_count)}</div>
                </div>
            `;
        }

        html += `</div>`;
        return html;
    }

    /**
     * Determine confidence level string
     */
    static getConfidenceLevel(confidence) {
        if (confidence >= 90) return 'Very High';
        if (confidence >= 70) return 'High';
        if (confidence >= 50) return 'Moderate';
        if (confidence >= 30) return 'Low';
        return 'Very Low';
    }

    /**
     * Handle error response
     */
    static handleError(response) {
        const message = response?.error?.message || 'Classification failed';
        const code = response?.error?.code || 'UNKNOWN';

        let alertClass = 'alert-error';
        let icon = 'fa-circle-exclamation';

        if (response?.status === 'warning') {
            alertClass = 'alert-warning';
            icon = 'fa-triangle-exclamation';
        }

        const html = `
            <div class="alert ${alertClass}">
                <i class="fas ${icon}"></i>
                <span>${message}</span>
                ${response?.error?.details ? `<small>(${code})</small>` : ''}
            </div>
        `;

        return {
            success: false,
            html,
            message,
            code
        };
    }

    /**
     * Format duration in readable format
     */
    static formatDuration(ms) {
        if (ms < 1000) return `${Math.round(ms)}ms`;
        return `${(ms / 1000).toFixed(2)}s`;
    }

    /**
     * Format large numbers with commas
     */
    static formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    }

    /**
     * Capitalize first letter
     */
    static capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// =============================================================================
// API CLIENT ENHANCEMENTS
// =============================================================================

class APIClient {
    constructor(baseURL = '/api') {
        this.baseURL = baseURL;
        this.defaultHeaders = {
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest'
        };
    }

    /**
     * Make API request with comprehensive error handling
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            method: options.method || 'GET',
            headers: { ...this.defaultHeaders, ...options.headers },
            signal: options.signal,
            ...options
        };

        // Don't send empty body for GET
        if (config.method === 'GET') {
            delete config.body;
        } else if (options.data) {
            config.body = JSON.stringify(options.data);
        }

        try {
            const response = await fetch(url, config);

            if (!response.ok) {
                const error = await response.json().catch(() => ({
                    error: { message: `HTTP ${response.status}` }
                }));
                throw error;
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'AbortError') {
                throw { error: { code: 'ABORTED', message: 'Request cancelled' } };
            }
            throw error;
        }
    }

    /**
     * Classify text with performance tracking
     * Backend returns flat structure - add client timing directly to response
     */
    async classifyText(text, options = {}) {
        const startTime = performance.now();

        try {
            const response = await this.request('/classify', {
                method: 'POST',
                data: { text },
                signal: options.signal
            });

            const endTime = performance.now();
            // Backend returns flat structure - add client_processing_time directly
            response.client_processing_time = endTime - startTime;

            return response;
        } catch (error) {
            console.error('Classification error:', error);
            throw error;
        }
    }

    /**
     * Upload and classify image
     */
    async classifyImage(file, options = {}) {
        const formData = new FormData();
        formData.append('image', file);

        try {
            return await this.request('/classify/image', {
                method: 'POST',
                body: formData,
                headers: {}, // Don't set Content-Type; browser will set it with boundary
                signal: options.signal
            });
        } catch (error) {
            console.error('Image classification error:', error);
            throw error;
        }
    }

    /**
     * Upload and classify audio
     */
    async classifyAudio(file, options = {}) {
        const formData = new FormData();
        formData.append('audio', file);

        try {
            return await this.request('/classify/audio', {
                method: 'POST',
                body: formData,
                headers: {}, // Don't set Content-Type; browser will set it with boundary
                signal: options.signal
            });
        } catch (error) {
            console.error('Audio classification error:', error);
            throw error;
        }
    }

    /**
     * Upload and classify video
     */
    async classifyVideo(file, options = {}) {
        const formData = new FormData();
        formData.append('video', file);

        try {
            return await this.request('/classify/video', {
                method: 'POST',
                body: formData,
                headers: {}, // Don't set Content-Type; browser will set it with boundary
                signal: options.signal
            });
        } catch (error) {
            console.error('Video classification error:', error);
            throw error;
        }
    }

    /**
     * Get health status
     */
    async healthCheck() {
        try {
            return await this.request('/health');
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }

    /**
     * Get categories
     */
    async getCategories() {
        try {
            return await this.request('/categories');
        } catch (error) {
            console.error('Failed to fetch categories:', error);
            throw error;
        }
    }

    /**
     * Get model information
     */
    async getModelInfo() {
        try {
            return await this.request('/model/info');
        } catch (error) {
            console.error('Failed to fetch model info:', error);
            throw error;
        }
    }
}

// =============================================================================
// UI COMPONENTS & RENDERING
// =============================================================================

class UIRenderer {
    /**
     * Render classification result with animations
     */
    static renderResult(element, handlerResult, animated = true) {
        if (!element) return;

        element.innerHTML = handlerResult.html;

        if (animated) {
            const cards = element.querySelectorAll('.result-card');
            cards.forEach((card, idx) => {
                card.style.animationDelay = `${idx * 0.1}s`;
            });
        }
    }

    /**
     * Show loading state
     */
    static showLoading(element, message = 'Analyzing...') {
        if (!element) return;

        element.innerHTML = `
            <div class="loading-overlay active">
                <div class="spinner"></div>
                <div class="loading-text">${message}</div>
                <div class="loading-subtext">Processing content</div>
            </div>
        `;
    }

    /**
     * Update progress
     */
    static updateProgress(element, percentage, message = '') {
        if (!element) return;

        element.innerHTML = `
            <div class="metric">
                <div class="metric-label">Processing</div>
                <div class="confidence-bar">
                    <div class="confidence-bar-fill" style="width: ${percentage}%;"></div>
                </div>
                ${message ? `<small>${message}</small>` : ''}
            </div>
        `;
    }

    /**
     * Display error alert
     */
    static showError(element, title, message, code = '') {
        if (!element) return;

        element.innerHTML = `
            <div class="alert alert-error">
                <i class="fas fa-circle-exclamation"></i>
                <div>
                    <strong>${title}</strong>
                    <p>${message}</p>
                    ${code ? `<small>Code: ${code}</small>` : ''}
                </div>
            </div>
        `;
    }

    /**
     * Display success message
     */
    static showSuccess(element, message) {
        element.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle"></i>
                <span>${message}</span>
            </div>
        `;
        setTimeout(() => {
            element.innerHTML = '';
        }, 3000);
    }
}

// =============================================================================
// PERFORMANCE & MONITORING
// =============================================================================

class PerformanceMonitor {
    constructor() {
        this.metrics = {};
    }

    record(key, value) {
        if (!this.metrics[key]) {
            this.metrics[key] = [];
        }
        this.metrics[key].push(value);
    }

    getStats(key) {
        const values = this.metrics[key] || [];
        if (values.length === 0) return null;

        return {
            min: Math.min(...values),
            max: Math.max(...values),
            avg: values.reduce((a, b) => a + b, 0) / values.length,
            count: values.length,
            total: values.reduce((a, b) => a + b, 0)
        };
    }

    getReport() {
        const report = {};
        for (const key in this.metrics) {
            report[key] = this.getStats(key);
        }
        return report;
    }
}

// =============================================================================
// EXPORTS
// =============================================================================

// Global instances
const apiClient = new APIClient();
const uiRenderer = new UIRenderer();
const performanceMonitor = new PerformanceMonitor();

// Export for use in main.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ResponseHandler,
        APIClient,
        UIRenderer,
        PerformanceMonitor,
        apiClient,
        uiRenderer,
        performanceMonitor
    };
}
